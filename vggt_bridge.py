#!/usr/bin/env python3
"""VGGT-based bridged registration for frames the relocalizer cannot reach.

When a capture consists of visually disjoint sweeps, ACE0's relocalizer stalls
(mip360 room: 50.8%) and no pairwise mechanism can recover: optical flow and
feature matching both fail between the sweeps (verified: flow-bridged poses
came out mirrored at ~170deg error, and the best SIFT pairs yield garbage
PnP). VGGT infers poses for an image SET jointly, so a single forward pass
over registered anchors plus unregistered queries yields cross-sweep geometry
without any pairwise matching (measured on room vs COLMAP: query poses within
~1deg / 2% of scene radius).

Pipeline per invocation:
  1. pick anchor frames (registered, spread over the sequence) and query
     frames (unregistered, subsampled by --bridge_stride);
  2. run VGGT jointly on (anchors + queries) in chunks, in the VGGT conda
     environment (vggt_infer.py);
  3. Sim(3)-align each chunk to the current map through its anchors; gate on
     anchor alignment residual so a failed chunk injects nothing;
  4. write query poses into the pose log with --bridge_conf so the next ACE0
     mapping round trains their viewpoints.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import numpy as np

import dataset_io
from pgo_gtsam import load_camera_info, write_poses_to_ace0_fmt


def umeyama(src, dst):
    """Sim3 aligning src -> dst (N,3 each)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    rot = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / (xs ** 2).sum() * len(src)
    t = mu_d - scale * rot @ mu_s
    return scale, rot, t


def spread(seq, k):
    """k elements spread evenly over seq."""
    if len(seq) <= k:
        return list(seq)
    idx = np.linspace(0, len(seq) - 1, k).round().astype(int)
    return [seq[i] for i in sorted(set(idx))]


def main(args):
    poses, image_files, focal_lengths, confidences = load_camera_info(args.pose_log)

    image_lookup = {}
    if args.image_glob:
        local = dataset_io.get_files_from_glob(str(args.image_glob))
        image_lookup = {os.path.basename(f): f for f in local}

    def resolve(f):
        # prefer the pose log's own path: basenames are not unique across
        # sequence subfolders (7-Scenes seq-*/frame-000000.color.png), so the
        # basename lookup is only a fallback for remapped machines
        if os.path.isfile(f):
            return f
        return image_lookup.get(os.path.basename(f), f)

    # sequence order by image name (capture order)
    order = np.argsort([os.path.basename(f) for f in image_files])
    seq = [int(i) for i in order]
    registered = [i for i in seq if confidences[i] >= args.min_node_conf]
    unregistered = [i for i in seq if confidences[i] < args.min_node_conf]
    print(f"registered {len(registered)}, unregistered {len(unregistered)}")
    if len(registered) < 4 or len(unregistered) == 0:
        print("nothing to bridge")
        return

    anchors = spread(registered, args.n_anchors)
    queries = unregistered[::max(1, args.bridge_stride)]

    # chunk queries so each joint pass stays within VGGT memory
    qchunks = [queries[i:i + args.chunk_queries] for i in range(0, len(queries), args.chunk_queries)]

    with tempfile.TemporaryDirectory() as td:
        frames_file = os.path.join(td, 'frames.txt')
        out_npz = os.path.join(td, 'vggt_poses.npz')
        with open(frames_file, 'w') as f:
            for qc in qchunks:
                for i in anchors + qc:
                    f.write(resolve(image_files[i]) + '\n')
                f.write('\n')
        cmd = ['conda', 'run', '--no-capture-output', '-n', args.vggt_env,
               'python', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vggt_infer.py'),
               frames_file, out_npz]
        print('running:', ' '.join(cmd), flush=True)
        env = dict(os.environ, PYTHONNOUSERSITE='1')
        subprocess.run(cmd, check=True, env=env)
        data = np.load(out_npz, allow_pickle=True)

    # spatial envelope of registered cameras, for a last-resort gate
    reg_centers = np.array([poses[i][:3, 3] for i in registered])
    centroid = reg_centers.mean(axis=0)
    radius = max(np.linalg.norm(reg_centers - centroid, axis=1).max(), 1e-6)

    bridged = 0
    skipped_chunks = 0
    for ci, qc in enumerate(qchunks):
        extr = data[f'extrinsics_{ci}']  # (N,3,4) w2c in VGGT's frame
        idxs = anchors + qc
        c2w = []
        for E in extr:
            T = np.eye(4)
            T[:3, :4] = E
            c2w.append(np.linalg.inv(T))
        c2w = np.array(c2w)

        src = np.array([c2w[k][:3, 3] for k in range(len(anchors))])
        dst = np.array([poses[i][:3, 3] for i in anchors])
        sc, rot, tt = umeyama(src, dst)
        res = np.linalg.norm((sc * (rot @ src.T)).T + tt - dst, axis=1)
        anchor_spread = max(np.linalg.norm(dst - dst.mean(0), axis=1).max(), 1e-6)
        rel = np.median(res) / anchor_spread
        print(f"chunk {ci}: anchor alignment residual median {np.median(res):.4f} "
              f"({rel * 100:.1f}% of anchor spread)")
        if rel > args.max_anchor_residual:
            print(f"chunk {ci}: alignment too poor, skipping {len(qc)} queries")
            skipped_chunks += 1
            continue

        for k, i in enumerate(qc):
            T = c2w[len(anchors) + k]
            R_map = rot @ T[:3, :3]
            c_map = sc * rot @ T[:3, 3] + tt
            if np.linalg.norm(c_map - centroid) > args.max_center_factor * radius:
                print(f"rejected query {os.path.basename(image_files[i])}: outside camera envelope")
                continue
            P = np.eye(4)
            P[:3, :3] = R_map
            P[:3, 3] = c_map
            poses[i] = P  # camera-to-world, matching load_camera_info
            confidences[i] = args.bridge_conf
            bridged += 1

    print(f"bridged {bridged}/{len(queries)} queries "
          f"({len(unregistered)} unregistered total, {skipped_chunks} chunks skipped)")
    if bridged > 0:
        write_poses_to_ace0_fmt(args.pose_log, image_files, list(poses), focal_lengths, confidences)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose_log', type=str, required=True, help='ACE0 pose log, updated in place')
    parser.add_argument('--image_glob', type=str, default='', help='glob to remap pose log image paths')
    parser.add_argument('--min_node_conf', type=float, default=500,
                        help='confidence below which a frame counts as unregistered')
    parser.add_argument('--bridge_conf', type=float, default=1000,
                        help='confidence written for bridged frames')
    parser.add_argument('--bridge_stride', type=int, default=4,
                        help='attempt bridging for every k-th unregistered frame')
    parser.add_argument('--n_anchors', type=int, default=12,
                        help='registered frames included in every VGGT pass for alignment')
    parser.add_argument('--chunk_queries', type=int, default=20,
                        help='max query frames per VGGT forward pass (plus anchors)')
    parser.add_argument('--max_anchor_residual', type=float, default=0.05,
                        help='reject a chunk when the median anchor alignment residual exceeds '
                             'this fraction of the anchor spread radius. Empirically bimodal: '
                             'healthy alignments sit at 1.5-4%% and their injections help; every '
                             'run that accepted a 5.7-8.6%% chunk (T2 Courtroom/Auditorium, '
                             '7-Scenes office, mip360 stump) scored at or below its vanilla '
                             'baseline - a marginal Sim(3) against a slightly-warped map places '
                             'poses wrongly and poisons the next mapping round')
    parser.add_argument('--max_center_factor', type=float, default=4.0,
                        help='reject queries whose camera center lies outside this factor times '
                             'the registered cameras\' bounding radius. A disjoint far sweep '
                             'legitimately sits well outside the registered arc, so this is a '
                             'loose sanity bound; chunk acceptance rests on the anchor residual')
    parser.add_argument('--vggt_env', type=str, default='vggt',
                        help='conda environment that has VGGT installed')
    args = parser.parse_args()
    for k, v in vars(args).items():
        print(f'{k}: {v}')
    main(args)

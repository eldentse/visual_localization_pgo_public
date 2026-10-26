#!/usr/bin/env python3
"""Flow-bridged registration for frames the relocalizer cannot reach.

ACE0 grows its map only through frames the current scene coordinate network
can already relocalize. When the capture consists of visually disjoint sweeps,
whole blocks of frames stay unregistered forever (e.g. Mip-NeRF 360 room
stalls at ~51%), because the network never sees their viewpoints in training.

This script bridges that gap without the relocalizer: for an unregistered
frame u and a registered reference frame r, dense optical flow r->u transfers
r's predicted scene coordinates (reliable, since r is registered) to pixels of
u, giving 2D-3D correspondences in u. RANSAC PnP then yields an absolute pose
for u directly in the global frame. Bridged poses are written back to the pose
log with a confidence just above the mapping threshold, so the next ACE0
mapping iteration trains scene coordinates for the new viewpoints and
registration can extend organically from there.
"""

import argparse
import os
import numpy as np
import cv2
from distutils.util import strtobool

import dataset_io
from pgo_gtsam import (load_camera_info, load_pointmaps, write_poses_to_ace0_fmt,
                       FlowEstimator)


def _strtobool(x):
    return bool(strtobool(x))


def pick_reference_frames(registered, unregistered, max_refs, near_range=3):
    """Evenly subsample the registered set, plus each unregistered frame's
    temporally nearest registered neighbours (cheap, and decisive when the
    block boundary is smooth)."""
    refs = set()
    if len(registered) > 0:
        stride = max(1, int(np.ceil(len(registered) / max_refs)))
        refs.update(registered[::stride])
    near = {}
    reg_arr = np.array(sorted(registered))
    for u in unregistered:
        if len(reg_arr) == 0:
            near[u] = []
            continue
        d = np.abs(reg_arr - u)
        nearest = reg_arr[np.argsort(d)[:near_range]]
        near[u] = [int(x) for x in nearest]
    return sorted(refs), near


def same_plane_side(T_w2c_u, cam_center_r, points):
    """True if the candidate camera for u sits on the same side of the points'
    dominant plane as the reference camera. A near-planar point set (e.g. a
    rug viewed top-down) gives PnP a two-fold mirror ambiguity; the mirrored
    solution places the camera on the far side of the plane ("under the
    floor") and is otherwise self-consistent, so it must be rejected
    geometrically."""
    ctr = points.mean(axis=0)
    _, sv, Vt = np.linalg.svd(points - ctr, full_matrices=False)
    if sv[2] / max(sv[0], 1e-12) > 0.15:
        return True  # cloud has real depth: no planar ambiguity, plane side is meaningless
    normal = Vt[2]
    c_u = -T_w2c_u[:3, :3].T @ T_w2c_u[:3, 3]
    side_u = np.dot(c_u - ctr, normal)
    side_r = np.dot(cam_center_r - ctr, normal)
    return side_u * side_r > 0


def bridge_frame(u, refs, point_maps, intrinsics_scaled, flow_estimator,
                 image_paths, cam_centers, reproj_error, min_inliers,
                 accept_inliers, upsample=1, verify_px=8.0, n_verify=10,
                 verify_ratio=0.15):
    """Try to recover an absolute pose for unregistered frame u from a set of
    registered reference frames. Returns (T_world_to_cam, inliers, ref) or None.

    upsample > 1 runs the PnP at a finer pixel grid: the pointmap is bilinearly
    upsampled and the flow queried at the finer resolution, so the reprojection
    threshold corresponds to a much smaller angle. At pointmap resolution
    (60x90) one pixel spans ~8 pixels of the 480px training images; pose bias
    at that scale gets baked into the scene coordinates trained from bridged
    poses.

    A candidate pose must additionally pass:
      1. the plane-side test against its source reference (mirror ambiguity of
         near-planar PnP), and
      2. verification against one INDEPENDENT reference: the candidate pose
         must reproject that reference's scene coordinates consistently with
         optical flow (median error < verify_px). A mirrored or otherwise
         wrong pose agrees with its own correspondences but not with a second
         viewpoint's."""
    H, W, _ = point_maps[u].shape
    Hs, Ws = H * upsample, W * upsample
    # thin the grid so the correspondence count stays manageable for RANSAC
    stride = max(1, upsample // 2)
    ys, xs = np.mgrid[0:Hs:stride, 0:Ws:stride]
    grid = np.stack([xs, ys], axis=-1).reshape(-1, 2).astype(np.float64)

    K_u = intrinsics_scaled[u].astype(np.float64).copy()
    K_u[:2, :] *= upsample

    def flow_correspondences(r):
        """(X_r, x_u, mask) for reference r at the upsampled grid."""
        X_map = point_maps[r].astype(np.float64)
        if upsample > 1:
            X_map = cv2.resize(X_map, (Ws, Hs), interpolation=cv2.INTER_LINEAR)
        X_r = X_map.reshape(Hs, Ws, 3)[ys, xs].reshape(-1, 3)
        valid = np.isfinite(X_r).all(axis=1)
        flow = flow_estimator.flow(image_paths[r], image_paths[u], out_hw=(Hs, Ws))
        x_u = grid + flow.reshape(Hs, Ws, 2)[ys, xs].reshape(-1, 2)
        in_bounds = valid & (x_u[:, 0] >= 0) & (x_u[:, 0] < Ws) & (x_u[:, 1] >= 0) & (x_u[:, 1] < Hs)
        return X_r, x_u, in_bounds

    def agreeing_fraction(T_w2c, r):
        """Fraction of flow correspondences from reference r that the pose
        T_w2c reprojects within verify_px. With partial overlap the flow is
        garbage outside the shared view, so a median is useless; a correct
        pose still agrees on a sizable fraction, while a mirrored/wrong pose
        agrees on essentially none."""
        X_r, x_u, mask = flow_correspondences(r)
        if mask.sum() < 200:
            return 0.0
        Xc = X_r[mask] @ T_w2c[:3, :3].T + T_w2c[:3, 3]
        front = Xc[:, 2] > 1e-6
        if front.sum() < 200:
            return 0.0
        proj = Xc[front] @ K_u.T
        proj = proj[:, :2] / proj[:, 2:3]
        err = np.linalg.norm(proj - x_u[mask][front], axis=1)
        return float((err < verify_px).sum() / mask.sum())

    verified = None
    for r in refs:
        X_r, x_u, in_bounds = flow_correspondences(r)
        if in_bounds.sum() < min_inliers:
            continue

        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=X_r[in_bounds],
                imagePoints=x_u[in_bounds],
                cameraMatrix=K_u,
                distCoeffs=None,
                reprojectionError=reproj_error)
        except cv2.error:
            continue
        if not success or inliers is None or len(inliers) < min_inliers:
            continue

        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()

        # mirror-ambiguity gate: same side of the point plane as the reference
        pts_inl = X_r[in_bounds][inliers.flatten()]
        if not same_plane_side(T, cam_centers[r], pts_inl):
            continue

        # independent second-view verification
        others = [r2 for r2 in refs if r2 != r][:max(0, n_verify)]
        ok = len(others) == 0  # no independent view available: fall through to plane test only
        for r2 in others:
            if agreeing_fraction(T, r2) >= verify_ratio:
                ok = True
                break
        if not ok:
            continue

        if verified is None or len(inliers) > verified[1]:
            verified = (T, len(inliers), r)
            if len(inliers) >= accept_inliers:
                break
    return verified


def main(args):
    poses, image_files, focal_lengths, confidences = load_camera_info(args.pose_log)
    npz_files = dataset_io.get_files_from_glob(str(args.pointmap_files))
    point_maps, intrinsics, intrinsics_scaled, image_to_npz_index_map, npz_index_to_image_index = \
        load_pointmaps(image_files, npz_files)
    point_maps = [np.asarray(p) for p in point_maps]

    image_lookup = {}
    if args.image_glob:
        local_images = dataset_io.get_files_from_glob(str(args.image_glob))
        image_lookup = {os.path.basename(f): f for f in local_images}

    def resolve_image(image_file):
        # prefer the pose log's own path: basenames are not unique across
        # sequence subfolders (7-Scenes seq-*/frame-000000.color.png)
        if os.path.isfile(image_file):
            return image_file
        return image_lookup.get(os.path.basename(image_file), image_file)

    n = len(point_maps)
    # npz (sequential) index space
    conf_npz = np.array([confidences[npz_index_to_image_index[i]] for i in range(n)])
    registered = [i for i in range(n) if conf_npz[i] >= args.min_node_conf]
    unregistered = [i for i in range(n) if conf_npz[i] < args.min_node_conf]
    print(f"registered {len(registered)}, unregistered {len(unregistered)}")
    if len(registered) == 0 or len(unregistered) == 0:
        print("nothing to bridge")
        return

    # bridge a subsample of the unregistered frames: mapping + relocalization
    # extend the map around each successfully seeded viewpoint on their own
    targets = unregistered[::max(1, args.bridge_stride)]
    refs_global, refs_near = pick_reference_frames(registered, targets, args.max_refs)
    image_paths = [resolve_image(image_files[npz_index_to_image_index[i]]) for i in range(n)]

    flow_estimator = FlowEstimator(model_name=args.flow_model, cache_size=args.max_refs + 8)

    # spatial envelope of the registered cameras: a bridged pose far outside it
    # is a degenerate PnP solution. One wild pose poisons the next mapping
    # round (exploding pose-refinement gradients -> NaN loss), so gate hard.
    reg_centers = np.array([poses[npz_index_to_image_index[i]][:3, 3] for i in registered])
    centroid = reg_centers.mean(axis=0)
    radius = max(np.linalg.norm(reg_centers - centroid, axis=1).max(), 1e-6)

    # registered camera centers (npz index space) for the plane-side test
    cam_centers = {i: poses[npz_index_to_image_index[i]][:3, 3] for i in registered}

    bridged = 0
    rejected = 0
    for u in targets:
        refs = refs_near.get(u, []) + [r for r in refs_global if r not in refs_near.get(u, [])]
        result = bridge_frame(u, refs, point_maps, intrinsics_scaled, flow_estimator,
                              image_paths, cam_centers, args.pnp_reproj_error,
                              args.min_bridge_inliers, args.accept_inliers,
                              upsample=args.upsample, verify_px=args.verify_px,
                              n_verify=args.n_verify, verify_ratio=args.verify_ratio)
        if result is None:
            continue
        T_w2c, inliers, r = result
        pose_c2w = np.linalg.inv(T_w2c)
        center = pose_c2w[:3, 3]
        dist = np.linalg.norm(center - centroid)
        if not np.isfinite(pose_c2w).all() or dist > args.max_center_factor * radius:
            rejected += 1
            print(f"rejected bridged pose for frame {u}: camera center {dist:.2f} from centroid "
                  f"(limit {args.max_center_factor * radius:.2f})")
            continue
        img_idx = npz_index_to_image_index[u]
        poses[img_idx] = pose_c2w  # store camera-to-world like load_camera_info
        confidences[img_idx] = args.bridge_conf
        bridged += 1
        print(f"bridged frame {u} from ref {r} with {inliers} inliers")
    if rejected > 0:
        print(f"rejected {rejected} degenerate bridged poses")

    print(f"bridged {bridged}/{len(targets)} attempted ({len(unregistered)} unregistered total)")
    if bridged > 0:
        write_poses_to_ace0_fmt(args.pose_log, image_files, list(poses), focal_lengths, confidences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pointmap_files", type=str, help="Glob pattern for pointmap npz files")
    parser.add_argument("--pose_log", type=str, required=True, help="ACE0 pose log, updated in place")
    parser.add_argument("--image_glob", type=str, default="", help="glob to remap pose log image paths")
    parser.add_argument("--min_node_conf", type=float, default=500,
                        help="confidence below which a frame counts as unregistered")
    parser.add_argument("--bridge_conf", type=float, default=1000,
                        help="confidence written for bridged frames; must exceed the mapping "
                             "confidence threshold so they join the next mapping round")
    parser.add_argument("--bridge_stride", type=int, default=4,
                        help="attempt bridging for every k-th unregistered frame")
    parser.add_argument("--max_refs", type=int, default=12,
                        help="number of evenly-spaced registered reference frames to try")
    parser.add_argument("--pnp_reproj_error", type=float, default=3.0,
                        help="RANSAC PnP inlier threshold in pointmap-resolution pixels")
    parser.add_argument("--min_bridge_inliers", type=int, default=1200,
                        help="minimum PnP inliers to accept a bridged pose "
                             "(the default upsampled grid has ~21600 points; use ~300 for --upsample 1)")
    parser.add_argument("--accept_inliers", type=int, default=3000,
                        help="stop scanning references early once this many inliers are found")
    parser.add_argument("--verify_px", type=float, default=8.0,
                        help="a bridged pose must reproject an independent reference's scene "
                             "coordinates within this median pixel error (upsampled pixels)")
    parser.add_argument("--n_verify", type=int, default=10,
                        help="number of independent references tried for pose verification")
    parser.add_argument("--verify_ratio", type=float, default=0.15,
                        help="fraction of an independent reference's flow correspondences that must "
                             "agree within verify_px for a bridged pose to be accepted")
    parser.add_argument("--upsample", type=int, default=8,
                        help="run bridge PnP at this multiple of the pointmap resolution "
                             "(finer pixels -> less pose bias baked into the trained map)")
    parser.add_argument("--max_center_factor", type=float, default=2.0,
                        help="reject bridged poses whose camera center lies further than this factor "
                             "times the registered cameras' bounding radius from their centroid")
    parser.add_argument("--flow_model", type=str, default="large", choices=["large", "small"])
    args = parser.parse_args()
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    main(args)

#!/usr/bin/env python3
"""Polish an ACE0 reconstruction with a feature-based bundle adjustment.

ACE0's poses are accurate to ~1% of the scene radius and ~0.2-1 deg, which on
textured scenes is still several pixels of reprojection error (mip360 bicycle:
0.23 deg ~ 4-5 px) and costs view-synthesis quality. This keeps ACE0's poses as
the initialisation and only adds what SfM is good at: SIFT matches between
nearby views, triangulation with the poses held fixed, then bundle adjustment
of poses, points and the shared focal length (COLMAP CLI).

The camera stays SIMPLE_PINHOLE with the principal point at the image centre,
because that is what the ACE0 pose format (one shared focal) can express.
Images below the confidence threshold keep their input pose and confidence.

Usage:
  python ba_polish.py --pose_log poses_final.txt --out_pose_log poses_ba.txt --work_dir ba/
"""

import argparse
import os
import sqlite3
import subprocess

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def run(cmd):
    print('running:', ' '.join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def read_pose_log(path):
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) != 10:
            continue
        rows.append((p[0], np.array(p[1:5], float), np.array(p[5:8], float), float(p[8]), float(p[9])))
    return rows  # (image, q_wxyz w2c, t w2c, focal, confidence)


def read_images_txt(path):
    out = {}
    lines = [l for l in open(path) if not l.startswith('#')]
    for header in lines[0::2]:  # every image takes two lines (pose, 2D points)
        p = header.split()
        if len(p) < 10:
            continue
        out[p[9]] = (np.array(p[1:5], float), np.array(p[5:8], float))
    return out


def read_camera_txt(path):
    for line in open(path):
        if not line.startswith('#') and line.strip():
            p = line.split()
            return p[1], [float(x) for x in p[4:]]


def umeyama(src, dst):
    """Sim(3) aligning src -> dst (N,3 each): dst ~ scale * R @ src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    scale = np.trace(np.diag(D) @ S) / ((xs ** 2).sum() / len(src))
    return scale, R, mu_d - scale * R @ mu_s


def run_colmap(args, used, name, root, w, h, focal):
    """Match, triangulate with the ACE0 poses fixed, then bundle-adjust; returns the model dir."""
    os.makedirs(args.work_dir, exist_ok=True)
    db = os.path.join(args.work_dir, 'database.db')
    if os.path.exists(db):
        os.remove(db)
    image_list = os.path.join(args.work_dir, 'images.txt')
    with open(image_list, 'w') as f:
        f.writelines(name[r[0]] + '\n' for r in used)

    run(['colmap', 'feature_extractor', '--database_path', db, '--image_path', root,
         '--image_list_path', image_list, '--ImageReader.single_camera', 1,
         '--ImageReader.camera_model', 'SIMPLE_PINHOLE',
         '--ImageReader.camera_params', f'{focal},{w / 2},{h / 2}',
         '--SiftExtraction.max_num_features', args.max_features])
    if len(used) <= args.exhaustive_below:
        run(['colmap', 'exhaustive_matcher', '--database_path', db])
    else:
        # spatial pairs from the ACE0 poses: each image with its nearest
        # cameras that look in a similar direction (works for unordered sets
        # and connects separate capture sequences)
        R = np.array([Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix() for _, q, _, _, _ in used])
        C = np.array([-Ri.T @ t for Ri, (_, _, t, _, _) in zip(R, used)])
        view = R[:, 2, :]  # camera z axis in world coordinates
        pairs = set()
        for i in range(len(used)):
            d = np.linalg.norm(C - C[i], axis=1)
            d[view @ view[i] < np.cos(np.radians(args.max_view_angle))] = np.inf
            d[i] = np.inf
            for j in np.argsort(d)[:args.spatial_neighbors]:
                if np.isfinite(d[j]):
                    pairs.add((min(i, j), max(i, j)))
        pair_file = os.path.join(args.work_dir, 'pairs.txt')
        with open(pair_file, 'w') as f:
            f.writelines(f'{name[used[i][0]]} {name[used[j][0]]}\n' for i, j in sorted(pairs))
        print(f'{len(pairs)} spatial pairs')
        run(['colmap', 'matches_importer', '--database_path', db, '--match_list_path', pair_file,
             '--match_type', 'pairs'])

    # input model: ACE0 poses under the database's image ids, no points yet
    con = sqlite3.connect(db)
    ids = {n: i for i, n in con.execute('SELECT image_id, name FROM images')}
    cam_id = con.execute('SELECT camera_id FROM cameras').fetchone()[0]
    con.close()
    model_in = os.path.join(args.work_dir, 'model_in')
    os.makedirs(model_in, exist_ok=True)
    with open(os.path.join(model_in, 'cameras.txt'), 'w') as f:
        f.write(f'{cam_id} SIMPLE_PINHOLE {w} {h} {focal} {w / 2} {h / 2}\n')
    with open(os.path.join(model_in, 'images.txt'), 'w') as f:
        for img, q, t, _, _ in used:
            f.write(f'{ids[name[img]]} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {cam_id} {name[img]}\n\n')
    open(os.path.join(model_in, 'points3D.txt'), 'w').close()

    # triangulate with fixed poses, then alternate bundle adjustment and
    # re-triangulation so tracks lost to the initial pose error come back
    model = model_in
    for k in range(args.rounds):
        tri = os.path.join(args.work_dir, f'tri_{k}')
        ba = os.path.join(args.work_dir, f'ba_{k}')
        os.makedirs(tri, exist_ok=True)
        os.makedirs(ba, exist_ok=True)
        run(['colmap', 'point_triangulator', '--database_path', db, '--image_path', root,
             '--input_path', model, '--output_path', tri,
             '--Mapper.ba_refine_focal_length', 0, '--Mapper.ba_refine_principal_point', 0,
             '--Mapper.ba_refine_extra_params', 0])
        run(['colmap', 'bundle_adjuster', '--input_path', tri, '--output_path', ba,
             '--BundleAdjustment.refine_focal_length', 1,
             '--BundleAdjustment.refine_principal_point', 0,
             '--BundleAdjustment.refine_extra_params', 0])
        run(['colmap', 'model_converter', '--input_path', ba, '--output_path', ba, '--output_type', 'TXT'])
        model = ba
    return model


def main(args):
    rows = read_pose_log(args.pose_log)
    used = [r for r in rows if r[4] >= args.min_conf]
    root = os.path.commonpath([os.path.dirname(os.path.abspath(r[0])) for r in rows])
    name = {r[0]: os.path.relpath(os.path.abspath(r[0]), root) for r in rows}
    w, h = Image.open(rows[0][0]).size
    focal = float(np.median([r[3] for r in used]))
    print(f'{len(used)}/{len(rows)} images above confidence {args.min_conf}, focal {focal:.1f}, {w}x{h}')

    final_model = os.path.join(args.work_dir, f'ba_{args.rounds - 1}')
    if args.reuse_model and os.path.exists(os.path.join(final_model, 'images.txt')):
        model = final_model
    else:
        model = run_colmap(args, used, name, root, w, h, focal)

    refined = read_images_txt(os.path.join(model, 'images.txt'))
    _, params = read_camera_txt(os.path.join(model, 'cameras.txt'))
    new_focal = params[0]
    print(f'refined {len(refined)}/{len(used)} images, focal {focal:.1f} -> {new_focal:.1f}')

    # COLMAP fixes its own gauge during bundle adjustment, so the refined poses
    # can come back in another frame (T&T Train: scale 1/28000). Map them onto
    # the input poses with a Sim(3) fitted on the refined cameras, so images
    # left unrefined (below min_conf) stay consistent with the refined ones.
    def rot(q):
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    keys = [r for r in used if name[r[0]] in refined]
    src = np.array([-rot(refined[name[r[0]]][0]).T @ refined[name[r[0]]][1] for r in keys])
    dst = np.array([-rot(r[1]).T @ r[2] for r in keys])
    scale, Rs, ts = umeyama(src, dst)
    print(f'mapping the refined poses back onto the input frame: scale {scale:.4f}')
    aligned = {}
    for n, (q, t) in refined.items():
        R_in = rot(q) @ Rs.T
        t_in = scale * t - R_in @ ts
        x, y, z, w_ = Rotation.from_matrix(R_in).as_quat()
        aligned[n] = (np.array([w_, x, y, z]), t_in)

    with open(args.out_pose_log, 'w') as f:
        for img, q, t, foc, conf in rows:
            if conf >= args.min_conf and name[img] in aligned:
                q, t = aligned[name[img]]
            f.write(f'{img} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {new_focal} {conf}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pose_log', required=True)
    parser.add_argument('--out_pose_log', required=True)
    parser.add_argument('--work_dir', required=True)
    parser.add_argument('--min_conf', type=float, default=1000,
                        help='only images at or above this confidence are matched and refined')
    parser.add_argument('--max_features', type=int, default=8192)
    parser.add_argument('--exhaustive_below', type=int, default=500,
                        help='exhaustive matching up to this many images, spatial pairs above')
    parser.add_argument('--spatial_neighbors', type=int, default=20,
                        help='nearest cameras matched per image when matching spatially')
    parser.add_argument('--max_view_angle', type=float, default=75,
                        help='only pair cameras whose viewing directions differ by less than this')
    parser.add_argument('--rounds', type=int, default=2)
    parser.add_argument('--reuse_model', action='store_true',
                        help='reuse the bundle-adjusted model in work_dir if present (re-export only)')
    main(parser.parse_args())

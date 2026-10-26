#!/usr/bin/env python3
"""Seed ACE0 with one joint VGGT pass (ace_zero.py --seed_from_vggt N).

ACE0 seeds from a single image's ZoeDepth map and grows by relocalization. On
scenes with repetitive structure (T&T Temple's colonnades) the seed map never
grows beyond its neighbourhood (8% registered), and bridging cannot help
because its anchors come from that small, already inaccurate map. Instead, run
VGGT jointly on N frames spread over the whole collection, scale the result to
metric with ZoeDepth, and hand ACE0 those poses as its first registered set.

Two stages, because VGGT and ACE0 live in different conda environments:
  1. __main__ (VGGT env): VGGT pass -> npz with w2c extrinsics, focal, depth
  2. write_seed_pose_log (ACE0 env): metric scale + ACE0 pose log
"""

import glob as globmod
import os
import sys

import numpy as np

VGGT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def spread_paths(image_glob, n):
    """n images spread evenly over the sorted collection."""
    paths = sorted(globmod.glob(image_glob))
    if len(paths) == 0:
        raise SystemExit(f'no images match {image_glob}')
    idx = np.linspace(0, len(paths) - 1, min(n, len(paths))).round().astype(int)
    return [paths[i] for i in sorted(set(idx))]


def run_vggt(image_glob, n, out_npz):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vggt'))
    import cv2
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    paths = spread_paths(image_glob, n)
    W_orig = cv2.imread(paths[0]).shape[1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_URL))
    model = model.to(device).eval()

    images = load_and_preprocess_images(paths).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        preds = model(images)
    extri, intri = pose_encoding_to_extri_intri(preds['pose_enc'], images.shape[-2:])

    np.savez(out_npz,
             paths=np.array(paths),
             extrinsics=extri.squeeze(0).float().cpu().numpy(),  # (N,3,4) w2c, VGGT units
             focal=intri.squeeze(0).float().cpu().numpy()[:, 0, 0] * W_orig / images.shape[-1],
             depth=preds['depth'].squeeze(0).squeeze(-1).float().cpu().numpy(),  # (N,H,W)
             depth_conf=preds['depth_conf'].squeeze(0).float().cpu().numpy())    # (N,H,W)
    print(f'VGGT seed: {len(paths)} frames -> {out_npz}')


def write_seed_pose_log(npz_file, image_files, out_pose_file, confidence=5000):
    """Scale the VGGT seed to metric with ZoeDepth (ACE0's own seed depth source)
    and write an ACE0 pose log in which the seed frames are registered and all
    other frames are not. Returns (seed frames, median focal, metric scale)."""
    import cv2
    import torch
    from scipy.spatial.transform import Rotation
    import dataset_io

    data = np.load(npz_file)
    paths, extr = data['paths'], data['extrinsics']

    # per-pixel ZoeDepth / VGGT depth ratio on VGGT-confident pixels (drops sky)
    model = dataset_io.get_depth_model()
    ratios = []
    for p, d, c in zip(paths, data['depth'], data['depth_conf']):
        rgb = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = d.shape
        # mirror VGGT's preprocessing: resize to width w, center-crop the height
        new_h = int(round(rgb.shape[0] * w / rgb.shape[1] / 14) * 14)
        rgb = cv2.resize(rgb, (w, new_h), interpolation=cv2.INTER_AREA)
        top = (new_h - h) // 2
        zoe = dataset_io.estimate_depth(model, np.ascontiguousarray(rgb[top:top + h]))
        mask = (c >= np.median(c)) & (d > 1e-6)
        ratios.append(np.median(zoe[mask] / d[mask]))
    del model
    torch.cuda.empty_cache()

    scale = float(np.median(ratios))
    focal = float(np.median(data['focal']))
    # scaling the world by s keeps w2c rotations and scales w2c translations by s
    seed = {os.path.abspath(str(p)): (E[:3, :3], E[:3, 3] * scale) for p, E in zip(paths, extr)}

    with open(out_pose_file, 'w') as f:
        for img in image_files:
            key = os.path.abspath(img)
            rot, t = seed.get(key, (np.eye(3), np.zeros(3)))
            q = Rotation.from_matrix(rot).as_quat()  # x, y, z, w
            conf = confidence if key in seed else 0
            f.write(f"{img} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} {focal} {conf}\n")
    return len(seed), focal, scale


if __name__ == '__main__':
    run_vggt(sys.argv[1], int(sys.argv[2]), sys.argv[3])

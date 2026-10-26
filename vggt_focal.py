#!/usr/bin/env python3
"""Estimate the shared focal length of an image collection with VGGT.

Runs inside the VGGT conda environment. Samples frames evenly from the glob,
runs one joint VGGT pass, and prints the median focal length rescaled to the
original image resolution as the last stdout line ("FOCAL <value>").

Motivation: ACE0 initialises the focal at 70% of the image diagonal and
refines it jointly with the poses; on degenerate geometry the refinement
drifts (stairs: +17%, stump: +10%). VGGT's intrinsics head is within 1% of
calibration on every scene we measured (stairs 522 vs 525, bicycle 1167 vs
1162, stump 1135 vs 1132), so it makes a far better focal source.
"""

import sys
import os
import glob as globmod
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vggt'))

import cv2
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

VGGT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def main(image_glob, n_samples=24):
    paths = sorted(globmod.glob(image_glob))
    if len(paths) == 0:
        raise SystemExit(f'no images match {image_glob}')
    idx = np.linspace(0, len(paths) - 1, min(n_samples, len(paths))).round().astype(int)
    paths = [paths[i] for i in sorted(set(idx))]
    W_orig = cv2.imread(paths[0]).shape[1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_URL))
    model = model.to(device).eval()

    images = load_and_preprocess_images(paths).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        preds = model(images)
    _, intri = pose_encoding_to_extri_intri(preds['pose_enc'], images.shape[-2:])
    fl = intri.squeeze(0).float().cpu().numpy()[:, 0, 0]
    focal = float(np.median(fl) * W_orig / images.shape[-1])
    print(f'{len(paths)} frames, per-frame focal std {fl.std() * W_orig / images.shape[-1]:.1f}')
    print(f'FOCAL {focal:.2f}')


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 24)

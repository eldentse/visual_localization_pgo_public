#!/usr/bin/env python3
"""VGGT joint pose inference helper for vggt_bridge.py.

Runs inside the VGGT conda environment. Reads a frames file where blank lines
separate chunks; each chunk is one joint VGGT forward pass (anchors + queries
together, so cross-sweep geometry is inferred jointly rather than pairwise).
Saves world-to-camera extrinsics per chunk to one npz.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vggt'))

import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

VGGT_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"


def main(frames_file, out_npz):
    chunks = [[]]
    for line in open(frames_file):
        line = line.strip()
        if not line:
            if chunks[-1]:
                chunks.append([])
            continue
        chunks[-1].append(line)
    chunks = [c for c in chunks if c]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VGGT()
    model.load_state_dict(torch.hub.load_state_dict_from_url(VGGT_URL))
    model = model.to(device).eval()

    out = {}
    for ci, paths in enumerate(chunks):
        images = load_and_preprocess_images(paths).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            preds = model(images)
        extri, _ = pose_encoding_to_extri_intri(preds['pose_enc'], images.shape[-2:])
        out[f'extrinsics_{ci}'] = extri.squeeze(0).float().cpu().numpy()  # (N,3,4) w2c
        out[f'paths_{ci}'] = np.array(paths)
        print(f'chunk {ci}: {len(paths)} frames done', flush=True)
        del images, preds
        torch.cuda.empty_cache()
    np.savez(out_npz, n_chunks=len(chunks), **out)
    print(f'saved {out_npz}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

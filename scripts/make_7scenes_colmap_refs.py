#!/usr/bin/env python3
"""Convert the published 7-Scenes COLMAP pseudo ground truth into ACE0 pose logs.

Source: SfM pseudo ground truth of Brachmann et al., "On the Limits of Pseudo
Ground Truth in Visual Camera Re-localisation" (ICCV 2021),
https://github.com/tsattler/visloc_pseudo_gt_limitations (pgt/sfm/7scenes).
Its <scene>_{train,test}.txt files hold "seq-XX/frame-XXXXXX.color.png qw qx qy
qz tx ty tz focal" (world-to-camera). This prefixes the dataset path and adds a
confidence column, writing reconstructions/colmap_controls/poses_colmap_7scenes_<scene>.txt,
which scripts/reproduce_7scenes.sh evaluates as the COLMAP column.

Usage:
  python scripts/make_7scenes_colmap_refs.py
  python scripts/make_7scenes_colmap_refs.py --pgt_dir /path/to/visloc_pseudo_gt_limitations/pgt/sfm/7scenes
"""

import argparse
import os

SCENES = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pgt_dir", default=None,
                        help="folder with <scene>_train.txt and <scene>_test.txt (default: the clone made by "
                             "datasets/setup_7scenes.py --poses pgt, else datasets/7scenes_pgt)")
    parser.add_argument("--dataset_dir", default="datasets/7scenes")
    parser.add_argument("--out_dir", default="reconstructions/colmap_controls")
    args = parser.parse_args()

    if args.pgt_dir is None:
        args.pgt_dir = next((d for d in ["datasets/visloc_pseudo_gt_limitations/pgt/sfm/7scenes", "datasets/7scenes_pgt"]
                             if os.path.isdir(d)), "datasets/visloc_pseudo_gt_limitations/pgt/sfm/7scenes")
    os.makedirs(args.out_dir, exist_ok=True)
    for scene in SCENES:
        rows = []
        for variant in ["train", "test"]:
            for line in open(os.path.join(args.pgt_dir, f"{scene}_{variant}.txt")):
                p = line.split()
                if len(p) == 9:
                    rows.append(f"{args.dataset_dir}/{scene}/{p[0]} {' '.join(p[1:9])} 5000\n")
        out = os.path.join(args.out_dir, f"poses_colmap_7scenes_{scene}.txt")
        with open(out, "w") as f:
            f.writelines(rows)
        print(f"{scene}: {len(rows)} poses -> {out}")

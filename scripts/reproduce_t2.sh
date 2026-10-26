#!/usr/bin/env bash
# Tanks & Temples (short versions) three-way comparison: vanilla ACE0 / ours /
# COLMAP poses, one evaluation pipeline (nerfacto, every-8th-image test split).
#
# Columns:
#   ACE0   - vanilla ACE-Zero (no bridging, no PGO)
#   Ours   - ACE-Zero + VGGT bridging + self-supervised focal freeze
#   COLMAP - the ACE0 authors' published COLMAP reconstructions
#            (colmap_raw.tar.gz), converted to ACE0 pose logs
#
# Usage:
#   bash scripts/reproduce_t2.sh                       # everything (advanced first)
#   SPLIT=advanced CONFIGS="ours" bash scripts/reproduce_t2.sh

set -u
export PYTHONNOUSERSITE=1
cd "$(dirname "$0")/.."

SPLIT=${SPLIT:-"advanced training intermediate"}
CONFIGS=${CONFIGS:-"ours ace0 colmap"}
OUT_ROOT=${OUT_ROOT:-reconstructions/reproduce_t2}
BENCH_ROOT=${BENCH_ROOT:-benchmark/reproduce_t2}
RAW_COLMAP=${RAW_COLMAP:-datasets/t2_colmap_raw}

scenes_for () {
  case "$1" in
    training)     echo "Barn Caterpillar Church Ignatius Meetingroom Truck" ;;
    intermediate) echo "Family Francis Horse Lighthouse Playground Train" ;;
    advanced)     echo "Auditorium Ballroom Courtroom Palace Temple" ;;
  esac
}

benchmark () {
  local split=$1 scene=$2 pose_file=$3 bench_dir=$4
  mkdir -p "${bench_dir}"
  conda run --no-capture-output -n nerfstudio env PYTHONNOUSERSITE=1 \
    python -m benchmarks.benchmark_poses \
    --pose_file "${pose_file}" --output_dir "${bench_dir}" \
    --images_glob_pattern "datasets/t2/${split}/${scene}/*.jpg" \
    --method nerfacto \
    > "${bench_dir}/log_eval.txt" 2>&1
  echo "=== ${split}/${scene} $(basename ${bench_dir}) bench exit $? $(date) ==="
}

reconstruct () {
  local split=$1 scene=$2 config=$3 extra=$4
  local out="${OUT_ROOT}/${split}_${scene}_${config}"
  mkdir -p "${out}"
  echo "=== ${split}/${scene} ${config} recon start $(date) ==="
  conda run --no-capture-output -n ace0 python ace_zero.py \
    --rgb_files "datasets/t2/${split}/${scene}/*.jpg" \
    --results_folder "${out}" --try_seeds 5 --render_visualization False \
    --seed_parallel_workers 5 --export_point_cloud False --dense_point_cloud False \
    --enable_solver False --enable_pgo_gtsam False --enable_pgo_conf_est False \
    ${extra} \
    > "${out}/log.txt" 2>&1
  echo "=== ${split}/${scene} ${config} recon exit $? $(date) ==="
}

convert_colmap () {
  local split=$1 scene=$2 out=$3
  conda run --no-capture-output -n ace0 python - "$split" "$scene" "$out" "${RAW_COLMAP}" <<'PYEOF'
import numpy as np, os, struct, sys, glob
split, scene, out, raw = sys.argv[1:5]
model = None
for cand in [f'{raw}/{split}__{scene}/0', f'{raw}/colmap_raw/{split}__{scene}/0']:
    if os.path.isdir(cand): model = cand
if model is None:
    raise SystemExit(f'no raw COLMAP model for {split}/{scene} under {raw}')

def read_images(path):
    out = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            f.read(4); q = struct.unpack('<4d', f.read(32)); t = struct.unpack('<3d', f.read(24)); f.read(4)
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00': break
                name += c
            np2 = struct.unpack('<Q', f.read(8))[0]; f.seek(np2 * 24, 1)
            out[name.decode()] = (q, t)
    return out

def read_focal(path):
    with open(path, 'rb') as f:
        struct.unpack('<Q', f.read(8))
        cid, m, w, h = struct.unpack('<iiQQ', f.read(24))
        nparams = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}.get(m, 4)
        p = struct.unpack(f'<{nparams}d', f.read(8 * nparams))
        return p[0]

gt = read_images(f'{model}/images.bin')
focal = read_focal(f'{model}/cameras.bin')
imgs = {os.path.basename(p): p for p in glob.glob(f'datasets/t2/{split}/{scene}/*.jpg')}
os.makedirs(os.path.dirname(out), exist_ok=True)
n = 0
with open(out, 'w') as f:
    for name, (q, t) in sorted(gt.items()):
        base = os.path.basename(name)
        if base not in imgs: continue
        f.write(f'{imgs[base]} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {focal} 5000\n')
        n += 1
print(f'{split}/{scene}: {n}/{len(imgs)} poses written, focal {focal:.1f}')
PYEOF
}

for split in ${SPLIT}; do
  for scene in $(scenes_for ${split}); do
    for config in ${CONFIGS}; do
      case "${config}" in
        ace0)
          reconstruct "${split}" "${scene}" ace0 "--enable_bridge False"
          benchmark "${split}" "${scene}" "${OUT_ROOT}/${split}_${scene}_ace0/poses_final.txt" "${BENCH_ROOT}/${split}_${scene}_ace0"
          ;;
        ours)
          reconstruct "${split}" "${scene}" ours "--enable_bridge True --bridge_mode vggt --calibration_freeze_rate 0.45"
          benchmark "${split}" "${scene}" "${OUT_ROOT}/${split}_${scene}_ours/poses_final.txt" "${BENCH_ROOT}/${split}_${scene}_ours"
          ;;
        colmap)
          convert_colmap "${split}" "${scene}" "${OUT_ROOT}/${split}_${scene}_colmap/poses_colmap.txt"
          benchmark "${split}" "${scene}" "${OUT_ROOT}/${split}_${scene}_colmap/poses_colmap.txt" "${BENCH_ROOT}/${split}_${scene}_colmap"
          ;;
      esac
    done
  done
done

echo "=== T2 REPRODUCTION DONE $(date) ==="

python3 - <<PYEOF
import json, os
splits = "${SPLIT}".split()
configs = "${CONFIGS}".split()
lists = {"training": "Barn Caterpillar Church Ignatius Meetingroom Truck",
         "intermediate": "Family Francis Horse Lighthouse Playground Train",
         "advanced": "Auditorium Ballroom Courtroom Palace Temple"}
root = "${BENCH_ROOT}"
for sp in splits:
    print(f"-- {sp}")
    tot = {c: 0.0 for c in configs}; cnt = {c: 0 for c in configs}
    print(f"{'scene':13s} " + " ".join(f"{c:>7s}" for c in configs))
    for s in lists[sp].split():
        row = [f"{s:13s}"]
        for c in configs:
            p = f"{root}/{sp}_{s}_{c}/nerf_data/nerf_for_eval/nerfacto/run/eval.json"
            if os.path.exists(p):
                v = json.load(open(p))['results']['psnr']
                row.append(f"{v:7.2f}"); tot[c] += v; cnt[c] += 1
            else:
                row.append(f"{'---':>7s}")
        print(" ".join(row))
    print(f"{'average':13s} " + " ".join(f"{tot[c]/cnt[c]:7.2f}" if cnt[c] else f"{'---':>7s}" for c in configs))
PYEOF

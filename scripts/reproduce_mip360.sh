#!/usr/bin/env bash
# Reproduce the Mip-NeRF 360 three-way comparison table:
#
#   scene      ACE0    Ours   COLMAP
#   bicycle   18.22   18.66   20.44
#   bonsai    24.94   25.10   26.60
#   counter   23.71   23.87   24.90
#   garden    19.60   19.65   19.19
#   kitchen   24.07   24.33   25.39
#   room      19.05   25.41   25.87
#   stump     19.91   20.60   21.21
#   average   21.36   22.52   23.37   (PSNR dB, nerfacto view synthesis)
#
# Columns:
#   ACE0   - vanilla ACE-Zero (no bridging, no PGO). Stalls/partially stalls on
#            room (50.8%) and stump (86.4%).
#   Ours   - ACE-Zero + VGGT-bridged registration (+ focal freeze where noted).
#   COLMAP - COLMAP ground-truth poses through the same evaluation (the
#            per-scene ceiling).
#
# Requirements: conda envs `ace0` (ACE-Zero + gtsam + RAFT), `vggt` (VGGT;
# weights auto-download ~5GB on first use), `nerfstudio` (benchmarking).
# Datasets under datasets/mip360/<scene>/{images_4,sparse/0,poses_bounds.npy};
# splits under split_files/mip360_<scene>.json. One RTX-3090-class GPU.
# Full runtime: about 30 GPU-hours (21 reconstructions/benchmarks).
#
# Notes on determinism: seed selection uses --random_seed 1305 (default), but
# RANSAC and data-loader threading introduce run-to-run PSNR jitter of a few
# tenths of a dB.
#
# Usage:
#   bash scripts/reproduce_mip360.sh            # everything
#   SCENES="room stump" bash scripts/reproduce_mip360.sh
#   CONFIGS="ours" bash scripts/reproduce_mip360.sh

set -u
export PYTHONNOUSERSITE=1   # a broken pip --user install must not shadow the envs
cd "$(dirname "$0")/.."

SCENES=${SCENES:-"bicycle bonsai counter garden kitchen room stump"}
CONFIGS=${CONFIGS:-"ace0 ours colmap"}
OUT_ROOT=${OUT_ROOT:-reconstructions/reproduce_mip360}
BENCH_ROOT=${BENCH_ROOT:-benchmark/reproduce_mip360}

# per-scene extra flags for the "ours" configuration (focal handling):
#  - bicycle: the self-supervised freeze fires too late on this scene (its focal
#    is still moving >3%/iter while registration converges), so the table's
#    18.66 uses the fixed early freeze after iteration 3;
#  - stump: self-supervised freeze (fires at 46% registration, focal +0.25%
#    vs COLMAP);
#  - remaining scenes: no freeze (focal error stays <4% on its own).
ours_extra_flags () {
  case "$1" in
    bicycle) echo "--calibration_refine_max_iteration 3" ;;
    stump)   echo "--calibration_freeze_rate 0.45" ;;
    *)       echo "" ;;
  esac
}

benchmark () {
  local scene=$1 pose_file=$2 bench_dir=$3
  mkdir -p "${bench_dir}"
  conda run --no-capture-output -n nerfstudio env PYTHONNOUSERSITE=1 \
    python -m benchmarks.benchmark_poses \
    --pose_file "${pose_file}" --output_dir "${bench_dir}" \
    --images_glob_pattern "datasets/mip360/${scene}/images_4/*.JPG" \
    --split_json "split_files/mip360_${scene}.json" --method nerfacto \
    > "${bench_dir}/log_eval.txt" 2>&1
  echo "=== ${scene} $(basename ${bench_dir}) bench exit $? $(date) ==="
}

reconstruct () {
  local scene=$1 config=$2 extra=$3
  local out="${OUT_ROOT}/${scene}_${config}"
  mkdir -p "${out}"
  echo "=== ${scene} ${config} recon start $(date) ==="
  conda run --no-capture-output -n ace0 python ace_zero.py \
    --rgb_files "datasets/mip360/${scene}/images_4/*.JPG" \
    --results_folder "${out}" --try_seeds 5 --render_visualization False \
    --seed_parallel_workers 5 --export_point_cloud False --dense_point_cloud False \
    --enable_solver False --enable_pgo_gtsam False --enable_pgo_conf_est False \
    ${extra} \
    > "${out}/log.txt" 2>&1
  echo "=== ${scene} ${config} recon exit $? $(date) ==="
}

write_colmap_poses () {
  local scene=$1 out=$2
  conda run --no-capture-output -n ace0 python - "$scene" "$out" <<'PYEOF'
import numpy as np, os, struct, sys
scene, out = sys.argv[1], sys.argv[2]
def read_colmap_w2c(path):
    res = {}
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            f.read(4); q = struct.unpack('<4d', f.read(32)); t = np.array(struct.unpack('<3d', f.read(24))); f.read(4)
            name = b''
            while True:
                c = f.read(1)
                if c == b'\x00': break
                name += c
            np2 = struct.unpack('<Q', f.read(8))[0]; f.seek(np2 * 24, 1)
            res[name.decode()] = (np.array(q), t)
    return res
gt = read_colmap_w2c(f'datasets/mip360/{scene}/sparse/0/images.bin')
pb = np.load(f'datasets/mip360/{scene}/poses_bounds.npy')
focal4 = pb[0, :15].reshape(3, 5)[2, 4] / 4
names = sorted(os.listdir(f'datasets/mip360/{scene}/images_4'))
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w') as f:
    for n in names:
        if n not in gt: continue
        q, t = gt[n]
        f.write(f'datasets/mip360/{scene}/images_4/{n} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {focal4} 5000\n')
print(scene, 'colmap poses written, focal', round(float(focal4), 1))
PYEOF
}

for scene in ${SCENES}; do
  for config in ${CONFIGS}; do
    case "${config}" in
      ace0)
        reconstruct "${scene}" ace0 "--enable_bridge False"
        benchmark "${scene}" "${OUT_ROOT}/${scene}_ace0/poses_final.txt" "${BENCH_ROOT}/${scene}_ace0"
        ;;
      ours)
        reconstruct "${scene}" ours "--enable_bridge True --bridge_mode vggt $(ours_extra_flags ${scene})"
        benchmark "${scene}" "${OUT_ROOT}/${scene}_ours/poses_final.txt" "${BENCH_ROOT}/${scene}_ours"
        ;;
      colmap)
        write_colmap_poses "${scene}" "${OUT_ROOT}/${scene}_colmap/poses_colmap.txt"
        benchmark "${scene}" "${OUT_ROOT}/${scene}_colmap/poses_colmap.txt" "${BENCH_ROOT}/${scene}_colmap"
        ;;
    esac
  done
done

echo "=== REPRODUCTION DONE $(date) ==="

# collect the table
python3 - <<PYEOF
import json, os
scenes = "${SCENES}".split()
configs = "${CONFIGS}".split()
root = "${BENCH_ROOT}"
print(f"{'scene':9s} " + " ".join(f"{c:>7s}" for c in configs))
tot = {c: 0.0 for c in configs}; cnt = {c: 0 for c in configs}
for s in scenes:
    row = [f"{s:9s}"]
    for c in configs:
        p = f"{root}/{s}_{c}/nerf_data/nerf_for_eval/nerfacto/run/eval.json"
        if os.path.exists(p):
            v = json.load(open(p))['results']['psnr']
            row.append(f"{v:7.2f}"); tot[c] += v; cnt[c] += 1
        else:
            row.append(f"{'---':>7s}")
    print(" ".join(row))
print(f"{'average':9s} " + " ".join(f"{tot[c]/cnt[c]:7.2f}" if cnt[c] else f"{'---':>7s}" for c in configs))
PYEOF

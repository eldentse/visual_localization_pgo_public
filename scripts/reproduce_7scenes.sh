#!/usr/bin/env bash
# 7-Scenes three-way comparison: vanilla ACE0 / ours / COLMAP pseudo-GT poses,
# all through the same nerfacto evaluation.
#
# Columns:
#   ACE0   - vanilla ACE-Zero (no bridging, no PGO)
#   Ours   - ACE-Zero + VGGT bridging + self-supervised focal freeze
#   COLMAP - the published COLMAP pseudo-GT poses (Brachmann & Sattler,
#            visloc_pseudo_gt_limitations), converted to ACE0 pose logs in
#            reconstructions/colmap_controls/poses_colmap_7scenes_<scene>.txt
#
# Scenes ordered small to large so results arrive early. Full runtime is
# multiple days on one RTX-3090-class GPU (2k-12k frames per scene).
#
# Usage:
#   bash scripts/reproduce_7scenes.sh
#   SCENES="chess" CONFIGS="ours" bash scripts/reproduce_7scenes.sh

set -u
export PYTHONNOUSERSITE=1
cd "$(dirname "$0")/.."

SCENES=${SCENES:-"heads stairs fire chess pumpkin office redkitchen"}
CONFIGS=${CONFIGS:-"ours ace0 colmap"}
OUT_ROOT=${OUT_ROOT:-reconstructions/reproduce_7scenes}
BENCH_ROOT=${BENCH_ROOT:-benchmark/reproduce_7scenes}

benchmark () {
  local scene=$1 pose_file=$2 bench_dir=$3
  mkdir -p "${bench_dir}"
  conda run --no-capture-output -n nerfstudio env PYTHONNOUSERSITE=1 \
    python -m benchmarks.benchmark_poses \
    --pose_file "${pose_file}" --output_dir "${bench_dir}" \
    --images_glob_pattern "datasets/7scenes/${scene}/seq-*/*.color.png" \
    --split_json "split_files/7scenes_${scene}.json" --method nerfacto \
    > "${bench_dir}/log_eval.txt" 2>&1
  echo "=== ${scene} $(basename ${bench_dir}) bench exit $? $(date) ==="
}

reconstruct () {
  local scene=$1 config=$2 extra=$3
  local out="${OUT_ROOT}/${scene}_${config}"
  mkdir -p "${out}"
  echo "=== ${scene} ${config} recon start $(date) ==="
  conda run --no-capture-output -n ace0 python ace_zero.py \
    --rgb_files "datasets/7scenes/${scene}/seq-*/*.color.png" \
    --results_folder "${out}" --try_seeds 5 --render_visualization False \
    --seed_parallel_workers 5 --export_point_cloud False --dense_point_cloud False \
    --enable_solver False --enable_pgo_gtsam False --enable_pgo_conf_est False \
    ${extra} \
    > "${out}/log.txt" 2>&1
  echo "=== ${scene} ${config} recon exit $? $(date) ==="
}

for scene in ${SCENES}; do
  for config in ${CONFIGS}; do
    case "${config}" in
      ace0)
        reconstruct "${scene}" ace0 "--enable_bridge False"
        benchmark "${scene}" "${OUT_ROOT}/${scene}_ace0/poses_final.txt" "${BENCH_ROOT}/${scene}_ace0"
        ;;
      ours)
        reconstruct "${scene}" ours "--enable_bridge True --bridge_mode vggt --calibration_freeze_rate 0.45"
        benchmark "${scene}" "${OUT_ROOT}/${scene}_ours/poses_final.txt" "${BENCH_ROOT}/${scene}_ours"
        ;;
      colmap)
        benchmark "${scene}" "reconstructions/colmap_controls/poses_colmap_7scenes_${scene}.txt" "${BENCH_ROOT}/${scene}_colmap"
        ;;
    esac
  done
done

echo "=== 7SCENES REPRODUCTION DONE $(date) ==="

python3 - <<PYEOF
import json, os
scenes = "${SCENES}".split()
configs = "${CONFIGS}".split()
root = "${BENCH_ROOT}"
print(f"{'scene':11s} " + " ".join(f"{c:>7s}" for c in configs))
tot = {c: 0.0 for c in configs}; cnt = {c: 0 for c in configs}
for s in scenes:
    row = [f"{s:11s}"]
    for c in configs:
        p = f"{root}/{s}_{c}/nerf_data/nerf_for_eval/nerfacto/run/eval.json"
        if os.path.exists(p):
            v = json.load(open(p))['results']['psnr']
            row.append(f"{v:7.2f}"); tot[c] += v; cnt[c] += 1
        else:
            row.append(f"{'---':>7s}")
    print(" ".join(row))
print(f"{'average':11s} " + " ".join(f"{tot[c]/cnt[c]:7.2f}" if cnt[c] else f"{'---':>7s}" for c in configs))
PYEOF

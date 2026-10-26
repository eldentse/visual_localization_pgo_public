#!/usr/bin/env bash
# Experiment runner for closing the T&T Advanced gap (exploration, not a
# reproduction script - frozen configurations live in reproduce_ours_v*.sh).
#
# Runs the current working tree with the v1 T&T flags plus EXTRA flags
# (argparse keeps the last occurrence, so EXTRA overrides v1 values).
# BENCH=1 also runs the nerfacto evaluation (~1h per scene); otherwise the
# reconstruction alone is scored by registration rate and ATE against COLMAP.
#
# Usage:
#   TAG=vggtfix SCENES="Temple" EXTRA="--focal_from_vggt True --refine_calibration False" \
#     bash scripts/exp_t2adv.sh

set -u
export PYTHONNOUSERSITE=1
cd "$(dirname "$0")/.."

TAG=${TAG:?set TAG}
SCENES=${SCENES:-"Auditorium Ballroom Courtroom Palace Temple"}
EXTRA=${EXTRA:-""}
BENCH=${BENCH:-0}
OUT_ROOT=reconstructions/exp_t2adv/${TAG}
BENCH_ROOT=benchmark/exp_t2adv/${TAG}

V1_FLAGS="--try_seeds 5 --seed_parallel_workers 5 --render_visualization False \
--export_point_cloud False --dense_point_cloud False \
--enable_solver False --enable_pgo_gtsam False --enable_pgo_conf_est False \
--enable_bridge True --bridge_mode vggt --calibration_freeze_rate 0.45"

for scene in ${SCENES}; do
  out="${OUT_ROOT}/${scene}"
  glob="datasets/t2/advanced/${scene}/*.jpg"
  if [ ! -f "${out}/poses_final.txt" ]; then
    mkdir -p "${out}"
    printf 'commit %s (dirty: %s)\nflags %s %s\nstarted %s\n' "$(git rev-parse HEAD)" \
      "$(git status --porcelain --untracked-files=no | wc -l)" "${V1_FLAGS}" "${EXTRA}" "$(date)" > "${out}/exp_config.txt"
    echo "=== ${TAG}/${scene} recon start $(date) ==="
    conda run --no-capture-output -n ace0 python ace_zero.py \
      --rgb_files "${glob}" --results_folder "${out}" ${V1_FLAGS} ${EXTRA} \
      > "${out}/log.txt" 2>&1
    echo "=== ${TAG}/${scene} recon exit $? $(date) ==="
  fi
  if [ "${BENCH}" = 1 ] && [ -f "${out}/poses_final.txt" ] \
     && [ ! -f "${BENCH_ROOT}/${scene}/nerf_data/nerf_for_eval/nerfacto/run/eval.json" ]; then
    mkdir -p "${BENCH_ROOT}/${scene}"
    conda run --no-capture-output -n nerfstudio env PYTHONNOUSERSITE=1 \
      python -m benchmarks.benchmark_poses \
      --pose_file "${out}/poses_final.txt" --output_dir "${BENCH_ROOT}/${scene}" \
      --images_glob_pattern "${glob}" --method nerfacto \
      > "${BENCH_ROOT}/${scene}/log_eval.txt" 2>&1
    echo "=== ${TAG}/${scene} bench exit $? $(date) ==="
  fi
done

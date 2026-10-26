#!/usr/bin/env bash
# Experiment runner (exploration, not a reproduction script - frozen
# configurations live in reproduce_ours_v*.sh).
#
# Runs the current working tree with the Ours v3 flags of each run id plus
# EXTRA flags (argparse keeps the last occurrence, so EXTRA overrides v3).
# BENCH=1 also runs the nerfacto evaluation; MAX_RES overrides its
# max_resolution (default 640, as in the reproduction scripts).
#
# Usage:
#   TAG=seed_norefine RUNS="7scenes/stairs mip360/bicycle" \
#     EXTRA="--seed_from_vggt 48 --refinement none" BENCH=1 bash scripts/exp_run.sh
#
# Run ids: mip360/<scene>, 7scenes/<scene>, t2/<split>/<Scene>.

set -u
export PYTHONNOUSERSITE=1
cd "$(dirname "$0")/.."

TAG=${TAG:?set TAG}
RUNS=${RUNS:?set RUNS}
EXTRA=${EXTRA:-""}
BENCH=${BENCH:-0}
MAX_RES=${MAX_RES:-640}
OUT_ROOT=reconstructions/exp/${TAG}
BENCH_ROOT=benchmark/exp/${TAG}

COMMON_FLAGS="--try_seeds 5 --seed_parallel_workers 5 --render_visualization False \
--export_point_cloud False --dense_point_cloud False \
--enable_solver False --enable_pgo_gtsam False --enable_pgo_conf_est False \
--vggt_rescue_below_rate 0.8 --vggt_rescue_frames 48"

v3_flags () {
  case "$1" in
    mip360/bicycle) echo "--enable_bridge False --calibration_refine_max_iteration 3" ;;
    mip360/stump)   echo "--enable_bridge True --bridge_mode vggt --calibration_freeze_rate 0.45" ;;
    mip360/*)       echo "--enable_bridge True --bridge_mode vggt" ;;
    7scenes/*)      echo "--enable_bridge True --bridge_mode vggt --calibration_freeze_rate 0.45" ;;
    t2/*)           echo "--enable_bridge True --bridge_mode vggt" ;;
  esac
}

rgb_glob () {
  case "$1" in
    mip360/*)  echo "datasets/mip360/${1#mip360/}/images_4/*.JPG" ;;
    7scenes/*) echo "datasets/7scenes/${1#7scenes/}/seq-*/*.color.png" ;;
    t2/*)      echo "datasets/t2/${1#t2/}/*.jpg" ;;
  esac
}

split_args () {
  case "$1" in
    mip360/*)  echo "--split_json split_files/mip360_${1#mip360/}.json" ;;
    7scenes/*) echo "--split_json split_files/7scenes_${1#7scenes/}.json" ;;
    t2/*)      echo "" ;;
  esac
}

for id in ${RUNS}; do
  name=${id//\//_}
  out="${OUT_ROOT}/${name}"
  bench="${BENCH_ROOT}/${name}"
  glob=$(rgb_glob "${id}")
  flags="${COMMON_FLAGS} $(v3_flags "${id}") ${EXTRA}"
  if [ ! -f "${out}/poses_final.txt" ]; then
    mkdir -p "${out}"
    printf 'commit %s (dirty: %s)\nflags %s\nstarted %s\n' "$(git rev-parse HEAD)" \
      "$(git status --porcelain --untracked-files=no | wc -l)" "${flags}" "$(date)" > "${out}/exp_config.txt"
    echo "=== ${TAG}/${id} recon start $(date) ==="
    conda run --no-capture-output -n ace0 python ace_zero.py \
      --rgb_files "${glob}" --results_folder "${out}" ${flags} > "${out}/log.txt" 2>&1
    echo "=== ${TAG}/${id} recon exit $? $(date) ==="
  fi
  if [ "${BENCH}" = 1 ] && [ -f "${out}/poses_final.txt" ] \
     && [ ! -f "${bench}/nerf_data/nerf_for_eval/nerfacto/run/eval.json" ]; then
    mkdir -p "${bench}"
    conda run --no-capture-output -n nerfstudio env PYTHONNOUSERSITE=1 \
      python -m benchmarks.benchmark_poses \
      --pose_file "${out}/poses_final.txt" --output_dir "${bench}" \
      --images_glob_pattern "${glob}" $(split_args "${id}") --method nerfacto \
      --max_resolution "${MAX_RES}" > "${bench}/log_eval.txt" 2>&1
    echo "=== ${TAG}/${id} bench exit $? $(date) ==="
  fi
done

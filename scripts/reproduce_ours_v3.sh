#!/usr/bin/env bash
# Frozen reproduction of "Ours" v3 = v2 with the focal freeze removed on
# Tanks & Temples. v1 and v2 stay reproducible on their own pins
# (reproduce_ours_v1.sh, reproduce_ours_v2.sh).
#
# T&T vs vanilla ACE0 (nerfacto PSNR): training 16.98 vs 16.62, intermediate
# 17.62 vs 16.74, advanced 15.23 vs 13.57 (v2: 17.02 / 17.15 / 15.10, or
# ~16.60 on training with Ignatius re-evaluated, see below).
#
# Why: on T&T the focal starts high (heuristic init) and descends ~2%/iteration
# toward the truth; the self-supervised freeze (stop once the focal moves <3%
# per iteration) fires partway down that slope. Over all 17 T&T scenes the
# frozen focal ends further from COLMAP's than ACE0's never-frozen refinement
# (mean |error| 5.7% vs 3.6%): Auditorium +15.7% vs +3.2%, Courtroom +7.4% vs
# +1.4%, Lighthouse -8.6% vs -0.4%, Meetingroom +4.7% vs +0.2% (v1's -2.4 dB
# regression there). mip360 and 7-Scenes keep their v2 focal handling.
#
# Code is identical to v2 (only flags change), so v3 pins the same commit.
#
# Reproducibility: all four configurations (reproduce_ours_v1.sh ... v4.sh)
# run on this repository's code. Each is a set of ace_zero.py flags and every
# later feature is opt-in (off by default), so v1-v3 behave as they did when
# they were frozen during development. Do NOT edit the flags below to try new
# ideas: copy this file to a new version instead.
#
# Results: reconstructions/ours_v3/<run>/ and benchmark/ours_v3/<run>/ (PSNR in
# .../nerf_data/nerf_for_eval/nerfacto/run/eval.json). Finished runs are
# skipped, so re-launching resumes. ACE0 training is not bit-deterministic:
# expect a few tenths of a dB of run-to-run jitter. The summary prints the
# reference values next to each result.
#
# Runtime on one RTX 3090: ~1-2h per run (reconstruction + nerfacto),
# ~2 days for all 31 runs.
#
# Usage:
#   bash scripts/reproduce_ours_v3.sh                         # everything
#   DATASETS="t2" bash scripts/reproduce_ours_v3.sh           # one dataset
#   ONLY="t2/advanced/Temple t2/advanced/Palace" bash scripts/reproduce_ours_v3.sh
#   DRY_RUN=1 bash scripts/reproduce_ours_v3.sh               # set up, print commands
#   SUMMARY_ONLY=1 bash scripts/reproduce_ours_v3.sh          # print the table
#
# Run ids: mip360/<scene>, 7scenes/<scene>, t2/<split>/<Scene>.

set -u
export PYTHONNOUSERSITE=1

REPO=$(cd "$(dirname "$0")/.." && pwd)
# absolute, so outputs land in this repository wherever the script is called from
OUT_ROOT=$(realpath -m "${OUT_ROOT:-${REPO}/reconstructions/ours_v3}")
BENCH_ROOT=$(realpath -m "${BENCH_ROOT:-${REPO}/benchmark/ours_v3}")
DATASETS=${DATASETS:-"mip360 7scenes t2"}
ONLY=${ONLY:-""}
DRY_RUN=${DRY_RUN:-0}
SUMMARY_ONLY=${SUMMARY_ONLY:-0}

ALL_RUNS="
mip360/bicycle mip360/bonsai mip360/counter mip360/garden mip360/kitchen mip360/room mip360/stump
7scenes/heads 7scenes/stairs 7scenes/fire 7scenes/chess 7scenes/pumpkin 7scenes/office 7scenes/redkitchen
t2/training/Barn t2/training/Caterpillar t2/training/Church t2/training/Ignatius t2/training/Meetingroom t2/training/Truck
t2/intermediate/Family t2/intermediate/Francis t2/intermediate/Horse t2/intermediate/Lighthouse t2/intermediate/Playground t2/intermediate/Train
t2/advanced/Auditorium t2/advanced/Ballroom t2/advanced/Courtroom t2/advanced/Palace t2/advanced/Temple
"

# === the configuration =======================================================

COMMON_FLAGS="--try_seeds 5 --seed_parallel_workers 5 --render_visualization False \
--export_point_cloud False --dense_point_cloud False \
--enable_solver False --enable_pgo_gtsam False --enable_pgo_conf_est False \
--vggt_rescue_below_rate 0.8 --vggt_rescue_frames 48"

ours_flags () {
  case "$1" in
    # bicycle: no bridging - its reference predates bridging, and bridging would
    # fire here (registration stays <90% for 7 iterations); focal frozen after
    # iteration 3 because the self-supervised freeze fires too late on it
    mip360/bicycle) echo "--enable_bridge False --calibration_refine_max_iteration 3" ;;
    # stump: self-supervised focal freeze at 45% registration
    mip360/stump)   echo "--enable_bridge True --bridge_mode vggt --calibration_freeze_rate 0.45" ;;
    # remaining mip360 scenes: focal error stays <4% without a freeze
    mip360/*)       echo "--enable_bridge True --bridge_mode vggt" ;;
    7scenes/*)      echo "--enable_bridge True --bridge_mode vggt --calibration_freeze_rate 0.45" ;;
    # T&T: no focal freeze (see header)
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

split_args () {   # held-out images for the nerfacto evaluation
  case "$1" in
    mip360/*)  echo "--split_json split_files/mip360_${1#mip360/}.json" ;;
    7scenes/*) echo "--split_json split_files/7scenes_${1#7scenes/}.json" ;;
    t2/*)      echo "" ;;   # default: every 8th image
  esac
}

# === assets ==================================================================

check_assets () {   # untracked inputs: see README, Installation and Data preparation
  for asset in datasets split_files vggt; do
    [ -e "${REPO}/${asset}" ] || echo "warning: ${asset}/ not found in ${REPO}"
  done
}

snapshot_envs () {   # record package versions once, in case the envs drift later
  mkdir -p "${OUT_ROOT}/env"
  for env in ace0 nerfstudio vggt; do
    [ -s "${OUT_ROOT}/env/${env}.txt" ] || conda list -n "${env}" > "${OUT_ROOT}/env/${env}.txt" 2>&1
  done
}

# === runs ====================================================================

selected () {
  case " ${DATASETS} " in *" ${1%%/*} "*) ;; *) return 1 ;; esac
  [ -z "${ONLY}" ] && return 0
  case " ${ONLY} " in *" $1 "*) return 0 ;; esac
  return 1
}

move_aside () {   # keep an interrupted run for inspection instead of deleting it
  [ -d "$1" ] && mv "$1" "$1.partial-$(date +%Y%m%d-%H%M%S)"
  return 0
}

run_one () {
  local id=$1 name=${1//\//_}
  local out="${OUT_ROOT}/${name}" bench="${BENCH_ROOT}/${name}"
  local flags="${COMMON_FLAGS} $(ours_flags "${id}")"
  local glob; glob=$(rgb_glob "${id}")

  if [ -f "${bench}/nerf_data/nerf_for_eval/nerfacto/run/eval.json" ]; then
    echo "=== ${id} already done, skipping ==="; return
  fi
  if [ "${DRY_RUN}" = 1 ]; then
    echo "[${id}] python ace_zero.py --rgb_files \"${glob}\" --results_folder ${out} ${flags}"
    echo "[${id}] python -m benchmarks.benchmark_poses --pose_file ${out}/poses_final.txt --output_dir ${bench} --images_glob_pattern \"${glob}\" $(split_args "${id}") --method nerfacto"
    return
  fi

  if [ ! -f "${out}/poses_final.txt" ]; then
    move_aside "${out}"
    mkdir -p "${out}"
    printf 'commit %s\nrgb_files %s\nflags %s\nstarted %s on %s\n' \
      "$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null || echo none)" "${glob}" "${flags}" "$(date)" "$(hostname)" > "${out}/ours_v3_config.txt"
    echo "=== ${id} recon start $(date) ==="
    conda run --no-capture-output -n ace0 python ace_zero.py \
      --rgb_files "${glob}" --results_folder "${out}" ${flags} \
      > "${out}/log.txt" 2>&1
    echo "=== ${id} recon exit $? $(date) ==="
    if [ ! -f "${out}/poses_final.txt" ]; then
      echo "=== ${id} FAILED: no poses_final.txt, see ${out}/log.txt ==="; return
    fi
  fi

  move_aside "${bench}"
  mkdir -p "${bench}"
  conda run --no-capture-output -n nerfstudio env PYTHONNOUSERSITE=1 \
    python -m benchmarks.benchmark_poses \
    --pose_file "${out}/poses_final.txt" --output_dir "${bench}" \
    --images_glob_pattern "${glob}" $(split_args "${id}") --method nerfacto \
    > "${bench}/log_eval.txt" 2>&1
  echo "=== ${id} bench exit $? $(date) ==="
}

summary () {
  local runs=""
  for id in ${ALL_RUNS}; do selected "${id}" && runs="${runs} ${id}"; done
  python3 - "${BENCH_ROOT}" "${runs}" <<'PYEOF'
import json, os, sys
root, runs = sys.argv[1], sys.argv[2].split()
# Reference PSNR (ours, vanilla ACE0) under the same nerfacto evaluation.
# mip360 / 7-Scenes runs share the v2 configuration, so their references are
# the v2 ones; T&T references are the v3 runs (2026-10-17/18).
# Ignatius: nerfacto evaluation spread is large here. Four evaluations of
# practically identical poses (v2 vs v3: 0.2cm / 0.04deg apart, same focal
# and training frames) gave 18.41 (2026-09-11), 15.85, 15.51 and 15.95
# (2026-10-17/19); expect ~15.8 rather than 18.41 when rerunning.
# The v3 reference is the mean of the two evaluations of its own poses.
REF = {
    "mip360/bicycle": (18.66, 18.22), "mip360/bonsai": (25.10, 24.94),
    "mip360/counter": (23.87, 23.71), "mip360/garden": (19.65, 19.60),
    "mip360/kitchen": (24.33, 24.07), "mip360/room": (25.41, 19.05),
    "mip360/stump": (20.68, 19.91),
    "7scenes/heads": (18.41, 18.13), "7scenes/stairs": (17.16, 17.40),
    "7scenes/fire": (22.07, 22.09), "7scenes/chess": (22.77, 22.88),
    "7scenes/pumpkin": (23.69, 22.27), "7scenes/office": (21.07, 21.35),
    "7scenes/redkitchen": (20.68, 20.49),
    "t2/training/Barn": (20.15, 16.62), "t2/training/Caterpillar": (16.15, 16.50),
    "t2/training/Church": (15.33, 14.76), "t2/training/Ignatius": (15.73, 17.10),
    "t2/training/Meetingroom": (16.38, 16.80), "t2/training/Truck": (18.13, 17.95),
    "t2/intermediate/Family": (16.68, 14.06), "t2/intermediate/Francis": (19.73, 19.46),
    "t2/intermediate/Horse": (17.88, 18.68), "t2/intermediate/Lighthouse": (16.63, 16.53),
    "t2/intermediate/Playground": (19.19, 16.08), "t2/intermediate/Train": (15.63, 15.61),
    "t2/advanced/Auditorium": (15.86, 15.64), "t2/advanced/Ballroom": (16.31, 16.27),
    "t2/advanced/Courtroom": (15.28, 15.23), "t2/advanced/Palace": (12.72, 10.94),
    "t2/advanced/Temple": (16.00, 9.75),
}
# office: median of three v1 runs (17.59, 21.30, 21.07). In every bridged run
# the pose refiner thrashes (44-92cm avg moves) at the start of the mapping
# round after bridging lifts registration 42% -> 99%; 2 of 3 recovered, the
# 17.59 run locked in ~34cm of drift and warped the map. Vanilla ACE0 never
# exceeds ~3cm there. Expect occasional failures on this scene.
STALE = set()   # references that are not results of this configuration

def psnr(rid):
    p = f"{root}/{rid.replace('/', '_')}/nerf_data/nerf_for_eval/nerfacto/run/eval.json"
    return json.load(open(p))['results']['psnr'] if os.path.exists(p) else None

groups = {}
for rid in runs:
    groups.setdefault(rid.rsplit('/', 1)[0], []).append(rid)
f = lambda v: f"{v:7.2f}" if v is not None else f"{'---':>7s}"
for g, ids in groups.items():
    print(f"-- {g}")
    print(f"{'scene':12s} {'v3':>7s} {'ref':>8s} {'ACE0':>7s} {'v3-ACE0':>8s}")
    done = []
    for rid in ids:
        v = psnr(rid); ours, ace0 = REF[rid]
        ref = f"{ours:7.2f}{'*' if rid in STALE else ' '}"
        d = f"{v - ace0:+8.2f}" if v is not None else f"{'':8s}"
        print(f"{rid.rsplit('/', 1)[1]:12s} {f(v)} {ref} {ace0:7.2f} {d}")
        if v is not None: done.append((v, ace0))
    if done:
        n = len(done); mv = sum(x for x, _ in done) / n; ma = sum(y for _, y in done) / n
        print(f"{'average':12s} {mv:7.2f} {'':8s} {ma:7.2f} {mv - ma:+8.2f}   ({n}/{len(ids)} runs)")
if STALE: print("* reference is not a result of this configuration")
PYEOF
}

if [ "${SUMMARY_ONLY}" != 1 ]; then
  for id in ${ONLY}; do
    case " $(echo ${ALL_RUNS}) " in *" ${id} "*) ;; *) echo "unknown run id: ${id}"; exit 1 ;; esac
  done
  mkdir -p "${OUT_ROOT}" "${BENCH_ROOT}"
  exec 9> "${OUT_ROOT}/.lock"
  flock -n 9 || { echo "another reproduce_ours_v3.sh is already running"; exit 1; }
  check_assets
  snapshot_envs
  cd "${REPO}"
  for id in ${ALL_RUNS}; do
    selected "${id}" && run_one "${id}"
  done
  echo "=== OURS v3 DONE $(date) ==="
fi
summary

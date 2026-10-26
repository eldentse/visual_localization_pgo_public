#!/usr/bin/env bash



# folder with image files
datasets_folder="datasets/7scenes"
# output directory for the reconstruction
out_dir="reconstructions/7scenes_pgo_sam"


# 7Scenes comes with a pre-defined train/test split, split files expected in this folder
# generate using create_7scenes_split.py
benchmarking_split_folder="split_files"

## 7 Scenes
# scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")

## Mip360
scenes=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump")


for scene in ${scenes[*]}; do
  # find color images for the reconstruction
  # input_rgb_files="${datasets_folder}/${scene}/seq-01"

  # 7 Scenes
  input_rgb_files="${datasets_folder}/${scene}/seq-*"

  scene_out_dir="${out_dir}/${scene}"
  POSE_FILE="${scene_out_dir}/poses_final.txt"
  BENCHMARK_FOLDER="${scene_out_dir}/benchmark_folder"


  mkdir -p ${BENCHMARK_FOLDER}

  python -m benchmarks.benchmark_poses \
  --pose_file ${POSE_FILE} \
  --output_dir ${BENCHMARK_FOLDER} \
  --images_glob_pattern "${input_rgb_files}/*.color.png" \
  --method nerfacto 2>&1 | tee ${BENCHMARK_FOLDER}/log_NeRF.txt

done
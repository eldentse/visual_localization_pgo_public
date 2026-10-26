#!/usr/bin/env bash



# folder with image files
datasets_folder="datasets/mip360"
# output directory for the reconstruction
# out_dir="reconstructions/mip360_pgo_w_conf"
out_dir="output/mip360"


# 7Scenes comes with a pre-defined train/test split, split files expected in this folder
# generate using create_7scenes_split.py
benchmarking_split_folder="split_files"


## Mip360
# scenes=("bicycle" "bonsai" "counter" "garden" "kitchen" "room" "stump")
scenes=("bonsai" "counter" "garden" "kitchen" "room" "stump")


for scene in ${scenes[*]}; do
  # find color images for the reconstruction
  input_rgb_files="${datasets_folder}/${scene}/images_4/*.JPG"
  scene_out_dir="${out_dir}/${scene}"
  POSE_FILE="${scene_out_dir}/pose.txt"
  BENCHMARK_FOLDER="${scene_out_dir}/benchmark_folder"


  mkdir -p ${BENCHMARK_FOLDER}

  python -m benchmarks.benchmark_poses \
  --pose_file ${POSE_FILE} \
  --output_dir ${BENCHMARK_FOLDER} \
  --images_glob_pattern "${input_rgb_files}" \
  --method nerfacto 2>&1 | tee ${BENCHMARK_FOLDER}/log_NeRF.txt

done
#!/usr/bin/env bash

reconstruction_exe="VGGT-SLAM-public/main.py"

# folder with image files
datasets_folder="datasets/7scenes"
# output directory for the reconstruction
out_dir="reconstructions/7scenes_se3"
# target directory for benchmarking results
benchmarking_out_dir="benchmark/7scenes_se3"

# render visualization of the reconstruction, slows down reconstruction considerably
render_visualization=false

# run view synthesis benchmarking after reconstruction
run_benchmark=true
# benchmarking needs to happen in the nerfstudio environment
benchmarking_environment="nerfstudio"
# benchmarking method, splatfacto or nerfacto
benchmarking_method="nerfacto"
# when using splatfacto, we need a point cloud initialization, either sparse or dense
# dense is recommended if you have very dense coverage of the scene, eg. > 2000 images
benchmarking_dense_pcinit=true

# 7Scenes comes with a pre-defined train/test split, split files expected in this folder
# generate using create_7scenes_split.py
benchmarking_split_folder="split_files"

scenes=("chess" "fire" "heads" "office" "pumpkin" "redkitchen" "stairs")

for scene in ${scenes[*]}; do
  # find color images for the reconstruction
  input_rgb_files="${datasets_folder}/${scene}/seq-01/"
  scene_out_dir="${out_dir}/${scene}"

  mkdir -p ${scene_out_dir}

  # run ACE0 reconstruction
  python $reconstruction_exe --image_folder "${input_rgb_files}" --max_loops 1 --log_results --log_path "${scene_out_dir}/pose.txt" --skip_dense_log --use_sim3 2>&1 | tee ${scene_out_dir}/log_${scene}.txt

  # # run benchmarking if requested
  # if $run_benchmark; then
  #   benchmarking_scene_dir="${benchmarking_out_dir}/${scene}"
  #   mkdir -p ${benchmarking_scene_dir}
  #   conda run --no-capture-output -n ${benchmarking_environment} python -m benchmarks.benchmark_poses --pose_file ${scene_out_dir}/poses_final.txt --output_dir ${benchmarking_scene_dir} --images_glob_pattern "${input_rgb_files}" --split_json ${benchmarking_split_folder}/7scenes_${scene}.json --method ${benchmarking_method} 2>&1 | tee ${benchmarking_out_dir}/log_${scene}.txt
  # fi
done
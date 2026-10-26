#!/usr/bin/env python3

import dataset_io
import argparse
import logging
from termcolor import colored
import cv2
from PIL import Image
import torch
import os

import sys

from distutils.util import strtobool
from torchvision import transforms as TF

from vggt_slam.solver import Solver
from vggt_slam.submap import Submap
from vggt_slam.loop_closure import ImageRetrieval

from scipy.spatial.transform import Rotation as R
from skimage import io
import numpy as np

_logger = logging.getLogger(__name__)

def _strtobool(x):
    return bool(strtobool(x))

def load_and_preprocess_images_size(image_path_list, target_size=518, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        img = Image.open(image_path)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # Make divisible by 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images

def load_camera_info(file_path, scale = 1, invert=False):
    poses = []
    image_paths = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 10:
                continue  # skip malformed lines
            image_path, qw, qx, qy, qz, x, y, z, _, _ = parts
            # quat = [float(qw), float(qx), float(qy), float(qz)]
            quat = [float(qx), float(qy), float(qz), float(qw)]
            trans = [float(x), float(y), float(z)]
            trans = [t * scale for t in trans]

            # Convert quaternion to rotation matrix
            rot = R.from_quat(quat).as_matrix()

            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = rot
            T[:3, 3] = trans

            if invert:
                T = np.linalg.inv(T)

            poses.append(T[:3, :])
            image_paths.append(image_path)

    return np.array(poses), image_paths

def load_pointmaps(image_files, npz_files, image_size=(518, 294)): # width, height
    pointmaps = []
    pointmaps_conf = []
    intrinsics = []

    for image_file in image_files:
        npz_name = os.path.basename(image_file.replace('.png', '.npz'))
        folder = os.path.basename(os.path.dirname(image_file))
        npz_file = [f for f in npz_files if f"{folder}/{npz_name}" in f]
        if len(npz_file) == 0:
            raise FileNotFoundError(f"Could not find npz file for image {image_file} in {folder} npz name {npz_name}")
        data = np.load(npz_file[0], allow_pickle=True)
        data = dict(data)
        reordered = np.transpose(data['pointmap'], (1, 2, 0))
        resized = cv2.resize(reordered, image_size, interpolation=cv2.INTER_LINEAR)
        pointmaps.append(resized)
        pointmaps_conf.append(np.ones(resized.shape, dtype=np.float32))
        intrinsics.append(data['intrinscis'])
        
    return np.array(pointmaps), np.array(pointmaps_conf), np.array(intrinsics)


if __name__ == "__main__":
    # Setup logging.
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('rgb_files', type=str, help="Glob pattern for RGB files, e.g. 'datasets/scene/*.jpg'")
    parser.add_argument("pointmap_files", type=str, help="Glob pattern for npz files, e.g. 'datasets/scene/*.npz'")
    parser.add_argument("--use_sim3", action="store_true", help="Use Sim3 instead of SL(4)")
    parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
    parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
    parser.add_argument('--image_resolution', type=int, default=480, help='base image resolution')
    parser.add_argument("--pose_log", type=str, help="log of the pose")
    

    opt = parser.parse_args()

    solver = Solver(
        init_conf_threshold=opt.conf_threshold,
        use_point_map=opt.use_point_map,
        use_sim3=opt.use_sim3,
        gradio_mode=False
    )

    # Extrinsics
    extrinsic, image_files = load_camera_info(opt.pose_log)

    # image_files = dataset_io.get_files_from_glob(opt.rgb_files)

    _logger.info("Loading Images")
    images = load_and_preprocess_images_size(image_files, target_size=opt.image_resolution)

    new_submap = Submap(0)
    _logger.info("Adding Images to submap")
    new_submap.add_all_frames(images)
    _logger.info("Setting Frame IDs")
    new_submap.set_frame_ids_modi(image_files)

    ids = new_submap.get_frame_ids()
    _logger.info(f"number of ids: {len(ids)}")
    _logger.info(f"number of unique: {len(set(ids))}")

    new_submap.set_all_retrieval_vectors(solver.image_retrieval.get_all_submap_embeddings(new_submap))

    solver.current_working_submap = new_submap

    detected_loops = solver.image_retrieval.find_loop_closures(solver.map, new_submap, max_loop_closures=1)
    if len(detected_loops) > 0:
        print(colored("detected_loops", "yellow"), detected_loops)
    else:
        print("no loop detected")
    

    
    height, width = images.numpy().shape[2:]

    npz_files = dataset_io.get_files_from_glob(opt.pointmap_files)
    world_points, world_points_conf, intrinsics = load_pointmaps(image_files, npz_files, (width, height))
    data = {}
    data["images"] = images.numpy()
    data["extrinsic"] = extrinsic
    data["intrinsic"] = intrinsics
    data["world_points"] = world_points
    data["world_points_conf"] = world_points_conf
    data["detected_loops"] = detected_loops

    print(f"images shape: {data['images'].shape}")
    print(f"extrinsic shape: {extrinsic.shape}")
    print(f"intrinsics shape: {intrinsics.shape}")
    print(f"world_points shape: {world_points.shape}")
    print(f"world_points_conf shape: {world_points_conf.shape}")

    print(f"first intrinsics: \n{intrinsics[0]}")

    solver.add_points(data)

    solver.graph.optimize()
    solver.map.update_submap_homographies(solver.graph)


    focal_lengths=[data["intrinsic"][:, 0, 0]]
    solver.map.write_poses_to_acezero_pose_file(opt.pose_log, image_files, focal_lengths, 1)
    # solver.map.write_poses_to_acezero_pose_file("test.txt", image_files, focal_lengths, 1)
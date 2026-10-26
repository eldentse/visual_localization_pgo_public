#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2023.

import logging
import shutil
from pathlib import Path
import os
import sys
import numpy as np
import argparse
from distutils.util import strtobool
import time
import ace_zero_util as zutil
from joblib import Parallel, delayed

import dataset_io

SOLVER_EXE = "./solver.py"
PGO_EXE = "./pgo_gtsam.py"
BRIDGE_EXE = "./bridge_registration.py"
VGGT_BRIDGE_EXE = "./vggt_bridge.py"

_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':

    # Setup logging levels.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Run ACE0 for a dataset or a scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--rgb_files', type=str, help="Glob pattern for RGB files, e.g. 'datasets/scene/*.jpg'")

    parser.add_argument('--results_folder', type=Path, help='path to output folder for result files')

    parser.add_argument('--depth_files', type=str, default=None,
                        help="Which depth to use for the seed image, Glob pattern. "
                             "Correspondence to rgb files via alphateical ordering. "
                             "None: estimate depth using ZoeDepth")

    # === Main reconstruction loop =====================================================================================

    parser.add_argument('--iterations_max', type=int, default=100,
                        help="Maximum number of ACE0 iterations, ie mapping and relocalisation rounds.")
    parser.add_argument('--registration_threshold', type=float, default=0.99,
                        help="Stop reconstruction when this ratio of images has been registered.")
    parser.add_argument('--relative_registration_threshold', type=float, default=0.01,
                        help="Stop reconstruction when less percent of images was registered wrt the last iteration.")
    parser.add_argument('--final_refine', type=_strtobool, default=True,
                        help="One more round of mapping when the stopping criteria have been met.")
    parser.add_argument('--final_refit', type=_strtobool, default=True,
                        help="Refit new (uninitialised) network in last iteration without early stopping")
    parser.add_argument('--final_refit_posewait', type=int, default=5000,
                        help="Fix poses for the first n training iterations of the final refit.")
    parser.add_argument('--refit_iterations', type=int, default=25000,
                        help='Number of training iterations for the final refit.')
    parser.add_argument('--registration_confidence', type=int, default=500,
                        help="Consider an image registered if it has this many inlier scene coordinates.")

    parser.add_argument('--try_seeds', type=int, default=5,
                        help="Number of random images to try when starting the reconstruction.")
    parser.add_argument('--seed_parallel_workers', type=int, default=3,
                        help="Number of parallel workers for seed mapping. "
                             "ZoeDepth might be a limiting factor in terms of GPU memory. "
                             "-1 -> all available cores, 1 -> no parallelism.")
    parser.add_argument('--seed_iterations', type=int, default=10000,
                        help="Maximum number of ACE training iterations for seed images.")

    parser.add_argument('--seed_network', type=Path, default=None,
                        help="Path to a pre-trained network to start the reconstruction.")

    parser.add_argument('--warmstart', type=_strtobool, default=True,
                        help="For each ACE0 mapping round, load the ACE weights of the last iteration.")

    parser.add_argument('--export_point_cloud', type=_strtobool, default=False,
                        help="Export the ACE0 point cloud after reconstruction, "
                             "for visualisation or to initialise splats")

    parser.add_argument('--dense_point_cloud', type=_strtobool, default=False,
                        help='when exporting a point cloud, do not filter points based on reprojection error, '
                             'bad for visualisation but good to initialise splats')

    # === Pose refinement ==================================================================================================

    parser.add_argument('--refinement', type=str, default="mlp", choices=['mlp', 'none', 'naive'],
                        help="How to refine poses. MLP: refinement network. Naive: Backprop to poses.")
    parser.add_argument('--refinement_ortho', type=str, default="gram-schmidt", choices=['gram-schmidt', 'procrustes'],
                        help="How to orthonormalise rotations when refining poses.")
    parser.add_argument('--pose_refinement_wait', type=int, default=0,
                        help="Keep poses frozen for the first n training iterations of ACE.")
    parser.add_argument('--pose_refinement_lr', type=float, default=0.001,
                        help="Learning rate for pose refinement.")

    # === Calibration refinement ===========================================================================================

    parser.add_argument('--refine_calibration', type=_strtobool, default=True,
                        help="Optimize focal length during mapping.")
    parser.add_argument('--calibration_refine_max_iteration', type=int, default=0,
                        help="Freeze focal length refinement after this ACE0 iteration "
                             "(0 = never freeze). Joint focal/pose refinement drifts the "
                             "focal upward after early iterations converge near the truth.")
    parser.add_argument('--calibration_freeze_rate', type=float, default=0,
                        help="Self-supervised focal freeze: freeze calibration refinement once the "
                             "registration rate reaches this value AND the focal has stabilised "
                             "(see --calibration_freeze_focal_delta). 0 disables. The refined focal "
                             "passes near the true value while coverage is still growing, then "
                             "drifts upward once refinement runs out of new constraints.")
    parser.add_argument('--calibration_freeze_focal_delta', type=float, default=0.03,
                        help="Focal stability condition for --calibration_freeze_rate: relative "
                             "focal change per iteration must be below this value.")
    parser.add_argument('--use_external_focal_length', type=float, default=-1,
                        help="Provide the focal length of images. Can be refined. "
                             "-1: Use 70%% of image diagonal as guess.")

    # === ACE Early Stopping ===============================================================================================

    parser.add_argument('--learning_rate_schedule', type=str, default="1cyclepoly",
                        choices=["circle", "constant", "1cyclepoly"],
                        help='circle: move from min to max to min, constant: stay at min, '
                             '1cyclepoly: linear approximation of 1cycle to support early stopping')
    parser.add_argument('--learning_rate_max', type=float, default=0.003, help="max learning rate of the lr schedule")
    parser.add_argument('--cooldown_iterations', type=int, default=5000,
                        help="train to min learning rate for this many iterations after early stopping criterion has been met")
    parser.add_argument('--cooldown_threshold', type=float, default=0.7,
                        help="Start cooldown after this percent of batch pixels are below the inlier reprojection error.")

    # === General ACE parameters ===========================================================================================

    parser.add_argument('--image_resolution', type=int, default=480,
                        help='base image resolution')
    parser.add_argument('--num_head_blocks', type=int, default=1,
                        help='depth of the regression head, defines the map size')
    parser.add_argument('--max_dataset_passes', type=int, default=10,
                        help='max number of repetition of mapping images (with different augmentations)')
    parser.add_argument('--repro_loss_type', type=str, default="tanh",
                        choices=["l1", "l1+sqrt", "l1+log", "tanh", "dyntanh"],
                        help='Loss function on the reprojection error. Dyn varies the soft clamping threshold')
    parser.add_argument('--repro_loss_hard_clamp', type=int, default=1000,
                        help='hard clamping threshold for the reprojection losses')
    parser.add_argument('--repro_loss_soft_clamp', type=int, default=50,
                        help='soft clamping threshold for the reprojection losses')
    parser.add_argument('--aug_rotation', type=int, default=15,
                        help='max inplane rotation angle')
    parser.add_argument('--num_data_workers', type=int, default=12,
                        help='number of data loading workers, set according to the number of available CPU cores')
    parser.add_argument('--training_buffer_cpu', type=_strtobool, default=False, 
                        help='store training buffer on CPU memory instead of GPU, '
                        'this allows running ACE0 on smaller GPUs, but is slower')

    # === Registration parameters ==========================================================================================

    parser.add_argument('--ransac_iterations', type=int, default=32,
                        help="Number of RANSAC hypothesis when registering mapping frames.")
    parser.add_argument('--ransac_threshold', type=float, default=10,
                        help='RANSAC inlier threshold in pixels')

    # === Visualisation parameters =========================================================================================

    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help="Render visualisation frames of the whole reconstruction process.")
    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help="Dataset images are 90deg flipped (like Wayspots).")
    parser.add_argument('--render_marker_size', type=float, default=0.03,
                        help="Size of the camera marker when rendering scenes.")
    parser.add_argument('--iterations_output', type=int, default=500,
                        help='how often to print the loss and render a frame')

    parser.add_argument('--random_seed', type=int, default=1305,
                        help='random seed, predominately used to select seed images')
    
    # === Pose Graph Optimization ===========================================================================================

    parser.add_argument('--enable_solver', type=_strtobool, default=False,
                        help="Enable Pose Graph Optmimization using SE3")

    parser.add_argument('--enable_pgo_gtsam', type=_strtobool, default=False,
                        help="Enable Pose Graph Optmimization GTSAM")

    parser.add_argument('--enable_pgo_conf_est', type=_strtobool, default=False,
                        help="Enable Pose Graph Optmimization GTSAM")

    parser.add_argument('--pgo_tau_start', type=float, default=0.2,
                        help="Initial geometric consistency threshold (m) for pose graph construction")

    parser.add_argument('--pgo_tau_end', type=float, default=0.05,
                        help="Final geometric consistency threshold (m) for pose graph construction")

    parser.add_argument('--pgo_tau_decay_iterations', type=int, default=10,
                        help="Number of iterations over which tau decays from start to end value")

    parser.add_argument('--pgo_loop_strides', type=str, default="20",
                        help="comma-separated strides for PGO loop closure candidate pairs")

    parser.add_argument('--pgo_skip_final', type=_strtobool, default=True,
                        help="Do not run PGO after the final mapping/refit round, so the "
                             "converged registration poses are the final output")

    # === Flow-bridged registration ========================================================================================

    parser.add_argument('--enable_bridge', type=_strtobool, default=False,
                        help="Bridge unregistered frames from registered ones "
                             "(rescues captures with visually disjoint sweeps where relocalization stalls)")

    parser.add_argument('--bridge_mode', type=str, default='vggt', choices=['vggt', 'flow'],
                        help="vggt: joint multi-view pose inference (works across disjoint sweeps). "
                             "flow: RAFT flow + PnP against a registered reference (fails when no "
                             "co-visible registered/unregistered pair exists)")

    parser.add_argument('--vggt_env', type=str, default='vggt',
                        help="conda environment with VGGT installed, for --bridge_mode vggt")

    parser.add_argument('--bridge_below_rate', type=float, default=0.9,
                        help="Only run bridging while the registration rate is below this value")

    parser.add_argument('--focal_from_vggt', type=_strtobool, default=False,
                        help="Estimate the focal length with one VGGT pass over sampled frames and "
                             "use it as the external focal length (within 1%% of calibration on "
                             "every scene measured, vs +3..17%% drift from joint refinement). "
                             "Combine with --refine_calibration False to hold it fixed.")

    parser.add_argument('--seed_from_vggt', type=int, default=0,
                        help="Seed the reconstruction with one joint VGGT pass over this many frames "
                             "spread over the collection (scaled to metric with ZoeDepth) instead of "
                             "single-image depth seeds; rescues scenes whose single-image seed never "
                             "grows (repetitive structure). 0 disables (default ACE0 seeding).")

    parser.add_argument('--vggt_rescue_below_rate', type=float, default=0,
                        help="If the reconstruction ends with fewer than this fraction of images registered, "
                             "redo it from a VGGT seed (--seed_from_vggt) without pose or focal refinement "
                             "and use that result. Only stalled scenes are affected. 0 disables.")
    parser.add_argument('--vggt_rescue_frames', type=int, default=48,
                        help="Frames in the VGGT seed of a rescue run (see --vggt_rescue_below_rate).")
    parser.add_argument('--vggt_seed_select', type=_strtobool, default=False,
                        help="Always run the VGGT-seeded reconstruction of --vggt_rescue_below_rate as well, "
                             "and keep whichever result has more registration inliers on average.")

    parser.add_argument('--ba_polish', type=_strtobool, default=False,
                        help="Polish the final poses and focal with a feature-based bundle adjustment "
                             "(COLMAP CLI, see ba_polish.py): ACE0's poses are kept as initialisation and "
                             "SIFT tracks between nearby views are triangulated and bundle-adjusted.")

    opt = parser.parse_args()
    # Print all arguments and their values
    for arg, value in vars(opt).items():
        print(f"{arg}: {value}")

    # create output directory
    os.makedirs(opt.results_folder, exist_ok=True)

    _logger.info(f"Starting reconstruction of files matching {opt.rgb_files}.")
    reconstruction_start_time = time.time()

    # We warm up the torch.hub cache and make sure the depth estimation model is available 
    # before we start the main ACE0 loop (ACE0 uses multiple processes for the initial seed
    # stage and the download should run only once).
    _logger.info(f"Downloading ZoeDepth model from the main process.")
    model = dataset_io.get_depth_model()
    del model
    _logger.info(f"Depth estimation model ready to use.")

    if opt.focal_from_vggt:
        _logger.info("Estimating focal length with VGGT.")
        import subprocess
        result = subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", opt.vggt_env, "python",
             str(Path(__file__).parent / "vggt_focal.py"), opt.rgb_files],
            capture_output=True, text=True, check=True,
            env=dict(os.environ, PYTHONNOUSERSITE='1'))
        focal_line = [l for l in result.stdout.splitlines() if l.startswith('FOCAL ')][-1]
        opt.use_external_focal_length = float(focal_line.split()[1])
        _logger.info(f"Using VGGT focal length estimate: {opt.use_external_focal_length:.2f}")

    if opt.seed_from_vggt > 0:
        # the seed pose log stands in for the registration of a seed network:
        # iteration 1 maps the VGGT frames from scratch (see vggt_seed.py)
        if opt.render_visualization:
            raise ValueError("--render_visualization is not supported with --seed_from_vggt")
        import subprocess
        import vggt_seed
        iteration_id = "iteration0_vggt"
        seed_npz = opt.results_folder / "vggt_seed.npz"
        _logger.info(f"Seeding with one joint VGGT pass over {opt.seed_from_vggt} frames.")
        subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", opt.vggt_env, "python",
             str(Path(__file__).parent / "vggt_seed.py"), opt.rgb_files, str(opt.seed_from_vggt), str(seed_npz)],
            check=True, env=dict(os.environ, PYTHONNOUSERSITE='1'))
        n_seed, seed_focal, seed_scale = vggt_seed.write_seed_pose_log(
            seed_npz, dataset_io.get_files_from_glob(opt.rgb_files),
            opt.results_folder / f"poses_{iteration_id}.txt")
        _logger.info(f"VGGT seed: {n_seed} frames, focal {seed_focal:.1f}, metric scale {seed_scale:.3f}.")
    elif opt.seed_network is not None:
        _logger.info(f"Using pre-trained network as seed: {opt.seed_network}")
        iteration_id = opt.seed_network.stem
    else:      
        # use individual images as seeds, try multiple and choose the one that registers the most images
        np.random.seed(opt.random_seed)
        seeds = np.random.uniform(size=opt.try_seeds)
        _logger.info(f"Trying seeds: {seeds}")

        # process seeds in parallel
        arg_instances = []
        for seed_idx, seed in enumerate(seeds):
            # show progress only for the first seed or if we are not using parallel workers
            verbose = (seed_idx == 0) or (opt.seed_parallel_workers == 1)
            arg_instances.append((seed_idx, seed, opt.rgb_files, opt.results_folder, opt, verbose, False, False))

        if opt.seed_parallel_workers != 1:
            _logger.info(f"Processing {len(arg_instances)} seeds in parallel.")

        # as we process initial seeds, keep track of their registration rates
        seed_reg_rates = Parallel(n_jobs=opt.seed_parallel_workers, verbose=1)(
            map(delayed(zutil.map_seed), arg_instances))

        for seed_idx, seed in enumerate(seeds):
            _logger.info(f"Seed {seed_idx}: {seed} -> {seed_reg_rates[seed_idx] * 100:.1f}%")

        # select the best seed
        best_seed = np.argmax(seed_reg_rates)
        iteration_id = zutil.get_seed_id(best_seed)

        _logger.info(f"Selected best seed {iteration_id} "
                     f"with registration rate: {seed_reg_rates[best_seed] * 100:.1f}%")

        # if a visualisation is requested, we need to re-map the best seed with visualisation enabled
        if opt.render_visualization:
            _logger.info(f"Re-mapping best seed {iteration_id} with visualisation enabled.")
            try:
                zutil.map_seed((best_seed, seeds[best_seed], opt.rgb_files, opt.results_folder, opt, True, True, True))
            except RuntimeError as e:
                # the re-map only produces visualisation frames; do not abort the reconstruction
                _logger.warning(f"Re-mapping best seed with visualisation failed, continuing without it: {e}")

    if opt.seed_from_vggt <= 0:
        _logger.info(f"Registering all images to best seed {iteration_id}.")

        # Register all images to the best seed. Also render visualisation if requested.
        # In some cases, this is redundant - when the dataset is small and the seed scoring already registered all images
        # AND no visualisation was requested. However, for small datasets, this is fast anyway.
        reg_cmd = [
            zutil.REGISTER_EXE,
            opt.rgb_files,
            opt.results_folder / f"{iteration_id}.pt",
            "--render_visualization", opt.render_visualization,
            "--render_target_path", zutil.get_render_path(opt.results_folder),
            "--render_marker_size", opt.render_marker_size,
            "--render_flipped_portrait", opt.render_flipped_portrait,
            "--session", f"{iteration_id}",
            "--confidence_threshold", opt.registration_confidence,
            "--use_external_focal_length", opt.use_external_focal_length,
            "--hypotheses", opt.ransac_iterations,
            "--threshold", opt.ransac_threshold,
            "--image_resolution", opt.image_resolution,
            "--num_data_workers", opt.num_data_workers,
            "--hypotheses_max_tries", 16
        ]
        zutil.run_cmd(reg_cmd)

    scheduled_to_stop_early = False
    prev_iteration_id = iteration_id
    calibration_frozen = False
    prev_focal = None

    # check the number of registered mapping images
    max_registration_rate = zutil.get_registration_rates(
        pose_file=opt.results_folder / f"poses_{iteration_id}.txt",
        thresholds=[opt.registration_confidence])[0]
    _logger.info(f"Best seed successfully registered {max_registration_rate * 100:.1f}% of mapping images.")

    # iterate mapping and registration starting from the best seed iteration
    for iteration in range(1, opt.iterations_max):

        iteration_id = f"iteration{iteration}"

        if scheduled_to_stop_early and opt.final_refit:
            # get full refitting mapping call
            mapping_cmd = zutil.get_refit_mapping_cmd(opt.rgb_files, iteration_id, opt.results_folder, opt)
        else:
            # get base mapping call
            mapping_cmd = zutil.get_base_mapping_cmd(opt.rgb_files, iteration_id, opt.results_folder, opt)

        # optionally freeze focal refinement after the early iterations: the
        # focal passes close to the true value early on, then drifts upward as
        # it compensates pose adjustments (and unmodeled lens distortion)
        refine_calibration = opt.refine_calibration and not calibration_frozen and (
            opt.calibration_refine_max_iteration <= 0 or iteration <= opt.calibration_refine_max_iteration)

        # setting parameters for mapping after initial seed
        mapping_cmd += [
            "--render_visualization", opt.render_visualization,
            "--use_ace_pose_file", f"{opt.results_folder}/poses_{prev_iteration_id}.txt",
            "--pose_refinement", opt.refinement,
            "--use_existing_vis_buffer", f"{prev_iteration_id}_register.pkl",
            "--refine_calibration", refine_calibration,
            "--num_data_workers", opt.num_data_workers,
        ]

        # load previous network weights starting from iteration 2, or if we started from a seed network
        if (opt.warmstart and iteration > 1) or (opt.warmstart and opt.seed_network is not None):
            # skip warmstart in final refit iteration
            if not (opt.final_refit and scheduled_to_stop_early):
                mapping_cmd += ["--load_weights", f"{opt.results_folder}/{prev_iteration_id}.pt"]

        zutil.run_cmd(mapping_cmd)

        # register all images to the current map
        _logger.info(f"Registering all images to map {iteration_id}.")

        reg_cmd = [
            zutil.REGISTER_EXE,
            opt.rgb_files,
            opt.results_folder / f"{iteration_id}.pt",
            "--render_visualization", opt.render_visualization,
            "--render_target_path", zutil.get_render_path(opt.results_folder),
            "--render_marker_size", opt.render_marker_size,
            "--session", iteration_id,
            "--confidence_threshold", opt.registration_confidence,
            "--render_flipped_portrait", opt.render_flipped_portrait,
            "--image_resolution", opt.image_resolution,
            "--hypotheses", opt.ransac_iterations,
            "--threshold", opt.ransac_threshold,
            "--num_data_workers", opt.num_data_workers,
            "--hypotheses_max_tries", 16
        ]

        # Mapping can abort without writing its pose file (e.g. NaN loss from a
        # poisoned pose); fall back to the previous iteration's poses instead
        # of crashing the whole reconstruction
        preliminary_pose_file = opt.results_folder / f"poses_{iteration_id}_preliminary.txt"
        if not preliminary_pose_file.exists():
            _logger.error(f"Mapping iteration {iteration_id} did not produce {preliminary_pose_file}. "
                          f"Stopping reconstruction; using poses of {prev_iteration_id} as final.")
            iteration_id = prev_iteration_id
            break

        # Get current focal length estimate from the pose file of the previous mapping iteration
        _, _, focal_lengths = dataset_io.load_dataset_ace(
            pose_file=preliminary_pose_file,
            confidence_threshold=opt.registration_confidence)

        # We support a single focal length.
        assert np.allclose(focal_lengths, focal_lengths[0])

        _logger.info("Passing previous focal length estimate to registration.")
        reg_cmd += ["--use_external_focal_length", focal_lengths[0]]

        zutil.run_cmd(reg_cmd)

        # check the number of registered mapping images
        registration_rate = zutil.get_registration_rates(
            pose_file=opt.results_folder / f"poses_{iteration_id}.txt",
            thresholds=[opt.registration_confidence])[0]

        _logger.info(f"Successfully registered {registration_rate*100:.1f}% of mapping images.")

        shutil.copy(opt.results_folder / f"poses_{iteration_id}.txt", opt.results_folder / f"poses_{iteration_id}_reg_map.txt")

        # self-supervised focal freeze: once coverage is substantial and the focal
        # has stopped moving between iterations, further "refinement" only drifts
        # it upward (compensating pose adjustments and unmodeled distortion)
        current_focal = float(focal_lengths[0])
        if (opt.calibration_freeze_rate > 0 and not calibration_frozen
                and registration_rate >= opt.calibration_freeze_rate
                and prev_focal is not None
                and abs(current_focal - prev_focal) / prev_focal < opt.calibration_freeze_focal_delta):
            calibration_frozen = True
            _logger.info(f"Freezing focal length refinement at {current_focal:.1f} "
                         f"(registration {registration_rate*100:.1f}%, "
                         f"focal change {(current_focal-prev_focal)/prev_focal*100:+.2f}%).")
        prev_focal = current_focal

        ## Call Solver.py object pose graph optimization
        if opt.enable_solver:
            solver_cmd = [
                SOLVER_EXE,
                opt.rgb_files,
                f"{opt.results_folder}/pointmaps/*/*.npz",
                "--image_resolution", opt.image_resolution,
                "--pose_log", f"{opt.results_folder}/poses_{iteration_id}.txt",
                "--use_point_map"
            ]
            zutil.run_cmd(solver_cmd)
            shutil.copy(opt.results_folder / f"poses_{iteration_id}.txt", opt.results_folder / f"poses_{iteration_id}_solver.txt")
            _logger.info(f"Successfully ran solver.")

        # skip PGO after the final mapping round: at convergence the registration
        # poses are maximally consistent with the trained scene model, and the
        # noisier pairwise measurements only perturb them (costs ~3.5dB PSNR)
        run_pgo_this_iteration = opt.enable_pgo_gtsam and not (scheduled_to_stop_early and opt.pgo_skip_final)

        if run_pgo_this_iteration:
            # progressively reduce the geometric consistency threshold to
            # enforce stricter correspondence checks as predictions improve
            tau_progress = min(1.0, (iteration - 1) / max(1, opt.pgo_tau_decay_iterations))
            pgo_tau = opt.pgo_tau_start + (opt.pgo_tau_end - opt.pgo_tau_start) * tau_progress
            pgo_cmd = [
                PGO_EXE,
                f"{opt.results_folder}/pointmaps/*/*.npz",
                "--pose_log", f"{opt.results_folder}/poses_{iteration_id}.txt",
                "--conf_est", f"{opt.enable_pgo_conf_est}",
                "--tau", f"{pgo_tau}",
                "--image_glob", opt.rgb_files,
                "--min_node_conf", opt.registration_confidence,
                "--loop_strides", opt.pgo_loop_strides
            ]
            zutil.run_cmd(pgo_cmd)
            shutil.copy(opt.results_folder / f"poses_{iteration_id}.txt", opt.results_folder / f"poses_{iteration_id}_pgo_gtsam.txt")
            _logger.info(f"Successfully ran pgo_gtsam.")

        # bridge unregistered frames into the map via optical flow so the next
        # mapping round trains scene coordinates for their viewpoints; without
        # this, captures made of visually disjoint sweeps stall permanently
        if opt.enable_bridge and registration_rate < opt.bridge_below_rate and not scheduled_to_stop_early:
            if opt.bridge_mode == 'vggt':
                bridge_cmd = [
                    VGGT_BRIDGE_EXE,
                    "--pose_log", f"{opt.results_folder}/poses_{iteration_id}.txt",
                    "--image_glob", opt.rgb_files,
                    "--min_node_conf", opt.registration_confidence,
                    "--vggt_env", opt.vggt_env,
                ]
            else:
                bridge_cmd = [
                    BRIDGE_EXE,
                    f"{opt.results_folder}/pointmaps/*/*.npz",
                    "--pose_log", f"{opt.results_folder}/poses_{iteration_id}.txt",
                    "--image_glob", opt.rgb_files,
                    "--min_node_conf", opt.registration_confidence,
                ]
            zutil.run_cmd(bridge_cmd)
            shutil.copy(opt.results_folder / f"poses_{iteration_id}.txt", opt.results_folder / f"poses_{iteration_id}_bridge.txt")
            _logger.info(f"Successfully ran flow bridging.")

        prev_iteration_id = iteration_id

        if scheduled_to_stop_early:
            # we are in the final refinement iteration and stop here
            break

        # check stopping criteria
        if (registration_rate >= opt.registration_threshold) or ((registration_rate-max_registration_rate) < opt.relative_registration_threshold):
            if opt.final_refine:
                # stopping criteria have been met, but we want to do one more round of mapping
                _logger.info(f"Stopping training loop in next iteration. Enough mapping images registered. "
                             f"(Threshold={opt.registration_threshold * 100:.1f}%")
                scheduled_to_stop_early = True
            else:
                # stopping criteria have been met, and we do not want to do one more round of mapping
                _logger.info(f"Stopping training loop. Enough mapping images registered. "
                             f"(Threshold={opt.registration_threshold * 100:.1f}%")
                break

        # stop in any case if we are approaching the maximum number of iterations
        if iteration >= (opt.iterations_max - 2):
            scheduled_to_stop_early = True

        max_registration_rate = max(registration_rate, max_registration_rate)

    if opt.render_visualization:
        _logger.info("Rendering final sweep.")

        zutil.run_cmd(["./render_final_sweep.py",
                       zutil.get_render_path(opt.results_folder),
                       "--render_marker_size", opt.render_marker_size
                       ])

        _logger.info("Converting to video.")

        # get ffmpeg path
        ffmpeg_path = shutil.which("ffmpeg")

        # run ffmpeg to convert the rendered images to a video
        zutil.run_cmd([ffmpeg_path,
                       "-y",
                       "-framerate", 30,
                       "-pattern_type", "glob",
                       "-i", f"{zutil.get_render_path(opt.results_folder)}/*.png",
                       "-c:v", "libx264",
                       "-pix_fmt", "yuv420p",
                       opt.results_folder / "reconstruction.mp4"
                       ])

    reconstruction_end_time = time.time()
    reconstruction_time = reconstruction_end_time - reconstruction_start_time
    _logger.info(f"Reconstructed in {reconstruction_time/60:.1f} Minutes.")

    # check the number of registered mapping images
    registration_rates = zutil.get_registration_rates(
        pose_file=opt.results_folder / f"poses_{iteration_id}.txt",
        thresholds=[500, 1000, 2000, 4000])

    # copy pose estimates of the final iteration to output file
    final_pose_file = opt.results_folder / f"poses_{iteration_id}.txt"
    shutil.copy(final_pose_file, final_pose_file.parent / f"poses_final.txt")

    # rescue a stalled reconstruction: redo it in a subfolder from a VGGT seed,
    # without pose/focal refinement (ACE's refiner degrades an accurate seed:
    # T&T Temple ATE 5cm -> 37cm), and adopt that result as final.
    # --vggt_seed_select runs that seeded reconstruction for every scene and
    # keeps whichever of the two maps registers the images with more inliers
    # on average: the seeded one wins where ACE's own map is warped (7-Scenes
    # stairs) or stalled, the ACE one where VGGT is confidently wrong
    # (symmetric rooms: T&T Courtroom, Ballroom)
    final_rate = zutil.get_registration_rates(
        pose_file=opt.results_folder / "poses_final.txt", thresholds=[opt.registration_confidence])[0]
    stalled = 0 < final_rate < opt.vggt_rescue_below_rate
    if (stalled or opt.vggt_seed_select) and opt.seed_from_vggt <= 0:
        rescue_folder = opt.results_folder / "vggt_rescue"
        if stalled:
            _logger.info(f"Only {final_rate * 100:.1f}% registered (< {opt.vggt_rescue_below_rate * 100:.0f}%): "
                         f"redoing the reconstruction from a VGGT seed in {rescue_folder}.")
        else:
            _logger.info(f"Running the VGGT-seeded alternative reconstruction in {rescue_folder}.")
        import subprocess
        alt = subprocess.run([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:] + [
            "--results_folder", str(rescue_folder),
            "--seed_from_vggt", str(opt.vggt_rescue_frames),
            "--refinement", "none", "--refine_calibration", "False",
            "--vggt_rescue_below_rate", "0", "--vggt_seed_select", "False", "--ba_polish", "False"])

        def mean_inliers(pose_file):
            return float(np.mean([float(l.split()[9]) for l in open(pose_file) if len(l.split()) == 10]))

        adopt = alt.returncode == 0 and (rescue_folder / "poses_final.txt").exists()
        if not adopt:
            _logger.warning("The VGGT-seeded reconstruction failed; keeping the ACE-seeded one.")
        elif opt.vggt_seed_select and not stalled:
            ace_inliers = mean_inliers(opt.results_folder / "poses_final.txt")
            vggt_inliers = mean_inliers(rescue_folder / "poses_final.txt")
            adopt = vggt_inliers > ace_inliers
            _logger.info(f"Mean registration inliers: ACE-seeded {ace_inliers:.0f}, VGGT-seeded {vggt_inliers:.0f}.")
        if adopt:
            shutil.copy(opt.results_folder / "poses_final.txt", opt.results_folder / "poses_final_stalled.txt")
            shutil.copy(rescue_folder / "poses_final.txt", opt.results_folder / "poses_final.txt")
            _logger.info("Using the VGGT-seeded reconstruction as final.")
        else:
            _logger.info("Keeping the ACE-seeded reconstruction.")

    if opt.ba_polish:
        shutil.copy(opt.results_folder / "poses_final.txt", opt.results_folder / "poses_final_prepolish.txt")
        _logger.info("Polishing the final poses with a feature-based bundle adjustment.")
        zutil.run_cmd(["./ba_polish.py",
                       "--pose_log", opt.results_folder / "poses_final_prepolish.txt",
                       "--out_pose_log", opt.results_folder / "poses_final.txt",
                       "--work_dir", opt.results_folder / "ba_polish"])

    # export point cloud if requested
    if opt.export_point_cloud:
        _logger.info("Exporting point cloud.")

        if not opt.dense_point_cloud and opt.render_visualization:
            vis_buffer_file = zutil.get_render_path(opt.results_folder) / f"{iteration_id}_mapping.pkl"
            _logger.info(f"Exporting point cloud from visualisation buffer file: {vis_buffer_file}")

            zutil.run_cmd(["./export_point_cloud.py",
                           opt.results_folder / "pc_final.ply",
                           "--visualization_buffer", vis_buffer_file,
                           "--convention", "opencv",
                           ])
        else:
            _logger.info(f"Exporting point cloud from last network and pose file.")

            zutil.run_cmd(["./export_point_cloud.py",
                           opt.results_folder / "pc_final.ply",
                           "--network", opt.results_folder / f"{iteration_id}.pt",
                           "--pose_file", opt.results_folder / f"poses_final.txt",
                           "--convention", "opencv",
                           "--dense_point_cloud", opt.dense_point_cloud,
                           ])

    stats_report = "Time (min) | Iterations | Reg. Rate @500 | @1000 | @2000 | @4000\n"
    stats_report += f"{reconstruction_time / 60:.1f} " \
                    f"{iteration} " \
                    f"{registration_rates[0] * 100:.1f}% " \
                    f"{registration_rates[1] * 100:.1f}% " \
                    f"{registration_rates[2] * 100:.1f}% " \
                    f"{registration_rates[3] * 100:.1f}%\n"

    _logger.info(stats_report)

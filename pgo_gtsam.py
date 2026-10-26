#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import argparse
import gtsam
from gtsam import Pose3, Rot3, Point3, BetweenFactorPose3, PriorFactorPose3, NonlinearFactorGraph, Values
from gtsam import noiseModel
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import dataset_io
import os
import cv2
from collections import OrderedDict
from distutils.util import strtobool

def _strtobool(x):
    return bool(strtobool(x))

def scale_intrinsics(K, input_size, output_size):
    H_in, W_in = input_size
    H_out, W_out = output_size
    sx = W_out / W_in
    sy = H_out / H_in
    K_scaled = K.copy()
    K_scaled[0, 0] *= sx  # fx
    K_scaled[1, 1] *= sy  # fy
    K_scaled[0, 2] *= sx  # cx
    K_scaled[1, 2] *= sy  # cy
    return K_scaled

def to_npz_path(image_file: str) -> str:
    """
    Convert any image path (or just a filename) to the corresponding .npz path.
    The original extension is stripped and replaced with '.npz'.
    """
    path = Path(image_file)
    return str(path.with_suffix('.npz'))

def load_pointmaps(image_files, npz_files):
    pointmaps = []
    intrinsics = []
    intrinsics_scaled = []

    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=True)
        data = dict(data)

        width = int(data['intrinscis'][0, 2]) * 2
        height = int(data['intrinscis'][1, 2]) * 2
        reordered = np.transpose(data['pointmap'], (1, 2, 0))

        pointmaps.append(reordered)

        scaled_K = scale_intrinsics(data['intrinscis'], (height, width), (reordered.shape[:2]))
        intrinsics.append(data['intrinscis'])
        intrinsics_scaled.append(scaled_K)

    # image_file to npz_index
    npz_basenames = [os.path.basename(f) for f in npz_files]
    image_to_npz_index_map = {}
    npz_index_to_image_index = {}
    for i, image_file in enumerate(image_files):
        npz_name = os.path.basename(to_npz_path(image_file))
        folder = os.path.basename(os.path.dirname(image_file))
        npz_file_indices = [j for j, f in enumerate(npz_files) if f.endswith(f"{folder}/{npz_name}")]
        if len(npz_file_indices) == 0:
            # pointmaps stored without a per-scene subfolder: match by file name only
            npz_file_indices = [j for j, f in enumerate(npz_basenames) if f == npz_name]
        if len(npz_file_indices) == 0:
            raise FileNotFoundError(f"Could not find npz file for image {image_file} in {folder} npz name {npz_name}")
        image_to_npz_index_map[image_file] = npz_file_indices[0]
        npz_index_to_image_index[npz_file_indices[0]] = i

    return pointmaps, np.array(intrinsics), np.array(intrinsics_scaled), image_to_npz_index_map, npz_index_to_image_index

def load_camera_info(file_path, scale = 1):
    poses = []
    image_paths = []
    focal_lengths = []
    confidences = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 10:
                continue  # skip malformed lines
            image_path, qw, qx, qy, qz, x, y, z, focal_length, confidence = parts
            quat = [float(qx), float(qy), float(qz), float(qw)]
            trans = [float(x), float(y), float(z)]
            trans = [t * scale for t in trans]

            # Convert quaternion to rotation matrix
            rot = R.from_quat(quat).as_matrix()

            # Create transformation matrix
            T_camera_world = np.eye(4)
            T_camera_world[:3, :3] = rot
            T_camera_world[:3, 3] = trans

            T_world_camera = np.linalg.inv(T_camera_world)

            poses.append(T_world_camera)
            image_paths.append(image_path)
            focal_lengths.append(float(focal_length))
            confidences.append(float(confidence))

    return np.array(poses), image_paths, np.array(focal_lengths), np.array(confidences)

def build_pairs(n_frames, step=1, loop_strides=None):
    """
    Build a minimal set of frame pairs for sequential data.

    Args:
        n_frames (int): number of frames
        step (int): connect i -> i+step (default 1 = immediate neighbors)
        loop_strides (list[int] | None): strides for loop closure candidates;
            multiple strides give the graph mid- and long-range constraints so
            global consistency has redundancy to work with

    Returns:
        List of (i, j) pairs (deduplicated)
    """
    pairs = set()
    # sequential neighbors
    for i in range(0, n_frames - step):
        j = i + step
        pairs.add((i, j))

    # optional loop closures
    for loop_stride in (loop_strides or []):
        for i in range(0, n_frames, loop_stride):
            j = i + loop_stride
            if j < n_frames:
                pairs.add((i, j))
    return sorted(pairs)

def mat4_to_pose3(mat):
    R = Rot3(mat[0:3,0:3])
    t = Point3(mat[0:3,3])
    return Pose3(R, t)

def pose3_to_mat4(pose):
    R = pose.rotation().matrix()
    t = pose.translation()
    T = np.eye(4)
    T[0:3,0:3] = R
    T[0:3,3] = np.array([t[0], t[1], t[2]])
    return T


def write_poses_to_ace0_fmt(file_name, image_names, poses, focal_lengths, confidences):
    """Write poses in ACE0 pose log format, preserving each image's refined
    focal length and registration confidence from the input pose log. PGO only
    refines poses; overwriting focal/confidence corrupts downstream mapping
    iterations and evaluation (unregistered images must stay below the
    registration confidence threshold)."""
    with open(file_name, "w") as f:
        for image_name, pose, focal_length, confidence in zip(image_names, poses, focal_lengths, confidences):
            pose = np.linalg.inv(pose)
            x, y, z = pose[0:3, 3]
            rotation_matrix = pose[0:3, 0:3]
            quaternion = R.from_matrix(rotation_matrix).as_quat() # x, y, z, w

            # write to pose file in this format -> filename qw qx qy qz x y z focal_length confidence
            pose_str = f"{image_name} " \
                f"{quaternion[3]} {quaternion[0]} {quaternion[1]} {quaternion[2]} " \
                f"{x} {y} {z} {focal_length} {confidence}\n"
            f.write(pose_str)

def skew_symmetric(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


class FlowEstimator:
    """RAFT-based dense optical flow (Eq. 7 in the paper), used for the
    flow-consistency uncertainty metric e_flow (Eq. 8)."""

    def __init__(self, model_name="large", device=None, max_long_side=520, cache_size=8):
        import torch
        from torchvision.models.optical_flow import raft_large, raft_small
        from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == "large":
            weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=weights)
        else:
            weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=weights)
        self.transforms = weights.transforms()
        self.model = self.model.to(self.device).eval()
        self.max_long_side = max_long_side
        self._cache = OrderedDict()
        self._cache_size = cache_size

    def _load_image(self, path):
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        s = min(1.0, self.max_long_side / max(H, W))
        # RAFT requires spatial dims divisible by 8
        newW = max(8, int(round(W * s / 8)) * 8)
        newH = max(8, int(round(H * s / 8)) * 8)
        img = cv2.resize(img, (newW, newH))
        tensor = self.torch.from_numpy(img).permute(2, 0, 1)
        self._cache[path] = tensor
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return tensor

    def flow(self, path_i, path_j, out_hw):
        """Dense flow F_{i->j}, resampled to resolution out_hw=(H, W) with the
        flow vectors expressed in out_hw pixel units. Returns (H, W, 2)."""
        torch = self.torch
        t1 = self._load_image(path_i).unsqueeze(0)
        t2 = self._load_image(path_j).unsqueeze(0)
        b1, b2 = self.transforms(t1, t2)
        with torch.no_grad():
            flow = self.model(b1.to(self.device), b2.to(self.device))[-1][0]  # (2, Hr, Wr)
        Hr, Wr = flow.shape[-2:]
        Hp, Wp = out_hw
        flow = torch.nn.functional.interpolate(
            flow.unsqueeze(0), size=(Hp, Wp), mode="bilinear", align_corners=False)[0]
        flow[0] *= Wp / Wr
        flow[1] *= Hp / Hr
        return flow.permute(1, 2, 0).cpu().numpy()


def find_matches(scene_coords_i, scene_coords_j, T_i, T_j, K_i, K_j, tau):
    """Geometrically consistent correspondences between two pointmaps (Sec. 3.2).

    Every scene coordinate X_i predicted for image I_i is projected into image
    I_j; the match is kept if the scene coordinate predicted at that pixel of
    I_j agrees within tau (e_match = ||X_j - X_i|| < tau).

    Poses are camera-to-world. Returns:
        matches_3d: (N, 3) matched points X_i in camera-i coordinates
        matches_2d: (N, 2) sub-pixel observations of the matched points in
                    image j: the projection of X_j, the scene coordinate
                    predicted by image j at the matched pixel. Together with
                    matches_3d this realizes Eq. 3 (X_j ~ R_ij X_i + t_ij) as
                    a PnP problem; the disagreement between the two images'
                    predictions is the measurement signal. The integer-rounded
                    pixel itself is too coarse: pointmaps are low resolution,
                    so one pixel spans several degrees at pointmap-scale focal
                    lengths.
        xy_i:       (N, 2) source pixel locations in image i
        xhat_j:     (N, 2) sub-pixel projections of X_i into image j
    """
    H, W, _ = scene_coords_i.shape
    X = scene_coords_i.reshape(-1, 3).astype(np.float64)

    T_i_inv = np.linalg.inv(T_i)
    T_j_inv = np.linalg.inv(T_j)

    # project all X_i into image j
    Xc_j = X @ T_j_inv[:3, :3].T + T_j_inv[:3, 3]
    z = Xc_j[:, 2]
    finite = np.isfinite(X).all(axis=1) & np.isfinite(z)
    in_front = finite & (z > 1e-6)

    uv = np.full((X.shape[0], 2), -1.0)
    proj = Xc_j[in_front] @ K_j.T
    uv[in_front] = proj[:, :2] / proj[:, 2:3]

    u = np.round(uv[:, 0]).astype(int)
    v = np.round(uv[:, 1]).astype(int)
    in_bounds = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)

    idx = np.flatnonzero(in_bounds)
    if idx.size == 0:
        return (np.empty((0, 3)), np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2)))

    # 3D consistency error e_match against the scene coordinates predicted in image j
    X_j = scene_coords_j[v[idx], u[idx]].astype(np.float64)
    e_match = np.linalg.norm(X_j - X[idx], axis=1)
    keep = e_match < tau
    idx = idx[keep]
    X_j = X_j[keep]

    # observation of the matched point in image j: where image j's own scene
    # coordinate prediction places it
    Xc_j_pred = X_j @ T_j_inv[:3, :3].T + T_j_inv[:3, 3]
    keep = Xc_j_pred[:, 2] > 1e-6
    idx = idx[keep]
    Xc_j_pred = Xc_j_pred[keep]
    proj_j = Xc_j_pred @ K_j.T
    matches_2d = proj_j[:, :2] / proj_j[:, 2:3]

    ys, xs = np.divmod(idx, W)
    xy_i = np.stack([xs, ys], axis=1).astype(np.float64)

    Xc_i = X[idx] @ T_i_inv[:3, :3].T + T_i_inv[:3, 3]
    xhat_j = uv[idx]

    return Xc_i, matches_2d, xy_i, xhat_j


def epipolar_errors(xy_i, xhat_j, T_i, T_j, K_i, K_j):
    """Epipolar uncertainty metric e_epi (Eqs. 4-6), vectorized over matches.

    Poses are camera-to-world. The essential matrix is built from the relative
    pose camera i -> camera j; the error is the perpendicular distance of the
    projected point to the epipolar line of x_i in image j.
    """
    T_ij = np.linalg.inv(T_j) @ T_i  # camera i -> camera j
    R_ij = T_ij[:3, :3]
    t_ij = T_ij[:3, 3]
    E_ij = skew_symmetric(t_ij) @ R_ij

    M = np.linalg.inv(K_j).T @ E_ij @ np.linalg.inv(K_i)

    xi_h = np.concatenate([xy_i, np.ones((xy_i.shape[0], 1))], axis=1)
    lines = xi_h @ M.T  # (N, 3) epipolar lines in image j

    xj_h = np.concatenate([xhat_j, np.ones((xhat_j.shape[0], 1))], axis=1)
    num = np.abs(np.sum(lines * xj_h, axis=1))
    den = np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2) + 1e-12
    return num / den


def flow_errors(flow_ij, xy_i, xhat_j):
    """Optical flow consistency metric e_flow (Eqs. 7-8), vectorized.

    flow_ij is the dense flow field F_{i->j} at pointmap resolution."""
    xs = xy_i[:, 0].astype(int)
    ys = xy_i[:, 1].astype(int)
    x_tilde = xy_i + flow_ij[ys, xs]
    return np.linalg.norm(xhat_j - x_tilde, axis=1)


def estimate_relative_pose(matches_3d, matches_2d, K, reproj_error, min_inliers):
    if len(matches_3d) < 6:
        return None
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=matches_3d.astype(np.float64),
        imagePoints=matches_2d.astype(np.float64),
        cameraMatrix=K.astype(np.float64),
        distCoeffs=None,
        reprojectionError=reproj_error
    )
    if not success or inliers is None or len(inliers) < min_inliers:
        return None
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T, len(inliers)

def main(args):

    # In ace0 order (randomly ordered)
    poses, image_files, focal_lengths, confidences = load_camera_info(args.pose_log) # (N,4,4)

    n = poses.shape[0]

    # In Sequential Order
    npz_files = dataset_io.get_files_from_glob(str(args.pointmap_files))
    point_maps, intrinsics, intrinsics_scaled, image_to_npz_index_map, npz_index_to_image_index = load_pointmaps(image_files, npz_files)

    # Optical flow based uncertainty needs the RGB images. The pose log may
    # reference paths from another machine/container, so an --image_glob can
    # remap them by file name.
    flow_estimator = None
    image_lookup = {}
    if args.conf_est and args.use_flow:
        if args.image_glob:
            local_images = dataset_io.get_files_from_glob(str(args.image_glob))
            image_lookup = {os.path.basename(f): f for f in local_images}
        missing = 0
        for f in image_files:
            if os.path.basename(f) not in image_lookup and not os.path.isfile(f):
                missing += 1
        if missing > 0:
            print(f"WARNING: {missing}/{len(image_files)} images not found, "
                  f"disabling optical flow uncertainty (epipolar only).")
        else:
            flow_estimator = FlowEstimator(model_name=args.flow_model)

    def resolve_image(image_file):
        # prefer the pose log's own path: basenames are not unique across
        # sequence subfolders (7-Scenes seq-*/frame-000000.color.png)
        if os.path.isfile(image_file):
            return image_file
        return image_lookup.get(os.path.basename(image_file), image_file)

    loop_strides = [int(s) for s in str(args.loop_strides).split(',') if s.strip()]
    pairs = build_pairs(n, step=1, loop_strides=loop_strides)
    print(f"pair candidates: {len(pairs)} (loop strides: {loop_strides})")
    edges = []

    reprojected_pose = 0
    gated = 0
    deviations = []

    def add_fallback_edge(i, j, pose_i, pose_j):
        # optional legacy behavior: constrain the pair with the relative pose
        # implied by the current absolute estimates. The paper instead drops
        # failed pairs and stabilizes unconnected nodes with loose priors.
        if args.fallback_edges:
            T_ij_orig = np.linalg.inv(pose_i) @ pose_j
            edges.append({"i": int(i), "j": int(j),
                          "T_ij": T_ij_orig.astype(np.float64), "sigma": None})

    skipped_unregistered = 0

    for (i, j) in tqdm(pairs, desc="pairs"):
        pose_i = poses[npz_index_to_image_index[i]]
        pose_j = poses[npz_index_to_image_index[j]]

        # scene coordinates of unregistered images are unreliable: edges built
        # from them are mostly gross outliers (near-180deg flips). Leave those
        # nodes to their loose priors instead.
        if args.min_node_conf > 0 and (
                confidences[npz_index_to_image_index[i]] < args.min_node_conf or
                confidences[npz_index_to_image_index[j]] < args.min_node_conf):
            skipped_unregistered += 1
            continue

        try:
            matches_3d, matches_2d, xy_i, xhat_j = find_matches(
                point_maps[i], point_maps[j],
                pose_i, pose_j,
                intrinsics_scaled[i], intrinsics_scaled[j],
                tau=args.tau)

            pnp_result = estimate_relative_pose(
                matches_3d, matches_2d, intrinsics_scaled[j],
                reproj_error=args.pnp_reproj_error, min_inliers=args.min_inliers)
            T_ij_inv, num_inliers = pnp_result if pnp_result is not None else (None, 0)

            sigma = None
            if args.conf_est and len(xy_i) > 0:
                # combined confidence score (Eq. 9): sigma = a1*e_epi + a2*e_flow
                e_epi = epipolar_errors(xy_i, xhat_j, pose_i, pose_j,
                                        intrinsics_scaled[i], intrinsics_scaled[j])
                if flow_estimator is not None:
                    flow_ij = flow_estimator.flow(
                        resolve_image(image_files[npz_index_to_image_index[i]]),
                        resolve_image(image_files[npz_index_to_image_index[j]]),
                        out_hw=point_maps[i].shape[:2])
                    e_flow = flow_errors(flow_ij, xy_i, xhat_j)
                    per_match = args.alpha1 * e_epi + args.alpha2 * e_flow
                else:
                    per_match = e_epi
                sigma = float(np.mean(per_match))

        except Exception as e:
            print(f"failed to estimate pose for pair ({i}, {j}): {e}")
            add_fallback_edge(i, j, pose_i, pose_j)
            continue

        if T_ij_inv is None:
            add_fallback_edge(i, j, pose_i, pose_j)
        else:
            T_ij = np.linalg.inv(T_ij_inv)

            # gate gross outliers: PnP can return wild poses on degenerate
            # match configurations; reject measurements absurdly far from the
            # current estimate instead of letting them shatter the graph
            D = np.linalg.inv(np.linalg.inv(pose_i) @ pose_j) @ T_ij
            rot_dev = np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))
            trans_dev = np.linalg.norm(D[:3, 3])
            deviations.append((rot_dev, trans_dev))
            if rot_dev > args.max_rot_dev or trans_dev > args.max_trans_dev:
                gated += 1
                add_fallback_edge(i, j, pose_i, pose_j)
                continue

            reprojected_pose += 1
            edges.append({"i": int(i), "j": int(j),
                          "T_ij": T_ij.astype(np.float64), "sigma": sigma,
                          "inliers": int(num_inliers)})

    init_poses = [poses[npz_index_to_image_index[i]] for i in range(n)]  # (N,4,4)
    init_poses = np.array(init_poses)
    N = init_poses.shape[0]

    print(f"actual edges by reprojection: {reprojected_pose} (gated as gross outliers: {gated}, "
          f"pairs skipped for unregistered nodes: {skipped_unregistered})")
    if len(deviations) > 0:
        dev = np.array(deviations)
        print(f"measurement deviation from current estimates: "
              f"rot median {np.median(dev[:, 0]):.2f} deg, p90 {np.percentile(dev[:, 0], 90):.2f}, max {dev[:, 0].max():.2f} | "
              f"trans median {np.median(dev[:, 1]):.3f} m, p90 {np.percentile(dev[:, 1], 90):.3f}, max {dev[:, 1].max():.3f}")

    # Build factor graph
    graph = NonlinearFactorGraph()
    initial = Values()

    # Tight prior on first pose to fix the gauge
    prior_noise = noiseModel.Diagonal.Sigmas(np.array([args.first_prior_sigma]*6))
    pose0 = mat4_to_pose3(init_poses[0])
    graph.add(PriorFactorPose3(0, pose0, prior_noise))

    # Add initial estimates
    for i in range(N):
        p = mat4_to_pose3(init_poses[i])
        initial.insert(i, p)

    # Base noise model for between factors, GTSAM Pose3 tangent ordering
    # [rot_x, rot_y, rot_z, tx, ty, tz]. Rotation (rad) and translation (m)
    # live on different scales, so they get separate sigmas: a shared sigma of
    # 0.1 makes rotation constraints ~5.7deg loose while translations are
    # tight, letting the optimizer trade rotation accuracy for translation fit.
    if args.between_sigma is not None:
        # legacy override: one sigma for all components
        base_sigmas = np.array([args.between_sigma] * 6)
    else:
        base_sigmas = np.array([args.between_sigma_rot] * 3 + [args.between_sigma_trans] * 3)

    def make_robust(base_model):
        # Huber loss reduces the influence of outlier constraints while
        # preserving sensitivity to small residuals
        return noiseModel.Robust.Create(
            noiseModel.mEstimator.Huber.Create(args.huber_k), base_model)

    # Scale the information matrix by the median confidence for numerical
    # stability: an edge with median confidence behaves exactly like the base
    # noise model, less confident edges are down-weighted (Eq. 10).
    edge_sigmas = [e["sigma"] for e in edges if e["sigma"] is not None]
    sigma_median = float(np.median(edge_sigmas)) if len(edge_sigmas) > 0 else 1.0
    if args.conf_est:
        print(f"confidence stats: {len(edge_sigmas)} weighted edges, median sigma {sigma_median:.4f}")
    if len(edges) > 0:
        inl = np.array([e["inliers"] for e in edges])
        print(f"inlier stats: min {inl.min()} median {np.median(inl):.0f} max {inl.max()}")

    connected = set()
    for e in edges:
        i = int(e["i"])
        j = int(e["j"])
        T_ij = e["T_ij"]  # 4x4 mapping i->j from registration
        sigma_ij = e["sigma"]

        pose_ij = mat4_to_pose3(T_ij)

        scale = 1.0
        if sigma_ij is not None and args.conf_est is True:
            # confidence weighting: median-confidence edge keeps the base
            # noise model; clamp so no edge becomes drastically tighter than
            # the base model and a single deceptive measurement cannot
            # dominate the whole graph
            scale *= float(np.clip(sigma_ij / (sigma_median + 1e-12), 1.0 / 3.0, 1e3))
        if args.inlier_ref > 0:
            # PnP inlier count is a direct measure of how well-supported the
            # measurement is: sigma ~ 1/sqrt(#inliers) under an iid noise
            # model, normalized so an edge with inlier_ref inliers keeps the
            # base noise model
            scale *= float(np.clip(np.sqrt(args.inlier_ref / max(e["inliers"], 1)), 0.5, 4.0))

        model = noiseModel.Diagonal.Sigmas(base_sigmas * scale)
        graph.add(BetweenFactorPose3(i, j, pose_ij, make_robust(model)))
        connected.add(i)
        connected.add(j)

    print(f"Graph added with conf_est set {args.conf_est}")

    # Loose priors on unconnected nodes prevent an under-constrained
    # optimization without introducing strong constraints
    unconnected = set(range(N)) - connected
    if len(unconnected) > 0:
        print(f"Adding loose priors to {len(unconnected)} unconnected nodes: {sorted(unconnected)}")
        loose_noise = noiseModel.Diagonal.Sigmas(np.array([args.loose_prior_sigma]*6))
        for i in sorted(unconnected):
            graph.add(PriorFactorPose3(i, mat4_to_pose3(init_poses[i]), loose_noise))

    # Optimize
    print("Number of factors:", graph.size())
    print("Keys in initial estimate:", len(list(initial.keys())))

    params = gtsam.LevenbergMarquardtParams()
    params.setRelativeErrorTol(1e-5)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    print("Optimization done.")

    # collect optimized poses
    optimized = np.zeros_like(init_poses)
    for i in range(N):
        p = result.atPose3(i)
        optimized[i] = pose3_to_mat4(p)

    # Reordered the sequential poses back to ordered of ace0 format (output pose already in world to camera)
    optimized_reordered = [optimized[image_to_npz_index_map[i]] for i in image_files]
    write_poses_to_ace0_fmt(args.pose_log, image_files, optimized_reordered, focal_lengths, confidences)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pointmap_files", type=str, help="Glob pattern for npz files, e.g. 'datasets/scene/*.npz'")
    parser.add_argument("--loop_strides", type=str, default="20",
                        help="comma-separated strides for sampled loop closure candidates, e.g. '5,10,20'")
    parser.add_argument("--pose_log", type=str, help="log of the pose")
    parser.add_argument("--conf_est", type=_strtobool, default=False, help="enable confidence estimation to add in pose graph")
    parser.add_argument("--tau", type=float, default=0.1,
                        help="geometric consistency threshold in meters; scheduled 0.2 -> 0.05 by the caller")
    parser.add_argument("--image_glob", type=str, default="",
                        help="glob for RGB images, used to remap pose log image paths for optical flow")
    parser.add_argument("--use_flow", type=_strtobool, default=True,
                        help="use RAFT optical flow consistency in the confidence score")
    parser.add_argument("--flow_model", type=str, default="large", choices=["large", "small"],
                        help="RAFT variant for optical flow")
    parser.add_argument("--alpha1", type=float, default=0.4, help="weight of the epipolar uncertainty metric")
    parser.add_argument("--alpha2", type=float, default=0.6, help="weight of the optical flow uncertainty metric")
    parser.add_argument("--between_sigma", type=float, default=None,
                        help="legacy: one stddev for all between-factor components; overrides "
                             "the separate rot/trans sigmas when set")
    parser.add_argument("--between_sigma_rot", type=float, default=0.1,
                        help="stddev of between factors, rotational components (rad). "
                             "Bicycle ablation: tighter (0.02) scored worse (17.67 vs 18.07), "
                             "so defaults keep the looser legacy weighting")
    parser.add_argument("--between_sigma_trans", type=float, default=0.1,
                        help="stddev of between factors, translational components (m)")
    parser.add_argument("--inlier_ref", type=float, default=0,
                        help="PnP inlier count at which an edge keeps the base noise model; "
                             "edges with fewer/more inliers are down/up-weighted by sqrt "
                             "(0 disables; bicycle ablation showed no gain)")
    parser.add_argument("--huber_k", type=float, default=1.0, help="Huber loss scale parameter")
    parser.add_argument("--first_prior_sigma", type=float, default=1e-6, help="stddev of the prior on the first pose")
    parser.add_argument("--loose_prior_sigma", type=float, default=10.0,
                        help="stddev of loose priors added to unconnected nodes")
    parser.add_argument("--pnp_reproj_error", type=float, default=3.0,
                        help="RANSAC PnP inlier reprojection error threshold in pixels (pointmap resolution)")
    parser.add_argument("--min_inliers", type=int, default=10,
                        help="minimum PnP inliers to accept a measured relative pose")
    parser.add_argument("--min_node_conf", type=float, default=500,
                        help="skip graph edges involving images whose registration confidence "
                             "is below this value (0 disables); matches ACE0's registration_confidence")
    parser.add_argument("--max_rot_dev", type=float, default=30.0,
                        help="reject measurements deviating more than this rotation (deg) from the current estimate")
    parser.add_argument("--max_trans_dev", type=float, default=1.0,
                        help="reject measurements deviating more than this translation (m) from the current estimate")
    parser.add_argument("--fallback_edges", type=_strtobool, default=False,
                        help="add edges from current pose estimates for pairs without a reliable "
                             "measurement (legacy behavior); default drops them per the paper")
    args = parser.parse_args()
    # Print all arguments and their values
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    main(args)

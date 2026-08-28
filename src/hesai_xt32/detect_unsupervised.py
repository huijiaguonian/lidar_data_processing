import argparse
import json
import math
import os

import cv2
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


DEFAULT_OUTPUT_DIR = os.path.join("outputs", "unsupervised")

# PandarXT coordinates produced by hesai_xt32.parse_pcap:
# +Y is azimuth 0 degrees (forward), +X is lateral, and +Z is upward.
ROI_X = (-50.0, 50.0)
ROI_Y = (-50.0, 50.0)
ROI_Z = (-0.20, 4.0)

MIN_RANGE = 0.30
MAX_RANGE = 50.0
MIN_INTENSITY = 0.0

RANSAC_DIST_THRESH = 0.10
RANSAC_MAX_ITER = 500
MAX_GROUND_TILT_DEG = 30.0
GROUND_CLEARANCE = ROI_Z[0]

VOXEL_SIZE = 0.03

# Overlapping radial bands prevent hard boundaries from splitting one object.
# (minimum range, maximum range, DBSCAN eps, DBSCAN min_samples)
RANGE_CLUSTER_CONFIGS = [
    (0.0, 10.0, 0.25, 5),
    (8.0, 20.0, 0.45, 4),
    (18.0, 35.0, 0.70, 3),
    (33.0, 50.1, 1.00, 3),
]


def load_and_prefilter(bin_path):
    data = np.fromfile(bin_path, dtype=np.float32)
    if data.size == 0 or data.size % 4 != 0:
        raise ValueError(
            f"Invalid BIN file: {data.size} float32 values; expected N x 4."
        )

    points = data.reshape(-1, 4)
    finite = np.isfinite(points).all(axis=1)
    distance = np.linalg.norm(points[:, :3], axis=1)
    keep = (
        finite
        & (distance >= MIN_RANGE)
        & (distance <= MAX_RANGE)
        & (points[:, 3] >= MIN_INTENSITY)
    )
    filtered = points[keep]
    print(
        f"[LOAD] {len(points):,} points; {len(filtered):,} remain after "
        "finite/range/intensity filtering"
    )
    return filtered[:, :3].astype(np.float64, copy=False)


def estimate_ground_plane(points):
    o3d.utility.random.seed(7)
    candidates = points[np.linalg.norm(points[:, :2], axis=1) <= 30.0]
    if len(candidates) < 100:
        raise ValueError("Too few points for ground-plane estimation.")

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(candidates))
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=RANSAC_DIST_THRESH,
        ransac_n=3,
        num_iterations=RANSAC_MAX_ITER,
    )
    plane_model = np.asarray(plane_model, dtype=np.float64)
    normal = plane_model[:3]
    norm = np.linalg.norm(normal)
    if norm == 0:
        raise ValueError("RANSAC returned a zero ground-plane normal.")
    plane_model /= norm
    normal = plane_model[:3]

    if normal[2] < 0:
        plane_model *= -1.0
        normal = plane_model[:3]

    tilt = math.degrees(math.acos(np.clip(normal[2], -1.0, 1.0)))
    inlier_ratio = len(inliers) / len(candidates)
    print(
        f"[GROUND] normal={normal}, tilt={tilt:.2f} deg, "
        f"inliers={len(inliers):,} ({inlier_ratio:.1%})"
    )
    if tilt > MAX_GROUND_TILT_DEG:
        raise ValueError(
            f"Dominant plane tilt {tilt:.1f} deg exceeds "
            f"{MAX_GROUND_TILT_DEG:.1f} deg; it is probably a wall."
        )

    ground_points = candidates[np.asarray(inliers, dtype=np.int64)]
    return plane_model, ground_points


def rotation_to_z_axis(normal):
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(normal, z_axis)
    s = np.linalg.norm(v)
    c = np.dot(normal, z_axis)
    if s < 1e-9:
        return np.eye(3)

    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def align_and_remove_ground(points, plane_model, ground_points):
    normal = plane_model[:3]
    rotation = rotation_to_z_axis(normal)
    aligned = points @ rotation.T
    aligned_ground = ground_points @ rotation.T
    ground_z = float(np.median(aligned_ground[:, 2]))
    aligned[:, 2] -= ground_z

    plane_distance = np.abs(points @ normal + plane_model[3])
    non_ground = plane_distance > RANSAC_DIST_THRESH
    result = aligned[non_ground]
    print(
        f"[GROUND] z shifted by {-ground_z:+.3f} m; "
        f"removed {len(aligned) - len(result):,} ground-plane points"
    )
    return aligned, result


def apply_roi(points):
    keep = (
        (points[:, 0] >= ROI_X[0])
        & (points[:, 0] <= ROI_X[1])
        & (points[:, 1] >= ROI_Y[0])
        & (points[:, 1] <= ROI_Y[1])
        & (points[:, 2] >= ROI_Z[0])
        & (points[:, 2] <= ROI_Z[1])
    )
    filtered = points[keep]
    print(f"[ROI] retained {len(filtered):,}/{len(points):,} non-ground points")
    return filtered


def voxel_downsample(points):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    downsampled = np.asarray(pcd.voxel_down_sample(VOXEL_SIZE).points)
    print(
        f"[VOXEL] {len(points):,} -> {len(downsampled):,} points "
        f"at {VOXEL_SIZE:.2f} m"
    )
    return downsampled


def cluster_points(points):
    """Run DBSCAN in overlapping range bands and merge shared clusters."""
    if len(points) < 3:
        return np.full(len(points), -1, dtype=np.int32)

    radius = np.linalg.norm(points[:, :2], axis=1)
    proposals = []

    for lo, hi, eps, min_samples in RANGE_CLUSTER_CONFIGS:
        indices = np.flatnonzero((radius >= lo) & (radius < hi))
        if len(indices) < min_samples:
            print(f"[DBSCAN {lo:.0f}-{hi:.0f}m] no usable points")
            continue

        local_labels = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=-1,
        ).fit_predict(points[indices])

        noise_count = int(np.count_nonzero(local_labels == -1))
        local_cluster_count = len(set(local_labels)) - (
            1 if -1 in local_labels else 0
        )
        print(
            f"[DBSCAN {lo:.0f}-{hi:.0f}m] points={len(indices):,}, "
            f"clusters={local_cluster_count}, noise={noise_count:,} "
            f"({noise_count / len(indices):.1%}), eps={eps:.2f}, "
            f"min_samples={min_samples}"
        )

        for local_label in sorted(set(local_labels) - {-1}):
            members = set(indices[local_labels == local_label].tolist())
            if members:
                proposals.append(members)

    if not proposals:
        return np.full(len(points), -1, dtype=np.int32)

    parent = list(range(len(proposals)))

    def find(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    point_owner = {}
    for proposal_index, members in enumerate(proposals):
        for point_index in members:
            previous_owner = point_owner.get(point_index)
            if previous_owner is None:
                point_owner[point_index] = proposal_index
            else:
                union(proposal_index, previous_owner)

    merged_members = {}
    for proposal_index, members in enumerate(proposals):
        root = find(proposal_index)
        merged_members.setdefault(root, set()).update(members)

    labels = np.full(len(points), -1, dtype=np.int32)
    ordered_clusters = sorted(
        merged_members.values(), key=lambda members: min(members)
    )
    for global_label, members in enumerate(ordered_clusters):
        labels[np.fromiter(members, dtype=np.int64)] = global_label

    assigned = int(np.count_nonzero(labels >= 0))
    print(
        f"[DBSCAN merged] clusters={len(ordered_clusters)}, "
        f"assigned={assigned:,}/{len(points):,}, "
        f"noise={len(points) - assigned:,}"
    )
    return labels


def split_oversized_cluster(cluster_points, depth=0):
    """Split a merged object row only when a clear horizontal gap exists."""
    if depth >= 2:
        return [cluster_points]

    center = cluster_points.mean(axis=0)
    range_m = float(np.linalg.norm(center[:2]))
    if range_m < 10.0:
        minimum_points, max_length, max_width, split_gap = 20, 6.0, 3.0, 0.65
    elif range_m < 20.0:
        minimum_points, max_length, max_width, split_gap = 8, 8.0, 4.0, 0.80
    else:
        minimum_points, max_length, max_width, split_gap = 5, 10.0, 5.0, 1.00

    if len(cluster_points) < 2 * minimum_points:
        return [cluster_points]

    xy = cluster_points[:, :2].astype(np.float64)
    hull = cv2.convexHull(xy.astype(np.float32).reshape(-1, 1, 2))
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(hull)
    if max(side_a, side_b) <= max_length and min(side_a, side_b) <= max_width:
        return [cluster_points]

    centered = xy - xy.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]

    for axis_index in range(2):
        projection = centered @ axes[:, axis_index]
        order = np.argsort(projection)
        sorted_projection = projection[order]
        gaps = np.diff(sorted_projection)
        if gaps.size == 0:
            continue

        gap_index = int(np.argmax(gaps))
        gap_size = float(gaps[gap_index])
        if gap_size < split_gap:
            continue

        cut = (sorted_projection[gap_index] + sorted_projection[gap_index + 1]) / 2.0
        left = cluster_points[projection <= cut]
        right = cluster_points[projection > cut]
        if len(left) < minimum_points or len(right) < minimum_points:
            continue

        parts = []
        parts.extend(split_oversized_cluster(left, depth + 1))
        parts.extend(split_oversized_cluster(right, depth + 1))
        return parts

    return [cluster_points]


def fit_filtered_obb(cluster_points, label, color):
    center_estimate = cluster_points.mean(axis=0)
    range_m = float(np.linalg.norm(center_estimate[:2]))

    if range_m < 10.0:
        minimum_points = 20
        min_length, max_length = 0.35, 6.0
        min_width, max_width = 0.25, 3.0
        min_height, max_height = 0.35, 3.5
        max_aspect_ratio = 6.0
        max_ground_gap = 0.70
    elif range_m < 20.0:
        minimum_points = 8
        min_length, max_length = 0.30, 8.0
        min_width, max_width = 0.05, 4.0
        min_height, max_height = 0.20, 4.0
        max_aspect_ratio = 30.0
        max_ground_gap = 1.0
    else:
        minimum_points = 5
        min_length, max_length = 0.20, 10.0
        min_width, max_width = 0.02, 5.0
        min_height, max_height = 0.15, 4.0
        max_aspect_ratio = 50.0
        max_ground_gap = 1.5

    if len(cluster_points) < minimum_points:
        return None

    xy = cluster_points[:, :2].astype(np.float32)
    hull = cv2.convexHull(xy.reshape(-1, 1, 2))
    rect = cv2.minAreaRect(hull)
    (center_x, center_y), (side_a, side_b), angle = rect
    length = float(max(side_a, side_b))
    width = float(min(side_a, side_b))
    z_min = float(cluster_points[:, 2].min())
    z_max = float(cluster_points[:, 2].max())
    height = z_max - z_min
    aspect_ratio = length / max(width, 1e-6)

    valid = (
        min_length <= length <= max_length
        and min_width <= width <= max_width
        and min_height <= height <= max_height
        and aspect_ratio <= max_aspect_ratio
        and z_min <= max_ground_gap
    )
    if not valid:
        return None

    box_2d = cv2.boxPoints(rect).astype(np.float64)
    vertices = np.vstack(
        [
            np.column_stack([box_2d, np.full(4, z_min)]),
            np.column_stack([box_2d, np.full(4, z_max)]),
        ]
    )
    lines = np.asarray(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int32,
    )
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(vertices),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(
        np.repeat(np.asarray(color)[None, :], len(lines), axis=0)
    )

    detection = {
        "cluster_label": int(label),
        "point_count": int(len(cluster_points)),
        "center": [center_x, center_y, (z_min + z_max) / 2.0],
        "dimensions": [length, width, height],
        "angle_degrees": float(angle),
        "ground_gap": z_min,
        "range_m": range_m,
    }
    return detection, line_set


def build_detections(points, labels):
    rng = np.random.default_rng(7)
    detections = []
    line_sets = []
    for label in sorted(set(labels) - {-1}):
        cluster = points[labels == label]
        candidates = split_oversized_cluster(cluster)
        if len(candidates) > 1:
            sizes = ", ".join(str(len(candidate)) for candidate in candidates)
            print(
                f"[SPLIT {label:03d}] {len(cluster):,} points -> "
                f"{len(candidates)} parts ({sizes})"
            )

        color = rng.random(3)
        for split_index, candidate in enumerate(candidates):
            result = fit_filtered_obb(candidate, label, color)
            if result is None:
                continue

            detection, line_set = result
            detection["split_index"] = int(split_index)
            detections.append(detection)
            line_sets.append(line_set)
            dims = detection["dimensions"]
            suffix = f".{split_index}" if len(candidates) > 1 else ""
            print(
                f"[OBB {label:03d}{suffix}] points={len(candidate):,}, "
                f"LWH={dims[0]:.2f},{dims[1]:.2f},{dims[2]:.2f} m, "
                f"center={np.round(detection['center'], 2)}"
            )

    print(f"[OBB] accepted {len(detections)} target proposals")
    return detections, line_sets


def combine_line_sets(line_sets):
    combined_points = []
    combined_lines = []
    combined_colors = []
    point_offset = 0
    for line_set in line_sets:
        points = np.asarray(line_set.points)
        lines = np.asarray(line_set.lines)
        colors = np.asarray(line_set.colors)
        combined_points.append(points)
        combined_lines.append(lines + point_offset)
        combined_colors.append(colors)
        point_offset += len(points)

    combined = o3d.geometry.LineSet()
    if combined_points:
        combined.points = o3d.utility.Vector3dVector(np.vstack(combined_points))
        combined.lines = o3d.utility.Vector2iVector(np.vstack(combined_lines))
        combined.colors = o3d.utility.Vector3dVector(np.vstack(combined_colors))
    return combined


def save_results(output_dir, environment_points, roi_points, detections, line_sets):
    os.makedirs(output_dir, exist_ok=True)

    environment_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(environment_points)
    )
    environment_cloud.paint_uniform_color([0.62, 0.62, 0.62])
    environment_path = os.path.join(output_dir, "environment_aligned.ply")
    o3d.io.write_point_cloud(environment_path, environment_cloud)

    roi_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(roi_points))
    roi_cloud.paint_uniform_color([0.20, 0.60, 1.00])
    roi_path = os.path.join(output_dir, "non_ground_roi.ply")
    o3d.io.write_point_cloud(roi_path, roi_cloud)

    obb_path = os.path.join(output_dir, "detected_obbs.ply")
    o3d.io.write_line_set(obb_path, combine_line_sets(line_sets))

    json_path = os.path.join(output_dir, "detections.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(detections, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] {environment_path}")
    print(f"[SAVE] {roi_path}")
    print(f"[SAVE] {obb_path}")
    print(f"[SAVE] {json_path}")
    return environment_cloud, roi_cloud


def parse_args():
    parser = argparse.ArgumentParser(
        description="Geometry-only target proposals for PandarXT frame BIN files"
    )
    parser.add_argument("--bin", required=True, help="input N x 4 float32 BIN")
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-view", action="store_true", help="save outputs without opening Open3D"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    points = load_and_prefilter(args.bin)
    plane_model, ground_points = estimate_ground_plane(points)
    environment_points, non_ground_points = align_and_remove_ground(
        points, plane_model, ground_points
    )
    roi_points = apply_roi(non_ground_points)
    if len(roi_points) == 0:
        raise ValueError("No points remain after ground removal and ROI filtering.")

    clustered_points = voxel_downsample(roi_points)
    labels = cluster_points(clustered_points)
    detections, line_sets = build_detections(clustered_points, labels)
    environment_cloud, roi_cloud = save_results(
        args.outdir,
        environment_points,
        roi_points,
        detections,
        line_sets,
    )

    if not args.no_view:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        o3d.visualization.draw_geometries(
            [environment_cloud, roi_cloud, axes] + line_sets,
            window_name="PandarXT non-ground clusters and OBB proposals",
        )


if __name__ == "__main__":
    main()

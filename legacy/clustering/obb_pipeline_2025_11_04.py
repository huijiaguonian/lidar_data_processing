import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import cv2
from scipy.spatial import ConvexHull
from dataclasses import dataclass
from typing import Optional


# ================== 参数 ==================
BIN_PATH = "/mnt/d/Hesai_data/frames_bin_fixed/frame_0000.bin"

# ROI（矩形 + 半径）
X_MIN, X_MAX = -20.0, 20.0
Y_MIN, Y_MAX =  -5.0,  5.0
Z_MIN, Z_MAX =  -3.0,  3.0
R_MAX = 70.0

# RANSAC 去地面
RANSAC_ITER = 200
RANSAC_DIST = 0.08

# 体素下采样
VOXEL_SIZE = 0.10

# 聚类分桶（自适应 eps）
R_BINS = [0, 10, 20, 35, 50, 80]

# 车辆/几何过滤阈值（单位：m）
MIN_L, MIN_W, MIN_H = 0.5, 0.5, 0.3
MAX_L, MAX_W, MAX_H = 10.0, 5.0, 3.5
ASPECT_MAX = 6.0       # L/W 上限
SPLIT_LEN = 8.0        # 大团二次切割阈值（沿主轴）

RESIDUAL_GROUND_Z = -1.5  # 残留地面抑制阈值（世界Z）

# ============== 数据结构（修复 OBB 返回） ==============
@dataclass
class OBBResult:
    geom: o3d.geometry.LineSet      # 用于显示的线框几何
    center: np.ndarray              # (3,)
    extent: np.ndarray              # (3,) 近似(l,w,h)

# ============== 工具函数 ==============
def estimate_ground_plane(points, max_iter=RANSAC_ITER, dist_thresh=RANSAC_DIST):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_thresh,
                                             ransac_n=3,
                                             num_iterations=max_iter)
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=float)
    normal /= np.linalg.norm(normal) + 1e-12
    ground_pts = points[inliers]
    ground_z = np.median(ground_pts[:, 2]) if len(ground_pts) > 0 else 0.0
    return plane_model, normal, ground_z, ground_pts

def align_to_z_axis(points, normal):
    z_axis = np.array([0, 0, 1.0], dtype=float)
    v = np.cross(normal, z_axis)
    s = np.linalg.norm(v)
    if s < 1e-6:
        return points  # 无需旋转
    c = float(np.dot(normal, z_axis))
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))
    return points @ R.T

def apply_rect_roi(points):
    """矩形ROI + 半径限制"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    r = np.sqrt(x**2 + y**2)
    mask = (
        (x >= X_MIN) & (x <= X_MAX) &
        (y >= Y_MIN) & (y <= Y_MAX) &
        (z >= Z_MIN) & (z <= Z_MAX) &
        (r < R_MAX)
    )
    return points[mask]

def voxel_downsample(points, voxel=VOXEL_SIZE):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd = pcd.voxel_down_sample(voxel)
    return np.asarray(pcd.points)

# ----------- OBB：用 minAreaRect 得到平面角度，再拉高到3D -----------
def _boxes_to_lineset(box_xy: np.ndarray, z_min: float, z_max: float) -> o3d.geometry.LineSet:
    # box_xy shape: (4, 2)
    vertices = np.vstack([
        np.c_[box_xy, np.full(4, z_min)],
        np.c_[box_xy, np.full(4, z_max)]
    ])
    lines = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set

def _center_extent_from_box(box_xy: np.ndarray, z_min: float, z_max: float):
    # center
    c_xy = box_xy.mean(axis=0)
    # extents: 取两条相邻边的长度作为 L/W，高度为 z_max - z_min
    e0 = np.linalg.norm(box_xy[1] - box_xy[0])
    e1 = np.linalg.norm(box_xy[2] - box_xy[1])
    L, W = (e0, e1) if e0 >= e1 else (e1, e0)
    H = float(z_max - z_min)
    center = np.array([c_xy[0], c_xy[1], (z_min + z_max) * 0.5], dtype=float)
    extent = np.array([L, W, H], dtype=float)
    return center, extent

def fit_tight_obb(points) -> Optional[OBBResult]:
    if len(points) < 10:
        return None
    try:
        pts2d = points[:, :2]
        if len(pts2d) > 10:
            hull = ConvexHull(pts2d)
            hull_pts = pts2d[hull.vertices]
        else:
            hull_pts = pts2d

        rect = cv2.minAreaRect(hull_pts.astype(np.float32))
        box = cv2.boxPoints(rect)  # (4,2)
        # 保证顺序一致（逆时针）
        cen = box.mean(axis=0)
        ang = np.arctan2(box[:, 1] - cen[1], box[:, 0] - cen[0])
        box = box[np.argsort(ang)[::-1]]

        z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())
        line_set = _boxes_to_lineset(box, z_min, z_max)
        center, extent = _center_extent_from_box(box, z_min, z_max)
        return OBBResult(geom=line_set, center=center, extent=extent)
    except Exception as e:
        print(f"OBB 拟合错误: {e}")
        return None

def fit_simple_obb(points) -> Optional[OBBResult]:
    if len(points) < 10:
        return None
    try:
        pts2d = points[:, :2]
        pca = PCA(n_components=2).fit(pts2d)
        main_axis = pca.components_[0]
        ang = np.arctan2(main_axis[1], main_axis[0])
        R = np.array([[np.cos(ang), -np.sin(ang)],
                      [np.sin(ang),  np.cos(ang)]], dtype=float)
        centered = pts2d - pts2d.mean(axis=0)
        rot = centered @ R.T
        minv, maxv = rot.min(axis=0), rot.max(axis=0)
        rect = np.array([[minv[0], minv[1]], [maxv[0], minv[1]],
                         [maxv[0], maxv[1]], [minv[0], maxv[1]]], dtype=float)
        corners = rect @ R + pts2d.mean(axis=0)
        z_min, z_max = float(points[:, 2].min()), float(points[:, 2].max())
        line_set = _boxes_to_lineset(corners, z_min, z_max)
        center, extent = _center_extent_from_box(corners, z_min, z_max)
        return OBBResult(geom=line_set, center=center, extent=extent)
    except Exception as e:
        print(f"简单 OBB 错误: {e}")
        return None

# ----------- 自适应分桶 DBSCAN -----------
def cluster_points(points_np):
    r = np.linalg.norm(points_np[:, :2], axis=1)
    labels = -np.ones(len(points_np), dtype=int)
    current_label = 0

    print("\n=== 自适应分桶 DBSCAN 聚类 ===")
    for lo, hi in zip(R_BINS[:-1], R_BINS[1:]):
        mask = (r >= lo) & (r < hi)
        num_points = int(mask.sum())
        if num_points < 20:
            continue

        mean_r = (lo + hi) * 0.5
        if mean_r < 15:
            eps = 0.2 + 0.006 * mean_r
            min_samples = int(max(10, min(30, 8 + 0.2 * mean_r)))
        else:
            eps = 0.15 + 0.007 * mean_r
            min_samples = int(max(6,  min(25, 6 + 0.15 * mean_r)))

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points_np[mask, :3])
        clab = db.labels_
        valid = clab >= 0
        clab[valid] += current_label
        labels[mask] = clab
        if valid.any():
            current_label = labels.max() + 1

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"✅ 聚类完成，共检测到 {n_clusters} 个簇")
    clusters = []
    for i in range(n_clusters):
        idx = np.where(labels == i)[0]
        if len(idx) >= 10:
            cpts = points_np[idx]
            clusters.append(cpts)
    return clusters

# ----------- PCA 二次切割 & 残留地面抑制 -----------
def split_large_cluster(pts, length_thresh=SPLIT_LEN):
    """簇很长时，沿主轴二分为两段"""
    if len(pts) < 30:
        return [pts]
    pca = PCA(n_components=1).fit(pts[:, :3])
    proj = pca.transform(pts[:, :3]).ravel()
    span = proj.max() - proj.min()
    if span < length_thresh:
        return [pts]
    mid = np.median(proj)
    left = pts[proj < mid]
    right = pts[proj >= mid]
    out = []
    if len(left) > 10: out.append(left)
    if len(right) > 10: out.append(right)
    return out if out else [pts]

def suppress_residual_ground(pts, z_thresh=RESIDUAL_GROUND_Z):
    return pts[pts[:, 2] > z_thresh] if len(pts) else pts

# ================== 主程序 ==================
points = np.fromfile(BIN_PATH, dtype=np.float32).reshape(-1, 4)
points_np = points[:, :3]
print(f"Loaded {len(points_np)} points")
print("[PCD] X:", points[:,0].min(), "→", points[:,0].max())
print("[PCD] Y:", points[:,1].min(), "→", points[:,1].max())
print("[PCD] Z:", points[:,2].min(), "→", points[:,2].max())

# --- 地面与姿态自检 ---
print("\n=== 地面平面自检 ===")
plane_model, normal, ground_z, ground_pts = estimate_ground_plane(points_np)
angle = float(np.degrees(np.arccos(np.clip(np.dot(normal, [0,0,1]), -1, 1))))
print(f"地面法向: {normal}, 与Z轴夹角: {angle:.3f}°")
print(f"地面中位高度: {ground_z:.3f} m")

if angle > 2.0:
    print("⚙️ 检测到俯仰偏转，自动姿态对齐...")
    points_np = align_to_z_axis(points_np, normal)
else:
    print("✅ 姿态对齐良好。")

# 零点化（把地面抬到 Z≈0）
points_np[:, 2] -= ground_z
print(f"✅ 地面已对齐至 Z≈0，高度平移 {-ground_z:+.3f} m")

# --- ROI + 体素下采样 ---
points_np = apply_rect_roi(points_np)
print(f"[ROI] kept: {points_np.shape[0]} points")
points_np = voxel_downsample(points_np)
print(f"[Voxel] kept: {points_np.shape[0]} points (voxel={VOXEL_SIZE} m)")

# --- 聚类 ---
raw_clusters = cluster_points(points_np)

# --- 后处理：二次切割 + 残留地面抑制 + OBB & 几何过滤 ---
final_clusters = []
for cpts in raw_clusters:
    # 抑制残留地面
    cpts = suppress_residual_ground(cpts)
    # 大团沿主轴切割
    sub_clusters = split_large_cluster(cpts)
    final_clusters.extend(sub_clusters)

obb_boxes = []
print("\n=== 拟合 OBB 并做几何过滤 ===")
for i, cluster_pts in enumerate(final_clusters):
    if len(cluster_pts) < 10:
        continue
    obb = fit_tight_obb(cluster_pts) or fit_simple_obb(cluster_pts)
    if obb is None:
        continue

    L, W, H = np.sort(obb.extent)[::-1]  # L>=W>=H
    # 几何过滤（车的合理范围 & 细长墙面/杆去除）
    if (L < MIN_L or W < MIN_W or H < MIN_H) or (L > MAX_L or W > MAX_W or H > MAX_H):
        continue
    if (L / max(W, 1e-6)) > ASPECT_MAX:
        continue

    color = np.random.rand(3)
    obb.geom.paint_uniform_color(color)
    obb_boxes.append(obb)

    print(f"[OBB] center={obb.center.round(3)}  extent(l,w,h)={np.array([L,W,H]).round(3)}")

print(f"\n🎯 生成 OBB 框: {len(obb_boxes)} 个")

# --- 可视化 ---
pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_np))
pcd.paint_uniform_color([0.3, 0.6, 1.0])

geoms = [pcd] + [b.geom for b in obb_boxes]
o3d.visualization.draw_geometries(geoms, window_name="Unsupervised OBB Clustering")

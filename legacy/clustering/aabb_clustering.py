import numpy as np
import open3d as o3d
import math
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt

# === 参数配置 ===
BIN_PATH = "/mnt/d/Hesai_data/frames_bin_fixed/frame_0000.bin"
EPS = 0.15         # 聚类半径 (米)
MIN_SAMPLES = 15   # 最小点数
VEHICLE_LENGTH = 4.5
VEHICLE_WIDTH = 2.0
SIZE_THRESHOLD = 1.3  # 超出多少倍才触发切割
Z_MIN, Z_MAX = -5.0, 10.0  # ✅ Z过滤阈值

# === 添加后处理：检查并排车辆 ===
def detect_side_by_side_vehicles(cluster_points):
    """检测并排停靠的车辆"""
    if len(cluster_points) < 50:
        return [cluster_points]
    
    pca = PCA(n_components=2)
    xy_points = cluster_points[:, :2]
    pca.fit(xy_points)
    
    # 在主方向和垂直方向都检查
    proj_main = xy_points @ pca.components_[0]
    proj_secondary = xy_points @ pca.components_[1]
    
    L_main = np.max(proj_main) - np.min(proj_main)
    L_secondary = np.max(proj_secondary) - np.min(proj_secondary)
    
    # 如果两个方向尺寸都很大，可能是并排车辆
    if L_main > 4.0 and L_secondary > 3.0:
        # 在次要方向进行切分
        clusters = []
        num_splits = int(np.ceil(L_secondary / 2.5))
        step = L_secondary / num_splits
        sec_min = np.min(proj_secondary)
        
        for k in range(num_splits):
            lo, hi = sec_min + k * step, sec_min + (k + 1) * step
            mask = (proj_secondary >= lo) & (proj_secondary < hi)
            sub_pts = cluster_points[mask]
            if len(sub_pts) >= 15:
                clusters.append(sub_pts)
        return clusters
    
    return [cluster_points]

# === Step 1. 读取点云 ===
points = np.fromfile(BIN_PATH, dtype=np.float32).reshape(-1, 4)
points_np = points[:, :3]  # (x, y, z)
intensity = points[:, 3]
print(f"Loaded {len(points_np)} points from {BIN_PATH}")

# === Step 1.5. Z过滤 ===
mask = (points_np[:, 2] > Z_MIN) & (points_np[:, 2] < Z_MAX)
points_np = points_np[mask]
intensity = intensity[mask]
print(f"Filtered points: {len(points_np)} (Z in [{Z_MIN}, {Z_MAX}])")

# === Step 2. 自适应分桶 DBSCAN 聚类 ===

# === Step 1. 计算点云距离 ===
r = np.linalg.norm(points_np[:, :2], axis=1)

# === Step 2. 分桶自适应聚类 ===
bins = [0, 10, 20, 35, 50, 80]
labels = -np.ones(len(points_np), dtype=int)
current_label = 0

print(f"\n=== 自适应分桶 DBSCAN 聚类 ===")
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (r >= lo) & (r < hi)
    num_points = np.sum(mask)
    if num_points < 20:
        continue

    mean_r = (lo + hi) / 2.0
    
    # 近处使用更小的 eps
    if mean_r < 15:
        eps = 0.15 + 0.005 * mean_r  # 近处更敏感
        min_samples = int(max(5, min(20, 5 + 0.15 * mean_r)))
    else:
        eps = 0.25 + 0.009 * mean_r
        min_samples = int(max(8, min(30, 8 + 0.18 * mean_r)))

    print(f"[DBSCAN {lo:>2}-{hi:<2}m] points={num_points:<6} eps={eps:<4.2f} min_samples={min_samples}")

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points_np[mask, :3])
    cluster_labels = db.labels_
    valid = cluster_labels >= 0
    cluster_labels[valid] += current_label
    labels[mask] = cluster_labels
    if np.any(valid):
        current_label = np.max(labels) + 1

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"✅ 分桶聚类完成，共检测到 {n_clusters} 个初始簇")

# === Step 3. 主轴切分（处理并排或过长车辆） ===
# === 加强切分条件 ===
MAX_CAR_LEN = 4.5   # 降低阈值，更早切分
SPLIT_UNIT  = 3.5   # 更小的切分单元
split_clusters = []

# 在切分逻辑中添加方向检查
for i in range(n_clusters):
    cluster_indices = np.where(labels == i)[0]
    cluster_points = points_np[cluster_indices]
    if len(cluster_points) < 30:
        continue

    pca = PCA(n_components=3)
    pca.fit(cluster_points[:, :3])
    v_main = pca.components_[0]
    proj = cluster_points[:, :3] @ v_main
    proj_min, proj_max = np.min(proj), np.max(proj)
    L = proj_max - proj_min

    # 添加宽度检查：如果宽度也很大，更可能是多车
    v_secondary = pca.components_[1]
    proj_w = cluster_points[:, :3] @ v_secondary
    W = np.max(proj_w) - np.min(proj_w)
    
    # 加强切分条件
    if L > MAX_CAR_LEN or (L > 3.5 and W > 2.5):
        num_splits = max(2, int(np.ceil(L / SPLIT_UNIT)))
        step = L / num_splits
        print(f"🚧 簇 {i}: 长度={L:.2f}m, 宽度={W:.2f}m → 切分为 {num_splits} 段")
        for k in range(num_splits):
            lo, hi = proj_min + k * step, proj_min + (k + 1) * step
            mask = (proj >= lo) & (proj < hi)
            sub_pts = cluster_points[mask]
            if len(sub_pts) >= 10:  # 提高点数要求
                split_clusters.append(sub_pts)
    else:
        split_clusters.append(cluster_points)

    final_clusters = []
    for cluster in split_clusters:
        sub_clusters = detect_side_by_side_vehicles(cluster)
        final_clusters.extend(sub_clusters)

print(f"✅ 切分后共生成 {len(split_clusters)} 个簇")

# === Step 4. 生成 AABB ===
boxes = []
for pts in split_clusters:
    box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(pts)
    )
    size = box.get_extent()
    # 尺寸过滤：排除地面与异常大块
    if size[0] > 10 or size[1] > 5 or size[2] > 3.5:
        continue
    if size[0] < 0.6 or size[1] < 0.4:
        continue
    boxes.append(box)

# === Step 5. 构建点云对象用于可视化 ===
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_np[:, :3])
pcd.paint_uniform_color([0.3, 0.6, 1.0])  # 点云颜色：淡蓝

# === Step 6. 为每个框随机上色 ===
for box in boxes:
    color = np.random.rand(3) * 0.8  # 颜色偏暗一点
    box.color = color

print(f"🎯 最终生成 {len(boxes)} 个 AABB 框")

# === Step 5. 可视化 ===
o3d.visualization.draw_geometries([pcd, *boxes])

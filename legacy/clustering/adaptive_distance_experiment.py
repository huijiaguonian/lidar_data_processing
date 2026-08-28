import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import os

# 距离分桶区间
DIST_BUCKETS = [(0, 15), (15, 30), (30, 50), (50, 80)]
BIN_PATH = "/mnt/d/Hesai_data/frames_bin_fixed/frame_0000.bin"

def adaptive_cluster(points):
    """
    自适应分桶 DBSCAN 聚类 + ROI 限制 + OBB 主轴切分 + 可视化与保存
    """
    boxes = []

    # === Step 1. ROI 限制 ===
    ROI_X = 20.0  # 前后方向 ±20m
    ROI_Y = 5.0   # 左右方向 ±5m
    mask = (np.abs(points[:, 0]) < ROI_X) & (np.abs(points[:, 1]) < ROI_Y)
    points = points[mask]
    print(f"✅ ROI过滤后点数: {len(points)}")

    # === Step 2. 计算水平距离 ===
    r = np.linalg.norm(points[:, :2], axis=1)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3]))
    pcd.paint_uniform_color([0.3, 0.6, 1.0])

    # === Step 3. 分桶聚类 ===
    for (r_min, r_max) in DIST_BUCKETS:
        mask = (r >= r_min) & (r < r_max)
        bucket_pts = points[mask]
        if len(bucket_pts) < 100:
            continue

        mean_r = (r_min + r_max) / 2
        eps = (0.15 + 0.007 * mean_r) * 0.9
        min_samples = int(max(6, min(25, 6 + 0.15 * mean_r)))

        # Z轴自适应降权
        scaled_pts = bucket_pts[:, :3].copy()
        if mean_r < 20:
            z_scale = 0.6
        elif mean_r < 40:
            z_scale = 0.4
        else:
            z_scale = 0.25
        scaled_pts[:, 2] *= z_scale

        print(f"[DBSCAN {r_min:2}-{r_max:2}m] points={len(bucket_pts)} eps={eps:.2f} min_samples={min_samples}")
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_pts)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"  → Detected {n_clusters} clusters")

        # === Step 4. OBB 生成与主轴切分 ===
        for i in range(n_clusters):
            cluster_pts = bucket_pts[labels == i, :3]
            if len(cluster_pts) < 30:
                continue

            cluster_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cluster_pts))
            obb = cluster_pcd.get_oriented_bounding_box()
            extent = obb.extent
            ratio = extent[0] / extent[1] if extent[0] > extent[1] else extent[1] / extent[0]

            # 如果簇太长，按主轴方向切分
            if ratio > 3.0 and max(extent[0], extent[1]) > 6.0:
                pca = PCA(n_components=3)
                pca.fit(cluster_pts)
                main_axis = pca.components_[0]
                main_axis /= np.linalg.norm(main_axis)

                proj = np.dot(cluster_pts, main_axis)
                proj_min, proj_max = proj.min(), proj.max()
                split_step = 4.5  # 平均一辆车长度
                split_points = np.arange(proj_min, proj_max, split_step)

                for s in split_points:
                    sub_mask = (proj >= s) & (proj < s + split_step)
                    sub_pts = cluster_pts[sub_mask]
                    if len(sub_pts) < 30:
                        continue
                    sub_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sub_pts))
                    sub_obb = sub_pcd.get_oriented_bounding_box()
                    sub_obb.color = np.random.rand(3)
                    boxes.append(sub_obb)
            else:
                # 普通目标直接生成OBB
                if extent[0] < 0.6 or extent[1] < 0.4 or extent[2] < 0.3:
                    continue
                if extent[0] > 10 or extent[1] > 5 or extent[2] > 3.5:
                    continue
                obb.color = np.random.rand(3)
                boxes.append(obb)

    # === Step 5. 保存与可视化 ===
    SAVE_PATH = "/mnt/d/Hesai_output/unsupervised_obb_roi_final.ply"
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    o3d.io.write_point_cloud(SAVE_PATH, pcd)
    print(f"📁 已保存点云到: {SAVE_PATH}")
    print(f"✅ 共生成 {len(boxes)} 个目标框")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Adaptive ROI + OBB", width=1280, height=720)
    vis.add_geometry(pcd)
    for b in boxes:
        vis.add_geometry(b)
    vis.run()
    vis.destroy_window()

    return pcd, boxes


# ======================== Example usage ========================
if __name__ == "__main__":

    points = np.fromfile(BIN_PATH, dtype=np.float32).reshape(-1, 4)
    points_np = points[:, :3]
    print(f"Loaded {len(points_np)} points")

    adaptive_cluster(points)

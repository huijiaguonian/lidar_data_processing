import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

# === 参数配置 ===
BIN_PATH = "/mnt/d/Hesai_data/frames_bin_fixed/frame_0000.bin"
Z_MIN, Z_MAX = -1.5, 2.0  # 调整Z范围，专注于车辆高度

def preprocess_point_cloud(points_np):
    """点云预处理"""
    # 1. 移除地面点（简单高度过滤）
    ground_mask = points_np[:, 2] > -1.0  # 保留高于地面1米的点
    filtered_points = points_np[ground_mask]
    
    # 2. 统计过滤（移除孤立点）
    from sklearn.neighbors import NearestNeighbors
    if len(filtered_points) > 100:
        nbrs = NearestNeighbors(n_neighbors=10).fit(filtered_points)
        distances, indices = nbrs.kneighbors(filtered_points)
        avg_distances = np.mean(distances, axis=1)
        
        # 保留距离适中的点
        distance_mask = avg_distances < 2.0
        filtered_points = filtered_points[distance_mask]
    
    print(f"预处理后点数: {len(filtered_points)}")
    return filtered_points

def adaptive_dbscan_clustering(points_np):
    """自适应的DBSCAN聚类"""
    clusters = []
    
    # 根据点云密度动态调整参数
    if len(points_np) < 1000:
        # 稀疏点云使用较小参数
        eps_values = [0.3, 0.4, 0.5]
        min_samples_values = [5, 8, 10]
    else:
        # 密集点云使用较大参数
        eps_values = [0.4, 0.6, 0.8]
        min_samples_values = [10, 15, 20]
    
    best_clusters = []
    best_score = 0
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(points_np)
            labels = db.labels_
            
            # 计算聚类质量（簇的数量和大小）
            unique_labels = set(labels)
            n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
            
            if n_clusters > 0:
                cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
                avg_size = np.mean(cluster_sizes)
                
                # 好的聚类应该有适中的簇数量和大小
                score = n_clusters * avg_size
                
                if score > best_score and 5 <= avg_size <= 500:
                    best_score = score
                    best_clusters = []
                    for i in range(n_clusters):
                        cluster_indices = np.where(labels == i)[0]
                        if 10 <= len(cluster_indices) <= 1000:  # 合理的簇大小范围
                            best_clusters.append(points_np[cluster_indices])
    
    print(f"最佳参数找到 {len(best_clusters)} 个簇")
    return best_clusters

def distance_based_clustering(points_np):
    """基于距离的分层聚类"""
    from sklearn.cluster import DBSCAN
    
    # 计算点到原点的距离
    distances = np.linalg.norm(points_np[:, :2], axis=1)
    
    clusters = []
    
    # 按距离分层处理
    distance_bins = [(0, 20), (20, 40), (40, 60), (60, 100)]
    
    for dist_min, dist_max in distance_bins:
        mask = (distances >= dist_min) & (distances < dist_max)
        bin_points = points_np[mask]
        
        if len(bin_points) < 10:
            continue
        
        # 根据距离调整聚类参数
        mean_dist = (dist_min + dist_max) / 2
        eps = 0.3 + 0.02 * mean_dist  # 远处使用更大的eps
        min_samples = max(5, int(8 + 0.1 * mean_dist))
        
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(bin_points)
        labels = db.labels_
        
        for i in range(np.max(labels) + 1):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) >= 8:  # 降低最小点数要求
                clusters.append(bin_points[cluster_indices])
    
    print(f"距离分层聚类找到 {len(clusters)} 个簇")
    return clusters

def simple_pca_bbox(points):
    """简单的PCA边界框拟合"""
    if len(points) < 8:
        return None
    
    try:
        # 使用PCA找到主方向
        pca = PCA(n_components=2)
        points_2d = points[:, :2]
        pca.fit(points_2d)
        
        # 主方向
        main_axis = pca.components_[0]
        angle = np.arctan2(main_axis[1], main_axis[0])
        
        # 旋转点云
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        centered_points = points_2d - np.mean(points_2d, axis=0)
        rotated_points = centered_points @ rotation_matrix.T
        
        # 计算边界
        min_vals = np.min(rotated_points, axis=0)
        max_vals = np.max(rotated_points, axis=0)
        
        # 创建边界框角点
        corners_rotated = np.array([
            [min_vals[0], min_vals[1]],
            [max_vals[0], min_vals[1]],
            [max_vals[0], max_vals[1]],
            [min_vals[0], max_vals[1]]
        ])
        
        # 旋转回原始坐标系
        inv_rotation = rotation_matrix.T
        corners_original = corners_rotated @ inv_rotation + np.mean(points_2d, axis=0)
        
        # Z范围
        z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
        
        # 创建3D边界框
        vertices = []
        for corner in corners_original:
            vertices.append([corner[0], corner[1], z_min])
        for corner in corners_original:
            vertices.append([corner[0], corner[1], z_max])
        
        vertices = np.array(vertices)
        
        # 线连接
        lines = [
            [0,1], [1,2], [2,3], [3,0],  # 底面
            [4,5], [5,6], [6,7], [7,4],  # 顶面
            [0,4], [1,5], [2,6], [3,7]   # 侧面
        ]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        
        return line_set
        
    except Exception as e:
        return None

def filter_clusters_by_size(clusters):
    """根据尺寸过滤簇"""
    filtered_clusters = []
    
    for cluster in clusters:
        if len(cluster) < 8 or len(cluster) > 1000:
            continue
            
        # 计算边界框尺寸
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(cluster)
        )
        size = bbox.get_extent()
        
        # 车辆尺寸范围（宽松条件）
        if (0.5 < size[0] < 8.0 and  # 长度
            0.3 < size[1] < 4.0 and  # 宽度  
            0.3 < size[2] < 3.5):    # 高度
            filtered_clusters.append(cluster)
        else:
            print(f"过滤尺寸: {size}")
    
    return filtered_clusters

# === 主程序 ===
def main():
    # 读取点云
    points = np.fromfile(BIN_PATH, dtype=np.float32).reshape(-1, 4)
    points_np = points[:, :3]
    print(f"原始点数: {len(points_np)}")
    
    # 预处理
    processed_points = preprocess_point_cloud(points_np)
    
    # 尝试不同的聚类方法
    print("\n尝试自适应DBSCAN聚类...")
    clusters1 = adaptive_dbscan_clustering(processed_points)
    
    print("\n尝试距离分层聚类...")
    clusters2 = distance_based_clustering(processed_points)
    
    # 合并结果
    all_clusters = clusters1 + clusters2
    
    # 去重（基于空间位置）
    unique_clusters = []
    cluster_centers = []
    
    for cluster in all_clusters:
        center = np.mean(cluster, axis=0)
        is_duplicate = False
        
        for existing_center in cluster_centers:
            if np.linalg.norm(center - existing_center) < 2.0:  # 2米内认为是重复
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_clusters.append(cluster)
            cluster_centers.append(center)
    
    print(f"去重后簇数量: {len(unique_clusters)}")
    
    # 尺寸过滤
    filtered_clusters = filter_clusters_by_size(unique_clusters)
    print(f"尺寸过滤后: {len(filtered_clusters)} 个簇")
    
    # 生成边界框
    bboxes = []
    for i, cluster in enumerate(filtered_clusters):
        color = np.random.rand(3)
        bbox = simple_pca_bbox(cluster)
        if bbox is not None:
            bbox.paint_uniform_color(color)
            bboxes.append(bbox)
    
    print(f"🎯 生成边界框: {len(bboxes)} 个")
    
    # 可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(processed_points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色点云
    
    if bboxes:
        # 为每个边界框添加标签
        geometries = [pcd] + bboxes
        o3d.visualization.draw_geometries(geometries)
    else:
        print("❌ 没有生成边界框，显示原始点云")
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
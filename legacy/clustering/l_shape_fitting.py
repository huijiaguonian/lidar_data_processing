import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# === 参数配置 ===
BIN_PATH = "/mnt/d/Hesai_data/frames_bin_fixed/frame_0000.bin"
Z_MIN, Z_MAX = -5.0, 10.0

# === 修复的 L-shape 拟合函数 ===
def fit_l_shape(points_2d):
    """拟合 L-shape 边界框 - 修复版本"""
    if len(points_2d) < 10:
        return None
    
    try:
        # 使用凸包减少噪声影响
        if len(points_2d) > 10:
            hull = ConvexHull(points_2d)
            hull_points = points_2d[hull.vertices]
        else:
            hull_points = points_2d
        
        def objective_function(angle):
            """目标函数：最小化旋转后的边界框面积 - 修复返回标量"""
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            # 旋转点云
            rotated_points = hull_points @ rotation_matrix.T
            
            # 计算轴对齐边界框
            min_vals = np.min(rotated_points, axis=0)
            max_vals = np.max(rotated_points, axis=0)
            
            # 边界框面积
            width = max_vals[0] - min_vals[0]
            height = max_vals[1] - min_vals[1]
            area = width * height
            
            return float(area)  # 确保返回标量
        
        # 优化找到最佳角度
        result = minimize(objective_function, 0, 
                         bounds=[(-np.pi/2, np.pi/2)], 
                         method='L-BFGS-B')
        
        if not result.success:
            print(f"优化失败: {result.message}")
            return None
            
        best_angle = result.x[0]
        cos_a, sin_a = np.cos(best_angle), np.sin(best_angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # 计算最佳边界框
        rotated_points = hull_points @ rotation_matrix.T
        min_vals = np.min(rotated_points, axis=0)
        max_vals = np.max(rotated_points, axis=0)
        
        # 计算边界框的四个角点（在旋转后的坐标系中）
        corners_rotated = np.array([
            [min_vals[0], min_vals[1]],
            [max_vals[0], min_vals[1]],
            [max_vals[0], max_vals[1]],
            [min_vals[0], max_vals[1]]
        ])
        
        # 旋转回原始坐标系
        inv_rotation_matrix = rotation_matrix.T
        corners_original = corners_rotated @ inv_rotation_matrix
        
        return corners_original, best_angle
        
    except Exception as e:
        print(f"L-shape 拟合错误: {e}")
        return None

# === 备用的简化 L-shape 拟合 ===
def fit_simple_l_shape(points_2d):
    """简化的 L-shape 拟合，使用 PCA 方向"""
    if len(points_2d) < 10:
        return None
    
    try:
        # 使用 PCA 找到主方向
        pca = PCA(n_components=2)
        pca.fit(points_2d)
        
        # 主方向角度
        main_axis = pca.components_[0]
        angle = np.arctan2(main_axis[1], main_axis[0])
        
        # 旋转矩阵
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        
        # 旋转点云
        centered_points = points_2d - np.mean(points_2d, axis=0)
        rotated_points = centered_points @ rotation_matrix.T
        
        # 计算边界框
        min_vals = np.min(rotated_points, axis=0)
        max_vals = np.max(rotated_points, axis=0)
        
        # 计算角点
        corners_rotated = np.array([
            [min_vals[0], min_vals[1]],
            [max_vals[0], min_vals[1]],
            [max_vals[0], max_vals[1]],
            [min_vals[0], max_vals[1]]
        ])
        
        # 旋转回原始坐标系
        inv_rotation_matrix = rotation_matrix.T
        corners_original = corners_rotated @ inv_rotation_matrix + np.mean(points_2d, axis=0)
        
        return corners_original, angle
        
    except Exception as e:
        print(f"简化 L-shape 拟合错误: {e}")
        return None

def create_l_shape_box_3d(points, color=None):
    """创建 3D 的 L-shape 边界框"""
    if len(points) < 10:
        return None
    
    # 2D 投影
    points_2d = points[:, :2]
    
    # 先尝试优化的 L-shape 拟合
    result = fit_l_shape(points_2d)
    
    # 如果失败，使用简化的 PCA 方法
    if result is None:
        print("优化 L-shape 失败，使用简化方法")
        result = fit_simple_l_shape(points_2d)
    
    if result is None:
        return None
    
    corners_2d, angle = result
    
    if color is None:
        color = np.random.rand(3)
    
    # Z 范围
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
    
    # 确保角点按顺时针顺序排列
    center = np.mean(corners_2d, axis=0)
    angles = []
    for corner in corners_2d:
        dx = corner[0] - center[0]
        dy = corner[1] - center[1]
        angles.append(np.arctan2(dy, dx))
    
    # 按角度排序（顺时针）
    sorted_indices = np.argsort(angles)[::-1]
    corners_2d = corners_2d[sorted_indices]
    
    # 创建 3D 边界框的8个顶点
    vertices = []
    # 底面顶点（顺时针）
    for corner in corners_2d:
        vertices.append([corner[0], corner[1], z_min])
    
    # 顶面顶点（顺时针，与底面顺序一致）
    for corner in corners_2d:
        vertices.append([corner[0], corner[1], z_max])
    
    vertices = np.array(vertices)
    
    # 创建线框 - 正确的连接顺序
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面四边形
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面四边形
        [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面连接
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    
    return line_set

# === 改进的聚类函数（放宽尺寸过滤）===
def cluster_points(points_np):
    """聚类点云 - 放宽尺寸过滤"""
    r = np.linalg.norm(points_np[:, :2], axis=1)
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
            eps = 0.15 + 0.005 * mean_r
            min_samples = int(max(5, min(20, 5 + 0.15 * mean_r)))
        else:
            eps = 0.25 + 0.009 * mean_r
            min_samples = int(max(8, min(30, 8 + 0.18 * mean_r)))

        print(f"[DBSCAN {lo:>2}-{hi:<2}m] eps={eps:<4.2f} min_samples={min_samples}")

        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points_np[mask, :3])
        cluster_labels = db.labels_
        valid = cluster_labels >= 0
        cluster_labels[valid] += current_label
        labels[mask] = cluster_labels
        if np.any(valid):
            current_label = np.max(labels) + 1

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"✅ 聚类完成，共检测到 {n_clusters} 个簇")
    
    # 提取聚类点云并放宽尺寸过滤
    clusters = []
    for i in range(n_clusters):
        cluster_indices = np.where(labels == i)[0]
        if len(cluster_indices) < 8:  # 降低点数要求
            continue
            
        cluster_pts = points_np[cluster_indices]
        
        # 放宽尺寸过滤条件
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(cluster_pts)
        )
        size = bbox.get_extent()
        
        # 更宽松的尺寸范围
        if (0.3 < size[0] < 15.0 and  # 放宽长度限制
            0.2 < size[1] < 8.0 and   # 放宽宽度限制
            0.2 < size[2] < 5.0):     # 放宽高度限制
            clusters.append(cluster_pts)
            print(f"保留簇 {i}: 尺寸 {size}, 点数 {len(cluster_pts)}")
        else:
            print(f"过滤簇 {i}: 尺寸异常 {size}")
    
    return clusters

# === 主程序 ===
def main():
    # 读取点云
    points = np.fromfile(BIN_PATH, dtype=np.float32).reshape(-1, 4)
    points_np = points[:, :3]
    print(f"Loaded {len(points_np)} points from {BIN_PATH}")
    
    # Z过滤
    mask = (points_np[:, 2] > Z_MIN) & (points_np[:, 2] < Z_MAX)
    points_np = points_np[mask]
    print(f"After Z-filter: {len(points_np)} points")
    
    # 聚类
    clusters = cluster_points(points_np)
    print(f"✅ 过滤后剩余 {len(clusters)} 个有效簇")
    
    # 生成 L-shape 框
    lshape_boxes = []
    success_count = 0
    
    for i, cluster_pts in enumerate(clusters):
        color = np.random.rand(3)
        
        lshape_box = create_l_shape_box_3d(cluster_pts, color)
        if lshape_box is not None:
            lshape_boxes.append(lshape_box)
            success_count += 1
            print(f"✅ 簇 {i}: 成功生成 L-shape 框 ({len(cluster_pts)} 点)")
        else:
            print(f"❌ 簇 {i}: L-shape 拟合失败 ({len(cluster_pts)} 点)")
    
    print(f"🎯 成功生成 L-shape 框: {success_count} 个")
    
    # 创建点云用于可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.paint_uniform_color([0.3, 0.6, 1.0])  # 蓝色点云
    
    # 可视化
    if len(lshape_boxes) > 0:
        visualization_geometries = [pcd] + lshape_boxes
        print("开始可视化 L-shape 边界框...")
        o3d.visualization.draw_geometries(visualization_geometries)
    else:
        print("❌ 没有生成任何 L-shape 框，只显示点云")
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
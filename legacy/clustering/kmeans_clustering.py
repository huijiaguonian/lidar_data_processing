import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# === 参数配置 ===
BIN_PATH = "/mnt/d/Hesai_data/frames_bin_fixed/frame_0000.bin"

def read_point_cloud():
    """读取点云"""
    points = np.fromfile(BIN_PATH, dtype=np.float32).reshape(-1, 4)
    points_np = points[:, :3]
    
    # 移除地面和过高点
    z_mask = (points_np[:, 2] > -1.0) & (points_np[:, 2] < 3.0)
    points_np = points_np[z_mask]
    
    print(f"处理后点数: {len(points_np)}")
    return points_np

def kmeans_with_elbow(points_np, max_k=20):
    """使用肘部法则确定K值的KMeans聚类"""
    from sklearn.metrics import silhouette_score
    
    # 提取特征：位置 + 局部特征
    features = []
    for i in range(len(points_np)):
        # 基本特征：位置 + 高度
        feature = [
            points_np[i, 0],  # x
            points_np[i, 1],  # y  
            points_np[i, 2],  # z
            np.linalg.norm(points_np[i, :2])  # 距离
        ]
        features.append(feature)
    
    features = np.array(features)
    features = StandardScaler().fit_transform(features)
    
    # 肘部法则找最佳K值
    inertias = []
    k_range = range(2, min(max_k, len(points_np)//10))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
    
    # 找肘点（斜率变化最大的点）
    differences = np.diff(inertias)
    second_diff = np.diff(differences)
    optimal_k = k_range[np.argmin(second_diff) + 2] if len(second_diff) > 0 else 8
    
    print(f"肘部法则确定最佳K值: {optimal_k}")
    
    # 使用最佳K值聚类
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    # 提取聚类
    clusters = []
    for i in range(optimal_k):
        cluster_points = points_np[labels == i]
        if len(cluster_points) >= 10:  # 过滤小簇
            clusters.append(cluster_points)
    
    return clusters

def gmm_clustering(points_np, max_components=15):
    """高斯混合模型聚类"""
    features = points_np.copy()
    features = StandardScaler().fit_transform(features)
    
    # 使用BIC准则选择最佳组件数
    best_gmm = None
    best_bic = np.inf
    
    for n_components in range(2, min(max_components, len(points_np)//20)):
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(features)
        bic = gmm.bic(features)
        
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
    
    labels = best_gmm.predict(features)
    
    clusters = []
    for i in range(best_gmm.n_components):
        cluster_points = points_np[labels == i]
        if len(cluster_points) >= 10:
            clusters.append(cluster_points)
    
    print(f"GMM找到 {len(clusters)} 个簇")
    return clusters

def create_bounding_box(points, color=None):
    """创建边界框"""
    if len(points) < 5:
        return None
    
    if color is None:
        color = np.random.rand(3)
    
    # 使用PCA找到主方向
    from sklearn.decomposition import PCA
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
    
    # 创建边界框
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
    
    lines = [
        [0,1], [1,2], [2,3], [3,0],
        [4,5], [5,6], [6,7], [7,4],
        [0,4], [1,5], [2,6], [3,7]
    ]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    
    return line_set

def main():
    points_np = read_point_cloud()
    
    print("=== 方法1: KMeans聚类 ===")
    clusters1 = kmeans_with_elbow(points_np)
    
    print("=== 方法2: GMM聚类 ===")
    clusters2 = gmm_clustering(points_np)
    
    # 合并结果
    all_clusters = clusters1 + clusters2
    
    # 生成边界框
    bboxes = []
    for cluster in all_clusters:
        bbox = create_bounding_box(cluster)
        if bbox is not None:
            bboxes.append(bbox)
    
    print(f"🎯 生成边界框: {len(bboxes)} 个")
    
    # 可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.paint_uniform_color([0.7, 0.7, 0.7])
    
    if bboxes:
        o3d.visualization.draw_geometries([pcd] + bboxes)
    else:
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
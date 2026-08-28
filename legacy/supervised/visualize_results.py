import open3d as o3d
import numpy as np
import json
import torch

def visualize_detections(point_cloud_file, result_file):
    """可视化点云和检测结果"""
    
    # 加载点云
    points = np.fromfile(point_cloud_file, dtype=np.float32).reshape(-1, 4)
    print(f"加载点云: {len(points)} 个点")
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色点云
    
    # 加载检测结果
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    # 创建检测框
    bbox_list = []
    colors = [
        [1, 0, 0],   # 红色
        [0, 1, 0],   # 绿色  
        [0, 0, 1],   # 蓝色
        [1, 1, 0],   # 黄色
        [1, 0, 1],   # 紫色
    ]
    
    frame = results[0]
    detections = frame['detections']
    
    print(f"可视化 {len(detections)} 个检测框")
    
    for i, det in enumerate(detections):
        if det['score'] < 0.3:  # 只显示高置信度检测
            continue
            
        # 获取框参数
        center = np.array([det['position']['x'], det['position']['y'], det['position']['z']])
        extent = np.array([det['dimensions']['length'], det['dimensions']['width'], det['dimensions']['height']])
        rotation = det['rotation']
        
        # 创建定向边界框
        bbox = o3d.geometry.OrientedBoundingBox(center, 
                                               o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, rotation]),
                                               extent)
        
        # 根据置信度设置颜色
        color_idx = min(int(det['score'] * len(colors)), len(colors)-1)
        bbox.color = colors[color_idx]
        
        bbox_list.append(bbox)
        
        # 显示检测信息
        if i < 5:  # 只显示前5个
            print(f"检测框 {i+1}: 分数={det['score']:.3f}, 位置=({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    
    # 可视化
    print("\n🎯 可视化说明:")
    print("   - 灰色: 点云")
    print("   - 红色→紫色: 检测框 (红色=低置信度, 紫色=高置信度)")
    print("   - 按 'H' 键显示帮助")
    print("   - 按 'Q' 键退出")
    
    o3d.visualization.draw_geometries([pcd] + bbox_list,
                                     window_name="3D检测结果可视化",
                                     width=1200,
                                     height=800)

if __name__ == "__main__":
    point_cloud_file = "/mnt/d/pcdet-pandaset/data/pandar128_custom/velodyne/frame_0000.bin"
    result_file = "/mnt/d/Hesai_output/my_results/detection_results.json"
    
    visualize_detections(point_cloud_file, result_file)
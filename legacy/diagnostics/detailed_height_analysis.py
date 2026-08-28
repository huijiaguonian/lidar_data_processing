import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gzip
import pickle

def load_pandaset_data(pandaset_path):
    """正确加载Pandaset的.pkl.gz文件"""
    try:
        # 方法1: 使用pandas读取
        df = pd.read_pickle(pandaset_path)
        print(f"Pandaset数据列: {df.columns.tolist()}")
        
        # 查找坐标列
        x_col = next((col for col in df.columns if col.lower() in ['x', 'coordx']), None)
        y_col = next((col for col in df.columns if col.lower() in ['y', 'coordy']), None) 
        z_col = next((col for col in df.columns if col.lower() in ['z', 'coordz']), None)
        i_col = next((col for col in df.columns if col.lower() in ['i', 'intensity']), None)
        
        if all(col is not None for col in [x_col, y_col, z_col]):
            points = df[[x_col, y_col, z_col]].values
            if i_col:
                intensity = df[i_col].values.reshape(-1, 1)
                points = np.hstack([points, intensity])
            else:
                points = np.hstack([points, np.zeros((len(points), 1))])
            return points
        else:
            raise ValueError("找不到坐标列")
            
    except Exception as e:
        print(f"Pandas读取失败: {e}")
        # 方法2: 直接使用pickle加载
        try:
            with gzip.open(pandaset_path, 'rb') as f:
                data = pickle.load(f)
                print(f"直接加载的数据类型: {type(data)}")
                
                if isinstance(data, np.ndarray):
                    return data
                elif isinstance(data, dict):
                    # 尝试从字典中提取点云
                    for key in ['points', 'lidar', 'cloud']:
                        if key in data:
                            return data[key]
                else:
                    raise ValueError(f"未知的数据格式: {type(data)}")
                    
        except Exception as e2:
            print(f"直接加载也失败: {e2}")
            return None

def detailed_height_analysis(pandaset_path, pandar_path):
    """修复版的高度分析"""
    
    # 加载数据
    points_pandaset = load_pandaset_data(pandaset_path)
    if points_pandaset is None:
        print("❌ 无法加载Pandaset数据")
        return
    
    points_pandar = np.fromfile(pandar_path, dtype=np.float32).reshape(-1, 4)
    
    print("📐 详细高度分析:")
    print("=" * 50)
    
    # Pandaset分析
    pandaset_z = points_pandaset[:, 2]
    pandaset_ground = np.percentile(pandaset_z, 2)
    pandaset_objects_mask = pandaset_z > pandaset_ground + 0.2
    pandaset_objects = pandaset_z[pandaset_objects_mask]
    
    print("Pandaset数据:")
    print(f"  点数: {len(points_pandaset)}")
    print(f"  Z范围: {pandaset_z.min():.2f} 到 {pandaset_z.max():.2f} m")
    print(f"  地面高度(2%分位): {pandaset_ground:.2f} m")
    if len(pandaset_objects) > 0:
        print(f"  物体点高度: {pandaset_objects.min():.2f} 到 {pandaset_objects.max():.2f} m")
    else:
        print("  没有检测到明显的物体点")
    print(f"  雷达安装高度估计: {pandaset_z.max():.2f} m")
    
    # Pandar128分析
    pandar_z = points_pandar[:, 2]
    pandar_ground = np.percentile(pandar_z, 2)
    pandar_objects_mask = pandar_z > pandar_ground + 0.2
    pandar_objects = pandar_z[pandar_objects_mask]
    
    print("\nPandar128数据:")
    print(f"  点数: {len(points_pandar)}")
    print(f"  Z范围: {pandar_z.min():.2f} 到 {pandar_z.max():.2f} m")
    print(f"  地面高度(2%分位): {pandar_ground:.2f} m")
    if len(pandar_objects) > 0:
        print(f"  物体点高度: {pandar_objects.min():.2f} 到 {pandar_objects.max():.2f} m")
    else:
        print("  没有检测到明显的物体点")
    print(f"  雷达安装高度估计: {pandar_z.max():.2f} m")
    
    # 计算偏移
    z_offset = pandaset_ground - pandar_ground
    print(f"\n📏 计算偏移:")
    print(f"  地面高度差: {z_offset:.2f} m")
    print(f"  你的雷达比Pandaset低: {abs(z_offset):.2f} m")
    
    # 可视化
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.hist(pandaset_z, bins=50, alpha=0.7, label='Pandaset', color='blue')
    plt.hist(pandar_z, bins=50, alpha=0.7, label='Pandar128', color='red')
    plt.axvline(pandaset_ground, color='blue', linestyle='--', label='Pandaset地面')
    plt.axvline(pandar_ground, color='red', linestyle='--', label='Pandar128地面')
    plt.xlabel('Z高度 (m)')
    plt.ylabel('点数')
    plt.legend()
    plt.title('高度分布对比')
    
    plt.subplot(1, 3, 2)
    if len(pandaset_objects) > 0 and len(pandar_objects) > 0:
        plt.hist(pandaset_objects, bins=50, alpha=0.7, label='Pandaset物体', color='blue')
        plt.hist(pandar_objects, bins=50, alpha=0.7, label='Pandar128物体', color='red')
        plt.xlabel('Z高度 (m)')
        plt.ylabel('点数')
        plt.legend()
        plt.title('物体高度分布')
    else:
        plt.text(0.5, 0.5, '没有足够的物体点', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('物体高度分布 (数据不足)')
    
    plt.subplot(1, 3, 3)
    # 随机采样避免点太多
    pandaset_sample = points_pandaset[np.random.choice(len(points_pandaset), min(5000, len(points_pandaset)), replace=False)]
    pandar_sample = points_pandar[np.random.choice(len(points_pandar), min(5000, len(points_pandar)), replace=False)]
    
    plt.scatter(pandaset_sample[:, 0], pandaset_sample[:, 2], s=1, alpha=0.5, label='Pandaset', color='blue')
    plt.scatter(pandar_sample[:, 0], pandar_sample[:, 2], s=1, alpha=0.5, label='Pandar128', color='red')
    plt.xlabel('X (m)')
    plt.ylabel('Z (m)')
    plt.legend()
    plt.title('XZ平面视图 (采样5000点)')
    
    plt.tight_layout()
    plt.show()
    
    return z_offset

# 运行分析
if __name__ == "__main__":
    pandaset_path = "/mnt/d/pandaset_down/001/lidar/00.pkl.gz"
    pandar_path = "/mnt/d/Hesai_data/frames_bin_fixed/frame_0000.bin"
    
    z_offset = detailed_height_analysis(pandaset_path, pandar_path)
    print(f"\n🎯 建议的Z偏移值: {z_offset:.2f} m")
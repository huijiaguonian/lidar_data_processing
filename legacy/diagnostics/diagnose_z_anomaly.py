import numpy as np
import matplotlib.pyplot as plt
import struct
from scipy import stats

def diagnose_bin_file_z_anomaly(bin_file_path):
    """
    诊断Pandar128 bin文件的Z坐标异常
    """
    print("🔍 Pandar128 Bin文件Z坐标异常诊断")
    print("=" * 60)
    
    # 1. 加载数据
    try:
        points = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
        print(f"✅ 文件加载成功: {bin_file_path}")
        print(f"   数据形状: {points.shape}")
    except Exception as e:
        print(f"❌ 文件加载失败: {e}")
        return
    
    # 2. 基本统计信息
    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    
    print(f"\n📊 基本统计信息:")
    print(f"   X范围: {x.min():8.2f} 到 {x.max():8.2f} m")
    print(f"   Y范围: {y.min():8.2f} 到 {y.max():8.2f} m")
    print(f"   Z范围: {z.min():8.2f} 到 {z.max():8.2f} m")
    print(f"   强度范围: {intensity.min():6.1f} 到 {intensity.max():6.1f}")
    
    # 3. Z坐标详细分析
    print(f"\n🎯 Z坐标详细分析:")
    
    # 分位数分析
    z_quantiles = np.percentile(z, [0, 1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100])
    quantile_labels = ['0%', '1%', '2%', '5%', '10%', '25%', '50%', '75%', '90%', '95%', '98%', '99%', '100%']
    
    print("   Z坐标分位数:")
    for label, value in zip(quantile_labels, z_quantiles):
        print(f"     {label}: {value:8.2f} m")
    
    # 4. 异常检测
    print(f"\n🚨 异常检测:")
    
    # 检查Z坐标是否在合理范围内
    reasonable_min, reasonable_max = -5, 10  # 合理的Z范围
    z_out_of_range = np.sum((z < reasonable_min) | (z > reasonable_max))
    z_out_of_range_ratio = z_out_of_range / len(z)
    
    print(f"   Z坐标超出合理范围({reasonable_min}~{reasonable_max}m): {z_out_of_range} 点 ({z_out_of_range_ratio:.1%})")
    
    if z_out_of_range_ratio > 0.5:
        print("   ❌ 严重异常: 超过50%的点Z坐标异常")
    elif z_out_of_range_ratio > 0.1:
        print("   ⚠️ 中度异常: 超过10%的点Z坐标异常")
    else:
        print("   ✅ Z坐标范围基本正常")
    
    # 5. 地面检测
    print(f"\n🏔️ 地面检测:")
    
    # 多种地面估计方法
    ground_methods = {
        '1%分位数': np.percentile(z, 1),
        '2%分位数': np.percentile(z, 2),
        '5%分位数': np.percentile(z, 5),
        '最低点簇均值': estimate_ground_cluster(z),
        '统计众数': estimate_ground_mode(z)
    }
    
    for method, value in ground_methods.items():
        print(f"   {method:12}: {value:8.2f} m")
    
    # 6. 数据分布可视化
    print(f"\n📈 生成可视化图表...")
    visualize_z_analysis(points, bin_file_path)
    
    # 7. 问题诊断建议
    print(f"\n💡 诊断建议:")
    
    ground_estimate = ground_methods['2%分位数']
    
    if ground_estimate < -10:
        print(f"   ❌ 严重问题: 地面高度异常低 ({ground_estimate:.2f}m)")
        print(f"      建议应用Z偏移: +{abs(ground_estimate) + 1.6:.2f} m")
    elif ground_estimate < -2:
        print(f"   ⚠️ 中度问题: 地面高度偏低 ({ground_estimate:.2f}m)")
        print(f"      建议应用Z偏移: +{abs(ground_estimate) + 1.6:.2f} m")
    elif -1 < ground_estimate < 1:
        print(f"   ✅ 正常: 地面高度合理 ({ground_estimate:.2f}m)")
    else:
        print(f"   ⚠️ 警告: 地面高度异常 ({ground_estimate:.2f}m)")
    
    # 8. 修复建议
    suggested_offset = -1.6 - ground_estimate  # 目标地面-1.6米
    print(f"\n🔧 修复建议:")
    print(f"   当前地面: {ground_estimate:.2f} m")
    print(f"   目标地面: -1.60 m (Pandaset标准)")
    print(f"   建议Z偏移: {suggested_offset:.2f} m")
    print(f"   修复命令: points[:, 2] += {suggested_offset:.2f}")

def estimate_ground_cluster(z_values, cluster_threshold=0.1):
    """使用聚类方法估计地面高度"""
    sorted_z = np.sort(z_values)
    # 取最低的10%点
    low_points = sorted_z[:len(sorted_z)//10]
    # 使用这些点的中位数作为地面估计
    return np.median(low_points)

def estimate_ground_mode(z_values, bin_width=0.05):
    """使用众数估计地面高度"""
    # 创建直方图
    hist, bin_edges = np.histogram(z_values, bins=100)
    # 找到最高频的bin
    mode_index = np.argmax(hist)
    # 返回该bin的中心值
    return (bin_edges[mode_index] + bin_edges[mode_index+1]) / 2

def visualize_z_analysis(points, file_path):
    """可视化Z坐标分析"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Pandar128 Bin文件Z坐标诊断 - {file_path}', fontsize=16)
    
    # 1. Z坐标直方图
    axes[0, 0].hist(z, bins=100, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].axvline(z.min(), color='blue', linestyle='--', label=f'最小值: {z.min():.2f}m')
    axes[0, 0].axvline(z.max(), color='green', linestyle='--', label=f'最大值: {z.max():.2f}m')
    axes[0, 0].axvline(np.percentile(z, 2), color='orange', linestyle='--', label=f'2%分位: {np.percentile(z, 2):.2f}m')
    axes[0, 0].set_xlabel('Z坐标 (m)')
    axes[0, 0].set_ylabel('点数')
    axes[0, 0].set_title('Z坐标分布')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. XZ平面视图
    sample_indices = np.random.choice(len(points), min(5000, len(points)), replace=False)
    sample_points = points[sample_indices]
    scatter = axes[0, 1].scatter(sample_points[:, 0], sample_points[:, 2], 
                                c=sample_points[:, 2], cmap='viridis', s=1, alpha=0.6)
    axes[0, 1].set_xlabel('X坐标 (m)')
    axes[0, 1].set_ylabel('Z坐标 (m)')
    axes[0, 1].set_title('XZ平面视图 (采样5000点)')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 1], label='Z坐标 (m)')
    
    # 3. YZ平面视图
    scatter = axes[0, 2].scatter(sample_points[:, 1], sample_points[:, 2], 
                                c=sample_points[:, 2], cmap='viridis', s=1, alpha=0.6)
    axes[0, 2].set_xlabel('Y坐标 (m)')
    axes[0, 2].set_ylabel('Z坐标 (m)')
    axes[0, 2].set_title('YZ平面视图 (采样5000点)')
    axes[0, 2].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 2], label='Z坐标 (m)')
    
    # 4. Z坐标箱线图
    axes[1, 0].boxplot(z, vert=True)
    axes[1, 0].set_ylabel('Z坐标 (m)')
    axes[1, 0].set_title('Z坐标箱线图')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 累积分布函数
    z_sorted = np.sort(z)
    cdf = np.arange(1, len(z_sorted)+1) / len(z_sorted)
    axes[1, 1].plot(z_sorted, cdf, linewidth=2)
    axes[1, 1].set_xlabel('Z坐标 (m)')
    axes[1, 1].set_ylabel('累积概率')
    axes[1, 1].set_title('Z坐标累积分布函数')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 修复前后对比
    ground_estimate = np.percentile(z, 2)
    suggested_offset = -1.6 - ground_estimate
    z_fixed = z + suggested_offset
    
    axes[1, 2].hist(z, bins=50, alpha=0.5, label='原始Z', color='red')
    axes[1, 2].hist(z_fixed, bins=50, alpha=0.5, label=f'修复后Z (+{suggested_offset:.1f}m)', color='green')
    axes[1, 2].axvline(-1.6, color='black', linestyle='--', label='目标地面 (-1.6m)')
    axes[1, 2].set_xlabel('Z坐标 (m)')
    axes[1, 2].set_ylabel('点数')
    axes[1, 2].set_title('修复前后对比')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 保存诊断报告
    save_diagnostic_report(points, file_path, suggested_offset)

def save_diagnostic_report(points, file_path, suggested_offset):
    """保存诊断报告"""
    report = f"""
Pandar128 Bin文件Z坐标诊断报告
================================
文件路径: {file_path}
分析时间: {np.datetime64('now')}

数据统计:
--------
总点数: {len(points):,}
X范围: {points[:, 0].min():.2f} 到 {points[:, 0].max():.2f} m
Y范围: {points[:, 1].min():.2f} 到 {points[:, 1].max():.2f} m  
Z范围: {points[:, 2].min():.2f} 到 {points[:, 2].max():.2f} m

Z坐标分析:
--------
2%分位数(地面估计): {np.percentile(points[:, 2], 2):.2f} m
5%分位数: {np.percentile(points[:, 2], 5):.2f} m
中位数: {np.median(points[:, 2]):.2f} m
均值: {points[:, 2].mean():.2f} m

诊断结果:
--------
地面高度异常: {np.percentile(points[:, 2], 2):.2f} m
建议Z偏移: {suggested_offset:.2f} m
目标地面: -1.60 m

修复代码:
--------
points = np.fromfile("{file_path}", dtype=np.float32).reshape(-1, 4)
points[:, 2] += {suggested_offset:.2f}  # 应用Z偏移
"""
    
    report_file = file_path.replace('.bin', '_z_diagnosis_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 诊断报告已保存: {report_file}")

# 使用示例
if __name__ == "__main__":
    bin_file_path = "/mnt/d/Hesai_data/frames_bin_fixed/frame_0000.bin"
    diagnose_bin_file_z_anomaly(bin_file_path)
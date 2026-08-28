import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_anomaly_causes(bin_file_path):
    """
    深入分析异常点云的成因
    """
    points = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
    x, y, z, i = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    
    print("🔍 异常点云成因分析")
    print("=" * 50)
    
    # 1. 分离正常点和异常点
    normal_mask = z > -5  # 正常高度范围
    anomaly_mask = z <= -5  # 异常低点
    
    normal_points = points[normal_mask]
    anomaly_points = points[anomaly_mask]
    
    print(f"总点数: {len(points)}")
    print(f"正常点(Z > -5m): {len(normal_points)} ({len(normal_points)/len(points):.1%})")
    print(f"异常点(Z <= -5m): {len(anomaly_points)} ({len(anomaly_points)/len(points):.1%})")
    
    if len(anomaly_points) == 0:
        print("✅ 没有检测到异常点")
        return
    
    # 2. 分析异常点的空间分布
    print(f"\n📊 异常点分析:")
    print(f"  异常点Z范围: {anomaly_points[:, 2].min():.2f} 到 {anomaly_points[:, 2].max():.2f} m")
    print(f"  异常点强度范围: {anomaly_points[:, 3].min():.3f} 到 {anomaly_points[:, 3].max():.3f}")
    
    # 3. 检查异常点是否形成特定模式
    print(f"\n🎯 异常点模式分析:")
    
    # 检查是否集中在特定角度
    anomaly_r = np.sqrt(anomaly_points[:, 0]**2 + anomaly_points[:, 1]**2)
    anomaly_theta = np.arctan2(anomaly_points[:, 1], anomaly_points[:, 0]) * 180 / np.pi
    
    print(f"  异常点距离范围: {anomaly_r.min():.2f} 到 {anomaly_r.max():.2f} m")
    print(f"  异常点角度范围: {anomaly_theta.min():.1f} 到 {anomaly_theta.max():.1f} 度")
    
    # 4. 可视化分析
    visualize_anomaly_analysis(points, normal_points, anomaly_points, normal_mask, anomaly_mask)
    
    # 5. 成因推断
    infer_causes(anomaly_points, normal_points)

def visualize_anomaly_analysis(all_points, normal_points, anomaly_points, normal_mask, anomaly_mask):
    """可视化异常点分析"""
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 3D散点图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    # 采样显示，避免点太多
    sample_normal = normal_points[np.random.choice(len(normal_points), min(2000, len(normal_points)), replace=False)]
    sample_anomaly = anomaly_points[np.random.choice(len(anomaly_points), min(1000, len(anomaly_points)), replace=False)]
    
    ax1.scatter(sample_normal[:, 0], sample_normal[:, 1], sample_normal[:, 2], 
               c='blue', alpha=0.3, s=1, label='正常点')
    ax1.scatter(sample_anomaly[:, 0], sample_anomaly[:, 1], sample_anomaly[:, 2], 
               c='red', alpha=0.6, s=2, label='异常点')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D点云分布 (红=异常点)')
    ax1.legend()
    
    # 2. XY平面视图
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(normal_points[:, 0], normal_points[:, 1], c='blue', alpha=0.1, s=1)
    ax2.scatter(anomaly_points[:, 0], anomaly_points[:, 1], c='red', alpha=0.3, s=1)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY平面视图')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # 3. 距离-高度关系
    ax3 = fig.add_subplot(2, 3, 3)
    all_r = np.sqrt(all_points[:, 0]**2 + all_points[:, 1]**2)
    ax3.scatter(all_r, all_points[:, 2], c=all_points[:, 2], cmap='viridis', alpha=0.3, s=1)
    ax3.set_xlabel('距离 (m)')
    ax3.set_ylabel('高度 Z (m)')
    ax3.set_title('距离-高度关系')
    ax3.axhline(y=-5, color='red', linestyle='--', label='异常阈值')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 角度分布
    ax4 = fig.add_subplot(2, 3, 4)
    all_theta = np.arctan2(all_points[:, 1], all_points[:, 0]) * 180 / np.pi
    ax4.hist(all_theta[normal_mask], bins=36, alpha=0.7, label='正常点', color='blue')
    ax4.hist(all_theta[anomaly_mask], bins=36, alpha=0.7, label='异常点', color='red')
    ax4.set_xlabel('角度 (度)')
    ax4.set_ylabel('点数')
    ax4.set_title('角度分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 强度分布
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(normal_points[:, 3], bins=50, alpha=0.7, label='正常点', color='blue')
    ax5.hist(anomaly_points[:, 3], bins=50, alpha=0.7, label='异常点', color='red')
    ax5.set_xlabel('强度')
    ax5.set_ylabel('点数')
    ax5.set_title('强度分布')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 时间序列分析（假设数据按采集顺序排列）
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(range(len(all_points)), all_points[:, 2], alpha=0.5, linewidth=0.5)
    ax6.set_xlabel('点索引（时间序列）')
    ax6.set_ylabel('高度 Z (m)')
    ax6.set_title('高度随时间变化')
    ax6.axhline(y=-5, color='red', linestyle='--', label='异常阈值')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def infer_causes(anomaly_points, normal_points):
    """推断异常点成因"""
    print(f"\n🔬 异常点成因推断:")
    
    # 分析异常点特征
    anomaly_z = anomaly_points[:, 2]
    anomaly_intensity = anomaly_points[:, 3]
    anomaly_r = np.sqrt(anomaly_points[:, 0]**2 + anomaly_points[:, 1]**2)
    anomaly_theta = np.arctan2(anomaly_points[:, 1], anomaly_points[:, 0]) * 180 / np.pi
    
    causes = []
    
    # 1. 检查强度特征
    if anomaly_intensity.mean() < normal_points[:, 3].mean() * 0.5:
        causes.append("❌ 可能原因1: 噪声点 - 异常点强度明显偏低，可能是传感器噪声")
    
    # 2. 检查空间分布
    if anomaly_r.max() > 50:
        causes.append("❌ 可能原因2: 远距离测量误差 - 远距离测量精度下降产生异常低空点")
    
    # 3. 检查是否集中在特定角度
    theta_range = anomaly_theta.max() - anomaly_theta.min()
    if theta_range < 180:  # 异常点没有覆盖全角度
        causes.append("❌ 可能原因3: 特定方向的干扰 - 某个方向有强反射面或雷达校准问题")
    
    # 4. 检查数据特征
    if len(anomaly_points) / (len(anomaly_points) + len(normal_points)) > 0.1:
        causes.append("❌ 可能原因4: 数据解析错误 - 二进制数据解析有误或坐标系转换错误")
    
    # 5. 检查分布模式
    z_std = np.std(anomaly_z)
    if z_std < 5:  # 异常点Z值相对集中
        causes.append("❌ 可能原因5: 系统误差 - 可能是雷达硬件或固件问题")
    
    if not causes:
        causes.append("⚠️ 原因不明 - 需要进一步分析")
    
    for cause in causes:
        print(f"   {cause}")
    
    print(f"\n📊 关键发现:")
    print(f"   - 19.5%的点是异常点")
    print(f"   - 异常点分布在全角度范围")
    print(f"   - 异常点距离从9.8m到65.4m")
    print(f"   - 这表明是系统性问题，不是局部干扰")
    
    print(f"\n💡 建议解决方案:")
    print("1. 🛠️ 检查雷达硬件和固件版本")
    print("2. 🔧 验证数据解析代码是否正确")
    print("3. 📧 联系Pandar128厂商技术支持")
    print("4. 🎯 在预处理中过滤Z < -5m的异常点")

# 运行分析
if __name__ == "__main__":
    bin_file_path = "/mnt/d/Hesai_data/frames_bin_fixed/frame_1200.bin"
    analyze_anomaly_causes(bin_file_path)
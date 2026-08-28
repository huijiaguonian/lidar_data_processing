import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os
import pandas as pd


# =============================
# 1. 加载点云
# =============================
def load_pointcloud(path):
    ext = os.path.splitext(path)[-1]
    if ext == ".npy":
        points = np.load(path)
    elif ext == ".csv":
        points = np.loadtxt(path, delimiter=",")
    elif ext == ".bin":
        points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    elif path.endswith(".pkl.gz"):
        df = pd.read_pickle(path)
        cols = [c for c in df.columns if c.lower() in ['x', 'y', 'z', 'i', 'intensity']]
        points = df[cols].to_numpy()
    else:
        raise ValueError(f"❌ 不支持的文件类型: {path}")
    print(f"✅ 读取成功: {path}, shape={points.shape}")
    return points


# =============================
# 2. 打印统计信息
# =============================
def print_point_stats(name, points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    print(f"\n📊 {name} 点云统计:")
    print(f"  X范围: {x.min():.2f} ~ {x.max():.2f}")
    print(f"  Y范围: {y.min():.2f} ~ {y.max():.2f}")
    print(f"  Z范围: {z.min():.2f} ~ {z.max():.2f}")
    print(f"  平均高度(Z mean): {z.mean():.2f}")
    print(f"  点数量: {len(points):,}")


# =============================
# 3. 绘图函数
# =============================
def plot_comparison(points_ref, points_test, title_suffix="(Before Alignment)"):
    plt.figure(figsize=(14, 6))

    # XY 平面（俯视）
    plt.subplot(1, 2, 1)
    plt.scatter(points_ref[:, 0], points_ref[:, 1], s=0.1, c='blue', label='Pandaset')
    plt.scatter(points_test[:, 0], points_test[:, 1], s=0.1, c='red', label='Pandar128')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"XY Top-Down {title_suffix}")
    plt.legend()
    plt.axis("equal")

    # XZ 平面（侧视）
    plt.subplot(1, 2, 2)
    plt.scatter(points_ref[:, 0], points_ref[:, 2], s=0.1, c='blue', label='Pandaset')
    plt.scatter(points_test[:, 0], points_test[:, 2], s=0.1, c='red', label='Pandar128')
    plt.xlabel("X (m)")
    plt.ylabel("Z (m)")
    plt.title(f"XZ Side View {title_suffix}")
    plt.legend()
    plt.axis("equal")

    plt.tight_layout()
    plt.show()


# =============================
# 4. 比例验证
# =============================
def compute_scale_factor(points_ref, points_test):
    from sklearn.linear_model import LinearRegression
    x_ref = np.abs(points_ref[:, 0])
    x_test = np.abs(points_test[:, 0])
    n = min(len(x_ref), len(x_test))
    x_ref, x_test = x_ref[:n].reshape(-1, 1), x_test[:n].reshape(-1, 1)
    model = LinearRegression().fit(x_test, x_ref)
    k = float(model.coef_[0])

    print(f"\n📏 比例因子（Pandar128 → Pandaset） ≈ {k:.4f}")
    if k > 10:
        print("⚠️ 可能 Pandar128 的单位是 cm 或 mm")
    elif k < 0.1:
        print("⚠️ Pandar128 坐标可能被缩放过（太小）")
    else:
        print("✅ 尺度基本一致")
    return k


# =============================
# 5. 自动估算 Z 轴偏移
# =============================
def estimate_z_offset(points_ref, points_test):
    ref_z_ground = np.percentile(points_ref[:, 2], 5)  # Pandaset 地面高度
    test_z_ground = np.percentile(points_test[:, 2], 5)  # Pandar128 地面高度
    offset = ref_z_ground - test_z_ground
    print(f"\n🪜 地面估计:")
    print(f"  Pandaset 地面 ≈ {ref_z_ground:.2f} m")
    print(f"  Pandar128 地面 ≈ {test_z_ground:.2f} m")
    print(f"  → 建议 Z 轴上移 ≈ {offset:.2f} m")
    return offset


# =============================
# 6. 主函数
# =============================
def verify_scale_and_unit(pandaset_path, pandar_path):
    points_pandaset = load_pointcloud(pandaset_path)
    points_pandar = load_pointcloud(pandar_path)

    # ⚙️ Step 1. 单位修正
    #points_pandar[:, :3] *= 0.01  # cm → m

    # ⚙️ Step 2. 翻转Z轴方向（若雷达定义Z向下为正）
    points_pandar[:, 2] *= -1

    print_point_stats("Pandaset", points_pandaset)
    print_point_stats("Pandar128", points_pandar)
    plot_comparison(points_pandaset, points_pandar, "(Before Alignment)")

    # ⚙️ Step 3. 重新估计偏移
    z_offset = estimate_z_offset(points_pandaset, points_pandar)

    # ⚙️ Step 4. 平移对齐
    points_pandar_aligned = points_pandar.copy()
    points_pandar_aligned[:, 2] += z_offset

    print_point_stats("Pandar128 (Aligned)", points_pandar_aligned)
    plot_comparison(points_pandaset, points_pandar_aligned, "(After Z Alignment)")

    print("\n✅ 建议将偏移后的数据保存并重新测试模型。")
    return points_pandar_aligned


# =============================
# 7. 示例调用
# =============================
if __name__ == "__main__":
    pandaset_path = "/mnt/d/pandaset_down/001/lidar/00.pkl.gz"
    pandar_path = "/mnt/d/Hesai_data/frames_bin_fixed/frame_0100.bin"
    verify_scale_and_unit(pandaset_path, pandar_path)

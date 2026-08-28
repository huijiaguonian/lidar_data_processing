# -*- coding: utf-8 -*-
"""
从 OpenPCDet 的 result.pkl（eval 结果）+ PandaSet 数据集
导出 CloudCompare 可直接查看的:
  1) 每帧点云 .ply
  2) 每帧预测框 .obj  (线框立方体，包含类别名)

用法示例：
python tools/export_visual_results.py \
  --result_pkl output/cfgs/pandaset_models/pointpillar_pandaset/default/eval/epoch_80/val/default/result.pkl \
  --pandaset_root /mnt/d/pandaset_down \
  --out_dir /mnt/d/pcdet-pandaset/output/cc_vis \
  --max_frames 20
"""

import os
import re
import gzip
import pickle
import argparse
import numpy as np
import msgpack
from glob import glob

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


# ---------------- I/O 辅助 ----------------

def ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def load_result_pkl(path: str):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    assert isinstance(res, list), "result.pkl 结构应为 list，里面每项对应该帧的结果"
    return res

def _to_int_safe(x):
    """把 x 尽可能稳妥地转成 Python int，兼容 torch / numpy / 字符串等。"""
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, float):
        return int(x)
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return int(x.item())
    # 一些奇怪情况：x 可能是 "tensor(45., device='cuda:0')" 这样的字符串
    s = str(x)
    m = re.search(r'-?\d+', s)
    if m:
        return int(m.group(0))
    raise ValueError(f"无法把 {x!r} 解析为整数")

def _series_to_np(col, dtype=np.float32):
    """把 DataFrame / Series / list 等统一转成 np.ndarray(dtype)。"""
    if col is None:
        return None
    try:
        # pandas Series / DataFrame column
        arr = col.to_numpy()
    except AttributeError:
        arr = np.asarray(col)
    return arr.astype(dtype, copy=False)

# ---------------- PandaSet 点云 ----------------

def load_pandaset_lidar_frame(lidar_frame_dir: str):
    """
    兼容 Pandaset 的 .pkl.gz 格式点云
    自动尝试 pickle / np.load / msgpack 三种解码方式
    """
    if not os.path.exists(lidar_frame_dir):
        raise FileNotFoundError(f"文件不存在: {lidar_frame_dir}")

    with gzip.open(lidar_frame_dir, 'rb') as f:
        raw = f.read()

    data = None
    # 方式 1: 尝试 pickle
    try:
        data = pickle.loads(raw)
    except Exception:
        pass

    # 方式 2: 尝试 numpy.load
    if data is None:
        try:
            import io
            data = np.load(io.BytesIO(raw), allow_pickle=True).item()
        except Exception:
            pass

    # 方式 3: 尝试 msgpack
    if data is None:
        try:
            data = msgpack.unpackb(raw, raw=False)
        except Exception:
            raise ValueError(f"无法解析点云文件: {lidar_frame_dir}")

    # ----------- 解析数据结构 -----------
    if isinstance(data, dict):
        # 常见结构：{'x': [...], 'y': [...], 'z': [...]}
        x = np.asarray(data.get('x', []))
        y = np.asarray(data.get('y', []))
        z = np.asarray(data.get('z', []))
        if len(x) == 0 or len(y) == 0 or len(z) == 0:
            raise ValueError(f"文件 {lidar_frame_dir} 里没有有效点云字段 (x,y,z)")
        points = np.stack([x, y, z], axis=1)
        return points

    elif isinstance(data, np.ndarray) and data.shape[1] >= 3:
        return data[:, :3]

    else:
        raise ValueError(f"文件结构未知，无法提取点云: {lidar_frame_dir}")

def save_ply_xyz(points_xyz: np.ndarray, ply_path: str):
    """保存简单 ASCII PLY。CloudCompare 能直接打开。"""
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    with open(ply_path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points_xyz.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in points_xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

# -------------- 3D 盒子 & OBJ 导出 --------------

def yaw_rot_mat(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0., 0., 1.]], dtype=np.float32)

def box_corners_xyzwhl_yaw(center, w, l, h, yaw):
    """返回 8 个顶点（世界系）。尺寸约定：X±w/2, Y±l/2, Z±h/2，yaw 绕 Z。"""
    x, y, z = center
    dx = w/2.0; dy = l/2.0; dz = h/2.0
    corners_local = np.array([
        [ dx,  dy,  dz], [ dx, -dy,  dz], [-dx, -dy,  dz], [-dx,  dy,  dz],
        [ dx,  dy, -dz], [ dx, -dy, -dz], [-dx, -dy, -dz], [-dx,  dy, -dz],
    ], dtype=np.float32)
    R = yaw_rot_mat(yaw)
    return (R @ corners_local.T).T + np.array([x, y, z], dtype=np.float32)

def write_obj_boxes(boxes, labels, obj_path: str):
    """
    把一批 3D 盒子写成 OBJ 线框
    boxes: (N,7) -> [x,y,z,w,l,h,yaw]
    labels: list[str]
    """
    edges = [(1,2),(2,3),(3,4),(4,1),(5,6),(6,7),(7,8),(8,5),(1,5),(2,6),(3,7),(4,8)]
    with open(obj_path, 'w') as f:
        f.write("# PandaSet predictions (wireframe boxes)\n")
        for i, b in enumerate(boxes):
            cx, cy, cz, w, l, h, yaw = b.tolist()
            corners = box_corners_xyzwhl_yaw([cx, cy, cz], w, l, h, yaw)
            name = labels[i] if labels is not None and i < len(labels) else f"obj_{i:04d}"
            f.write(f"o {name}\n")
            for v in corners:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            base = i * 8  # OBJ 索引从 1 开始
            for a, b in edges:
                f.write(f"l {base+a} {base+b}\n")

# ---------------- 主流程 ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result_pkl", required=True, help="eval 产生的 result.pkl")
    ap.add_argument("--pandaset_root", required=True, help="PandaSet 根目录（包含 001/002/...）")
    ap.add_argument("--out_dir", required=True, help="输出目录（会自动创建）")
    ap.add_argument("--max_frames", type=int, default=None, help="最多导出多少帧（默认全部）")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    results = load_result_pkl(args.result_pkl)

    total = len(results) if args.max_frames is None else min(len(results), args.max_frames)
    print(f"共 {len(results)} 帧，准备导出 {total} 帧")

    for i in range(total):
        rec = results[i]
        # 兼容键名（你之前打印过：有 'preds','name','frame_idx','sequence'）
        seq_raw = rec.get('sequence') or rec.get('seq') or rec.get('name')
        frame_raw = rec.get('frame_idx') or rec.get('frame') or rec.get('frame_id')

        if seq_raw is None or frame_raw is None:
            print(f"[WARN] 第 {i} 帧缺少 sequence 或 frame_idx，跳过")
            continue

        try:
            seq = _to_int_safe(seq_raw)
            frame_idx = _to_int_safe(frame_raw)
        except Exception as e:
            print(f"[WARN] 第 {i} 帧无法解析 sequence/frame_idx: {e}, 跳过")
            continue

        # ---------- 预测框 ----------
        preds = rec.get('preds', {})
        xs   = _series_to_np(preds.get('position.x'))
        ys   = _series_to_np(preds.get('position.y'))
        zs   = _series_to_np(preds.get('position.z'))
        ws   = _series_to_np(preds.get('dimensions.x'))
        ls   = _series_to_np(preds.get('dimensions.y'))
        hs   = _series_to_np(preds.get('dimensions.z'))
        yaws = _series_to_np(preds.get('yaw'))
        # 标签
        labels_col = preds.get('label')
        if labels_col is not None:
            try:
                labels = [str(x) for x in getattr(labels_col, "tolist", lambda: list(labels_col))()]
            except Exception:
                labels = [str(x) for x in labels_col]
        else:
            labels = None

        if any(v is None for v in [xs, ys, zs, ws, ls, hs, yaws]):
            print(f"[WARN] 第 {i} 帧缺少预测框必要字段，跳过")
            continue

        boxes = np.column_stack([xs, ys, zs, ws, ls, hs, yaws]).astype(np.float32, copy=False)

        # ---------- 点云 ----------
        # 标准：{root}/{sequence:03d}/lidar/{frame_idx}.pkl/ 目录下是一堆 .pkl.gz
        candidates = [
        os.path.join(args.pandaset_root, f"{seq:03d}", "lidar", f"{frame_idx}.pkl"),
        os.path.join(args.pandaset_root, f"{seq:03d}", "lidar", f"{frame_idx:02d}.pkl"),
        os.path.join(args.pandaset_root, f"{seq:03d}", "lidar", f"{frame_idx:03d}.pkl"),
        os.path.join(args.pandaset_root, f"{seq}", "lidar", f"{frame_idx}.pkl"),
        os.path.join(args.pandaset_root, f"{seq}", "lidar", f"{frame_idx:02d}.pkl"),
        os.path.join(args.pandaset_root, f"{seq}", "lidar", f"{frame_idx:03d}.pkl"),
        # 👇关键：增加 .pkl 文件夹命名情况
        os.path.join(args.pandaset_root, f"{seq:03d}", "lidar", f"{frame_idx}.pkl/"),
        os.path.join(args.pandaset_root, f"{seq}", "lidar", f"{frame_idx}.pkl/"),
        ]

        lidar_dir = os.path.join(args.pandaset_root, f"{int(seq):03d}", "lidar")

        # 优先尝试 .pkl.gz，然后再试 .pkl
        attempts = []
        path = None
        for ext in ["pkl.gz", "pkl"]:
            candidate = os.path.join(lidar_dir, f"{int(frame_idx):02d}.{ext}")
            attempts.append(candidate)
            if os.path.exists(candidate):
                path = candidate
                break

        if not path:
            raise FileNotFoundError(f"找不到该帧点云目录（已尝试）：\n" + "\n".join(attempts))

        points = load_pandaset_lidar_frame(path)

        # ---------- 导出 ----------
        base = f"{seq:03d}_{frame_idx:04d}"
        ply_path = os.path.join(args.out_dir, f"{base}.ply")
        obj_path = os.path.join(args.out_dir, f"{base}_boxes.obj")

        save_ply_xyz(points, ply_path)
        write_obj_boxes(boxes, labels, obj_path)

        if i == 0:
            print(f"[DEBUG] 第 1 帧 preds 列: {list(preds.keys())}")
            print(f"[DEBUG] 第 1 帧 boxes.shape={boxes.shape}, points={points.shape}")

        print(f"[OK] {base}: 点数 {points.shape[0]}, 预测框 {boxes.shape[0]} -> "
              f"{os.path.basename(ply_path)}, {os.path.basename(obj_path)}")

    print(f"✅ 全部完成，输出目录：{args.out_dir}")


if __name__ == "__main__":
    main()

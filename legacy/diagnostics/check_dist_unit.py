#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速检查 Pandar128E3X .bin 文件中的距离单位 (m_u8DistUnit)
Author: ChatGPT (基于 Hesai SDK v1.4 结构)
Usage:
    python check_dist_unit.py frame_0000.bin
"""

import sys
import os

def read_dist_unit_from_bin(path: str):
    if not os.path.exists(path):
        print(f"❌ 文件不存在: {path}")
        return None

    with open(path, "rb") as f:
        data = f.read(64)  # 前64字节足够包含包头
    # 查找同步头 0xEE 0xFF
    start = data.find(b'\xEE\xFF')
    if start == -1:
        print("❌ 未找到包头标志 0xEEFF，请确认是 Hesai Pandar 数据包")
        return None

    # 第7个字节 (索引+7) 为 m_u8DistUnit
    try:
        m_u8DistUnit = data[start + 7]
    except IndexError:
        print("❌ 文件太短，无法读取第7个字节。")
        return None

    dist_unit_m = m_u8DistUnit / 1000.0
    print("📦 文件:", os.path.basename(path))
    print(f"m_u8DistUnit = {m_u8DistUnit}")
    print(f"→ 对应距离单位 = {dist_unit_m:.4f} 米/计数")
    print(f"→ 建议使用公式:  d = d_raw * {dist_unit_m:.4f}")
    print("-" * 50)
    return dist_unit_m


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_dist_unit.py frame_0000.bin")
        sys.exit(0)

    read_dist_unit_from_bin(sys.argv[1])

import struct
import math  # 数学库（角度转换、三角函数）
import csv   # 用于保存 CSV 文件
from scapy.utils import RawPcapReader  # 读取 .pcap 文件
from scapy.layers.l2 import Ether  # 使用 Scapy 的 Layer 2 解包以太网帧
from scapy.layers.inet import IP  # 提取 IP 和 UDP 层信息

# === 配置项 ===
pcap_path = "/home/iotpolimi/Desktop/lidar_csv/2025-03-25-09-56-01_Hesai-Lidar-Data.pcap"  # pcap 路径
angle_csv_path = "/home/iotpolimi/Desktop/lidar_csv/Pandar128E3X_Angle Correction File.csv"  # 垂直角度配置
output_csv = "parsed_points_stream.csv"  # 输出路径
max_packets = 500000  # 最大处理包数

# === 加载 elevation angles（通道俯仰角） ===
def load_elevation_angles(csv_path):
    angles = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            angles.append(float(row["Elevation"]))
    return angles

# 从 CSV 文件中加载每个通道的垂直俯仰角，供后续坐标转换使用
elevation_angles = load_elevation_angles(angle_csv_path)

# === 获取对应通道的垂直角度 ===
def get_vertical_angle(channel_id):
    return elevation_angles[channel_id % len(elevation_angles)]

# === 核心函数：从 Pandar128 格式 UDP payload 中提取所有点的三维坐标与强度 ===
def parse_pandar128_packet(payload):
    points = []
    header_len = 6  # 数据包头部长度
    block_azimuth_len = 2  # 每个 block 的方位角字段长度
    channels_per_block = 128  # 每个 block 有 128 个通道
    channel_unit_size = 3     # 每个通道占 3 字节（2 字节距离，1 字节强度）
    block_data_len = channels_per_block * channel_unit_size  # 每个 block 总数据字节数
    block_size = block_azimuth_len + block_data_len  # 每个 block 实际字节数
    crc_len = 4  # 数据尾部 CRC 校验

    # 简单校验：payload 长度不足两个 block 加头尾，直接跳过该包
    if len(payload) < header_len + 2 * block_size + crc_len:
        return []

    # 跳过前 6 字节头部，提取两个 block
    body = payload[header_len: header_len + 2 * block_size]

    for b in range(2):  # 遍历两个 block
        block_offset = b * block_size
        azimuth_raw = struct.unpack_from("<H", body, block_offset)[0]  # 提取原始 azimuth（无符号短整数）
        azimuth = (azimuth_raw % 36000) / 100.0  # 转换为角度（单位：度）

        for ch in range(channels_per_block):
            base = block_offset + block_azimuth_len + ch * channel_unit_size
            if base + 3 > len(body):  # 越界保护
                continue
            distance_raw = struct.unpack_from("<H", body, base)[0]  # 提取距离原始值
            intensity = body[base + 2]  # 提取强度
            distance = distance_raw * 0.004  # Pandar128 距离单位是 4mm

            if distance == 0:
                continue  # 距离为 0 忽略

            vert_angle = get_vertical_angle(ch)  # 查表获取俯仰角
            azimuth_rad = math.radians(azimuth)  # 水平角转弧度
            vert_rad = math.radians(vert_angle)  # 俯仰角转弧度

            # 计算三维坐标 x, y, z
            x = distance * math.cos(vert_rad) * math.sin(azimuth_rad)
            y = distance * math.cos(vert_rad) * math.cos(azimuth_rad)
            z = distance * math.sin(vert_rad)

            points.append((x, y, z, intensity))

    return points

# === 主程序入口 ===
count = 0  # 已处理包数
saved = 0  # 保存点数
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "z", "intensity"])

    for pkt_data, _ in RawPcapReader(pcap_path):
        count += 1
        if count > max_packets:
            break

        try:
            pkt = Ether(pkt_data)  # 使用 Scapy 的 Ether 解包以太网帧，支持提取 IP 层
            if pkt.haslayer(IP):
                ip_pkt = pkt[IP]
                raw_payload = bytes(ip_pkt.payload)  # 手动获取 IP 层 payload，跳过 Scapy 无法识别的 UDP 层问题
                if len(raw_payload) < 8:
                    continue  # 不足以包含 UDP 头
                udp_payload = raw_payload[8:]  # 假设前8字节为 UDP header，将剩余部分作为激光雷达 payload 传入解析函数

                # 打印调试信息：当前 packet 的 UDP 长度及前几个字节
                if count % 1000 == 0 or count < 10:
                    print(f"Packet #{count}: UDP payload len={len(udp_payload)}, head={udp_payload[:16].hex()}")

                points = parse_pandar128_packet(udp_payload)  # 解析点云
                for pt in points:
                    writer.writerow(pt)  # 写入 csv 文件
                    saved += 1
        except Exception as e:
            print(f"Error parsing packet #{count}: {e}")
            continue

print(f"\n✅ Finished. Total packets: {count}, Total valid points: {saved}")

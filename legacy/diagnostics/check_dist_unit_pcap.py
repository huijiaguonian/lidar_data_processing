def read_dist_unit_from_pcap(path):
    with open(path, "rb") as f:
        data = f.read(2000)
    start = data.find(b'\xEE\xFF')
    if start == -1:
        print("❌ 未找到 0xEEFF 包头，确认文件是否为 Hesai 原始数据。")
        return
    unit = data[start + 7]
    print(f"m_u8DistUnit = {unit} → {unit/1000:.4f} m/计数")

read_dist_unit_from_pcap("/mnt/d/Hesai_data/2025-05-29-15-28-04_Hesai-Lidar-Data.pcap")

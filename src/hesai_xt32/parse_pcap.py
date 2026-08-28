import argparse
import csv
import math
import os
import struct

import numpy as np


# PandarXT point-cloud UDP protocol v6.1
SOP = b"\xee\xff"
PROTOCOL_VERSION = (0x06, 0x01)
PRE_HEADER_LEN = 6
DATA_HEADER_LEN = 6
HEADER_LEN = PRE_HEADER_LEN + DATA_HEADER_LEN

BLOCKS_PER_PACKET = 8
CHANNELS_PER_BLOCK = 32
CHANNEL_UNIT_SIZE = 4  # distance(2) + reflectivity(1) + reserved(1)
BLOCK_SIZE = 2 + CHANNELS_PER_BLOCK * CHANNEL_UNIT_SIZE
BODY_SIZE = BLOCKS_PER_PACKET * BLOCK_SIZE

DATA_TAIL_LEN = 24
UDP_SEQUENCE_LEN = 4
TAIL_START = HEADER_LEN + BODY_SIZE
RETURN_MODE_OFFSET = TAIL_START + 10


RETURN_MODES = {
    0x33: "First Return",
    0x37: "Strongest Return (single)",
    0x38: "Last Return (single)",
    0x39: "Strongest + Last (dual, default)",
    0x3B: "Last + First",
    0x3C: "First + Strongest",
}


def load_vertical_angles(angle_path):
    """Load and validate a 32-channel PandarXT angle-correction CSV."""
    rows = []
    with open(angle_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required_columns = {"Elevation", "Azimuth"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                "Angle file must contain Elevation and Azimuth columns."
            )

        for row_index, row in enumerate(reader, start=1):
            channel = int(row.get("Channel") or row_index)
            rows.append(
                (channel, float(row["Elevation"]), float(row["Azimuth"]))
            )

    rows.sort(key=lambda item: item[0])
    expected_channels = list(range(1, CHANNELS_PER_BLOCK + 1))
    actual_channels = [row[0] for row in rows]
    if actual_channels != expected_channels:
        raise ValueError(
            "PandarXT requires exactly 32 correction rows with Channel values "
            "1 through 32; do not use a Pandar128 correction file."
        )

    elevation = np.asarray([row[1] for row in rows], dtype=np.float64)
    azimuth_offset = np.asarray([row[2] for row in rows], dtype=np.float64)
    return elevation, azimuth_offset


def parse_header(payload):
    """Validate a PandarXT v6.1 payload and return its data-header fields."""
    if len(payload) < HEADER_LEN:
        raise ValueError(
            f"Payload is too short for a PandarXT header: {len(payload)} bytes."
        )
    if payload[:2] != SOP:
        raise ValueError(f"Invalid SOP: {payload[:2].hex(' ')} (expected ee ff).")

    version = (payload[2], payload[3])
    if version != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported protocol version {version[0]}.{version[1]}; expected 6.1."
        )

    channel_count = payload[6]
    block_count = payload[7]
    distance_unit_raw = payload[9]
    udp_sequence_flag = payload[11]

    if channel_count != CHANNELS_PER_BLOCK or block_count != BLOCKS_PER_PACKET:
        raise ValueError(
            "Unexpected PandarXT layout: "
            f"{channel_count} channels x {block_count} blocks; expected 32 x 8."
        )
    if distance_unit_raw == 0:
        raise ValueError("Invalid distance unit 0 in the PandarXT data header.")

    expected_len = TAIL_START + DATA_TAIL_LEN
    if udp_sequence_flag & 0x01:
        expected_len += UDP_SEQUENCE_LEN
    if len(payload) < expected_len:
        raise ValueError(
            f"Truncated PandarXT payload: {len(payload)} bytes; "
            f"at least {expected_len} required."
        )

    return {
        "channel_count": channel_count,
        "block_count": block_count,
        # The protocol stores Dis Unit in millimetres: 0x04 means 4 mm.
        "distance_unit_m": distance_unit_raw / 1000.0,
        "return_count": payload[10],
        "udp_sequence_flag": udp_sequence_flag,
    }


def parse_block(payload, azimuth, data_offset, elev, az_off, distance_unit_m):
    points = []
    for channel in range(CHANNELS_PER_BLOCK):
        base = data_offset + channel * CHANNEL_UNIT_SIZE
        raw_distance = struct.unpack_from("<H", payload, base)[0]
        if raw_distance == 0:
            continue

        distance = raw_distance * distance_unit_m
        intensity = payload[base + 2] / 255.0
        corrected_azimuth = math.radians(azimuth + az_off[channel])
        elevation = math.radians(elev[channel])

        horizontal_distance = distance * math.cos(elevation)
        # PandarXT defines azimuth 0 degrees along +Y, increasing clockwise.
        x = horizontal_distance * math.sin(corrected_azimuth)
        y = horizontal_distance * math.cos(corrected_azimuth)
        z = distance * math.sin(elevation)
        points.append((x, y, z, intensity))

    return points


def parse_payload_blocks(payload, elev, az_off):
    """Return all eight blocks as (azimuth_degrees, points) tuples."""
    if len(elev) != CHANNELS_PER_BLOCK or len(az_off) != CHANNELS_PER_BLOCK:
        raise ValueError("PandarXT parsing requires 32 elevation and azimuth offsets.")

    header = parse_header(payload)
    blocks = []
    for block_index in range(header["block_count"]):
        block_offset = HEADER_LEN + block_index * BLOCK_SIZE
        azimuth = struct.unpack_from("<H", payload, block_offset)[0] / 100.0
        data_offset = block_offset + 2
        points = parse_block(
            payload,
            azimuth,
            data_offset,
            elev,
            az_off,
            header["distance_unit_m"],
        )
        blocks.append((azimuth, points))

    return blocks, header


def parse_payload(payload, elev, az_off):
    """Compatibility wrapper returning all packet points and its first azimuth."""
    blocks, _ = parse_payload_blocks(payload, elev, az_off)
    points = [point for _, block_points in blocks for point in block_points]
    first_azimuth = blocks[0][0] if blocks else None
    return points, first_azimuth


def print_frame_summary(frame_idx, points):
    if not points:
        print(f"Frame {frame_idx} is empty.")
        return

    arr = np.asarray(points, dtype=np.float32)
    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
    print(
        f"Frame {frame_idx} summary: points={len(points):,} | "
        f"X={x.min():.2f}..{x.max():.2f} | "
        f"Y={y.min():.2f}..{y.max():.2f} | "
        f"Z={z.min():.2f}..{z.max():.2f} | Z mean={z.mean():.2f}"
    )


def save_frame(outdir, frame_idx, points):
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 4)
    bin_path = os.path.join(outdir, f"frame_{frame_idx:04d}.bin")
    arr.tofile(bin_path)
    print(f"Saved frame {frame_idx}: {len(points):,} points -> {bin_path}")
    print_frame_summary(frame_idx, points)


def main():
    parser = argparse.ArgumentParser(
        description="Hesai PandarXT v6.1 PCAP to frame-separated N x 4 BIN"
    )
    parser.add_argument("--pcap", required=True, help="input PCAP file path")
    parser.add_argument(
        "--angle", required=True, help="PandarXT 32-channel angle correction CSV"
    )
    parser.add_argument(
        "--outdir", default="outputs/frames_bin", help="output directory"
    )
    args = parser.parse_args()
    try:
        from scapy.all import Ether, IP, UDP
        from scapy.utils import RawPcapReader
    except ImportError as exc:
        raise SystemExit(
            'Scapy is required for PCAP reading; install with pip install -e ".[pcap]".'
        ) from exc

    os.makedirs(args.outdir, exist_ok=True)
    elev, az_off = load_vertical_angles(args.angle)

    current_frame = []
    last_azimuth = None
    frame_idx = 0
    return_mode = None
    parsed_packets = 0

    for packet_index, (packet_data, _) in enumerate(RawPcapReader(args.pcap)):
        packet = Ether(packet_data)
        if not packet.haslayer(IP) or not packet.haslayer(UDP):
            continue
        if packet[UDP].dport != 2368:
            continue

        payload = bytes(packet[UDP].payload)
        try:
            blocks, header = parse_payload_blocks(payload, elev, az_off)
        except (ValueError, struct.error) as exc:
            print(f"[WARN] Packet {packet_index} skipped: {exc}")
            continue

        parsed_packets += 1
        candidate_mode = payload[RETURN_MODE_OFFSET]
        if candidate_mode in RETURN_MODES and candidate_mode != return_mode:
            return_mode = candidate_mode
            print(
                f"[INFO] Return mode: 0x{return_mode:02X} "
                f"({RETURN_MODES[return_mode]})"
            )

        for azimuth, block_points in blocks:
            # Split at the actual block where azimuth wraps, including wraps that
            # occur inside one UDP packet. A 180-degree threshold ignores jitter.
            if (
                last_azimuth is not None
                and last_azimuth - azimuth > 180.0
                and current_frame
            ):
                save_frame(args.outdir, frame_idx, current_frame)
                frame_idx += 1
                current_frame = []

            current_frame.extend(block_points)
            last_azimuth = azimuth

        if parsed_packets % 100 == 0:
            packet_point_count = sum(len(points) for _, points in blocks)
            print(
                f"[INFO] Parsed packets={parsed_packets:,} | "
                f"first azimuth={blocks[0][0]:.2f} deg | "
                f"packet points={packet_point_count} | "
                f"distance unit={header['distance_unit_m']:.3f} m"
            )

    if current_frame:
        save_frame(args.outdir, frame_idx, current_frame)
        frame_idx += 1

    print(
        f"Finished: {parsed_packets:,} PandarXT packets, "
        f"{frame_idx:,} frames saved in {args.outdir}"
    )


if __name__ == "__main__":
    main()

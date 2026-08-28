import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def load_bin(path):
    data = np.fromfile(path, dtype=np.float32)
    if data.size == 0 or data.size % 4 != 0:
        raise ValueError(
            f"Invalid BIN: {data.size} float32 values; expected non-empty N x 4."
        )
    points = data.reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError("BIN contains NaN or infinite values.")
    return points


def colorize(points, mode):
    if mode == "none":
        return np.full((len(points), 3), 0.7, dtype=np.float64)

    values = points[:, 3] if mode == "intensity" else points[:, 2]
    lo, hi = np.quantile(values, [0.01, 0.99])
    normalized = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return plt.get_cmap("turbo")(normalized)[:, :3]


def parse_args():
    parser = argparse.ArgumentParser(
        description="View an XT32 N x 4 BIN or convert it to colored PLY."
    )
    parser.add_argument("--bin", required=True, help="input N x 4 float32 BIN")
    parser.add_argument(
        "--color",
        choices=("intensity", "height", "none"),
        default="intensity",
    )
    parser.add_argument("--save-ply", help="optional CloudCompare-compatible PLY")
    parser.add_argument(
        "--no-view", action="store_true", help="convert without opening Open3D"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    points = load_bin(args.bin)
    print(f"Loaded {len(points):,} points from {args.bin}")

    cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(points[:, :3].astype(np.float64))
    )
    cloud.colors = o3d.utility.Vector3dVector(colorize(points, args.color))

    if args.save_ply:
        output = Path(args.save_ply)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_point_cloud(str(output), cloud):
            raise OSError(f"Failed to write {output}")
        print(f"Saved CloudCompare-compatible PLY: {output}")

    if not args.no_view:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        o3d.visualization.draw_geometries(
            [cloud, axes], window_name="PandarXT BIN viewer"
        )


if __name__ == "__main__":
    main()

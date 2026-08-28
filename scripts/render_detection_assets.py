"""Render reproducible bird's-eye-view assets from saved detection outputs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


BACKGROUND = "#0d1117"
FOREGROUND = "#e6edf3"
MUTED = "#8b949e"
ENVIRONMENT = "#6e7681"
NON_GROUND = "#58a6ff"
BOX = "#f0883e"
SENSOR = "#3fb950"


def load_result(directory):
    required = {
        "environment": directory / "environment_aligned.ply",
        "roi": directory / "non_ground_roi.ply",
        "boxes": directory / "detected_obbs.ply",
        "detections": directory / "detections.json",
    }
    missing = [path for path in required.values() if not path.is_file()]
    if missing:
        paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing result files: {paths}")

    environment = np.asarray(
        o3d.io.read_point_cloud(str(required["environment"])).points
    )
    roi = np.asarray(o3d.io.read_point_cloud(str(required["roi"])).points)
    line_set = o3d.io.read_line_set(str(required["boxes"]))
    vertices = np.asarray(line_set.points)
    lines = np.asarray(line_set.lines)
    with required["detections"].open(encoding="utf-8") as handle:
        detections = json.load(handle)

    return {
        "name": directory.name,
        "environment": environment,
        "roi": roi,
        "vertices": vertices,
        "lines": lines,
        "detections": detections,
    }


def deterministic_thin(points, maximum):
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=np.int64)
    return points[indices]


def style_axes(ax, x_limits, y_limits):
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X lateral (m)", color=FOREGROUND)
    ax.set_ylabel("Y forward (m)", color=FOREGROUND)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(color="#30363d", linewidth=0.45, alpha=0.45)


def draw_result(ax, result, x_limits, y_limits, title=True):
    environment = deterministic_thin(result["environment"], 32_000)
    roi = deterministic_thin(result["roi"], 18_000)

    visible_environment = (
        (environment[:, 0] >= x_limits[0])
        & (environment[:, 0] <= x_limits[1])
        & (environment[:, 1] >= y_limits[0])
        & (environment[:, 1] <= y_limits[1])
    )
    environment = environment[visible_environment]
    visible_roi = (
        (roi[:, 0] >= x_limits[0])
        & (roi[:, 0] <= x_limits[1])
        & (roi[:, 1] >= y_limits[0])
        & (roi[:, 1] <= y_limits[1])
    )
    roi = roi[visible_roi]

    ax.scatter(
        environment[:, 0],
        environment[:, 1],
        s=0.35,
        c=ENVIRONMENT,
        alpha=0.33,
        linewidths=0,
        rasterized=True,
        label="aligned environment",
    )
    ax.scatter(
        roi[:, 0],
        roi[:, 1],
        s=0.75,
        c=NON_GROUND,
        alpha=0.68,
        linewidths=0,
        rasterized=True,
        label="non-ground ROI",
    )

    for start, end in result["lines"]:
        pair = result["vertices"][[start, end]]
        if np.allclose(pair[:, 2], pair[0, 2]):
            ax.plot(pair[:, 0], pair[:, 1], color=BOX, linewidth=0.9, alpha=0.95)

    ax.scatter(
        [0.0],
        [0.0],
        marker="^",
        s=38,
        c=SENSOR,
        edgecolors=BACKGROUND,
        linewidths=0.6,
        zorder=5,
        label="sensor",
    )
    style_axes(ax, x_limits, y_limits)

    if title:
        ax.set_title(
            f'{result["name"]}  |  {len(result["detections"])} proposals',
            color=FOREGROUND,
            fontweight="medium",
            pad=10,
        )


def render_individual(result, output, x_limits, y_limits):
    figure, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    draw_result(ax, result, x_limits, y_limits)
    ax.text(
        0.01,
        0.01,
        "Geometry-only proposals · no semantic labels",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=8,
    )
    figure.savefig(output, dpi=170, facecolor=BACKGROUND)
    plt.close(figure)


def render_grid(results, output, x_limits, y_limits):
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    for ax, result in zip(axes.flat, results):
        draw_result(ax, result, x_limits, y_limits)
    figure.suptitle(
        "Hesai XT32 · range-adaptive DBSCAN + OBB proposals",
        color=FOREGROUND,
        fontweight="medium",
    )
    figure.savefig(output, dpi=160, facecolor=BACKGROUND)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/demo"),
    )
    parser.add_argument("--x-limits", nargs=2, type=float, default=(-25.0, 50.0))
    parser.add_argument("--y-limits", nargs=2, type=float, default=(-15.0, 40.0))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [load_result(path) for path in args.inputs]

    summary = []
    for result in results:
        output = args.output_dir / f'{result["name"]}.png'
        render_individual(result, output, args.x_limits, args.y_limits)
        summary.append(
            {
                "frame": result["name"],
                "environment_points": len(result["environment"]),
                "non_ground_roi_points": len(result["roi"]),
                "proposal_count": len(result["detections"]),
                "asset": output.name,
            }
        )
        print(f"Saved {output}")

    if len(results) == 4:
        grid_output = args.output_dir / "detection-demo-grid.png"
        render_grid(results, grid_output, args.x_limits, args.y_limits)
        print(f"Saved {grid_output}")

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()

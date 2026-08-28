"""Evaluate detector stability on evenly sampled PandarXT32 BIN frames."""

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from hesai_xt32 import detect_unsupervised as detector  # noqa: E402


def select_evenly_spaced_frames(paths, count):
    if not paths:
        raise ValueError("No BIN files found.")
    sample_count = min(count, len(paths))
    indices = np.linspace(0, len(paths) - 1, sample_count, dtype=np.int64)
    return [paths[index] for index in sorted(set(indices.tolist()))]


def filter_complete_frames(paths, minimum_size_ratio):
    if not paths:
        raise ValueError("No BIN files found.")
    median_size = float(np.median([path.stat().st_size for path in paths]))
    minimum_size = median_size * minimum_size_ratio
    eligible = [path for path in paths if path.stat().st_size >= minimum_size]
    return eligible, len(paths) - len(eligible), median_size


def numeric_summary(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": mean,
        "standard_deviation": float(values.std()),
        "coefficient_of_variation": (
            float(values.std() / mean) if mean != 0.0 else 0.0
        ),
    }


def evaluate_frame(path):
    captured_log = io.StringIO()
    with redirect_stdout(captured_log):
        points = detector.load_and_prefilter(path)
        plane_model, ground_points = detector.estimate_ground_plane(points)
        ground = detector.ground_plane_metrics(plane_model)
        _, non_ground_points = detector.align_and_remove_ground(
            points, plane_model, ground_points
        )
        roi_points = detector.apply_roi(non_ground_points)
        clustered_points = detector.voxel_downsample(roi_points)
        labels = detector.cluster_points(clustered_points)
        detections, _ = detector.build_detections(clustered_points, labels)

    cluster_count = len(set(labels.tolist()) - {-1})
    thin_or_tiny = sum(
        detection["dimensions"][1] < 0.5
        or detection["point_count"] < 15
        for detection in detections
    )
    return {
        "frame": path.name,
        "status": "ok",
        "raw_points": path.stat().st_size // 16,
        "filtered_points": len(points),
        "sensor_height_m": ground["sensor_height_m"],
        "ground_tilt_degrees": ground["tilt_degrees"],
        "ground_inliers": len(ground_points),
        "non_ground_roi_points": len(roi_points),
        "clustered_points": len(clustered_points),
        "cluster_count": cluster_count,
        "proposal_count": len(detections),
        "thin_or_tiny_proposals": thin_or_tiny,
    }


def build_report(
    input_dir,
    frames,
    records,
    total_frames,
    excluded_incomplete_frames,
):
    successful = [record for record in records if record["status"] == "ok"]
    failed = [record for record in records if record["status"] != "ok"]
    if not successful:
        raise RuntimeError("Every sampled frame failed validation.")

    proposal_total = sum(record["proposal_count"] for record in successful)
    thin_or_tiny_total = sum(
        record["thin_or_tiny_proposals"] for record in successful
    )
    summary = {
        "source_series": input_dir.name,
        "total_frames": total_frames,
        "excluded_incomplete_frames": excluded_incomplete_frames,
        "sampled_frames": len(frames),
        "successful_frames": len(successful),
        "failed_frames": len(failed),
        "sensor_height_m": numeric_summary(
            [record["sensor_height_m"] for record in successful]
        ),
        "ground_tilt_degrees": numeric_summary(
            [record["ground_tilt_degrees"] for record in successful]
        ),
        "non_ground_roi_points": numeric_summary(
            [record["non_ground_roi_points"] for record in successful]
        ),
        "proposal_count": numeric_summary(
            [record["proposal_count"] for record in successful]
        ),
        "thin_or_tiny_proposal_fraction": (
            thin_or_tiny_total / proposal_total if proposal_total else 0.0
        ),
    }
    return {"summary": summary, "frames": records}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument(
        "--minimum-size-ratio",
        type=float,
        default=0.80,
        help="exclude partial frames smaller than this fraction of median size",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/demo/stability-report.json"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be at least 1.")
    if not 0.0 < args.minimum_size_ratio <= 1.0:
        raise ValueError("--minimum-size-ratio must be in (0, 1].")

    all_paths = sorted(args.input_dir.glob("*.bin"))
    paths, excluded_incomplete, median_size = filter_complete_frames(
        all_paths, args.minimum_size_ratio
    )
    print(
        f"Eligible frames: {len(paths)}/{len(all_paths)}; "
        f"median size={median_size / 16:,.0f} points; "
        f"excluded incomplete={excluded_incomplete}"
    )
    frames = select_evenly_spaced_frames(paths, args.count)
    records = []
    for index, path in enumerate(frames, start=1):
        try:
            record = evaluate_frame(path)
            print(
                f"[{index:02d}/{len(frames):02d}] {path.name}: "
                f"height={record['sensor_height_m']:.3f} m, "
                f"tilt={record['ground_tilt_degrees']:.2f} deg, "
                f"ROI={record['non_ground_roi_points']:,}, "
                f"proposals={record['proposal_count']}"
            )
        except Exception as error:  # keep the batch audit running
            record = {
                "frame": path.name,
                "status": "failed",
                "error": str(error),
            }
            print(f"[{index:02d}/{len(frames):02d}] {path.name}: FAILED: {error}")
        records.append(record)

    report = build_report(
        args.input_dir,
        frames,
        records,
        total_frames=len(all_paths),
        excluded_incomplete_frames=excluded_incomplete,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

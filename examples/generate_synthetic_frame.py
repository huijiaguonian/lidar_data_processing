"""Generate a deterministic N x 4 point-cloud frame for a local demo."""

import argparse
from pathlib import Path

import numpy as np


def sample_box_surface(rng, center, dimensions, count):
    """Sample five visible faces of an axis-aligned box."""
    length, width, height = dimensions
    local = np.empty((count, 3), dtype=np.float32)
    faces = rng.integers(0, 5, size=count)

    local[:, 0] = rng.uniform(-length / 2, length / 2, size=count)
    local[:, 1] = rng.uniform(-width / 2, width / 2, size=count)
    local[:, 2] = rng.uniform(0.05, height, size=count)

    local[faces == 0, 0] = -length / 2
    local[faces == 1, 0] = length / 2
    local[faces == 2, 1] = -width / 2
    local[faces == 3, 1] = width / 2
    local[faces == 4, 2] = height
    local += np.asarray(center, dtype=np.float32)

    intensity = rng.uniform(0.35, 0.9, size=(count, 1)).astype(np.float32)
    return np.hstack([local, intensity])


def build_scene(seed):
    rng = np.random.default_rng(seed)
    ground_count = 16_000
    ground = np.column_stack(
        [
            rng.uniform(-30.0, 30.0, size=ground_count),
            rng.uniform(-5.0, 35.0, size=ground_count),
            rng.normal(0.0, 0.012, size=ground_count),
            rng.uniform(0.08, 0.28, size=ground_count),
        ]
    ).astype(np.float32)

    boxes = [
        sample_box_surface(rng, (4.0, 9.0, 0.0), (4.2, 1.9, 1.6), 1_100),
        sample_box_surface(rng, (-6.0, 16.0, 0.0), (4.5, 2.0, 1.7), 850),
        sample_box_surface(rng, (7.0, 26.0, 0.0), (4.3, 1.8, 1.5), 650),
    ]
    return np.vstack([ground, *boxes]).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/examples/synthetic_frame.bin"),
    )
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    points = build_scene(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    points.tofile(args.output)
    print(f"Saved {len(points):,} points to {args.output}")


if __name__ == "__main__":
    main()

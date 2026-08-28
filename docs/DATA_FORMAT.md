# Data formats

## PandarXT UDP payload

The parser targets PandarXT point-cloud protocol v6.1. Offsets below are relative to the start of the UDP payload.

| Offset | Size | Field used by this project |
|---:|---:|---|
| 0 | 2 | Start of packet, `EE FF` |
| 2 | 2 | Protocol version, expected `06 01` |
| 6 | 1 | Laser/channel count, expected 32 |
| 7 | 1 | Block count, expected 8 |
| 9 | 1 | Distance unit in millimetres |
| 10 | 1 | Return count |
| 11 | 1 | Flags, including optional UDP sequence |
| 12 | 1040 | Eight measurement blocks |

Each measurement block is 130 bytes:

```text
2-byte little-endian azimuth (0.01 degree)
32 channels x (
    2-byte little-endian distance
    1-byte reflectivity
    1-byte reserved
)
```

The data tail starts after the eight blocks. Only the return-mode byte and optional sequence-length validation are used by the maintained parser.

## Distance conversion

The packet header is authoritative:

```python
distance_unit_m = distance_unit_raw / 1000.0
distance_m = raw_distance * distance_unit_m
```

Examples:

| Header byte | Metres per raw unit |
|---:|---:|
| `0x01` | `0.001 m` |
| `0x04` | `0.004 m` |

The project capture was verified at `0.001 m`. Hard-coding `0.004 m` produced an incorrectly scaled scene, so the regression test covers both values.

## Angle correction CSV

The correction file must have exactly 32 channel rows:

```csv
Channel,Elevation,Azimuth
1,...,...
2,...,...
```

Rows are sorted by `Channel` and validated as the complete range 1 through 32. A Pandar64 or Pandar128 calibration file is rejected.

## Frame BIN

Each output frame is a headerless little-endian `float32` array with shape `N x 4`:

| Column | Meaning | Unit/range |
|---:|---|---|
| 0 | X, lateral | metres |
| 1 | Y, forward at azimuth 0 | metres |
| 2 | Z, upward | metres |
| 3 | normalized reflectivity | `[0, 1]` |

A file therefore contains exactly `N * 4 * 4` bytes.

## Detection JSON

`detections.json` is a list of class-agnostic proposals. Each item contains:

- `cluster_label` and `split_index`;
- `point_count`;
- `center: [x, y, z]` in metres;
- `dimensions: [length, width, height]` in metres;
- `angle_degrees`;
- `ground_gap`;
- `range_m`.

These records describe geometry only. They do not contain a vehicle, pedestrian, or cyclist class probability.

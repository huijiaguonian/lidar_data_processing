# Legacy experiments

This directory preserves selected experiments that explain how the final design evolved. None of these files is part of the supported package.

| Directory | Contents | Status |
|---|---|---|
| `clustering/` | AABB, K-means, L-shape, early OBB, and adaptive-distance experiments | Superseded |
| `diagnostics/` | Distance-unit, scale, height, and anomaly investigations | Reference only |
| `parsers/` | Original Pandar128 streaming parser uploaded to the repository | Historical |
| [`supervised/`](supervised/README.md) | PointPillars inference/export attempt, evidence index, and OpenPCDet patch | Stopped |
| `openpcdet/` | Upstream base commit and Apache 2.0 attribution | Reference |

Legacy scripts intentionally retain some old assumptions and machine-specific paths because they document the experiment state. Do not use them as production entry points.

Supported commands are installed from `src/hesai_xt32/`. The development story and reasons for stopping the supervised path are summarized in [../docs/RETROSPECTIVE.md](../docs/RETROSPECTIVE.md).

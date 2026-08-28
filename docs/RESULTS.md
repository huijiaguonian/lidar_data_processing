# Qualitative results

## Scope

These assets were produced from four real PandarXT32 frames in the corrected `frames_bin_xt32` series, using one unchanged parameter set and the maintained height-constrained detector. They demonstrate repeatability of the processing path; they are not an accuracy benchmark because the capture has no ground-truth 3D boxes.

![Four-frame detection overview](assets/demo/detection-demo-grid.png)

| Frame | Aligned environment points | Non-ground ROI points | Geometric proposals |
|---|---:|---:|---:|
| `frame_0000` | 45,849 | 28,047 | 36 |
| `frame_0001` | 48,125 | 29,947 | 35 |
| `frame_0020` | 51,368 | 33,297 | 37 |
| `frame_0100` | 49,971 | 30,497 | 43 |

Individual bird's-eye-view frames:

- [frame 0000](assets/demo/frame_0000.png)
- [frame 0001](assets/demo/frame_0001.png)
- [frame 0020](assets/demo/frame_0020.png)
- [frame 0100](assets/demo/frame_0100.png)

## Sequence stability audit

To avoid judging the detector from hand-picked frames alone, the repository includes an evenly spaced 20-frame audit over the full corrected sequence. The final partial frame was automatically excluded because its size was below 80% of the sequence median.

| Metric | Result |
|---|---:|
| Complete frames eligible for sampling | 1,362 / 1,363 |
| Sampled frames completed | 20 / 20 |
| Estimated sensor height | 1.177-1.287 m |
| Sensor-height mean +/- standard deviation | 1.218 +/- 0.027 m |
| Non-ground ROI coefficient of variation | 5.8% |
| Proposal-count range | 32-46 |
| Proposal-count mean +/- standard deviation | 40.55 +/- 3.88 |

The earlier four-frame run produced 49, 18, 11, and 62 proposals because single-plane RANSAC sometimes selected a plane near the sensor origin. Height-constrained multi-plane selection changed those counts to 36, 35, 37, and 43. This is evidence of improved cross-frame consistency, not proof of detection accuracy.

The audit is reproducible with:

```powershell
python scripts\evaluate_sequence_stability.py `
  D:\Hesai_data\frames_bin_xt32 `
  --count 20 `
  --output docs\assets\demo\stability-report.json
```

The full per-frame measurements are retained in [stability-report.json](assets/demo/stability-report.json).

## How the assets were generated

The detector outputs aligned environment PLY, non-ground ROI PLY, OBB line geometry, and JSON metadata. The repository renderer projects those saved artifacts into a fixed bird's-eye view:

```powershell
python scripts\render_detection_assets.py `
  path\to\frame_0000 `
  path\to\frame_0001 `
  path\to\frame_0020 `
  path\to\frame_0100 `
  --source-series frames_bin_xt32
```

The underlying PCAP and BIN data are intentionally excluded because of size. The PNG files and [summary.json](assets/demo/summary.json) provide a compact, traceable record of these runs.

## Interpretation

The strongest result is not a semantic class score. It is a stable engineering chain from raw packet layout to inspectable geometric proposals:

- packet validation and correct scale;
- block-level revolution splitting;
- height-constrained multi-plane ground selection and alignment;
- density adaptation with range;
- conservative OBB construction;
- portable outputs for Open3D and CloudCompare.

Some visible objects remain missed and some compact structures remain plausible proposals. Without annotations, proposal count must not be interpreted as object count, recall, or precision.

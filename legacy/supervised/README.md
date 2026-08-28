# Supervised-learning exploration (stopped)

## Status

This route is preserved as a documented experiment and is no longer maintained. It is not part of the main installation, CI, or supported command line interface.

## Question explored

Could a PointPillars model trained through OpenPCDet provide semantic 3D detection for the recorded Hesai point cloud without building a new target-domain annotation set?

## Work performed

1. Tested PointPillars with KITTI-trained weights and KITTI-style data conventions.
2. Investigated PandaSet and available Hesai data as closer LiDAR domains.
3. Added PandaSet dataset/configuration changes to OpenPCDet.
4. Added explicit XT32-to-OpenPCDet coordinate conversion for inference.
5. Exported predicted boxes and point clouds for Open3D and CloudCompare inspection.

## Evidence retained

| Artifact | Purpose |
|---|---|
| [configs/pointpillar_pandaset.yaml](configs/pointpillar_pandaset.yaml) | Last PointPillars/PandaSet experiment configuration |
| [demo_pointpillar_xt32.py](demo_pointpillar_xt32.py) | XT32 BIN inference and coordinate conversion |
| [export_visual_results.py](export_visual_results.py) | Export of evaluation results for visual inspection |
| [visualize_results.py](visualize_results.py) | Early Open3D result viewer |
| [openpcdet_changes.patch](openpcdet_changes.patch) | Complete tracked changes against the recorded upstream base |
| [../openpcdet/README.md](../openpcdet/README.md) | Upstream commit, source link, and license |

## Why it was stopped

- KITTI and the recorded XT32 scene differ in sensor geometry, sampling density, classes, and scene distribution.
- Available Hesai examples used different devices, including Pandar64 and PandarGT, rather than the recorded PandarXT32.
- The short target recording did not provide a practical labeled training and evaluation set.
- Coordinate and feature conversion can make files structurally compatible, but cannot remove the sensor and domain gap.
- GPU access and iteration cost were not justified by the observed output quality.
- Generated boxes were less credible than the final geometry-only baseline.

## Conclusion

The experiment established that format compatibility is not model compatibility. Without XT32 target-domain labels and an independent evaluation split, further supervised tuning would risk presenting visually plausible but unsupported results.

Resuming this route would require a real annotation plan, a matching sensor/domain dataset, and quantitative evaluation. Until then, the unsupervised pipeline is the supported project.

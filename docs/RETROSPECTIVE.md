# Project retrospective

## Initial objective

The initial plan was a complete LiDAR perception pipeline:

```text
raw packets -> point-cloud processing -> learning -> detection -> tracking -> rendering
```

The final project did not reach reliable semantic detection or tracking. The portfolio version documents that boundary instead of presenting geometric boxes as trained object recognition.

## What took the most engineering effort

Raw UDP decoding was the critical path. The Ethernet, IP, UDP, and PandarXT headers were readable, but a wrong interpretation at the measurement block or distance-unit level distorted every downstream result. Two details were especially important:

- an XT32 packet contains eight blocks of 32 channels, with a reserved byte after reflectivity;
- distance scale must come from the packet header. The project capture was verified at `0.001 m`, while early hard-coded `0.004 m` parsing was visibly wrong.

Frame boundaries also need block-level azimuth inspection because a revolution can wrap inside a packet.

## Unsupervised experiments

DBSCAN, K-means, PCA, RANSAC, axis-aligned boxes, oriented boxes, and L-shape fitting were explored. A single global clustering threshold performed poorly because angular sampling produces dense near-field points and sparse distant points.

The maintained baseline therefore uses overlapping distance bands and range-specific DBSCAN parameters. Conservative size and gap checks reduced false positives in the evaluated scene, but distant vehicles were still missed. This is a precision-oriented geometric proposal system, not a general detector.

## Supervised experiment

PointPillars was tested through OpenPCDet with KITTI and available Hesai/PandaSet-related data. The result did not transfer reliably to the captured XT32 scene because of several interacting gaps:

- sensor models and beam patterns did not match;
- coordinate and feature conventions required conversion;
- KITTI classes and scene distribution did not match the target capture;
- the small target recording had no practical annotation set;
- training and inference resources were constrained.

The code changes are archived as a patch rather than keeping a complete OpenPCDet copy in this repository.

## Final scope decision

The defensible deliverable is:

- a tested PandarXT32 parser;
- one BIN per azimuth revolution;
- portable visualization/export;
- a transparent CPU-only proposal baseline;
- a documented record of unsuccessful approaches and why they were stopped.

That scope is smaller than the original ambition, but it is reproducible and technically honest.

## Reasonable future work

Future effort would only be justified with target-domain labels or a matching pretrained sensor/domain dataset. The most useful next steps would be a small labeled evaluation split, parameter configuration outside the source file, temporal aggregation, and quantitative precision/recall reporting. Without those prerequisites, further detector tuning risks overfitting one short recording.

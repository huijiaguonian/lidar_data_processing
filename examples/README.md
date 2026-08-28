# Examples

Recorded PCAP, calibration, BIN, and model files are intentionally excluded from Git.

Run the deterministic synthetic generator to create an `N x 4` frame containing a level ground plane and three box-shaped obstacles:

```powershell
python examples\generate_synthetic_frame.py
hesai-xt32-view --bin outputs\examples\synthetic_frame.bin
hesai-xt32-detect --bin outputs\examples\synthetic_frame.bin --no-view
```

The synthetic frame validates the file and command workflow. It is not intended as an accuracy benchmark for the detector.

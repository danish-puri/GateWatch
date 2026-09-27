"""Perception: detect -> track -> entry/exit direction.

- calibration.py: normalized (0..1) geometry, calibrated per stream
- roi.py: crop the tiny gate region; run the detector there at native res
- gating.py: motion-gate the detector (run YOLO only on motion at the gate)
- detector.py / tracker.py: YOLO + ByteTrack
- motion.py: track-level dwell filter (parked vehicles never count)
- crossing.py: oriented gate-mouth entry/exit counting
- dewarp.py: fisheye Stage-2 seam (Stage 1 works in distorted space)
"""

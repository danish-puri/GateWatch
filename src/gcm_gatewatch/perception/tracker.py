"""Multi-object tracking via supervision's ByteTrack.

Replaces v1's two hand-rolled centroid trackers (one in main.py, a different one in
run.py). ByteTrack assigns stable IDs across frames so a vehicle is counted once, not
per frame.
"""

from __future__ import annotations


class VehicleTracker:
    """Assigns and maintains stable track IDs across frames (supervision ByteTrack)."""

    def __init__(self) -> None:
        self._tracker = None  # lazily built so importing this module needs no supervision

    def _load(self):
        if self._tracker is None:
            import supervision as sv

            self._tracker = sv.ByteTrack()
        return self._tracker

    def update(self, detections):  # (supervision.Detections) -> supervision.Detections
        """Attach tracker IDs to this frame's detections."""
        return self._load().update_with_detections(detections)

    def reset(self) -> None:
        """Clear tracker state (e.g. between separate video sources)."""
        if self._tracker is not None:
            self._tracker.reset()

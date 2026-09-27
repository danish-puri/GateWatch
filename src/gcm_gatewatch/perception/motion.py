"""Dwell / stationary filter -- tell a transiting vehicle from a parked one.

The gate-exit camera's forecourt is also a bus depot: 10+ buses sit parked for hours.
Counting at the gate mouth (crossing.py) already means buses parked *in the yard* never
touch the counter. This filter is the safety net for the cases geometry alone can't
catch:

  - a bus parked on or right beside the gate line,
  - tracker ID churn where a long-parked bus is re-detected with a new ID near the line
    and its first frames look like a crossing,
  - detection-box jitter that oscillates a stationary centroid back and forth over the
    line.

It classifies a track by how much it has *moved recently*, not by a permanent label -- so
a bus that has been parked and then pulls out to exit correctly becomes "moving" and its
gate crossing is counted. Pure Python, no deps, unit-testable without a camera.
"""

from __future__ import annotations

from collections import defaultdict, deque

Point = tuple[float, float]  # normalized (x, y) in 0..1


class MotionFilter:
    """Tracks recent centroids per vehicle and reports whether each is moving.

    "Moving" = the spread of the last ``window`` centroids exceeds ``move_radius``
    (normalized). A parked vehicle's centroid barely spreads, so it reads as stationary
    however long it sits there; a vehicle driving through the gate spreads well past the
    radius over a few frames.
    """

    def __init__(self, *, window: int = 5, move_radius: float = 0.02) -> None:
        self.window = window
        self.move_radius = move_radius
        self._hist: dict[int, deque[Point]] = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id: int, centroid: Point) -> None:
        self._hist[track_id].append(centroid)

    @staticmethod
    def _spread(points: deque[Point]) -> float:
        """Diagonal of the bounding box of the recent centroids."""
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5

    def is_moving(self, track_id: int) -> bool:
        """True once there is real, sustained motion. Two frames minimum to judge."""
        hist = self._hist.get(track_id)
        if not hist or len(hist) < 2:
            return False  # not enough evidence -> don't count until motion is confirmed
        return self._spread(hist) >= self.move_radius

    def is_stationary(self, track_id: int) -> bool:
        """True only once a full window confirms the track is parked (not merely new)."""
        hist = self._hist.get(track_id)
        if not hist or len(hist) < self.window:
            return False
        return self._spread(hist) < self.move_radius

    def forget(self, track_id: int) -> None:
        """Drop a retired track to bound memory."""
        self._hist.pop(track_id, None)

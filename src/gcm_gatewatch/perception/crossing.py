"""Entry/exit counting at the gate mouth (not across the yard).

The gate-exit camera overlooks a forecourt/bus depot; the actual gate is a small
sliding gate at the far bottom-left, oblique to the camera (see docs/detection_tuning.md).
v1's full-width horizontal line sat over the *parking* area and counted parked-bus
jitter. v2 instead places a short **oriented line over the gate mouth** and decides
direction from which side of that line a track moves to:

    gate -> yard  = entry (IN)
    yard -> gate  = exit  (OUT)

The line is stored in **normalized** coordinates (0..1), so the same calibration works
across the camera's different stream resolutions (2880 / 2461 / 1195 wide) -- unlike v1's
hardcoded ``y=500``. Detection centroids are normalized by frame size before they reach
this module.

Geometry is pure Python (no numpy / supervision), so it is unit-testable without a GPU,
a camera, or heavy deps. The pipeline adapts the tracker's output (supervision ByteTrack
Detections) into the lightweight ``Observation`` records this module consumes.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from gcm_gatewatch.perception.calibration import to_normalized as normalize
from gcm_gatewatch.perception.motion import MotionFilter

__all__ = [
    "Crossing",
    "Direction",
    "GateLine",
    "GateLineCounter",
    "Kind",
    "Observation",
    "normalize",
]

Point = tuple[float, float]  # normalized (x, y) in 0..1


class Direction(str, Enum):
    IN = "in"    # gate -> yard : a subject entering the college
    OUT = "out"  # yard -> gate : a subject leaving the college


class Kind(str, Enum):
    """What sort of thing crossed. Both walk and drive through Gate 1."""

    VEHICLE = "vehicle"  # bus, car, motorcycle, truck, ...
    PERSON = "person"    # people on foot, overwhelmingly students at this gate


@dataclass(frozen=True)
class Crossing:
    """One subject crossing the gate line."""

    track_id: int
    subject_type: str          # 'bus', 'car', 'person', ... the detector's label
    direction: Direction
    kind: Kind = Kind.VEHICLE  # defaulted so vehicle call sites stay unchanged


@dataclass(frozen=True)
class Observation:
    """One tracked subject on one frame, centroid in normalized coords."""

    track_id: int
    subject_type: str
    centroid: Point
    kind: Kind = Kind.VEHICLE


def _sign(x: float) -> int:
    return 1 if x > 0.0 else -1 if x < 0.0 else 0


def _cross(o: Point, a: Point, b: Point) -> float:
    """Z of (a - o) x (b - o). Sign tells which side of line o->a the point b is on."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """True if segment p1->p2 properly crosses segment p3->p4."""
    d1 = _cross(p3, p4, p1)
    d2 = _cross(p3, p4, p2)
    d3 = _cross(p1, p2, p3)
    d4 = _cross(p1, p2, p4)
    return (
        ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0))
        and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0))
    )


class GateLine:
    """An oriented gate-mouth line with a designated yard (inside) side.

    ``segment`` is the short line drawn across the gate opening; ``yard_point`` is any
    point clearly inside the courtyard. Crossing toward the yard side is an entry.
    """

    def __init__(self, segment: tuple[Point, Point], yard_point: Point) -> None:
        self.a, self.b = segment
        self._yard_sign = _sign(_cross(self.a, self.b, yard_point))
        if self._yard_sign == 0:
            raise ValueError("yard_point lies on the gate line; pick a point off the line")

    def classify(self, prev: Point, curr: Point) -> Direction | None:
        """Direction of a crossing from prev->curr, or None if it didn't cross the mouth.

        Because we test the movement segment against the gate *segment* (not an infinite
        line), only motion through the gate opening counts -- vehicles moving elsewhere in
        the yard are ignored.
        """
        if not _segments_intersect(prev, curr, self.a, self.b):
            return None
        side = _sign(_cross(self.a, self.b, curr))
        if side == 0:
            return None
        return Direction.IN if side == self._yard_sign else Direction.OUT


class GateLineCounter:
    """Emits entry/exit crossings for tracked subjects at the gate mouth.

    Vehicles and people go through the same geometry. A student walking in and a bus
    driving in both cut the same short line over the gate opening, and the direction test
    does not care how big the thing was. Only the label and the kind differ, and both ride
    along on the Observation.

    Keeps each track's last centroid and reports a crossing when its path cuts the gate
    line. Three guards keep spurious counts off the counter:

    - ``min_travel`` (normalized) drops single-frame sub-pixel jitter,
    - the optional ``motion`` filter suppresses crossings from tracks without sustained
      recent motion, so a bus parked on the gate line (or a re-detected parked bus) never
      counts. A parked bus that pulls out to leave reads as moving again and *is* counted.
    - ``cooldown`` debounces a single track: once it crosses, it can't cross again for
      ``cooldown`` updates. A large vehicle *turning* at the gate wobbles its centroid
      back and forth over the line and would otherwise emit OUT-then-IN on one track;
      the cooldown collapses that maneuver into a single event. A genuine later re-cross
      (the same vehicle leaving minutes afterward) is well past the window and still
      counts.
    """

    def __init__(
        self,
        gate: GateLine,
        *,
        min_travel: float = 0.01,
        motion: MotionFilter | None = None,
        cooldown: int = 40,
    ) -> None:
        self.gate = gate
        self.min_travel = min_travel
        self.motion = motion
        self.cooldown = cooldown
        self._last: dict[int, Point] = {}
        self._update_idx = 0
        self._last_cross_idx: dict[int, int] = {}

    def update(self, observations: Iterable[Observation]) -> list[Crossing]:
        """Process one frame's tracked vehicles; return any crossings that occurred."""
        self._update_idx += 1
        crossings: list[Crossing] = []
        for obs in observations:
            if self.motion is not None:
                self.motion.update(obs.track_id, obs.centroid)
            prev = self._last.get(obs.track_id)
            self._last[obs.track_id] = obs.centroid
            if prev is None:
                continue
            if _dist(prev, obs.centroid) < self.min_travel:
                continue
            direction = self.gate.classify(prev, obs.centroid)
            if direction is None:
                continue
            # parked/stationary tracks never count, even if their box jitters over the line
            if self.motion is not None and not self.motion.is_moving(obs.track_id):
                continue
            # debounce a wobbling/turning track: one crossing per cooldown window
            last = self._last_cross_idx.get(obs.track_id)
            if last is not None and (self._update_idx - last) < self.cooldown:
                continue
            self._last_cross_idx[obs.track_id] = self._update_idx
            crossings.append(
                Crossing(obs.track_id, obs.subject_type, direction, obs.kind)
            )
        return crossings

    def forget(self, track_id: int) -> None:
        """Drop a track that the tracker has retired, to bound memory."""
        self._last.pop(track_id, None)
        self._last_cross_idx.pop(track_id, None)
        if self.motion is not None:
            self.motion.forget(track_id)


def _dist(p: Point, q: Point) -> float:
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5

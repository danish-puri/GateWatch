"""Tests for gate-mouth entry/exit direction.

v1's run.py had the IN/OUT labels swapped and a counter that could go negative, and none
of it was tested. These pin the direction semantics so that regression can't return. The
geometry is dependency-free, so these run without a GPU, camera, or heavy deps.

Gate mouth: a short line at bottom-left; the yard (inside) is to the right of it.
Convention: crossing toward the yard = entry (IN); toward the gate = exit (OUT).
"""

from __future__ import annotations

from gcm_gatewatch.perception.crossing import (
    Direction,
    GateLine,
    GateLineCounter,
    Observation,
    normalize,
)

# A near-vertical gate line at x~0.14; yard is to the right (larger x).
GATE = GateLine(segment=((0.14, 0.55), (0.14, 0.90)), yard_point=(0.55, 0.45))


def test_gate_to_yard_is_entry() -> None:
    # move left -> right across the gate line
    assert GATE.classify((0.10, 0.72), (0.20, 0.72)) is Direction.IN


def test_yard_to_gate_is_exit() -> None:
    # move right -> left across the gate line
    assert GATE.classify((0.20, 0.72), (0.10, 0.72)) is Direction.OUT


def test_motion_elsewhere_in_yard_does_not_count() -> None:
    # movement well to the right of the gate never touches the mouth segment
    assert GATE.classify((0.50, 0.50), (0.70, 0.50)) is None


def test_motion_missing_the_short_segment_does_not_count() -> None:
    # crosses the infinite line's x, but above the segment's y-span -> no crossing
    assert GATE.classify((0.10, 0.20), (0.20, 0.20)) is None


def test_counter_needs_two_frames_then_emits_once() -> None:
    counter = GateLineCounter(GATE, min_travel=0.0)
    assert counter.update([Observation(1, "bus", (0.10, 0.72))]) == []  # first sighting
    crossings = counter.update([Observation(1, "bus", (0.20, 0.72))])
    assert len(crossings) == 1
    assert crossings[0].direction is Direction.IN
    assert crossings[0].subject_type == "bus"


def test_counter_ignores_sub_threshold_jitter() -> None:
    # a parked vehicle shivering on the line must not be counted
    counter = GateLineCounter(GATE, min_travel=0.05)
    counter.update([Observation(2, "car", (0.139, 0.72))])
    crossings = counter.update([Observation(2, "car", (0.141, 0.72))])  # tiny move
    assert crossings == []


def test_normalize_is_resolution_independent() -> None:
    # the frame centre normalizes to (0.5, 0.5) at any stream resolution
    assert normalize((1440, 810), (2880, 1620)) == (0.5, 0.5)
    assert normalize((597, 335), (1194, 670)) == (0.5, 0.5)

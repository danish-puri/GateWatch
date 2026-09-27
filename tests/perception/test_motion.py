"""Tests for the dwell/stationary filter and its effect on gate counting.

The scenario that matters: a bus parked on the gate line whose detection box jitters
back and forth across it must never be counted, while a bus that actually drives through
the gate must be counted exactly once. This is v1's single biggest error source here.
"""

from __future__ import annotations

from gcm_gatewatch.perception.crossing import (
    Direction,
    GateLine,
    GateLineCounter,
    Observation,
)
from gcm_gatewatch.perception.motion import MotionFilter

GATE = GateLine(segment=((0.14, 0.55), (0.14, 0.90)), yard_point=(0.55, 0.45))


def test_moving_track_is_moving_not_stationary() -> None:
    m = MotionFilter(window=5, move_radius=0.02)
    for x in (0.10, 0.13, 0.16, 0.19, 0.22):  # marching right, spread 0.12
        m.update(1, (x, 0.72))
    assert m.is_moving(1) is True
    assert m.is_stationary(1) is False


def test_parked_track_is_stationary_not_moving() -> None:
    m = MotionFilter(window=5, move_radius=0.02)
    for _ in range(6):  # sitting still (tiny wobble)
        m.update(2, (0.140, 0.72))
        m.update(2, (0.142, 0.72))
    assert m.is_moving(2) is False
    assert m.is_stationary(2) is True


def test_new_track_is_not_yet_stationary() -> None:
    # a freshly-seen track must not be declared parked before the window fills
    m = MotionFilter(window=5, move_radius=0.02)
    m.update(3, (0.14, 0.72))
    assert m.is_stationary(3) is False


def test_parked_bus_jittering_over_the_line_is_not_counted() -> None:
    # centroid oscillates across x=0.14 by ~0.015 each frame: each step crosses the line
    # geometrically and clears min_travel, but the spread stays under move_radius.
    counter = GateLineCounter(
        GATE, min_travel=0.01, motion=MotionFilter(window=5, move_radius=0.02)
    )
    total = []
    for _ in range(10):
        total += counter.update([Observation(4, "bus", (0.1325, 0.72))])
        total += counter.update([Observation(4, "bus", (0.1475, 0.72))])
    assert total == []  # never counted despite repeatedly cutting the line


def test_transiting_vehicle_still_counts_with_filter_on() -> None:
    counter = GateLineCounter(
        GATE, min_travel=0.01, motion=MotionFilter(window=5, move_radius=0.02)
    )
    crossings: list = []
    for x in (0.10, 0.13, 0.16, 0.19):  # a real drive-through, spread 0.09
        crossings += counter.update([Observation(5, "car", (x, 0.72))])
    assert len(crossings) == 1
    assert crossings[0].direction is Direction.IN


def test_cooldown_debounces_a_turning_vehicle_double_cross() -> None:
    # a large vehicle turning at the gate wobbles across the line: it would emit
    # OUT then IN on the same track. The cooldown collapses that to a single event.
    counter = GateLineCounter(
        GATE, min_travel=0.01, motion=MotionFilter(window=5, move_radius=0.02), cooldown=40
    )
    emitted: list = []
    # build motion history, then oscillate across the line on one track
    xs = [0.10, 0.13, 0.16, 0.19, 0.16, 0.13, 0.16, 0.19]  # right, then wobble back/forth
    for x in xs:
        emitted += counter.update([Observation(9, "bus", (x, 0.72))])
    assert len(emitted) == 1  # one crossing, not several


def test_genuine_recross_after_cooldown_counts() -> None:
    counter = GateLineCounter(
        GATE, min_travel=0.01, motion=MotionFilter(window=5, move_radius=0.02), cooldown=3
    )
    emitted: list = []
    for x in (0.10, 0.13, 0.16, 0.19):   # enter (IN)
        emitted += counter.update([Observation(10, "car", (x, 0.72))])
    # many frames of clear motion pass (past the cooldown), then it leaves (OUT)
    for x in (0.19, 0.16, 0.13, 0.10):   # exit
        emitted += counter.update([Observation(10, "car", (x, 0.72))])
    assert [c.direction for c in emitted] == [Direction.IN, Direction.OUT]


def test_parked_bus_that_pulls_out_to_exit_is_counted() -> None:
    # bus sits parked just inside the yard, then drives out through the gate
    counter = GateLineCounter(
        GATE, min_travel=0.01, motion=MotionFilter(window=5, move_radius=0.02)
    )
    out: list = []
    for _ in range(5):  # parked inside the yard, right of the line
        out += counter.update([Observation(6, "bus", (0.20, 0.72))])
    for x in (0.19, 0.16, 0.13, 0.10):  # pulls out toward the gate
        out += counter.update([Observation(6, "bus", (x, 0.72))])
    assert len(out) == 1
    assert out[0].direction is Direction.OUT

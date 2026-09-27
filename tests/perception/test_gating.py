"""Tests for the motion-gate decision policy (threshold + cooldown).

The point: the detector runs while there is motion at the gate and for a short cooldown
after, then stops -- saving CPU on an idle yard without dropping a vehicle that pauses
mid-crossing.
"""

from __future__ import annotations

from gcm_gatewatch.perception.gating import MotionGate


def test_motion_above_threshold_triggers_detection() -> None:
    gate = MotionGate(min_foreground_ratio=0.02, cooldown_frames=3)
    assert gate.should_detect(0.05) is True


def test_idle_below_threshold_does_not_detect() -> None:
    gate = MotionGate(min_foreground_ratio=0.02, cooldown_frames=3)
    assert gate.should_detect(0.0) is False


def test_cooldown_keeps_detecting_then_stops() -> None:
    gate = MotionGate(min_foreground_ratio=0.02, cooldown_frames=3)
    assert gate.should_detect(0.10) is True   # motion -> detect, arm cooldown=3
    # motion stops; detector keeps running for 3 more frames
    assert gate.should_detect(0.0) is True     # cooldown 3 -> 2
    assert gate.should_detect(0.0) is True     # 2 -> 1
    assert gate.should_detect(0.0) is True     # 1 -> 0
    assert gate.should_detect(0.0) is False    # cooldown exhausted -> stop


def test_new_motion_rearms_cooldown() -> None:
    gate = MotionGate(min_foreground_ratio=0.02, cooldown_frames=2)
    gate.should_detect(0.10)          # arm
    gate.should_detect(0.0)           # 2 -> 1
    assert gate.should_detect(0.10) is True   # fresh motion re-arms to 2
    assert gate.should_detect(0.0) is True    # 2 -> 1
    assert gate.should_detect(0.0) is True    # 1 -> 0
    assert gate.should_detect(0.0) is False

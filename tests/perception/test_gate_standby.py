"""Tests for the motion gate standing down when it is not paying for itself.

Measured on the real footage, the gate can only skip 2.4% of frames on a busy afternoon
while costing about a seventh of a detection to run, so blindly running it loses time.
These cover the policy that stops that: judge the skip rate against real measured costs,
go to standby when it falls short, and relearn the background before judging again.

`foreground_ratio` is stubbed throughout, so none of this needs OpenCV or a frame.
"""

from __future__ import annotations

import pytest

from gcm_gatewatch.perception.gating import GateState, MotionGate


def _gate(**kwargs) -> MotionGate:
    defaults = {
        "min_foreground_ratio": 0.02,
        "cooldown_frames": 0,
        "review_frames": 10,
        "standby_frames": 5,
        "warmup_frames": 2,
        "min_skip_rate": 0.15,
    }
    return MotionGate(**{**defaults, **kwargs})


def _drive(gate: MotionGate, ratios, monkeypatch) -> list[bool]:
    """Run the gate over a sequence of foreground ratios, skipping the pixel work."""
    values = iter(ratios)
    monkeypatch.setattr(MotionGate, "foreground_ratio", lambda self, crop: next(values))
    return [gate.observe_frame(None) for _ in ratios]


def test_warmup_never_skips(monkeypatch) -> None:
    """A half-learned background reads as wall-to-wall motion; its verdict means nothing."""
    gate = _gate(warmup_frames=3)
    decisions = _drive(gate, [0.0, 0.0, 0.0], monkeypatch)

    assert decisions == [True, True, True]   # detected despite a zero ratio
    assert gate.frames_gated == 0            # and none of it counted as evidence
    assert gate.state is GateState.ACTIVE    # warmup done, now judging for real


def test_a_busy_gate_goes_to_standby(monkeypatch) -> None:
    """The afternoon case: nothing gets skipped, so the gate should stop charging for it."""
    gate = _gate()
    _drive(gate, [1.0] * 2 + [1.0] * 10, monkeypatch)  # warmup, then a full review window

    assert gate.state is GateState.STANDBY
    assert gate.standbys == 1


def test_a_quiet_gate_stays_active(monkeypatch) -> None:
    """The night case: it skips nearly everything, which is the whole point of having it."""
    gate = _gate()
    _drive(gate, [1.0] * 2 + [0.0] * 10, monkeypatch)

    assert gate.state is GateState.ACTIVE
    assert gate.frames_skipped == 10


def test_standby_costs_nothing_and_detects_everything(monkeypatch) -> None:
    """In standby the pixel work must not run at all, or it has not saved anything."""
    gate = _gate()
    _drive(gate, [1.0] * 2 + [1.0] * 10, monkeypatch)
    assert gate.state is GateState.STANDBY

    calls = 0

    def counting_ratio(self, crop):
        nonlocal calls
        calls += 1
        return 0.0

    monkeypatch.setattr(MotionGate, "foreground_ratio", counting_ratio)
    assert [gate.observe_frame(None) for _ in range(4)] == [True] * 4
    assert calls == 0


def test_standby_wakes_through_warmup_not_straight_into_judgement(monkeypatch) -> None:
    """The trap: waking mid-relearn scores a zero skip rate and sleeps forever.

    The gate must relearn the background before its verdict counts, otherwise a yard that
    empties overnight is never noticed and the gate never switches back on.
    """
    gate = _gate(standby_frames=3, warmup_frames=2)
    _drive(gate, [1.0] * 2 + [1.0] * 10, monkeypatch)
    assert gate.state is GateState.STANDBY

    # sleep out the standby, then feed the ratios a relearning model would produce:
    # everything looks like motion at first, then the yard reads as genuinely empty
    _drive(gate, [0.0] * 3, monkeypatch)          # standby elapses -> warmup
    assert gate.state is GateState.WARMUP

    _drive(gate, [1.0, 1.0], monkeypatch)         # relearning frames, verdict ignored
    assert gate.state is GateState.ACTIVE
    assert gate._seen == 0                        # relearning did not count against it

    _drive(gate, [0.0] * 10, monkeypatch)         # a genuinely quiet window
    assert gate.state is GateState.ACTIVE         # so it stays on and keeps saving
    assert gate.frames_skipped == 10


def test_break_even_comes_from_measured_costs() -> None:
    """Whether the gate pays depends on the machine, so it should not be a guess."""
    gate = _gate(min_skip_rate=0.15)
    assert gate.break_even_skip_rate() == 0.15   # fallback until both costs are known

    gate._gate_cost = 0.0065      # what subtraction measured at
    gate.note_detector_cost(0.045)  # what a detection measured at
    assert gate.break_even_skip_rate() == pytest.approx(0.0065 / 0.045, rel=0.02)


def test_break_even_is_capped_when_the_gate_costs_more_than_detecting() -> None:
    """If subtraction were dearer than detection, no skip rate could justify it."""
    gate = _gate()
    gate._gate_cost = 0.10
    gate.note_detector_cost(0.045)
    assert gate.break_even_skip_rate() == 1.0


def test_health_reports_whether_it_is_worth_running() -> None:
    gate = _gate()
    assert gate.health()["state"] == "warmup"
    assert gate.health()["skip_rate"] is None

"""Tests for the visit repository.

Covers: crossings persist; an entry followed by an exit pairs into one closed visit; an
entry with no exit stays open; count_by_type windows by time and direction; recent
crossings come back newest-first; plate matching beats the FIFO fallback; and an exit
with no matching entry is recorded without closing a visit.

Uses in-memory SQLite (the repository holds one connection, so ":memory:" persists for
its lifetime) -- runs with no filesystem and no pytest fixtures.
"""

from __future__ import annotations

from gcm_gatewatch.perception.crossing import Crossing, Direction
from gcm_gatewatch.storage.repository import VisitRepository


def _repo() -> VisitRepository:
    repo = VisitRepository(":memory:")
    repo.initialize()
    return repo


def _bus(track_id: int) -> Crossing:
    return Crossing(track_id=track_id, subject_type="bus", direction=Direction.IN)


def test_entry_then_exit_forms_one_closed_visit() -> None:
    repo = _repo()
    repo.record_crossing(_bus(1), camera="exit", ts=1000)  # IN
    repo.record_crossing(
        Crossing(2, "bus", Direction.OUT), camera="exit", ts=1600  # OUT, 10 min later
    )
    assert repo.open_visits() == []  # the visit is closed
    # dwell is entered_at..exited_at
    row = repo._conn.execute("SELECT entered_at, exited_at FROM visits").fetchone()
    assert row["entered_at"] == 1000
    assert row["exited_at"] == 1600


def test_entry_without_exit_is_open_visit() -> None:
    repo = _repo()
    repo.record_crossing(_bus(1), camera="exit", ts=1000)
    open_v = repo.open_visits()
    assert len(open_v) == 1
    assert open_v[0]["subject_type"] == "bus"
    assert open_v[0]["exited_at"] is None


def test_count_by_type_windows_by_time_and_direction() -> None:
    repo = _repo()
    repo.record_crossing(Crossing(1, "bus", Direction.IN), camera="exit", ts=100)
    repo.record_crossing(Crossing(2, "car", Direction.IN), camera="exit", ts=200)
    repo.record_crossing(Crossing(3, "bus", Direction.IN), camera="exit", ts=5000)  # outside
    repo.record_crossing(Crossing(4, "bus", Direction.OUT), camera="exit", ts=150)  # other dir
    counts = repo.count_by_type(since=0, until=1000, direction="in")
    assert counts == {"bus": 1, "car": 1}  # ts=5000 excluded, OUT excluded


def test_recent_crossings_newest_first_with_limit() -> None:
    repo = _repo()
    for i, ts in enumerate((10, 20, 30), start=1):
        repo.record_crossing(Crossing(i, "car", Direction.IN), camera="exit", ts=ts)
    recent = repo.recent_crossings(limit=2)
    assert [c["ts"] for c in recent] == [30, 20]


def test_plate_match_beats_fifo_fallback() -> None:
    repo = _repo()
    # two buses open; the second carries a plate
    repo.record_crossing(Crossing(1, "bus", Direction.IN), camera="exit", ts=100)  # no plate
    repo.record_crossing(
        Crossing(2, "bus", Direction.IN), camera="exit", plate_text="BA9CHA4812", ts=200
    )
    # an exit with that plate must close the plated visit, not the older FIFO one
    repo.record_crossing(
        Crossing(3, "bus", Direction.OUT), camera="exit", plate_text="BA9CHA4812", ts=900
    )
    open_v = repo.open_visits()
    assert len(open_v) == 1
    assert open_v[0]["entered_at"] == 100  # the un-plated, older visit is still open


def test_exit_without_entry_records_crossing_but_no_visit() -> None:
    repo = _repo()
    repo.record_crossing(Crossing(1, "truck", Direction.OUT), camera="exit", ts=500)
    assert repo.open_visits() == []
    assert repo._conn.execute("SELECT COUNT(*) AS n FROM visits").fetchone()["n"] == 0
    # the crossing itself is still on record
    assert repo.recent_crossings(limit=5)[0]["direction"] == "out"

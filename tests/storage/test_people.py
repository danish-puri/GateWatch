"""Tests for logging people alongside vehicles.

Covers: a person crossing is recorded; a person never opens a visit (there is no identity
to pair against); a plate on a person is refused; the plate CHECK holds at the SQL level
too; counts and the daily view separate the two kinds; and net_flow reads entries minus
exits.
"""

from __future__ import annotations

import sqlite3

import pytest

from gcm_gatewatch.perception.crossing import Crossing, Direction, Kind
from gcm_gatewatch.storage.repository import VisitRepository


def _repo() -> VisitRepository:
    repo = VisitRepository(":memory:")
    repo.initialize()
    return repo


def _person(track_id: int, direction: Direction = Direction.IN) -> Crossing:
    return Crossing(track_id, "person", direction, Kind.PERSON)


def test_person_crossing_is_recorded() -> None:
    repo = _repo()
    repo.record_crossing(_person(1), camera="entry", ts=1000)
    rows = repo.recent_crossings(limit=5)
    assert len(rows) == 1
    assert rows[0]["kind"] == "person"
    assert rows[0]["subject_type"] == "person"
    assert rows[0]["direction"] == "in"


def test_person_crossing_never_opens_a_visit() -> None:
    """Hundreds of identical person detections a day make FIFO pairing meaningless."""
    repo = _repo()
    repo.record_crossing(_person(1), camera="entry", ts=1000)
    repo.record_crossing(_person(2, Direction.OUT), camera="exit", ts=2000)
    assert repo.open_visits() == []
    assert repo._conn.execute("SELECT COUNT(*) AS n FROM visits").fetchone()["n"] == 0


def test_person_exit_does_not_close_a_vehicle_visit() -> None:
    """A student walking out must never be mistaken for the bus leaving."""
    repo = _repo()
    repo.record_crossing(Crossing(1, "bus", Direction.IN), camera="entry", ts=1000)
    repo.record_crossing(_person(2, Direction.OUT), camera="exit", ts=1500)
    open_visits = repo.open_visits()
    assert len(open_visits) == 1
    assert open_visits[0]["exited_at"] is None


def test_plate_on_a_person_is_refused() -> None:
    repo = _repo()
    with pytest.raises(ValueError, match="cannot carry a plate"):
        repo.record_crossing(_person(1), camera="entry", plate_text="BA9CHA4812", ts=1000)


def test_schema_itself_rejects_a_person_with_a_plate() -> None:
    """The writer guards it, and so does the table, so no other writer can slip past."""
    repo = _repo()
    with pytest.raises(sqlite3.IntegrityError):
        repo._conn.execute(
            "INSERT INTO crossings "
            "(track_id, camera, kind, subject_type, direction, plate_text, ts) "
            "VALUES (1, 'entry', 'person', 'person', 'in', 'BA9CHA4812', 1000)"
        )


def test_plate_confidence_needs_a_plate() -> None:
    repo = _repo()
    with pytest.raises(ValueError, match="without a plate_text"):
        repo.record_crossing(
            Crossing(1, "bus", Direction.IN), camera="entry", plate_confidence=0.9, ts=1000
        )


def test_counts_separate_people_from_vehicles() -> None:
    repo = _repo()
    repo.record_crossing(Crossing(1, "bus", Direction.IN), camera="entry", ts=100)
    repo.record_crossing(_person(2), camera="entry", ts=110)
    repo.record_crossing(_person(3), camera="entry", ts=120)

    assert repo.count_by_type(since=0, until=999, direction="in", kind=Kind.VEHICLE) == {"bus": 1}
    assert repo.count_by_type(since=0, until=999, direction="in", kind=Kind.PERSON) == {"person": 2}
    # no kind filter counts everything together
    assert repo.count_by_type(since=0, until=999, direction="in") == {"bus": 1, "person": 2}


def test_net_flow_is_entries_minus_exits() -> None:
    repo = _repo()
    for track_id in (1, 2, 3):
        repo.record_crossing(_person(track_id), camera="entry", ts=100 + track_id)
    repo.record_crossing(_person(4, Direction.OUT), camera="exit", ts=200)

    assert repo.net_flow(since=0, until=999, kind=Kind.PERSON) == 2
    assert repo.net_flow(since=0, until=999, kind=Kind.VEHICLE) == 0


def test_recent_crossings_can_filter_by_kind() -> None:
    repo = _repo()
    repo.record_crossing(Crossing(1, "car", Direction.IN), camera="entry", ts=100)
    repo.record_crossing(_person(2), camera="entry", ts=200)
    people = repo.recent_crossings(limit=10, kind=Kind.PERSON)
    assert [r["subject_type"] for r in people] == ["person"]


def test_daily_flow_groups_by_day_kind_and_direction() -> None:
    repo = _repo()
    repo.record_crossing(Crossing(1, "bus", Direction.IN), camera="entry", ts=1_700_000_000)
    repo.record_crossing(_person(2), camera="entry", ts=1_700_000_100)
    repo.record_crossing(_person(3), camera="entry", ts=1_700_000_200)

    flow = repo.daily_flow()
    people_in = [r for r in flow if r["kind"] == "person" and r["direction"] == "in"]
    assert len(people_in) == 1
    assert people_in[0]["crossings"] == 2
    assert all(set(r) == {"day", "kind", "subject_type", "direction", "crossings"} for r in flow)

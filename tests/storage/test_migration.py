"""Tests that a database written by the vehicles-only schema survives the upgrade.

The first schema had `vehicle_type` and no notion of a person crossing. `CREATE TABLE IF
NOT EXISTS` would leave such a table untouched and every later insert would fail, so the
repository migrates it on initialize(). Real gate data must not be lost to that.
"""

from __future__ import annotations

import sqlite3

from gcm_gatewatch.perception.crossing import Crossing, Direction, Kind
from gcm_gatewatch.storage.repository import VisitRepository

LEGACY_SCHEMA = """
CREATE TABLE crossings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id     INTEGER NOT NULL,
    camera       TEXT    NOT NULL,
    vehicle_type TEXT    NOT NULL,
    direction    TEXT    NOT NULL CHECK (direction IN ('in', 'out')),
    plate_text   TEXT,
    ts           INTEGER NOT NULL
);
CREATE TABLE visits (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_type      TEXT    NOT NULL,
    plate_text        TEXT,
    entered_at        INTEGER NOT NULL,
    exited_at         INTEGER,
    entry_crossing_id INTEGER NOT NULL REFERENCES crossings (id),
    exit_crossing_id  INTEGER REFERENCES crossings (id)
);
"""


def _legacy_db(path) -> None:
    conn = sqlite3.connect(str(path))
    conn.executescript(LEGACY_SCHEMA)
    conn.execute(
        "INSERT INTO crossings (track_id, camera, vehicle_type, direction, plate_text, ts) "
        "VALUES (7, 'exit', 'bus', 'in', NULL, 1000)"
    )
    conn.execute(
        "INSERT INTO visits (vehicle_type, entered_at, entry_crossing_id) "
        "VALUES ('bus', 1000, 1)"
    )
    conn.commit()
    conn.close()


def test_legacy_rows_survive_and_become_vehicles(tmp_path) -> None:
    db = tmp_path / "legacy.db"
    _legacy_db(db)

    repo = VisitRepository(db)
    repo.initialize()

    rows = repo.recent_crossings(limit=10)
    assert len(rows) == 1
    assert rows[0]["subject_type"] == "bus"   # renamed from vehicle_type
    assert rows[0]["kind"] == "vehicle"       # backfilled, since v1 logged vehicles only
    assert rows[0]["track_id"] == 7           # the actual data is untouched

    open_visits = repo.open_visits()
    assert len(open_visits) == 1
    assert open_visits[0]["subject_type"] == "bus"
    repo.close()


def test_migrated_database_accepts_person_crossings(tmp_path) -> None:
    """The point of the migration: new rows work in an old database."""
    db = tmp_path / "legacy.db"
    _legacy_db(db)

    repo = VisitRepository(db)
    repo.initialize()
    repo.record_crossing(Crossing(1, "person", Direction.IN, Kind.PERSON), camera="entry", ts=2000)

    assert repo.count_by_type(since=0, until=9999, direction="in", kind=Kind.PERSON) == {
        "person": 1
    }
    repo.close()


def test_migration_is_idempotent(tmp_path) -> None:
    """initialize() runs on every start, so running it twice must be harmless."""
    db = tmp_path / "legacy.db"
    _legacy_db(db)

    repo = VisitRepository(db)
    repo.initialize()
    repo.initialize()
    assert len(repo.recent_crossings(limit=10)) == 1
    repo.close()

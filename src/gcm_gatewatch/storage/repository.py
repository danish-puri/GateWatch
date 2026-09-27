"""Data access for crossings and visits.

The rest of the app (pipeline, agents, API) goes through this repository rather than
touching SQL directly. It records raw crossings for both kinds of subject that pass the
gate, vehicles and people, pairs vehicle crossings into visits, and answers the read
queries the dashboard and the agents need.

Pairing an exit to its entry is best-effort and vehicles-only. With a readable plate we
match on it; without one (the current cameras can't resolve plates -- see
docs/feasibility.md) there is no reliable identity across an hours-long visit, because
tracker IDs are per-session and differ between the entry and exit crossings. So the
fallback is a heuristic: close the oldest still-open visit of the same vehicle type
(FIFO). An exit with no match is still recorded as a crossing (and is itself a useful
anomaly signal for the monitor agent); it simply does not close a visit.

People are logged as crossings and aggregated, never paired. See schema.sql for why
FIFO is defensible for a few dozen vehicles a day and meaningless for hundreds of
identical person detections.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from gcm_gatewatch.perception.crossing import Crossing, Direction, Kind

_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class VisitRepository:
    """SQLite-backed store for crossings and visits."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = db_path
        # check_same_thread=False: the pipeline writes from a capture thread while the API
        # reads from another. SQLite serializes writes; keep operations short.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")

    def initialize(self) -> None:
        """Bring the database up to the current schema (idempotent)."""
        self._migrate_legacy()
        self._conn.executescript(_SCHEMA_PATH.read_text())
        self._conn.commit()

    def _migrate_legacy(self) -> None:
        """Upgrade a pre-`kind` database in place, before the schema script runs.

        The first schema logged vehicles only, so it had `vehicle_type` and no notion of
        a person crossing. `CREATE TABLE IF NOT EXISTS` would silently leave such a table
        alone and every insert would then fail, so old columns are renamed and the new
        ones added here first.

        One honest limitation: SQLite cannot add a CHECK to an existing table, so a
        migrated database keeps its data and its indices but not the plate/kind
        constraints. A database created fresh gets them. Rebuilding the tables to add
        constraints would risk real logged data for a guard the writer already enforces,
        which is not a trade worth making.
        """
        for table in ("crossings", "visits"):
            columns = {
                row["name"]
                for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if not columns:
                continue  # table does not exist yet; the schema script will create it
            if "vehicle_type" in columns:
                self._conn.execute(
                    f"ALTER TABLE {table} RENAME COLUMN vehicle_type TO subject_type"
                )
            if "kind" not in columns:
                self._conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN kind TEXT NOT NULL DEFAULT 'vehicle'"
                )
        crossing_columns = {
            row["name"] for row in self._conn.execute("PRAGMA table_info(crossings)").fetchall()
        }
        if crossing_columns and "plate_confidence" not in crossing_columns:
            self._conn.execute("ALTER TABLE crossings ADD COLUMN plate_confidence REAL")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------ writes

    def record_crossing(
        self,
        crossing: Crossing,
        *,
        camera: str,
        plate_text: str | None = None,
        plate_confidence: float | None = None,
        ts: int | None = None,
    ) -> int:
        """Persist one crossing and update the visit it belongs to. Returns the row id.

        Only vehicle crossings touch the visits table. A person crossing is recorded and
        nothing more, because there is no identity to pair it against.
        """
        if plate_text is not None and crossing.kind is not Kind.VEHICLE:
            raise ValueError(
                f"a {crossing.kind.value} crossing cannot carry a plate: {plate_text!r}"
            )
        if plate_confidence is not None and plate_text is None:
            raise ValueError("plate_confidence given without a plate_text")

        ts = int(time.time()) if ts is None else ts
        cur = self._conn.execute(
            "INSERT INTO crossings "
            "(track_id, camera, kind, subject_type, direction, plate_text, plate_confidence, ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (crossing.track_id, camera, crossing.kind.value, crossing.subject_type,
             crossing.direction.value, plate_text, plate_confidence, ts),
        )
        crossing_id = int(cur.lastrowid)

        if crossing.kind is Kind.VEHICLE:
            if crossing.direction is Direction.IN:
                self._open_visit(
                    crossing, plate_text=plate_text, ts=ts, entry_crossing_id=crossing_id
                )
            else:  # Direction.OUT
                self._close_matching_visit(
                    crossing, plate_text=plate_text, ts=ts, exit_crossing_id=crossing_id
                )

        self._conn.commit()
        return crossing_id

    def _open_visit(
        self, crossing: Crossing, *, plate_text: str | None, ts: int, entry_crossing_id: int
    ) -> None:
        self._conn.execute(
            "INSERT INTO visits (kind, subject_type, plate_text, entered_at, entry_crossing_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (crossing.kind.value, crossing.subject_type, plate_text, ts, entry_crossing_id),
        )

    def _close_matching_visit(
        self, crossing: Crossing, *, plate_text: str | None, ts: int, exit_crossing_id: int
    ) -> None:
        row = None
        if plate_text:
            # exact identity: the most recent open visit with this plate
            row = self._conn.execute(
                "SELECT id FROM visits WHERE exited_at IS NULL AND plate_text = ? "
                "ORDER BY entered_at DESC LIMIT 1",
                (plate_text,),
            ).fetchone()
        if row is None:
            # heuristic fallback: oldest open visit of the same vehicle type (FIFO).
            # entered_at <= ts keeps the CHECK on visits satisfied, and skipping a visit
            # that started after this exit is right anyway: it cannot be the match.
            row = self._conn.execute(
                "SELECT id FROM visits WHERE exited_at IS NULL AND subject_type = ? "
                "AND entered_at <= ? ORDER BY entered_at ASC LIMIT 1",
                (crossing.subject_type, ts),
            ).fetchone()
        if row is not None:
            self._conn.execute(
                "UPDATE visits SET exited_at = ?, exit_crossing_id = ? WHERE id = ?",
                (ts, exit_crossing_id, row["id"]),
            )
        # else: exit with no matching entry -> crossing stands alone, no visit closed

    # ------------------------------------------------------------------ reads

    def count_by_type(
        self, *, since: int, until: int, direction: str, kind: Kind | str | None = None
    ) -> dict[str, int]:
        """Counts per subject type in [since, until] (inclusive) for one direction.

        `kind` narrows to vehicles or people; leave it None to count both together.
        """
        sql = (
            "SELECT subject_type, COUNT(*) AS n FROM crossings "
            "WHERE ts BETWEEN ? AND ? AND direction = ?"
        )
        params: list = [since, until, direction]
        if kind is not None:
            sql += " AND kind = ?"
            params.append(kind.value if isinstance(kind, Kind) else kind)
        rows = self._conn.execute(sql + " GROUP BY subject_type", params).fetchall()
        return {r["subject_type"]: r["n"] for r in rows}

    def net_flow(self, *, since: int, until: int, kind: Kind | str) -> int:
        """Entries minus exits for one kind over a window.

        For people this is the only occupancy signal available, and it is an estimate:
        anyone who was already inside when the window opened is not counted, and a missed
        detection skews it either way. Read it as a trend, not as a headcount.
        """
        value = kind.value if isinstance(kind, Kind) else kind
        row = self._conn.execute(
            "SELECT "
            "  SUM(CASE WHEN direction = 'in'  THEN 1 ELSE 0 END) - "
            "  SUM(CASE WHEN direction = 'out' THEN 1 ELSE 0 END) AS net "
            "FROM crossings WHERE ts BETWEEN ? AND ? AND kind = ?",
            (since, until, value),
        ).fetchone()
        return int(row["net"] or 0)

    def daily_flow(self, *, day: str | None = None) -> list[dict]:
        """Rows from the v_daily_flow view, optionally for one 'YYYY-MM-DD' local day."""
        if day is None:
            rows = self._conn.execute(
                "SELECT * FROM v_daily_flow ORDER BY day DESC, kind, subject_type, direction"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM v_daily_flow WHERE day = ? ORDER BY kind, subject_type, direction",
                (day,),
            ).fetchall()
        return [dict(r) for r in rows]

    def open_visits(self) -> list[dict]:
        """Vehicles that entered but have no recorded exit. Feeds anomaly detection."""
        rows = self._conn.execute(
            "SELECT * FROM visits WHERE exited_at IS NULL ORDER BY entered_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def recent_crossings(self, *, limit: int, kind: Kind | str | None = None) -> list[dict]:
        """Most recent crossings, newest first, for the monitor agent to reason over."""
        sql = "SELECT * FROM crossings"
        params: list = []
        if kind is not None:
            sql += " WHERE kind = ?"
            params.append(kind.value if isinstance(kind, Kind) else kind)
        params.append(limit)
        rows = self._conn.execute(sql + " ORDER BY ts DESC, id DESC LIMIT ?", params).fetchall()
        return [dict(r) for r in rows]

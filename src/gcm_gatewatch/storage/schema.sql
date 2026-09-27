-- GCM GateWatch schema.
--
-- v1 stored one flat row per crossing with TEXT timestamps, no indices, and no way to
-- pair an entry with its matching exit. v2 models a "visit": a subject enters, and
-- (usually) later exits. Crossings are the raw events; visits are the useful unit.
--
-- The gate log covers two KINDS of subject, because both walk and drive through Gate 1:
--
--   kind = 'vehicle'   bus, car, motorcycle, truck, ... (subject_type says which)
--   kind = 'person'    people on foot, overwhelmingly students at this gate
--
-- `plate_text` belongs to vehicles only, and a CHECK enforces it, so a person row can
-- never carry a plate. Plates stay nullable even for vehicles, because the two current
-- cameras cannot resolve Devanagari plate characters (docs/feasibility.md). The column
-- and the OCR path are in place so the log starts filling in on its own the day a
-- plate-grade camera is added, with no schema change.

PRAGMA journal_mode = WAL;   -- concurrent reads while the pipeline writes
PRAGMA foreign_keys = ON;

-- Raw line crossings, one row per event. This is the entry/exit log proper.
CREATE TABLE IF NOT EXISTS crossings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id         INTEGER NOT NULL,
    camera           TEXT    NOT NULL,             -- 'entry' | 'exit'
    kind             TEXT    NOT NULL CHECK (kind IN ('vehicle', 'person')),
    subject_type     TEXT    NOT NULL,             -- bus | car | motorcycle | person | ...
    direction        TEXT    NOT NULL CHECK (direction IN ('in', 'out')),
    plate_text       TEXT,                         -- vehicles only, often unavailable
    plate_confidence REAL,                         -- OCR confidence 0..1 for plate_text
    ts               INTEGER NOT NULL,             -- unix epoch seconds

    -- a plate is a property of a vehicle, never of a person
    CHECK (kind = 'vehicle' OR plate_text IS NULL),
    -- a confidence only means something next to an actual plate read
    CHECK (plate_confidence IS NULL
           OR (plate_text IS NOT NULL AND plate_confidence BETWEEN 0.0 AND 1.0))
);
CREATE INDEX IF NOT EXISTS idx_crossings_ts ON crossings (ts);
CREATE INDEX IF NOT EXISTS idx_crossings_track ON crossings (track_id);
CREATE INDEX IF NOT EXISTS idx_crossings_kind_ts ON crossings (kind, ts);
CREATE INDEX IF NOT EXISTS idx_crossings_plate ON crossings (plate_text)
    WHERE plate_text IS NOT NULL;

-- A visit pairs an entry crossing with its later exit crossing.
--
-- Vehicles only, deliberately. Pairing needs identity, and without a readable plate the
-- only fallback is "close the oldest open visit of the same type" (FIFO). That is a
-- defensible guess for vehicles, a few dozen a day spread across distinguishable types.
-- It would be meaningless for people, where hundreds of identical 'person' detections
-- cross daily and FIFO would invent arrivals and departures that never happened. So
-- people are logged as crossings and aggregated, and no pairing is invented for them.
-- `kind` is still carried here so every query reads the same across both tables, and so
-- person visits become recordable unchanged once a real identity source exists.
CREATE TABLE IF NOT EXISTS visits (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    kind              TEXT    NOT NULL CHECK (kind IN ('vehicle', 'person')),
    subject_type      TEXT    NOT NULL,
    plate_text        TEXT,
    entered_at        INTEGER NOT NULL,
    exited_at         INTEGER,                  -- null while the subject is still inside
    entry_crossing_id INTEGER NOT NULL REFERENCES crossings (id),
    exit_crossing_id  INTEGER REFERENCES crossings (id),

    CHECK (kind = 'vehicle' OR plate_text IS NULL),
    CHECK (exited_at IS NULL OR exited_at >= entered_at)
);
CREATE INDEX IF NOT EXISTS idx_visits_entered ON visits (entered_at);
CREATE INDEX IF NOT EXISTS idx_visits_open ON visits (exited_at) WHERE exited_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_visits_plate ON visits (plate_text)
    WHERE plate_text IS NOT NULL;

-- Daily footfall and traffic, the shape the college actually asks for. Dates render in
-- local time via SQLite's 'localtime' modifier, so a "day" means a day at the gate
-- rather than a UTC day, which would split Kathmandu evenings across two rows.
CREATE VIEW IF NOT EXISTS v_daily_flow AS
SELECT
    date(ts, 'unixepoch', 'localtime') AS day,
    kind,
    subject_type,
    direction,
    COUNT(*)                           AS crossings
FROM crossings
GROUP BY day, kind, subject_type, direction;

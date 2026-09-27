"""Tests for the pipeline wiring.

The frame loop itself needs a camera and YOLO, so what is tested here is the logic
around it that decides whether a row is correct: refusing a stream whose crop no longer
matches its calibration, turning tracked crop detections into whole-frame observations of
the right kind, writing crossings for both kinds, and reporting a quiet stream as
degraded.
"""

from __future__ import annotations

import threading
import time
import types

import numpy as np
import pytest

from gcm_gatewatch.config import Settings
from gcm_gatewatch.perception.crossing import Crossing, Direction, Kind
from gcm_gatewatch.service.pipeline import Pipeline, StreamWorker, resolve_config_path
from gcm_gatewatch.storage.repository import VisitRepository

CONFIG = """
cameras:
  exit:
    primary: main
    streams:
      main:
        rtsp_env: TEST_CAM_RTSP
        resolution: [2880, 1620]
        gate:
          line: [[0.12, 0.55], [0.16, 0.90]]
          yard_point: [0.55, 0.45]
perception:
  model_path: models/yolo.pt
  vehicle_classes: [2, 5]
  person_classes: [0]
vision: {}
plate: {}
storage:
  db_path: gate.db
agents: {}
service: {}
logging: {}
"""


@pytest.fixture
def settings(tmp_path) -> Settings:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    path = config_dir / "config.yaml"
    path.write_text(CONFIG)
    return Settings.load(path)


@pytest.fixture
def repository() -> VisitRepository:
    repo = VisitRepository(":memory:")
    repo.initialize()
    return repo


@pytest.fixture
def worker(settings, repository) -> StreamWorker:
    return StreamWorker(
        "exit",
        settings.cameras["exit"].stream(),
        settings=settings,
        repository=repository,
        stop=threading.Event(),
    )


def _frame(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _tracked(rows: list[tuple[list[float], int, int | None]]):
    """A stand-in for supervision.Detections carrying only what the pipeline reads."""
    return types.SimpleNamespace(
        xyxy=np.array([r[0] for r in rows], dtype=float),
        class_id=np.array([r[1] for r in rows]),
        tracker_id=np.array([r[2] for r in rows]),
    )


def test_prepare_accepts_a_different_resolution_of_the_same_view(worker) -> None:
    """Normalized calibration is the whole point: 1195x671 is the same 16:9 scene."""
    assert worker._prepare(_frame(1195, 671)) is True
    assert worker._resolution == (1195, 671)
    assert worker._roi_box is not None


def test_prepare_refuses_a_different_crop(worker) -> None:
    """2461x1228 is 2:1, a different crop, so the normalized gate points somewhere else."""
    assert worker._prepare(_frame(2461, 1228)) is False
    assert worker._roi_box is None
    assert "recalibrate" in worker.last_error


def test_a_miscalibrated_stream_complains_once_not_every_frame(worker, monkeypatch) -> None:
    """8 frames a second of the same error would bury everything else in the log."""
    errors = []
    monkeypatch.setattr(
        "gcm_gatewatch.service.pipeline.log",
        types.SimpleNamespace(error=lambda *a, **k: errors.append(k), info=lambda *a, **k: None),
    )

    for _ in range(10):
        assert worker._prepare(_frame(2461, 1228)) is False

    assert len(errors) == 1


def test_observations_carry_the_right_kind_and_label(worker) -> None:
    worker.detector = types.SimpleNamespace(names={0: "person", 5: "bus"})
    worker._prepare(_frame(2880, 1620))

    observations = worker._observations(
        _tracked([([0.0, 0.0, 10.0, 10.0], 5, 1), ([20.0, 20.0, 30.0, 30.0], 0, 2)]),
        worker._roi_box,
        worker._resolution,
    )

    assert [(o.subject_type, o.kind) for o in observations] == [
        ("bus", Kind.VEHICLE),
        ("person", Kind.PERSON),
    ]


def test_observations_map_crop_pixels_back_to_whole_frame(worker) -> None:
    """A detection made in the crop has to land in the same 0..1 space as the gate line."""
    worker.detector = types.SimpleNamespace(names={5: "bus"})
    worker._prepare(_frame(2880, 1620))
    x0, y0, _, _ = worker._roi_box

    observations = worker._observations(
        _tracked([([0.0, 0.0, 20.0, 40.0], 5, 1)]), worker._roi_box, worker._resolution
    )

    x, y = observations[0].centroid
    assert x == pytest.approx((x0 + 10.0) / 2880)
    assert y == pytest.approx((y0 + 20.0) / 1620)
    assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def test_observations_skip_unconfirmed_tracks(worker) -> None:
    """ByteTrack leaves tracker_id None until it is sure; counting those double-counts."""
    worker.detector = types.SimpleNamespace(names={5: "bus"})
    worker._prepare(_frame(2880, 1620))

    observations = worker._observations(
        _tracked([([0.0, 0.0, 10.0, 10.0], 5, None)]), worker._roi_box, worker._resolution
    )

    assert observations == []


def test_observations_skip_classes_the_config_does_not_know(worker) -> None:
    """A model returning an unconfigured class must not guess at its kind."""
    worker.detector = types.SimpleNamespace(names={11: "kite"})
    worker._prepare(_frame(2880, 1620))

    observations = worker._observations(
        _tracked([([0.0, 0.0, 10.0, 10.0], 11, 1)]), worker._roi_box, worker._resolution
    )

    assert observations == []


def test_record_writes_both_kinds(worker, repository) -> None:
    worker._record(Crossing(1, "bus", Direction.IN))
    worker._record(Crossing(2, "person", Direction.IN, Kind.PERSON))

    rows = repository.recent_crossings(limit=5)
    assert {r["kind"] for r in rows} == {"vehicle", "person"}
    assert all(r["camera"] == "exit" for r in rows)
    assert worker.crossings_written == 2
    assert worker.last_write_at is not None


def test_record_survives_a_plate_failure(worker, repository, settings) -> None:
    """A plate is a bonus. Losing the crossing to get it would be the wrong trade."""
    settings.plate.enabled = True
    worker._read_plate = lambda: (_ for _ in ()).throw(RuntimeError("cloud vision is down"))

    worker._record(Crossing(1, "bus", Direction.IN))

    rows = repository.recent_crossings(limit=5)
    assert len(rows) == 1
    assert rows[0]["plate_text"] is None


def test_health_reports_a_quiet_stream_as_degraded(settings, repository) -> None:
    pipeline = Pipeline(settings, repository=repository)
    pipeline.workers = pipeline.build_workers()

    assert pipeline.health()["status"] == "degraded"  # no frames yet

    pipeline.workers[0].last_frame_at = time.time()
    assert pipeline.health()["status"] == "ok"

    pipeline.workers[0].last_frame_at = time.time() - (Pipeline.STALE_FRAME_SECONDS + 1)
    health = pipeline.health()
    assert health["status"] == "degraded"
    assert "exit" in health["detail"]


def test_a_quiet_gate_is_not_unhealthy(settings, repository) -> None:
    """3am with no crossings is normal. Alerting on it teaches everyone to ignore alerts."""
    pipeline = Pipeline(settings, repository=repository)
    pipeline.workers = pipeline.build_workers()
    pipeline.workers[0].last_frame_at = time.time()

    health = pipeline.health()

    assert health["status"] == "ok"
    assert health["streams"][0]["crossings_written"] == 0


def test_one_worker_per_camera(settings, repository) -> None:
    pipeline = Pipeline(settings, repository=repository)
    assert [w.name for w in pipeline.build_workers()] == ["exit"]


def test_config_path_comes_from_argv_then_env_then_default(monkeypatch) -> None:
    monkeypatch.delenv("GCM_GATEWATCH_CONFIG", raising=False)
    assert resolve_config_path(["gcm-gatewatch"]).as_posix() == "config/config.yaml"

    monkeypatch.setenv("GCM_GATEWATCH_CONFIG", "/etc/gatewatch.yaml")
    assert resolve_config_path(["gcm-gatewatch"]).as_posix() == "/etc/gatewatch.yaml"
    assert resolve_config_path(["gcm-gatewatch", "/tmp/other.yaml"]).as_posix() == "/tmp/other.yaml"

"""Pipeline wiring and process entry point.

Ties the pieces together. Per camera stream, each frame flows:

    capture frame
      -> compute the gate ROI once per stream (roi.gate_roi from the normalized gate)
      -> crop the ROI (roi.crop)
      -> motion gate (gating.MotionGate.observe_frame): run the detector only on motion
      -> detector.detect on the crop at native res
      -> map crop centroids to full-frame normalized coords (roi.local_to_global_norm)
      -> tracker (ByteTrack) -> Observation records
      -> crossing.GateLineCounter (+ motion.MotionFilter dwell filter) -> Crossings
      -> storage.VisitRepository

Both kinds of subject ride that one path. A student walking in and a bus driving in are
the same geometry problem, and the detector's class id decides which kind gets written.

Runs the monitor and watchdog agents on their intervals alongside, and launches the
FastAPI service. This is `gcm-gatewatch` (see [project.scripts]); it replaces v1's two
ad-hoc scripts with one supervised, headless process.
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from pathlib import Path

from gcm_gatewatch.capture.stream import ReconnectingStream
from gcm_gatewatch.config import ConfigError, Settings, StreamConfig
from gcm_gatewatch.observability.logging import configure as configure_logging
from gcm_gatewatch.observability.logging import get_logger
from gcm_gatewatch.perception import roi
from gcm_gatewatch.perception.calibration import aspect_matches
from gcm_gatewatch.perception.crossing import GateLine, GateLineCounter, Kind, Observation
from gcm_gatewatch.perception.detector import VehicleDetector
from gcm_gatewatch.perception.gating import MotionGate
from gcm_gatewatch.perception.motion import MotionFilter
from gcm_gatewatch.perception.tracker import VehicleTracker
from gcm_gatewatch.storage.repository import VisitRepository

log = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path("config/config.yaml")


class StreamWorker:
    """Runs one camera stream's capture -> perception -> storage loop.

    One worker per stream, each with its own tracker, counter and detector. None of that
    state is shareable: ByteTrack IDs, the dwell history and the background model all
    describe one particular view, and crossing the wires between the entry and exit
    cameras would produce confident nonsense.
    """

    def __init__(
        self,
        name: str,
        stream_config: StreamConfig,
        *,
        settings: Settings,
        repository: VisitRepository,
        stop: threading.Event,
        max_reconnects: int | None = None,
    ) -> None:
        self.name = name
        self.stream_config = stream_config
        self.settings = settings
        self.repository = repository
        self.stop = stop
        # None means reconnect forever, which is what a live camera wants. Set it to 0 to
        # make one pass over a finite source, which is how the accuracy runs in testing/
        # replay the recorded clips through this exact loop instead of a parallel one.
        self.max_reconnects = max_reconnects

        perception = settings.perception
        self.detector = VehicleDetector(
            str(perception.model_path),
            classes=perception.detect_classes,
            confidence=perception.confidence,
            class_labels=perception.class_labels,
        )
        self.tracker = VehicleTracker()
        self.motion_gate = MotionGate(
            min_foreground_ratio=perception.motion_gate_min_foreground,
            cooldown_frames=perception.motion_gate_cooldown,
            sample_width=perception.motion_gate_sample_width,
            review_frames=perception.motion_gate_review_frames,
            standby_frames=perception.motion_gate_standby_frames,
            warmup_frames=perception.motion_gate_warmup_frames,
            min_skip_rate=perception.motion_gate_min_skip_rate,
        )
        self.counter = GateLineCounter(
            GateLine(stream_config.gate.line, stream_config.gate.yard_point),
            motion=MotionFilter(
                window=perception.motion_window,
                move_radius=perception.motion_move_radius,
            ),
            cooldown=perception.crossing_cooldown,
        )
        self.stream: ReconnectingStream | None = None

        # health signals, read by /healthz and the watchdog
        self.frames_seen = 0
        self.detections_run = 0
        self.crossings_written = 0
        self.last_frame_at: float | None = None
        self.last_write_at: float | None = None
        self.last_error: str | None = None
        self._roi_box: tuple[int, int, int, int] | None = None
        self._resolution: tuple[int, int] | None = None
        self._rejected: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------ per frame

    def _prepare(self, frame) -> bool:
        """Fix the ROI to this stream's real frame size. False means do not use the stream.

        The configured `resolution` is what the stream was calibrated against. A live
        frame of a different size but the same aspect is fine, since the gate is stored
        normalized. A different aspect is a different crop of the scene, so the normalized
        gate points at the wrong place and the only honest move is to refuse to count
        rather than to log wrong crossings.
        """
        height, width = frame.shape[:2]
        actual = (width, height)
        if self._resolution == actual:
            return True
        configured = self.stream_config.resolution
        if not aspect_matches(configured, actual):
            self.last_error = (
                f"stream delivers {width}x{height}, calibrated for "
                f"{configured[0]}x{configured[1]}; different crop, recalibrate this stream"
            )
            # Once per resolution, not once per frame. A miscalibrated stream still
            # delivers 8 frames a second, and repeating this forever would bury every
            # other event in the log of a process that runs for weeks.
            if actual not in self._rejected:
                self._rejected.add(actual)
                log.error("stream_aspect_mismatch", stream=self.name, detail=self.last_error)
            return False
        self._resolution = actual
        self._roi_box = roi.gate_roi(
            self.stream_config.gate.line,
            resolution=actual,
            margin=self.settings.perception.gate_roi_margin,
        )
        log.info(
            "stream_calibrated",
            stream=self.name,
            resolution=f"{width}x{height}",
            roi=self._roi_box,
        )
        return True

    def _observations(self, tracked, box, resolution) -> list[Observation]:
        """Turn tracked crop detections into whole-frame normalized observations."""
        names = self.detector.names
        perception = self.settings.perception
        observations: list[Observation] = []
        for xyxy, class_id, track_id in zip(
            tracked.xyxy, tracked.class_id, tracked.tracker_id, strict=False
        ):
            if track_id is None:
                continue  # ByteTrack has not confirmed this detection yet
            x1, y1, x2, y2 = (float(v) for v in xyxy)
            centroid = roi.local_to_global_norm(
                ((x1 + x2) / 2.0, (y1 + y2) / 2.0), box, resolution
            )
            class_id = int(class_id)
            try:
                kind = perception.kind_for_class(class_id)
            except ConfigError:
                continue  # a class the model returned but the config does not classify
            observations.append(
                Observation(
                    track_id=int(track_id),
                    subject_type=names.get(class_id, str(class_id)),
                    centroid=centroid,
                    kind=kind,
                )
            )
        return observations

    def _record(self, crossing) -> None:
        """Write one crossing, reading a plate first when that path is switched on."""
        plate_text = plate_confidence = None
        if self.settings.plate.enabled and crossing.kind is Kind.VEHICLE:
            try:
                plate_text, plate_confidence = self._read_plate()
            # a plate is a bonus; never lose a crossing over one
            except Exception as exc:  # noqa: BLE001
                log.warning("plate_read_failed", stream=self.name, error=str(exc))

        self.repository.record_crossing(
            crossing,
            camera=self.name,
            plate_text=plate_text,
            plate_confidence=plate_confidence,
        )
        self.crossings_written += 1
        self.last_write_at = time.time()
        log.info(
            "crossing",
            stream=self.name,
            kind=crossing.kind.value,
            subject_type=crossing.subject_type,
            direction=crossing.direction.value,
            track_id=crossing.track_id,
            plate=plate_text,
        )

    def _read_plate(self) -> tuple[str | None, float | None]:
        """Seam for plate OCR (plate/ocr.py -> plate/nepali_plate.py).

        Left unwired on purpose. The current cameras cannot resolve Devanagari plate
        characters (docs/feasibility.md), so `plate.enabled` is false by default and
        nothing calls this. Wiring the crop and the Cloud Vision call in here is the whole
        job once a plate-grade camera exists.
        """
        return None, None

    # ------------------------------------------------------------------ loop

    def run(self) -> None:
        url = self.stream_config.rtsp_url()
        self.stream = ReconnectingStream(url, max_reconnects=self.max_reconnects)
        stride = self.settings.perception.frame_stride
        log.info("stream_starting", stream=self.name, stride=stride)

        try:
            for frame in self.stream.frames(stride=stride):
                if self.stop.is_set():
                    break
                self.frames_seen += 1
                self.last_frame_at = time.time()
                try:
                    self._process(frame)
                except Exception as exc:
                    # One bad frame must not end a run that is supposed to last weeks,
                    # so every failure is caught here, logged, and stepped over.
                    self.last_error = str(exc)
                    log.exception("frame_failed", stream=self.name, error=str(exc))
        finally:
            self.stream.close()
            log.info("stream_stopped", stream=self.name, frames=self.frames_seen)

    def _process(self, frame) -> None:
        if not self._prepare(frame):
            return
        assert self._roi_box is not None and self._resolution is not None

        crop = roi.crop(frame, self._roi_box)
        if not self.motion_gate.observe_frame(crop):
            return  # nothing moving at the gate, skip the expensive part

        self.detections_run += 1
        started = time.perf_counter()
        detections = self.detector.detect(crop)
        # tell the gate what it just saved by not skipping, so it can work out for itself
        # whether it is worth running on this machine
        self.motion_gate.note_detector_cost(time.perf_counter() - started)
        tracked = self.tracker.update(detections)
        for crossing in self.counter.update(
            self._observations(tracked, self._roi_box, self._resolution)
        ):
            self._record(crossing)

    def health(self) -> dict:
        """This stream's health signals."""
        now = time.time()
        return {
            "stream": self.name,
            "frames_seen": self.frames_seen,
            "detections_run": self.detections_run,
            "crossings_written": self.crossings_written,
            "seconds_since_frame": (
                None if self.last_frame_at is None else round(now - self.last_frame_at, 1)
            ),
            "seconds_since_write": (
                None if self.last_write_at is None else round(now - self.last_write_at, 1)
            ),
            "reconnects": self.stream.reconnects if self.stream else 0,
            "motion_gate": self.motion_gate.health(),
            "last_error": self.last_error,
        }


class Pipeline:
    """Owns the capture -> perception -> storage loop and the background agents."""

    # A stream that has delivered nothing for this long is presumed broken, even though
    # the reconnect logic keeps trying. It is what /healthz reports on.
    STALE_FRAME_SECONDS = 120

    def __init__(self, settings: Settings, *, repository: VisitRepository | None = None) -> None:
        self.settings = settings
        self.repository = repository or VisitRepository(settings.storage.db_path)
        self.stop = threading.Event()
        self.workers: list[StreamWorker] = []
        self._threads: list[threading.Thread] = []
        self.started_at = time.time()

    def build_workers(self) -> list[StreamWorker]:
        """One worker per camera's primary stream."""
        workers = []
        for camera_name, camera in self.settings.cameras.items():
            workers.append(
                StreamWorker(
                    camera_name,
                    camera.stream(),
                    settings=self.settings,
                    repository=self.repository,
                    stop=self.stop,
                )
            )
        return workers

    def run(self) -> None:
        """Start capture loops and the API; run until stopped."""
        self.repository.initialize()
        self.workers = self.build_workers()

        for worker in self.workers:
            thread = threading.Thread(target=worker.run, name=f"stream-{worker.name}", daemon=True)
            thread.start()
            self._threads.append(thread)
        log.info("pipeline_started", streams=[w.name for w in self.workers])

        try:
            self._serve()
        finally:
            self.shutdown()

    def _serve(self) -> None:
        """Serve the API in the foreground. Blocks until the server exits."""
        import uvicorn

        from gcm_gatewatch.service.api import create_app

        uvicorn.run(
            create_app(self, analyst=None),
            host=self.settings.service.host,
            port=self.settings.service.port,
            log_config=None,  # keep structlog's handler, do not let uvicorn replace it
        )

    def shutdown(self) -> None:
        """Ask the workers to stop and wait briefly for them."""
        if self.stop.is_set():
            return
        log.info("pipeline_stopping")
        self.stop.set()
        for worker in self.workers:
            if worker.stream is not None:
                worker.stream.close()
        for thread in self._threads:
            thread.join(timeout=10)
        self.repository.close()
        log.info("pipeline_stopped")

    def health(self) -> dict:
        """Current health signals for the watchdog and /healthz.

        Unhealthy means a stream has gone quiet past STALE_FRAME_SECONDS, or has never
        produced a frame at all. A gate with no crossings is not unhealthy, since a quiet
        gate at 3am is the normal case and alerting on it would train everyone to ignore
        the alerts.
        """
        streams = [worker.health() for worker in self.workers]
        stale = [
            s["stream"]
            for s in streams
            if s["seconds_since_frame"] is None
            or s["seconds_since_frame"] > self.STALE_FRAME_SECONDS
        ]
        return {
            "status": "ok" if self.workers and not stale else "degraded",
            "detail": f"no frames from: {', '.join(stale)}" if stale else None,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "streams": streams,
        }


def resolve_config_path(argv: list[str]) -> Path:
    """Config path from the command line, then the environment, then the default."""
    if len(argv) > 1:
        return Path(argv[1])
    return Path(os.environ.get("GCM_GATEWATCH_CONFIG", DEFAULT_CONFIG_PATH))


def main() -> None:
    """Console-script entry point: load config, build the pipeline, run it."""
    try:
        settings = Settings.load(resolve_config_path(sys.argv))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    configure_logging(level=settings.logging.level, json=settings.logging.as_json)

    # Fail at startup with the list of what to fix. An unattended process that starts
    # "successfully" and then cannot reach a camera is much harder to diagnose later.
    missing = settings.missing_secrets()
    if missing:
        log.error("missing_secrets", variables=missing)
        print(
            "missing environment variables: " + ", ".join(missing) + "\n"
            "Set them in .env (see .env.example) or in the service environment.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    pipeline = Pipeline(settings)

    def _handle_signal(signum, _frame):
        log.info("signal_received", signal=signal.Signals(signum).name)
        pipeline.shutdown()

    # systemd sends SIGTERM on stop; Ctrl-C sends SIGINT. Both mean shut down cleanly so
    # the SQLite connection closes and the WAL is checkpointed.
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    pipeline.run()

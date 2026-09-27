"""Typed configuration loaded from config.yaml + .env.

Replaces v1's hardcoded constants (pixel lines, RTSP URLs with embedded credentials,
Dash port). Non-secret settings come from YAML; secrets come from the environment and
are never written to the YAML or to source.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from pydantic_settings import BaseSettings

from gcm_gatewatch.perception.crossing import Kind

Point = tuple[float, float]      # normalized (x, y) in 0..1, resolution-independent
Resolution = tuple[int, int]     # (width, height) in pixels


class ConfigError(RuntimeError):
    """Raised when configuration is missing or unusable, always with what to fix."""


def load_env_file(path: Path) -> dict[str, str]:
    """Read a KEY=VALUE .env file into os.environ without clobbering the real environment.

    Real environment variables win over the file on purpose. A systemd unit or a Docker
    `-e` flag is the deliberate, deployed configuration, and a stray .env left in the
    working directory should never quietly override it.

    Returns the variables this call actually set, which keeps it testable and lets the
    caller log the names (never the values) at startup.
    """
    applied: dict[str, str] = {}
    if not path.exists():
        return applied
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


class GateZone(BaseModel):
    """The oriented gate-mouth line and which side is the yard (inside).

    `line` is a short segment across the gate opening (bottom-left in the exit camera),
    not a full-width line across the yard. `yard_point` is any point clearly inside the
    courtyard; crossing toward it counts as an entry. All coordinates are normalized 0..1.
    """

    line: tuple[Point, Point]
    yard_point: Point


class StreamConfig(BaseModel):
    """One RTSP stream of a camera, calibrated on its own.

    Normalized zones transfer across resolutions of the *same* field of view but not
    across a different crop/aspect, so each stream (e.g. main vs sub) carries its own
    `resolution` and `gate`. `resolution` is the size this stream delivers -- used to map
    pixel<->normalized and to aspect-check live frames against this calibration.
    """

    rtsp_env: str          # env-var name holding the real rtsp://user:pass@... URL
    resolution: Resolution
    gate: GateZone

    def rtsp_url(self) -> str:
        """Resolve the RTSP URL from the environment.

        The URL carries the camera password, so it lives only in the environment. v1 had
        it inline in run.py, which is how it ended up in git history.
        """
        url = os.environ.get(self.rtsp_env)
        if not url:
            raise ConfigError(
                f"environment variable {self.rtsp_env} is not set. "
                f"Add it to .env (see .env.example) or export it before starting."
            )
        return url


class CameraConfig(BaseModel):
    """One physical camera and its independently-calibrated streams."""

    streams: dict[str, StreamConfig]   # e.g. {"main": ..., "sub": ...}
    primary: str = "main"              # stream the pipeline runs on by default

    def stream(self, name: str | None = None) -> StreamConfig:
        """Return a stream's config (the primary one by default)."""
        return self.streams[name or self.primary]


class PerceptionConfig(BaseModel):
    model_path: Path
    vehicle_classes: list[int]
    # People on foot, logged as kind='person'. At GCM these are overwhelmingly students,
    # but the detector reports a class and not an identity, so the log says 'person'.
    # Set to [] to stop detecting people entirely.
    person_classes: list[int] = [0]     # COCO id 0
    confidence: float = 0.4
    frame_stride: int = 3       # process every Nth frame to stay light on CPU
    # dwell/stationary filter: parked vehicles must never touch the counter
    motion_window: int = 5      # frames of history used to judge motion
    motion_move_radius: float = 0.02  # normalized spread below this = parked
    # debounce: after a track crosses, ignore its crossings for this many processed
    # frames -- collapses a vehicle turning/wobbling at the gate into one event
    crossing_cooldown: int = 40
    # gate ROI: crop the gate region and run the detector there at native res
    gate_roi_margin: float = 0.06        # fraction of frame padded around the gate line
    # motion gate: only run the detector when the ROI shows motion (CPU saver, no GPU).
    # Measured on the real footage, this only pays off when the gate is quiet, so it now
    # stands down when it is not earning. See perception/gating.py for the numbers.
    motion_gate_min_foreground: float = 0.005  # ROI moving-pixel ratio that triggers detection
    motion_gate_cooldown: int = 15       # keep detecting this many frames after motion stops
    motion_gate_sample_width: int = 320  # downscale the ROI to this before subtracting
    motion_gate_review_frames: int = 300     # evidence gathered before judging the gate
    motion_gate_standby_frames: int = 2000   # how long to stay out of the way once judged
    motion_gate_warmup_frames: int = 150     # relearn the background before trusting it
    # fallback break-even skip rate, used only until the gate and detector costs are
    # both measured; after that the real ratio decides
    motion_gate_min_skip_rate: float = 0.15
    # fisheye: Stage 1 works in distorted space; enable to undistort first (Stage 2).
    # See perception/dewarp.py and docs/detection_tuning.md.
    dewarp_enabled: bool = False
    # class id -> label. COCO lacks Nepali types (tempo/auto/micro); a fine-tuned model
    # adds those ids and their labels go here. Empty = use the model's built-in names.
    class_labels: dict[int, str] = {}

    @model_validator(mode="after")
    def _classes_do_not_overlap(self) -> PerceptionConfig:
        both = set(self.vehicle_classes) & set(self.person_classes)
        if both:
            raise ValueError(
                f"class ids {sorted(both)} appear in both vehicle_classes and "
                f"person_classes; a subject cannot be logged as both kinds"
            )
        return self

    @property
    def detect_classes(self) -> list[int]:
        """Every class id the detector should return, vehicles and people together.

        One detector pass covers both. Running YOLO twice over the same crop to keep the
        two apart would double the CPU cost for nothing, since the class id already says
        which is which.
        """
        return sorted(set(self.vehicle_classes) | set(self.person_classes))

    def kind_for_class(self, class_id: int) -> Kind:
        """Map a detector class id to the log's subject kind."""
        if class_id in self.person_classes:
            return Kind.PERSON
        if class_id in self.vehicle_classes:
            return Kind.VEHICLE
        raise ConfigError(
            f"class id {class_id} is in neither perception.vehicle_classes nor "
            f"perception.person_classes, so its crossings could not be labelled"
        )


class VisionConfig(BaseModel):
    """Google Cloud Vision -- the 'eyes' of v2 (OCR + frame image analysis).

    Replaces v1's EasyOCR and the earlier plan's Claude-vision usage. Claude stays the
    agent brain; every image-understanding call goes through Cloud Vision.
    """

    # env-var name holding the path to the GCP service-account key JSON
    credentials_env: str = "GOOGLE_APPLICATION_CREDENTIALS"
    # bias the detector toward Nepali, Hindi, and English scripts
    language_hints: list[str] = ["ne", "hi", "en"]


class PlateConfig(BaseModel):
    enabled: bool = False   # current cameras cannot resolve plates (docs/feasibility.md)
    min_confidence: float = 0.5


class StorageConfig(BaseModel):
    db_path: Path


class AgentsConfig(BaseModel):
    analyst_model: str = "claude-opus-4-8"
    monitor_model: str = "claude-haiku-4-5"
    escalation_model: str = "claude-opus-4-8"
    monitor_interval_seconds: int = 300
    operating_hours: tuple[int, int] = (6, 21)


class ServiceConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080


class LoggingConfig(BaseModel):
    # The YAML key stays `json`, but the attribute is `as_json`: a field literally named
    # `json` shadows BaseModel.json and makes pydantic warn on every startup, which is
    # noise a service meant to run unattended for weeks does not need.
    model_config = ConfigDict(populate_by_name=True)

    level: str = "INFO"
    as_json: bool = Field(default=True, alias="json")


class Settings(BaseSettings):
    """Top-level settings. Instantiate with `Settings.load(path)`."""

    cameras: dict[str, CameraConfig]
    perception: PerceptionConfig
    vision: VisionConfig
    plate: PlateConfig
    storage: StorageConfig
    agents: AgentsConfig
    service: ServiceConfig
    logging: LoggingConfig

    # Where the config file was read from, so relative paths inside it can be resolved
    # against the project rather than against whatever directory the process started in.
    root: Path = Path(".")

    @classmethod
    def load(cls, config_path: Path, *, env_path: Path | None = None) -> Settings:
        """Read YAML, pull secrets in from .env, validate, and resolve relative paths.

        Layout assumed is the repo's own: `config/config.yaml` with `.env` one level up.
        Pass `env_path` to override that.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigError(
                f"no config file at {config_path}. "
                f"Copy config/config.example.yaml to config/config.yaml and edit it."
            )
        root = config_path.parent.parent
        load_env_file(env_path if env_path is not None else root / ".env")

        raw = yaml.safe_load(config_path.read_text())
        if not isinstance(raw, dict):
            raise ConfigError(f"{config_path} did not parse into a mapping of settings")
        raw["root"] = root

        try:
            settings = cls.model_validate(raw)
        except ValidationError as exc:
            raise ConfigError(f"{config_path} is not valid:\n{exc}") from exc

        # Resolve the two paths the process writes to or reads from. A systemd unit runs
        # with whatever WorkingDirectory it was given, and a relative db_path would
        # otherwise create a second, empty database somewhere surprising.
        settings.storage.db_path = settings._resolve(settings.storage.db_path)
        settings.perception.model_path = settings._resolve(settings.perception.model_path)
        return settings

    def _resolve(self, path: Path) -> Path:
        return path if path.is_absolute() else (self.root / path).resolve()

    def missing_secrets(self) -> list[str]:
        """Names of environment variables the config references but that are not set.

        Called at startup so an unattended process fails immediately with a list of what
        to fix, rather than an hour later on the first reconnect attempt.
        """
        missing = [
            stream.rtsp_env
            for camera in self.cameras.values()
            for stream in camera.streams.values()
            if not os.environ.get(stream.rtsp_env)
        ]
        return sorted(set(missing))

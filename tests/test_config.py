"""Tests for configuration loading.

Covers: the shipped example config actually parses; .env fills secrets in without
overriding the real environment; relative paths resolve against the project rather than
the working directory; missing secrets are reported at load time; and class ids map to
the right subject kind.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gcm_gatewatch.config import ConfigError, Settings, load_env_file
from gcm_gatewatch.perception.crossing import Kind

EXAMPLE = Path(__file__).resolve().parents[1] / "config" / "config.example.yaml"

MINIMAL = """
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


def _write(tmp_path: Path, yaml_text: str = MINIMAL, env_text: str | None = None) -> Path:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml_text)
    if env_text is not None:
        (tmp_path / ".env").write_text(env_text)
    return config_path


def test_example_config_parses() -> None:
    """The file people copy has to be valid, or the first run fails for everyone."""
    settings = Settings.load(EXAMPLE)
    assert set(settings.cameras) == {"entry", "exit"}
    assert settings.cameras["exit"].stream().resolution == (2880, 1620)


def test_missing_config_file_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config.example.yaml"):
        Settings.load(tmp_path / "nope.yaml")


def test_env_file_supplies_the_rtsp_url(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TEST_CAM_RTSP", raising=False)
    config_path = _write(tmp_path, env_text="TEST_CAM_RTSP=rtsp://user:pw@10.0.0.5:554/video1\n")

    settings = Settings.load(config_path)

    assert settings.missing_secrets() == []
    assert settings.cameras["exit"].stream().rtsp_url() == "rtsp://user:pw@10.0.0.5:554/video1"


def test_real_environment_wins_over_the_env_file(tmp_path: Path, monkeypatch) -> None:
    """A systemd or Docker variable is deliberate; a stray .env must not override it."""
    monkeypatch.setenv("TEST_CAM_RTSP", "rtsp://from-the-service")
    config_path = _write(tmp_path, env_text="TEST_CAM_RTSP=rtsp://from-the-file\n")

    settings = Settings.load(config_path)

    assert settings.cameras["exit"].stream().rtsp_url() == "rtsp://from-the-service"


def test_missing_secret_is_reported_not_raised(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("TEST_CAM_RTSP", raising=False)
    config_path = _write(tmp_path)

    settings = Settings.load(config_path)

    assert settings.missing_secrets() == ["TEST_CAM_RTSP"]
    with pytest.raises(ConfigError, match="TEST_CAM_RTSP"):
        settings.cameras["exit"].stream().rtsp_url()


def test_relative_paths_resolve_against_the_project(tmp_path: Path, monkeypatch) -> None:
    """A service started from / must not create its database in /."""
    config_path = _write(tmp_path)
    monkeypatch.chdir(tmp_path.parent)

    settings = Settings.load(config_path)

    assert settings.storage.db_path == (tmp_path / "gate.db").resolve()
    assert settings.perception.model_path == (tmp_path / "models" / "yolo.pt").resolve()


def test_class_ids_map_to_kinds(tmp_path: Path) -> None:
    settings = Settings.load(_write(tmp_path))
    perception = settings.perception

    assert perception.kind_for_class(0) is Kind.PERSON
    assert perception.kind_for_class(5) is Kind.VEHICLE
    assert perception.detect_classes == [0, 2, 5]
    with pytest.raises(ConfigError, match="neither"):
        perception.kind_for_class(99)


def test_a_class_cannot_be_both_kinds(tmp_path: Path) -> None:
    overlapping = MINIMAL.replace("person_classes: [0]", "person_classes: [0, 2]")
    with pytest.raises(ConfigError, match="cannot be logged as both"):
        Settings.load(_write(tmp_path, overlapping))


def test_load_env_file_reports_only_what_it_set(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALREADY_SET", "keep-me")
    monkeypatch.delenv("BRAND_NEW", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# a comment\n"
        "\n"
        "ALREADY_SET=overwrite-me\n"
        'BRAND_NEW="quoted value"\n'
        "not-a-pair\n"
    )

    applied = load_env_file(env_path)

    assert applied == {"BRAND_NEW": "quoted value"}
    assert os.environ["ALREADY_SET"] == "keep-me"


def test_load_env_file_tolerates_a_missing_file(tmp_path: Path) -> None:
    """Running from environment variables alone is a normal deployment, not an error."""
    assert load_env_file(tmp_path / "nothing-here") == {}

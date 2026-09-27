"""Tests for resolution-independent geometry and per-stream calibration.

Uses the three real gate-exit stream sizes to prove the core point: a zone calibrated on
a 16:9 stream transfers to another 16:9 stream, but NOT to the 2:1 crop -- which is
exactly why each stream is calibrated separately (v1's hardcoded y=500 ignored all of
this).
"""

from __future__ import annotations

import math

from gcm_gatewatch.perception.calibration import (
    aspect,
    aspect_matches,
    to_normalized,
    to_pixel,
)

DAY = (2880, 1620)      # 16:9
NIGHT = (1195, 671)     # 16:9 (different resolution, same field of view)
DAWN_CROP = (2461, 1228)  # ~2:1 (a different crop -> different field of view)


def test_pixel_normalized_round_trip() -> None:
    px = (1440.0, 810.0)
    norm = to_normalized(px, DAY)
    assert norm == (0.5, 0.5)
    assert to_pixel(norm, DAY) == px


def test_same_normalized_point_maps_across_same_aspect_resolutions() -> None:
    # the gate mouth at normalized (0.14, 0.72) lands proportionally on both 16:9 streams
    norm = (0.14, 0.72)
    assert to_pixel(norm, DAY) == (0.14 * 2880, 0.72 * 1620)
    assert to_pixel(norm, NIGHT) == (0.14 * 1195, 0.72 * 671)


def test_aspect_values() -> None:
    assert math.isclose(aspect(DAY), 16 / 9, rel_tol=1e-3)
    assert math.isclose(aspect(NIGHT), 16 / 9, rel_tol=2e-3)
    assert math.isclose(aspect(DAWN_CROP), 2.0, rel_tol=1e-2)


def test_same_fov_streams_share_calibration() -> None:
    assert aspect_matches(DAY, NIGHT) is True


def test_different_crop_needs_its_own_calibration() -> None:
    # 16:9 vs 2:1 -> a zone from one must not be reused on the other
    assert aspect_matches(DAY, DAWN_CROP) is False

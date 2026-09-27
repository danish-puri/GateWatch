"""Resolution-independent geometry and per-stream calibration.

v1 hardcoded a pixel line (``y=500``) that became meaningless the moment the same camera
delivered a different resolution. v2 stores every zone in **normalized 0..1** coordinates.

Key subtlety this module enforces: normalized coordinates transfer across resolutions of
the *same* field of view, but **not** across streams with a different crop / aspect
ratio. The real gate-exit frames prove it -- 2880x1620 and 1195x671 are both 16:9
(1.778), but 2461x1228 is 2:1 (2.004), a different crop. So each stream is calibrated
independently, and incoming frames are aspect-checked against their calibration at
runtime; an aspect mismatch means the normalized zone does not apply and the operator
must recalibrate that stream.

Pure Python, no deps -- unit-testable without a camera.
"""

from __future__ import annotations

Point = tuple[float, float]        # normalized (x, y) in 0..1
Resolution = tuple[int, int]       # (width, height) in pixels


def to_normalized(pixel: tuple[float, float], res: Resolution) -> Point:
    """Pixel coordinate -> normalized 0..1."""
    w, h = res
    return (pixel[0] / w, pixel[1] / h)


def to_pixel(norm: Point, res: Resolution) -> tuple[float, float]:
    """Normalized 0..1 -> pixel coordinate for a given resolution."""
    w, h = res
    return (norm[0] * w, norm[1] * h)


def aspect(res: Resolution) -> float:
    """Width / height."""
    w, h = res
    return w / h


def aspect_matches(a: Resolution, b: Resolution, *, tol: float = 0.02) -> bool:
    """True if two resolutions share a field of view (aspect within ``tol``).

    When this is False, a normalized zone calibrated on ``a`` must NOT be reused on ``b``
    -- the crop differs, so the same normalized point lands on a different part of the
    scene. Calibrate that stream separately.
    """
    return abs(aspect(a) - aspect(b)) <= tol * aspect(a)

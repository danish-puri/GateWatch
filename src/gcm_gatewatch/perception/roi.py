"""Gate ROI: crop the tiny gate region and run detection there at native resolution.

The gate mouth is a small slice of a 2880-wide frame. Downscaling the whole frame to fit
YOLO would shrink the far gate vehicles below the detector's reach; cropping the gate
region and running at native pixels keeps them detectable *and* is far cheaper than a
full-frame pass (there is no GPU at GCM).

This module is pure geometry (no OpenCV): compute the ROI pixel box from the normalized
gate zone, and map a detection made in the crop back to full-frame normalized coordinates
so `crossing.py` sees the same 0..1 space regardless of cropping.
"""

from __future__ import annotations

Point = tuple[float, float]        # normalized (x, y) in 0..1
Resolution = tuple[int, int]       # (width, height) in pixels
PixelBox = tuple[int, int, int, int]  # (x0, y0, x1, y1) in pixels


def gate_roi(
    gate_line: tuple[Point, Point],
    *,
    resolution: Resolution,
    margin: float = 0.06,
) -> PixelBox:
    """Padded, frame-clamped pixel box around the (normalized) gate line.

    `margin` is a fraction of the frame added on every side, so an approaching or
    departing vehicle is fully inside the crop, not just the instant it touches the line.
    """
    (ax, ay), (bx, by) = gate_line
    x0 = min(ax, bx) - margin
    x1 = max(ax, bx) + margin
    y0 = min(ay, by) - margin
    y1 = max(ay, by) + margin
    w, h = resolution
    px0 = max(0, int(x0 * w))
    py0 = max(0, int(y0 * h))
    px1 = min(w, int(x1 * w + 0.999))  # ceil, clamped to frame
    py1 = min(h, int(y1 * h + 0.999))
    return (px0, py0, px1, py1)


def crop(frame, box: PixelBox):
    """Return the ROI sub-image. Thin numpy slice; kept trivial on purpose."""
    x0, y0, x1, y1 = box
    return frame[y0:y1, x0:x1]


def local_to_global_norm(
    local_px: tuple[float, float],
    box: PixelBox,
    resolution: Resolution,
) -> Point:
    """A point measured in the crop (local pixels) -> full-frame normalized 0..1.

    This is what lets detection run on the crop while the tracker and gate counter keep
    working in the whole-frame normalized space.
    """
    x0, y0, _, _ = box
    w, h = resolution
    return ((x0 + local_px[0]) / w, (y0 + local_px[1]) / h)

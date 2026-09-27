"""Tests for the gate ROI crop geometry and crop->global coordinate mapping."""

from __future__ import annotations

from gcm_gatewatch.perception.roi import gate_roi, local_to_global_norm

DAY = (2880, 1620)
GATE = ((0.12, 0.55), (0.16, 0.90))


def test_roi_pads_and_pixel_maps() -> None:
    # x: 0.12..0.16 padded 0.06 -> 0.06..0.22 ; y: 0.55..0.90 padded -> 0.49..0.96
    box = gate_roi(GATE, resolution=DAY, margin=0.06)
    x0, y0, x1, y1 = box
    assert x0 == int(0.06 * 2880)          # 172
    assert y0 == int(0.49 * 1620)          # 793
    assert x1 == min(2880, int(0.22 * 2880 + 0.999))
    assert y1 == min(1620, int(0.96 * 1620 + 0.999))
    # the ROI is a small slice, not the whole frame
    assert (x1 - x0) < DAY[0] * 0.25
    assert (y1 - y0) < DAY[1] * 0.55


def test_roi_clamps_to_frame_edges() -> None:
    # a gate hard against the left/bottom with a big margin must not go negative or exceed
    box = gate_roi(((0.02, 0.90), (0.05, 0.99)), resolution=DAY, margin=0.10)
    x0, y0, x1, y1 = box
    assert x0 == 0
    assert y1 == 1620
    assert 0 <= x0 < x1 <= DAY[0]
    assert 0 <= y0 < y1 <= DAY[1]


def test_local_to_global_norm_reconstructs_full_frame_point() -> None:
    box = gate_roi(GATE, resolution=DAY, margin=0.06)
    x0, y0, _, _ = box
    # a detection at the crop's centre maps back to that pixel in the full frame
    local = (10.0, 20.0)
    gx, gy = local_to_global_norm(local, box, DAY)
    assert gx == (x0 + 10.0) / 2880
    assert gy == (y0 + 20.0) / 1620

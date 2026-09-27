"""Tests for the ANPR camera placement calculator.

Run with: python3 -m pytest development/v2/nepaliOCR/ -q
"""

from __future__ import annotations

import math

import pytest

from gcm_gatewatch.plate.camera_placement import (
    CHAR_PX_RELIABLE,
    CURRENT_OVERHEAD,
    ENTRY_RAMP,
    EXIT_THROAT,
    SENSOR_1_2_7,
    Placement,
    Sensor,
)


class TestFramingLaw:
    def test_character_pixels_depend_on_framing_not_distance(self):
        # The core claim: two cameras framing the same lane width get the same
        # character resolution even from very different distances, as long as
        # the view angles match.
        near = Placement("near", 3.0, 0.5, 3.0, 0.0)
        far = Placement("far", 9.0, 0.5, 3.0, 0.0)
        assert math.isclose(
            near.char_pixels(SENSOR_1_2_7), far.char_pixels(SENSOR_1_2_7)
        )

    def test_wider_framing_gives_smaller_characters(self):
        tight = Placement("tight", 5.0, 0.5, 3.0, 0.0)
        wide = Placement("wide", 5.0, 0.5, 20.0, 0.0)
        assert tight.char_pixels(SENSOR_1_2_7) > wide.char_pixels(SENSOR_1_2_7)

    def test_head_on_single_lane_clears_the_reliable_threshold(self):
        p = Placement("lane", 5.0, 0.5, 3.5, 0.0)
        assert p.char_pixels(SENSOR_1_2_7) >= CHAR_PX_RELIABLE


class TestAngles:
    def test_higher_mount_increases_tilt(self):
        low = Placement("low", 5.0, 1.0, 3.0, 0.0)
        high = Placement("high", 5.0, 4.0, 3.0, 0.0)
        assert high.tilt_deg > low.tilt_deg

    def test_obliquity_reduces_resolution(self):
        head_on = Placement("head on", 5.0, 0.5, 3.0, 0.0)
        oblique = Placement("oblique", 5.0, 0.5, 3.0, 40.0)
        assert oblique.char_pixels(SENSOR_1_2_7) < head_on.char_pixels(SENSOR_1_2_7)

    def test_tilt_uses_plate_height_not_ground(self):
        # A plate sits about half a metre up, so a 1.1 m mount looks down only
        # slightly, not as if aiming at the road surface.
        p = Placement("throat", 4.0, 1.1, 3.0, 0.0, plate_height_m=0.5)
        expected = math.degrees(math.atan2(1.1 - 0.5, 4.0))
        assert math.isclose(p.tilt_deg, expected)


class TestLens:
    def test_focal_length_scales_with_distance(self):
        near = Placement("near", 4.0, 1.0, 3.0, 0.0)
        far = Placement("far", 8.0, 1.0, 3.0, 0.0)
        assert math.isclose(
            far.focal_length_mm(SENSOR_1_2_7),
            2 * near.focal_length_mm(SENSOR_1_2_7),
        )

    def test_recommended_lenses_are_commonly_available(self):
        # A 5 to 6 m stand off on a single lane should want a lens in the
        # ordinary varifocal range, not an exotic telephoto.
        for placement in (EXIT_THROAT, ENTRY_RAMP):
            f = placement.focal_length_mm(SENSOR_1_2_7)
            assert 4.0 <= f <= 12.0


class TestGatePlacements:
    @pytest.mark.parametrize("placement", [EXIT_THROAT, ENTRY_RAMP])
    def test_proposed_placements_are_good(self, placement):
        assert placement.verdict(SENSOR_1_2_7) == "GOOD"
        assert placement.char_pixels(SENSOR_1_2_7) >= CHAR_PX_RELIABLE

    def test_current_overhead_is_poor(self):
        # Reframing the existing bullet onto the lane still fails, which is the
        # whole reason a new camera is needed.
        assert CURRENT_OVERHEAD.verdict(SENSOR_1_2_7) == "POOR"
        # And it reproduces the roughly 8 px measured off real footage.
        assert 7 <= CURRENT_OVERHEAD.char_pixels(SENSOR_1_2_7) <= 10

    def test_verdict_needs_both_resolution_and_angle(self):
        # Enough pixels but a wild angle is not GOOD.
        sharp_but_oblique = Placement("bad angle", 5.0, 0.5, 2.5, 45.0)
        assert sharp_but_oblique.char_pixels(SENSOR_1_2_7) >= CHAR_PX_RELIABLE
        assert sharp_but_oblique.verdict(SENSOR_1_2_7) != "GOOD"


class TestSensorChoice:
    def test_larger_sensor_needs_longer_lens_for_same_framing(self):
        big = Sensor("big", 2880, 7.18)
        p = EXIT_THROAT
        assert p.focal_length_mm(big) > p.focal_length_mm(SENSOR_1_2_7)

    def test_same_resolution_sensor_gives_same_character_pixels(self):
        # Character pixels depend on pixel count and framing, not sensor
        # physical size, so a 1/1.8 in unit at the same resolution reads the
        # same. Its advantage is light gathering, not sharpness.
        big = Sensor("big", 2880, 7.18)
        assert math.isclose(
            EXIT_THROAT.char_pixels(big), EXIT_THROAT.char_pixels(SENSOR_1_2_7)
        )

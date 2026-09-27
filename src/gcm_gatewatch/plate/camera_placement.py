"""Where to put a dedicated ANPR camera at GCM Gate 2.

The finding from resolution_budget.py is that the existing cameras fail on
plates not because plates are far but because each camera frames a 25 m wide
scene. Spreading 2880 pixels across 25 m leaves a plate character 8 px tall.
The fix is framing, not distance.

The governing relation makes this precise. To read a character of physical
height h you need it to land on some number of pixels p, and the horizontal
ground width the frame covers at the plate is

    W = image_width_px * h / p

Distance cancels out. W depends only on sensor resolution and how many pixels
you demand per character. So the design task is to frame the capture zone to a
width W narrow enough that p is comfortable, then choose the lens and stand off
distance that achieve that framing without an ugly view angle.

    p    = image_width_px * h / W          characters per pixel, from framing
    f_mm = distance_m * sensor_width_mm / W  lens to achieve that framing

This module turns a proposed placement into those numbers and flags the two
angles that wreck ANPR: looking down too steeply, and viewing the plate too
far off the axis of travel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

CHAR_HEIGHT_M = 0.07  # Devanagari glyph height on a Nepali plate
CHAR_PX_RELIABLE = 30.0
CHAR_PX_MARGINAL = 20.0

# ANPR view-angle limits. Beyond these, perspective foreshortening and glare
# start destroying characters regardless of resolution. Industry rule of thumb
# is to keep both the vertical (tilt) and horizontal (pan) angle under 30 deg,
# and ideally under 20.
MAX_ANGLE_GOOD = 20.0
MAX_ANGLE_OK = 30.0


@dataclass(frozen=True)
class Sensor:
    name: str
    width_px: int
    width_mm: float  # physical sensor width


# 1/2.7 inch is what the current UNV bullet uses, so a matching dedicated ANPR
# unit is the like for like reference. 1/1.8 inch is a common step up for
# purpose built ANPR cameras and gathers more light for night capture.
SENSOR_1_2_7 = Sensor("1/2.7in 5MP", 2880, 5.37)
SENSOR_1_1_8 = Sensor("1/1.8in", 2880, 7.18)


@dataclass(frozen=True)
class Placement:
    """A proposed camera position relative to the capture zone."""

    label: str
    distance_m: float  # camera to plate along the ground
    mount_height_m: float  # lens height above ground
    frame_width_m: float  # ground width the frame should cover (the lane)
    pan_deg: float  # horizontal angle off the axis of vehicle travel
    plate_height_m: float = 0.5  # typical plate centre height above ground

    @property
    def tilt_deg(self) -> float:
        """Downward view angle from lens to plate."""
        rise = self.mount_height_m - self.plate_height_m
        return math.degrees(math.atan2(rise, self.distance_m))

    def char_pixels(self, sensor: Sensor) -> float:
        """Character height in pixels, from framing alone."""
        head_on = sensor.width_px * CHAR_HEIGHT_M / self.frame_width_m
        # Oblique viewing shrinks the plate along whichever axis it is rotated
        # about. Tilt and pan compound.
        foreshorten = math.cos(math.radians(self.tilt_deg)) * math.cos(
            math.radians(self.pan_deg)
        )
        return head_on * foreshorten

    def focal_length_mm(self, sensor: Sensor) -> float:
        """Lens focal length that frames `frame_width_m` at `distance_m`."""
        return self.distance_m * sensor.width_mm / self.frame_width_m

    def verdict(self, sensor: Sensor) -> str:
        px = self.char_pixels(sensor)
        worst_angle = max(self.tilt_deg, abs(self.pan_deg))
        if px >= CHAR_PX_RELIABLE and worst_angle <= MAX_ANGLE_GOOD:
            return "GOOD"
        if px >= CHAR_PX_MARGINAL and worst_angle <= MAX_ANGLE_OK:
            return "OK"
        return "POOR"

    def report(self, sensor: Sensor) -> str:
        px = self.char_pixels(sensor)
        return "\n".join(
            (
                f"{self.label}",
                f"  sensor           {sensor.name}",
                f"  stand off         {self.distance_m:>5.1f} m",
                f"  mount height      {self.mount_height_m:>5.1f} m",
                f"  frame width       {self.frame_width_m:>5.1f} m  (the lane)",
                f"  tilt (down)       {self.tilt_deg:>5.1f} deg",
                f"  pan (off axis)    {abs(self.pan_deg):>5.1f} deg",
                f"  -> character size {px:>5.1f} px",
                f"  -> lens needed    {self.focal_length_mm(sensor):>5.1f} mm",
                f"  -> verdict        {self.verdict(sensor)}",
            )
        )


# --------------------------------------------------------------------------
# The two gate placements
# --------------------------------------------------------------------------

# Exit throat. The sliding gate opening is one vehicle wide and vehicles crawl
# through it. Mount low on the brick pillar, aimed back along the throat at the
# rear plate of a departing vehicle, nearly on axis.
EXIT_THROAT = Placement(
    label="Gate 2 EXIT  - low on the gate pillar, aimed along the throat",
    distance_m=4.0,
    mount_height_m=1.1,
    frame_width_m=3.0,
    pan_deg=10.0,
)

# Entry ramp. Vehicles slow to cross the sidewalk ramp into the gate. Mount low
# facing the approach so the front plate is captured as the vehicle noses in.
ENTRY_RAMP = Placement(
    label="Gate 2 ENTRY - low beside the ramp, facing approaching vehicles",
    distance_m=5.0,
    mount_height_m=1.2,
    frame_width_m=3.5,
    pan_deg=15.0,
)

# For contrast, the current overhead bullet reframed onto the same lane. Shows
# why the existing mount cannot be salvaged for ANPR by tuning alone.
CURRENT_OVERHEAD = Placement(
    label="Current overhead bullet - covering the whole gate scene",
    distance_m=12.0,
    mount_height_m=4.0,
    frame_width_m=20.0,
    pan_deg=25.0,
)


if __name__ == "__main__":
    for placement in (EXIT_THROAT, ENTRY_RAMP, CURRENT_OVERHEAD):
        print(placement.report(SENSOR_1_2_7))
        print()

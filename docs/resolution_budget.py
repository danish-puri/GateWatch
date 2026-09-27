"""Optical resolution budget for Nepali plate reading.

Answers the only question that matters before any model work. Given a camera
and a mounting geometry, is the plate physically legible? No amount of
fine-tuning recovers detail the sensor never captured, so this runs first.

The pinhole model gives focal length in pixels from the sensor width and the
horizontal field of view:

    f_px = (image_width / 2) / tan(hfov / 2)

A feature of real-world height h metres at distance d metres projects to

    pixels = f_px * h / d

Solving for d gives the furthest a plate can sit and still resolve a required
character height. Obliquity costs another factor of cos(theta) on the axis the
plate is rotated about, which at a high gate mount is the dominant loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Character height on a Nepali plate. The plate standard fixes overall plate
# size (45x11 cm single row, 30x18.5 cm two row) and the glyphs occupy roughly
# 7 cm of height in both layouts, which is why the distance limit below comes
# out the same for either layout.
CHAR_HEIGHT_M = 0.07

# Character height needed for reliable recognition. Devanagari needs more than
# Latin because the marks separating digits such as 3/4 and 6/7, and letters
# such as BA/VA, are finer than any Latin distinction.
CHAR_PX_RELIABLE = 30.0
CHAR_PX_MARGINAL = 20.0


@dataclass(frozen=True)
class Camera:
    """A camera's intrinsics, as far as this calculation needs them."""

    name: str
    image_width_px: int
    hfov_deg: float

    @property
    def focal_px(self) -> float:
        return (self.image_width_px / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    def pixels_at(self, feature_height_m: float, distance_m: float) -> float:
        """Pixel height of a feature viewed head on."""
        if distance_m <= 0:
            raise ValueError("distance must be positive")
        return self.focal_px * feature_height_m / distance_m

    def max_distance_m(self, required_char_px: float, obliquity_deg: float = 0.0) -> float:
        """Furthest a plate can be and still resolve `required_char_px`.

        `obliquity_deg` is the angle between the plate normal and the optical
        axis. A camera mounted high and looking down at a plate that faces the
        horizon is oblique by roughly the depression angle.
        """
        if required_char_px <= 0:
            raise ValueError("required_char_px must be positive")
        d = self.focal_px * CHAR_HEIGHT_M / required_char_px
        return d * math.cos(math.radians(obliquity_deg))

    def distance_of_plate(self, plate_px_width: float, plate_width_m: float) -> float:
        """Back-solve how far away an observed plate was.

        Useful for sanity checking a measurement taken off a real frame.
        """
        if plate_px_width <= 0:
            raise ValueError("plate_px_width must be positive")
        return self.focal_px * plate_width_m / plate_px_width


# The two cameras currently mounted at GCM Gate 1.
GCM_GATE_CAM = Camera(
    name="UNV IPC2225SE-DF40K-WL-I0 (4mm)",
    image_width_px=2880,
    hfov_deg=80.2,
)


def report(camera: Camera, obliquity_deg: float = 0.0) -> str:
    lines = [
        f"Camera        {camera.name}",
        f"Image width   {camera.image_width_px} px",
        f"Horizontal FOV{camera.hfov_deg:>6.1f} deg",
        f"Focal length  {camera.focal_px:>6.0f} px",
        f"Obliquity     {obliquity_deg:>6.1f} deg",
        "",
        "Furthest a plate can sit and still be readable",
    ]
    for label, required in (
        ("reliable", CHAR_PX_RELIABLE),
        ("marginal", CHAR_PX_MARGINAL),
    ):
        d = camera.max_distance_m(required, obliquity_deg)
        lines.append(f"  {label:<9} {required:>4.0f} px chars   {d:>5.1f} m")

    lines += ["", "Character height against distance"]
    for d in (2, 4, 6, 8, 10, 15, 20):
        px = camera.pixels_at(CHAR_HEIGHT_M, d) * math.cos(math.radians(obliquity_deg))
        if px >= CHAR_PX_RELIABLE - 1e-6:
            verdict = "readable"
        elif px >= CHAR_PX_MARGINAL - 1e-6:
            verdict = "marginal"
        else:
            verdict = "unreadable"
        lines.append(f"  {d:>3} m   {px:>5.1f} px   {verdict}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(report(GCM_GATE_CAM))
    print()
    print("Sanity check against a plate measured off the busy_day.jpg exit camera frame")
    # The Toyota Hiace two row plate measured 53 px wide. A two row rear plate
    # is 30 cm across.
    d = GCM_GATE_CAM.distance_of_plate(plate_px_width=53, plate_width_m=0.30)
    print(f"  53 px wide two row plate implies a range of {d:.1f} m")
    print(f"  giving {GCM_GATE_CAM.pixels_at(CHAR_HEIGHT_M, d):.0f} px characters")

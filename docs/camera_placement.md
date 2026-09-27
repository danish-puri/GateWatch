# Where to put an ANPR camera at GCM Gate 2

A siting guide for one dedicated plate reading camera per direction. Every
number here comes from `anpr_camera_placement.py`, which is derived from the
same optics as the feasibility study and validated against the real footage.

## The one idea that matters

The existing cameras cannot read plates, but not because plates are far away.
They fail because each one frames the entire gate scene, roughly 20 to 25 m
wide. Spreading 2880 pixels across 25 m leaves a plate character about 8 px
tall, which is unreadable. The calculator reproduces this: the current overhead
bullet comes out at 8.8 px, matching what I measured by hand off
`busy_day.jpg`.

The width the frame covers is what sets character size, and distance cancels
out of that relation:

    character_pixels = sensor_width_px * 0.07 m / frame_width_m

So the entire job is to frame one lane, not the whole scene. A 5 MP sensor
pointed at a single 3 m lane puts a plate character at roughly 65 px, which is
more than twice the reliable threshold. Same sensor, same plates, different
framing.

## The chokepoint at each gate

Gate 2 is one gate with two views. Every vehicle passes through a single
opening one vehicle wide, and slows to do it. That opening is the natural ANPR
chokepoint, and both new cameras aim at it from opposite sides.

### Exit throat

The sliding metal gate opening sits against a brick pillar. Vehicles crawl
through it one at a time, which is close to ideal for plate capture.

```
  PLAN VIEW, exit throat

     brick pillar
        |
        |####  <- sliding gate
        |   \
   [CAM]|    \        vehicle leaving  --->  ( rear plate )
    1.1m|     \                              |
   high |      \___ throat, ~3 m wide _______|
        |                                    |
        |  4 m stand off, aimed along travel, ~10 deg off axis
```

Mount the camera low on the pillar, about 1.1 m up, aimed back along the throat
so it looks at the rear plate of a departing vehicle almost straight on.

| Parameter | Value |
|---|---|
| Stand off distance | 4 m |
| Mount height | 1.1 m |
| Frame width | 3 m, the throat |
| Tilt down | 8.5 deg |
| Pan off travel axis | 10 deg |
| **Character height** | **65 px** |
| **Lens** | **~7 mm** |
| **Verdict** | **GOOD** |

### Entry ramp

Vehicles turn off the public road and cross the sidewalk on a ramp to enter,
slowing almost to a stop. Mount a second camera low beside that ramp, facing
the approaching vehicle so it captures the front plate as the vehicle noses in.

```
  PLAN VIEW, entry ramp

   public road  ===============================
                     |  vehicle turning in
                     v
                 [ ramp over sidewalk ]
                     |
                     |  ( front plate )  <--- vehicle approaching
                     |         ^
                 5 m |         |
                     |      [CAM] 1.2 m high, ~15 deg off axis
                 gate throat
```

| Parameter | Value |
|---|---|
| Stand off distance | 5 m |
| Mount height | 1.2 m |
| Frame width | 3.5 m |
| Tilt down | 8 deg |
| Pan off travel axis | 15 deg |
| **Character height** | **55 px** |
| **Lens** | **~8 mm** |
| **Verdict** | **GOOD** |

## The rules these placements follow

Four constraints drive every choice above. Break any one and the read rate
collapses even with a good camera.

**Frame one lane, not the scene.** This is the whole game. Aim for 2.5 to 3.5 m
of frame width. Wider than about 6 m and a 5 MP sensor drops below the reliable
threshold.

**Mount low.** Plate height, roughly 0.5 to 1.5 m, not gate height. A high
mount looks down at a plate that faces the horizon, and that tilt foreshortens
the characters. Keep the downward tilt under 30 deg, ideally under 20. The
placements above sit near 8 deg.

**Aim along the direction of travel.** Keep the horizontal angle off the
travel axis under 30 deg, ideally under 20. A plate viewed from the side is
compressed sideways and the characters merge. This is why the camera looks
along the throat rather than across it.

**Freeze the motion.** A plate smeared across even two pixels is unreadable, so
this is not optional. Use a fixed fast shutter, on the order of 1/500 to
1/1000 s, rather than auto. Vehicles crawl at the chokepoint, which helps, but
the shutter still has to be locked fast. A fast shutter needs light, which is
the next point.

## Night, and a wrinkle worth knowing

A fast shutter starves the sensor of light, so night capture needs dedicated
illumination aimed at the plate, not just the camera's scene floodlight.

Most ANPR systems lean on plates being retroreflective and light them with
infrared, which makes the plate glow white and the rest of the scene go dark,
a very clean capture. Nepal's newer plate guidelines require plates to be non
reflective, so that trick is weaker here than in most countries. Plan on
visible white light at the chokepoint, sized for a 1/500 s exposure, and prefer
a camera with a larger 1/1.8 in sensor for its light gathering. Note that a
larger sensor at the same 5 MP resolution reads the same sharpness, its
advantage is purely low light, so pay for it only if night traffic matters.

## Camera and lens shopping list

Per direction, so two of these.

- 5 MP or higher, global or fast rolling shutter, manual shutter speed control.
- A varifocal lens covering roughly 6 to 12 mm, so the framing can be dialled
  in on site. Both placements want about 7 to 8 mm, and a varifocal removes the
  guesswork of a fixed lens.
- A dedicated ANPR or LPR model if the budget allows, since those ship with the
  fast shutter, plate exposure logic, and IR or white light integrated. A plain
  bullet like the current UNV can work if its shutter can be locked fast.
- White light illuminator at the chokepoint for night, unless the camera has an
  adequate one built in.

## How this feeds the software

Framing one lane does more than fix resolution, it also constrains the
pipeline. Because the camera sees one vehicle in the throat at a time, roughly
frontal, the plate detector and the constrained decoder in `nepali_plate.py`
work in their best case rather than hunting for small plates across a wide
scene. Decode every frame while the vehicle is in the throat and fuse them,
which the throat's slow crawl makes easy since it yields many frames per
vehicle.

The existing overhead cameras keep doing what they are good at, detecting and
counting vehicles and timing entry and exit. The two ANPR cameras add the plate
number. They are complementary, not a replacement.

## Verify on site

The table numbers assume the geometry I read off the sample stills. Confirm
during installation:

- Measure the actual throat width and the real stand off. If the throat is
  wider than 3.5 m, either move closer or accept fewer pixels.
- Check the vehicle mix. A motorcycle plate sits lower and smaller than a bus
  plate, so set the mount height and tilt for the smallest plate you care
  about.
- Capture ten vehicles per direction after install and measure the character
  height in the saved frames. If it is below 30 px, the framing is too wide,
  which is a lens adjustment, not a new camera.

## Files

`anpr_camera_placement.py` turns any proposed placement into character pixels,
required lens, and view angles, and flags whether it will work. Edit the
`Placement` values at the bottom to test a different spot, or run it as is to
print the two recommendations and the current overhead camera for contrast.

`test_anpr_placement.py` covers the calculator, including the check that the
current overhead mount reproduces the roughly 8 px measured off real footage.

# Detection & counting tuning — from the real gate-exit footage

Grounded in four real frames from the exit camera, not assumptions. The frames themselves are kept private because people and plates are identifiable in them.

| Frame | Resolution | Time | What it shows |
|---|---|---|---|
| `busy_day.jpg` | 2880×1620 | 16:17, bright sun | ~10–12 buses parked in rows + buses maneuvering in foreground; dense, heavy occlusion, hard shadows |
| `packed_parcking.jpg` | 2880×1620 | 16:15, bright sun | row of buses parked nose-out along the back; van + bus in foreground; guard on court |
| `night.png` | 1195×671 | 00:06, dark | near-empty; one van + one tempo; low-res, floodlight glare, strong vignette |
| `GateExit_empty.png` | 2461×1228 | 04:46, dawn | empty court, clean ground — the calibration frame |

## Scene reality (why v1's model fails here)

The camera watches a **forecourt/depot**, and the **gate is a small sliding gate at the far bottom-left**, oblique to the camera. Vehicles enter through that gate and then **park** across the yard. Consequences:

- A full-width horizontal line across the middle (v1's `y=500`) sits over the *parking area*, not the gate. It counts parked-bus jitter, not real entries/exits.
- The lens is **fisheye** — straight lines bow; geometry done in raw pixels is wrong near the edges (and the gate is right at a distorted edge).
- The camera appears at **three resolutions**. Any pixel-hardcoded geometry (v1's `y=500`) breaks. Coordinates must be **normalized (0–1)** or calibrated per stream.

## What to do differently from v1

1. **Count at the gate, not across the yard.** Place a short **oriented line / small polygon zone over the gate mouth** (bottom-left), with an orientation vector: a track crossing gate→yard is an **entry**, yard→gate is an **exit**. `supervision.LineZone` with a short, angled segment (not full width) or a `PolygonZone` at the gate.
2. **Ignore parked vehicles.** A stationary-track / dwell filter: only tracks with real displacement *through the gate zone* count. The 10+ buses sitting in the lot must never increment a counter. (This is the biggest source of error v1 has here.)
3. **One gate camera can do both IN and OUT** by travel direction through the gate zone. The separate entry/exit cameras become redundancy/confirmation, not separate IN-only / OUT-only counters.
4. **ROI-crop the gate and run detection there at high resolution.** The gate is a tiny fraction of a 2880-wide frame; a downscaled full-frame pass will miss small vehicles at the gate. Run a **cheap coarse full-frame pass** (motion/occupancy) plus a **fine pass on the cropped gate ROI** at native res.
5. **Motion-gate the detector (CPU win, no GPU here).** Use the empty frames as a background model; only run YOLO on the gate ROI when background subtraction shows motion there. Saves most of the compute on an idle yard.
6. **Normalized geometry + per-stream calibration.** Store the gate zone as relative coordinates and calibrate per resolution, since day/dawn/night arrive at different sizes. Never hardcode pixels like v1.
7. **Optional dewarp.** Undistort with the lens model before geometry for cleaner crossing math. Adds calibration burden and CPU; defer — a polygon drawn in distorted space is good enough to start.
8. **Nepali vehicle classes.** COCO covers bus/car/truck/motorcycle but not tempo/auto/micro; expect bus↔truck and van↔car confusion. Plan a light fine-tune on local vehicles once the pipeline runs.
9. **Separate day/night handling** (see per-image params).

## Best parameters per image

Common baseline: YOLO (v8/v11), vehicle classes only (car, motorcycle, bus, truck), ByteTrack for IDs, geometry only at the gate zone.

### `busy_day.jpg` / `packed_parcking.jpg` — bright day, 2880×1620, dense
- `imgsz`: 1280–1536 (small far vehicles at the gate need the resolution).
- `conf`: 0.35–0.40.
- **NMS `iou`: 0.7** (high) — buses overlap heavily; a low IoU threshold suppresses adjacent buses. Keep them separate.
- Gate-ROI crop run at native res; coarse full-frame at 1280.
- Strong shadows + occlusion → lean on the tracker (larger `track_buffer`, e.g. 60–90 frames) so an occluded transiting vehicle keeps its ID.
- `frame_stride`: 3–5 for the yard, but sample the **gate ROI every frame** (vehicles move fast there).
- Must apply the stationary/parked filter — otherwise the parked fleet dominates.

### `GateExit_empty.png` — dawn, empty, 2461×1228, clean
- This is the **calibration frame**, not a detection target. Use it to draw the gate zone + the parking exclusion polygon and to seed the background model.
- Detection params irrelevant (nothing to detect); `conf` default 0.4.

### `night.png` — midnight, 1195×671, dark + glare + low-res
- `conf`: 0.20–0.25 (weak appearance) — but pair with motion gating or false positives spike.
- Pre-process: CLAHE / gamma lift; **mask the two floodlight glare blobs** — the right one sits near the gate/building, exactly where a vehicle at the gate would appear.
- Appearance is unreliable at night → **background subtraction / motion is the primary signal**; YOLO confirms.
- `imgsz`: run at 960–1280 (upscaling the small frame recovers no real detail, but helps the detector's stride).
- Larger `track_buffer`, lower `conf`; expect degraded accuracy — this is a camera/optics limit, consistent with `feasibility.md`.

## Geometry, fisheye, and vehicle classes (implemented / planned)

### Normalized geometry, calibrated per stream (done)
All zones are stored in **normalized 0..1** coordinates (`config.py` `GateZone`), not pixels
— v1's `y=500` broke the moment the stream resolution changed. Normalized coords transfer
across resolutions of the *same* field of view, but **not** across a different crop: the
real frames are 2880×1620 and 1195×671 (both 16:9 = 1.778) versus 2461×1228 (2:1 = 2.004).
So each camera holds a `streams` map, and **every stream is calibrated separately**
(`StreamConfig`: `resolution` + its own `gate`). At runtime, a live frame's aspect is
checked against the stream's calibrated `resolution` (`perception/calibration.aspect_matches`);
a mismatch means the wrong stream/crop and the zone must not be applied. Pixel↔normalized
conversion: `perception/calibration.to_normalized` / `to_pixel`.

### Fisheye (Stage 1 now, Stage 2 later)
The lens is wide-angle and worst at the edges — where the gate mouth sits. **Stage 1**
(current): draw the gate and test crossings **in distorted image space**. The gate zone
and the vehicle centroids live in the same distorted space, so the crossing geometry is
self-consistent and correct with no undistortion and no calibration. **Stage 2** (later,
`perception/dewarp.py`): calibrate the fisheye lens (OpenCV `K` + `D`) and undistort frames
first — needed only if metric distance/speed or a top-down homography is wanted. Toggle
with `perception.dewarp_enabled`.

### Nepali vehicle classes (planned fine-tune)
COCO has car/bus/truck/motorcycle but **not** tempo, auto-rickshaw, or microbus, and it
confuses van↔car and bus↔truck. Plan: collect and label local frames from this exact
camera, fine-tune the detector to add the missing classes, swap `perception.model_path`,
and map the new class ids to labels via `perception.class_labels`. Until then, run COCO
classes and accept the known confusions.

## How this maps to the code

- Gate zone + per-stream calibration → `config.py` (`GateZone` / `StreamConfig`), consumed by `perception/crossing.py` via `perception/calibration.py`.
- Stationary/dwell filter → `perception/motion.py`, gating `perception/crossing.py`.
- ROI-crop + coarse/fine passes → `perception/detector.py`.
- Fisheye undistort (Stage 2) → `perception/dewarp.py`.
- Day/night parameter sets → a profile switch keyed on hour or measured luminance in `config.yaml`.

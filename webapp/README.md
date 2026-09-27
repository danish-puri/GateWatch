# GateWatch dashboard

The Gate 1 detection dashboard. Plain HTML, CSS and JavaScript, no build step and no
dependencies, so it can be dropped next to the service and served as static files.

Design source is the Paper file `neoBird · Gate Detection`. Tokens, spacing and the
motion curves in `styles.css` mirror that file directly.

## Running it

```
cd webapp
python3 -m http.server 8777
```

Then open http://127.0.0.1:8777/

The camera panel plays a real 60 second clip from the exit camera with real YOLO v26
boxes over it. Nothing is faked: crossings, counts and the tally all come from the
detections, so the dashboard moves because vehicles moved. Add `?mode=live` to pull the
surrounding figures from the service instead of the seeded ones.

## Building the demo clip

The dashboard plays a short clip from the exit camera with real YOLO boxes drawn over it.
The detections file `assets/gate-exit.detections.json` is committed. The video itself is
not, because people and number plates are identifiable in real CCTV footage. Without it the
dashboard still loads, it just has nothing to play.

To run it against your own footage, build a matching clip and detections file with

```
uv run python tools/build_clip.py \
    --source path/to/footage.mp4 --start 140 --duration 60 \
    --weights models/yolo26n.pt
```

That transcodes the segment to 960px h264 and runs detection with supervision's
ByteTrack, the same tracker `perception/tracker.py` uses. Boxes are stored as
percentages of frame size so the overlay fits the video element at any width.

The clip I used was 60 s long, with 750 detected frames and 175 tracks.

## Wiring it to the real service

The dashboard is not connected to `service/api.py` yet. The frontend expects three
endpoints, and `DataSource.apply()` in `app.js` is the only place that needs to agree with
them.

`GET /stats`

```json
{
  "inside_now": 23,
  "total_today": 209,
  "by_class": { "moto": 96, "bus": 42, "van": 38, "cycle": 27, "truck": 6 },
  "by_hour": [ { "hour": "06", "v": 18 }, { "hour": "08", "v": 52 } ]
}
```

`GET /crossings?limit=6` returns rows straight off the `crossings` table, newest first.
`vehicle_type` may use COCO names (`car`, `motorcycle`, `bicycle`), `normaliseClass()`
maps them onto the five the gate cares about.

```json
[ { "track_id": 4471, "vehicle_type": "car", "direction": "out", "ts": 1784500841, "confidence": 0.97 } ]
```

`GET /healthz`

```json
{ "status": "ok", "last_write_seconds": 2 }
```

Two of these need work on the backend before live mode is useful:

- **`last_write_seconds`** does not exist yet. The liveness dot and the stale state are
  built on it and there is no substitute for it.
- **`confidence` is not persisted.** The schema has no column for it, so `cf` renders as
  an em dash. `OcrResult.confidence` exists on the object but is dropped before the write.

Once the API is real, serve this directory from FastAPI with

```python
app.mount("/", StaticFiles(directory="webapp", html=True), name="webapp")
```

mounted last, after the JSON routes, so it does not shadow them.

## Notes on the design

**A detection is a capsule.** The overlay boxes, the tally bars, the wordmark and the
ambient background are all the same shape at different sizes, which is why the motion
reads as one system rather than four effects.

**Four motions, defined once in `styles.css`:** `det-in` springs a box out of the object's
centroid, `det-flash` plus `emit` throws the count into the crossings list, the tally
capsules transition height from a shared centre line, and `drift-a/b/c` keep the ground
moving on a 48 second loop.

**The gate line is a segment, not an infinite line.** Testing a point against an
infinite line counted parked buses at the far end of the yard every time their boxes
wobbled. Crossings are now a segment intersection between the vehicle's movement step
and the gate, which is what `supervision`'s LineZone does on the backend.

**The anchor is the centroid, not the bottom edge.** Vehicles pass directly under this
camera, so their boxes get clipped by the frame edge and the foot point pins to y=100,
hiding the crossing entirely. The centroid does not saturate.

**Class comes from a vote, not the last frame.** The same bus reads `bus` on one frame
and `truck` on the next while the track id stays stable, so each track's class is decided
by confidence-weighted vote across its whole life.

**Crossings are debounced by 2 seconds per track.** Without it a box jittering on the
line fires three counts in a row, which is exactly what track #44 did.

**People are drawn but never counted.** They are detected and shown in neutral gray
because the counted unit at this gate is vehicles, and rendering people in a class colour
implied they were part of the tally.

**A failed poll never blanks the screen.** Content freezes and the status pill goes stale
rather than showing zeros, because an empty dashboard reads as "no traffic" when it
actually means "no data".

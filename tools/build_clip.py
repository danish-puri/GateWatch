"""Build the webapp's demo clip: transcode a segment, then detect and track it.

The dashboard needs two things to show real detection over real footage without
a live pipeline: a web-playable video, and the boxes that belong to it. This
produces both, so the demo is genuine output from the same model the service
runs rather than hand-placed rectangles.

    python tools/build_clip.py \
        --source path/to/footage.mp4 \
        --start 140 --duration 60 \
        --weights models/yolo26n.pt

Boxes are written as percentages of frame size, so the overlay tracks the video
element at any width. Coordinates never need to know the render size.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# COCO ids -> the five classes this gate actually counts, plus people.
# People are detected so the overlay is honest about what the model sees, but
# they are never part of the vehicle tally.
COCO_TO_CLASS = {
    0: "person",
    1: "cycle",
    2: "van",     # COCO "car" covers the vans and hatchbacks at this gate
    3: "moto",
    5: "bus",
    7: "truck",
}

DEFAULT_WEIGHTS = Path.home() / "yolo26n.pt"


def transcode(source: Path, out: Path, start: float, duration: float, width: int, crf: int) -> None:
    """Cut the segment and re-encode it small enough to ship and stream."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", str(start), "-t", str(duration), "-i", str(source),
        "-an",                                  # no audio, this is CCTV
        "-vf", f"scale={width}:-2,fps=25",
        "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
        "-crf", str(crf), "-preset", "slow",
        "-movflags", "+faststart",              # so it starts before it finishes downloading
        str(out),
    ]
    subprocess.run(cmd, check=True)


def detect(clip: Path, weights: Path, stride: int, conf: float, imgsz: int) -> dict:
    """Run detection + tracking over the clip and collect per-frame boxes.

    Uses supervision's ByteTrack rather than the tracker built into ultralytics,
    matching perception/tracker.py. That is also what the service already depends
    on, so this needs no extra packages.
    """
    import cv2
    import supervision as sv
    from ultralytics import YOLO

    model = YOLO(str(weights))
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {clip}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames: list[dict] = []
    seen_tracks: set[int] = set()
    keep = sorted(COCO_TO_CLASS)

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride:
            idx += 1
            continue

        h, w = frame.shape[:2]
        result = model.predict(frame, classes=keep, conf=conf, imgsz=imgsz, verbose=False)[0]

        # Tracking gives each vehicle a stable id, which is what makes a crossing
        # one event instead of one event per frame.
        dets = tracker.update_with_detections(sv.Detections.from_ultralytics(result))

        out = []
        for (x1, y1, x2, y2), cls_id, cf, tid in zip(
            dets.xyxy, dets.class_id, dets.confidence, dets.tracker_id
        ):
            name = COCO_TO_CLASS.get(int(cls_id))
            if name is None:
                continue
            seen_tracks.add(int(tid))
            out.append({
                "c": name,
                "i": int(tid),
                "p": round(float(cf), 2),
                # percentages of frame size, rounded to keep the JSON small
                "x": round(float(x1) / w * 100, 2),
                "y": round(float(y1) / h * 100, 2),
                "w": round(float(x2 - x1) / w * 100, 2),
                "h": round(float(y2 - y1) / h * 100, 2),
            })

        frames.append({"t": round(idx / fps, 2), "d": out})
        if len(frames) % 50 == 0:
            print(f"  t={idx / fps:6.2f}s  {len(out)} detections", file=sys.stderr)
        idx += 1

    cap.release()
    return {
        "fps": fps / stride,
        "duration": frames[-1]["t"] if frames else 0,
        "tracks": len(seen_tracks),
        "frames": frames,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True, help="full-resolution CCTV clip")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "webapp" / "assets")
    ap.add_argument("--name", default="gate-exit")
    ap.add_argument("--start", type=float, default=140.0)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--crf", type=int, default=30)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--stride", type=int, default=2, help="detect every Nth frame")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--skip-transcode", action="store_true")
    args = ap.parse_args()

    clip = args.out_dir / f"{args.name}.mp4"
    meta = args.out_dir / f"{args.name}.detections.json"

    if not args.skip_transcode:
        print(f"transcoding {args.source.name} [{args.start}s +{args.duration}s] -> {clip}", file=sys.stderr)
        transcode(args.source, clip, args.start, args.duration, args.width, args.crf)

    print(f"detecting with {args.weights.name} (stride {args.stride})", file=sys.stderr)
    data = detect(clip, args.weights.expanduser(), args.stride, args.conf, args.imgsz)

    meta.write_text(json.dumps(data, separators=(",", ":")))
    print(
        f"wrote {meta.name}: {len(data['frames'])} frames, "
        f"{data['tracks']} tracks, {meta.stat().st_size / 1024:.0f} KB",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

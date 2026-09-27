"""Turn the raw gate videos into a label-ready detection dataset.

The slow part of measuring mAP is drawing ground-truth boxes by hand. The trick
that removes most of that work is model-assisted labelling: run a strong
detector once to produce *candidate* boxes, then a human only corrects them
instead of drawing from scratch. Roboflow, CVAT, and Label Studio all import
this YOLO format directly and let you fix the boxes in a browser.

    Roboflow: model-assisted labelling / Autodistill.
    https://docs.autodistill.com/

This script:
  1. samples frames from video/*.mp4 at a chosen rate,
  2. pre-labels each with a YOLO model (default yolo11l, the largest local one,
     because a heavier model makes fewer mistakes for a human to fix),
  3. writes an Ultralytics-format dataset (images/, labels/, data.yaml) that
     `model.val()` and the supervision library both read without conversion.

The pre-labels are a starting point, not ground truth. Correct them before
reporting any accuracy number, or you are only measuring the pre-labeller
against itself.

Usage:
    python3 prepare_dataset.py --fps 0.5 --limit 5        # quick smoke test
    python3 prepare_dataset.py --fps 2 --model yolo11l.pt # full sampling
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

# COCO ids we care about at the gate, mapped to the project's vehicle names.
# tempo/auto/microbus do not exist in COCO and need a fine-tuned model; they are
# documented here so the class list stays stable when that model arrives.
COCO_VEHICLE_CLASSES: dict[int, str] = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

REPO_ROOT = Path(__file__).resolve().parents[3]
VIDEO_DIR = REPO_ROOT / "video"
DEFAULT_MODEL = REPO_ROOT / "development/detect_store_vehicle_info_v1/yolo11l.pt"


@dataclass
class Sampled:
    video: str
    frame_index: int
    time_s: float
    n_prelabels: int


def _class_list() -> list[str]:
    """Contiguous 0..N names for data.yaml, in a fixed order."""
    return [COCO_VEHICLE_CLASSES[k] for k in sorted(COCO_VEHICLE_CLASSES)]


def _remap() -> dict[int, int]:
    """COCO id -> contiguous dataset id (0..N), matching _class_list order."""
    names = _class_list()
    return {
        coco_id: names.index(name) for coco_id, name in COCO_VEHICLE_CLASSES.items()
    }


def prepare(
    out_dir: Path,
    *,
    fps: float,
    model_path: Path,
    imgsz: int,
    confidence: float,
    limit: int | None,
) -> list[Sampled]:
    import cv2  # deferred: heavy, and keeps --help fast
    from ultralytics import YOLO

    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    remap = _remap()
    wanted = set(COCO_VEHICLE_CLASSES)
    sampled: list[Sampled] = []

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no videos found in {VIDEO_DIR}")

    for video in videos:
        cap = cv2.VideoCapture(str(video))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        step = max(1, round(src_fps / fps))
        # Seek to each target frame rather than decoding every frame in between;
        # on these 2880-wide clips that is the difference between seconds and
        # minutes.
        targets = list(range(0, n_frames, step))
        if limit:
            targets = targets[:limit]
        for idx in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            stem = f"{video.stem}_f{idx:06d}"
            cv2.imwrite(str(images_dir / f"{stem}.jpg"), frame)
            res = model(frame, imgsz=imgsz, classes=sorted(wanted), verbose=False)[0]
            lines: list[str] = []
            for cls_id, xywh, conf in zip(
                res.boxes.cls.tolist(),
                res.boxes.xywhn.tolist(),  # normalized cx,cy,w,h -- YOLO label format
                res.boxes.conf.tolist(),
            ):
                if conf < confidence:
                    continue
                ds_id = remap[int(cls_id)]
                cx, cy, bw, bh = xywh
                lines.append(f"{ds_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (labels_dir / f"{stem}.txt").write_text("\n".join(lines))
            sampled.append(Sampled(video.name, idx, idx / src_fps, len(lines)))
        cap.release()

    _write_data_yaml(out_dir)
    return sampled


def _write_data_yaml(out_dir: Path) -> None:
    names = _class_list()
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(names))
    (out_dir / "data.yaml").write_text(
        "# Ultralytics dataset. Point train/val at the corrected split.\n"
        f"path: {out_dir}\n"
        "train: images\n"
        "val: images\n"
        f"nc: {len(names)}\n"
        "names:\n"
        f"{names_block}\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("dataset"))
    ap.add_argument("--fps", type=float, default=0.5, help="frames sampled per second")
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=None, help="max frames per video")
    args = ap.parse_args()

    sampled = prepare(
        args.out,
        fps=args.fps,
        model_path=args.model,
        imgsz=args.imgsz,
        confidence=args.conf,
        limit=args.limit,
    )
    total_labels = sum(s.n_prelabels for s in sampled)
    print(f"sampled {len(sampled)} frames -> {args.out}")
    print(f"pre-labelled {total_labels} vehicle boxes (candidates, correct before use)")
    for s in sampled[:10]:
        print(f"  {s.video}  t={s.time_s:6.1f}s  {s.n_prelabels} boxes")
    if len(sampled) > 10:
        print(f"  ... and {len(sampled) - 10} more")
    print(f"\nNext: import {args.out} into CVAT/Roboflow, correct the boxes, then")
    print("run evaluate.py (or `yolo val`) against the corrected labels.")


if __name__ == "__main__":
    main()

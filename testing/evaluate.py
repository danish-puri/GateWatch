"""Score a detector against corrected ground-truth labels: mAP, IoU, P, R.

Two ways to get the same numbers, and you should agree them against each other
at least once:

  1. The Ultralytics one-liner, which is what you will use day to day:

        from ultralytics import YOLO
        metrics = YOLO("model.pt").val(data="dataset/data.yaml")
        print(metrics.box.map, metrics.box.map50)   # mAP50-95, mAP50
        # docs.ultralytics.com/modes/val -- results.box.map / .map50 / .map75

  2. This script, which loads the same YOLO-format labels and computes the
     metrics with the readable reference in detection_metrics.py. Use it to
     understand or audit a number, or when you want per-frame control that
     `val` does not give.

Both read the exact dataset that prepare_dataset.py writes. Point `--labels` at
the *corrected* labels, not the pre-labels, or you are grading the pre-labeller
against itself (which is what the self-check below deliberately does to prove
the harness works).

Usage:
    python3 evaluate.py --model yolo11n.pt --dataset DATASET_DIR
    python3 evaluate.py --model yolo11n.pt --dataset DATASET_DIR --imgsz 1280
"""

from __future__ import annotations

import argparse
from pathlib import Path

from detection_metrics import Box, counting_error, evaluate_dataset


def _load_yolo_labels(label_path: Path, names: list[str], w: int, h: int) -> list[Box]:
    """Read one YOLO .txt label file into pixel-space Boxes."""
    if not label_path.exists():
        return []
    boxes: list[Box] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id, cx, cy, bw, bh = (
            int(parts[0]),
            *(float(p) for p in parts[1:5]),
        )
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        label = names[cls_id] if cls_id < len(names) else str(cls_id)
        boxes.append(Box(x1, y1, x2, y2, label))
    return boxes


def _read_names(data_yaml: Path) -> list[str]:
    """Parse the ordered class names out of data.yaml without a yaml dep."""
    names: dict[int, str] = {}
    in_names = False
    for line in data_yaml.read_text().splitlines():
        if line.strip().startswith("names:"):
            in_names = True
            continue
        if in_names:
            s = line.strip()
            if not s or ":" not in s or not s[0].isdigit():
                if s and not s[0].isdigit():
                    break
                continue
            idx, name = s.split(":", 1)
            names[int(idx)] = name.strip()
    return [names[i] for i in sorted(names)]


def run(model_path: str, dataset: Path, imgsz: int, confidence: float):
    import cv2
    from ultralytics import YOLO

    names = _read_names(dataset / "data.yaml")
    name_to_id = {n: i for i, n in enumerate(names)}
    model = YOLO(model_path)
    images = sorted((dataset / "images").glob("*.jpg"))
    if not images:
        raise SystemExit(f"no images in {dataset/'images'}")

    per_frame: list[tuple[list[Box], list[Box]]] = []
    pred_counts: dict[str, int] = {}
    gt_counts: dict[str, int] = {}

    for img_path in images:
        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]
        gts = _load_yolo_labels(
            dataset / "labels" / f"{img_path.stem}.txt", names, w, h
        )
        res = model(
            frame,
            imgsz=imgsz,
            classes=sorted(name_to_id[n] for n in names if n in name_to_id),
            verbose=False,
        )[0]
        preds: list[Box] = []
        for cls_id, xyxy, conf in zip(
            res.boxes.cls.tolist(),
            res.boxes.xyxy.tolist(),
            res.boxes.conf.tolist(),
        ):
            if conf < confidence:
                continue
            label = model.names[int(cls_id)]
            preds.append(Box(*xyxy, label=label, score=conf))
        per_frame.append((preds, gts))
        for b in preds:
            pred_counts[b.label] = pred_counts.get(b.label, 0) + 1
        for b in gts:
            gt_counts[b.label] = gt_counts.get(b.label, 0) + 1

    return evaluate_dataset(per_frame), counting_error(pred_counts, gt_counts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    ev, ce = run(args.model, args.dataset, args.imgsz, args.conf)
    print("Detection accuracy (per-frame matched, pooled)")
    print(f"  mAP@50       {ev.map50:.3f}")
    print(f"  mAP@75       {ev.map75:.3f}")
    print(f"  mAP@50-95    {ev.map50_95:.3f}   (COCO primary metric)")
    print(f"  precision    {ev.precision:.3f}")
    print(f"  recall       {ev.recall:.3f}")
    print(f"  F1           {ev.f1:.3f}")
    print("  per-class AP@50")
    for cls, ap_val in sorted(ev.per_class_ap50.items()):
        print(f"    {cls:12} {ap_val:.3f}")
    print("\nBox-count error (sanity signal, not a substitute for gate counting)")
    print(f"  predicted {ce.total_pred} vs ground truth {ce.total_true}")
    print(f"  MAE {ce.mae:.2f}   total error {ce.percent_error:.1f}%")


if __name__ == "__main__":
    main()

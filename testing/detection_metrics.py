"""Detection accuracy metrics from scratch: IoU, precision, recall, AP, mAP.

This exists so the numbers in the test report are ones we can explain, not a
black box. It is pure stdlib and matches the standard definitions used by COCO
and PASCAL VOC, so a result here lines up with what Ultralytics `val` or the
supervision library would report. For a real run you would usually call one of
those one liners (see TEST_PLAN.md); this module is the readable reference they
agree with, and it doubles as the thing our unit tests check against.

Definitions, with sources.

IoU, intersection over union, measures how well a predicted box overlaps a
ground truth box: area of overlap divided by area of union. A prediction counts
as correct only if its IoU with a ground truth box clears a threshold.
    Everingham et al., The PASCAL VOC Challenge, IJCV 2010.

Precision is TP / (TP + FP), the fraction of detections that were real. Recall
is TP / (TP + FN), the fraction of real objects that were found.

Average Precision, AP, is the area under the precision-recall curve for one
class. mAP is AP averaged over classes.
    Ultralytics glossary, Mean Average Precision.
    https://www.ultralytics.com/glossary/mean-average-precision-map

mAP@0.5 uses a single IoU threshold of 0.5 and is lenient. mAP@[0.5:0.95],
the COCO primary metric, averages mAP over ten IoU thresholds from 0.5 to 0.95
in steps of 0.05, rewarding tight localisation.
    Lin et al., Microsoft COCO, ECCV 2014, and the COCO evaluation protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Box:
    """An axis aligned box in pixel coordinates, x1 y1 top left."""

    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "vehicle"
    score: float = 1.0  # detections carry a confidence; ground truth is 1.0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes, in [0, 1]."""
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


@dataclass
class PRCurve:
    """A precision-recall curve for one class at one IoU threshold."""

    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    tp: int = 0
    fp: int = 0
    n_ground_truth: int = 0

    @property
    def fn(self) -> int:
        return self.n_ground_truth - self.tp

    @property
    def final_precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def final_recall(self) -> float:
        return self.tp / self.n_ground_truth if self.n_ground_truth else 0.0

    def f1(self) -> float:
        p, r = self.final_precision, self.final_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _match_one_class(
    preds: list[Box], gts: list[Box], iou_threshold: float
) -> PRCurve:
    """Greedily match detections to ground truth for a single class.

    Detections are taken in descending confidence. Each is matched to the
    highest IoU unclaimed ground truth box that clears the threshold, which is
    the COCO and VOC matching rule. A second detection of an already matched
    object is a false positive, which is how duplicate boxes are penalised.
    """
    curve = PRCurve(n_ground_truth=len(gts))
    claimed = [False] * len(gts)
    tp_cum = fp_cum = 0

    for pred in sorted(preds, key=lambda b: b.score, reverse=True):
        best_iou, best_j = iou_threshold, -1
        for j, gt in enumerate(gts):
            if claimed[j]:
                continue
            ov = iou(pred, gt)
            if ov >= best_iou:
                best_iou, best_j = ov, j
        if best_j >= 0:
            claimed[best_j] = True
            tp_cum += 1
        else:
            fp_cum += 1
        curve.precision.append(tp_cum / (tp_cum + fp_cum))
        curve.recall.append(tp_cum / len(gts) if gts else 0.0)

    curve.tp, curve.fp = tp_cum, fp_cum
    return curve


def average_precision(curve: PRCurve) -> float:
    """Area under the precision-recall curve, all points interpolation.

    The precision envelope is made monotonically non increasing in recall, then
    the area is summed exactly over every recall step. This is the modern COCO
    and updated VOC method, more accurate than the older 11 point sampling.
    """
    if curve.n_ground_truth == 0:
        return 0.0
    # Prepend (recall 0, precision 1) and append (last recall, precision 0).
    recalls = [0.0] + curve.recall + [curve.recall[-1] if curve.recall else 0.0]
    precisions = [1.0] + curve.precision + [0.0]

    # Make precision monotonically decreasing from the right.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


def _labels(boxes: list[Box]) -> set[str]:
    return {b.label for b in boxes}


def map_at_iou(
    preds: list[Box], gts: list[Box], iou_threshold: float
) -> tuple[float, dict[str, float]]:
    """mAP at one IoU threshold, plus per class AP.

    A class that appears only in predictions (never in ground truth) is skipped
    for AP, matching COCO, since AP is undefined without positives. Its false
    positives still hurt any class it is confused with through matching.
    """
    classes = sorted(_labels(gts))
    per_class: dict[str, float] = {}
    for c in classes:
        curve = _match_one_class(
            [p for p in preds if p.label == c],
            [g for g in gts if g.label == c],
            iou_threshold,
        )
        per_class[c] = average_precision(curve)
    mean = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return mean, per_class


# The ten COCO IoU thresholds, 0.50 to 0.95 step 0.05.
COCO_IOU_THRESHOLDS = [round(0.50 + 0.05 * i, 2) for i in range(10)]


@dataclass
class Evaluation:
    map50: float
    map50_95: float
    map75: float
    per_class_ap50: dict[str, float]
    precision: float
    recall: float
    f1: float


def evaluate(preds: list[Box], gts: list[Box]) -> Evaluation:
    """Full detection evaluation over a set of matched frames.

    `preds` and `gts` are the pooled boxes across every evaluated image. In a
    real run you would offset coordinates per image, or call this per image and
    average, so that a box in image A cannot match a box in image B. The helper
    `evaluate_dataset` below does the per image bookkeeping.
    """
    map50, per_class = map_at_iou(preds, gts, 0.50)
    map75, _ = map_at_iou(preds, gts, 0.75)
    per_t = [map_at_iou(preds, gts, t)[0] for t in COCO_IOU_THRESHOLDS]
    map50_95 = sum(per_t) / len(per_t)

    # Precision, recall, F1 reported at the lenient 0.5 threshold, pooled.
    tp = fp = ngt = 0
    for c in sorted(_labels(gts) | _labels(preds)):
        curve = _match_one_class(
            [p for p in preds if p.label == c],
            [g for g in gts if g.label == c],
            0.50,
        )
        tp += curve.tp
        fp += curve.fp
        ngt += curve.n_ground_truth
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / ngt if ngt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return Evaluation(map50, map50_95, map75, per_class, precision, recall, f1)


def evaluate_dataset(
    frames: list[tuple[list[Box], list[Box]]],
) -> Evaluation:
    """Evaluate a list of (predictions, ground_truth) per frame.

    Boxes are shifted into disjoint coordinate blocks per frame so matching
    never crosses frame boundaries, then evaluated together. This is the entry
    point a harness should call with real labelled frames.
    """
    all_preds: list[Box] = []
    all_gts: list[Box] = []
    offset = 0.0
    for preds, gts in frames:
        span = 1e6  # each frame gets its own coordinate block
        for b in preds:
            all_preds.append(
                Box(b.x1 + offset, b.y1, b.x2 + offset, b.y2, b.label, b.score)
            )
        for b in gts:
            all_gts.append(Box(b.x1 + offset, b.y1, b.x2 + offset, b.y2, b.label, b.score))
        offset += span
    return evaluate(all_preds, all_gts)


# --------------------------------------------------------------------------
# Counting error, the metric that actually matters for a gate counter
# --------------------------------------------------------------------------


@dataclass
class CountingError:
    """Per class counting error against a hand count.

    For a gate monitor the product question is not mAP, it is did we count the
    right number of vehicles. MAE is the mean absolute error across classes,
    and percentage error normalises by the true total.
    """

    per_class_abs_error: dict[str, int]
    mae: float
    total_true: int
    total_pred: int

    @property
    def total_abs_error(self) -> int:
        return abs(self.total_pred - self.total_true)

    @property
    def percent_error(self) -> float:
        return 100.0 * self.total_abs_error / self.total_true if self.total_true else 0.0


def counting_error(
    predicted: dict[str, int], truth: dict[str, int]
) -> CountingError:
    classes = set(predicted) | set(truth)
    errs = {c: abs(predicted.get(c, 0) - truth.get(c, 0)) for c in classes}
    mae = sum(errs.values()) / len(errs) if errs else 0.0
    return CountingError(
        per_class_abs_error=errs,
        mae=mae,
        total_true=sum(truth.values()),
        total_pred=sum(predicted.values()),
    )

# Accuracy test plan for the gate pipeline

How I plan to measure whether the `gcm_gatewatch` pipeline is any good, using
three real Gate-Exit clips (kept private), with the least manual
labelling I can get away with. This is the answer to the practical question:
how do I actually get an mAP and IoU number out of this footage, and what else
should I measure.

Every metric below has a citation in the Sources section, because a number
without a definition is not worth reporting.

## What the footage is, and what it can and cannot tell us

Three clips, all from the existing overhead Gate-Exit camera, 2880x1620 at
25 fps, about 26 minutes total. The view is a bus depot forecourt with the
actual gate a small sliding gate at the bottom-left.

That shapes the whole plan. The scene is mostly parked buses with a few
vehicles moving through the gate, so two different things need measuring and
they do not track each other:

- whether the detector finds and localises vehicles at all (Tier A), and
- whether the system counts the right number of entries and exits (Tier B),
  which is the actual product.

A detector can score a mediocre mAP yet count perfectly, because counting only
depends on the handful of vehicles that cross the gate line, not on the fifty
parked buses. The reverse is also true. So I measure both.

What this footage cannot validate: entry-side direction logic (no entry
camera clip exists) and plate read-rate from the dedicated ANPR camera (that
camera does not exist yet, see Task 3 and CAMERA_PLACEMENT.md). Plate reading
is reported as best-effort on the overhead footage, never gated.

## The metrics, defined

### IoU, intersection over union

Overlap between a predicted box and a ground-truth box: area of intersection
over area of union, in [0, 1]. A prediction counts as correct only if its IoU
with a ground-truth box clears a threshold. This is the localisation test that
every detection metric is built on [1].

### Precision, recall, F1

Precision is TP / (TP + FP), the fraction of detections that were real. Recall
is TP / (TP + FN), the fraction of real vehicles that were found. F1 is their
harmonic mean. Reported at IoU 0.5.

### AP and mAP

Average Precision (AP) is the area under the precision-recall curve for one
class. mAP is AP averaged over classes [2]. Two variants matter:

- **mAP@50** uses a single IoU threshold of 0.5. Lenient; rewards finding the
  vehicle even if the box is loose.
- **mAP@[50:95]** (the COCO primary metric) averages mAP over ten IoU
  thresholds from 0.5 to 0.95 in steps of 0.05. Strict; rewards tight boxes [2][3].

I report both. mAP@50 is the headline; mAP@[50:95] catches sloppy localisation
that mAP@50 hides.

### Counting error (Tier B, the product metric)

Absolute error between counted and true entries/exits per class, summarised as
MAE across classes and as a percentage of the true total. This is what GCM and
Nepal Police actually care about: did the log say the right number of buses
left at 4pm.

### Tracking quality (Tier C)

**HOTA** (Higher Order Tracking Accuracy), reported over the older MOTA and
IDF1. MOTA is biased toward detection and IDF1 toward association; HOTA
balances the two and exposes them separately as DetA and AssA, and it is now
the primary metric on the major tracking benchmarks [6]. This matters here
because a track that fragments as a bus passes behind another bus can double
count, and only an association-aware metric catches it.

## The three tiers

| Tier | Question | Metric | Target (initial) |
|---|---|---|---|
| A | Does the detector find vehicles? | mAP@50, mAP@[50:95], P, R | mAP@50 >= 0.75 |
| B | Does it count correctly? | entry/exit MAE, % error | % error <= 10% |
| C | Are tracks stable? | HOTA (DetA, AssA) | HOTA >= 0.60 |

Targets are first guesses to revise once real numbers land. They are not
promises.

## The easiest path to each number

This is the part worth optimising, because the slow step is never the maths, it
is the labelling.

### Ground truth without drawing boxes by hand

The trick is model-assisted labelling: let a model draw candidate boxes, then a
human only fixes them. Two ways, in order of how much I trust them here:

1. **Pre-label with the local model.** `prepare_dataset.py` samples frames and
   runs `yolo11l.pt` to produce candidate boxes in Ultralytics YOLO format. A
   heavier model is used on purpose, since fewer mistakes means less to correct.
   Import the result into CVAT, Roboflow, or Label Studio and correct in the
   browser.
2. **Zero-shot auto-labelling** with Autodistill + Grounding DINO or
   Grounded-SAM-2, prompted with "bus, car, motorcycle" [5]. Useful if the
   local model misses the Nepali-specific vehicle types, because a
   text-prompted open-vocabulary model is not limited to COCO classes.

Either way, correct before reporting. Grading a model against another model's
guesses measures nothing.

### mAP and IoU

Once labels are corrected, the day-to-day command is one line [4]:

```python
from ultralytics import YOLO
metrics = YOLO("model.pt").val(data="dataset/data.yaml")
print(metrics.box.map, metrics.box.map50, metrics.box.map75)  # 50-95, 50, 75
```

`evaluate.py` computes the same numbers through `detection_metrics.py`, the
readable reference implementation, so any surprising `val` result can be
audited line by line. `supervision`, already a project dependency, offers
`MeanAveragePrecision` as a third cross-check [8].

Agree at least two of these three against each other once, then trust `val`.

### Small parked-bus detections in wide frames

The buses at the back of the forecourt are small in a 2880px frame, which is
exactly where detectors miss. **SAHI** (Slicing Aided Hyper Inference) runs the
detector on overlapping tiles so small objects occupy more pixels per tile, and
reports AP gains of roughly 5 to 13% with no retraining [7]. Evaluate with and
without SAHI and keep it only if the gate-relevant vehicles improve; do not pay
its latency cost for parked buses that never cross the line.

### Counting

Score the project's own `GateLineCounter` (`perception/crossing.py`) against a
hand count of the clips. It already encodes the gate-mouth line and direction
logic. `supervision`'s `LineZone` is the off-the-shelf equivalent if a
independent cross-check is wanted.

### Tracking

Export tracker output and ground-truth tracks in MOT Challenge format and score
with **TrackEval** (the reference HOTA implementation) or `motmetrics` [6].

## Ground-truth protocol

- **Detection set:** sample at ~1 frame per 2 s across the three clips (about
  750 frames), pre-label, correct. Enough boxes for a stable mAP without a
  week of labelling.
- **Classes:** the five COCO vehicle types the current model knows (`bicycle`,
  `car`, `motorcycle`, `bus`, `truck`). Nepali-specific types (tempo, auto,
  microbus) are out of COCO and need a fine-tuned model; the class list in
  `prepare_dataset.py` is where they get added when that model exists.
- **Counting set:** watch each clip and tally actual entries and exits per
  class. One tally per clip. This is the Tier B ground truth and cannot be
  auto-generated.
- **Plate set:** the one clearly readable plate in clip 3, `बा ५ ख ७०७६`
  (BA 5 KHA 7076), is the OCR ground-truth sample. It parses cleanly through
  `plate/nepali_plate.py`, so it doubles as an end-to-end grammar check.

## Running it

```bash
cd development/v2/testing

# unit tests for the metric maths (currently 17, all green)
uv run --with pytest --python 3.12 python -m pytest . -q

# build a label-ready dataset from the videos (pre-labelled candidates)
python prepare_dataset.py --out dataset --fps 0.5 --imgsz 1920

# ... correct the labels in CVAT / Roboflow / Label Studio ...

# score a model against the corrected labels
python evaluate.py --model yolo11l.pt --dataset dataset --imgsz 1920
# cross-check: yolo val data=dataset/data.yaml
```

## Next steps, not built yet

- `to_mot.py`: export tracker and GT tracks to MOT format for TrackEval (Tier C).
- A SAHI on/off toggle in `evaluate.py` to quantify the small-object gain.
- A fine-tuned detector for Nepali vehicle types, which changes the class list
  and raises the achievable mAP on this footage.

## Sources

1. Everingham et al., *The PASCAL Visual Object Classes (VOC) Challenge*,
   IJCV 2010. IoU, precision/recall, AP as area under the PR curve.
2. Ultralytics, *What is Mean Average Precision (mAP)?*
   https://www.ultralytics.com/glossary/mean-average-precision-map
3. Lin et al., *Microsoft COCO: Common Objects in Context*, ECCV 2014, and the
   COCO evaluation protocol. mAP@[0.5:0.95].
4. Ultralytics, *Model Validation (val mode)*.
   https://docs.ultralytics.com/modes/val (`box.map`, `.map50`, `.map75`).
5. Roboflow Autodistill and Grounded-SAM-2: https://docs.autodistill.com and
   https://github.com/autodistill/autodistill-grounded-sam-2 (auto-labelling).
6. Luiten et al., *HOTA: A Higher Order Metric for Evaluating Multi-Object
   Tracking*, IJCV 2021, arXiv:2009.07736; TrackEval:
   https://github.com/JonathonLuiten/TrackEval (HOTA vs MOTA/IDF1).
7. Akyon et al., *Slicing Aided Hyper Inference and Fine-tuning for Small
   Object Detection*, arXiv:2202.06934; https://github.com/obss/sahi.
8. Roboflow Supervision, *Mean Average Precision*.
   https://supervision.roboflow.com/develop/metrics/mean_average_precision/

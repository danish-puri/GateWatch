"""Tests for the detection metrics, checked against hand computed values.

Run with: python3 -m pytest development/v2/testing/ -q
"""

from __future__ import annotations

import math

from detection_metrics import (
    Box,
    counting_error,
    evaluate,
    evaluate_dataset,
    iou,
    map_at_iou,
)


class TestIoU:
    def test_identical_boxes(self):
        b = Box(0, 0, 10, 10)
        assert iou(b, b) == 1.0

    def test_disjoint_boxes(self):
        assert iou(Box(0, 0, 10, 10), Box(20, 20, 30, 30)) == 0.0

    def test_half_overlap(self):
        # Two 10x10 boxes overlapping in a 5x10 strip.
        a, b = Box(0, 0, 10, 10), Box(5, 0, 15, 10)
        inter = 5 * 10  # 50
        union = 100 + 100 - 50  # 150
        assert math.isclose(iou(a, b), inter / union)

    def test_quarter_containment(self):
        # A 5x5 box fully inside a 10x10 box.
        outer, inner = Box(0, 0, 10, 10), Box(0, 0, 5, 5)
        assert math.isclose(iou(outer, inner), 25 / 100)

    def test_symmetric(self):
        a, b = Box(1, 1, 6, 6), Box(3, 2, 8, 9)
        assert math.isclose(iou(a, b), iou(b, a))


class TestMatchingAndAP:
    def test_perfect_detection_scores_one(self):
        gts = [Box(0, 0, 10, 10, "car"), Box(50, 50, 60, 60, "car")]
        preds = [
            Box(0, 0, 10, 10, "car", 0.9),
            Box(50, 50, 60, 60, "car", 0.8),
        ]
        m, per = map_at_iou(preds, gts, 0.5)
        assert math.isclose(m, 1.0)
        assert math.isclose(per["car"], 1.0)

    def test_duplicate_prediction_is_a_false_positive(self):
        gts = [Box(0, 0, 10, 10, "car")]
        preds = [
            Box(0, 0, 10, 10, "car", 0.9),  # TP
            Box(0, 0, 10, 10, "car", 0.8),  # duplicate -> FP
        ]
        _curve, _ = map_at_iou(preds, gts, 0.5)
        # AP with one TP then one FP: precision drops but AP stays 1.0 because
        # recall never advances past the first detection. The FP shows up in
        # precision, which the evaluate() path reports separately.
        ev = evaluate(preds, gts)
        assert ev.precision == 0.5  # 1 TP out of 2 detections
        assert ev.recall == 1.0

    def test_missed_object_lowers_recall(self):
        gts = [Box(0, 0, 10, 10, "car"), Box(50, 50, 60, 60, "car")]
        preds = [Box(0, 0, 10, 10, "car", 0.9)]  # second car missed
        ev = evaluate(preds, gts)
        assert ev.recall == 0.5
        assert ev.precision == 1.0

    def test_loose_box_fails_strict_threshold(self):
        gts = [Box(0, 0, 10, 10, "car")]
        # A box shifted so IoU is about 0.68, passes at 0.5 but not at 0.75.
        preds = [Box(2, 0, 12, 10, "car", 0.9)]
        assert iou(preds[0], gts[0]) > 0.5
        assert iou(preds[0], gts[0]) < 0.75
        assert map_at_iou(preds, gts, 0.5)[0] == 1.0
        assert map_at_iou(preds, gts, 0.75)[0] == 0.0

    def test_ap_of_half_precision_ranking(self):
        # Classic worked example: two GT, detections TP, FP, TP by confidence.
        # Recall reaches 1.0, precision at full recall is 2/3.
        gts = [Box(0, 0, 10, 10, "v"), Box(100, 0, 110, 10, "v")]
        preds = [
            Box(0, 0, 10, 10, "v", 0.9),  # TP
            Box(0, 40, 10, 50, "v", 0.8),  # FP, matches nothing
            Box(100, 0, 110, 10, "v", 0.7),  # TP
        ]
        m, _ = map_at_iou(preds, gts, 0.5)
        # PR points: (0.5,1.0), (0.5,0.5), (1.0,0.666). After making precision
        # monotonic from the right the envelope is 1.0 up to recall 0.5 then
        # 0.666 to recall 1.0: AP = 0.5*1.0 + 0.5*0.666 = 0.833.
        assert math.isclose(m, 0.5 * 1.0 + 0.5 * (2 / 3), rel_tol=1e-9)


class TestFullEvaluation:
    def test_map50_95_between_zero_and_one(self):
        gts = [Box(0, 0, 10, 10, "car")]
        preds = [Box(1, 1, 11, 11, "car", 0.9)]
        ev = evaluate(preds, gts)
        assert 0.0 <= ev.map50_95 <= ev.map50 <= 1.0

    def test_perfect_gives_all_ones(self):
        gts = [Box(0, 0, 10, 10, "bus")]
        preds = [Box(0, 0, 10, 10, "bus", 0.95)]
        ev = evaluate(preds, gts)
        assert ev.map50 == ev.map75 == ev.map50_95 == 1.0
        assert ev.f1 == 1.0

    def test_per_frame_matching_does_not_cross_frames(self):
        # Same coordinates in two frames must not match each other.
        frame_a = ([Box(0, 0, 10, 10, "car", 0.9)], [Box(0, 0, 10, 10, "car")])
        frame_b = ([], [Box(0, 0, 10, 10, "car")])  # a miss in frame B
        ev = evaluate_dataset([frame_a, frame_b])
        # Two GT cars, one found: recall 0.5, not 1.0.
        assert ev.recall == 0.5

    def test_multiclass_map_averages_over_classes(self):
        gts = [Box(0, 0, 10, 10, "car"), Box(50, 0, 60, 10, "bus")]
        preds = [
            Box(0, 0, 10, 10, "car", 0.9),  # car perfect
            # bus missed entirely
        ]
        m, per = map_at_iou(preds, gts, 0.5)
        assert per["car"] == 1.0
        assert per["bus"] == 0.0
        assert math.isclose(m, 0.5)  # mean of the two classes


class TestCounting:
    def test_exact_count_has_zero_error(self):
        ce = counting_error({"car": 10, "bus": 3}, {"car": 10, "bus": 3})
        assert ce.mae == 0.0
        assert ce.percent_error == 0.0

    def test_mae_and_percent(self):
        ce = counting_error({"car": 8, "bus": 5}, {"car": 10, "bus": 4})
        assert ce.per_class_abs_error == {"car": 2, "bus": 1}
        assert math.isclose(ce.mae, 1.5)
        assert ce.total_true == 14
        assert ce.total_pred == 13
        assert math.isclose(ce.percent_error, 100 * 1 / 14)

    def test_missing_class_counts_as_full_error(self):
        ce = counting_error({"car": 5}, {"car": 5, "motorcycle": 3})
        assert ce.per_class_abs_error["motorcycle"] == 3

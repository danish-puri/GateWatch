"""Subject detection via ultralytics YOLO.

Thin wrapper so the rest of the pipeline never imports ultralytics directly. Returns
detections in supervision's format so tracking and line-crossing compose cleanly.

One pass covers both kinds of subject at the gate, vehicles and people. `classes` is the
union of the two, and the caller maps each returned class id back to a kind. Running the
model twice to keep them apart would double the CPU cost for information the class id
already carries, and there is no GPU at GCM.
"""

from __future__ import annotations

import numpy as np


class VehicleDetector:
    """Detects vehicles (car, bus, motorbike, truck, ...) and people in a single frame.

    `class_labels` maps class ids to names. COCO lacks Nepali types (tempo/auto/micro);
    a fine-tuned model adds those ids and supplies their labels here (see
    docs/detection_tuning.md).
    """

    def __init__(
        self,
        model_path: str,
        *,
        classes: list[int],
        confidence: float,
        class_labels: dict[int, str] | None = None,
        imgsz: int = 1280,
    ) -> None:
        self.model_path = model_path
        self.classes = classes
        self.confidence = confidence
        self.class_labels = class_labels or {}
        self.imgsz = imgsz  # small far gate vehicles need more than the default 640
        self._model = None  # lazily loaded so importing this module needs no torch

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        return self._model

    @property
    def names(self) -> dict[int, str]:
        """Class id -> label (config overrides win over the model's own names)."""
        return {**self._load().names, **self.class_labels}

    def detect(self, frame: np.ndarray):  # -> supervision.Detections
        """Run YOLO and return detections in supervision's format."""
        import supervision as sv

        result = self._load()(
            frame,
            conf=self.confidence,
            classes=self.classes,
            imgsz=self.imgsz,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)

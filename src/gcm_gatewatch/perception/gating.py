"""Motion gate: skip the detector when nothing is happening at the gate, but only while
that is actually cheaper than detecting.

The original idea was that the yard is empty most of the time, so background subtraction
on the gate ROI could keep YOLO asleep. Measured against the real Gate-1 footage, that
premise only holds part of the day:

    16:21 clip   2.4% of frames contain nothing detectable
    10:48 clip  19.8% of frames contain nothing detectable

The first number is the ceiling on what any gate could ever save at that hour, and it is
smaller than what the gate itself cost to run: background subtraction over the native-res
ROI measured 22-24 ms a frame against 45 ms for the detector. Through the afternoon the
gate was paying half the detector's price to skip a fortieth of its work, so it was a net
loss and the reason the earlier version ran YOLO on 94-100% of frames anyway.

Two changes make it earn its keep:

1. It is cheap now. The ROI is downscaled to `sample_width` before subtraction and the
   speckle is opened away, which took 22-24 ms down to about 6.5 ms.
2. It stands down when it is not paying. The gate watches how often it actually skips a
   frame, compares that against the break-even skip rate implied by its own measured cost
   and the detector's, and goes to standby when it falls short. In standby it does nothing
   at all, so a busy afternoon costs nothing beyond an occasional probe. When the gate
   quietens down, at night or over a holiday, the probe finds a high skip rate and the
   gate stays on, which is where the real saving was always going to come from.

This is frame-level gating (whether to detect at all), distinct from motion.py's
track-level dwell filter (whether a tracked vehicle counts).

The decision and standby policy is pure and unit-tested; only `foreground_ratio` touches
OpenCV.
"""

from __future__ import annotations

import time
from enum import Enum


class GateState(str, Enum):
    """Whether the gate is currently earning its keep."""

    ACTIVE = "active"    # subtracting, and its verdict is used
    STANDBY = "standby"  # doing nothing, every frame goes to the detector
    WARMUP = "warmup"    # subtracting to relearn the background, verdict not yet trusted


class MotionGate:
    """Decides, per frame, whether to run the detector on the gate ROI.

    A cooldown keeps the detector running for a few frames after motion falls below the
    threshold, so a vehicle that briefly stops mid-gate (or that the crude foreground
    ratio under-reads) is not dropped part-way through its crossing.
    """

    def __init__(
        self,
        *,
        min_foreground_ratio: float = 0.005,
        cooldown_frames: int = 15,
        sample_width: int = 320,
        review_frames: int = 300,
        standby_frames: int = 2000,
        warmup_frames: int = 150,
        min_skip_rate: float = 0.15,
    ) -> None:
        self.min_foreground_ratio = min_foreground_ratio
        self.cooldown_frames = cooldown_frames
        self.sample_width = sample_width
        # how many frames of evidence to gather before judging whether the gate pays
        self.review_frames = review_frames
        # how long to stay out of the way once it has been judged not to pay. At stride 3
        # this is a few minutes, so the gate rechecks often enough to catch the yard
        # emptying at the end of the day without probing constantly.
        self.standby_frames = standby_frames
        # MOG2 needs to see the scene again before its verdict means anything. Waking
        # straight into judgement would read the relearning frames as wall-to-wall motion,
        # score a zero skip rate, and put the gate back to sleep for good.
        self.warmup_frames = warmup_frames
        # fallback break-even until both costs have been measured; overridden by the
        # real ratio as soon as the pipeline reports what a detection costs
        self.min_skip_rate = min_skip_rate

        self._cooldown = 0
        self._subtractor = None  # built on first frame so importing this needs no OpenCV
        self.state = GateState.WARMUP
        self._state_left = warmup_frames

        # rolling evidence for the current review window
        self._seen = 0
        self._skipped = 0
        # measured costs, in seconds per frame
        self._gate_cost: float | None = None
        self._detector_cost: float | None = None

        # lifetime counters, surfaced through /healthz
        self.frames_gated = 0
        self.frames_skipped = 0
        self.standbys = 0

    # ------------------------------------------------------------------ policy

    def should_detect(self, foreground_ratio: float) -> bool:
        """Pure policy: given the ROI's moving-pixel ratio, run the detector?

        Above threshold -> detect and re-arm the cooldown. Below -> keep detecting while
        the cooldown lasts, then stop.
        """
        if foreground_ratio >= self.min_foreground_ratio:
            self._cooldown = self.cooldown_frames
            return True
        if self._cooldown > 0:
            self._cooldown -= 1
            return True
        return False

    def note_detector_cost(self, seconds: float) -> None:
        """Tell the gate what one detection costs, so break-even is a fact not a guess.

        Smoothed, because a single frame's timing on a shared CPU says very little.
        """
        self._detector_cost = (
            seconds if self._detector_cost is None else 0.9 * self._detector_cost + 0.1 * seconds
        )

    def break_even_skip_rate(self) -> float:
        """The skip rate below which running the gate costs more than it saves.

        Skipping a frame saves one detection and every frame pays for the gate, so the
        gate is worth running only while `skip_rate * detector_cost > gate_cost`.
        """
        if not self._gate_cost or not self._detector_cost:
            return self.min_skip_rate
        return min(1.0, self._gate_cost / self._detector_cost)

    def _review(self) -> None:
        """Decide whether the last window justified the gate's own cost."""
        if self._seen < self.review_frames:
            return
        skip_rate = self._skipped / self._seen
        if skip_rate < self.break_even_skip_rate():
            self.state = GateState.STANDBY
            self._state_left = self.standby_frames
            self.standbys += 1
        self._seen = self._skipped = 0

    # ------------------------------------------------------------------ pixels

    def foreground_ratio(self, roi_crop) -> float:
        """Fraction of the ROI that the background subtractor calls moving foreground.

        The crop is downscaled to `sample_width` first. At native resolution this was the
        single most expensive thing in the loop after the detector, and the extra pixels
        bought nothing: the question is only whether something sizeable is moving, which
        survives downscaling intact.

        MOG2 runs with shadow detection on and shadows (marked 127) are dropped, so only
        hard foreground counts. A bus casts a long shadow across the forecourt in the
        Kathmandu afternoon, and counting shadow as motion would hold the gate open for
        most of the day. A morphological opening then clears the speckle, which measured
        in the thousands of tiny blobs per frame and made the ratio read high on frames
        where nothing of interest was moving at all.
        """
        import cv2

        if self._subtractor is None:
            # history 200 at stride 3 is roughly 20 seconds of scene, long enough that a
            # vehicle waiting at the gate does not melt into the background
            self._subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=32, detectShadows=True
            )
            self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        height, width = roi_crop.shape[:2]
        if width > self.sample_width:
            scale = self.sample_width / width
            roi_crop = cv2.resize(
                roi_crop,
                (self.sample_width, max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        mask = self._subtractor.apply(roi_crop)
        if mask is None or mask.size == 0:
            return 0.0
        import numpy as np

        hard = (mask == 255).astype(np.uint8)
        opened = cv2.morphologyEx(hard, cv2.MORPH_OPEN, self._kernel)
        return float(opened.sum()) / float(opened.size)

    # ------------------------------------------------------------------ per frame

    def observe_frame(self, roi_crop) -> bool:
        """Apply background subtraction to the ROI crop and decide whether to detect."""
        if self.state is GateState.STANDBY:
            self._state_left -= 1
            if self._state_left <= 0:
                # the background model went stale while asleep; relearn before judging
                self.state = GateState.WARMUP
                self._state_left = self.warmup_frames
            return True

        started = time.perf_counter()
        ratio = self.foreground_ratio(roi_crop)
        elapsed = time.perf_counter() - started
        self._gate_cost = (
            elapsed if self._gate_cost is None else 0.9 * self._gate_cost + 0.1 * elapsed
        )

        decision = self.should_detect(ratio)

        if self.state is GateState.WARMUP:
            self._state_left -= 1
            if self._state_left <= 0:
                self.state = GateState.ACTIVE
                self._seen = self._skipped = 0
            return True  # never skip on a verdict from a half-learned background

        self.frames_gated += 1
        self._seen += 1
        if not decision:
            self._skipped += 1
            self.frames_skipped += 1
        self._review()
        return decision

    def health(self) -> dict:
        """What the gate is doing and whether it is worth doing."""
        return {
            "state": self.state.value,
            "frames_gated": self.frames_gated,
            "frames_skipped": self.frames_skipped,
            "skip_rate": (
                round(self.frames_skipped / self.frames_gated, 3) if self.frames_gated else None
            ),
            "break_even_skip_rate": round(self.break_even_skip_rate(), 3),
            "standbys": self.standbys,
        }

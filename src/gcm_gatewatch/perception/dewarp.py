"""Fisheye handling.

The gate cameras are wide-angle: straight lines bow, and the distortion is worst at the
frame edges -- exactly where the gate mouth sits (bottom-left). Two-stage plan:

  Stage 1 (now): work in **distorted image space**. The gate line and any zones are drawn
  directly on the raw frame and crossings are tested there. Because the gate zone and the
  vehicle centroids live in the *same* distorted space, the crossing geometry is
  self-consistent and correct without undistortion. No calibration needed to start.

  Stage 2 (later): calibrate the lens (OpenCV fisheye model: camera matrix K + distortion
  coeffs D) and **undistort** frames before geometry. This makes distances and directions
  metric, which helps speed estimation and any homography to a top-down view -- but costs
  a per-frame remap and a calibration session. Deferred until Stage 1 accuracy is
  measured and found wanting.

This module is the Stage-2 seam. Left as a stub so Stage 1 does not depend on it.
"""

from __future__ import annotations

import numpy as np


class Dewarper:
    """Undistorts fisheye frames using a calibrated lens model (Stage 2)."""

    def __init__(self, camera_matrix, distortion_coeffs) -> None:
        self.camera_matrix = camera_matrix      # K
        self.distortion_coeffs = distortion_coeffs  # D

    @classmethod
    def from_calibration_file(cls, path) -> Dewarper:
        """Load K and D produced by an offline fisheye calibration. Stub."""
        raise NotImplementedError

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Remap a distorted frame to a rectilinear one. Stub."""
        raise NotImplementedError

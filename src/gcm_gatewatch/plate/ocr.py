"""Nepali license-plate OCR via Google Cloud Vision.

v2 uses Google Cloud Vision for reading plate text (replacing v1's EasyOCR, which was
also mistakenly configured English-only). Cloud Vision has real Devanagari/Nepali text
detection, so it is a much stronger reader than EasyOCR.

Honest caveat (see docs/feasibility.md): the two current gate cameras cannot resolve
Devanagari plate characters -- that is an optics limit, not an OCR-engine limit, so no
reader fixes it on the present hardware. This module pays off on the rare favorable
frame and once a plate-grade camera is added. It is gated off by default in config.

Flow:  plate crop (bytes) -> Cloud Vision text detection (with ne/hi/en hints)
       -> raw text candidates -> gcm_gatewatch.plate.nepali_plate.parse_plate / decode
       -> validated NepaliPlate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OcrResult:
    """One Cloud Vision text read of a plate crop."""

    text: str          # raw detected text (may mix Devanagari and Latin)
    confidence: float  # Cloud Vision's confidence for the best block


class PlateOCR:
    """Reads plate text from an image crop using Google Cloud Vision.

    Auth uses application default credentials; point GOOGLE_APPLICATION_CREDENTIALS at a
    service-account key (see .env.example). `language_hints` biases the detector toward
    Nepali, Hindi, and English scripts.
    """

    def __init__(
        self,
        *,
        language_hints: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> None:
        self.language_hints = language_hints or ["ne", "hi", "en"]
        self.min_confidence = min_confidence
        # constructs a google.cloud.vision.ImageAnnotatorClient

    def read(self, image_bytes: bytes) -> OcrResult | None:
        """Run Cloud Vision text detection on a plate crop. Stub.

        Returns the best text candidate, or None if nothing clears min_confidence.
        Caller passes the text to nepali_plate.parse_plate to validate/normalize it.
        """
        raise NotImplementedError

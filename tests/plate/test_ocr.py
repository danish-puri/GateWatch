"""Tests for the Google Cloud Vision plate OCR.

Once implemented, these should mock the Cloud Vision client (no network in tests) and
check that: a detected string is returned as an OcrResult, a read below min_confidence
yields None, and the raw text feeds cleanly into nepali_plate.parse_plate. The parser
itself is covered by test_nepali_plate.py.
"""

from __future__ import annotations

import pytest

from gcm_gatewatch.plate.ocr import PlateOCR


@pytest.mark.skip(reason="scaffold: PlateOCR.read not implemented yet")
def test_read_returns_none_below_min_confidence() -> None:
    ocr = PlateOCR(min_confidence=0.9)
    # with a mocked Cloud Vision response whose best block scores < 0.9
    assert ocr.read(b"...") is None

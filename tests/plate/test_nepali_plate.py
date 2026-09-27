"""Tests for the Nepali plate grammar and constrained decoder.

Run with: python3 -m pytest development/v2/nepaliOCR/ -q
"""

from __future__ import annotations

import math

import pytest

from gcm_gatewatch.plate.nepali_plate import (
    DEVANAGARI_DIGITS,
    VEHICLE_CLASSES,
    ZONES,
    Decoded,
    NepaliPlate,
    Ownership,
    PlateColour,
    Weight,
    classes_with_colour,
    decode,
    fuse,
    is_valid,
    parse_plate,
    to_devanagari_digits,
    to_western_digits,
)

# Modelled on a van seen in the gate footage: a CHA class plate painted black,
# which makes it a public light vehicle. The digits are made up so no real
# vehicle is identified.
HIACE_DEVANAGARI = "बा९च४८१२"


class TestDigits:
    def test_roundtrip(self):
        assert to_western_digits(DEVANAGARI_DIGITS) == "0123456789"
        assert to_devanagari_digits("0123456789") == DEVANAGARI_DIGITS

    def test_leaves_other_characters_alone(self):
        assert to_western_digits("बा२ख") == "बा2ख"


class TestParsing:
    def test_parses_devanagari(self):
        p = parse_plate("बा २ ख २४३६")
        assert p is not None
        assert p.zone.latin == "BA"
        assert p.lot == 2
        assert p.vehicle_class.latin == "KHA"
        assert p.serial == 2436

    def test_parses_latin(self):
        p = parse_plate("BA 2 KHA 2436")
        assert p is not None
        assert p.canonical() == "BA-2-KHA-2436"

    def test_scripts_agree(self):
        assert parse_plate("बा २ ख २४३६") == parse_plate("BA 2 KHA 2436")

    def test_hiace_from_footage(self):
        p = parse_plate(HIACE_DEVANAGARI)
        assert p is not None
        assert p.serial == 4812
        # A CHA plate is private, so it should be red. The observed plate is
        # black, which is the sort of contradiction the colour check catches.
        assert p.expected_colour is PlateColour.RED

    def test_tolerates_punctuation_and_spacing(self):
        assert parse_plate("  BA-2 KHA_2436 ") == parse_plate("BA2KHA2436")

    def test_ignores_case(self):
        assert parse_plate("ba 2 kha 2436") == parse_plate("BA 2 KHA 2436")

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "   ",
            "XY 2 KHA 2436",  # no such zone
            "BA 2 ZZ 2436",  # no such class
            "BA 2 KHA 24361",  # serial too long
            "BA 123 KHA 2436",  # lot too long
            "BA KHA 2436",  # missing lot
            "BA 2 KHA",  # missing serial
            "2 BA KHA 2436",  # wrong order
            "इ२00",  # what EasyOCR actually returned on the Hiace plate
            "तःष",  # and on the bus plate
        ],
    )
    def test_rejects_impossible_plates(self, bad):
        assert parse_plate(bad) is None
        assert not is_valid(bad)


class TestColour:
    def test_colour_narrows_the_class_slot(self):
        # Colour is the signal that survives low resolution, and it cuts the
        # 12 class letters down to a handful.
        for colour in PlateColour:
            assert len(classes_with_colour(colour)) < len(VEHICLE_CLASSES)
        assert {c.latin for c in classes_with_colour(PlateColour.BLACK)} == {
            "KHA",
            "JA",
            "PHA",
        }

    def test_every_class_has_a_colour(self):
        for cls in VEHICLE_CLASSES.values():
            assert isinstance(cls.colour, PlateColour)

    def test_tourist_class_has_no_single_weight(self):
        assert VEHICLE_CLASSES["य"].weight is None
        assert VEHICLE_CLASSES["य"].ownership is Ownership.TOURIST

    def test_heavy_private_is_red(self):
        cls = VEHICLE_CLASSES["क"]
        assert cls.weight is Weight.HEAVY
        assert cls.ownership is Ownership.PRIVATE
        assert cls.colour is PlateColour.RED


class TestPlateModel:
    def test_rejects_out_of_range(self):
        zone, cls = ZONES["बा"], VEHICLE_CLASSES["ख"]
        with pytest.raises(ValueError):
            NepaliPlate(zone, 100, cls, 1)
        with pytest.raises(ValueError):
            NepaliPlate(zone, 1, cls, 10000)

    def test_roundtrips_through_both_scripts(self):
        p = parse_plate("बा २ ख २४३६")
        assert parse_plate(p.devanagari()) == p
        assert parse_plate(p.latin()) == p


def lattice_from(text: str, correct_prob: float, distractor: str = "५") -> list:
    """Build a lattice where each true character competes with a distractor."""
    return [
        [(ch, correct_prob), (distractor, 1.0 - correct_prob)] for ch in text
    ]


class TestConstrainedDecoding:
    def test_recovers_the_plate_when_the_signal_is_clean(self):
        out = decode(lattice_from("बा२ख२४३६", 0.9))
        assert out
        assert out[0].plate.canonical() == "BA-2-KHA-2436"

    def test_split_zone_glyph(self):
        # बा arriving as ब then ा rather than as one combined glyph.
        lattice = [[("ब", 0.9)], [("ा", 0.9)]] + lattice_from("२ख२४३६", 0.9)
        out = decode(lattice)
        assert out and out[0].plate.zone.latin == "BA"

    def test_rejects_a_lattice_with_no_legal_parse(self):
        # A leading digit cannot start a plate.
        assert decode(lattice_from("२२ख२४३६", 0.9)) == []

    def test_must_consume_the_whole_lattice(self):
        # Trailing junk means no parse rather than a truncated guess.
        assert decode(lattice_from("बा२ख२४३६", 0.9) + [[("क", 0.9)]]) == []

    def test_colour_prunes_the_candidates(self):
        # ख (public, black) and घ (corporation, yellow) compete in the class
        # slot, with the wrong one scoring higher. Colour breaks the tie.
        lattice = (
            lattice_from("बा२", 0.9)
            + [[("घ", 0.7), ("ख", 0.3)]]
            + lattice_from("२४३६", 0.9)
        )
        assert decode(lattice)[0].plate.vehicle_class.latin == "GHA"
        best = decode(lattice, plate_colour=PlateColour.BLACK)[0]
        assert best.plate.vehicle_class.latin == "KHA"

    def test_ranked_by_probability(self):
        out = decode(lattice_from("बा२ख२४३६", 0.6), top_k=5)
        assert len(out) > 1
        assert all(
            out[i].log_prob >= out[i + 1].log_prob for i in range(len(out) - 1)
        )

    def test_offers_alternative_serials_for_human_review(self):
        # Two ambiguous digit slots. A reviewer needs to see the competing
        # serials, not just the single best one.
        lattice = (
            lattice_from("बा", 0.9)
            + [[("२", 0.6), ("३", 0.4)]]
            + lattice_from("ख", 0.9)
            + [[("२", 0.6), ("७", 0.4)]]
            + lattice_from("४", 0.9)
            + [[("३", 0.5), ("८", 0.5)]]
            + lattice_from("६", 0.9)
        )
        out = decode(lattice, top_k=5)
        serials = {d.plate.serial for d in out}
        assert len(serials) > 1, "decoder must surface competing serials"
        assert out[0].plate.canonical() == "BA-2-KHA-2436"
        assert all(d.plate.vehicle_class.latin == "KHA" for d in out)

    def test_confidence_is_a_probability(self):
        out = decode(lattice_from("बा२ख२४३६", 0.9))
        assert 0.0 < out[0].confidence <= 1.0
        assert math.isclose(out[0].confidence, math.exp(out[0].log_prob))

    def test_empty_lattice(self):
        assert decode([]) == []


class TestFusion:
    def _decoded(self, text: str, conf: float) -> Decoded:
        return Decoded(parse_plate(text), math.log(conf))

    def test_majority_wins_over_a_confident_outlier(self):
        right = [self._decoded("BA 2 KHA 2436", 0.5)]
        wrong = [self._decoded("BA 2 KHA 2438", 0.95)]
        fused = fuse([right, right, right, right, wrong])
        assert fused.plate.serial == 2436
        assert fused.n_supporting_frames == 4
        assert fused.n_frames == 5
        assert math.isclose(fused.agreement, 0.8)

    def test_ignores_frames_that_failed_to_decode(self):
        good = [self._decoded("BA 2 KHA 2436", 0.8)]
        fused = fuse([[], good, [], good, []])
        assert fused.plate.serial == 2436
        assert fused.n_frames == 2  # only the frames that produced a plate

    def test_returns_none_when_nothing_decoded(self):
        assert fuse([[], [], []]) is None
        assert fuse([]) is None

    def test_min_agreement_suppresses_a_split_vote(self):
        a = [self._decoded("BA 2 KHA 2436", 0.6)]
        b = [self._decoded("BA 2 KHA 2438", 0.6)]
        # An even split should not be reported as a confident read.
        assert fuse([a, b], min_agreement=0.75) is None
        assert fuse([a, a, a, b], min_agreement=0.75) is not None

"""Grammar and constrained decoding for Nepali vehicle registration plates.

At the resolution a gate camera gives us, a raw character recogniser is not
good enough on its own. What rescues it is that a Nepali plate is not an
arbitrary string. Under the pre 2020 system, which is still most of the fleet
at GCM, a plate reads

    ZONE   LOT   CLASS   SERIAL
    बा     २     ख       २४३६

with only 14 legal zone tokens and 13 legal class letters. The search space is
small enough that forcing the recogniser to emit a structurally legal plate
removes most of its errors for free.

Plate background colour carries the ownership category, and colour survives
low resolution far better than glyph shape does. Passing an observed colour in
prunes the class slot before decoding, which is the single cheapest accuracy
win available to us.

This module is deliberately pure stdlib and has no opinion about how the
characters were recognised, so it can sit behind EasyOCR now and a fine tuned
recogniser later.
"""

from __future__ import annotations

import math
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

# --------------------------------------------------------------------------
# Character sets
# --------------------------------------------------------------------------

DEVANAGARI_DIGITS = "०१२३४५६७८९"
WESTERN_DIGITS = "0123456789"

_DEV_TO_WEST = {d: w for d, w in zip(DEVANAGARI_DIGITS, WESTERN_DIGITS)}
_WEST_TO_DEV = {w: d for d, w in zip(DEVANAGARI_DIGITS, WESTERN_DIGITS)}


def to_western_digits(text: str) -> str:
    """Rewrite Devanagari digits as ASCII, leaving everything else alone."""
    return "".join(_DEV_TO_WEST.get(ch, ch) for ch in text)


def to_devanagari_digits(text: str) -> str:
    """Rewrite ASCII digits as Devanagari, leaving everything else alone."""
    return "".join(_WEST_TO_DEV.get(ch, ch) for ch in text)


class Ownership(Enum):
    PRIVATE = "private"
    PUBLIC = "public"
    GOVERNMENT = "government"
    CORPORATION = "national corporation"
    TOURIST = "tourist"
    DIPLOMATIC = "diplomatic"


class Weight(Enum):
    HEAVY = "heavy"
    LIGHT = "light"
    MOTORCYCLE = "motorcycle"


class PlateColour(Enum):
    """Background colour of the plate, which encodes ownership.

    This is the low resolution signal. At 15 px character height the glyphs are
    gone but the background is still several hundred pixels of flat colour.
    """

    RED = "red"
    BLACK = "black"
    WHITE = "white"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"


COLOUR_OF_OWNERSHIP = {
    Ownership.PRIVATE: PlateColour.RED,
    Ownership.PUBLIC: PlateColour.BLACK,
    Ownership.GOVERNMENT: PlateColour.WHITE,
    Ownership.CORPORATION: PlateColour.YELLOW,
    Ownership.TOURIST: PlateColour.GREEN,
    Ownership.DIPLOMATIC: PlateColour.BLUE,
}


@dataclass(frozen=True)
class Zone:
    devanagari: str
    latin: str
    name: str


# The 14 zones of the pre 2020 registration system.
ZONES: dict[str, Zone] = {
    z.devanagari: z
    for z in (
        Zone("मे", "ME", "Mechi"),
        Zone("को", "KO", "Koshi"),
        Zone("स", "SA", "Sagarmatha"),
        Zone("ज", "JA", "Janakpur"),
        Zone("बा", "BA", "Bagmati"),
        Zone("ना", "NA", "Narayani"),
        Zone("ग", "GA", "Gandaki"),
        Zone("लु", "LU", "Lumbini"),
        Zone("ध", "DH", "Dhawalagiri"),
        Zone("रा", "RA", "Rapti"),
        Zone("भे", "BHE", "Bheri"),
        Zone("क", "KA", "Karnali"),
        Zone("से", "SE", "Seti"),
        Zone("म", "MA", "Mahakali"),
    )
}

ZONE_BY_LATIN = {z.latin: z for z in ZONES.values()}


@dataclass(frozen=True)
class VehicleClass:
    devanagari: str
    latin: str
    ownership: Ownership
    # Tourist plates share one letter across every weight, so weight is not
    # always recoverable from the class letter alone.
    weight: Weight | None

    @property
    def colour(self) -> PlateColour:
        return COLOUR_OF_OWNERSHIP[self.ownership]


VEHICLE_CLASSES: dict[str, VehicleClass] = {
    c.devanagari: c
    for c in (
        # Heavy
        VehicleClass("क", "KA", Ownership.PRIVATE, Weight.HEAVY),
        VehicleClass("ख", "KHA", Ownership.PUBLIC, Weight.HEAVY),
        VehicleClass("ग", "GA", Ownership.GOVERNMENT, Weight.HEAVY),
        VehicleClass("घ", "GHA", Ownership.CORPORATION, Weight.HEAVY),
        # Light
        VehicleClass("च", "CHA", Ownership.PRIVATE, Weight.LIGHT),
        VehicleClass("ज", "JA", Ownership.PUBLIC, Weight.LIGHT),
        VehicleClass("झ", "JHA", Ownership.GOVERNMENT, Weight.LIGHT),
        VehicleClass("ञ", "NYA", Ownership.CORPORATION, Weight.LIGHT),
        # Motorcycle
        VehicleClass("प", "PA", Ownership.PRIVATE, Weight.MOTORCYCLE),
        VehicleClass("फ", "PHA", Ownership.PUBLIC, Weight.MOTORCYCLE),
        VehicleClass("ब", "BA", Ownership.GOVERNMENT, Weight.MOTORCYCLE),
        # Any weight
        VehicleClass("य", "YA", Ownership.TOURIST, None),
    )
}

CLASS_BY_LATIN = {c.latin: c for c in VEHICLE_CLASSES.values()}

MAX_LOT_DIGITS = 2
MAX_SERIAL_DIGITS = 4


def classes_with_colour(colour: PlateColour) -> list[VehicleClass]:
    """Every class letter consistent with an observed plate background."""
    return [c for c in VEHICLE_CLASSES.values() if c.colour is colour]


# --------------------------------------------------------------------------
# Plate
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NepaliPlate:
    """A structurally valid plate under the pre 2020 system."""

    zone: Zone
    lot: int
    vehicle_class: VehicleClass
    serial: int

    def __post_init__(self) -> None:
        if not 0 <= self.lot < 10**MAX_LOT_DIGITS:
            raise ValueError(f"lot out of range: {self.lot}")
        if not 0 <= self.serial < 10**MAX_SERIAL_DIGITS:
            raise ValueError(f"serial out of range: {self.serial}")

    @property
    def ownership(self) -> Ownership:
        return self.vehicle_class.ownership

    @property
    def expected_colour(self) -> PlateColour:
        return self.vehicle_class.colour

    def devanagari(self) -> str:
        return " ".join(
            (
                self.zone.devanagari,
                to_devanagari_digits(str(self.lot)),
                self.vehicle_class.devanagari,
                to_devanagari_digits(str(self.serial)),
            )
        )

    def latin(self) -> str:
        return f"{self.zone.latin} {self.lot} {self.vehicle_class.latin} {self.serial}"

    def canonical(self) -> str:
        """Stable key for database storage and comparison."""
        return f"{self.zone.latin}-{self.lot}-{self.vehicle_class.latin}-{self.serial}"

    def __str__(self) -> str:
        return self.latin()


# --------------------------------------------------------------------------
# Parsing a finished string
# --------------------------------------------------------------------------


def _strip(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return "".join(ch for ch in text if not ch.isspace() and ch not in ".-_")


def parse_plate(text: str) -> NepaliPlate | None:
    """Parse a plate string in either script. Returns None if it is not legal.

    Tolerant of spacing and punctuation, strict about structure. A string that
    does not describe a plate that could exist comes back as None rather than
    being coerced into something plausible.
    """
    s = _strip(text)
    if not s:
        return None

    upper = s.upper()
    # Latin form, e.g. BA2KH2436
    for zone_latin in sorted(ZONE_BY_LATIN, key=len, reverse=True):
        if not upper.startswith(zone_latin):
            continue
        rest = upper[len(zone_latin) :]
        lot_str = ""
        while rest and rest[0] in WESTERN_DIGITS and len(lot_str) < MAX_LOT_DIGITS:
            lot_str, rest = lot_str + rest[0], rest[1:]
        if not lot_str:
            continue
        for class_latin in sorted(CLASS_BY_LATIN, key=len, reverse=True):
            if not rest.startswith(class_latin):
                continue
            serial_str = rest[len(class_latin) :]
            if (
                serial_str
                and serial_str.isdigit()
                and len(serial_str) <= MAX_SERIAL_DIGITS
            ):
                return NepaliPlate(
                    ZONE_BY_LATIN[zone_latin],
                    int(lot_str),
                    CLASS_BY_LATIN[class_latin],
                    int(serial_str),
                )

    # Devanagari form, e.g. बा२ख२४३६
    for zone_dev in sorted(ZONES, key=len, reverse=True):
        if not s.startswith(zone_dev):
            continue
        rest = s[len(zone_dev) :]
        lot_str = ""
        while rest and rest[0] in DEVANAGARI_DIGITS and len(lot_str) < MAX_LOT_DIGITS:
            lot_str, rest = lot_str + rest[0], rest[1:]
        if not lot_str or not rest:
            continue
        cls = VEHICLE_CLASSES.get(rest[0])
        if cls is None:
            continue
        serial_str = rest[1:]
        if (
            serial_str
            and all(ch in DEVANAGARI_DIGITS for ch in serial_str)
            and len(serial_str) <= MAX_SERIAL_DIGITS
        ):
            return NepaliPlate(
                ZONES[zone_dev],
                int(to_western_digits(lot_str)),
                cls,
                int(to_western_digits(serial_str)),
            )
    return None


def is_valid(text: str) -> bool:
    """True when the string describes a plate that could legally exist."""
    return parse_plate(text) is not None


# --------------------------------------------------------------------------
# Constrained decoding
# --------------------------------------------------------------------------

# A lattice is one slot per recognised character position, each slot holding
# candidate characters with probabilities. This is what a CTC or per character
# classifier gives you before argmax throws the alternatives away.
Slot = Sequence[tuple[str, float]]
Lattice = Sequence[Slot]

_NEG_INF = float("-inf")


@dataclass(frozen=True)
class Decoded:
    plate: NepaliPlate
    log_prob: float

    @property
    def confidence(self) -> float:
        return math.exp(self.log_prob)


def _slot_score(slot: Slot, char: str) -> float:
    for candidate, prob in slot:
        if candidate == char:
            return math.log(prob) if prob > 0 else _NEG_INF
    return _NEG_INF


def _token_score(lattice: Lattice, pos: int, token: str) -> tuple[float, int]:
    """Score a multi character token against consecutive slots.

    Returns the log probability and how many slots it consumed. A token such as
    बा may arrive as one combined glyph or as two, so both are tried.
    """
    # Whole token sitting in a single slot.
    single = _slot_score(lattice[pos], token) if pos < len(lattice) else _NEG_INF
    best, width = single, 1

    # Token spread across consecutive slots.
    if len(token) > 1 and pos + len(token) <= len(lattice):
        total = 0.0
        for offset, ch in enumerate(token):
            total += _slot_score(lattice[pos + offset], ch)
            if total == _NEG_INF:
                break
        if total > best:
            best, width = total, len(token)
    return best, width


def decode(
    lattice: Lattice,
    *,
    plate_colour: PlateColour | None = None,
    top_k: int = 5,
    beam: int = 8,
) -> list[Decoded]:
    """Find the best structurally legal plates explaining the whole lattice.

    Every returned parse consumes the lattice exactly, so scores are directly
    comparable. Passing `plate_colour` restricts the class slot to letters that
    match the observed background, which is the cheapest accuracy win here.

    Digit runs are explored with a beam rather than taken greedily. Greedy is
    optimal for the single best plate, because digit positions carry no
    constraint on each other, but it can only ever return one serial. Ranked
    alternatives are what a human needs when reviewing a low confidence read,
    so `beam` controls how many competing digit runs stay alive.
    """
    n = len(lattice)
    if n == 0:
        return []

    allowed_classes = (
        classes_with_colour(plate_colour)
        if plate_colour is not None
        else list(VEHICLE_CLASSES.values())
    )
    if not allowed_classes:
        return []

    results: list[Decoded] = []

    def digits_from(pos: int, max_digits: int) -> Iterable[tuple[str, float, int]]:
        """Yield digit runs starting at pos as (text, log_prob, next_pos).

        Runs of every length up to `max_digits` are produced, keeping the top
        `beam` hypotheses at each step.
        """
        frontier: list[tuple[str, float, int]] = [("", 0.0, pos)]
        for _ in range(max_digits):
            extended: list[tuple[str, float, int]] = []
            for run, score, p in frontier:
                if p >= n:
                    continue
                for ch, prob in lattice[p]:
                    west = _DEV_TO_WEST.get(ch, ch)
                    if west not in WESTERN_DIGITS or prob <= 0:
                        continue
                    extended.append((run + west, score + math.log(prob), p + 1))
            if not extended:
                return
            extended.sort(key=lambda t: t[1], reverse=True)
            frontier = extended[:beam]
            yield from frontier

    for zone in ZONES.values():
        zone_lp, zone_width = _token_score(lattice, 0, zone.devanagari)
        if zone_lp == _NEG_INF:
            continue
        for lot_str, lot_lp, after_lot in digits_from(zone_width, MAX_LOT_DIGITS):
            for cls in allowed_classes:
                cls_lp, cls_width = _token_score(lattice, after_lot, cls.devanagari)
                if cls_lp == _NEG_INF:
                    continue
                after_cls = after_lot + cls_width
                for serial_str, serial_lp, after_serial in digits_from(
                    after_cls, MAX_SERIAL_DIGITS
                ):
                    if after_serial != n:
                        continue  # must consume the lattice exactly
                    results.append(
                        Decoded(
                            NepaliPlate(zone, int(lot_str), cls, int(serial_str)),
                            zone_lp + lot_lp + cls_lp + serial_lp,
                        )
                    )

    results.sort(key=lambda d: d.log_prob, reverse=True)
    return results[:top_k]


# --------------------------------------------------------------------------
# Multi frame fusion
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FusedPlate:
    plate: NepaliPlate
    score: float
    n_supporting_frames: int
    n_frames: int

    @property
    def agreement(self) -> float:
        """Fraction of frames that voted for this plate."""
        return self.n_supporting_frames / self.n_frames if self.n_frames else 0.0


def fuse(
    per_frame: Sequence[Sequence[Decoded]],
    *,
    min_agreement: float = 0.0,
) -> FusedPlate | None:
    """Combine per frame decodes of one tracked vehicle into one answer.

    A vehicle crossing the gate is seen for tens of frames at varying range and
    angle, and the errors are largely independent between frames. Aggregating
    the evidence is worth more than any single frame improvement, which is why
    a real deployment should decode every frame and fuse rather than picking a
    single best shot.

    Returns None when no frame produced a legal plate, or when the winner is
    supported by fewer than `min_agreement` of the frames that did decode.
    """
    frames_with_a_decode = [f for f in per_frame if f]
    if not frames_with_a_decode:
        return None

    weight: dict[str, float] = defaultdict(float)
    votes: dict[str, int] = defaultdict(int)
    plates: dict[str, NepaliPlate] = {}

    for frame in frames_with_a_decode:
        best_key = frame[0].plate.canonical()
        votes[best_key] += 1
        for decoded in frame:
            key = decoded.plate.canonical()
            plates[key] = decoded.plate
            weight[key] += decoded.confidence

    winner = max(weight, key=lambda k: (weight[k], votes[k]))
    n_frames = len(frames_with_a_decode)
    fused = FusedPlate(
        plate=plates[winner],
        score=weight[winner] / n_frames,
        n_supporting_frames=votes[winner],
        n_frames=n_frames,
    )
    if fused.agreement < min_agreement:
        return None
    return fused

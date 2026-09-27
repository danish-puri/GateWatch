"""Tests for the reconnecting RTSP reader.

The reconnect policy is the part that decides whether an unattended run survives the
night, so it is tested against a fake capture rather than a real camera. No OpenCV, no
network, no real sleeping.

Covers: frames come through; stride grabs rather than sleeps past skipped frames; a
stalled stream reconnects; backoff grows and is capped; a good frame resets the backoff
so a flaky link does not drift towards the cap; and close() stops the loop.
"""

from __future__ import annotations

import numpy as np
import pytest

from gcm_gatewatch.capture.stream import ReconnectingStream


class FakeCapture:
    """Stands in for cv2.VideoCapture. Delivers `frames` then reports failure."""

    def __init__(self, frames: int, *, opens: bool = True) -> None:
        self._remaining = frames
        self._opens = opens
        self.grabs = 0
        self.reads = 0
        self.released = False

    def isOpened(self) -> bool:  # mirrors the OpenCV API, hence the camelCase
        return self._opens

    def grab(self) -> bool:
        self.grabs += 1
        if self._remaining <= 0:
            return False
        self._remaining -= 1
        return True

    def read(self):
        self.reads += 1
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((4, 4, 3), dtype=np.uint8)

    def release(self) -> None:
        self.released = True


def _stream(captures, **kwargs) -> tuple[ReconnectingStream, list[float]]:
    """A stream over a queue of fake captures, with sleeps recorded instead of taken."""
    slept: list[float] = []
    queue = list(captures)

    def factory(_url):
        return queue.pop(0) if queue else FakeCapture(0, opens=False)

    stream = ReconnectingStream(
        "rtsp://fake",
        capture_factory=factory,
        sleep=slept.append,
        **kwargs,
    )
    return stream, slept


def test_yields_frames_until_the_source_runs_out() -> None:
    stream, _ = _stream([FakeCapture(3)], max_reconnects=0)
    assert len(list(stream.frames())) == 3
    assert stream.frames_read == 3


def test_stride_grabs_the_skipped_frames() -> None:
    """Skipping by sleeping would let the camera buffer run ahead of the log."""
    capture = FakeCapture(6)
    stream, _ = _stream([capture], max_reconnects=0)

    frames = list(stream.frames(stride=3))

    assert len(frames) == 2          # 6 frames, 1 kept per 3
    assert capture.reads == 2        # only the kept frames are decoded
    # 2 skipped per kept frame is 4, plus the grab that runs into the empty source and
    # reports the stall
    assert capture.grabs == 5


def test_reconnects_when_the_stream_stalls() -> None:
    first, second = FakeCapture(2), FakeCapture(1)
    stream, slept = _stream([first, second], max_reconnects=1)

    frames = list(stream.frames())

    assert len(frames) == 3          # both captures contributed
    assert stream.reconnects >= 1
    assert first.released and second.released
    assert slept  # it waited before retrying instead of spinning


def test_backoff_grows_and_is_capped() -> None:
    stream, _ = _stream([], reconnect_backoff_seconds=2.0, max_backoff_seconds=60.0)

    assert stream.backoff_for(1) == 2.0
    assert stream.backoff_for(2) == 4.0
    assert stream.backoff_for(3) == 8.0
    assert stream.backoff_for(10) == 60.0   # capped, so recovery stays prompt


def test_a_camera_that_opens_but_never_delivers_still_backs_off() -> None:
    """The nasty case: the socket connects, so opening looks fine, but no frames come.

    Resetting the failure count on a successful open would pin the backoff at its
    minimum and retry forever at full speed. Only a delivered frame counts as proof.
    """
    stream, slept = _stream([FakeCapture(0) for _ in range(4)], max_reconnects=3)

    assert list(stream.frames()) == []
    assert slept == [2.0, 4.0, 8.0]


def test_a_delivered_frame_resets_the_backoff() -> None:
    """A link that drops once an hour should not creep towards the cap over a week."""
    stream, slept = _stream([FakeCapture(1), FakeCapture(1), FakeCapture(0)], max_reconnects=2)

    frames = list(stream.frames())

    assert len(frames) == 2
    assert slept[0] == 2.0 and slept[1] == 2.0  # not 2.0 then 4.0


def test_close_stops_the_loop() -> None:
    capture = FakeCapture(100)
    stream, _ = _stream([capture])

    taken = []
    for frame in stream.frames():
        taken.append(frame)
        if len(taken) == 2:
            stream.close()

    assert len(taken) == 2
    assert capture.released


def test_stride_must_be_positive() -> None:
    stream, _ = _stream([FakeCapture(1)])
    with pytest.raises(ValueError, match="at least 1"):
        list(stream.frames(stride=0))

"""RTSP frame source with automatic reconnect.

v1 opened the stream once and had no recovery path; a dropped connection ended the
run. For unsupervised operation the camera link must recover on its own. This reader
yields frames and transparently reconnects with backoff when the stream stalls or
closes.

The OpenCV calls sit behind two injectable seams, `capture_factory` and `sleep`, so the
reconnect policy is unit-testable with a fake capture rather than a real camera and real
wall-clock waits. That policy is the part that has to be right, since nobody is watching
when it runs.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import numpy as np


class _StreamStalled(Exception):
    """Internal signal: the capture stopped delivering frames."""


def _open_rtsp(url: str):
    """Default capture factory. Imported lazily so tests need no OpenCV."""
    import cv2

    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    # Keep the buffer at one frame. The default queues frames, and on a slow CPU the
    # pipeline would drift further and further behind real time, logging a crossing
    # minutes after the vehicle actually passed. `set` reports False on backends that do
    # not support the property, which is fine; there is nothing to fall back to.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class ReconnectingStream:
    """Yields frames from an RTSP source, reconnecting on failure.

    Uses opencv-python-headless (no display) so it runs on a server with no GUI.

    The stream is treated as endless: end-of-stream from a live camera means the link
    dropped, so it is a reconnect and not a reason to stop. `max_reconnects` bounds that
    for tests and for finite video files.
    """

    def __init__(
        self,
        url: str,
        *,
        reconnect_backoff_seconds: float = 2.0,
        max_backoff_seconds: float = 60.0,
        max_reconnects: int | None = None,
        capture_factory: Callable[[str], object] = _open_rtsp,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.url = url
        self.reconnect_backoff_seconds = reconnect_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.max_reconnects = max_reconnects
        self._capture_factory = capture_factory
        self._sleep = sleep
        self._capture = None
        self._closed = False
        # counters /healthz reads
        self.frames_read = 0
        self.reconnects = 0

    def backoff_for(self, failures: int) -> float:
        """Seconds to wait after `failures` consecutive failures.

        Exponential and capped. A camera that is down stays down for a while, so retrying
        every two seconds for hours just fills the log; the cap keeps recovery prompt once
        it comes back.
        """
        delay = self.reconnect_backoff_seconds * (2 ** max(0, failures - 1))
        return min(delay, self.max_backoff_seconds)

    def frames(self, *, stride: int = 1) -> Iterator[np.ndarray]:
        """Yield frames, skipping `stride - 1` between each to stay light on CPU.

        Skipped frames are grabbed rather than slept through. `grab()` pulls a frame off
        the socket without decoding it, which is the cheap part; skipping by sleeping
        would let the camera's buffer run ahead and the log would fall behind real time.
        """
        if stride < 1:
            raise ValueError(f"stride must be at least 1, got {stride}")

        failures = 0
        while not self._closed:
            capture = self._connect()
            if capture is None:
                failures += 1
                if not self._wait_or_give_up(failures):
                    return
                continue

            try:
                while not self._closed:
                    for _ in range(stride - 1):
                        if not capture.grab():
                            raise _StreamStalled
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        raise _StreamStalled
                    self.frames_read += 1
                    failures = 0  # a delivered frame is the only proof the link is good
                    yield frame
            except _StreamStalled:
                self.close(_final=False)
                failures += 1
                if not self._wait_or_give_up(failures):
                    return

    def _connect(self):
        """Open the capture, or return None if it will not open."""
        capture = self._capture_factory(self.url)
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            return None
        self._capture = capture
        return capture

    def _wait_or_give_up(self, failures: int) -> bool:
        """Back off before the next attempt. False means stop retrying."""
        if self.max_reconnects is not None and failures > self.max_reconnects:
            return False
        self.reconnects += 1
        self._sleep(self.backoff_for(failures))
        return not self._closed

    def close(self, *, _final: bool = True) -> None:
        """Release the underlying capture, and stop the frame loop when called publicly."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if _final:
            self._closed = True

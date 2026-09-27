"""Watchdog agent: keeps the pipeline itself alive and healthy.

Planned, not implemented yet.

v1 was meant to run for hours unattended but had no supervision of its own health.
The watchdog closes that gap. It reviews pipeline health signals -- last frame time per
camera, per-stage heartbeats, disk headroom, error rates -- and either self-heals
(reconnect a camera, restart a stalled stage) or, if it can't, alerts a human.

Kept deliberately simple and cheap; it runs often. Escalates to a stronger model only
when a decision is genuinely ambiguous.
"""

from __future__ import annotations

DEFAULT_MODEL = "claude-haiku-4-5"


class WatchdogAgent:
    """Supervises pipeline health and takes corrective action."""

    def __init__(self, client, tools: list, *, model: str = DEFAULT_MODEL) -> None:
        self.client = client
        self.tools = tools
        self.model = model

    def run_once(self, health: dict) -> None:
        """Assess health signals and self-heal or alert. Stub.

        `health` carries last-frame timestamps, stage heartbeats, disk usage, etc.
        """
        raise NotImplementedError

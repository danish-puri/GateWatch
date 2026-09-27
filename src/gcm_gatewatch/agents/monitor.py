"""Monitor agent: periodic anomaly detection over recent gate activity.

Planned, not implemented yet.

On an interval, reviews recent crossings and open visits (and, when useful, a Google
Cloud Vision analysis of the current frame via the inspect_frame tool) and decides
whether anything is off:
  - a vehicle that exited with no matching entry (or vice-versa)
  - an entry outside operating hours
  - a vehicle loitering at the gate
  - an unusual spike in traffic

Runs on Claude Haiku 4.5 for cheap, frequent checks, and escalates ambiguous cases to
Opus 4.8. When it decides an alert is warranted it calls the send_alert tool.
"""

from __future__ import annotations

DEFAULT_MODEL = "claude-haiku-4-5"
ESCALATION_MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = (
    "You watch a college gate for unusual vehicle activity. Review the recent log and "
    "current view, and raise an alert ONLY when something genuinely warrants a human "
    "look. Prefer silence over false alarms."
)


class MonitorAgent:
    """Runs anomaly checks on a schedule and raises alerts when warranted."""

    def __init__(
        self,
        client,
        tools: list,
        *,
        model: str = DEFAULT_MODEL,
        escalation_model: str = ESCALATION_MODEL,
        operating_hours: tuple[int, int] = (6, 21),
    ) -> None:
        self.client = client
        self.tools = tools
        self.model = model
        self.escalation_model = escalation_model
        self.operating_hours = operating_hours

    def run_once(self) -> None:
        """One anomaly-detection pass. Escalate to Opus if the Haiku pass is unsure. Stub."""
        raise NotImplementedError

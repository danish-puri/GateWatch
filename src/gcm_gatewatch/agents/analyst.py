"""Analyst agent: natural-language questions over the gate log.

Planned, not implemented yet.

Answers things like "how many buses came in today?", "when did that plate last enter?",
or "summarize yesterday's traffic." Runs a Claude tool-use loop over the shared tool
surface. Default model: Claude Opus 4.8.

This is capability v1 simply did not have -- v1 only counted and logged.
"""

from __future__ import annotations

DEFAULT_MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = (
    "You answer questions about vehicle traffic at Gate 1 of Global College of "
    "Management, Baneshwor. Use the provided tools to read the log; do not guess. "
    "If the data cannot answer the question, say so plainly."
)


class AnalystAgent:
    """Wraps a Claude tool-runner loop for read-only questions over the log."""

    def __init__(self, client, tools: list, *, model: str = DEFAULT_MODEL) -> None:
        self.client = client
        self.tools = tools
        self.model = model

    def ask(self, question: str) -> str:
        """Run the tool loop and return a natural-language answer. Stub."""
        raise NotImplementedError

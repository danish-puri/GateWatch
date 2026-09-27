"""Shared tool surface for the agents.

These are the typed functions the Claude agents may call, defined with the Anthropic
SDK's `@beta_tool` decorator and driven by `client.beta.messages.tool_runner`. The agent
loop runs on our own server (no GPU, no external sandbox), which suits an unsupervised
on-prem deployment.

Division of labour: Claude is the agent **brain** (it decides which tools to call and
reasons over the results). **Google Cloud Vision is the eyes** -- all image understanding
(OCR + frame analysis) happens through it, not through a vision-capable LLM. The tools
below hand the agent structured Cloud Vision results (labels, objects, detected text),
which the agent reasons over.

The three agents (analyst, monitor, watchdog) draw from this one surface so behaviour
stays consistent. Bind the repository / vision / ocr / alerter via `build_tools(...)`
rather than using globals, so the tools are testable.
"""

from __future__ import annotations

from gcm_gatewatch.plate.ocr import PlateOCR
from gcm_gatewatch.storage.repository import VisitRepository


def build_tools(
    repo: VisitRepository,
    *,
    vision,
    ocr: PlateOCR,
    alerter,
    frame_source,
) -> list:
    """Construct the @beta_tool callables bound to live dependencies. Stub.

    Returns the list passed to `client.beta.messages.tool_runner(tools=...)`.

    `vision` is the Google Cloud Vision inspector (labels/objects/text on a frame);
    `ocr` is the Cloud Vision plate reader; both are the "eyes." Tools to expose:
      - query_visits(since, until, direction): counts/records from the log
      - get_recent_frame(camera): the current JPEG frame (raw bytes)
      - inspect_frame(camera): Google Cloud Vision analysis of the current frame --
          detected labels, objects, and text as structured data for the agent to reason
          over (replaces the earlier plan's Claude-vision scene description)
      - read_plate(camera): Cloud Vision OCR on a plate crop -> validated Nepali plate
          (gated off until a plate-grade camera exists; see docs/feasibility.md)
      - daily_report(date): a rolled-up summary of a day's traffic
      - send_alert(severity, message): raise an alert to security staff
    """
    raise NotImplementedError

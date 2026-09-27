"""Tests for the agent tool surface.

The tools are the boundary between Claude and our data, so they carry the tests: each
tool should return well-formed, typed results from a known repository state, and
send_alert should route to the configured alerter. The agents themselves (which call
Claude) are covered by higher-level integration tests, not unit tests.
"""

from __future__ import annotations

import pytest

from gcm_gatewatch.agents.tools import build_tools


@pytest.mark.skip(reason="scaffold: build_tools not implemented yet")
def test_build_tools_returns_expected_tool_names() -> None:
    tools = build_tools(repo=..., vision=..., ocr=..., alerter=..., frame_source=...)
    names = {t.name for t in tools}
    # inspect_frame / read_plate are Google Cloud Vision backed (the "eyes")
    assert {
        "query_visits",
        "get_recent_frame",
        "inspect_frame",
        "read_plate",
        "daily_report",
        "send_alert",
    } <= names

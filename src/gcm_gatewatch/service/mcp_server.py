"""MCP server exposing gate data as tools.

Publishes the gate log's read queries over the Model Context Protocol so any MCP client
(Claude apps, other agents) can ask about GCM gate traffic without going through our
HTTP API. Wraps the same repository the analyst agent uses.

Optional surface -- the FastAPI /ask endpoint covers the common case -- but it makes the
gate data first-class for the wider agent ecosystem.
"""

from __future__ import annotations

from gcm_gatewatch.storage.repository import VisitRepository


def build_mcp_server(repo: VisitRepository):
    """Construct and return the MCP server exposing gate-log tools. Stub."""
    raise NotImplementedError

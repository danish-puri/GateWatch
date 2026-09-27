"""FastAPI service.

Headless replacement for v1's Dash dashboard (which required a display). Exposes:
  - GET  /healthz            liveness/health for unattended operation
  - GET  /stats              today's counts for vehicles and people, plus open visits
  - GET  /crossings          the recent entry/exit log
  - GET  /daily              per-day footfall and traffic from the v_daily_flow view
  - POST /ask                natural-language question -> analyst agent (503 until it is built)

A small optional web UI can be served from here later; the API is the contract.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from gcm_gatewatch.perception.crossing import Kind


def today_bounds(now: float | None = None) -> tuple[int, int]:
    """Local midnight to now, as unix seconds.

    The gate is asked about days ("how many buses came in today"), and a UTC day would
    cut a Kathmandu evening in half, so the window is local.
    """
    now = time.time() if now is None else now
    local = time.localtime(now)
    midnight = time.mktime((local.tm_year, local.tm_mon, local.tm_mday, 0, 0, 0, 0, 0, -1))
    return int(midnight), int(now)


def create_app(pipeline, analyst) -> FastAPI:
    """Build the FastAPI app bound to the running pipeline and analyst agent."""
    app = FastAPI(
        title="GCM GateWatch",
        summary="Entry and exit log for Gate 1, Global College of Management",
        version="2.0",
    )
    repository = pipeline.repository

    @app.get("/healthz")
    def healthz() -> JSONResponse:
        """Liveness plus per-stream detail.

        Returns 503 when degraded, so systemd, a container probe or an uptime checker can
        act on it without parsing the body.
        """
        health = pipeline.health()
        return JSONResponse(health, status_code=200 if health["status"] == "ok" else 503)

    @app.get("/stats")
    def stats() -> dict:
        """Today's counts for both kinds of subject, plus vehicles still inside."""
        since, until = today_bounds()
        return {
            "since": since,
            "until": until,
            "vehicles": {
                "in": repository.count_by_type(
                    since=since, until=until, direction="in", kind=Kind.VEHICLE
                ),
                "out": repository.count_by_type(
                    since=since, until=until, direction="out", kind=Kind.VEHICLE
                ),
                "open_visits": len(repository.open_visits()),
            },
            "people": {
                "in": repository.count_by_type(
                    since=since, until=until, direction="in", kind=Kind.PERSON
                ),
                "out": repository.count_by_type(
                    since=since, until=until, direction="out", kind=Kind.PERSON
                ),
                # entries minus exits since midnight. An estimate and not a headcount,
                # since anyone already inside at midnight is not in it. See net_flow.
                "net_inside_estimate": repository.net_flow(
                    since=since, until=until, kind=Kind.PERSON
                ),
            },
        }

    @app.get("/crossings")
    def crossings(
        limit: int = Query(50, ge=1, le=1000),
        kind: str | None = Query(None, pattern="^(vehicle|person)$"),
    ) -> dict:
        """The raw entry/exit log, newest first."""
        return {"crossings": repository.recent_crossings(limit=limit, kind=kind)}

    @app.get("/daily")
    def daily(day: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$")) -> dict:
        """Per-day counts by kind, subject type and direction."""
        return {"flow": repository.daily_flow(day=day)}

    @app.post("/ask")
    def ask(question: str) -> dict:
        """Natural-language question over the log, answered by the analyst agent."""
        if analyst is None:
            raise HTTPException(
                status_code=503,
                detail="the analyst agent is not wired up yet; use /stats and /daily",
            )
        return {"question": question, "answer": analyst.ask(question)}

    return app

"""Tests for the HTTP contract.

Covers: /healthz answers 503 when a stream has gone quiet, so a probe can act on the
status code alone; /stats separates vehicles from people; /crossings filters by kind;
/daily reports the view; and /ask says so plainly while the analyst agent is unwired.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from gcm_gatewatch.perception.crossing import Crossing, Direction, Kind
from gcm_gatewatch.service.api import create_app
from gcm_gatewatch.storage.repository import VisitRepository


class FakePipeline:
    """Just the two things the API touches on a pipeline."""

    def __init__(self, repository: VisitRepository, *, status: str = "ok") -> None:
        self.repository = repository
        self._status = status

    def health(self) -> dict:
        return {"status": self._status, "detail": None, "uptime_seconds": 1.0, "streams": []}


@pytest.fixture
def repository() -> VisitRepository:
    repo = VisitRepository(":memory:")
    repo.initialize()
    return repo


def _client(repository: VisitRepository, *, status: str = "ok", analyst=None) -> TestClient:
    return TestClient(create_app(FakePipeline(repository, status=status), analyst))


def test_healthz_is_200_when_ok(repository) -> None:
    response = _client(repository).get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_healthz_is_503_when_degraded(repository) -> None:
    """A probe should not have to parse the body to know something is wrong."""
    response = _client(repository, status="degraded").get("/healthz")
    assert response.status_code == 503


def test_stats_separates_vehicles_from_people(repository) -> None:
    now = int(time.time())
    repository.record_crossing(Crossing(1, "bus", Direction.IN), camera="entry", ts=now)
    repository.record_crossing(
        Crossing(2, "person", Direction.IN, Kind.PERSON), camera="entry", ts=now
    )
    repository.record_crossing(
        Crossing(3, "person", Direction.OUT, Kind.PERSON), camera="exit", ts=now
    )

    body = _client(repository).get("/stats").json()

    assert body["vehicles"]["in"] == {"bus": 1}
    assert body["vehicles"]["open_visits"] == 1
    assert body["people"]["in"] == {"person": 1}
    assert body["people"]["out"] == {"person": 1}
    assert body["people"]["net_inside_estimate"] == 0


def test_crossings_filters_by_kind(repository) -> None:
    repository.record_crossing(Crossing(1, "car", Direction.IN), camera="entry", ts=100)
    repository.record_crossing(
        Crossing(2, "person", Direction.IN, Kind.PERSON), camera="entry", ts=200
    )
    client = _client(repository)

    assert len(client.get("/crossings").json()["crossings"]) == 2
    people = client.get("/crossings", params={"kind": "person"}).json()["crossings"]
    assert [row["subject_type"] for row in people] == ["person"]


def test_crossings_rejects_an_unknown_kind(repository) -> None:
    response = _client(repository).get("/crossings", params={"kind": "bicycle"})
    assert response.status_code == 422


def test_daily_reports_the_view(repository) -> None:
    repository.record_crossing(Crossing(1, "bus", Direction.IN), camera="entry", ts=1_700_000_000)
    flow = _client(repository).get("/daily").json()["flow"]
    assert flow and flow[0]["kind"] == "vehicle"


def test_ask_is_explicit_while_the_analyst_is_unwired(repository) -> None:
    response = _client(repository).post("/ask", params={"question": "how many buses today"})
    assert response.status_code == 503
    assert "not wired up yet" in response.json()["detail"]

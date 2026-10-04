"""API integration tests using Starlette's TestClient (background tasks run in its loop)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from grader.api import create_app
from grader.config import Settings
from grader.jobs import InMemoryJobStore, new_job
from grader.llm.fake import FakeLLMClient
from grader.models import JobStatus
from tests.conftest import RUBRICS_DIR


def _settings(**overrides) -> Settings:
    base = {
        "USE_FAKE_LLM": True,
        "rubrics_dir": RUBRICS_DIR,
        "log_level": "WARNING",
        "max_concurrency": 2,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[call-arg]


@pytest.fixture
def client():
    app = create_app(settings=_settings(), llm=FakeLLMClient())
    with TestClient(app) as test_client:
        yield test_client


def _payload(conversations, rubric_id="sales_v1"):
    return {
        "rubric_id": rubric_id,
        "conversations": [c.model_dump(mode="json") for c in conversations],
    }


def _wait_done(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in (JobStatus.DONE, JobStatus.FAILED):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_health_and_rubrics(client: TestClient):
    assert client.get("/health").json()["status"] == "ok"
    rubrics = client.get("/rubrics").json()
    assert rubrics[0]["id"] == "sales_v1"
    assert "compliance" in rubrics[0]["criteria"]


def test_job_lifecycle_and_report(client: TestClient, conversations):
    response = client.post("/jobs", json=_payload(conversations))
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == JobStatus.QUEUED
    job_id = body["job_id"]

    job = _wait_done(client, job_id)
    assert job["status"] == JobStatus.DONE
    assert job["progress"] == {
        "total": len(conversations),
        "completed": len(conversations),
        "failed": 0,
    }
    assert len(job["results"]) == len(conversations)
    assert job["usage"]["input_tokens"] > 0

    report = client.get(f"/jobs/{job_id}/report").json()
    assert report["job_id"] == job_id
    assert report["conversations"] == len(conversations)
    assert {c["criterion_id"] for c in report["by_criterion"]} == {
        "opening",
        "discovery",
        "objection_handling",
        "next_steps",
        "compliance",
    }
    agents = report["by_agent"]
    assert sum(a["conversations"] for a in agents) == len(conversations)
    assert agents == sorted(agents, key=lambda a: a["mean_overall"], reverse=True)
    assert any(r["code"] == "undue_promise" for r in report["top_risks"])
    # The report mean must match the per-conversation results it was built from.
    expected_mean = sum(r["overall_score"] for r in job["results"]) / len(job["results"])
    assert abs(report["mean_overall"] - expected_mean) < 1e-3


def test_unknown_rubric_is_404(client: TestClient, conversations):
    response = client.post("/jobs", json=_payload(conversations[:1], rubric_id="nope"))
    assert response.status_code == 404


def test_unknown_job_is_404(client: TestClient):
    assert client.get("/jobs/does-not-exist").status_code == 404
    assert client.get("/jobs/does-not-exist/report").status_code == 404


def test_duplicate_ids_rejected(client: TestClient, conversations):
    response = client.post("/jobs", json=_payload([conversations[0], conversations[0]]))
    assert response.status_code == 422
    assert "duplicate" in response.json()["detail"]


def test_invalid_payload_is_422(client: TestClient):
    response = client.post(
        "/jobs",
        json={
            "rubric_id": "sales_v1",
            "conversations": [{"id": "x", "agent_id": "a", "messages": []}],
        },
    )
    assert response.status_code == 422
    response = client.post(
        "/jobs",
        json={
            "rubric_id": "sales_v1",
            "conversations": [
                {"id": "x", "agent_id": "a", "messages": [{"role": "alien", "text": "hi"}]}
            ],
        },
    )
    assert response.status_code == 422


def test_limits_are_enforced(conversations):
    app = create_app(
        settings=_settings(max_conversations_per_job=1, max_messages_per_conversation=3),
        llm=FakeLLMClient(),
    )
    with TestClient(app) as client:
        assert client.post("/jobs", json=_payload(conversations[:2])).status_code == 413
        assert client.post("/jobs", json=_payload([conversations[0]])).status_code == 413


def test_report_on_queued_job_is_409():
    store = InMemoryJobStore()
    app = create_app(settings=_settings(), llm=FakeLLMClient(), store=store)
    with TestClient(app) as client:
        job = new_job("sales_v1", total=1)
        client.portal.call(store.create, job)
        response = client.get(f"/jobs/{job.id}/report")
        assert response.status_code == 409


def test_api_key_protection(conversations):
    app = create_app(settings=_settings(API_KEY="secret-123"), llm=FakeLLMClient())
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200  # health is public
        assert client.get("/rubrics").status_code == 401
        assert client.post("/jobs", json=_payload(conversations[:1])).status_code == 401
        assert client.get("/rubrics", headers={"X-API-Key": "wrong"}).status_code == 401
        ok = client.post(
            "/jobs", json=_payload(conversations[:1]), headers={"X-API-Key": "secret-123"}
        )
        assert ok.status_code == 202


def test_get_app_factory_builds_without_injection(monkeypatch):
    from grader import api

    monkeypatch.setattr(api, "get_settings", lambda: _settings())
    with TestClient(api.get_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.app.state.llm.model == "fake-judge-v1"


def test_real_llm_is_selected_when_fake_disabled(monkeypatch):
    from grader import api

    class DummyAnthropic:
        def __init__(self, **kwargs):
            self.messages = None

    import anthropic

    monkeypatch.setattr(anthropic, "Anthropic", DummyAnthropic)
    llm = api._build_llm(_settings(USE_FAKE_LLM=False, LLM_MODEL="claude-opus-5-5"))
    assert type(llm).__name__ == "AnthropicLLMClient"
    assert llm.model == "claude-opus-5-5"

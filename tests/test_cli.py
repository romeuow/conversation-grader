from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from grader.cli import app, load_conversations
from grader.config import PROJECT_ROOT

runner = CliRunner()
EXAMPLES = PROJECT_ROOT / "examples" / "conversations.jsonl"
EVALS = PROJECT_ROOT / "evals" / "sales_v1.yaml"


@pytest.fixture(autouse=True)
def quiet_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from grader.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_run_table(capfd):
    result = runner.invoke(app, ["run", str(EXAMPLES), "--rubric", "sales_v1", "--fake"])
    assert result.exit_code == 0, result.output
    assert "conv-001" in result.output and "compliance" in result.output
    assert "mean_overall=" in result.output
    assert "top risks:" in result.output


def test_run_json():
    result = runner.invoke(app, ["run", str(EXAMPLES), "--fake", "-o", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert len(payload["results"]) == 8
    assert payload["report"]["rubric_id"] == "sales_v1"


def test_run_unknown_rubric():
    result = runner.invoke(app, ["run", str(EXAMPLES), "--rubric", "nope", "--fake"])
    assert result.exit_code == 2
    assert "unknown rubric" in result.output


def test_run_real_without_key_fails():
    result = runner.invoke(app, ["run", str(EXAMPLES), "--real"])
    assert result.exit_code == 2
    assert "ANTHROPIC_API_KEY" in result.output


def test_run_rejects_bad_jsonl(tmp_path: Path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x"}\n', encoding="utf-8")
    result = runner.invoke(app, ["run", str(bad), "--fake"])
    assert result.exit_code != 0
    assert "bad.jsonl:1" in result.output


def test_load_conversations_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "c.jsonl"
    path.write_text(
        '\n{"id":"a","agent_id":"x","messages":[{"role":"agent","text":"oi"}]}\n\n',
        encoding="utf-8",
    )
    assert [c.id for c in load_conversations(path)] == ["a"]


def test_eval_passes():
    result = runner.invoke(app, ["eval", "--cases", str(EVALS), "--fake"])
    assert result.exit_code == 0, result.output
    assert "-> PASS" in result.output


def test_eval_fails_on_regression(tmp_path: Path):
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        """
rubric_id: sales_v1
default_tolerance: 0
cases:
  - id: drift
    expected: {opening: 5}
    conversation:
      id: d
      agent_id: a
      messages:
        - {role: agent, text: "Quer contratar?"}
""",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["eval", "--cases", str(suite), "--fake"])
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_serve_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch):
    import uvicorn

    calls: list[tuple] = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: calls.append((a, k)))
    result = runner.invoke(app, ["serve", "--port", "9999"])
    assert result.exit_code == 0
    assert calls[0][0] == ("grader.api:get_app",)
    assert calls[0][1]["port"] == 9999 and calls[0][1]["factory"] is True

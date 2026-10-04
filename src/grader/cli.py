"""Command line interface: `grader run`, `grader eval`, `grader serve`."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from grader.aggregate import build_report
from grader.config import get_settings
from grader.evals import EvalReport, load_eval_suite, run_eval_suite
from grader.graph.build import ConversationGrader
from grader.llm.base import LLMClient
from grader.log import configure_logging
from grader.models import Conversation, ConversationResult, TokenUsage
from grader.rubric import RubricError, RubricRegistry

app = typer.Typer(help="LLM-as-judge grader for sales/support conversations.", no_args_is_help=True)


def _llm(fake: bool | None) -> LLMClient:
    settings = get_settings()
    use_fake = settings.use_fake_llm if fake is None else fake
    if use_fake:
        from grader.llm.fake import FakeLLMClient

        return FakeLLMClient()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        typer.echo("ANTHROPIC_API_KEY is not set; use --fake for demo mode.", err=True)
        raise typer.Exit(code=2)
    from grader.llm.anthropic_client import AnthropicLLMClient

    return AnthropicLLMClient(model=settings.llm_model, max_tokens=settings.llm_max_tokens)


def load_conversations(path: Path) -> list[Conversation]:
    """Read a JSONL file (one conversation per line)."""
    conversations: list[Conversation] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                conversations.append(Conversation.model_validate_json(line))
            except ValueError as exc:
                raise typer.BadParameter(f"{path}:{line_no}: {exc}") from exc
    return conversations


def _table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows, strict=True)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*("-" * w for w in widths))]
    lines += [fmt.format(*row) for row in rows]
    return "\n".join(lines)


def render_results(results: list[ConversationResult]) -> str:
    criteria_ids = [g.criterion_id for g in results[0].criteria] if results else []
    headers = ["conversation", "agent", "overall", *criteria_ids, "risks"]
    rows = []
    for result in results:
        scores = {g.criterion_id: g for g in result.criteria}
        row = [result.conversation_id, result.agent_id, f"{result.overall_score:.2f}"]
        for cid in criteria_ids:
            grade = scores.get(cid)
            mark = "" if grade is None or grade.evidence_verified else "*"
            row.append(f"{grade.score}{mark}" if grade else "-")
        row.append(",".join(r.code for r in result.risks) or "-")
        rows.append(row)
    return _table(rows, headers) + "\n(* = evidence not verified)"


def render_eval(report: EvalReport) -> str:
    rows = []
    for case in report.cases:
        for check in case.checks:
            rows.append(
                [
                    case.case_id,
                    check.criterion_id,
                    str(check.expected),
                    "-" if check.actual is None else str(check.actual),
                    f"±{check.tolerance}",
                    "PASS" if check.passed else "FAIL",
                ]
            )
    return _table(rows, ["case", "criterion", "expected", "actual", "tol", "status"])


@app.command()
def run(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="JSONL file")],
    rubric: Annotated[str, typer.Option(help="Rubric id (file name under rubrics/)")] = "sales_v1",
    fake: Annotated[
        bool | None, typer.Option("--fake/--real", help="Use the deterministic fake judge")
    ] = None,
    output: Annotated[str, typer.Option("--output", "-o", help="table or json")] = "table",
    rubrics_dir: Annotated[Path | None, typer.Option(help="Override rubrics directory")] = None,
) -> None:
    """Grade every conversation in a JSONL file and print the results."""
    settings = get_settings()
    configure_logging(settings.log_level)
    try:
        registry = RubricRegistry(rubrics_dir or settings.rubrics_dir)
        rubric_obj = registry.get(rubric)
    except RubricError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    grader = ConversationGrader(_llm(fake), rubric_obj, settings=settings)
    conversations = load_conversations(path)
    results = [grader.grade(c) for c in conversations]
    usage = TokenUsage()
    for result in results:
        usage = usage.add(result.usage)
    report = build_report("cli", rubric_obj.id, results, usage)
    if output == "json":
        typer.echo(
            json.dumps(
                {
                    "results": [r.model_dump(mode="json") for r in results],
                    "report": report.model_dump(mode="json"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    typer.echo(render_results(results))
    typer.echo("")
    typer.echo(f"conversations={report.conversations} mean_overall={report.mean_overall:.2f}")
    typer.echo(
        f"tokens: input={usage.input_tokens} output={usage.output_tokens} "
        f"est_cost_usd={usage.estimated_cost_usd}"
    )
    if report.top_risks:
        typer.echo("top risks: " + ", ".join(f"{r.code}x{r.count}" for r in report.top_risks))


@app.command("eval")
def eval_cmd(
    cases: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = Path("evals/sales_v1.yaml"),
    fake: Annotated[
        bool | None, typer.Option("--fake/--real", help="Use the deterministic fake judge")
    ] = None,
    rubrics_dir: Annotated[Path | None, typer.Option(help="Override rubrics directory")] = None,
) -> None:
    """Run the prompt regression suite; exit 1 if any score drifts beyond tolerance."""
    settings = get_settings()
    configure_logging(settings.log_level)
    suite = load_eval_suite(cases)
    registry = RubricRegistry(rubrics_dir or settings.rubrics_dir)
    grader = ConversationGrader(_llm(fake), registry.get(suite.rubric_id), settings=settings)
    report = run_eval_suite(suite, grader)
    typer.echo(render_eval(report))
    failed = report.failed_checks
    typer.echo("")
    typer.echo(
        f"model={report.model} checks={report.total_checks} failed={len(failed)} "
        f"-> {'PASS' if report.passed else 'FAIL'}"
    )
    if not report.passed:
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "0.0.0.0",
    port: Annotated[int, typer.Option()] = 8000,
) -> None:
    """Start the HTTP API with uvicorn."""
    import uvicorn

    uvicorn.run("grader.api:get_app", host=host, port=port, factory=True)


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

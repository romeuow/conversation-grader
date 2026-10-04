"""FastAPI application exposing jobs, reports and rubrics."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field

from grader import __version__
from grader.aggregate import build_report
from grader.config import Settings, get_settings
from grader.graph.build import ConversationGrader
from grader.jobs import InMemoryJobStore, JobRunner, JobStore, new_job
from grader.llm.base import LLMClient
from grader.log import configure_logging, get_logger
from grader.models import Conversation, Job, JobStatus, Report
from grader.rubric import RubricError, RubricRegistry

log = get_logger(__name__)


class CreateJobRequest(BaseModel):
    rubric_id: str = Field(min_length=1, max_length=64)
    conversations: list[Conversation] = Field(min_length=1)


class CreateJobResponse(BaseModel):
    job_id: str
    status: JobStatus


class RubricSummary(BaseModel):
    id: str
    version: str
    name: str
    criteria: list[str]


def _build_llm(settings: Settings) -> LLMClient:
    if settings.use_fake_llm:
        from grader.llm.fake import FakeLLMClient

        return FakeLLMClient()
    from grader.llm.anthropic_client import AnthropicLLMClient

    return AnthropicLLMClient(model=settings.llm_model, max_tokens=settings.llm_max_tokens)


def create_app(
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    store: JobStore | None = None,
    registry: RubricRegistry | None = None,
) -> FastAPI:
    """Application factory. Every dependency can be injected (used heavily by tests)."""
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.llm = llm or _build_llm(settings)
        app.state.store = store or InMemoryJobStore()
        app.state.registry = registry or RubricRegistry(settings.rubrics_dir)
        app.state.runner = JobRunner(app.state.store, max_concurrency=settings.max_concurrency)
        app.state.tasks: set[asyncio.Task[Job]] = set()
        log.info(
            "app_started",
            extra={
                "fake_llm": settings.use_fake_llm,
                "rubrics": [r.id for r in app.state.registry.list()],
            },
        )
        yield
        for task in list(app.state.tasks):
            task.cancel()

    app = FastAPI(
        title="conversation-grader",
        version=__version__,
        description="LLM-as-judge grading of sales/support conversations.",
        lifespan=lifespan,
    )

    async def require_api_key(
        request: Request, x_api_key: Annotated[str | None, Header()] = None
    ) -> None:
        expected = request.app.state.settings.api_key
        if expected and x_api_key != expected:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key")

    auth = Depends(require_api_key)

    @app.get("/health", tags=["ops"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/rubrics", response_model=list[RubricSummary], dependencies=[auth], tags=["rubrics"])
    async def list_rubrics(request: Request) -> list[RubricSummary]:
        return [
            RubricSummary(
                id=r.id, version=r.version, name=r.name, criteria=[c.id for c in r.criteria]
            )
            for r in request.app.state.registry.list()
        ]

    @app.post(
        "/jobs",
        status_code=status.HTTP_202_ACCEPTED,
        response_model=CreateJobResponse,
        dependencies=[auth],
        tags=["jobs"],
    )
    async def create_job(payload: CreateJobRequest, request: Request) -> CreateJobResponse:
        state = request.app.state
        limits: Settings = state.settings
        if len(payload.conversations) > limits.max_conversations_per_job:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"too many conversations (max {limits.max_conversations_per_job})",
            )
        for conversation in payload.conversations:
            if len(conversation.messages) > limits.max_messages_per_conversation:
                raise HTTPException(
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    detail=f"conversation {conversation.id} has too many messages",
                )
        ids = [c.id for c in payload.conversations]
        if len(ids) != len(set(ids)):
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_CONTENT, detail="duplicate conversation ids"
            )
        try:
            rubric = state.registry.get(payload.rubric_id)
        except RubricError as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

        job = new_job(rubric.id, total=len(payload.conversations))
        await state.store.create(job)
        grader = ConversationGrader(state.llm, rubric, settings=limits)
        task = asyncio.create_task(state.runner.run(job.id, payload.conversations, grader))
        state.tasks.add(task)
        task.add_done_callback(state.tasks.discard)
        log.info(
            "job_accepted", extra={"job_id": job.id, "rubric_id": rubric.id, "total": len(ids)}
        )
        return CreateJobResponse(job_id=job.id, status=job.status)

    async def _get_job(request: Request, job_id: str) -> Job:
        job = await request.app.state.store.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="job not found")
        return job

    @app.get("/jobs/{job_id}", response_model=Job, dependencies=[auth], tags=["jobs"])
    async def get_job(job_id: str, request: Request) -> Job:
        return await _get_job(request, job_id)

    @app.get("/jobs/{job_id}/report", response_model=Report, dependencies=[auth], tags=["jobs"])
    async def get_report(job_id: str, request: Request) -> Report:
        job = await _get_job(request, job_id)
        if job.status in (JobStatus.QUEUED,):
            raise HTTPException(status.HTTP_409_CONFLICT, detail="job has not started yet")
        return build_report(job.id, job.rubric_id, job.results, job.usage)

    return app


def get_app() -> FastAPI:
    """Entry point for `uvicorn grader.api:get_app --factory`."""
    return create_app()

"""Asynchronous job orchestration with a pluggable store."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime
from typing import Protocol

from grader.graph.build import ConversationGrader
from grader.log import get_logger
from grader.models import Conversation, ConversationResult, Job, JobProgress, JobStatus, TokenUsage

log = get_logger(__name__)


class JobStore(Protocol):
    """Persistence boundary for jobs. Swap for Redis/Postgres without touching the runner."""

    async def create(self, job: Job) -> None: ...

    async def get(self, job_id: str) -> Job | None: ...

    async def save(self, job: Job) -> None: ...

    async def list(self) -> list[Job]: ...


class InMemoryJobStore:
    """Dict-backed store guarded by an asyncio lock. State is lost on restart."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = asyncio.Lock()

    async def create(self, job: Job) -> None:
        async with self._lock:
            self._jobs[job.id] = job.model_copy(deep=True)

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    async def save(self, job: Job) -> None:
        async with self._lock:
            self._jobs[job.id] = job.model_copy(deep=True)

    async def list(self) -> list[Job]:
        async with self._lock:
            return [j.model_copy(deep=True) for j in self._jobs.values()]


def new_job(rubric_id: str, total: int) -> Job:
    now = datetime.now(UTC)
    return Job(
        id=uuid.uuid4().hex,
        rubric_id=rubric_id,
        created_at=now,
        updated_at=now,
        progress=JobProgress(total=total, completed=0),
    )


class JobRunner:
    """Grades every conversation of a job with bounded concurrency and persists progress."""

    def __init__(self, store: JobStore, *, max_concurrency: int = 4) -> None:
        self._store = store
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))

    async def run(
        self, job_id: str, conversations: list[Conversation], grader: ConversationGrader
    ) -> Job:
        job = await self._store.get(job_id)
        if job is None:
            raise KeyError(job_id)
        job.status = JobStatus.RUNNING
        job.updated_at = datetime.now(UTC)
        await self._store.save(job)
        log.info("job_started", extra={"job_id": job_id, "total": len(conversations)})

        try:
            await asyncio.gather(
                *(self._grade_one(job_id, conversation, grader) for conversation in conversations)
            )
            job = await self._store.get(job_id)
            assert job is not None
            job.status = JobStatus.DONE
        except Exception as exc:  # pragma: no cover - defensive: graph errors are caught per item
            log.exception("job_failed", extra={"job_id": job_id})
            job = await self._store.get(job_id)
            assert job is not None
            job.status = JobStatus.FAILED
            job.error = str(exc)
        job.updated_at = datetime.now(UTC)
        await self._store.save(job)
        log.info(
            "job_finished",
            extra={
                "job_id": job_id,
                "status": job.status,
                "completed": job.progress.completed,
                "failed": job.progress.failed,
                "input_tokens": job.usage.input_tokens,
                "output_tokens": job.usage.output_tokens,
                "estimated_cost_usd": job.usage.estimated_cost_usd,
            },
        )
        return job

    async def _grade_one(
        self, job_id: str, conversation: Conversation, grader: ConversationGrader
    ) -> None:
        async with self._semaphore:
            try:
                result = await grader.agrade(conversation)
            except Exception as exc:
                log.exception(
                    "conversation_failed",
                    extra={"job_id": job_id, "conversation_id": conversation.id},
                )
                result = ConversationResult(
                    conversation_id=conversation.id,
                    agent_id=conversation.agent_id,
                    channel=conversation.channel,
                    rubric_id=grader.rubric.id,
                    rubric_version=grader.rubric.version,
                    overall_score=0.0,
                    criteria=[],
                    risks=[],
                    pii_redactions={},
                    usage=TokenUsage(),
                    error=f"{type(exc).__name__}: {exc}",
                )
        job = await self._store.get(job_id)
        if job is None:  # pragma: no cover
            return
        job.results.append(result)
        job.progress.completed += 1
        if result.error:
            job.progress.failed += 1
        job.usage = job.usage.add(result.usage)
        job.updated_at = datetime.now(UTC)
        await self._store.save(job)

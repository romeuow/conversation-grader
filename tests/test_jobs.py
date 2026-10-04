from __future__ import annotations

import pytest

from grader.jobs import InMemoryJobStore, JobRunner, new_job
from grader.models import JobStatus


async def test_store_crud():
    store = InMemoryJobStore()
    job = new_job("sales_v1", total=2)
    assert await store.get(job.id) is None
    await store.create(job)
    fetched = await store.get(job.id)
    assert fetched == job and fetched is not job  # copies, never shared references
    fetched.status = JobStatus.RUNNING
    assert (await store.get(job.id)).status == JobStatus.QUEUED
    await store.save(fetched)
    assert (await store.get(job.id)).status == JobStatus.RUNNING
    assert [j.id for j in await store.list()] == [job.id]


async def test_runner_tracks_progress_and_usage(grader, conversations):
    store = InMemoryJobStore()
    job = new_job("sales_v1", total=len(conversations))
    await store.create(job)
    runner = JobRunner(store, max_concurrency=3)
    finished = await runner.run(job.id, conversations, grader)
    assert finished.status == JobStatus.DONE
    assert finished.progress.completed == len(conversations)
    assert finished.progress.failed == 0
    assert {r.conversation_id for r in finished.results} == {c.id for c in conversations}
    assert finished.usage.input_tokens == sum(r.usage.input_tokens for r in finished.results)
    assert finished.updated_at >= finished.created_at


async def test_runner_isolates_conversation_failures(grader, conversations, monkeypatch):
    async def boom(conversation):
        if conversation.id == "conv-002":
            raise RuntimeError("graph exploded")
        return await type(grader).agrade(grader, conversation)

    monkeypatch.setattr(grader, "agrade", boom)
    store = InMemoryJobStore()
    job = new_job("sales_v1", total=3)
    await store.create(job)
    finished = await JobRunner(store).run(job.id, conversations[:3], grader)
    assert finished.status == JobStatus.DONE
    assert finished.progress.failed == 1
    failed = next(r for r in finished.results if r.conversation_id == "conv-002")
    assert failed.error == "RuntimeError: graph exploded"
    assert failed.criteria == []


async def test_runner_unknown_job(grader):
    with pytest.raises(KeyError):
        await JobRunner(InMemoryJobStore()).run("missing", [], grader)

"""In-process job queue for slow work (transcription, generation, rendering).

Swap for a real worker (RQ, Celery, arq) when needed; the routes only use
submit() and get(). Every job saves progress to the project, so an
interrupted job can be retried from the last good step.
"""
from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable, Optional


@dataclass
class Job:
    id: str
    project_id: str
    label: str
    status: str = "queued"  # queued | running | done | failed
    stage: str = ""
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


JOBS: dict[str, Job] = {}
_tasks: set[asyncio.Task] = set()
Work = Callable[[Callable[[str], None]], Awaitable[None]]


def submit(project_id: str, label: str, work: Work) -> Job:
    job = Job(id=f"job_{uuid.uuid4().hex[:8]}", project_id=project_id, label=label)
    JOBS[job.id] = job

    def progress(stage: str) -> None:
        job.stage = stage

    async def run() -> None:
        job.status = "running"
        try:
            await work(progress)
            job.status = "done"
        except Exception as e:  # surfaced to the UI as a clear error
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            traceback.print_exc()
        finally:
            job.finished_at = time.time()

    task = asyncio.create_task(run())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return job


def get(job_id: str) -> Optional[Job]:
    return JOBS.get(job_id)


def for_project(project_id: str) -> list[dict]:
    return [asdict(j) for j in sorted(JOBS.values(), key=lambda j: -j.created_at) if j.project_id == project_id]


async def wait_all() -> None:
    while _tasks:
        await asyncio.gather(*list(_tasks), return_exceptions=True)

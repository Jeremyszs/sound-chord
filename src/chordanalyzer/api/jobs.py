"""In-memory store for async jobs.

Phase 4 of the plan replaces this with RQ + Redis so jobs survive restarts and
can run on separate worker machines; the HTTP contract stays the same.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    status: str = "queued"
    created: float = field(default_factory=time.time)
    result: dict | None = None
    error: dict | None = None
    task: asyncio.Task | None = None


class JobStore:
    def __init__(self, ttl_s: float = 24 * 3600, max_jobs: int = 10_000):
        self.ttl_s, self.max_jobs = ttl_s, max_jobs
        self._jobs: dict[str, Job] = {}

    def _expire(self) -> None:
        cutoff = time.time() - self.ttl_s
        for jid in [j.id for j in self._jobs.values()
                    if j.created < cutoff and j.status in ("succeeded", "failed")]:
            del self._jobs[jid]
        while len(self._jobs) >= self.max_jobs:
            self._jobs.pop(next(iter(self._jobs)))

    def create(self) -> Job:
        self._expire()
        job = Job(id=uuid.uuid4().hex)
        self._jobs[job.id] = job
        return job

    def attach(self, job_id: str, task: asyncio.Task) -> None:
        self._jobs[job_id].task = task

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def set_running(self, job_id: str) -> None:
        self._jobs[job_id].status = "running"

    def succeed(self, job_id: str, result: dict) -> None:
        job = self._jobs[job_id]
        job.status, job.result, job.task = "succeeded", result, None

    def fail(self, job_id: str, error: dict) -> None:
        job = self._jobs[job_id]
        job.status, job.error, job.task = "failed", error, None

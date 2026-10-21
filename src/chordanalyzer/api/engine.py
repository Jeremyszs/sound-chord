"""Runs analyses off the event loop: a process pool, with models loaded once per worker."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import threading
from collections.abc import Callable
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from typing import Any

from ..pipeline import AnalysisOptions, Analyzer
from ..settings import Settings

_worker_analyzer: Analyzer | None = None


def _init_worker(settings: Settings) -> None:
    global _worker_analyzer
    _worker_analyzer = Analyzer(settings)


def _worker_info() -> dict[str, Any]:
    assert _worker_analyzer is not None
    return {"engine": _worker_analyzer.engine_info().model_dump(),
            "beat_tracker_fallback": bool(_worker_analyzer.beat_warnings)}


def _worker_analyze(path: str, options: dict, sha256: str) -> tuple[dict, dict]:
    assert _worker_analyzer is not None
    result = _worker_analyzer.analyze(path, AnalysisOptions(**options), sha256=sha256)
    return result.model_dump(mode="json", exclude_none=True), _worker_analyzer.last_timings.steps


class Engine:
    """Bounded front door to the analysis workers.

    `capacity` is the number of analyses allowed in flight (running plus waiting);
    beyond that `submit` raises QueueFull so the API can answer 503.
    """

    def __init__(self, settings: Settings, executor: Executor | None = None):
        self.settings = settings
        self.capacity = settings.workers + settings.max_queue
        self._in_flight = 0
        self._lock = threading.Lock()
        self.ready = False
        self.info: dict[str, Any] = {}
        self.error: str | None = None
        if executor is None:
            executor = ProcessPoolExecutor(max_workers=settings.workers,
                                           mp_context=mp.get_context("spawn"),
                                           initializer=_init_worker, initargs=(settings,))
        self.executor = executor

    @classmethod
    def in_process(cls, settings: Settings, analyzer: Analyzer) -> Engine:
        """Run analyses in a thread with an already-loaded analyzer (tests, small deployments)."""
        global _worker_analyzer
        _worker_analyzer = analyzer
        return cls(settings, ThreadPoolExecutor(max_workers=1))

    async def warm_up(self) -> None:
        """Load the models in every worker; the health check reports ready afterwards."""
        loop = asyncio.get_running_loop()
        try:
            n = getattr(self.executor, "_max_workers", 1)
            infos = await asyncio.gather(*[loop.run_in_executor(self.executor, _worker_info)
                                           for _ in range(n)])
            self.info = infos[0]
            self.ready = True
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            raise

    def _acquire(self) -> None:
        with self._lock:
            if self._in_flight >= self.capacity:
                raise QueueFull()
            self._in_flight += 1

    def _release(self) -> None:
        with self._lock:
            self._in_flight -= 1

    async def analyze(self, path: str, options: AnalysisOptions, sha256: str
                      ) -> tuple[dict, dict]:
        self._acquire()
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(self.executor, _worker_analyze, path,
                                              asdict(options), sha256)
        finally:
            self._release()

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)


class QueueFull(Exception):
    pass


EngineFactory = Callable[[Settings], Engine]

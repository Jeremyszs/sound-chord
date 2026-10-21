"""HTTP API: upload an audio file, get tempo, key and chords as JSON."""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .. import __version__
from ..audio import DurationOutOfRange, UnsupportedAudio
from ..pipeline import AnalysisOptions
from ..schema import AnalysisResult, ChordVocabulary, Notation
from ..settings import Settings
from .engine import Engine, QueueFull
from .jobs import JobStore

log = logging.getLogger("chordanalyzer.api")

INCLUDE_CHOICES = {"beats", "tempo_curve", "bars", "chord_probabilities"}
PROBLEM_BASE = "https://chordanalyzer.dev/problems/"
CHUNK = 1 << 20


class Problem(Exception):
    """An error answered as application/problem+json (RFC 9457)."""

    def __init__(self, status: int, slug: str, title: str, detail: str = "",
                 headers: dict[str, str] | None = None):
        self.status, self.slug, self.title, self.detail = status, slug, title, detail
        self.headers = headers or {}


def problem_body(p: Problem) -> dict:
    body = {"type": PROBLEM_BASE + p.slug, "title": p.title, "status": p.status}
    if p.detail:
        body["detail"] = p.detail
    return body


def problem_response(p: Problem) -> JSONResponse:
    return JSONResponse(problem_body(p), status_code=p.status, headers=p.headers,
                        media_type="application/problem+json")


class ProblemModel(BaseModel):
    type: str
    title: str
    status: int
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    version: str
    auth: Literal["enabled", "disabled"]
    engine: dict | None = None
    beat_tracker_fallback: bool | None = None
    error: str | None = None


class JobAccepted(BaseModel):
    job_id: str
    status: str


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    result: AnalysisResult | None = None
    error: ProblemModel | None = None


ERROR_RESPONSES = {
    s: {"model": ProblemModel, "content": {"application/problem+json": {}}}
    for s in (400, 401, 413, 415, 422, 503)
}


class ResultCache:
    """Results keyed by file hash + engine version + options. On disk if CACHE_DIR is set."""

    def __init__(self, directory: Path | None, max_memory_items: int = 256):
        self.directory = directory
        self.max_items = max_memory_items
        self.memory: dict[str, dict] = {}
        if directory:
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(sha256: str, options: AnalysisOptions) -> str:
        opts = f"{options.chord_vocabulary}|{','.join(sorted(options.include))}|{options.notation}"
        return hashlib.sha256(f"{sha256}|{__version__}|{opts}".encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        if key in self.memory:
            return self.memory[key]
        if self.directory and (self.directory / f"{key}.json").exists():
            return json.loads((self.directory / f"{key}.json").read_text())
        return None

    def put(self, key: str, value: dict) -> None:
        if len(self.memory) >= self.max_items:
            self.memory.pop(next(iter(self.memory)))
        self.memory[key] = value
        if self.directory:
            tmp = self.directory / f"{key}.{uuid.uuid4().hex}.tmp"
            tmp.write_text(json.dumps(value))
            tmp.replace(self.directory / f"{key}.json")


def parse_options(chord_vocabulary: str, include: str, notation: str) -> AnalysisOptions:
    items = [x.strip() for x in include.split(",") if x.strip()]
    bad = sorted(set(items) - INCLUDE_CHOICES)
    if bad:
        raise Problem(400, "bad-option", "Invalid option value",
                      f"unknown include value(s): {', '.join(bad)}; "
                      f"allowed: {', '.join(sorted(INCLUDE_CHOICES))}")
    return AnalysisOptions(chord_vocabulary=chord_vocabulary, include=tuple(items),
                           notation=notation)


async def save_upload(upload: UploadFile, dest_dir: str, max_bytes: int) -> tuple[str, str]:
    """Copy the upload to a temp file with a hard size cap. Returns (path, sha256)."""
    h = hashlib.sha256()
    size = 0
    fd, path = tempfile.mkstemp(prefix="upload-", dir=dest_dir)
    try:
        with os.fdopen(fd, "wb") as out:
            while chunk := await upload.read(CHUNK):
                size += len(chunk)
                if size > max_bytes:
                    raise Problem(413, "file-too-large", "File too large",
                                  f"uploads are limited to {max_bytes // (1 << 20)} MB")
                h.update(chunk)
                out.write(chunk)
    except BaseException:
        os.unlink(path)
        raise
    if size == 0:
        os.unlink(path)
        raise Problem(400, "missing-file", "Missing file", "the uploaded file is empty")
    return path, h.hexdigest()


def create_app(settings: Settings | None = None, engine: Engine | None = None) -> FastAPI:
    settings = settings or Settings()
    key_hashes = [hashlib.sha256(k.encode()).digest() for k in settings.api_keys]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not key_hashes:
            log.warning("API_KEYS is not set: authentication is disabled")
        app.state.engine = engine or Engine(settings)
        app.state.jobs = JobStore()
        warm = asyncio.create_task(app.state.engine.warm_up())
        warm.add_done_callback(lambda t: t.exception() and log.error(
            "model loading failed: %s", t.exception()))
        yield
        warm.cancel()
        app.state.engine.shutdown()

    app = FastAPI(
        title="Chord Analyzer API",
        version=__version__,
        description=("Upload an audio file and get its tempo, key and beat-aligned chord "
                     "progression as JSON. Send the API key in the X-API-Key header."),
        lifespan=lifespan,
    )
    if settings.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins,
                           allow_methods=["GET", "POST"], allow_headers=["*"])
    cache = ResultCache(settings.cache_dir)

    @app.exception_handler(Problem)
    async def _problem(_: Request, exc: Problem):
        return problem_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        missing_file = any(e.get("loc", [None, None])[-1] == "file" for e in exc.errors())
        if missing_file:
            return problem_response(Problem(400, "missing-file", "Missing file",
                                            "send the audio as a multipart field named 'file'"))
        detail = "; ".join(f"{'.'.join(str(x) for x in e['loc'][1:])}: {e['msg']}"
                           for e in exc.errors())
        return problem_response(Problem(400, "bad-option", "Invalid option value", detail))

    def check_key(api_key: str | None) -> None:
        if not key_hashes:
            return
        given = hashlib.sha256((api_key or "").encode()).digest()
        if not any(hmac.compare_digest(given, k) for k in key_hashes):
            raise Problem(401, "unauthorized", "Missing or invalid API key",
                          "send a valid key in the X-API-Key header")

    def check_ready(request: Request) -> Engine:
        eng: Engine = request.app.state.engine
        if not eng.ready:
            raise Problem(503, "not-ready", "Models are still loading",
                          eng.error or "try again shortly", {"Retry-After": "10"})
        return eng

    async def run_analysis(request: Request, upload: UploadFile, options: AnalysisOptions
                           ) -> tuple[dict, dict[str, str]]:
        eng = check_ready(request)
        tmp_dir = tempfile.mkdtemp(prefix="chordanalyzer-")
        started = time.perf_counter()
        path = None
        try:
            path, sha = await save_upload(upload, tmp_dir, settings.max_upload_mb << 20)
            key = cache.key(sha, options)
            cached = cache.get(key)
            if cached is not None:
                return cached, {"X-Cache": "hit"}
            try:
                result, timings = await eng.analyze(path, options, sha)
            except QueueFull:
                raise Problem(503, "busy", "Server busy", "too many analyses in progress",
                              {"Retry-After": "30"}) from None
            except UnsupportedAudio as e:
                raise Problem(415, "unsupported-media", "Unsupported audio format",
                              str(e)) from None
            except DurationOutOfRange as e:
                raise Problem(422, "duration-out-of-range", "Audio duration out of range",
                              str(e)) from None
            cache.put(key, result)
            log.info("analyzed sha256=%s in %.2fs %s", sha[:12], time.perf_counter() - started,
                     timings)
            return result, {"X-Cache": "miss"}
        finally:
            if path and os.path.exists(path):
                os.unlink(path)
            os.rmdir(tmp_dir)

    def precheck_length(request: Request) -> None:
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > (settings.max_upload_mb << 20) + CHUNK:
            raise Problem(413, "file-too-large", "File too large",
                          f"uploads are limited to {settings.max_upload_mb} MB")

    FileField = Annotated[UploadFile, File(description="The audio file (mp3, wav, flac, m4a, ogg)")]
    VocabField = Annotated[ChordVocabulary, Form(description="'majmin' (25 labels) or 'full' (170)")]
    IncludeField = Annotated[str, Form(description=(
        "Comma list of optional fields: beats, tempo_curve, bars, chord_probabilities"))]
    NotationField = Annotated[Notation, Form(description="Chord display: name, roman or both")]
    KeyHeader = Annotated[str | None, Header(alias="X-API-Key")]

    @app.post("/v1/analyze", response_model=AnalysisResult, response_model_exclude_none=True,
              responses=ERROR_RESPONSES, summary="Analyze an audio file and wait for the result")
    async def analyze(request: Request, file: FileField,
                      chord_vocabulary: VocabField = "full",
                      include: IncludeField = "beats,bars",
                      notation: NotationField = "both",
                      x_api_key: KeyHeader = None):
        check_key(x_api_key)
        precheck_length(request)
        options = parse_options(chord_vocabulary, include, notation)
        started = time.perf_counter()
        result, headers = await run_analysis(request, file, options)
        headers["X-Processing-Time"] = f"{time.perf_counter() - started:.3f}"
        return JSONResponse(result, headers=headers)

    @app.post("/v1/jobs", status_code=202, response_model=JobAccepted, responses=ERROR_RESPONSES,
              summary="Upload an audio file and return immediately; poll the job for the result")
    async def create_job(request: Request, file: FileField,
                         chord_vocabulary: VocabField = "full",
                         include: IncludeField = "beats,bars",
                         notation: NotationField = "both",
                         x_api_key: KeyHeader = None):
        check_key(x_api_key)
        precheck_length(request)
        options = parse_options(chord_vocabulary, include, notation)
        check_ready(request)
        # Save now: the upload stream is closed once this request returns.
        tmp_dir = tempfile.mkdtemp(prefix="chordanalyzer-job-")
        path, sha = await save_upload(file, tmp_dir, settings.max_upload_mb << 20)
        jobs: JobStore = request.app.state.jobs
        job = jobs.create()

        async def run():
            jobs.set_running(job.id)
            try:
                key = cache.key(sha, options)
                result = cache.get(key)
                if result is None:
                    result, _ = await request.app.state.engine.analyze(path, options, sha)
                    cache.put(key, result)
                jobs.succeed(job.id, result)
            except QueueFull:
                jobs.fail(job.id, problem_body(Problem(503, "busy", "Server busy",
                                                       "too many analyses in progress")))
            except UnsupportedAudio as e:
                jobs.fail(job.id, problem_body(Problem(415, "unsupported-media",
                                                       "Unsupported audio format", str(e))))
            except DurationOutOfRange as e:
                jobs.fail(job.id, problem_body(Problem(422, "duration-out-of-range",
                                                       "Audio duration out of range", str(e))))
            except Exception:  # noqa: BLE001
                log.exception("job %s failed", job.id)
                jobs.fail(job.id, problem_body(Problem(500, "internal", "Analysis failed")))
            finally:
                os.unlink(path)
                os.rmdir(tmp_dir)

        jobs.attach(job.id, asyncio.create_task(run()))
        return JSONResponse({"job_id": job.id, "status": "queued"}, status_code=202,
                            headers={"Location": f"/v1/jobs/{job.id}"})

    @app.get("/v1/jobs/{job_id}", response_model=JobStatus, response_model_exclude_none=True,
             responses={404: ERROR_RESPONSES[400], 401: ERROR_RESPONSES[401]},
             summary="Poll an async job")
    async def get_job(request: Request, job_id: str, x_api_key: KeyHeader = None):
        check_key(x_api_key)
        job = request.app.state.jobs.get(job_id)
        if job is None:
            raise Problem(404, "job-not-found", "Job not found",
                          "unknown job id, or the job expired")
        body = {"job_id": job.id, "status": job.status}
        if job.result is not None:
            body["result"] = job.result
        if job.error is not None:
            body["error"] = job.error
        return JSONResponse(body)

    @app.get("/v1/health", response_model=HealthResponse, response_model_exclude_none=True,
             summary="Liveness and loaded model versions")
    async def health(request: Request):
        eng: Engine = request.app.state.engine
        body = {"status": "ok" if eng.ready else ("error" if eng.error else "loading"),
                "version": __version__, "auth": "enabled" if key_hashes else "disabled"}
        if eng.ready:
            body.update(eng.info)
        if eng.error:
            body["error"] = eng.error
        return JSONResponse(body, status_code=200 if eng.ready else 503)

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(create_app(), host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()

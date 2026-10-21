"""API behaviour with a stub analyzer (fast, no models needed)."""

import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from chordanalyzer.api.app import create_app
from chordanalyzer.api.engine import Engine
from chordanalyzer.audio import load
from chordanalyzer.pipeline import Timings
from chordanalyzer.schema import AnalysisResult, EngineInfo
from chordanalyzer.settings import Settings

FIXTURES = Path(__file__).parent / "fixtures"


class StubAnalyzer:
    beat_warnings: list[str] = []

    def __init__(self):
        self.calls = 0
        self.last_timings = Timings()

    def engine_info(self, vocabulary="full"):
        return EngineInfo(version="test", chord_model=f"stub-{vocabulary}", beat_model="stub",
                          key_model="stub")

    def analyze(self, path, options, sha256=None):
        self.calls += 1
        audio = load(path)  # real validation, so error paths are exercised
        result = json.loads((FIXTURES / "stub_result.json").read_text())
        result["file"]["sha256"] = sha256
        result["file"]["duration"] = round(audio.duration, 2)
        result["chord_vocabulary"] = options.chord_vocabulary
        return AnalysisResult.model_validate(result)


def make_client(tmp_path, **overrides):
    settings = Settings(**{"models_dir": tmp_path, "max_upload_mb": 1, **overrides})
    stub = StubAnalyzer()
    app = create_app(settings, Engine.in_process(settings, stub))
    return TestClient(app), stub


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=6:sample_rate=8000", str(path)], check=True)
    return path.read_bytes()


def test_health_and_analyze(tmp_path, wav):
    client, stub = make_client(tmp_path)
    with client:
        h = client.get("/v1/health")
        assert h.status_code == 200 and h.json()["status"] == "ok"
        assert h.json()["auth"] == "disabled"
        r = client.post("/v1/analyze", files={"file": ("tone.wav", wav, "audio/wav")},
                        data={"chord_vocabulary": "majmin"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["schema_version"] == "1.0"
        assert body["chord_vocabulary"] == "majmin"
        assert body["file"]["duration"] == pytest.approx(6.0, abs=0.05)
        assert r.headers["X-Cache"] == "miss" and float(r.headers["X-Processing-Time"]) >= 0
        # Same bytes and options again: served from the cache.
        r2 = client.post("/v1/analyze", files={"file": ("again.wav", wav, "audio/wav")},
                         data={"chord_vocabulary": "majmin"})
        assert r2.headers["X-Cache"] == "hit" and r2.json() == body and stub.calls == 1


def _assert_problem(r, status):
    assert r.status_code == status, r.text
    assert r.headers["content-type"].startswith("application/problem+json")
    body = r.json()
    assert body["status"] == status and body["type"].startswith("https://") and body["title"]
    return body


def test_auth(tmp_path, wav):
    client, _ = make_client(tmp_path, api_keys=["secret-key"])
    with client:
        files = {"file": ("tone.wav", wav, "audio/wav")}
        _assert_problem(client.post("/v1/analyze", files=files), 401)
        _assert_problem(client.post("/v1/analyze", files=files,
                                    headers={"X-API-Key": "wrong"}), 401)
        r = client.post("/v1/analyze", files=files, headers={"X-API-Key": "secret-key"})
        assert r.status_code == 200
        assert client.get("/v1/health").json()["auth"] == "enabled"


def test_errors(tmp_path, wav):
    client, _ = make_client(tmp_path)
    with client:
        _assert_problem(client.post("/v1/analyze"), 400)
        _assert_problem(client.post("/v1/analyze", files={"file": ("t.wav", wav)},
                                    data={"chord_vocabulary": "jazz"}), 400)
        body = _assert_problem(client.post("/v1/analyze", files={"file": ("t.wav", wav)},
                                           data={"include": "beats,lyrics"}), 400)
        assert "lyrics" in body["detail"]
        _assert_problem(client.post("/v1/analyze",
                                    files={"file": ("big.wav", b"\0" * (3 << 20))}), 413)
        _assert_problem(client.post("/v1/analyze",
                                    files={"file": ("x.mp3", b"not audio at all")}), 415)
        short = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                                "sine=duration=2:sample_rate=8000", "-f", "wav", "-"],
                               check=True, capture_output=True).stdout
        _assert_problem(client.post("/v1/analyze", files={"file": ("s.wav", short)}), 422)


def test_async_job(tmp_path, wav):
    client, _ = make_client(tmp_path)
    with client:
        r = client.post("/v1/jobs", files={"file": ("tone.wav", wav)})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        assert r.headers["Location"] == f"/v1/jobs/{job_id}"
        for _ in range(200):
            status = client.get(f"/v1/jobs/{job_id}").json()
            if status["status"] in ("succeeded", "failed"):
                break
        assert status["status"] == "succeeded", status
        assert status["result"]["schema_version"] == "1.0"
        _assert_problem(client.get("/v1/jobs/nope"), 404)


def test_failed_job_reports_problem(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        job_id = client.post("/v1/jobs", files={"file": ("x.mp3", b"junk")}).json()["job_id"]
        for _ in range(200):
            status = client.get(f"/v1/jobs/{job_id}").json()
            if status["status"] in ("succeeded", "failed"):
                break
        assert status["status"] == "failed" and status["error"]["status"] == 415


def test_openapi_contract_matches_committed_copy(tmp_path):
    """Changing the API contract must be deliberate: regenerate with
    `python scripts/export_openapi.py` and commit openapi.json."""
    client, _ = make_client(tmp_path)
    committed = json.loads((Path(__file__).parents[1] / "openapi.json").read_text())
    spec = client.get("/openapi.json").json()
    spec["info"]["version"] = committed["info"]["version"] = "x"
    assert spec == committed

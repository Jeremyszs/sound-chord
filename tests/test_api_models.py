"""The real service: process pool, models loaded in the worker, a real upload."""

import pytest
from conftest import MODELS_DIR
from fastapi.testclient import TestClient

from chordanalyzer.api.app import create_app
from chordanalyzer.settings import Settings

pytestmark = pytest.mark.models


def test_upload_returns_analysis(synth_pop):
    path, _ = synth_pop
    app = create_app(Settings(models_dir=MODELS_DIR, workers=1, api_keys=["k"]))
    with TestClient(app) as client:
        for _ in range(600):
            health = client.get("/v1/health")
            if health.status_code == 200:
                break
            import time
            time.sleep(0.1)
        assert health.json()["status"] == "ok", health.json()
        with open(path, "rb") as f:
            r = client.post("/v1/analyze", files={"file": ("pop.wav", f, "audio/wav")},
                            data={"include": "beats,bars,tempo_curve"},
                            headers={"X-API-Key": "k"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["key"]["name"] == "C major"
        assert abs(body["tempo"]["bpm"] - 120) < 5
        assert [p["name"] for p in body["progression"] if p["name"] != "N.C."][:4] == \
            ["C", "G", "Am", "F"]
        assert "chord_probabilities" not in body

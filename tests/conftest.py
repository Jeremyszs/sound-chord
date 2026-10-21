import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
HAVE_MODELS = all((MODELS_DIR / f).exists()
                  for f in ("btc_majmin.safetensors", "btc_full.safetensors"))


def pytest_collection_modifyitems(config, items):
    if HAVE_MODELS:
        return
    skip = pytest.mark.skip(reason="BTC weights missing: run scripts/fetch_models.py")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def analyzer():
    from chordanalyzer import Analyzer
    from chordanalyzer.settings import Settings

    return Analyzer(Settings(models_dir=MODELS_DIR))


@pytest.fixture(scope="session")
def synth_pop(tmp_path_factory):
    """C G Am F at 120 BPM, one chord per bar, 4 times."""
    from synth import render, write_wav

    y, truth = render(["C:maj", "G:maj", "A:min", "F:maj"], 120)
    path = tmp_path_factory.mktemp("audio") / "pop.wav"
    write_wav(path, y)
    return path, truth

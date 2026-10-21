"""End-to-end runs with the real BTC weights on synthesised songs with known answers."""

import json
import os
import zlib
from pathlib import Path

import pytest
from synth import render, write_wav

from chordanalyzer import AnalysisOptions

pytestmark = pytest.mark.models
GOLDEN = Path(__file__).parent / "golden"

SONGS = {
    "pop_C_120": (["C:maj", "G:maj", "A:min", "F:maj"], 120, 4),
    "minor_Am_96": (["A:min", "F:maj", "C:maj", "E:maj"], 96, 4),
    "sevenths_Bb_104": (["Bb:maj7", "G:min7", "C:min7", "F:7"], 104, 4),
    "sharp_E_140": (["E:maj", "C#:min", "A:maj", "B:maj"], 140, 4),
    "waltz_D_150": (["D:maj", "B:min", "G:maj", "A:maj"], 150, 3),
}


def _summary(result):
    return {
        "bpm": round(result.tempo.bpm),
        "beats_per_bar": result.tempo.beats_per_bar,
        "key": result.key.name,
        "progression": [f"{p.name} ({p.roman}) x{p.bars}" for p in result.progression],
    }


@pytest.mark.parametrize("name", SONGS)
def test_golden(analyzer, tmp_path, name):
    """The summary must match the reviewed snapshot in tests/golden/.

    After an intended change, regenerate with UPDATE_GOLDEN=1 and review the diff.
    These are regression snapshots, not ground truth: the librosa-fallback waltz is
    known to be wrong (4/4 assumed).
    """
    prog, bpm, meter = SONGS[name]
    y, _ = render(prog, bpm, beats_per_bar=meter, repeats=3, seed=zlib.crc32(name.encode()))
    path = tmp_path / f"{name}.wav"
    write_wav(path, y)
    summary = _summary(analyzer.analyze(path))
    # Snapshots are per beat tracker: the librosa fallback is weaker (e.g. it assumes 4/4).
    golden = GOLDEN / analyzer.beat_tracker.name / f"{name}.json"
    if os.environ.get("UPDATE_GOLDEN"):
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(json.dumps(summary, indent=2) + "\n")
    if not golden.exists():
        pytest.skip(f"no snapshot for {analyzer.beat_tracker.name}; create with UPDATE_GOLDEN=1")
    assert summary == json.loads(golden.read_text())


def test_known_answer(analyzer, synth_pop):
    path, truth = synth_pop
    result = analyzer.analyze(path, AnalysisOptions(include=("beats", "bars", "tempo_curve")))
    assert result.tempo.bpm == pytest.approx(120, rel=0.04)
    assert result.key.name == "C major"
    # Chord label at the middle of every bar matches the rendered chord.
    hits = 0
    for start, end, label in truth:
        mid = (start + end) / 2
        seg = next(c for c in result.chords if c.start <= mid < c.end)
        hits += seg.label == {"C:maj": "C:maj", "G:maj": "G:maj", "A:min": "A:min",
                              "F:maj": "F:maj"}[label]
    assert hits / len(truth) >= 0.9
    assert [p.name for p in result.progression if p.name != "N.C."][:4] == ["C", "G", "Am", "F"]
    assert result.beats and result.bars and result.tempo_curve
    assert result.chords[0].start == 0 and result.chords[-1].end == result.file.duration


def test_majmin_vocabulary_and_notation(analyzer, synth_pop):
    path, _ = synth_pop
    result = analyzer.analyze(path, AnalysisOptions(chord_vocabulary="majmin", notation="name",
                                                    include=("chord_probabilities",)))
    assert result.engine.chord_model == "btc-majmin"
    assert all(c.roman is None for c in result.chords)
    assert len(result.chord_probabilities.labels) == 25
    assert result.beats is None


def test_deterministic(analyzer, synth_pop):
    path, _ = synth_pop
    assert analyzer.analyze(path).model_dump() == analyzer.analyze(path).model_dump()

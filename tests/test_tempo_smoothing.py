import numpy as np
import pytest

from chordanalyzer.beats import BeatResult
from chordanalyzer.pipeline import BeatGrid
from chordanalyzer.smoothing import beat_intervals, merge_segments, pool_frames, viterbi
from chordanalyzer.tempo import estimate_tempo


@pytest.mark.parametrize("bpm,meter", [(60, 4), (97.5, 4), (128, 4), (174, 4), (90, 3)])
def test_tempo_from_click_grid(bpm, meter):
    beats = 0.37 + np.arange(64) * 60 / bpm
    res = estimate_tempo(beats, beats[::meter])
    assert res.bpm == pytest.approx(bpm, rel=0.01)
    assert res.beats_per_bar == meter
    assert res.confidence > 0.99
    assert res.bpm_alternatives == [round(res.bpm / 2, 1), round(res.bpm * 2, 1)]


def test_tempo_confidence_drops_with_jitter():
    rng = np.random.default_rng(0)
    beats = np.cumsum(0.5 + rng.normal(0, 0.1, 64))
    assert estimate_tempo(beats, beats[::4]).confidence < 0.9


def test_tempo_curve_shows_tempo_change():
    beats = np.concatenate([np.arange(16) * 0.5, 8 + np.arange(1, 17) * 0.4])
    curve = estimate_tempo(beats, beats[::4]).curve
    assert curve[0][2] == pytest.approx(120) and curve[-1][2] == pytest.approx(150)


def test_viterbi_removes_single_step_flicker():
    probs = np.array([[0.9, 0.1]] * 5 + [[0.4, 0.6]] + [[0.9, 0.1]] * 5)
    assert viterbi(probs, switch_penalty=2.0).tolist() == [0] * 11
    assert viterbi(probs, switch_penalty=0.0)[5] == 1


def test_viterbi_keeps_real_change():
    probs = np.array([[0.9, 0.1]] * 5 + [[0.1, 0.9]] * 5)
    assert viterbi(probs, 2.0).tolist() == [0] * 5 + [1] * 5


def test_pool_and_merge_cover_whole_file():
    frame_period = 0.1
    probs = np.zeros((100, 2))
    probs[:50, 0] = probs[50:, 1] = 1.0
    intervals = beat_intervals(np.arange(1, 10) * 1.0, 10.0)
    assert intervals[0, 0] == 0 and intervals[-1, 1] == 10.0
    pooled = pool_frames(probs, frame_period, intervals)
    segs = merge_segments(intervals, viterbi(pooled, 2.0), pooled)
    assert [(s.start, s.end, s.index) for s in segs] == [(0.0, 5.0, 0), (5.0, 10.0, 1)]


def test_beat_grid_numbering_with_pickup():
    beats = np.arange(10) * 0.5 + 0.5
    grid = BeatGrid(BeatResult(beats, beats[2::4]), 4, 6.0)
    assert grid.beat_positions[:4] == [(0, 3), (0, 4), (1, 1), (1, 2)]
    assert grid.position(0.1) == (0, 0)
    assert grid.position(1.5) == (1, 1)
    assert grid.bars()[0] == (0, 0.0, 1.5)

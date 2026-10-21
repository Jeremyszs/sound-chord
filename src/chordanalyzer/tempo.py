"""Tempo, tempo curve and meter derived from beat and downbeat times."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TempoResult:
    bpm: float
    confidence: float
    beats_per_bar: int
    bpm_alternatives: list[float]
    curve: list[tuple[int, float, float]] = field(default_factory=list)  # (bar, start, bpm)


def estimate_tempo(beats: np.ndarray, downbeats: np.ndarray) -> TempoResult:
    beats = np.asarray(beats, float)
    if len(beats) < 3:
        return TempoResult(0.0, 0.0, 4, [])
    ibi = np.diff(beats)
    bpm = round(60.0 / float(np.median(ibi)), 1)
    cv = float(np.std(ibi) / np.mean(ibi))
    confidence = round(float(np.clip(1.0 - cv, 0.0, 1.0)), 3)
    beats_per_bar = meter_from_downbeats(beats, downbeats)
    alternatives = [round(bpm / 2, 1), round(bpm * 2, 1)]
    return TempoResult(bpm, confidence, beats_per_bar, alternatives,
                       tempo_curve(beats, downbeats))


def meter_from_downbeats(beats: np.ndarray, downbeats: np.ndarray) -> int:
    """Median number of beats between consecutive downbeats (3 or 4 in practice)."""
    if len(downbeats) < 2:
        return 4
    idx = np.searchsorted(beats, np.asarray(downbeats) - 1e-3)
    counts = np.diff(idx)
    counts = counts[counts > 0]
    return int(np.median(counts)) if len(counts) else 4


def tempo_curve(beats: np.ndarray, downbeats: np.ndarray) -> list[tuple[int, float, float]]:
    """BPM per bar, from the beats inside each bar (bar numbers start at 1)."""
    curve = []
    downbeats = np.asarray(downbeats, float)
    for i, start in enumerate(downbeats):
        end = downbeats[i + 1] if i + 1 < len(downbeats) else np.inf
        inside = beats[(beats >= start - 1e-3) & (beats < end + 1e-3)]
        if len(inside) >= 2:
            curve.append((i + 1, round(float(start), 2),
                          round(60.0 / float(np.mean(np.diff(inside))), 1)))
    return curve

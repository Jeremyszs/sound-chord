"""Global key estimation: chroma key profiles plus a vote from the detected chords."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .chords.vocab import Chord

# Krumhansl-Kessler probe-tone profiles, tonic first.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MINOR_QUALITIES = {"min", "min6", "min7", "minmaj7"}
MAJOR_QUALITIES = {"maj", "maj6", "maj7"}


@dataclass(frozen=True)
class Key:
    tonic: int  # pitch class
    mode: str  # 'major' | 'minor'


@dataclass
class KeyResult:
    key: Key
    confidence: float
    runner_up: Key
    scores: np.ndarray  # (24,): 0-11 major keys, 12-23 minor keys


def _key_of(index: int) -> Key:
    return Key(index % 12, "major" if index < 12 else "minor")


def profile_correlations(chroma: np.ndarray) -> np.ndarray:
    """Pearson correlation of a 12-bin chroma vector with all 24 rotated profiles."""
    out = np.empty(24)
    if np.allclose(chroma, chroma[0]):
        return np.zeros(24)
    for mode, profile in enumerate((KK_MAJOR, KK_MINOR)):
        for tonic in range(12):
            out[mode * 12 + tonic] = np.corrcoef(chroma, np.roll(profile, tonic))[0, 1]
    return out


def chord_chroma(chords: Sequence[tuple[Chord, float]]) -> np.ndarray:
    """Duration-weighted pitch-class histogram of a chord sequence, root counted twice."""
    hist = np.zeros(12)
    for chord, dur in chords:
        if chord.is_chord:
            for pc in chord.pitch_classes():
                hist[pc] += dur
            hist[chord.root] += dur
    return hist


def tonic_share(chords: Sequence[tuple[Chord, float]]) -> np.ndarray:
    """For each of the 24 keys, the fraction of chord time spent on its tonic chord.

    A dominant seventh is not a tonic: it counts half towards the key a fifth below
    (both modes), since it usually resolves there.
    """
    share = np.zeros(24)
    total = sum(d for c, d in chords if c.is_chord) or 1.0
    for chord, dur in chords:
        if not chord.is_chord:
            continue
        if chord.quality in MAJOR_QUALITIES:
            share[chord.root] += dur
        elif chord.quality in MINOR_QUALITIES:
            share[12 + chord.root] += dur
        elif chord.quality == "7":
            target = (chord.root + 5) % 12
            share[target] += 0.5 * dur
            share[12 + target] += 0.5 * dur
    return share / total


class KeyEstimator(Protocol):
    name: str

    def estimate(self, y: np.ndarray, sr: int, chords: Sequence[tuple[Chord, float]],
                 chroma: np.ndarray | None = None) -> KeyResult: ...


class ProfileKeyEstimator:
    """Krumhansl-Kessler correlation on CQT chroma, combined with chord evidence.

    The weights are untuned starting values; phase 3 tunes them on the evaluation set.
    """

    name = "kk-profile+chord-vote"

    def __init__(self, w_audio: float = 0.5, w_chords: float = 1.0, w_tonic: float = 0.5,
                 temperature: float = 0.1):
        self.w_audio, self.w_chords, self.w_tonic = w_audio, w_chords, w_tonic
        self.temperature = temperature

    def audio_chroma(self, y: np.ndarray, sr: int) -> np.ndarray:
        import librosa

        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
        return chroma.sum(axis=1)

    def estimate(self, y: np.ndarray, sr: int, chords: Sequence[tuple[Chord, float]],
                 chroma: np.ndarray | None = None) -> KeyResult:
        """chroma: optional precomputed (n_frames, 12) chroma, to avoid a second CQT."""
        song_chroma = chroma.sum(axis=0) if chroma is not None else self.audio_chroma(y, sr)
        scores = self.w_audio * profile_correlations(song_chroma)
        cc = chord_chroma(chords)
        if cc.sum() > 0:
            scores = scores + self.w_chords * profile_correlations(cc)
            scores = scores + self.w_tonic * tonic_share(chords)
        order = np.argsort(scores)[::-1]
        z = (scores - scores.max()) / self.temperature
        p = np.exp(z) / np.exp(z).sum()
        confidence = float(p[order[0]] - p[order[1]])
        return KeyResult(_key_of(int(order[0])), round(confidence, 3),
                         _key_of(int(order[1])), scores)

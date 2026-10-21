"""Synthesise test songs with a known tempo, key and chord progression."""

from __future__ import annotations

import numpy as np

from chordanalyzer.chords.vocab import parse_harte

SR = 22050


def _tone(freq: float, dur: float, sr: int = SR) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    env = np.exp(-2.5 * t) * np.minimum(1, t / 0.005)
    wave = sum(np.sin(2 * np.pi * freq * h * t) / h ** 1.5 for h in range(1, 7))
    return (wave * env).astype(np.float32)


def _kick(sr: int = SR) -> np.ndarray:
    t = np.arange(int(0.15 * sr)) / sr
    return (np.sin(2 * np.pi * (50 + 80 * np.exp(-30 * t)) * t) * np.exp(-25 * t)).astype(np.float32)


def _hat(rng: np.random.Generator, sr: int = SR) -> np.ndarray:
    n = int(0.03 * sr)
    return (rng.standard_normal(n) * np.exp(-np.arange(n) / (0.005 * sr))).astype(np.float32)


def render(progression: list[str], bpm: float, beats_per_bar: int = 4, repeats: int = 4,
           sr: int = SR, seed: int = 0) -> tuple[np.ndarray, list[tuple[float, float, str]]]:
    """One chord per bar. Returns (audio, [(start, end, harte_label), ...])."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = beat * beats_per_bar
    lead_in = 1.0
    n_bars = len(progression) * repeats
    y = np.zeros(int((lead_in + n_bars * bar + 2.0) * sr), np.float32)
    truth = []
    for b in range(n_bars):
        label = progression[b % len(progression)]
        chord = parse_harte(label)
        start = lead_in + b * bar
        truth.append((start, start + bar, label))
        for k in range(beats_per_bar):
            t0 = int((start + k * beat) * sr)
            # Chord stab on every beat: root in the bass, chord tones around middle C.
            for i, pc in enumerate(chord.pitch_classes()):
                midi = 60 + pc if i else 36 + chord.root
                stab = _tone(440 * 2 ** ((midi - 69) / 12), beat * 0.95, sr) * (0.6 if i == 0 else 0.35)
                y[t0:t0 + len(stab)] += stab
            hit = _kick(sr) if k == 0 else _hat(rng, sr) * 0.3
            y[t0:t0 + len(hit)] += hit
    y /= np.max(np.abs(y))
    return y * 0.9, truth


def write_wav(path, y: np.ndarray, sr: int = SR) -> None:
    import soundfile as sf

    sf.write(str(path), y, sr, subtype="PCM_16")

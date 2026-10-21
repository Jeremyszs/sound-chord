"""Beat-synchronous pooling, Viterbi smoothing and segment merging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def beat_intervals(beats: np.ndarray, duration: float) -> np.ndarray:
    """Split [0, duration] at the beat times. Returns (n, 2) start/end pairs.

    Audio before the first beat and after the last beat become their own
    intervals, so the whole file is covered.
    """
    b = np.asarray(beats, float)
    b = b[(b > 0.05) & (b < duration - 0.05)]
    edges = np.concatenate([[0.0], b, [duration]])
    return np.stack([edges[:-1], edges[1:]], axis=1)


def pool_frames(probs: np.ndarray, frame_period: float, intervals: np.ndarray) -> np.ndarray:
    """Average frame posteriors inside each interval. Frame i is centred on i * frame_period."""
    times = np.arange(len(probs)) * frame_period
    out = np.empty((len(intervals), probs.shape[1]), np.float64)
    for k, (start, end) in enumerate(intervals):
        mask = (times >= start) & (times < end)
        if not mask.any():
            # Interval shorter than a frame: take the nearest frame.
            mask = np.zeros(len(times), bool)
            mask[min(int(round((start + end) / 2 / frame_period)), len(times) - 1)] = True
        out[k] = probs[mask].mean(axis=0)
    return out


def viterbi(probs: np.ndarray, switch_penalty: float) -> np.ndarray:
    """Most likely state path under a uniform 'stay or switch' transition model.

    probs: (n, k) posteriors per step. switch_penalty: log-probability cost of
    changing state between steps (larger means fewer changes).
    """
    n, k = probs.shape
    log_e = np.log(np.clip(probs, 1e-12, None))
    score = log_e[0].copy()
    back = np.zeros((n, k), np.int64)
    for t in range(1, n):
        best_prev = int(np.argmax(score))
        switch = score[best_prev] - switch_penalty
        stay = score
        take_switch = switch > stay
        back[t] = np.where(take_switch, best_prev, np.arange(k))
        score = np.where(take_switch, switch, stay) + log_e[t]
    path = np.empty(n, np.int64)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


@dataclass
class Segment:
    start: float
    end: float
    index: int  # vocabulary index
    confidence: float


def merge_segments(intervals: np.ndarray, path: np.ndarray, pooled: np.ndarray) -> list[Segment]:
    """Merge consecutive intervals with the same state into segments."""
    segments: list[Segment] = []
    run_start = 0
    for t in range(1, len(path) + 1):
        if t == len(path) or path[t] != path[run_start]:
            idx = int(path[run_start])
            span = intervals[run_start:t]
            w = span[:, 1] - span[:, 0]
            conf = float(np.average(pooled[run_start:t, idx], weights=w)) if w.sum() > 0 else 0.0
            segments.append(Segment(float(span[0, 0]), float(span[-1, 1]), idx, conf))
            run_start = t
    return segments


def smooth_chords(probs: np.ndarray, frame_period: float, beats: np.ndarray, duration: float,
                  switch_penalty: float = 2.0) -> list[Segment]:
    intervals = beat_intervals(beats, duration)
    pooled = pool_frames(probs, frame_period, intervals)
    path = viterbi(pooled, switch_penalty)
    return merge_segments(intervals, path, pooled)

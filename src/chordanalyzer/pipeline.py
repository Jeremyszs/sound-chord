"""Orchestrates the analysis: decode, beats, tempo, chords, key, smoothing, spelling."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__
from .audio import load
from .beats import BeatResult, make_beat_tracker
from .chords.btc import BTCChordModel
from .chords.vocab import Chord
from .key import ProfileKeyEstimator
from .schema import (
    DEFAULT_INCLUDE,
    AnalysisResult,
    Bar,
    Beat,
    ChordProbabilities,
    ChordSegment,
    EngineInfo,
    FileInfo,
    KeyInfo,
    KeyResultInfo,
    ProgressionEntry,
    TempoCurvePoint,
    TempoInfo,
)
from .settings import Settings
from .smoothing import Segment, smooth_chords
from .spelling import display_name, harte_label, key_label, key_name, roman_numeral, tonic_name
from .tempo import estimate_tempo

log = logging.getLogger(__name__)

LOW_TEMPO_CONFIDENCE = 0.6
AMBIGUOUS_KEY_CONFIDENCE = 0.15
MOSTLY_NO_CHORD = 0.5
CLIPPED_FRACTION = 0.001


@dataclass
class AnalysisOptions:
    chord_vocabulary: str = "full"
    include: Sequence[str] = DEFAULT_INCLUDE
    notation: str = "both"
    switch_penalty: float = 2.0


@dataclass
class Timings:
    steps: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str, start: float) -> float:
        now = time.perf_counter()
        self.steps[name] = round(now - start, 3)
        return now


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _r(x: float) -> float:
    return round(float(x), 2)


class Analyzer:
    """Loads the models once; `analyze` can then be called many times."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        s = self.settings
        if s.torch_threads:
            import torch

            torch.set_num_threads(s.torch_threads)
        self.chord_models = {v: BTCChordModel(v, s.models_dir, s.device) for v in ("majmin", "full")}
        self.beat_tracker, self.beat_warnings = make_beat_tracker(
            s.beat_tracker, s.beat_checkpoint, s.models_dir, s.device)
        if self.beat_warnings:
            log.warning("Beat This! unavailable, using the librosa fallback beat tracker")
        self.key_estimator = ProfileKeyEstimator()
        self.last_timings = Timings()

    def engine_info(self, vocabulary: str = "full") -> EngineInfo:
        return EngineInfo(version=__version__, chord_model=self.chord_models[vocabulary].name,
                          beat_model=self.beat_tracker.name, key_model=self.key_estimator.name)

    def analyze(self, path: str | Path, options: AnalysisOptions | None = None,
                sha256: str | None = None) -> AnalysisResult:
        opts = options or AnalysisOptions()
        timings = Timings()
        t = time.perf_counter()
        digest = sha256 or sha256_file(path)
        audio = load(path, self.settings.min_duration_s, self.settings.max_duration_s)
        t = timings.mark("decode", t)

        beats = self.beat_tracker.track(audio.y, audio.sr)
        t = timings.mark("beats", t)
        model = self.chord_models[opts.chord_vocabulary]
        frames = model.predict(audio.y, audio.sr)
        t = timings.mark("chords", t)

        tempo = estimate_tempo(beats.beats, beats.downbeats)
        segments = smooth_chords(frames.probs, frames.frame_period, beats.beats,
                                 audio.duration, opts.switch_penalty)
        vocab = frames.vocabulary
        chord_seq = [(vocab.chords[s.index], s.end - s.start) for s in segments]
        key_res = self.key_estimator.estimate(audio.y, audio.sr, chord_seq, frames.chroma)
        key = key_res.key
        t = timings.mark("key_and_smoothing", t)

        grid = BeatGrid(beats, tempo.beats_per_bar, audio.duration)
        name_on = opts.notation in ("name", "both")
        roman_on = opts.notation in ("roman", "both")

        def describe(chord: Chord) -> dict:
            return {"label": harte_label(chord, key),
                    "name": display_name(chord, key) if name_on else None,
                    "roman": roman_numeral(chord, key) if roman_on else None}

        chords_out = []
        for seg in segments:
            bar, beat = grid.position(seg.start)
            chords_out.append(ChordSegment(start=_r(seg.start), end=_r(seg.end), bar=bar, beat=beat,
                                           confidence=round(min(max(seg.confidence, 0), 1), 3),
                                           **describe(vocab.chords[seg.index])))
        progression = [ProgressionEntry(bar=b, bars=n, **describe(vocab.chords[i]))
                       for b, n, i in grid.progression(segments)]

        warnings = list(self.beat_warnings)
        if tempo.confidence < LOW_TEMPO_CONFIDENCE:
            warnings.append("low_tempo_confidence")
        if key_res.confidence < AMBIGUOUS_KEY_CONFIDENCE:
            warnings.append("ambiguous_key")
        no_chord = sum(d for c, d in chord_seq if not c.is_chord)
        if no_chord > MOSTLY_NO_CHORD * audio.duration:
            warnings.append("mostly_no_chord")
        if audio.clipped_fraction > CLIPPED_FRACTION:
            warnings.append("clipped_audio")

        def key_info(k) -> dict:
            return {"tonic": tonic_name(k), "mode": k.mode, "name": key_name(k),
                    "label": key_label(k)}

        include = set(opts.include)
        result = AnalysisResult(
            engine=self.engine_info(opts.chord_vocabulary),
            file=FileInfo(sha256=digest, duration=_r(audio.duration),
                          format=audio.probe.format_name, codec=audio.probe.codec),
            tempo=TempoInfo(bpm=tempo.bpm, confidence=tempo.confidence,
                            bpm_alternatives=tempo.bpm_alternatives,
                            beats_per_bar=tempo.beats_per_bar,
                            time_signature=f"{tempo.beats_per_bar}/4"),
            key=KeyResultInfo(**key_info(key), confidence=key_res.confidence,
                              runner_up=KeyInfo(**key_info(key_res.runner_up))),
            chord_vocabulary=opts.chord_vocabulary,
            chords=chords_out,
            progression=progression,
            warnings=warnings,
        )
        if "beats" in include:
            result.beats = [Beat(time=_r(b), bar=bar, beat=beat)
                            for b, (bar, beat) in zip(grid.beats, grid.beat_positions, strict=True)]
        if "bars" in include:
            result.bars = [Bar(bar=n, start=_r(s), end=_r(e)) for n, s, e in grid.bars()]
        if "tempo_curve" in include:
            result.tempo_curve = [TempoCurvePoint(bar=b, start=s, bpm=v) for b, s, v in tempo.curve]
        if "chord_probabilities" in include:
            result.chord_probabilities = ChordProbabilities(
                labels=[c.harte() for c in vocab.chords], frame_period=frames.frame_period,
                frames=np.round(frames.probs, 3).tolist())
        timings.mark("format", t)
        self.last_timings = timings
        return result


class BeatGrid:
    """Bar and beat numbering from beat and downbeat times."""

    def __init__(self, beats: BeatResult, beats_per_bar: int, duration: float):
        self.beats = np.asarray(beats.beats, float)
        self.duration = duration
        self.beats_per_bar = beats_per_bar
        down = np.asarray(beats.downbeats, float)
        is_down = np.zeros(len(self.beats), bool)
        for d in down:
            if len(self.beats):
                i = int(np.argmin(np.abs(self.beats - d)))
                if abs(self.beats[i] - d) < 0.07:
                    is_down[i] = True
        if len(self.beats) and not is_down.any():
            is_down[::beats_per_bar] = True
        self.is_down = is_down
        self.beat_positions = self._number()

    def _number(self) -> list[tuple[int, int]]:
        positions = []
        first_down = int(np.argmax(self.is_down)) if self.is_down.any() else 0
        bar, beat = 0, 0
        for i in range(len(self.beats)):
            if self.is_down[i]:
                bar, beat = bar + 1, 1
            elif bar == 0:
                beat = max(1, self.beats_per_bar - (first_down - i) + 1)
            else:
                beat += 1
            positions.append((bar, beat))
        return positions

    def position(self, t: float) -> tuple[int, int]:
        """(bar, beat) of the beat at or just before t; (0, 0) before the first beat."""
        i = int(np.searchsorted(self.beats, t + 0.05, side="right")) - 1
        if i < 0:
            return 0, 0
        return self.beat_positions[i]

    def bars(self) -> list[tuple[int, float, float]]:
        starts = self.beats[self.is_down]
        out = []
        if len(starts) and starts[0] > 0.5:
            out.append((0, 0.0, float(starts[0])))
        for n, s in enumerate(starts, start=1):
            e = starts[n] if n < len(starts) else self.duration
            out.append((n, float(s), float(e)))
        return out

    def progression(self, segments: list[Segment]) -> list[tuple[int, int, int]]:
        """One chord per bar (the one sounding longest), consecutive repeats collapsed.

        Returns (first bar, number of bars, vocabulary index).
        """
        per_bar = []
        for n, s, e in self.bars():
            overlap: dict[int, float] = {}
            for seg in segments:
                o = min(e, seg.end) - max(s, seg.start)
                if o > 0:
                    overlap[seg.index] = overlap.get(seg.index, 0.0) + o
            if overlap:
                per_bar.append((n, max(overlap, key=overlap.get)))
        out: list[list[int]] = []
        for n, idx in per_bar:
            if out and out[-1][2] == idx:
                out[-1][1] += 1
            else:
                out.append([n, 1, idx])
        return [tuple(x) for x in out]

"""Accuracy evaluation with mir_eval.

A dataset is a directory of audio files with annotations next to them:

    song.mp3        audio
    song.lab        chords: "start end label" per line (Harte syntax), as in Isophonics/Billboard
    song.key        optional: one line, e.g. "A minor" or "Bb major"
    song.bpm        optional: one line with the annotated tempo
    song.beats      optional: one beat time per line (seconds)

Usage:
    python eval/run_eval.py DATASET_DIR [--out results.csv] [--vocabulary full]

Prints per-song scores and the duration-weighted means, in the metrics the
plan defines (chord WCSR root/majmin/mirex/sevenths, key MIREX score, tempo
Accuracy1/Accuracy2, beat F-measure).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import mir_eval
import numpy as np

from chordanalyzer import AnalysisOptions, Analyzer
from chordanalyzer.settings import Settings

AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
CHORD_METRICS = ["root", "majmin", "mirex", "sevenths"]


def chord_scores(ref_lab: Path, result) -> dict[str, float]:
    ref_int, ref_lab_ = mir_eval.io.load_labeled_intervals(str(ref_lab))
    est_int = np.array([[c.start, c.end] for c in result.chords])
    est_lab = [c.label for c in result.chords]
    est_int, est_lab = mir_eval.util.adjust_intervals(
        est_int, est_lab, ref_int.min(), ref_int.max(), mir_eval.chord.NO_CHORD,
        mir_eval.chord.NO_CHORD)
    intervals, ref_l, est_l = mir_eval.util.merge_labeled_intervals(ref_int, ref_lab_, est_int, est_lab)
    durations = mir_eval.util.intervals_to_durations(intervals)
    out = {}
    for m in CHORD_METRICS:
        comparisons = getattr(mir_eval.chord, m)(ref_l, est_l)
        out[f"chord_{m}"] = float(mir_eval.chord.weighted_accuracy(comparisons, durations))
    out["chord_segmentation"] = float(mir_eval.chord.seg(ref_int, est_int))
    return out


def key_score(ref_key: str, result) -> float:
    tonic, mode = ref_key.split()
    est = f"{result.key.tonic} {result.key.mode}"
    return float(mir_eval.key.weighted_score(f"{tonic} {mode.lower()}", est))


def tempo_scores(ref_bpm: float, est_bpm: float) -> dict[str, float]:
    acc1 = abs(est_bpm - ref_bpm) <= 0.04 * ref_bpm
    acc2 = any(abs(est_bpm - ref_bpm * f) <= 0.04 * ref_bpm * f for f in (1, 2, 3, 1 / 2, 1 / 3))
    return {"tempo_acc1": float(acc1), "tempo_acc2": float(acc2)}


def evaluate(dataset: Path, analyzer: Analyzer, options: AnalysisOptions) -> list[dict]:
    rows = []
    for audio in sorted(p for p in dataset.iterdir() if p.suffix.lower() in AUDIO_EXT):
        result = analyzer.analyze(audio, options)
        row: dict = {"song": audio.name, "duration": result.file.duration,
                     "bpm": result.tempo.bpm, "key": result.key.name}
        if (lab := audio.with_suffix(".lab")).exists():
            row.update(chord_scores(lab, result))
        if (k := audio.with_suffix(".key")).exists():
            row["key_ref"] = k.read_text().strip()
            row["key_mirex"] = key_score(row["key_ref"], result)
        if (b := audio.with_suffix(".bpm")).exists():
            row["bpm_ref"] = float(b.read_text().strip())
            row.update(tempo_scores(row["bpm_ref"], result.tempo.bpm))
        if (bt := audio.with_suffix(".beats")).exists() and result.beats:
            ref_beats = np.loadtxt(bt, ndmin=1)
            est_beats = np.array([x.time for x in result.beats])
            row["beat_f"] = float(mir_eval.beat.f_measure(
                mir_eval.beat.trim_beats(ref_beats), mir_eval.beat.trim_beats(est_beats)))
        rows.append(row)
        print(row, file=sys.stderr)
    return rows


def summarize(rows: list[dict]) -> dict[str, float]:
    summary = {}
    metrics = sorted({k for r in rows for k, v in r.items()
                      if isinstance(v, float) and k not in ("duration", "bpm", "bpm_ref")})
    for m in metrics:
        vals = [(r[m], r["duration"]) for r in rows if m in r]
        if m.startswith("chord_"):  # duration-weighted, as in MIREX
            summary[m] = sum(v * d for v, d in vals) / sum(d for _, d in vals)
        else:
            summary[m] = sum(v for v, _ in vals) / len(vals)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dataset", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--vocabulary", choices=["majmin", "full"], default="full")
    args = p.parse_args()
    analyzer = Analyzer(Settings())
    rows = evaluate(args.dataset, analyzer, AnalysisOptions(chord_vocabulary=args.vocabulary,
                                                            include=("beats",)))
    if args.out and rows:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "song", k))
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"\n{len(rows)} songs, engine {analyzer.engine_info(args.vocabulary).model_dump()}")
    for k, v in summarize(rows).items():
        print(f"{k:22s} {v:.3f}")


if __name__ == "__main__":
    main()

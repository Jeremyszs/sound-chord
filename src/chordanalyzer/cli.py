"""Command line: `chordanalyzer song.mp3 [--json out.json]`."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .audio import AudioError
from .schema import DEFAULT_INCLUDE


def _summary(result) -> str:
    lines = [
        f"Tempo: {result.tempo.bpm} BPM ({result.tempo.time_signature}, "
        f"confidence {result.tempo.confidence})",
        f"Key:   {result.key.name} (confidence {result.key.confidence}, "
        f"runner-up {result.key.runner_up.name})",
        "Progression (one chord per bar, repeats collapsed):",
    ]
    row = []
    for p in result.progression:
        cell = p.name or p.label
        if p.roman:
            cell += f" ({p.roman})"
        row.append(cell + (f" x{p.bars}" if p.bars > 1 else ""))
    lines.append("  " + " | ".join(row))
    if result.warnings:
        lines.append("Warnings: " + ", ".join(result.warnings))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="chordanalyzer",
                                description="Detect tempo, key and chords in an audio file.")
    p.add_argument("audio", type=Path)
    p.add_argument("--json", type=Path, help="Write the full JSON result here ('-' for stdout)")
    p.add_argument("--vocabulary", choices=["majmin", "full"], default="full")
    p.add_argument("--notation", choices=["name", "roman", "both"], default="both")
    p.add_argument("--include", default=",".join(DEFAULT_INCLUDE),
                   help="Comma list of beats, tempo_curve, bars, chord_probabilities")
    p.add_argument("--beat-tracker", choices=["auto", "beat_this", "librosa"])
    p.add_argument("-v", "--verbose", action="store_true", help="Print step timings")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    from .pipeline import AnalysisOptions, Analyzer
    from .settings import Settings

    settings = Settings()
    if args.beat_tracker:
        settings.beat_tracker = args.beat_tracker
    analyzer = Analyzer(settings)
    include = [x.strip() for x in args.include.split(",") if x.strip()]
    try:
        result = analyzer.analyze(args.audio, AnalysisOptions(
            chord_vocabulary=args.vocabulary, include=include, notation=args.notation))
    except AudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    payload = json.dumps(result.model_dump(exclude_none=True), indent=2, ensure_ascii=False)
    if args.json and str(args.json) == "-":
        print(payload)
    else:
        if args.json:
            args.json.write_text(payload + "\n", encoding="utf-8")
        print(_summary(result))
    if args.verbose:
        print("Timings (s): " + json.dumps(analyzer.last_timings.steps), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

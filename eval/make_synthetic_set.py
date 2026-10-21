"""Write a small synthetic dataset (audio + .lab/.key/.bpm/.beats) to smoke-test run_eval.py.

Synthetic songs are clean and easy; they check the plumbing, not real-world accuracy.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from synth import render, write_wav  # noqa: E402

SONGS = [
    ("pop_C", ["C:maj", "G:maj", "A:min", "F:maj"], 120, "C major"),
    ("minor_Am", ["A:min", "F:maj", "C:maj", "E:maj"], 96, "A minor"),
    ("sevenths_Bb", ["A#:maj7", "G:min7", "C:min7", "F:7"], 104, "Bb major"),
    ("sharp_E", ["E:maj", "C#:min", "A:maj", "B:maj"], 140, "E major"),
]

out = Path(sys.argv[1] if len(sys.argv) > 1 else "eval/synthetic")
out.mkdir(parents=True, exist_ok=True)
for i, (name, prog, bpm, key) in enumerate(SONGS):
    y, truth = render(prog, bpm, repeats=3, seed=i)
    write_wav(out / f"{name}.wav", y)
    lines = [f"0.000 {truth[0][0]:.3f} N"] + [f"{s:.3f} {e:.3f} {lab}" for s, e, lab in truth]
    lines.append(f"{truth[-1][1]:.3f} {len(y) / 22050:.3f} N")
    (out / f"{name}.lab").write_text("\n".join(lines) + "\n")
    (out / f"{name}.key").write_text(key + "\n")
    (out / f"{name}.bpm").write_text(f"{bpm}\n")
    beat = 60 / bpm
    beats = [truth[0][0] + k * beat for k in range(len(truth) * 4)]
    (out / f"{name}.beats").write_text("\n".join(f"{b:.3f}" for b in beats) + "\n")
print(f"wrote {len(SONGS)} songs to {out}")

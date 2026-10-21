"""Chord vocabularies of the BTC checkpoints and Harte label helpers."""

from __future__ import annotations

from dataclasses import dataclass

PITCH_CLASSES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Order of qualities in the 170-class BTC output (upstream utils/mir_eval_modules.py).
LARGE_QUALITIES = ["min", "maj", "dim", "aug", "min6", "maj6", "min7", "minmaj7", "maj7",
                   "7", "dim7", "hdim7", "sus2", "sus4"]

# Chord tones as semitone intervals above the root.
QUALITY_INTERVALS: dict[str, tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "maj6": (0, 4, 7, 9),
    "min6": (0, 3, 7, 9),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "minmaj7": (0, 3, 7, 11),
    "dim7": (0, 3, 6, 9),
    "hdim7": (0, 3, 6, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
}

NO_CHORD = "N"
UNKNOWN_CHORD = "X"

_NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


@dataclass(frozen=True)
class Chord:
    """A chord as a pitch-class root and a Harte quality, or N / X (root is None)."""

    root: int | None
    quality: str | None

    @property
    def is_chord(self) -> bool:
        return self.root is not None

    @property
    def special(self) -> str | None:
        if self.root is not None:
            return None
        return self.quality or NO_CHORD

    def pitch_classes(self) -> tuple[int, ...]:
        if self.root is None or self.quality is None:
            return ()
        return tuple((self.root + i) % 12 for i in QUALITY_INTERVALS[self.quality])

    def harte(self) -> str:
        """Harte label with sharp spelling, e.g. 'C#:min7'."""
        if self.root is None:
            return self.special or NO_CHORD
        return f"{PITCH_CLASSES_SHARP[self.root]}:{self.quality}"


N = Chord(None, NO_CHORD)
X = Chord(None, UNKNOWN_CHORD)


def note_to_pc(name: str) -> int:
    pc = _NOTE_TO_PC[name[0].upper()]
    for acc in name[1:]:
        if acc in "#♯":
            pc += 1
        elif acc in "b♭":
            pc -= 1
        else:
            raise ValueError(f"bad note name: {name!r}")
    return pc % 12


def parse_harte(label: str) -> Chord:
    """Parse the subset of Harte syntax the models produce ('C', 'Bb:min7', 'N', 'X').

    Bass notes ('/3') are dropped, since the models do not predict inversions.
    """
    label = label.strip()
    if label in (NO_CHORD, UNKNOWN_CHORD):
        return N if label == NO_CHORD else X
    label = label.split("/")[0]
    root, _, quality = label.partition(":")
    quality = quality or "maj"
    if quality not in QUALITY_INTERVALS:
        raise ValueError(f"unsupported chord quality: {quality!r}")
    return Chord(note_to_pc(root), quality)


@dataclass(frozen=True)
class Vocabulary:
    name: str
    chords: tuple[Chord, ...]

    def __len__(self) -> int:
        return len(self.chords)

    def index(self, chord: Chord) -> int:
        return self.chords.index(chord)


def _majmin() -> Vocabulary:
    # Index 2*pc is major, 2*pc+1 minor, 24 is N (upstream idx2chord).
    chords = []
    for pc in range(12):
        chords += [Chord(pc, "maj"), Chord(pc, "min")]
    return Vocabulary("majmin", tuple(chords + [N]))


def _full() -> Vocabulary:
    # Index 14*pc + q, then 168 = X and 169 = N (upstream idx2voca_chord).
    chords = [Chord(pc, q) for pc in range(12) for q in LARGE_QUALITIES]
    return Vocabulary("full", tuple(chords + [X, N]))


MAJMIN = _majmin()
FULL = _full()
VOCABULARIES = {"majmin": MAJMIN, "full": FULL}

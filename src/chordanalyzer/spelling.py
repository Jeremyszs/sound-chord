"""Display names, key-aware enharmonic spelling and Roman numerals."""

from __future__ import annotations

from .chords.vocab import Chord
from .key import Key

LETTERS = "CDEFGAB"
LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

MAJOR_TONICS = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_TONICS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]

# Semitones above the tonic -> (scale-degree step, accidental relative to the major scale).
# Minor keys use the same table, so A minor reads i, bIII, iv, v, bVI, bVII.
INTERVAL_DEGREE = {0: (0, ""), 1: (1, "b"), 2: (1, ""), 3: (2, "b"), 4: (2, ""), 5: (3, ""),
                   6: (3, "#"), 7: (4, ""), 8: (5, "b"), 9: (5, ""), 10: (6, "b"), 11: (6, "")}
ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]

DISPLAY_SUFFIX = {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7",
                  "minmaj7": "m(maj7)", "maj6": "6", "min6": "m6", "dim": "dim",
                  "dim7": "dim7", "hdim7": "m7b5", "aug": "aug", "sus2": "sus2", "sus4": "sus4"}
LOWER_CASE = {"min", "min7", "minmaj7", "min6", "dim", "dim7", "hdim7"}
ROMAN_SUFFIX = {"maj": "", "min": "", "7": "7", "maj7": "maj7", "min7": "7",
                "minmaj7": "(maj7)", "maj6": "6", "min6": "6", "dim": "°", "dim7": "°7",
                "hdim7": "ø7", "aug": "+", "sus2": "sus2", "sus4": "sus4"}


def tonic_name(key: Key) -> str:
    return (MAJOR_TONICS if key.mode == "major" else MINOR_TONICS)[key.tonic]


def uses_flats(key: Key) -> bool:
    return "b" in tonic_name(key) or tonic_name(key) in (
        ("F",) if key.mode == "major" else ("D", "G", "C", "F"))


def spell_pc(pc: int, key: Key | None) -> str:
    """Name a pitch class so it reads naturally in `key` (F# in D major, Gb in Db major)."""
    if key is None:
        return SHARP_NAMES[pc]
    tonic = tonic_name(key)
    step, _ = INTERVAL_DEGREE[(pc - key.tonic) % 12]
    letter = LETTERS[(LETTERS.index(tonic[0]) + step) % 7]
    offset = (pc - LETTER_PC[letter] + 6) % 12 - 6
    if offset in (-1, 0, 1):
        return letter + {-1: "b", 0: "", 1: "#"}[offset]
    return (FLAT_NAMES if uses_flats(key) else SHARP_NAMES)[pc]


def display_name(chord: Chord, key: Key | None) -> str:
    if not chord.is_chord:
        return "N.C." if chord.special == "N" else "X"
    return spell_pc(chord.root, key) + DISPLAY_SUFFIX[chord.quality]


def harte_label(chord: Chord, key: Key | None) -> str:
    """Harte label with the root spelled for the key, e.g. 'Bb:maj'."""
    if not chord.is_chord:
        return chord.special or "N"
    return f"{spell_pc(chord.root, key)}:{chord.quality}"


def roman_numeral(chord: Chord, key: Key) -> str | None:
    if not chord.is_chord:
        return None
    step, accidental = INTERVAL_DEGREE[(chord.root - key.tonic) % 12]
    numeral = ROMAN[step]
    if chord.quality in LOWER_CASE:
        numeral = numeral.lower()
    return accidental + numeral + ROMAN_SUFFIX[chord.quality]


def key_name(key: Key) -> str:
    return f"{tonic_name(key)} {key.mode}"


def key_label(key: Key) -> str:
    return f"{tonic_name(key)}:{'maj' if key.mode == 'major' else 'min'}"

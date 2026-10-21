import pytest

from chordanalyzer.chords.vocab import FULL, MAJMIN, Chord, parse_harte
from chordanalyzer.key import Key
from chordanalyzer.spelling import (
    display_name,
    harte_label,
    key_label,
    key_name,
    roman_numeral,
    spell_pc,
)

C_MAJ, A_MIN, D_MAJ, DB_MAJ, F_MAJ = (Key(0, "major"), Key(9, "minor"), Key(2, "major"),
                                      Key(1, "major"), Key(5, "major"))


def test_vocabulary_sizes_and_order():
    assert len(MAJMIN) == 25 and len(FULL) == 170
    assert MAJMIN.chords[2] == Chord(1, "maj") and MAJMIN.chords[3] == Chord(1, "min")
    assert MAJMIN.chords[24].special == "N"
    assert FULL.chords[1] == Chord(0, "maj") and FULL.chords[14 + 6] == Chord(1, "min7")
    assert FULL.chords[168].special == "X" and FULL.chords[169].special == "N"


@pytest.mark.parametrize("label,chord", [
    ("C", Chord(0, "maj")), ("Bb:min7", Chord(10, "min7")), ("F#:hdim7", Chord(6, "hdim7")),
    ("E:maj/3", Chord(4, "maj")), ("N", Chord(None, "N")), ("X", Chord(None, "X")),
])
def test_parse_harte(label, chord):
    assert parse_harte(label) == chord


@pytest.mark.parametrize("pc,key,name", [
    (6, D_MAJ, "F#"), (6, DB_MAJ, "Gb"), (10, F_MAJ, "Bb"), (10, C_MAJ, "Bb"),
    (8, C_MAJ, "Ab"), (6, C_MAJ, "F#"), (8, A_MIN, "G#"), (1, None, "C#"),
])
def test_spell_pc(pc, key, name):
    assert spell_pc(pc, key) == name


@pytest.mark.parametrize("label,key,name,roman", [
    ("D:min7", C_MAJ, "Dm7", "ii7"),
    ("G:7", C_MAJ, "G7", "V7"),
    ("A#:maj", C_MAJ, "Bb", "bVII"),
    ("B:hdim7", C_MAJ, "Bm7b5", "viiø7"),
    ("C:maj", A_MIN, "C", "bIII"),
    ("E:7", A_MIN, "E7", "V7"),
    ("F#:min", D_MAJ, "F#m", "iii"),
    ("C#:maj", DB_MAJ, "Db", "I"),
    ("G#:dim7", A_MIN, "G#dim7", "vii°7"),
])
def test_names_and_roman(label, key, name, roman):
    chord = parse_harte(label)
    assert display_name(chord, key) == name
    assert roman_numeral(chord, key) == roman


def test_no_chord():
    n = parse_harte("N")
    assert display_name(n, C_MAJ) == "N.C." and roman_numeral(n, C_MAJ) is None
    assert harte_label(n, C_MAJ) == "N"


def test_key_names():
    assert key_name(A_MIN) == "A minor" and key_label(DB_MAJ) == "Db:maj"
    assert harte_label(parse_harte("A#:min7"), DB_MAJ) == "Bb:min7"

"""The JSON contract. Adding fields is non-breaking; renaming or removing one bumps
SCHEMA_VERSION and the URL prefix."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

ChordVocabulary = Literal["majmin", "full"]
Notation = Literal["name", "roman", "both"]
IncludeField = Literal["beats", "tempo_curve", "bars", "chord_probabilities"]
DEFAULT_INCLUDE: tuple[IncludeField, ...] = ("beats", "bars")


class EngineInfo(BaseModel):
    version: str
    chord_model: str
    beat_model: str
    key_model: str


class FileInfo(BaseModel):
    sha256: str
    duration: float = Field(description="Seconds")
    format: str
    codec: str


class TempoInfo(BaseModel):
    bpm: float
    confidence: float = Field(ge=0, le=1)
    bpm_alternatives: list[float] = Field(description="Half and double tempo")
    beats_per_bar: int
    time_signature: str


class KeyInfo(BaseModel):
    tonic: str = Field(description="Spelled tonic, e.g. 'F#' or 'Bb'")
    mode: Literal["major", "minor"]
    name: str = Field(description="e.g. 'A minor'")
    label: str = Field(description="Harte-style key label, e.g. 'A:min'")


class KeyResultInfo(KeyInfo):
    confidence: float = Field(ge=0, le=1)
    runner_up: KeyInfo


class ChordSegment(BaseModel):
    start: float
    end: float
    label: str = Field(description="Harte syntax, root:quality, or N (no chord) / X (unknown)")
    name: str | None = Field(None, description="Display name, e.g. 'F#m7'")
    roman: str | None = Field(None, description="Roman numeral relative to the key, e.g. 'ii7'")
    bar: int = Field(description="1-based from the first downbeat; 0 = pickup before it")
    beat: int = Field(description="1-based beat within the bar")
    confidence: float = Field(ge=0, le=1)


class ProgressionEntry(BaseModel):
    bar: int = Field(description="First bar of this entry")
    bars: int = Field(description="Number of bars the chord is held (repeats collapsed)")
    label: str
    name: str | None = None
    roman: str | None = None


class Beat(BaseModel):
    time: float
    bar: int
    beat: int


class Bar(BaseModel):
    bar: int
    start: float
    end: float


class TempoCurvePoint(BaseModel):
    bar: int
    start: float
    bpm: float


class ChordProbabilities(BaseModel):
    labels: list[str]
    frame_period: float = Field(description="Seconds between frames; frame i is centred on i * frame_period")
    frames: list[list[float]]


class AnalysisResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    engine: EngineInfo
    file: FileInfo
    tempo: TempoInfo
    key: KeyResultInfo
    chord_vocabulary: ChordVocabulary
    chords: list[ChordSegment]
    progression: list[ProgressionEntry]
    beats: list[Beat] | None = None
    bars: list[Bar] | None = None
    tempo_curve: list[TempoCurvePoint] | None = None
    chord_probabilities: ChordProbabilities | None = None
    warnings: list[str] = Field(default_factory=list, description=(
        "Codes such as low_tempo_confidence, ambiguous_key, mostly_no_chord, "
        "clipped_audio, beat_tracker_fallback"))

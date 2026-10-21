"""Beat and downbeat tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class BeatResult:
    beats: np.ndarray  # seconds
    downbeats: np.ndarray  # seconds, a subset of beats


class BeatTracker(Protocol):
    name: str

    def track(self, y: np.ndarray, sr: int) -> BeatResult: ...


class BeatThisTracker:
    """Beat This! (Foscarin et al., ISMIR 2024), MIT code and weights."""

    def __init__(self, checkpoint: str = "final0", models_dir: Path | None = None,
                 device: str = "cpu"):
        from beat_this.inference import Audio2Beats

        self.name = f"beat_this-{checkpoint}"
        local = Path(models_dir) / f"beat_this-{checkpoint}.ckpt" if models_dir else None
        # A shortname makes beat_this download the checkpoint into the torch hub cache.
        path = str(local) if local is not None and local.exists() else checkpoint
        self._model = Audio2Beats(checkpoint_path=path, device=device, dbn=False)

    def track(self, y: np.ndarray, sr: int) -> BeatResult:
        beats, downbeats = self._model(y, sr)
        return BeatResult(np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float))


class LibrosaBeatTracker:
    """Fallback when Beat This! weights are unavailable: onset-based beats, 4/4 assumed.

    Noticeably weaker than Beat This!, and its downbeats are a guess: the beat
    phase with the strongest low-frequency onsets is taken as beat one.
    """

    name = "librosa-beat_track"

    def __init__(self, beats_per_bar: int = 4):
        self.beats_per_bar = beats_per_bar

    def track(self, y: np.ndarray, sr: int) -> BeatResult:
        import librosa

        hop = 512
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
        _, frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop,
                                            trim=False)
        beats = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
        if len(beats) < self.beats_per_bar:
            return BeatResult(beats, beats[:1])
        mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop, n_mels=64, fmax=8000)
        low = librosa.onset.onset_strength(S=librosa.power_to_db(mel[:8]), sr=sr,
                                           hop_length=hop)
        frames = np.clip(frames, 0, len(low) - 1)
        scores = [low[frames[p::self.beats_per_bar]].mean() for p in range(self.beats_per_bar)]
        phase = int(np.argmax(scores))
        return BeatResult(beats, beats[phase::self.beats_per_bar])


def make_beat_tracker(kind: str = "auto", checkpoint: str = "final0",
                      models_dir: Path | None = None, device: str = "cpu"
                      ) -> tuple[BeatTracker, list[str]]:
    """Build the configured tracker. Returns the tracker and any warnings.

    kind: 'beat_this', 'librosa', or 'auto' (Beat This!, falling back to librosa
    if its weights cannot be loaded).
    """
    if kind == "librosa":
        return LibrosaBeatTracker(), ["beat_tracker_fallback"]
    try:
        return BeatThisTracker(checkpoint, models_dir, device), []
    except Exception:
        if kind == "beat_this":
            raise
        return LibrosaBeatTracker(), ["beat_tracker_fallback"]

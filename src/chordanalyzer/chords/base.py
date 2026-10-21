"""Interface every chord model implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .vocab import Vocabulary


@dataclass
class ChordFrames:
    """Per-frame chord posteriors.

    probs has shape (n_frames, len(vocabulary)); frame i is centred on
    time i * frame_period seconds.
    """

    probs: np.ndarray
    frame_period: float
    vocabulary: Vocabulary
    chroma: np.ndarray | None = None  # optional (n_frames, 12) chroma from the model's features

    @property
    def times(self) -> np.ndarray:
        return np.arange(len(self.probs)) * self.frame_period


class ChordModel(Protocol):
    name: str
    vocabulary: Vocabulary

    def predict(self, y: np.ndarray, sr: int) -> ChordFrames: ...

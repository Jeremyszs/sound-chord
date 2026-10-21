"""BTC inference: log-CQT features and overlapping-window prediction."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch

from .base import ChordFrames
from .btc_model import BTCModel
from .vocab import FULL, MAJMIN, Vocabulary

SR = 22050
HOP = 2048
N_BINS = 144
BINS_PER_OCTAVE = 24
WINDOW = 108  # frames per model input, about 10 s (fixed by the model's positional encoding)

WEIGHT_FILES = {"majmin": "btc_majmin.safetensors", "full": "btc_full.safetensors"}


def cqt_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Log-magnitude CQT as BTC was trained on it: (n_frames, 144)."""
    if sr != SR:
        raise ValueError(f"BTC expects {SR} Hz audio, got {sr}")
    c = librosa.cqt(y, sr=SR, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE, hop_length=HOP)
    return np.log(np.abs(c) + 1e-6).T.astype(np.float32)


def cqt_to_chroma(log_cqt: np.ndarray) -> np.ndarray:
    """Fold the 144-bin log-CQT (2 bins per semitone from C1) into 12 pitch classes."""
    mag = np.exp(log_cqt)
    center, upper = mag[:, 0::2], mag[:, 1::2]  # bins on each semitone and a quarter-tone above
    lower = np.pad(upper[:, :-1], ((0, 0), (1, 0)))
    semitones = center + 0.5 * (upper + lower)
    return semitones.reshape(len(mag), -1, 12).sum(axis=1)


def load_btc_weights(path: Path) -> tuple[dict[str, torch.Tensor], float, float]:
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    mean = float(tensors.pop("feature_mean"))
    std = float(tensors.pop("feature_std"))
    return tensors, mean, std


class BTCChordModel:
    """BTC with 50%-overlapping windows; softmax outputs are averaged where windows overlap."""

    def __init__(self, vocabulary: str, models_dir: Path, device: str = "cpu"):
        self.vocabulary: Vocabulary = {"majmin": MAJMIN, "full": FULL}[vocabulary]
        self.name = f"btc-{vocabulary}"
        self.device = torch.device(device)
        state, self.mean, self.std = load_btc_weights(Path(models_dir) / WEIGHT_FILES[vocabulary])
        self.model = BTCModel(num_chords=len(self.vocabulary))
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def frame_probs(self, features: np.ndarray, hop: int = WINDOW // 2) -> np.ndarray:
        """Run the model over normalised features of shape (n_frames, 144)."""
        n = len(features)
        x = (features - self.mean) / self.std
        starts = list(range(0, max(n - WINDOW, 0) + 1, hop))
        if starts[-1] + WINDOW < n:
            starts.append(n - WINDOW)
        padded = np.pad(x, ((0, max(0, WINDOW - n)), (0, 0)))
        batch = np.stack([padded[s:s + WINDOW] for s in starts])
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(batch).to(self.device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        # Taper window edges, where the model has the least context.
        weight = np.hanning(WINDOW + 2)[1:-1].astype(np.float32)
        total = np.zeros((max(n, WINDOW), probs.shape[-1]), np.float32)
        norm = np.zeros(max(n, WINDOW), np.float32)
        for s, p in zip(starts, probs, strict=True):
            total[s:s + WINDOW] += p * weight[:, None]
            norm[s:s + WINDOW] += weight
        return (total / norm[:, None])[:n]

    def predict(self, y: np.ndarray, sr: int) -> ChordFrames:
        features = cqt_features(y, sr)
        probs = self.frame_probs(features)
        return ChordFrames(probs=probs, frame_period=HOP / SR, vocabulary=self.vocabulary,
                           chroma=cqt_to_chroma(features))

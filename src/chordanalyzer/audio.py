"""Validate uploads with ffprobe and decode them once with ffmpeg."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 22050


class AudioError(Exception):
    """Base class for problems with the input audio."""


class UnsupportedAudio(AudioError):
    """The file is not decodable audio."""


class DurationOutOfRange(AudioError):
    def __init__(self, duration: float, min_s: float, max_s: float):
        self.duration = duration
        super().__init__(
            f"audio is {duration:.1f} s long; allowed range is {min_s:g} to {max_s:g} s")


@dataclass
class ProbeResult:
    format_name: str
    codec: str
    duration: float
    sample_rate: int
    channels: int


def probe(path: str | Path, timeout: float = 30) -> ProbeResult:
    """Inspect the real container and codec, ignoring the file extension."""
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
           "-show_streams", "-select_streams", "a:0", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=True).stdout
        info = json.loads(out)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        raise UnsupportedAudio("file could not be read as audio") from e
    streams = info.get("streams") or []
    if not streams:
        raise UnsupportedAudio("file has no audio stream")
    s, fmt = streams[0], info.get("format", {})
    duration = s.get("duration") or fmt.get("duration")
    if duration is None:
        raise UnsupportedAudio("could not determine audio duration")
    return ProbeResult(
        format_name=fmt.get("format_name", ""),
        codec=s.get("codec_name", ""),
        duration=float(duration),
        sample_rate=int(s.get("sample_rate", 0) or 0),
        channels=int(s.get("channels", 0) or 0),
    )


def decode(path: str | Path, sr: int = SR, timeout: float = 120) -> np.ndarray:
    """Decode any ffmpeg-readable file to mono float32 at `sr` Hz."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:a:0",
           "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=True).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        raise UnsupportedAudio("file could not be decoded") from e
    y = np.frombuffer(out, dtype=np.float32).copy()
    if y.size == 0:
        raise UnsupportedAudio("file decoded to no samples")
    return y


@dataclass
class LoadedAudio:
    y: np.ndarray  # peak-normalised mono float32
    sr: int
    duration: float
    clipped_fraction: float
    probe: ProbeResult


def load(path: str | Path, min_duration: float = 5.0, max_duration: float = 600.0) -> LoadedAudio:
    info = probe(path)
    if not (min_duration <= info.duration <= max_duration):
        raise DurationOutOfRange(info.duration, min_duration, max_duration)
    y = decode(path)
    clipped = float(np.mean(np.abs(y) >= 0.999))
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y / peak
    return LoadedAudio(y=y, sr=SR, duration=len(y) / SR, clipped_fraction=clipped, probe=info)

import subprocess

import numpy as np
import pytest

from chordanalyzer.audio import DurationOutOfRange, UnsupportedAudio, decode, load, probe


def _make(tmp_path, ext, seconds=6.0, extra=()):
    path = tmp_path / f"tone{ext}"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={seconds}:sample_rate=44100", "-ac", "2",
                    *extra, str(path)], check=True)
    return path


@pytest.mark.parametrize("ext", [".wav", ".mp3", ".flac", ".ogg", ".m4a"])
def test_decodes_common_formats(tmp_path, ext):
    audio = load(_make(tmp_path, ext))
    assert audio.sr == 22050
    assert audio.duration == pytest.approx(6.0, abs=0.1)
    assert np.max(np.abs(audio.y)) == pytest.approx(1.0)


def test_extension_is_not_trusted(tmp_path):
    real = _make(tmp_path, ".mp3")
    renamed = tmp_path / "song.wav"
    renamed.write_bytes(real.read_bytes())
    assert probe(renamed).codec == "mp3"


def test_rejects_non_audio(tmp_path):
    bad = tmp_path / "notes.mp3"
    bad.write_text("this is not audio")
    with pytest.raises(UnsupportedAudio):
        load(bad)
    with pytest.raises(UnsupportedAudio):
        decode(bad)


def test_rejects_video_without_audio(tmp_path):
    path = tmp_path / "video.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=32x32:d=6",
                    str(path)], check=True)
    with pytest.raises(UnsupportedAudio):
        load(path)


def test_duration_limits(tmp_path):
    with pytest.raises(DurationOutOfRange):
        load(_make(tmp_path, ".wav", seconds=2.0))
    with pytest.raises(DurationOutOfRange):
        load(_make(tmp_path, ".wav", seconds=8.0), max_duration=7.0)

"""Tempo, key and chord-progression analysis for audio files."""

__version__ = "0.1.0"

from .pipeline import AnalysisOptions, Analyzer  # noqa: E402

__all__ = ["Analyzer", "AnalysisOptions", "__version__"]

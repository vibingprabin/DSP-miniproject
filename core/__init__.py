"""
core — DSP engine modules for the Acoustic HCI System.
"""

from .audio_streamer import AudioStreamer
from .calibration_engine import SpectroTemporalProfiler
from .event_detector import EventDetector
from .macro_executor import MacroExecutor, MACRO_TYPE_HINTS
from . import dsp_utils
from . import macro_store

__all__ = [
    "AudioStreamer",
    "SpectroTemporalProfiler",
    "EventDetector",
    "MacroExecutor",
    "MACRO_TYPE_HINTS",
    "dsp_utils",
    "macro_store",
]

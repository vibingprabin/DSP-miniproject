"""
gui — PyQt6 dashboard modules for the Acoustic HCI System.

Components:
    MainWindow        : Top-level window with layout orchestration
    CalibrationPanel  : Calibration buttons and status display
    VisualizerPanel   : Time-domain, FFT, and Z-plane plots
    ConsolePanel      : Timestamped event log console
"""

from .main_window import MainWindow
from .calibration_panel import CalibrationPanel
from .visualizer_panel import VisualizerPanel
from .console_panel import ConsolePanel
from .macro_dialog import MacroDialog

__all__ = [
    "MainWindow",
    "CalibrationPanel",
    "VisualizerPanel",
    "ConsolePanel",
    "MacroDialog",
]

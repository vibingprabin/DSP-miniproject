import sys
import os
from PyQt6.QtWidgets import QApplication
from core import AudioStreamer, SpectroTemporalProfiler, EventDetector, MacroExecutor
from gui import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Load dark theme
    style_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.qss')
    if os.path.exists(style_path):
        with open(style_path, 'r') as f:
            app.setStyleSheet(f.read())
            
    # Initialize Core DSP Engine
    streamer = AudioStreamer()
    calibration = SpectroTemporalProfiler()
    detector = EventDetector(calibration)
    macro_exec = MacroExecutor()
    
    # Initialize and show GUI
    window = MainWindow(streamer, calibration, detector, macro_exec)
    window.show()
    
    # Cleanup on exit
    exit_code = app.exec()
    if streamer.is_active:
        streamer.stop_stream()
        
    sys.exit(exit_code)

if __name__ == '__main__':
    main()

import threading
import numpy as np
import scipy.signal as sig
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QMessageBox
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QFont

from .calibration_panel import CalibrationPanel
from .visualizer_panel import VisualizerPanel, ZPlanePlot
from .console_panel import ConsolePanel
from .macro_dialog import MacroDialog
from core import macro_store
import config

class MainWindow(QMainWindow):
    def __init__(self, streamer, calibration, detector, macro_exec):
        super().__init__()
        self.streamer = streamer
        self.calibration = calibration # SpectroTemporalProfiler
        self.detector = detector # EventDetector
        self.macro_exec = macro_exec
        
        self.setWindowTitle("🎤 Acoustic HCI — STEC v2 Dynamic Gestures")
        self.setMinimumSize(1200, 750)
        
        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        
        # Left Column: Calibration
        self.calibration_panel = CalibrationPanel()
        self.calibration_panel.setFixedWidth(320)
        main_layout.addWidget(self.calibration_panel)
        
        # Connect calibration signals
        self.calibration_panel.record_gesture_clicked.connect(self._on_record_gesture)
        self.calibration_panel.delete_gesture_clicked.connect(self._on_delete_gesture)
        self.calibration_panel.add_mapping_clicked.connect(self._on_add_mapping)
        self.calibration_panel.delete_mapping_clicked.connect(self._on_delete_mapping)
        self.calibration_panel.new_macro_clicked.connect(self._on_new_macro)
        self.calibration_panel.delete_macro_clicked.connect(self._on_delete_macro)

        # Route macro-executor status messages into the console.
        self.macro_exec.logger = lambda msg: self.console.log(msg, "warning")
        
        # Center Column: Visualizers + Console
        center_layout = QVBoxLayout()
        self.visualizer = VisualizerPanel()
        center_layout.addWidget(self.visualizer, stretch=2)
        
        self.console = ConsolePanel()
        center_layout.addWidget(self.console, stretch=1)
        
        main_layout.addLayout(center_layout, stretch=1)
        
        # Right Column: Z-Plane + Controls
        right_layout = QVBoxLayout()
        right_layout.setSpacing(15)
        
        self.zplane_plot = ZPlanePlot()
        self.zplane_plot.setFixedSize(300, 300)
        right_layout.addWidget(self.zplane_plot)
        
        # Show poles/zeros for the first STEC band just as a DSP visual.
        # Filters are now second-order sections (SOS), so use sos2zpk.
        z, p, k = sig.sos2zpk(self.calibration.filters[0])
        self.zplane_plot.update_zpk(z, p)
        
        # State Indicator
        self.lbl_state = QLabel("● IDLE")
        font = QFont()
        font.setPointSize(24)
        font.setBold(True)
        self.lbl_state.setFont(font)
        self.lbl_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_state.setStyleSheet(f"color: {config.COLORS['secondary']};")
        right_layout.addWidget(self.lbl_state)
        
        # Controls
        self.btn_start = QPushButton("▶ Start Listening")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.setEnabled(False) # Require calibration first
        self.btn_start.clicked.connect(self._start_listening)
        
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setMinimumHeight(50)
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_listening)
        
        right_layout.addWidget(self.btn_start)
        right_layout.addWidget(self.btn_stop)
        right_layout.addStretch()
        
        main_layout.addLayout(right_layout)
        
        # Main Processing Timer
        self.timer = QTimer(self)
        self.timer.setInterval(config.GUI_UPDATE_INTERVAL_MS)
        self.timer.timeout.connect(self._process_audio)

        # Load any custom macros the user saved in a previous session.
        n_loaded = macro_store.load_into_config()
        self.calibration_panel.refresh_macro_list()

        self.console.log("System initialized. Please add a gesture...", "info")
        if n_loaded:
            self.console.log(f"Loaded {n_loaded} custom macro(s) from disk.", "success")

    def _on_record_gesture(self, gesture_name):
        # Validate unique name
        if gesture_name in self.calibration.templates:
            self.console.log(f"Gesture '{gesture_name}' already exists. Please delete it first or use a new name.", "warning")
            return
            
        self.console.log(
            f"Recording '{gesture_name}' for 3 seconds. Repeat the sound a few "
            f"times clearly — each repetition becomes a template exemplar.", "warning")
        self.calibration_panel.set_recording_active(True)
        
        def task():
            audio = self.streamer.record_duration(config.CALIBRATION_GESTURE_DURATION)
            return audio
            
        result_container = {}
        def target():
            try:
                result_container['audio'] = task()
            except Exception as e:
                result_container['error'] = str(e)
                
        thread = threading.Thread(target=target)
        thread.start()
        
        def on_done():
            if 'error' in result_container:
                self.calibration_panel.set_recording_active(False)
                self.console.log(f"Recording crash: {result_container['error']}", "error")
            else:
                self._on_gesture_recorded(gesture_name, result_container['audio'])
                
        self._check_thread(thread, on_done)

    def _on_gesture_recorded(self, gesture_name, audio):
        self.calibration_panel.set_recording_active(False)
        
        try:
            self.calibration.train_gesture(audio, gesture_name)
            self.console.log(f"Successfully trained structural profile for '{gesture_name}'.", "success")
        except ValueError as e:
            self.console.log(f"Recording failed: {str(e)}", "error")
            QMessageBox.warning(self, "Recording Failed", str(e))
            
        self._refresh_ui_lists()

    def _on_delete_gesture(self, gesture_name):
        self.calibration.delete_gesture(gesture_name)
        # We must also clean up mappings that used this gesture
        config.ACTIVE_MAPPINGS = [m for m in config.ACTIVE_MAPPINGS if gesture_name not in m[0]]
        self.console.log(f"Deleted gesture '{gesture_name}'.", "warning")
        self._refresh_ui_lists()
        
    def _on_add_mapping(self, g1, g2, mac):
        seq = (g1, g2) if g2 else (g1,)
        # Check if seq already exists
        for existing_seq, _ in config.ACTIVE_MAPPINGS:
            if existing_seq == seq:
                self.console.log("This gesture sequence is already mapped.", "warning")
                return
                
        config.ACTIVE_MAPPINGS.append((seq, mac))
        self.console.log(f"Mapped sequence {seq} to macro {mac}", "success")
        self._refresh_ui_lists()
        
    def _on_delete_mapping(self, row):
        if 0 <= row < len(config.ACTIVE_MAPPINGS):
            seq, mac = config.ACTIVE_MAPPINGS.pop(row)
            self.console.log(f"Deleted mapping for sequence {seq}", "warning")
            self._refresh_ui_lists()

    def _on_new_macro(self):
        dialog = MacroDialog(self, executor=self.macro_exec)
        if dialog.exec():
            result = dialog.result_macro()
            if not result:
                return
            macro_id, defn = result
            if macro_id in config.MACRO_DEFINITIONS and not config.MACRO_DEFINITIONS[macro_id].get("custom"):
                self.console.log(f"'{defn['label']}' clashes with a built-in macro name.", "warning")
                return
            macro_store.add_macro(macro_id, defn)
            self.calibration_panel.refresh_macro_list()
            self.console.log(f"Added custom macro '{defn['label']}' ({defn['type']}).", "success")

    def _on_delete_macro(self, macro_id):
        defn = config.MACRO_DEFINITIONS.get(macro_id, {})
        if not defn.get("custom"):
            self.console.log("Only custom macros (★) can be deleted; built-ins are protected.", "warning")
            return
        # Drop mappings that referenced it.
        config.ACTIVE_MAPPINGS = [m for m in config.ACTIVE_MAPPINGS if m[1] != macro_id]
        macro_store.delete_macro(macro_id)
        self.calibration_panel.refresh_macro_list()
        self._refresh_ui_lists()
        self.console.log(f"Deleted custom macro '{defn.get('label', macro_id)}'.", "warning")

    def _refresh_ui_lists(self):
        gestures = list(self.calibration.templates.keys())
        self.calibration_panel.refresh_gesture_list(gestures)
        self.calibration_panel.refresh_mapping_list(config.ACTIVE_MAPPINGS)
        
        if len(gestures) > 0 and len(config.ACTIVE_MAPPINGS) > 0:
            self.btn_start.setEnabled(True)
        else:
            self.btn_start.setEnabled(False)

    def _check_thread(self, thread, callback):
        if thread.is_alive():
            QTimer.singleShot(100, lambda: self._check_thread(thread, callback))
        else:
            callback()

    def _start_listening(self):
        if not self.calibration.is_calibrated:
            self.console.log("Cannot start: Missing calibration.", "error")
            return
            
        self.streamer.start_stream()
        self.timer.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.console.log("Started live listening...", "success")

    def _stop_listening(self):
        self.timer.stop()
        self.streamer.stop_stream()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_state.setText("● IDLE")
        self.lbl_state.setStyleSheet(f"color: {config.COLORS['secondary']};")
        self.console.log("Stopped listening.", "warning")

    def _process_audio(self):
        while True:
            chunk = self.streamer.get_chunk()
            if chunk is None:
                break
                
            chunk_flat = chunk.flatten()
            
            # Update visuals
            self.visualizer.update_waveform(chunk_flat)
            freqs, mags = self.calibration.compute_fft(chunk_flat)
            self.visualizer.update_spectrum(freqs, mags)
            
            # Detect events
            events = self.detector.process_chunk(chunk_flat)

            # Live onset-detection-function curve.
            self.visualizer.update_odf(self.detector.odf_history)
            for event in events:
                if event.get('state_change'):
                    self.console.log(f"State transition -> {event['state_change']} (Match: {event.get('similarity', 0):.2f})", "detect")
                    self._update_state_display(event['state_change'])
                elif event.get('gesture') != 'timeout':
                    is_unrec = "Unrecognized" in event['gesture']
                    suffix = "" if is_unrec else " - added to sequence"
                    color = "warning" if is_unrec else "info"
                    margin = event.get('margin', 0.0)
                    self.console.log(
                        f"Heard {event['gesture']} (Match: {event.get('similarity', 0):.2f}, "
                        f"margin: {margin:.2f}){suffix}", color)
                    # Refresh the live STEC fingerprint heatmap.
                    name, score = self.detector.last_match
                    self.visualizer.update_profile(
                        self.detector.last_profile,
                        f"Last STEC Fingerprint — {name} ({score:.2f})")
                    
                if 'macro' in event:
                    macro = event['macro']
                    self.console.log(f"Firing MACRO: {macro}", "error")
                    success = self.macro_exec.execute(macro)
                    if success:
                        self.console.log(f"Macro {macro} executed successfully.", "success")
                    else:
                        self.console.log(f"Macro {macro} failed.", "error")

    def _update_state_display(self, state):
        self.lbl_state.setText(f"● {state}")
        color = config.COLORS['secondary']
        if state == 'ARMED':
            color = config.COLORS['warning']
        elif state == 'EXECUTE':
            color = config.COLORS['error']
        self.lbl_state.setStyleSheet(f"color: {color};")

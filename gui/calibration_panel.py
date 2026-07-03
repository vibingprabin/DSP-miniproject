from PyQt6.QtWidgets import (QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
                               QProgressBar, QComboBox, QFrame, QLineEdit, QListWidget, QFormLayout)
from PyQt6.QtCore import pyqtSignal
import config

class CalibrationPanel(QGroupBox):
    record_gesture_clicked = pyqtSignal(str)
    delete_gesture_clicked = pyqtSignal(str)
    add_mapping_clicked = pyqtSignal(str, str, str) # gesture1, gesture2, macro
    delete_mapping_clicked = pyqtSignal(int)
    new_macro_clicked = pyqtSignal()
    delete_macro_clicked = pyqtSignal(str)

    def __init__(self):
        super().__init__("Dynamic Gestures & Macros")
        layout = QVBoxLayout(self)
        
        # 1. Add Gesture Section
        gesture_group = QGroupBox("Record New Gesture")
        glayout = QVBoxLayout(gesture_group)
        
        row1 = QHBoxLayout()
        self.input_gesture_name = QLineEdit()
        self.input_gesture_name.setPlaceholderText("e.g., Left Snap")
        self.btn_record_gesture = QPushButton("🔴 Record")
        self.btn_record_gesture.clicked.connect(self._on_record_clicked)
        row1.addWidget(self.input_gesture_name)
        row1.addWidget(self.btn_record_gesture)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        
        glayout.addLayout(row1)
        glayout.addWidget(self.progress_bar)
        
        self.list_gestures = QListWidget()
        glayout.addWidget(QLabel("Trained Gestures:"))
        glayout.addWidget(self.list_gestures)
        
        self.btn_delete_gesture = QPushButton("Delete Selected Gesture")
        self.btn_delete_gesture.clicked.connect(self._on_delete_gesture)
        glayout.addWidget(self.btn_delete_gesture)
        
        layout.addWidget(gesture_group)
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)
        
        # 2. Macro Allocator Section
        macro_group = QGroupBox("Macro Sequences")
        mlayout = QVBoxLayout(macro_group)
        
        form = QFormLayout()
        self.combo_trig1 = QComboBox()
        self.combo_trig2 = QComboBox()
        self.combo_trig2.addItem("-- None (Single Action) --", "")
        
        self.combo_macro = QComboBox()
        self.refresh_macro_list()

        form.addRow("Trigger 1:", self.combo_trig1)
        form.addRow("Trigger 2 (Opt):", self.combo_trig2)
        form.addRow("Action:", self.combo_macro)

        # Custom macro management.
        macro_btn_row = QHBoxLayout()
        self.btn_new_macro = QPushButton("＋ New Macro")
        self.btn_new_macro.clicked.connect(lambda: self.new_macro_clicked.emit())
        self.btn_del_macro = QPushButton("🗑 Delete Macro")
        self.btn_del_macro.clicked.connect(self._on_delete_macro)
        macro_btn_row.addWidget(self.btn_new_macro)
        macro_btn_row.addWidget(self.btn_del_macro)
        form.addRow("Actions:", macro_btn_row)

        self.btn_add_mapping = QPushButton("Add Sequence")
        self.btn_add_mapping.clicked.connect(self._on_add_mapping)
        form.addRow("", self.btn_add_mapping)
        
        mlayout.addLayout(form)
        
        self.list_mappings = QListWidget()
        mlayout.addWidget(QLabel("Active Sequences:"))
        mlayout.addWidget(self.list_mappings)
        
        self.btn_delete_mapping = QPushButton("Delete Selected Sequence")
        self.btn_delete_mapping.clicked.connect(self._on_delete_mapping)
        mlayout.addWidget(self.btn_delete_mapping)
        
        layout.addWidget(macro_group)
        
    def _on_record_clicked(self):
        name = self.input_gesture_name.text().strip()
        if name:
            self.record_gesture_clicked.emit(name)
            
    def _on_delete_gesture(self):
        item = self.list_gestures.currentItem()
        if item:
            name = item.text()
            self.delete_gesture_clicked.emit(name)
            
    def _on_add_mapping(self):
        g1 = self.combo_trig1.currentData()
        g2 = self.combo_trig2.currentData()
        mac = self.combo_macro.currentData()
        if g1:
            self.add_mapping_clicked.emit(g1, g2, mac)
            
    def _on_delete_mapping(self):
        row = self.list_mappings.currentRow()
        if row >= 0:
            self.delete_mapping_clicked.emit(row)

    def _on_delete_macro(self):
        macro_id = self.combo_macro.currentData()
        if macro_id:
            self.delete_macro_clicked.emit(macro_id)

    def refresh_macro_list(self):
        """Repopulate the action combo from config, preserving selection."""
        current = self.combo_macro.currentData()
        self.combo_macro.clear()
        for key, val in config.MACRO_DEFINITIONS.items():
            tag = "  ★" if val.get("custom") else ""
            self.combo_macro.addItem(f"{val['label']} [{val.get('type','?')}]{tag}", key)
        if current is not None:
            i = self.combo_macro.findData(current)
            if i >= 0:
                self.combo_macro.setCurrentIndex(i)

    def set_recording_progress(self, value: int):
        if value < 100:
            self.progress_bar.show()
            self.progress_bar.setValue(value)
        else:
            self.progress_bar.hide()
            self.progress_bar.setValue(0)

    def set_recording_active(self, active: bool):
        self.btn_record_gesture.setEnabled(not active)
        self.input_gesture_name.setEnabled(not active)

    def refresh_gesture_list(self, gestures: list[str]):
        self.list_gestures.clear()
        self.combo_trig1.clear()
        
        # Reset trig2 keeping the "None" option
        self.combo_trig2.clear()
        self.combo_trig2.addItem("-- None (Single Action) --", "")
        
        for g in gestures:
            self.list_gestures.addItem(g)
            self.combo_trig1.addItem(g, g)
            self.combo_trig2.addItem(g, g)
            
    def refresh_mapping_list(self, mappings: list):
        self.list_mappings.clear()
        for seq, mac in mappings:
            macro_label = config.MACRO_DEFINITIONS.get(mac, {}).get("label", mac)
            label = " → ".join(seq) + f"  :  {macro_label}"
            self.list_mappings.addItem(label)

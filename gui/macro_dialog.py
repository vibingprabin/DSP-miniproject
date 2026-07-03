"""
macro_dialog.py — Modal dialog for creating a custom macro at runtime.

Lets the user pick an action type (hotkey, text, url, open, launch, shell, ...),
enter a label and a value, and returns a (macro_id, definition) pair.
"""

import re

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QLineEdit, QComboBox, QDialogButtonBox,
    QLabel, QPushButton, QHBoxLayout,
)

from core import MACRO_TYPE_HINTS


class MacroDialog(QDialog):
    def __init__(self, parent=None, executor=None):
        super().__init__(parent)
        self.executor = executor
        self.setWindowTitle("Create Custom Macro")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.input_label = QLineEdit()
        self.input_label.setPlaceholderText("e.g., Open ChatGPT")

        self.combo_type = QComboBox()
        for t in MACRO_TYPE_HINTS:
            self.combo_type.addItem(t, t)
        self.combo_type.currentTextChanged.connect(self._update_hint)

        self.input_value = QLineEdit()
        self.hint = QLabel()
        self.hint.setStyleSheet("color: #8b949e; font-size: 11px;")

        form.addRow("Name:", self.input_label)
        form.addRow("Type:", self.combo_type)
        form.addRow("Value:", self.input_value)
        form.addRow("", self.hint)
        layout.addLayout(form)

        # Test-fire the macro before committing.
        test_row = QHBoxLayout()
        self.btn_test = QPushButton("Test (dry-run)")
        self.btn_test.clicked.connect(self._on_test)
        self.lbl_test = QLabel("")
        test_row.addWidget(self.btn_test)
        test_row.addWidget(self.lbl_test, stretch=1)
        layout.addLayout(test_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._update_hint(self.combo_type.currentText())
        self._result = None

    def _update_hint(self, atype):
        self.hint.setText("Hint: " + MACRO_TYPE_HINTS.get(atype, ""))
        self.input_value.setPlaceholderText(MACRO_TYPE_HINTS.get(atype, ""))

    def _build_definition(self):
        label = self.input_label.text().strip()
        atype = self.combo_type.currentData()
        value = self.input_value.text().strip()
        if not label or not value:
            return None, None

        if atype in ("hotkey", "press"):
            keys = [k.strip().lower() for k in re.split(r"[+\s,]+", value) if k.strip()]
            defn = {"label": label, "type": atype, "keys": keys}
        else:
            defn = {"label": label, "type": atype, "value": value}

        macro_id = "USER_" + re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").upper()
        return macro_id, defn

    def _on_test(self):
        mid, defn = self._build_definition()
        if not defn:
            self.lbl_test.setText("Enter a name and value first.")
            return
        if self.executor is None:
            self.lbl_test.setText("(no executor available)")
            return
        # Temporarily register + dry-run so the user can validate the action.
        from core import MacroExecutor
        dry = MacroExecutor(dry_run=True, logger=lambda m: self.lbl_test.setText(m))
        dry.macros = {mid: defn}
        ok = dry.execute(mid)
        self.lbl_test.setText(("valid: " if ok else "invalid: ") + self.lbl_test.text())

    def _on_accept(self):
        mid, defn = self._build_definition()
        if not defn:
            self.lbl_test.setText("A name and value are required.")
            return
        self._result = (mid, defn)
        self.accept()

    def result_macro(self):
        return self._result

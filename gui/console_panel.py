import datetime
from PyQt6.QtWidgets import QGroupBox, QVBoxLayout, QTextEdit
import config

class ConsolePanel(QGroupBox):
    def __init__(self):
        super().__init__("Event Console")
        self.layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.layout.addWidget(self.text_edit)
        self.log_lines = []

    def log(self, message: str, level: str = 'info'):
        color_map = {
            'info': 'white',
            'success': config.COLORS['secondary'],
            'warning': config.COLORS['warning'],
            'error': config.COLORS['error'],
            'system': config.COLORS['primary'],
            'detect': config.COLORS['accent_cyan']
        }
        color = color_map.get(level, 'white')
        timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
        html = f'<span style="color: {config.COLORS["text_dim"]};">{timestamp}</span> <span style="color: {color};">{message}</span>'
        
        self.log_lines.append(html)
        if len(self.log_lines) > config.CONSOLE_MAX_LINES:
            self.log_lines.pop(0)
            
        self.text_edit.setHtml("<br>".join(self.log_lines))
        
        # Scroll to bottom
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear(self):
        self.log_lines.clear()
        self.text_edit.clear()

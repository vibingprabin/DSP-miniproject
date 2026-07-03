"""
config.py — Application-wide constants for the Acoustic HCI / STEC v2 system.

All tunable DSP, GUI, and calibration parameters live here so they can be
adjusted without touching the core logic.
"""

# ────────────────────────────────────────────
# Audio
# ────────────────────────────────────────────
SAMPLE_RATE: int = 44100
CHUNK_SIZE: int = 1024
CHANNELS: int = 1
AUDIO_DTYPE: str = "float32"

# ────────────────────────────────────────────
# STEC — Spectro-Temporal Envelope Correlation
# ────────────────────────────────────────────
STEC_NUM_BANDS: int = 12                # 12 log-spaced bands, 500 Hz – 12 kHz
STEC_LOW_FREQ: float = 500.0
STEC_HIGH_FREQ: float = 12000.0
STEC_FILTER_ORDER: int = 4              # 4th-order Butterworth (SOS)

STEC_LP_CUTOFF: float = 50.0            # envelope extraction lowpass (Hz)
STEC_PRE_MS: float = 30.0               # pre-onset window (ms)
STEC_POST_MS: float = 120.0             # post-onset window (ms)
STEC_TIME_FRAMES: int = 64              # resampled envelope length
STEC_LOG_COMPRESS: float = 30.0         # gain before log1p — perceptual compression
STEC_USE_DELTAS: bool = True            # append temporal delta rows

# ────────────────────────────────────────────
# Onset detection  (spectral flux ODF)
# ────────────────────────────────────────────
ONSET_FRAME_SIZE: int = 512             # STFT frame (≈12 ms @ 44.1 kHz)
ONSET_HOP: int = 256                    # STFT hop
ONSET_HFC_WEIGHT: bool = True           # weight bins by frequency
ONSET_MEDIAN_FRAMES: int = 11           # neighbourhood for adaptive threshold
ONSET_THRESH_MULT: float = 2.5          # k * local_std
ONSET_MIN_FLUX: float = 1e-4            # floor threshold
ONSET_MIN_INTERVAL_MS: float = 80.0     # minimum gap between onsets (ms)
MIN_ONSET_ENERGY: float = 1e-6          # absolute energy floor

# ────────────────────────────────────────────
# Template matching / gesture acceptance
# ────────────────────────────────────────────
SIMILARITY_THRESHOLD: float = 0.75      # minimum cosine similarity
REJECT_MARGIN: float = 0.05             # must beat runner-up by this margin

# Guarded adaptive learning
ADAPTIVE_LEARNING_THRESHOLD: float = 0.85
ADAPTIVE_MIN_MARGIN: float = 0.10
ADAPTIVE_LEARNING_RATE: float = 0.15    # EMA blend factor

MAX_EXEMPLARS: int = 12                 # max exemplars per gesture

# ────────────────────────────────────────────
# Event detection / state machine
# ────────────────────────────────────────────
COOLDOWN_MS: int = 800                  # debounce between accepted gestures (ms)
ARMED_TIMEOUT_S: float = 3.0            # state-machine armed timeout (s)

# ────────────────────────────────────────────
# Calibration
# ────────────────────────────────────────────
CALIBRATION_GESTURE_DURATION: float = 3.0  # seconds per recording

# ────────────────────────────────────────────
# Macros
# ────────────────────────────────────────────
USER_MACRO_STORE: str = "user_macros.json"

MACRO_DEFINITIONS: dict = {
    "TAB_SWITCH": {
        "label": "Tab Switch", "type": "hotkey", "keys": ["alt", "tab"],
    },
    "PLAY_PAUSE": {
        "label": "Play / Pause", "type": "press", "keys": ["playpause"],
    },
    "VOLUME_MUTE": {
        "label": "Volume Mute", "type": "press", "keys": ["volumemute"],
    },
    "COPY": {
        "label": "Copy", "type": "hotkey", "keys": ["ctrl", "c"],
    },
    "PASTE": {
        "label": "Paste", "type": "hotkey", "keys": ["ctrl", "v"],
    },
    "UNDO": {
        "label": "Undo", "type": "hotkey", "keys": ["ctrl", "z"],
    },
    "ENTER": {
        "label": "Enter", "type": "press", "keys": ["enter"],
    },
    "ESC": {
        "label": "Escape", "type": "press", "keys": ["escape"],
    },
    "OPEN_CHATGPT": {
        "label": "Open ChatGPT", "type": "url",
        "value": "https://chatgpt.com",
    },
    "OPEN_DOWNLOADS": {
        "label": "Open Downloads", "type": "open", "value": "~/Downloads",
    },
}

ACTIVE_MAPPINGS: list = []   # [(("Snap",), "TAB_SWITCH"), ...]

# ────────────────────────────────────────────
# GUI — colours (GitHub Dark-inspired)
# ────────────────────────────────────────────
COLORS: dict = {
    "background":  "#0d1117",
    "primary":     "#58a6ff",
    "secondary":   "#3fb950",
    "warning":     "#d29922",
    "error":       "#f85149",
    "accent_cyan": "#79c0ff",
    "border":      "#30363d",
    "text_dim":    "#8b949e",
}

FFT_SIZE: int = 2048
FFT_DISPLAY_MAX_FREQ: float = 8000.0
TIME_DOMAIN_DISPLAY_S: float = 3.0
GUI_UPDATE_INTERVAL_MS: int = 50
CONSOLE_MAX_LINES: int = 2000

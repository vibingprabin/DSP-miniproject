import pyqtgraph as pg
from PyQt6.QtWidgets import QWidget, QGridLayout
import numpy as np
import config

pg.setConfigOptions(antialias=True)


class VisualizerPanel(QWidget):
    """
    2x2 dashboard of the live DSP chain:
        ┌───────────────────────┬───────────────────────┐
        │ time-domain waveform   │ onset detection func   │
        ├───────────────────────┼───────────────────────┤
        │ FFT + STEC filterbank  │ last STEC fingerprint  │
        └───────────────────────┴───────────────────────┘
    """

    def __init__(self):
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        bg = config.COLORS["background"]
        axis = config.COLORS["border"]

        # ── Time domain ──
        self.td_plot = pg.PlotWidget(title="Time-Domain Waveform")
        self.td_plot.setBackground(bg)
        self.td_plot.getAxis("bottom").setPen(axis)
        self.td_plot.getAxis("left").setPen(axis)
        self.td_plot.setYRange(-1, 1)
        self.td_data = np.zeros(int(config.SAMPLE_RATE * config.TIME_DOMAIN_DISPLAY_S))
        self.td_curve = self.td_plot.plot(
            self.td_data, pen=pg.mkPen(config.COLORS["secondary"], width=1.5)
        )
        layout.addWidget(self.td_plot, 0, 0)

        # ── Onset detection function ──
        self.odf_plot = pg.PlotWidget(title="Onset Detection Function (Spectral Flux)")
        self.odf_plot.setBackground(bg)
        self.odf_plot.getAxis("bottom").setPen(axis)
        self.odf_plot.getAxis("left").setPen(axis)
        self.odf_curve = self.odf_plot.plot(
            pen=pg.mkPen(config.COLORS["accent_cyan"], width=1.8),
            fillLevel=0, brush=pg.mkBrush(config.COLORS["accent_cyan"] + "40"),
        )
        layout.addWidget(self.odf_plot, 0, 1)

        # ── FFT spectrum + filterbank ──
        self.fft_plot = pg.PlotWidget(title="Spectrum & STEC Filterbank")
        self.fft_plot.setBackground(bg)
        self.fft_plot.getAxis("bottom").setPen(axis)
        self.fft_plot.getAxis("left").setPen(axis)
        self.fft_plot.setXRange(0, config.FFT_DISPLAY_MAX_FREQ)
        edges = np.logspace(
            np.log10(config.STEC_LOW_FREQ), np.log10(config.STEC_HIGH_FREQ),
            config.STEC_NUM_BANDS + 1,
        )
        for i in range(config.STEC_NUM_BANDS):
            region = pg.LinearRegionItem(
                values=[edges[i], edges[i + 1]], movable=False,
                brush=pg.mkBrush(config.COLORS["error"] + "12"),
            )
            self.fft_plot.addItem(region)
        self.fft_curve = self.fft_plot.plot(
            pen=pg.mkPen(config.COLORS["primary"], width=1.5),
            fillLevel=-100, brush=pg.mkBrush(config.COLORS["primary"] + "40"),
        )
        layout.addWidget(self.fft_plot, 1, 0)

        # ── Last STEC fingerprint (heatmap) ──
        self.profile_plot = pg.PlotWidget(title="Last STEC Fingerprint — (awaiting gesture)")
        self.profile_plot.setBackground(bg)
        self.profile_plot.getAxis("bottom").setPen(axis)
        self.profile_plot.getAxis("left").setPen(axis)
        self.profile_plot.setLabel("left", "band")
        self.profile_plot.setLabel("bottom", "time frame")
        self.profile_img = pg.ImageItem()
        self.profile_img.setLookupTable(_magma_lut())
        self.profile_plot.addItem(self.profile_img)
        self.profile_plot.invertY(False)
        layout.addWidget(self.profile_plot, 1, 1)

    def update_waveform(self, chunk: np.ndarray):
        n = len(chunk)
        if n == 0:
            return
        self.td_data[:-n] = self.td_data[n:]
        self.td_data[-n:] = chunk
        self.td_curve.setData(self.td_data)

    def update_spectrum(self, freqs: np.ndarray, magnitudes_db: np.ndarray):
        if len(freqs) and len(magnitudes_db):
            self.fft_curve.setData(freqs, magnitudes_db)

    def update_odf(self, odf_history):
        if odf_history:
            self.odf_curve.setData(np.asarray(odf_history))

    def update_profile(self, profile_2d, title=None):
        if profile_2d is None:
            return
        img = np.asarray(profile_2d).T  # (time x band) for column-major display
        self.profile_img.setImage(img, autoLevels=True)
        if title:
            self.profile_plot.setTitle(title)


def _magma_lut():
    """A compact magma-like colormap lookup table for the heatmap."""
    stops = np.array([
        [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
        [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 253, 191],
    ], dtype=float)
    xs = np.linspace(0, 255, len(stops))
    lut = np.zeros((256, 3), dtype=np.ubyte)
    for c in range(3):
        lut[:, c] = np.interp(np.arange(256), xs, stops[:, c]).astype(np.ubyte)
    return lut


class ZPlanePlot(pg.PlotWidget):
    def __init__(self):
        super().__init__(title="Z-Plane (STEC Band 1)")
        self.setBackground(config.COLORS["background"])
        self.getAxis("bottom").setPen(config.COLORS["border"])
        self.getAxis("left").setPen(config.COLORS["border"])
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setXRange(-1.5, 1.5)
        self.setYRange(-1.5, 1.5)

        theta = np.linspace(0, 2 * np.pi, 200)
        self.plot(np.cos(theta), np.sin(theta),
                  pen=pg.mkPen(config.COLORS["text_dim"], style=pg.QtCore.Qt.PenStyle.DashLine))

        self.zeros_scatter = pg.ScatterPlotItem(
            symbol="o", pen=config.COLORS["primary"], brush=config.COLORS["background"], size=12)
        self.poles_scatter = pg.ScatterPlotItem(
            symbol="x", pen=config.COLORS["error"], size=15)
        self.addItem(self.zeros_scatter)
        self.addItem(self.poles_scatter)

    def update_zpk(self, zeros: np.ndarray, poles: np.ndarray):
        self.zeros_scatter.setData(
            pos=np.column_stack([zeros.real, zeros.imag]) if len(zeros) else np.empty((0, 2)))
        self.poles_scatter.setData(
            pos=np.column_stack([poles.real, poles.imag]) if len(poles) else np.empty((0, 2)))

"""
dsp_utils.py — Shared, stateless DSP primitives for the STEC pipeline.

These helpers are used by both the offline trainer (batch onset segmentation of
a multi-snap calibration recording) and the live streaming detector, so that the
exact same signal processing defines "what a transient looks like" in training
and at run time.

Design notes / robustness rationale:
  * Filters are built as second-order sections (SOS). High-order Butterworth
    band-passes at 44.1 kHz are numerically unstable in transfer-function (b, a)
    form; SOS + `sosfiltfilt` keeps the sub-band envelopes clean and phase-free.
  * Onset detection uses HFC-weighted spectral flux — the textbook onset
    detection function (Bello et al. 2005) — which is far more selective for
    percussive transients (snaps/claps) than a raw energy threshold and is
    robust to slow background-noise level changes.
"""

import numpy as np
import scipy.signal as sig


# ──────────────────────────────────────────────
# Filter design
# ──────────────────────────────────────────────
def design_filterbank(fs, num_bands, low_freq, high_freq, order):
    """Log-spaced Butterworth band-pass filterbank, returned as a list of SOS."""
    nyq = fs / 2.0
    high_freq = min(high_freq, 0.98 * nyq)
    edges = np.logspace(np.log10(low_freq), np.log10(high_freq), num_bands + 1)
    filters = []
    for i in range(num_bands):
        wn = [edges[i] / nyq, edges[i + 1] / nyq]
        sos = sig.butter(order, wn, btype="bandpass", output="sos")
        filters.append(sos)
    return filters, edges


def design_lowpass(fs, cutoff, order=2):
    """Envelope-extraction lowpass as SOS."""
    return sig.butter(order, cutoff / (fs / 2.0), btype="low", output="sos")


# ──────────────────────────────────────────────
# Onset detection function (spectral flux)
# ──────────────────────────────────────────────
def spectral_flux(signal, frame_size, hop, hfc_weight=True):
    """
    Compute an HFC-weighted spectral-flux onset detection function (ODF).

    Returns
    -------
    odf : np.ndarray            # one value per STFT frame, >= 0
    frame_centers : np.ndarray  # sample index at the centre of each frame
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n < frame_size:
        return np.zeros(0), np.zeros(0, dtype=int)

    window = np.hanning(frame_size)
    n_frames = 1 + (n - frame_size) // hop
    freqs = np.fft.rfftfreq(frame_size)  # normalized 0..0.5
    # High-frequency emphasis: transients dump energy into the upper bands.
    # Soft weighting keeps low-frequency transients (e.g. claps) detectable
    # instead of biasing entirely toward snaps/clicks.
    weight = (0.3 + 0.7 * freqs / freqs.max()) if hfc_weight else np.ones_like(freqs)

    # Vectorised framing (stride tricks) + batched rFFT along the frame axis.
    starts = np.arange(n_frames) * hop
    idx = starts[:, None] + np.arange(frame_size)[None, :]
    frames = signal[idx] * window                       # (n_frames, frame_size)
    mag = np.abs(np.fft.rfft(frames, axis=1)) * weight   # (n_frames, n_bins)

    diff = np.diff(mag, axis=0)
    diff[diff < 0] = 0.0                                  # half-wave rectify
    odf = np.zeros(n_frames)
    odf[1:] = diff.sum(axis=1)
    odf /= (frame_size * 0.5)                             # frame-size invariant

    centers = starts + frame_size // 2
    return odf, centers


def adaptive_peak_pick(odf, median_frames, thresh_mult, min_flux, min_gap_frames):
    """
    Pick onset frames from an ODF using a *scale-relative* adaptive threshold
    (local mean + k·std) plus a local-maximum + refractory constraint.

    Using the local standard deviation instead of an absolute additive bias
    keeps the detector volume-invariant: in near-silence std→0 so the threshold
    collapses to `min_flux`, while during a transient the flux spike towers over
    `mean + k·std` regardless of the absolute loudness.

    Returns a list of frame indices that qualify as onsets.
    """
    n = len(odf)
    if n == 0:
        return []

    half = max(1, median_frames // 2)
    onsets = []
    last = -min_gap_frames - 1

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local = odf[lo:hi]
        thr = max(min_flux, np.mean(local) + thresh_mult * np.std(local))

        # Strict local maximum over a small neighbourhood.
        a = max(0, i - 1)
        b = min(n, i + 2)
        if odf[i] >= thr and odf[i] >= np.max(odf[a:b]) and (i - last) >= min_gap_frames:
            onsets.append(i)
            last = i

    return onsets

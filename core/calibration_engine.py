"""
calibration_engine.py — Spectro-Temporal Envelope Correlation (STEC) v2.

Extracts a volume-invariant 2D structural "fingerprint" of an acoustic transient
and stores it as a *multi-exemplar* template so classification is robust to the
natural variation between repetitions of the same gesture.

Feature pipeline (per detected transient):
    onset-aligned window
      ─▶ peak-normalise (loudness invariance)
      ─▶ SOS log-spaced band-pass filterbank
      ─▶ full-wave rectify
      ─▶ envelope lowpass (zero-phase)
      ─▶ log compression  log1p(gain * env)   (perceptual dynamic range)
      ─▶ resample to a fixed number of time frames
      ─▶ optional temporal deltas (attack/decay dynamics)
      ─▶ global L2 normalise  (bounded cosine similarity)
"""

import numpy as np
import scipy.signal as sig

import config
from .dsp_utils import (
    design_filterbank,
    design_lowpass,
    spectral_flux,
    adaptive_peak_pick,
)


class SpectroTemporalProfiler:
    def __init__(self, sample_rate=config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.num_bands = config.STEC_NUM_BANDS

        self.filters, self.band_edges = design_filterbank(
            sample_rate,
            self.num_bands,
            config.STEC_LOW_FREQ,
            config.STEC_HIGH_FREQ,
            config.STEC_FILTER_ORDER,
        )
        self.lp_sos = design_lowpass(sample_rate, config.STEC_LP_CUTOFF)

        self.pre_samples = int(config.STEC_PRE_MS / 1000.0 * sample_rate)
        self.post_samples = int(config.STEC_POST_MS / 1000.0 * sample_rate)
        self.win_len = self.pre_samples + self.post_samples

        # gesture_name -> {"exemplars": [profile, ...], "mean": profile}
        self.templates = {}

    # ────────────────────────────────────────────
    # Feature extraction
    # ────────────────────────────────────────────
    def profile_from_window(self, window: np.ndarray) -> np.ndarray:
        """
        Build a normalised STEC profile from an already onset-aligned window.
        Returns a 2D array (bands[*2 with deltas] x STEC_TIME_FRAMES).
        """
        window = np.asarray(window, dtype=np.float64).flatten()
        if len(window) < self.win_len:
            window = np.pad(window, (0, self.win_len - len(window)))
        else:
            window = window[: self.win_len]

        # Sub-band envelopes.
        env_rows = []
        for sos in self.filters:
            subband = sig.sosfiltfilt(sos, window)
            envelope = sig.sosfiltfilt(self.lp_sos, np.abs(subband))
            envelope = np.clip(envelope, 0.0, None)
            # Per-band noise-floor subtraction: remove the flat pedestal so the
            # (later) log stage does not amplify background noise in bands that
            # carry no transient energy — this is what makes claps stop leaking
            # into the high-band "click" template under noise.
            floor = np.percentile(envelope, 15.0)
            envelope = np.clip(envelope - floor, 0.0, None)
            env_rows.append(sig.resample(envelope, config.STEC_TIME_FRAMES))

        profile = np.clip(np.asarray(env_rows), 0.0, None)  # (bands x T)

        # Loudness invariance that is robust to single-sample noise spikes:
        # scale by the global envelope maximum (a smoothed quantity), not the
        # raw waveform peak.
        gmax = profile.max()
        if gmax > 1e-9:
            profile = profile / gmax

        # Perceptual dynamic-range compression on the scale-normalised envelope.
        profile = np.log1p(config.STEC_LOG_COMPRESS * profile)

        if config.STEC_USE_DELTAS:
            deltas = np.diff(profile, axis=1, prepend=profile[:, :1])
            profile = np.vstack([profile, deltas])  # (2*bands x T)

        norm = np.linalg.norm(profile)
        if norm > 1e-10:
            profile = profile / norm
        return profile

    def _find_onset_windows(self, audio: np.ndarray):
        """Segment every strong transient in a recording into aligned windows."""
        audio = np.asarray(audio, dtype=np.float64).flatten()
        odf, centers = spectral_flux(
            audio, config.ONSET_FRAME_SIZE, config.ONSET_HOP, config.ONSET_HFC_WEIGHT
        )
        min_gap = max(1, int(
            (config.ONSET_MIN_INTERVAL_MS / 1000.0 * self.sample_rate) / config.ONSET_HOP
        ))
        onset_frames = adaptive_peak_pick(
            odf,
            config.ONSET_MEDIAN_FRAMES,
            config.ONSET_THRESH_MULT,
            config.ONSET_MIN_FLUX,
            min_gap,
        )

        windows = []
        for f in onset_frames:
            approx = int(centers[f])
            # Refine to the true local energy peak near the flux onset.
            lo = max(0, approx - self.pre_samples)
            hi = min(len(audio), approx + self.post_samples)
            if hi - lo < 4:
                continue
            local_peak = lo + int(np.argmax(np.abs(audio[lo:hi])))
            start = max(0, local_peak - self.pre_samples)
            end = min(len(audio), local_peak + self.post_samples)
            win = audio[start:end]
            if np.max(win ** 2) >= config.MIN_ONSET_ENERGY:
                windows.append(win)
        return windows

    # ────────────────────────────────────────────
    # Training
    # ────────────────────────────────────────────
    def train_gesture(self, audio: np.ndarray, gesture_name: str):
        """
        Validate the recording, segment every transient it contains, and store
        each as an exemplar. Multiple snaps in one 3 s recording therefore yield
        multiple exemplars, which makes the template robust to variation.
        """
        audio = np.asarray(audio, dtype=np.float64).flatten()

        windows = self._find_onset_windows(audio)
        if not windows:
            raise ValueError(
                "No distinct transient detected. Snap/clap clearly and close to "
                "the mic (a sharp, explosive sound works best)."
            )

        exemplars = [self.profile_from_window(w) for w in windows]
        if len(exemplars) > config.MAX_EXEMPLARS:
            exemplars = exemplars[-config.MAX_EXEMPLARS:]

        self.templates[gesture_name] = {
            "exemplars": exemplars,
            "mean": self._mean_profile(exemplars),
        }
        return len(exemplars)

    def add_exemplar(self, gesture_name: str, window: np.ndarray):
        """Add a single onset-aligned window as an extra exemplar."""
        profile = self.profile_from_window(window)
        entry = self.templates.setdefault(gesture_name, {"exemplars": [], "mean": None})
        entry["exemplars"].append(profile)
        if len(entry["exemplars"]) > config.MAX_EXEMPLARS:
            entry["exemplars"] = entry["exemplars"][-config.MAX_EXEMPLARS:]
        entry["mean"] = self._mean_profile(entry["exemplars"])

    @staticmethod
    def _mean_profile(exemplars):
        mean = np.mean(np.stack(exemplars, axis=0), axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 1e-10 else mean

    # ────────────────────────────────────────────
    # Matching
    # ────────────────────────────────────────────
    def match(self, profile: np.ndarray):
        """
        Score a live profile against all gestures.

        Score for a gesture = max cosine similarity over {its exemplars, mean}
        (nearest-exemplar rule → robust to intra-gesture variation).

        Returns
        -------
        best_name, best_score, second_score, best_exemplar_index
        """
        flat = profile.flatten()
        ranking = []
        best_idx_map = {}
        for name, entry in self.templates.items():
            candidates = list(entry["exemplars"])
            if entry.get("mean") is not None:
                candidates = candidates + [entry["mean"]]
            scores = [float(np.dot(c.flatten(), flat)) for c in candidates]
            gi = int(np.argmax(scores))
            ranking.append((name, scores[gi]))
            best_idx_map[name] = gi if gi < len(entry["exemplars"]) else -1

        if not ranking:
            return None, 0.0, 0.0, -1

        ranking.sort(key=lambda x: x[1], reverse=True)
        best_name, best_score = ranking[0]
        second_score = ranking[1][1] if len(ranking) > 1 else 0.0
        return best_name, best_score, second_score, best_idx_map[best_name]

    def adapt_template(self, gesture_name, live_profile, exemplar_index, learning_rate):
        """
        EMA-blend a confident live hit into the exemplar it matched, then
        renormalise. Adapting the nearest exemplar (instead of a single global
        template) preserves the diversity captured during training.
        """
        entry = self.templates.get(gesture_name)
        if not entry or not entry["exemplars"]:
            return
        idx = exemplar_index if 0 <= exemplar_index < len(entry["exemplars"]) else 0
        old = entry["exemplars"][idx]
        blended = (1.0 - learning_rate) * old + learning_rate * live_profile
        norm = np.linalg.norm(blended)
        if norm > 1e-10:
            entry["exemplars"][idx] = blended / norm
            entry["mean"] = self._mean_profile(entry["exemplars"])

    def delete_gesture(self, gesture_name: str):
        self.templates.pop(gesture_name, None)

    @property
    def is_calibrated(self) -> bool:
        return len(self.templates) > 0

    def get_display_profile(self, gesture_name: str):
        """Return the mean band-envelope profile (bands x T) for GUI heatmaps."""
        entry = self.templates.get(gesture_name)
        if not entry or entry.get("mean") is None:
            return None
        return entry["mean"][: self.num_bands]

    # ────────────────────────────────────────────
    # GUI-only spectrum helper (unchanged behaviour)
    # ────────────────────────────────────────────
    def compute_fft(self, chunk: np.ndarray):
        if len(chunk) == 0:
            return np.array([]), np.array([])
        fft_size = config.FFT_SIZE
        pad_len = max(0, fft_size - len(chunk))
        padded = np.pad(chunk, (0, pad_len)) if pad_len > 0 else chunk[:fft_size]
        windowed = padded * np.hanning(fft_size)
        mags = np.abs(np.fft.rfft(windowed))
        mags_db = 20 * np.log10(mags + 1e-10)
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / self.sample_rate)
        return freqs, mags_db

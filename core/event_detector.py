"""
event_detector.py — Streaming transient detector + gesture state machine.

Robustness upgrades over the original energy-threshold design:

  1. Continuous RING BUFFER: the STEC window is always cut from a buffer that
     spans chunk boundaries, so a snap that lands between two audio callbacks is
     never truncated or missed.
  2. Spectral-flux ONSET detection with an adaptive (median/mean) threshold and
     peak-picking, instead of "peak energy > k * rolling average". This tracks a
     drifting noise floor and fires selectively on percussive transients.
  3. Discriminative REJECTION: a gesture is only accepted when the best template
     score clears the similarity threshold *and* beats the runner-up by a margin,
     which sharply cuts false positives from unknown sounds.
  4. GUARDED adaptation: templates only self-update on high-confidence,
     unambiguous hits, preventing drift/poisoning from noise.
"""

import collections
import time

import numpy as np

import config
from .dsp_utils import spectral_flux, adaptive_peak_pick


class EventDetector:
    def __init__(self, profiler, cooldown_ms=config.COOLDOWN_MS,
                 armed_timeout=config.ARMED_TIMEOUT_S):
        self.profiler = profiler
        self.cooldown_ms = cooldown_ms
        self.armed_timeout = armed_timeout

        # Ring buffer big enough for the adaptive threshold history + full window.
        self.buf_len = int(0.75 * profiler.sample_rate)
        self.buffer = np.zeros(self.buf_len, dtype=np.float64)
        self.total_samples = 0                 # samples ever ingested
        self.last_onset_abs = -(10 ** 12)      # absolute sample idx of last onset

        self.min_gap_samples = int(
            config.ONSET_MIN_INTERVAL_MS / 1000.0 * profiler.sample_rate
        )
        self.min_gap_frames = max(1, self.min_gap_samples // config.ONSET_HOP)

        # Debounce between *accepted* gestures, measured in audio samples (not
        # wall-clock) so behaviour is identical live and in offline replay.
        self.cooldown_samples = int(cooldown_ms / 1000.0 * profiler.sample_rate)
        self.last_accept_abs = -(10 ** 12)

        # State machine
        self.state = "IDLE"
        self.state_enter_time = 0.0
        self.current_sequence = []

        # For GUI: recent onset-detection-function values + last live analysis.
        self.odf_history = collections.deque(maxlen=200)
        self.last_odf = 0.0
        self.last_threshold = 0.0
        self.last_profile = None          # 2D band-envelope of the last transient
        self.last_match = (None, 0.0)     # (gesture_name, similarity)

    # ────────────────────────────────────────────
    def _push(self, samples):
        n = len(samples)
        if n >= self.buf_len:
            self.buffer[:] = samples[-self.buf_len:]
        else:
            self.buffer[:-n] = self.buffer[n:]
            self.buffer[-n:] = samples
        self.total_samples += n

    def _buf_start_abs(self):
        return self.total_samples - self.buf_len

    # ────────────────────────────────────────────
    def process_chunk(self, raw_chunk: np.ndarray) -> list:
        events = []
        now = time.time()

        # State-machine timeouts.
        if self.state == "ARMED" and (now - self.state_enter_time) > self.armed_timeout:
            self.state = "IDLE"
            self.current_sequence = []
            events.append({"gesture": "timeout", "energy": 0, "time": now,
                           "state_change": "IDLE"})
        if self.state == "EXECUTE":
            self.state = "IDLE"
            self.current_sequence = []

        chunk = np.asarray(raw_chunk, dtype=np.float64).flatten()
        if len(chunk) == 0:
            return events
        self._push(chunk)

        # Onset detection over the current buffer.
        odf, centers = spectral_flux(
            self.buffer, config.ONSET_FRAME_SIZE, config.ONSET_HOP,
            config.ONSET_HFC_WEIGHT,
        )
        if len(odf) == 0:
            return events

        self.last_odf = float(odf[-1])
        self.odf_history.append(self.last_odf)

        onset_frames = adaptive_peak_pick(
            odf, config.ONSET_MEDIAN_FRAMES, config.ONSET_THRESH_MULT,
            config.ONSET_MIN_FLUX, self.min_gap_frames,
        )
        if not onset_frames:
            return events

        buf_start = self._buf_start_abs()
        post = self.profiler.post_samples
        pre = self.profiler.pre_samples

        for f in onset_frames:
            c_rel = int(centers[f])
            c_abs = buf_start + c_rel

            # Ignore already-processed onsets and enforce refractory spacing.
            if c_abs <= self.last_onset_abs + self.min_gap_samples:
                continue
            # Require the full post-onset window to be present in the buffer.
            if (self.total_samples - c_abs) < post:
                continue

            # Refine to the local energy peak, then cut the aligned window.
            lo = max(0, c_rel - pre)
            hi = min(self.buf_len, c_rel + post)
            if hi - lo < 4:
                continue
            local_peak = lo + int(np.argmax(np.abs(self.buffer[lo:hi])))
            w_start = max(0, local_peak - pre)
            w_end = min(self.buf_len, local_peak + post)
            window = self.buffer[w_start:w_end]

            peak_energy = float(np.max(window ** 2)) if len(window) else 0.0
            if peak_energy < config.MIN_ONSET_ENERGY:
                continue

            self.last_onset_abs = c_abs

            events.extend(self._classify(window, peak_energy, c_abs, now))

        return events

    # ────────────────────────────────────────────
    def _classify(self, window, peak_energy, c_abs, now):
        events = []
        if not self.profiler.is_calibrated:
            return events

        profile = self.profiler.profile_from_window(window)
        best_name, best_score, second_score, ex_idx = self.profiler.match(profile)
        margin = best_score - second_score

        # Expose the raw band-envelope (pre-delta rows) for GUI heatmaps.
        self.last_profile = profile[: self.profiler.num_bands]
        self.last_match = (best_name, best_score)

        if best_name is None:
            return events

        accepted = (best_score >= config.SIMILARITY_THRESHOLD
                    and margin >= config.REJECT_MARGIN)

        # Guarded adaptive learning.
        if (accepted and best_score >= config.ADAPTIVE_LEARNING_THRESHOLD
                and margin >= config.ADAPTIVE_MIN_MARGIN):
            self.profiler.adapt_template(
                best_name, profile, ex_idx, config.ADAPTIVE_LEARNING_RATE
            )

        if not accepted:
            events.append({
                "gesture": f"Unrecognized (Best: {best_name})",
                "energy": peak_energy, "similarity": best_score,
                "margin": margin, "state_change": None,
            })
            return events

        # Debounce repeated firings of the same physical event (audio-time).
        if (c_abs - self.last_accept_abs) < self.cooldown_samples:
            return events
        self.last_accept_abs = c_abs

        self.current_sequence.append(best_name)
        matched_macro = self._evaluate_sequence()

        if matched_macro:
            self.state = "EXECUTE"
            state_change = "EXECUTE"
        else:
            self.state = "ARMED"
            state_change = "ARMED"
        self.state_enter_time = now

        event = {"gesture": best_name, "energy": peak_energy,
                 "similarity": best_score, "margin": margin, "time": now,
                 "state_change": state_change}
        if matched_macro:
            event["macro"] = matched_macro
        events.append(event)
        return events

    def _evaluate_sequence(self):
        for seq, macro_id in config.ACTIVE_MAPPINGS:
            if tuple(self.current_sequence) == seq:
                return macro_id
        # Prune the sequence if it can no longer be a prefix of any mapping.
        if not any(tuple(self.current_sequence) == tuple(seq[:len(self.current_sequence)])
                   for seq, _ in config.ACTIVE_MAPPINGS):
            self.current_sequence = self.current_sequence[-1:]
        return None

    def get_state(self) -> str:
        return self.state

    def reset(self):
        self.state = "IDLE"
        self.current_sequence = []
        self.buffer[:] = 0.0
        self.total_samples = 0
        self.last_onset_abs = -(10 ** 12)
        self.last_accept_abs = -(10 ** 12)
        self.odf_history.clear()
        self.last_odf = 0.0
        self.last_threshold = 0.0

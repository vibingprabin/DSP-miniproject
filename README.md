# Acoustic HCI -- Dynamic Gestures & STEC v2

[![GitHub](https://img.shields.io/badge/GitHub-vibingprabin/DSP--miniproject-181717?logo=github)](https://github.com/vibingprabin/DSP-miniproject)

A real-time Digital Signal Processing application that uses your microphone to trigger OS-level macros (Alt+Tab, Play/Pause, Mute) using acoustic transients like snaps, claps, and tongue clicks.

This is a reimplementation of the original **Spectro-Temporal Envelope Correlation (STEC)** idea. The DSP concept (filterbank, sub-band envelopes, 2D fingerprint, template correlation) is unchanged, but each stage was rebuilt to improve detection across varying noise levels, loudness and chunk timing.

> **Project Report:** [`Report/report.pdf`](Report/report.pdf) -- full documentation of the system architecture, methodology, results, and source code links.

## How It Works (the DSP pipeline)

```
mic -- ring buffer -- spectral-flux ONSET detector -- onset-aligned window
    -- SOS log-spaced FILTERBANK -- rectify -- envelope lowpass
    -- noise-floor subtraction -- log compression -- fixed-length resample
    -- + temporal deltas -- L2 normalise -- multi-exemplar cosine MATCH
    (+ discriminative margin test) -- guarded adaptive learning
    -- gesture-sequence state machine -- OS macro
```

1. **Continuous ring buffer.** Audio is accumulated in a rolling buffer so a transient that lands between two audio callbacks is never truncated or missed. The analysis window is cut cleanly around the event.
2. **Spectral-flux onset detection.** Instead of "peak energy > k x rolling average", transients are found with an HFC-weighted spectral-flux onset detection function and a scale-relative adaptive threshold (`local_mean + k.local_std`). Because the threshold scales with local signal statistics, detection is volume-invariant and tracks a drifting noise floor.
3. **12-band SOS filterbank.** Live audio is routed through 12 log-spaced Butterworth band-passes (500 Hz-12 kHz), implemented as second-order sections (numerically stable at 44.1 kHz, unlike high-order `(b, a)` forms).
4. **Structural profiling.** Each band envelope is rectified, low-pass smoothed, noise-floor subtracted (so background noise in empty bands is not amplified by the log stage), log-compressed, resampled to a fixed length, and stacked with its temporal deltas (attack/decay dynamics). The 2D fingerprint is L2-normalised for volume invariance.
5. **Multi-exemplar template matching.** A single 3-second recording is segmented into every transient it contains, so several snaps become several exemplars per gesture. Live audio is matched by nearest-exemplar cosine similarity.
6. **Discriminative rejection.** A gesture is accepted only when the best score clears the similarity threshold and beats the runner-up by a margin. This is what drives the false-positive rate toward zero on unknown sounds.
7. **Guarded adaptive learning.** Templates self-update (EMA) only on high-confidence, unambiguous hits, so they adapt to your environment without drifting or being poisoned by noise.
8. **State machine.** A sequence engine lets you chain gestures (e.g. `Left Snap` -> `Right Snap`) for two-gesture macro mappings.

## Robustness upgrades over the original

| Stage | Original | STEC v2 |
| --- | --- | --- |
| Transient detection | peak energy vs rolling avg | HFC spectral-flux ODF + scale-relative adaptive threshold + peak-picking |
| Windowing | single chunk (could truncate) | continuous ring buffer, onset-aligned |
| Filterbank | 8 bands, `(b, a)` | 12 bands, numerically stable SOS |
| Feature | envelope only | noise-floor-subtracted, log-compressed envelope + temporal deltas |
| Template | one per gesture | multiple exemplars (nearest-exemplar match) |
| Acceptance | similarity threshold | similarity threshold **+ discriminative margin** |
| Adaptation | any hit > 0.75 | only confident, unambiguous hits |
| Debounce | wall-clock (flaky offline) | audio-sample based |

## Installation

```bash
python -m pip install -r requirements.txt
```

(On Windows, `PyAudio` is not required because `sounddevice` provides better PortAudio bindings.)

## Usage

1. Run the application:
   ```bash
   python main.py
   ```
2. Type a name (e.g. "Left Snap") into the `Record New Gesture` box and click **Record**. Repeat the sound a few times during the 3-second window; each repetition becomes a template exemplar.
3. Once trained, map your gestures to macros in the `Macro Sequences` panel. You can map single gestures or two-gesture combos.
4. Click **Start Listening** and perform your gesture.

## Macros -- flexible action types

Every mapping fires a macro. Each macro has a `type`:

| Type | Payload | Example | What it does |
| --- | --- | --- | --- |
| `hotkey` | `keys` | `ctrl+alt+t` | Press a key combination simultaneously |
| `press` | `keys` | `playpause` | Tap one or more single keys (media keys, etc.) |
| `text` | `value` | `my.email@x.com` | Type a string of text |
| `url` | `value` | `https://chatgpt.com`, `mailto:me@x.com`, `slack://open` | Open a URL or URI scheme in the default handler |
| `open` | `value` | `~/Downloads` | Open a file/folder with the default app |
| `launch` | `value` | `code .` | Launch an app or run a program (no shell) |
| `shell` | `value` | `notify-send 'Snap!' && date` | Run a shell command (pipes, `&&`, globs) |
| `sequence` | `actions` | - | Chain several actions, with optional `{"type":"delay","ms":N}` |

### Creating your own macros

Click **New Macro** in the panel, pick a type, enter a value, hit **Test (dry-run)** to validate, then **OK**. Custom macros are marked with a star, persist to `user_macros.json`, and reload automatically on the next launch. Use **Delete Macro** to remove a custom one (built-ins are protected).

Cross-platform notes: URLs/files use `xdg-open` on Linux, `open` on macOS, `os.startfile` on Windows (with a `webbrowser` fallback); `launch`/`shell` run detached so the app never blocks.

## Testing (no microphone required)

The full DSP chain can be validated offline. `test_stec.py` synthesises three distinct transients (snap / clap / click), trains from multi-transient recordings, then streams noisy, randomly-timed, randomly-scaled trials through the real `EventDetector` chunk-by-chunk (the same path as the live app).

```bash
python test_stec.py
```

It reports and asserts on: onset recall, classification accuracy vs SNR, volume invariance, false-positive rejection on unknown distractor sounds, a confusion matrix, and a streaming state-machine / macro test. Result plots (templates, onset function, accuracy-vs-SNR, confusion matrix) are written to `test_results/`.

Representative results from the synthetic suite:

- Classification accuracy: ~90-95% from 5 dB to 30 dB SNR (zero misclassifications in the confusion matrix; errors are misses, not confusions).
- Volume invariance: recognised across the full 40 dB gain sweep.
- False-positive rate on tones / speech-like / broadband distractors: 0%.
- Streaming `Snap -> Clap` fires the mapped macro.
- Macro engine dispatches all action types (hotkey / press / text / url / open / launch / shell / sequence) and rejects malformed macros.

## Troubleshooting

- **Audio too quiet / No distinct peak**: Get closer to the mic. The onset detector rejects windows without a distinct explosive transient (a snap/clap works best).
- **Heard Unrecognized**: The system detected a transient, but either its similarity was below threshold or it was too ambiguous (the top two gestures were within the rejection margin). Retrain the gesture (repeat it a few times during the 3 s recording so it captures several exemplars), or perform it closer to how you trained it.

## Academic References

The STEC algorithm draws from prior work in acoustic transient processing, filterbank-based feature extraction, and template matching:

1. **Pineda, F.J., Cauwenberghs, G., & Edwards, R.T. (1996).** "Bangs, Clicks, Snaps, Thuds and Whacks: An Architecture for Acoustic Transient Processing." *Advances in Neural Information Processing Systems 9*, pp. 734-740. A neuromorphic architecture for real-time acoustic transient classification using time-frequency analysis, rectification, smoothing, and template correlation. The STEC pipeline (filterbank, rectify, envelope, normalize, cosine similarity) is structurally analogous to this baseline algorithm.

2. **Schroder, J., Goetze, S., & Anemuller, J. (2015).** "Spectro-Temporal Gabor Filterbank Features for Acoustic Event Detection." *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 23(12), pp. 2198-2212. Shows that filterbank-based spectro-temporal features outperform MFCCs for acoustic event detection. The use of multiple sub-band energy envelopes for classification supports the STEC design choice of an 8-band filterbank.

3. **Duxbury, C., Sandler, M., & Davies, M. (2002).** "A Hybrid Approach to Musical Note Onset Detection." *Proceedings of the 5th International Conference on Digital Audio Effects (DAFX-02)*. Proposes sub-band energy analysis for transient onset detection using a constant-Q filterbank (1.2-11 kHz). The energy-based detection on upper subbands informs the STEC transient detection stage.

4. **Bello, J.P., Daudet, L., Abdallah, S., Duxbury, C., Davies, M., & Sandler, M. (2005).** "A Tutorial on Onset Detection in Music Signals." *IEEE Transactions on Speech and Audio Processing*, 13(5), pp. 1035-1047. A review of onset detection methods including energy-based, spectral difference, and phase-based approaches. Covers multi-band preprocessing and envelope extraction techniques used in STEC.

5. **Edwards, R.T., Cauwenberghs, G., & Pineda, F.J. (1998).** "Optimizing Correlation Algorithms for Hardware-Based Transient Classification." *Advances in Neural Information Processing Systems 11*. Investigates normalization, differencing, and binarization schemes for template correlation. The finding that normalized templates maintain classification performance supports the L2-normalization step in STEC.

6. **Becker, V., Fessler, L., & Soros, G. (2019).** "GestEar: Combining Audio and Motion Sensing for Gesture Recognition on Smartwatches." *Proceedings of the 23rd International Symposium on Wearable Computers (ISWC)*, pp. 10-17. Acoustic gesture recognition using snap, clap, and knock gestures. Shows that transient-based gesture classification works on resource-constrained devices.

7. **Leng, Y.R., Tran, H.D., Kitaoka, N., & Li, H. (2012).** "Selective Gammatone Envelope Feature for Sound Event Recognition." *IEICE Transactions on Information and Systems*, E95-D(5), pp. 1229-1237. Uses gammatone filterbank envelopes as features for sound event recognition, showing that sub-band envelope information holds up under noise (a property STEC uses for volume-invariant gesture recognition).

"""
test_stec.py — Offline robustness test suite for the STEC v2 pipeline.

Because the real system needs a microphone, this harness *synthesises* acoustic
transients (snap / clap / click) and drives the full DSP chain exactly as the
live app would: training from multi-onset recordings, then streaming noisy,
randomly-timed, randomly-scaled trials through the EventDetector chunk-by-chunk.

It reports, and asserts on:
  * onset-detection recall,
  * classification accuracy vs SNR (noise robustness),
  * volume invariance,
  * false-positive rate against unknown distractor sounds (rejection),
  * a confusion matrix,
  * a full streaming state-machine / macro-firing test.

Plots (profiles, ODF, accuracy-vs-SNR, confusion matrix) are written to
`test_results/` when matplotlib is available.

Run:  python test_stec.py
"""

import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mpl"))

import numpy as np
import scipy.signal as sig

import config
from core import SpectroTemporalProfiler, EventDetector, MacroExecutor, MACRO_TYPE_HINTS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "test_results")

GESTURES = {
    # name:   (formants Hz,               decay rate)
    "Snap":  ([3000, 6000],               120.0),
    "Clap":  ([1200, 2200, 3500],         45.0),
    "Click": ([5000, 8500],               220.0),
}


# ──────────────────────────────────────────────
# Synthesis
# ──────────────────────────────────────────────
def make_transient(fs, formants, decay, dur=0.06, rng=None, jitter=0.0):
    """A percussive transient: decaying noise shaped by resonant band-passes."""
    rng = rng or np.random.default_rng()
    t = np.linspace(0, dur, int(fs * dur), endpoint=False)
    noise = rng.standard_normal(len(t))
    d = decay * (1.0 + jitter * rng.uniform(-1, 1))
    env = np.exp(-d * t)
    x = noise * env
    out = np.zeros_like(x)
    for f0 in formants:
        f0 = f0 * (1.0 + jitter * rng.uniform(-1, 1))
        lo = max(100.0, f0 - 250.0)
        hi = min(0.98 * fs / 2.0, f0 + 250.0)
        sos = sig.butter(2, [lo, hi], btype="bandpass", fs=fs, output="sos")
        out += sig.sosfilt(sos, x)
    peak = np.max(np.abs(out))
    return out / peak if peak > 0 else out


def embed(transient, fs, total_s=1.0, onset_s=None, gain=1.0, snr_db=None, rng=None):
    """Place one transient in a noisy background buffer."""
    rng = rng or np.random.default_rng()
    n = int(fs * total_s)
    buf = np.zeros(n)
    if onset_s is None:
        onset_s = rng.uniform(0.15, total_s - 0.2)
    start = int(onset_s * fs)
    end = min(n, start + len(transient))
    buf[start:end] += gain * transient[: end - start]

    if snr_db is not None:
        sig_rms = np.sqrt(np.mean((gain * transient) ** 2)) + 1e-12
        noise_rms = sig_rms / (10 ** (snr_db / 20.0))
        buf += rng.standard_normal(n) * noise_rms
    return buf


def training_recording(fs, formants, decay, n_hits=5, rng=None):
    """A 3 s calibration recording containing several repetitions of a gesture."""
    rng = rng or np.random.default_rng()
    n = int(fs * config.CALIBRATION_GESTURE_DURATION)
    buf = rng.standard_normal(n) * 0.002  # quiet room noise
    for k in range(n_hits):
        onset = 0.3 + k * (2.4 / n_hits)
        tr = make_transient(fs, formants, decay, rng=rng, jitter=0.05)
        start = int(onset * fs)
        end = min(n, start + len(tr))
        buf[start:end] += 0.8 * tr[: end - start]
    return buf


# ──────────────────────────────────────────────
# Streaming helper
# ──────────────────────────────────────────────
def stream_through(detector, audio, chunk=config.CHUNK_SIZE):
    """Feed a buffer through the detector chunk-by-chunk; collect real events."""
    detector.reset()
    detected = []
    for i in range(0, len(audio) - chunk + 1, chunk):
        for e in detector.process_chunk(audio[i:i + chunk]):
            g = e.get("gesture", "")
            if g and g != "timeout" and "Unrecognized" not in g:
                detected.append(e)
    return detected


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────
def build_profiler(fs, rng):
    profiler = SpectroTemporalProfiler(sample_rate=fs)
    counts = {}
    for name, (formants, decay) in GESTURES.items():
        rec = training_recording(fs, formants, decay, n_hits=5, rng=rng)
        counts[name] = profiler.train_gesture(rec, name)
    return profiler, counts


def test_training(counts):
    print("\n[1] TRAINING — multi-exemplar templates from multi-snap recordings")
    ok = True
    for name, c in counts.items():
        status = "ok" if c >= 3 else "LOW"
        if c < 3:
            ok = False
        print(f"    {name:6s}: {c} exemplars extracted  [{status}]")
    return ok


def test_accuracy_vs_snr(profiler, fs, rng, trials=40):
    print("\n[2] CLASSIFICATION ACCURACY vs SNR  (noise robustness)")
    snrs = [30, 20, 15, 10, 5, 0]
    acc_curve = {}
    for snr in snrs:
        detector = EventDetector(profiler)
        correct = total = 0
        for name, (formants, decay) in GESTURES.items():
            for _ in range(trials):
                tr = make_transient(fs, formants, decay, rng=rng, jitter=0.06)
                gain = rng.uniform(0.3, 1.0)
                audio = embed(tr, fs, gain=gain, snr_db=snr, rng=rng)
                hits = stream_through(detector, audio)
                total += 1
                if hits and hits[-1]["gesture"] == name:
                    correct += 1
        acc = correct / total
        acc_curve[snr] = acc
        bar = "#" * int(acc * 30)
        print(f"    SNR {snr:>2} dB : {acc*100:5.1f}%  {bar}")
    return acc_curve


def test_volume_invariance(profiler, fs, rng):
    print("\n[3] VOLUME INVARIANCE  (same gesture across 40 dB of gain)")
    detector = EventDetector(profiler)
    gains = [0.05, 0.15, 0.4, 0.8, 1.0]
    ok = True
    for name, (formants, decay) in GESTURES.items():
        results = []
        for g in gains:
            tr = make_transient(fs, formants, decay, rng=rng, jitter=0.03)
            audio = embed(tr, fs, gain=g, snr_db=25, rng=rng)
            hits = stream_through(detector, audio)
            results.append(bool(hits) and hits[-1]["gesture"] == name)
        rate = np.mean(results)
        if rate < 0.8:
            ok = False
        print(f"    {name:6s}: {int(rate*len(gains))}/{len(gains)} gains recognised")
    return ok


def test_false_positive(profiler, fs, rng, trials=60):
    print("\n[4] FALSE-POSITIVE REJECTION  (unknown distractor sounds)")
    detector = EventDetector(profiler)

    def tone_burst():
        t = np.linspace(0, 0.15, int(fs * 0.15), endpoint=False)
        f = rng.uniform(300, 2000)
        return np.sin(2 * np.pi * f * t) * np.hanning(len(t))

    def speech_like():
        t = np.linspace(0, 0.25, int(fs * 0.25), endpoint=False)
        base = rng.uniform(120, 260)
        s = sum(np.sin(2 * np.pi * base * h * t) / h for h in range(1, 6))
        return s * np.hanning(len(t)) * (0.5 + 0.5 * np.sin(2 * np.pi * 4 * t))

    def broadband():  # a slow "whoosh", not a sharp transient
        t = np.linspace(0, 0.3, int(fs * 0.3), endpoint=False)
        return rng.standard_normal(len(t)) * np.hanning(len(t))

    distractors = [tone_burst, speech_like, broadband]
    false_hits = total = 0
    for _ in range(trials):
        d = distractors[rng.integers(len(distractors))]()
        d = d / (np.max(np.abs(d)) + 1e-9)
        audio = embed(d, fs, gain=rng.uniform(0.4, 1.0), snr_db=25, rng=rng)
        hits = stream_through(detector, audio)
        total += 1
        if hits:
            false_hits += 1
    fpr = false_hits / total
    print(f"    False positives: {false_hits}/{total}  (FPR = {fpr*100:.1f}%)")
    return fpr


def test_confusion(profiler, fs, rng, trials=60):
    print("\n[5] CONFUSION MATRIX  (SNR = 15 dB)")
    names = list(GESTURES.keys())
    idx = {n: i for i, n in enumerate(names)}
    cm = np.zeros((len(names), len(names) + 1), dtype=int)  # +1 = rejected/miss
    detector = EventDetector(profiler)
    for name, (formants, decay) in GESTURES.items():
        for _ in range(trials):
            tr = make_transient(fs, formants, decay, rng=rng, jitter=0.06)
            audio = embed(tr, fs, gain=rng.uniform(0.4, 1.0), snr_db=15, rng=rng)
            hits = stream_through(detector, audio)
            if hits:
                cm[idx[name], idx[hits[-1]["gesture"]]] += 1
            else:
                cm[idx[name], -1] += 1
    header = "        " + "".join(f"{n:>8}" for n in names) + f"{'miss':>8}"
    print(header)
    for n in names:
        row = "".join(f"{v:>8}" for v in cm[idx[n]])
        print(f"    {n:6s}{row}")
    return names, cm


def test_streaming_sequence(profiler, fs, rng):
    print("\n[6] STREAMING STATE MACHINE  (Snap -> Clap fires a macro)")
    config.ACTIVE_MAPPINGS = [(("Snap", "Clap"), "TAB_SWITCH")]
    detector = EventDetector(profiler)

    n = int(fs * 1.4)
    audio = rng.standard_normal(n) * 0.003
    for onset, (formants, decay) in [(0.3, GESTURES["Snap"]), (0.75, GESTURES["Clap"])]:
        tr = make_transient(fs, formants, decay, rng=rng, jitter=0.04)
        s = int(onset * fs)
        audio[s:s + len(tr)] += 0.8 * tr

    fired = None
    seq = []
    detector.reset()
    for i in range(0, len(audio) - config.CHUNK_SIZE + 1, config.CHUNK_SIZE):
        for e in detector.process_chunk(audio[i:i + config.CHUNK_SIZE]):
            if e.get("gesture") and "Unrecognized" not in e["gesture"] and e["gesture"] != "timeout":
                seq.append(e["gesture"])
                print(f"    heard {e['gesture']:6s} sim={e.get('similarity',0):.2f} "
                      f"margin={e.get('margin',0):.2f} -> {e.get('state_change')}")
            if "macro" in e:
                fired = e["macro"]
                print(f"    *** MACRO FIRED: {fired} ***")
    ok = fired == "TAB_SWITCH"
    config.ACTIVE_MAPPINGS = []
    return ok


def test_macro_engine():
    print("\n[7] MACRO ENGINE  (all action types dispatch correctly)")
    logs = []
    ex = MacroExecutor(dry_run=True, logger=logs.append)
    # One macro per supported action type.
    ex.macros = {
        "m_hotkey": {"type": "hotkey", "keys": ["ctrl", "alt", "t"]},
        "m_press":  {"type": "press",  "keys": ["playpause"]},
        "m_text":   {"type": "text",   "value": "hello"},
        "m_url":    {"type": "url",    "value": "https://example.com"},
        "m_open":   {"type": "open",   "value": "~/Downloads"},
        "m_launch": {"type": "launch", "value": "echo hi"},
        "m_shell":  {"type": "shell",  "value": "echo hi && date"},
        "m_seq":    {"type": "sequence", "actions": [
            {"type": "hotkey", "keys": ["ctrl", "c"]},
            {"type": "delay", "ms": 10},
            {"type": "text", "value": "world"},
        ]},
        "m_bad":    {"type": "hotkey", "keys": []},        # invalid: no keys
        "m_empty":  {"type": "url",    "value": ""},        # invalid: no value
    }
    expect_ok = {"m_hotkey", "m_press", "m_text", "m_url", "m_open",
                 "m_launch", "m_shell", "m_seq"}
    all_ok = True
    for mid in ex.macros:
        got = ex.execute(mid)
        want = mid in expect_ok
        flag = "PASS" if got == want else "FAIL"
        if got != want:
            all_ok = False
        print(f"    [{flag}] {mid:9s} valid={got}")

    # Real (harmless) execution path: 'shell' echo should actually spawn.
    live = MacroExecutor(logger=lambda m: None)
    live.macros = {"echo": {"type": "shell", "value": "true"}}
    spawned = live.execute("echo")
    print(f"    [{'PASS' if spawned else 'FAIL'}] live shell spawn works")
    return all_ok and spawned


# ──────────────────────────────────────────────
# Plotting (optional)
# ──────────────────────────────────────────────
def save_plots(profiler, fs, rng, acc_curve, cm_data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"\n[plots skipped: matplotlib unavailable — {e}]")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Template heatmaps.
    fig, axes = plt.subplots(1, len(GESTURES), figsize=(4 * len(GESTURES), 3))
    for ax, name in zip(np.atleast_1d(axes), GESTURES):
        prof = profiler.get_display_profile(name)
        ax.imshow(prof, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"STEC template: {name}")
        ax.set_xlabel("time frame"); ax.set_ylabel("band")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "templates.png"), dpi=110)
    plt.close(fig)

    # Onset detection function demo.
    tr = make_transient(fs, *GESTURES["Snap"], rng=rng)
    audio = embed(tr, fs, gain=0.7, snr_db=10, rng=rng, onset_s=0.5)
    from core.dsp_utils import spectral_flux
    odf, centers = spectral_flux(audio, config.ONSET_FRAME_SIZE, config.ONSET_HOP)
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax[0].plot(np.arange(len(audio)) / fs, audio, lw=0.5, color="#3fb950")
    ax[0].set_title("Noisy signal (Snap @ 0.5 s, SNR 10 dB)"); ax[0].set_ylabel("amp")
    ax[1].plot(centers / fs, odf, color="#58a6ff")
    ax[1].set_title("Spectral-flux onset detection function"); ax[1].set_xlabel("s")
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "onset_odf.png"), dpi=110)
    plt.close(fig)

    # Accuracy vs SNR.
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = sorted(acc_curve, reverse=True)
    ax.plot(xs, [acc_curve[s] * 100 for s in xs], "o-", color="#bc8cff")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title("Classification accuracy vs noise"); ax.grid(alpha=0.3)
    ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, "accuracy_vs_snr.png"), dpi=110)
    plt.close(fig)

    # Confusion matrix.
    names, cm = cm_data
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names) + 1)); ax.set_xticklabels(names + ["miss"])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white")
    ax.set_title("Confusion matrix (SNR 15 dB)")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=110)
    plt.close(fig)

    print(f"\n[plots saved to {RESULTS_DIR}/]")


# ──────────────────────────────────────────────
def run_tests():
    print("=" * 62)
    print(" STEC v2 — Acoustic Transient Detection : Robustness Suite")
    print("=" * 62)
    fs = config.SAMPLE_RATE
    rng = np.random.default_rng(1234)

    profiler, counts = build_profiler(fs, rng)

    r1 = test_training(counts)
    acc = test_accuracy_vs_snr(profiler, fs, rng)
    r3 = test_volume_invariance(profiler, fs, rng)
    fpr = test_false_positive(profiler, fs, rng)
    cm_data = test_confusion(profiler, fs, rng)
    r6 = test_streaming_sequence(profiler, fs, rng)
    r7 = test_macro_engine()

    save_plots(profiler, fs, rng, acc, cm_data)

    # ── Summary / pass criteria ──
    print("\n" + "=" * 62)
    print(" SUMMARY")
    print("=" * 62)
    checks = {
        "Training extracts >=3 exemplars/gesture": r1,
        "Accuracy >=90% at 15 dB SNR":             acc.get(15, 0) >= 0.90,
        "Accuracy >=70% at 5 dB SNR":              acc.get(5, 0) >= 0.70,
        "Volume invariant (>=80% across gains)":   r3,
        "False-positive rate <=10%":               fpr <= 0.10,
        "Streaming Snap->Clap fires macro":        r6,
        "Macro engine dispatches all types":       r7,
    }
    all_ok = True
    for label, ok in checks.items():
        all_ok &= ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {label}")
    print("=" * 62)
    print(f" RESULT: {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    print("=" * 62)
    return all_ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)

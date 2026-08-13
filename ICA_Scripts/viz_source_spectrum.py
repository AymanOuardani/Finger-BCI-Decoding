"""
viz_source_spectrum.py

Where in the frequency spectrum a source's trial energy comes from.

A source can correlate with a finger task while carrying almost nothing in the
mu/beta range that motor imagery is supposed to modulate — which usually means
the association is drift or muscle rather than cortical. This script shows that
directly.

Two modes:

  task   (default)  Per-task view of the trial segments:
                    left  — mean Welch PSD per finger (1-45 Hz, log), alpha and
                            beta shaded
                    right — band power per finger (delta/theta/alpha/beta/gamma)
                    The band powers and the share of power below 8 Hz are also
                    printed, since that fraction is the quickest tell for drift.

  full              Welch PSD of the whole continuous source, no trials, no
                    tasks, with the bands drawn as coloured surfaces.

Usage:
    python viz_source_spectrum.py <subj> <sess> <nclass> <model> <src> [mode] [task]

    python viz_source_spectrum.py 4 5 3 Orig 80         # per-task spectrum
    python viz_source_spectrum.py 4 5 3 Orig 80 full    # whole-source spectrum

Figures -> RESULTS_ROOT/Source Diagnostics/S{XX}/
"""

import os
import sys
import glob
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import corr_heatmap as ch
import fill_energy_table as fet
import energy_stats as es
from Functions import build_raw_from_mat_files, get_folder
from config import SOURCE_FIGS_DIR


FMIN, FMAX = 1.0, 45.0
NPERSEG = 1024

PLOT_BANDS = [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13),
              ("beta", 13, 30), ("gamma", 30, 45)]
BAND_COLORS = ["#c7d9f1", "#bfe3d0", "#ffe0b3", "#f7c0c0", "#ddd0ec"]


def _load_source(subj, sess, nclass, model, src, task):
    """The continuous ICA source signal, its trials and their classes."""
    folder = get_folder(subj, task, sess, nclass, model)
    mats = sorted(glob.glob(os.path.join(folder, "*.mat")))
    raw, _, _ = build_raw_from_mat_files(mats)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)
    sfreq = float(raw.info["sfreq"])

    ica = ch._load_offline_ica(subj, task)
    sources = ica.get_sources(raw)
    signal = sources.get_data()[src]

    trials = fet._chronological_trials(raw)
    classes = np.array([c for c, _s, _e in trials])

    del raw, sources
    gc.collect()
    return signal, trials, classes, sfreq


def spectrum_per_task(signal, trials, classes, sfreq, fingers, head, save_path):
    psd_by_finger = {f: [] for f in fingers}
    freqs = None
    for cls, start, end in trials:
        segment = signal[start:end]
        if len(segment) < NPERSEG or cls not in psd_by_finger:
            continue
        freqs, power = welch(segment, fs=sfreq, nperseg=NPERSEG,
                             noverlap=NPERSEG // 2)
        psd_by_finger[cls].append(power)

    usable = [f for f in fingers if psd_by_finger[f]]
    if freqs is None or not usable:
        print("  [SKIP] no trial long enough for a Welch estimate")
        return

    mask = (freqs >= FMIN) & (freqs <= FMAX)
    df = freqs[1] - freqs[0]
    mean_psd = {f: np.mean(np.array(psd_by_finger[f]), axis=0) for f in usable}

    band_power = {}
    for f in usable:
        band_power[f] = [mean_psd[f][(freqs >= lo) & (freqs < hi)].sum() * df
                         for _lbl, lo, hi in PLOT_BANDS]

    print(f"{head} — band power per task:")
    print("  band   " + "  ".join(f"{f:>10s}" for f in usable))
    for bi, (lbl, _lo, _hi) in enumerate(PLOT_BANDS):
        print(f"  {lbl:5s}  " + "  ".join(f"{band_power[f][bi]:10.3e}" for f in usable))
    for f in usable:
        total = mean_psd[f][mask].sum()
        low = mean_psd[f][(freqs >= FMIN) & (freqs < 8)].sum()
        print(f"  {f}: {100 * low / total:.0f}% of 1-45 Hz power is below 8 Hz")

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5))

    for f in usable:
        ax_l.semilogy(freqs[mask], mean_psd[f][mask],
                      color=es.CLASS_COLORS.get(f, "gray"), lw=2, label=f)
    ax_l.axvspan(8, 13, color="gray", alpha=0.08)
    ax_l.axvspan(13, 30, color="gray", alpha=0.12)
    ax_l.text(10.5, ax_l.get_ylim()[1], "α", ha="center", va="top", color="gray")
    ax_l.text(21, ax_l.get_ylim()[1], "β", ha="center", va="top", color="gray")
    ax_l.set_xlim(FMIN, FMAX)
    ax_l.set_xlabel("Frequency (Hz)")
    ax_l.set_ylabel("PSD  (a.u. / Hz, log)")
    ax_l.set_title("mean PSD per task")
    ax_l.legend()
    ax_l.grid(alpha=0.25, which="both")

    x = np.arange(len(PLOT_BANDS))
    width = 0.8 / len(usable)
    for fi, f in enumerate(usable):
        ax_r.bar(x + fi * width, band_power[f], width=width,
                 color=es.CLASS_COLORS.get(f, "gray"), label=f, alpha=0.9)
    ax_r.set_yscale("log")
    ax_r.set_xticks(x + 0.4 - width / 2)
    ax_r.set_xticklabels([b[0] for b in PLOT_BANDS])
    ax_r.set_ylabel("Band power  Σ PSD·Δf  (log)")
    ax_r.set_title("band power per task")
    ax_r.legend()
    ax_r.grid(axis="y", alpha=0.25, which="both")

    fig.suptitle(f"{head} | where the energy comes from", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.basename(save_path)}")


def spectrum_full(signal, sfreq, head, save_path):
    freqs, power = welch(signal, fs=sfreq, nperseg=NPERSEG, noverlap=NPERSEG // 2)
    mask = (freqs >= FMIN) & (freqs <= FMAX)

    fig, ax = plt.subplots(figsize=(9, 5))
    for (lbl, lo, hi), color in zip(PLOT_BANDS, BAND_COLORS):
        ax.axvspan(lo, hi, color=color, alpha=0.55, zorder=0)
    ax.semilogy(freqs[mask], power[mask], color="black", lw=1.6, zorder=3)
    ax.set_xlim(FMIN, FMAX)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD  (a.u. / Hz, log)")
    ax.set_title(f"{head} | whole-source Welch spectrum")
    ax.grid(alpha=0.25, which="both")
    ax.legend(handles=[Patch(facecolor=c, alpha=0.55, label=l)
                       for (l, _lo, _hi), c in zip(PLOT_BANDS, BAND_COLORS)],
              loc="upper right", ncol=len(PLOT_BANDS), fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.basename(save_path)}")


def main():
    if len(sys.argv) < 6:
        print(__doc__)
        sys.exit(1)

    subj = int(sys.argv[1])
    sess = int(sys.argv[2])
    nclass = int(sys.argv[3])
    model = sys.argv[4]
    src = int(sys.argv[5])
    mode = sys.argv[6].lower() if len(sys.argv) > 6 else "task"
    task = sys.argv[7].upper() if len(sys.argv) > 7 else "MI"

    if mode not in ("task", "full"):
        print(f"unknown mode {mode!r}; expected task or full")
        sys.exit(1)

    out_dir = os.path.join(SOURCE_FIGS_DIR, f"S{subj:02d}")
    tag = f"S{subj:02d}_src{src}_Sess{sess:02d}_{nclass}class_{model}"
    head = f"S{subj:02d} · src{src} · Sess{sess:02d} {nclass}class {model}"
    print(f"\n=== {tag}  [spectrum: {mode}] -> {out_dir} ===")

    signal, trials, classes, sfreq = _load_source(subj, sess, nclass, model, src, task)

    if mode == "full":
        spectrum_full(signal, sfreq, head,
                      os.path.join(out_dir, f"{tag}_spectrum_full.png"))
    else:
        fingers = es.present_fingers(classes, nclass)
        spectrum_per_task(signal, trials, classes, sfreq, fingers, head,
                          os.path.join(out_dir, f"{tag}_spectrum_per_task.png"))

    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    main()

"""_spectrum_src.py — TEMP. Per-task power spectrum of ONE ICA source.

Shows WHERE (which frequencies) the trial energy comes from, per task, so we can
see the frequency origin of the Thumb energy of S04/src80.

Left  : mean Welch PSD per task (1-45 Hz, log), alpha/beta shaded.
Right : mean band power per task (delta/theta/alpha/beta/gamma), log.

Usage:  python _spectrum_src.py <subj> <sess> <nclass> <model> <src>   (default 4 5 3 Orig 80)
"""
import os, sys, glob, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import corr_heatmap as ch
import fill_energy_table as fet
from Functions import build_raw_from_mat_files, get_folder
from config import TASK_LABELS

CLASS_COL = {"Thumb": "#1f77b4", "Index": "#ff7f0e",
             "Middle": "#9467bd", "Pinky": "#d62728"}
BANDS = [("delta", 1, 4), ("theta", 4, 8), ("alpha", 8, 13),
         ("beta", 13, 30), ("gamma", 30, 45)]
FMIN, FMAX = 1.0, 45.0


def main():
    subj = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    sess = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    ncl  = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    model = sys.argv[4] if len(sys.argv) > 4 else "Orig"
    src  = int(sys.argv[5]) if len(sys.argv) > 5 else 80
    task = "MI"

    folder = get_folder(subj, task, sess, ncl, model)
    mats = sorted(glob.glob(os.path.join(folder, "*.mat")))
    raw, _, _ = build_raw_from_mat_files(mats)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)
    sfreq = float(raw.info["sfreq"])

    ica = ch._load_offline_ica(subj, task)
    one = ica.get_sources(raw).copy().pick([ica.get_sources(raw).ch_names[src]])
    sig = one.get_data()[0]

    trials = fet._chronological_trials(raw)
    classes = np.array([c for c, _s, _e in trials])
    tasks = [t for t in TASK_LABELS if t in set(classes)]

    nperseg = 1024
    psd_by_task = {T: [] for T in tasks}
    freqs = None
    for cls, s, e in trials:
        seg = sig[s:e]
        if len(seg) < nperseg:
            continue
        f, p = welch(seg, fs=sfreq, nperseg=nperseg, noverlap=nperseg // 2)
        freqs = f
        psd_by_task[cls].append(p)
    mask = (freqs >= FMIN) & (freqs <= FMAX)
    fm = freqs[mask]
    mean_psd = {T: np.mean(np.array(psd_by_task[T]), axis=0)[mask] for T in tasks}

    # band power per task (integrate PSD over each band)
    df = freqs[1] - freqs[0]
    bandpow = {T: [] for T in tasks}
    for T in tasks:
        full = np.mean(np.array(psd_by_task[T]), axis=0)
        for _lbl, lo, hi in BANDS:
            bm = (freqs >= lo) & (freqs < hi)
            bandpow[T].append(full[bm].sum() * df)

    # print numeric summary
    print(f"S{subj:02d} src{src} Sess{sess:02d} {ncl}class {model} — band power per task:")
    print("  band   " + "  ".join(f"{T:>10s}" for T in tasks))
    for bi, (lbl, lo, hi) in enumerate(BANDS):
        print(f"  {lbl:5s}  " + "  ".join(f"{bandpow[T][bi]:10.3e}" for T in tasks))
    # fraction of (1-45) power below 8 Hz
    for T in tasks:
        full = np.mean(np.array(psd_by_task[T]), axis=0)
        tot = full[(freqs >= FMIN) & (freqs <= FMAX)].sum()
        low = full[(freqs >= FMIN) & (freqs < 8)].sum()
        print(f"  {T}: {100*low/tot:.0f}% of 1-45 Hz power is below 8 Hz")

    # ── figure ────────────────────────────────────────────────────────────
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))

    for T in tasks:
        axL.semilogy(fm, mean_psd[T], color=CLASS_COL.get(T, "gray"),
                     lw=2, label=T)
    axL.axvspan(8, 13, color="gray", alpha=0.08)
    axL.axvspan(13, 30, color="gray", alpha=0.12)
    axL.text(10.5, axL.get_ylim()[1], "α", ha="center", va="top", color="gray")
    axL.text(21, axL.get_ylim()[1], "β", ha="center", va="top", color="gray")
    axL.set_xlim(FMIN, FMAX)
    axL.set_xlabel("Frequency (Hz)")
    axL.set_ylabel("PSD  (a.u. / Hz, log)")
    axL.set_title(f"S{subj:02d} · src{src} · mean PSD per task")
    axL.legend()
    axL.grid(alpha=0.25, which="both")

    x = np.arange(len(BANDS))
    w = 0.8 / len(tasks)
    for ti, T in enumerate(tasks):
        axR.bar(x + ti * w, bandpow[T], width=w, color=CLASS_COL.get(T, "gray"),
                label=T, alpha=0.9)
    axR.set_yscale("log")
    axR.set_xticks(x + 0.4 - w / 2)
    axR.set_xticklabels([b[0] for b in BANDS])
    axR.set_ylabel("Band power  Σ PSD·Δf  (log)")
    axR.set_title(f"S{subj:02d} · src{src} · band power per task")
    axR.legend()
    axR.grid(axis="y", alpha=0.25, which="both")

    fig.suptitle(f"S{subj:02d} src{src} | Sess{sess:02d} {ncl}class {model} | "
                 f"where the energy comes from", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"C:/Users/aymen/Desktop/S{subj:02d}/S{subj:02d}_src{src}_Sess{sess:02d}_{ncl}class_{model}_spectrum.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nfigure -> {out}")
    del raw, one
    gc.collect()


if __name__ == "__main__":
    main()

"""_spectrum_source_full.py — TEMP. Welch PSD of the WHOLE ICA source 80.

Simple single-panel figure: Welch power spectrum of the entire continuous source
(no trials, no tasks), with the frequency bands drawn as coloured surfaces.

Usage: python _spectrum_source_full.py <subj> <sess> <nclass> <model> <src>
       (default 4 5 3 Orig 80)
"""
import os, sys, glob, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import corr_heatmap as ch
from Functions import build_raw_from_mat_files, get_folder

# (label, lo, hi, colour)
BANDS = [("delta (1-4 Hz)",   1,  4,  "#3b6ea5"),
         ("theta (4-8 Hz)",   4,  8,  "#2ca25f"),
         ("alpha (8-13 Hz)",  8,  13, "#f1a340"),
         ("beta (13-30 Hz)",  13, 30, "#de2d26"),
         ("gamma (30-45 Hz)", 30, 45, "#756bb1")]
FMIN, FMAX = 0.5, 45.0


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

    sources = ica = ch._load_offline_ica(subj, task)
    sources = ica.get_sources(raw)
    sig = sources.get_data(picks=src)[0]          # whole continuous source

    # Welch over the entire signal: long windows -> fine low-freq resolution
    # and many averages -> smooth PSD.
    f, p = welch(sig, fs=sfreq, nperseg=4096, noverlap=2048)
    m = (f >= FMIN) & (f <= FMAX)
    f, p = f[m], p[m]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    # coloured band surfaces (full height)
    for lbl, lo, hi, col in BANDS:
        ax.axvspan(lo, hi, color=col, alpha=0.18, lw=0)
    ax.semilogy(f, p, color="black", lw=1.6, zorder=5)
    ax.set_xlim(FMIN, FMAX)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD  (a.u. / Hz, log)")
    ax.set_title(f"S{subj:02d} · ICA source {src} · Welch PSD (full source) · "
                 f"Sess{sess:02d} {ncl}class {model}")
    ax.grid(alpha=0.25, which="both")
    handles = [Patch(facecolor=c, alpha=0.35, label=l) for l, _lo, _hi, c in BANDS]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = (f"C:/Users/aymen/Desktop/S{subj:02d}/"
           f"S{subj:02d}_src{src}_Sess{sess:02d}_{ncl}class_{model}_full_spectrum.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")
    del raw, sources
    gc.collect()


if __name__ == "__main__":
    main()

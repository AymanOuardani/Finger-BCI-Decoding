"""_gen_band_figs.py — TEMP. Per-task energy-by-class figures for ALL bands,
one ICA source, on a single SHARED energy axis (so bands are comparable).

Generates for S04/src80: per-trial (fullband) + by-class strips for
Fullband, delta, alpha, beta, gamma. All on one common log energy axis.
"""
import os, sys, glob, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")

import corr_heatmap as ch
import fill_energy_table as fet
import _gen_clean_figs as gcf            # reuse _strip_by_class / _per_trial_fig / CLASS_COL
from Functions import build_raw_from_mat_files, get_folder
from config import TASK_LABELS

SUBJ  = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SESS  = int(sys.argv[2]) if len(sys.argv) > 2 else 5
NCL   = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MODEL = sys.argv[4] if len(sys.argv) > 4 else "Orig"
SRC   = int(sys.argv[5]) if len(sys.argv) > 5 else 80
TASK  = "MI"

# (label, lo, hi, filename-tag)
BANDS = [("Fullband", None, None, "2_fullband"),
         ("Delta",   1.0, 4.0,  "4_delta"),
         ("Alpha",   8.0, 13.0, "5_alpha"),
         ("Beta",    13.0, 30.0, "3_beta"),
         ("Gamma",   30.0, 45.0, "6_gamma")]


def main():
    folder = get_folder(SUBJ, TASK, SESS, NCL, MODEL)
    mats = sorted(glob.glob(os.path.join(folder, "*.mat")))
    raw, _, _ = build_raw_from_mat_files(mats)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)

    ica = ch._load_offline_ica(SUBJ, TASK)
    src_raw = ica.get_sources(raw)
    one = src_raw.copy().pick([src_raw.ch_names[SRC]])
    sig_full = one.get_data()[0]

    trials = fet._chronological_trials(raw)
    classes = np.array([c for c, _s, _e in trials])
    tasks = [t for t in TASK_LABELS if t in set(classes)]

    def energy(sig):
        return np.array([np.sum(sig[s:e] ** 2) for _c, s, e in trials])

    E_by = {}
    for lbl, lo, hi, _fn in BANDS:
        if lo is None:
            sig = sig_full
        else:
            sig = one.copy().filter(lo, hi, verbose=False).get_data()[0]
        E_by[lbl] = energy(sig)

    # one common energy axis over ALL bands
    allE = np.concatenate([E[E > 0] for E in E_by.values()])
    ylim = (allE.min() * 0.7, allE.max() * 1.4)
    print(f"shared energy axis over all bands: {ylim[0]:.2e} .. {ylim[1]:.2e}")

    out_dir = f"C:/Users/aymen/Desktop/S{SUBJ:02d}"
    tag = f"S{SUBJ:02d}_src{SRC}_Sess{SESS:02d}_{NCL}class_{MODEL}"
    head = f"S{SUBJ:02d} · ICA source {SRC}"

    gcf._per_trial_fig(
        E_by["Fullband"], classes, tasks,
        f"{head} · Fullband energy per trial",
        os.path.join(out_dir, f"{tag}_1_energy_per_trial.png"), ylim=ylim)

    for lbl, lo, hi, fn in BANDS:
        band_title = "Fullband" if lo is None else f"{lbl}-band ({lo:g}–{hi:g} Hz)"
        gcf._strip_by_class(
            E_by[lbl], classes, tasks,
            f"{head} · {band_title} energy by task",
            r"Energy  $\Sigma x^2$  (log scale)",
            os.path.join(out_dir, f"{tag}_{fn}_by_class.png"), ylim=ylim)
        print(f"  saved: {tag}_{fn}_by_class.png")

    del raw, src_raw, one
    gc.collect()
    print("done.")


if __name__ == "__main__":
    main()

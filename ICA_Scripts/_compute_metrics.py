"""_compute_metrics.py — TEMP. Compute the full metric suite for the report.

Loads S09 Sess01 3class Orig and S04 Sess05 3class Orig, computes per-trial
fullband + beta energy for all 128 ICA sources, then for selected sources
reports: raw Pearson, log-energy Pearson, trimmed-5% Pearson, Spearman,
AUC (Mann-Whitney), rank-biserial (2*AUC-1), beta Pearson, artifact flag,
and tail descriptors. Prints a parseable report.
"""
import os, sys, glob, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr, kurtosis

import corr_heatmap as ch
import fill_energy_table as fet
from Functions import (build_raw_from_mat_files, get_folder, compute_exclusions)
from config import TASK_LABELS, EOG_THRESHOLD, EMG_SLOPE_THRESH


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def metrics(E, classes, T):
    m = (classes == T).astype(float)
    n = len(E)
    r_raw = pearson(E, m)
    r_log = pearson(np.log10(E + 1e-30), m)
    k = max(1, int(round(0.05 * n)))
    keep = np.argsort(E)[:-k]
    r_trim = pearson(E[keep], m[keep])
    rho = float(spearmanr(E, m)[0])
    inm = classes == T
    n1, n0 = int(inm.sum()), int((~inm).sum())
    U = mannwhitneyu(E[inm], E[~inm], alternative="two-sided").statistic
    auc = float(U / (n1 * n0))
    rb = 2 * auc - 1
    return dict(r_raw=r_raw, r_log=r_log, r_trim=r_trim, rho=rho,
                auc=auc, rb=rb)


def load(subj, sess, ncl, model, task="MI"):
    folder = get_folder(subj, task, sess, ncl, model)
    mats = sorted(glob.glob(os.path.join(folder, "*.mat")))
    raw, _, _ = build_raw_from_mat_files(mats)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)
    ica = ch._load_offline_ica(subj, task)

    art = set()
    try:
        rf = raw.copy().filter(1.0, None, verbose=False)
        _, eog, emg = compute_exclusions(
            ica, rf, muscle_thresh=EMG_SLOPE_THRESH, eog_thresh=EOG_THRESHOLD,
            ch_names=rf.info["ch_names"])
        art = set(eog) | set(emg)
        del rf
    except Exception as ex:
        print(f"  [warn] artifact detection failed: {ex}")

    src = ica.get_sources(raw)
    Dfull = src.get_data()
    Dbeta = src.copy().filter(13, 30, verbose=False).get_data()
    trials = fet._chronological_trials(raw)
    classes = np.array([c for c, _s, _e in trials])
    ncomp = Dfull.shape[0]
    nt = len(trials)
    Ef = np.empty((nt, ncomp)); Eb = np.empty((nt, ncomp))
    for ti, (_c, s, e) in enumerate(trials):
        Ef[ti] = np.sum(Dfull[:, s:e] ** 2, axis=1)
        Eb[ti] = np.sum(Dbeta[:, s:e] ** 2, axis=1)
    del raw, src, Dfull, Dbeta
    gc.collect()
    return Ef, Eb, classes, art, ncomp


def row(tag, src, E, Eb, classes, T, art):
    mm = metrics(E[:, src], classes, T)
    rb_beta = pearson(Eb[:, src], (classes == T).astype(float))
    med = np.median(E[:, src])
    kurt = float(kurtosis(E[:, src]))
    mx = float(E[:, src].max() / med)
    flag = "Yes" if src in art else "No"
    print(f"ROW|{tag}|{src}|{T}|{mm['r_raw']:.3f}|{mm['r_log']:.3f}|"
          f"{mm['r_trim']:.3f}|{mm['rho']:.3f}|{mm['auc']:.3f}|{mm['rb']:.3f}|"
          f"{rb_beta:.3f}|{flag}|{kurt:.1f}|{mx:.1f}")


def main():
    print("\n########## S09 Sess01 3class Orig ##########")
    Ef, Eb, classes, art, ncomp = load(9, 1, 3, "Orig")
    print(f"n_trials={len(classes)}  ncomp={ncomp}  artifacts={len(art)}")

    # Rank all sources by raw Pearson to Pinky.
    mP = (classes == "Pinky").astype(float)
    pear = np.array([pearson(Ef[:, c], mP) for c in range(ncomp)])
    order = np.argsort(pear)[::-1]
    print("\nTOP raw-Pearson(Pinky) sources (src: r_raw, artifact):")
    for c in order[:12]:
        print(f"  src {c:3d}: r={pear[c]:.3f}  artifact={c in art}")

    # Selected: top high-Pearson-to-Pinky sources (exclude 15, shown separately).
    selected = [c for c in order if c != 15][:6]
    print(f"\nSELECTED high-Pearson(Pinky) S09 sources: {selected}")

    print("\n--- S09 src15 (all tasks) ---")
    for T in [t for t in TASK_LABELS if t in set(classes)]:
        row("S09s15", 15, Ef, Eb, classes, T, art)

    print("\n--- S09 selected sources (Pinky) ---")
    for c in selected:
        row("S09sel", c, Ef, Eb, classes, "Pinky", art)

    del Ef, Eb; gc.collect()

    print("\n########## S04 Sess05 3class Orig ##########")
    Ef4, Eb4, classes4, art4, ncomp4 = load(4, 5, 3, "Orig")
    print(f"n_trials={len(classes4)}  ncomp={ncomp4}  artifacts={len(art4)}")
    print("\n--- S04 src80 (all tasks) ---")
    for T in [t for t in TASK_LABELS if t in set(classes4)]:
        row("S04s80", 80, Ef4, Eb4, classes4, T, art4)


if __name__ == "__main__":
    main()

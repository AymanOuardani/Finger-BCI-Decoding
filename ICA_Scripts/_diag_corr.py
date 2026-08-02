"""_diag_corr.py — TEMP diagnostic.

Decompose the energy<->task Pearson correlation for ONE ICA source of ONE
recording, exactly as fill_energy_table / _gen_energy_images compute it, and
add robust comparisons to explain a suspicious value.

Usage:
    python _diag_corr.py <subj> <sess> <nclass> <model> <src> [task]
"""
import os, sys, glob, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import rankdata, kurtosis, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import corr_heatmap as ch
import fill_energy_table as fet
from Functions import (build_raw_from_mat_files, get_folder, get_offline_folder,
                       compute_exclusions)
from config import TASK_LABELS, EOG_THRESHOLD, EMG_SLOPE_THRESH

CLASS_COL = {"Thumb": "#2196F3", "Index": "#FF9800",
             "Middle": "#9C27B0", "Pinky": "#F44336"}


def per_trial_energy(sig1d, trials):
    return np.array([np.sum(sig1d[s:e] ** 2) for (_c, s, e) in trials])


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def auc(E, mask):
    """P(energy_in_class > energy_out) — robust effect size, 0.5 = none."""
    x, y = E[mask], E[~mask]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    U = mannwhitneyu(x, y, alternative="two-sided").statistic
    return float(U / (len(x) * len(y)))


def main():
    subj = int(sys.argv[1]); sess = int(sys.argv[2]); ncl = int(sys.argv[3])
    model = sys.argv[4]; src = int(sys.argv[5])
    task = "MI"
    focus_task = sys.argv[6] if len(sys.argv) > 6 else None

    print(f"\n{'='*70}\n  S{subj:02d} | {task} | Sess{sess:02d} {ncl}class {model} | "
          f"ICA source {src}\n{'='*70}")

    folder = get_folder(subj, task, sess, ncl, model)
    mats = sorted(glob.glob(os.path.join(folder, "*.mat")))
    raw, _, _ = build_raw_from_mat_files(mats)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)

    ica = ch._load_offline_ica(subj, task)

    # Per-recording EOG/EMG flags (cheap: online raw).
    rec_eog = rec_emg = set()
    try:
        rf = raw.copy().filter(1.0, None, verbose=False)
        _, e_idx, m_idx = compute_exclusions(
            ica, rf, muscle_thresh=EMG_SLOPE_THRESH, eog_thresh=EOG_THRESHOLD,
            ch_names=rf.info["ch_names"])
        rec_eog, rec_emg = set(e_idx), set(m_idx)
        del rf
    except Exception as ex:
        print(f"  [warn] per-recording detection failed: {ex}")

    src_raw = ica.get_sources(raw)
    chname = src_raw.ch_names[src]
    one = src_raw.copy().pick([chname])
    sig_full = one.get_data()[0]
    sig_alpha = one.copy().filter(8, 13, verbose=False).get_data()[0]
    sig_beta = one.copy().filter(13, 30, verbose=False).get_data()[0]

    trials = fet._chronological_trials(raw)
    classes = np.array([c for c, _s, _e in trials])
    tasks = [t for t in TASK_LABELS if t in set(classes)]
    n = len(trials)

    Ef = per_trial_energy(sig_full, trials)
    Ea = per_trial_energy(sig_alpha, trials)
    Eb = per_trial_energy(sig_beta, trials)

    print(f"\n  n_trials={n}  classes={ {t:int((classes==t).sum()) for t in tasks} }")
    print(f"  source {src}: in per-recording EOG={src in rec_eog}  "
          f"EMG={src in rec_emg}")

    med = np.median(Ef)
    print(f"\n  FULLBAND energy distribution (source {src}):")
    print(f"    median={med:.3e}  mean={Ef.mean():.3e}  max={Ef.max():.3e}")
    print(f"    max/median={Ef.max()/med:.1f}  p99/median="
          f"{np.percentile(Ef,99)/med:.1f}  kurtosis(excess)={kurtosis(Ef):.1f}")
    order = np.argsort(Ef)[::-1][:8]
    print("    top-8 energy trials (rank: class, energy, x median):")
    for r, i in enumerate(order, 1):
        print(f"      {r}: {classes[i]:6s}  {Ef[i]:.3e}  ({Ef[i]/med:.1f}x)")

    print(f"\n  {'task':7s} {'nIn':>4s} {'r_full':>8s} {'r_logE':>8s} "
          f"{'r_trim5%':>9s} {'r_alpha':>8s} {'r_beta':>8s} {'AUC_full':>9s} "
          f"{'medIn/medOut':>13s}")
    results = {}
    logEf = np.log10(Ef + 1e-30)
    # trimmed: drop the top 5% highest-energy trials entirely
    k = max(1, int(round(0.05 * n)))
    keep = np.argsort(Ef)[:-k]
    for T in tasks:
        m = (classes == T).astype(float)
        r_full = pearson(Ef, m)
        r_log = pearson(logEf, m)
        r_trim = pearson(Ef[keep], m[keep])
        r_a = pearson(Ea, m)
        r_b = pearson(Eb, m)
        a = auc(Ef, classes == T)
        in_med = np.median(Ef[classes == T]); out_med = np.median(Ef[classes != T])
        ratio = in_med / out_med if out_med else float("nan")
        results[T] = dict(r_full=r_full, r_log=r_log, r_trim=r_trim,
                          r_a=r_a, r_b=r_b, auc=a, ratio=ratio)
        print(f"  {T:7s} {int(m.sum()):4d} {r_full:8.3f} {r_log:8.3f} "
              f"{r_trim:9.3f} {r_a:8.3f} {r_b:8.3f} {a:9.3f} {ratio:13.2f}")

    if focus_task is None:
        focus_task = max(tasks, key=lambda T: abs(results[T]["r_full"]))
    print(f"\n  >>> FOCUS task = {focus_task}: "
          f"r_full={results[focus_task]['r_full']:.3f}  "
          f"r_logE={results[focus_task]['r_log']:.3f}  "
          f"r_trim5%={results[focus_task]['r_trim']:.3f}  "
          f"AUC={results[focus_task]['auc']:.3f}")

    # ── figure ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    cols = [CLASS_COL.get(c, "gray") for c in classes]
    ax[0].scatter(range(n), Ef, c=cols, s=18)
    ax[0].set_yscale("log"); ax[0].set_title(
        f"S{subj:02d} src{src} | Fullband energy per trial (chronological)")
    ax[0].set_xlabel("trial (time order)"); ax[0].set_ylabel("Σx² (log)")
    for j, (E, lbl) in enumerate([(Ef, "Fullband"), (Eb, "Beta")]):
        a_ = ax[1 + j]
        for ti, T in enumerate(tasks):
            v = E[classes == T]
            xj = np.random.normal(ti, 0.06, len(v))
            a_.scatter(xj, v, c=CLASS_COL.get(T, "gray"), s=18, alpha=0.7)
            a_.scatter([ti], [np.median(v)], marker="_", s=900, c="black", zorder=6)
        a_.set_yscale("log"); a_.set_xticks(range(len(tasks)))
        a_.set_xticklabels(tasks); a_.set_title(f"{lbl} energy by class (median=bar)")
        a_.set_ylabel("Σx² (log)")
    fig.suptitle(f"S{subj:02d} Sess{sess:02d} {ncl}class {model} src{src} | "
                 f"focus {focus_task}: r_full={results[focus_task]['r_full']:.2f} "
                 f"AUC={results[focus_task]['auc']:.2f}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"C:/Users/aymen/Desktop/_diag_S{subj:02d}_Sess{sess:02d}_{ncl}c_{model}_src{src}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"\n  figure -> {out}")

    del raw, src_raw, one
    gc.collect()


if __name__ == "__main__":
    main()

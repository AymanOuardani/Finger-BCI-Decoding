"""_convince.py — TEMP. Eye vs metric: a supervisor-facing figure.

Two columns: S09/src15 (Pinky), S04/src80 (Thumb).
Row A: all trials SORTED by Fullband energy (x=rank), coloured by task, log y.
       Shows how separable the task is (= what AUC measures). The right end
       (highest energy) is what the eye notices.
Row B: per-trial energy distribution by task (strip + median).

Annotated with AUC = P(energy higher in task) and the % of the top-n1 highest-
energy trials that actually belong to the task.
"""
import os, sys, glob, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import rankdata, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import corr_heatmap as ch
import fill_energy_table as fet
from Functions import build_raw_from_mat_files, get_folder
from config import TASK_LABELS

CLASS_COL = {"Thumb": "#1f77b4", "Index": "#ff7f0e",
             "Middle": "#9467bd", "Pinky": "#d62728"}

CASES = [  # tag, subj, sess, ncl, model, src, focus task
    ("S09/src15", 9, 1, 3, "Orig", 15, "Pinky"),
    ("S04/src80", 4, 5, 3, "Orig", 80, "Thumb"),
]


def energy(subj, sess, ncl, model, src):
    folder = get_folder(subj, "MI", sess, ncl, model)
    mats = sorted(glob.glob(os.path.join(folder, "*.mat")))
    raw, _, _ = build_raw_from_mat_files(mats)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)
    ica = ch._load_offline_ica(subj, "MI")
    s = ica.get_sources(raw)
    sig = s.get_data(picks=src)[0]
    trials = fet._chronological_trials(raw)
    classes = np.array([c for c, _a, _b in trials])
    E = np.array([np.sum(sig[a:b] ** 2) for _c, a, b in trials])
    del raw, s
    gc.collect()
    return E, classes


def auc_of(E, classes, T):
    inm = classes == T
    n1, n0 = int(inm.sum()), int((~inm).sum())
    U = mannwhitneyu(E[inm], E[~inm], alternative="two-sided").statistic
    return U / (n1 * n0), n1


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for col, (tag, subj, sess, ncl, model, src, T) in enumerate(CASES):
        E, classes = energy(subj, sess, ncl, model, src)
        tasks = [t for t in TASK_LABELS if t in set(classes)]
        N = len(E)
        auc, n1 = auc_of(E, classes, T)

        # top-n1 highest-energy trials: how many are actually task T?
        top = np.argsort(E)[::-1][:n1]
        pct_top = 100.0 * np.mean(classes[top] == T)

        # ── Row A: sorted by energy, coloured by task ──────────────────────
        axA = axes[0][col]
        order = np.argsort(E)                 # ascending
        x = np.arange(N)
        cols = [CLASS_COL.get(c, "gray") for c in classes[order]]
        axA.scatter(x, E[order], c=cols, s=20, zorder=3)
        axA.set_yscale("log")
        axA.axvline(N - n1, color="black", ls="--", lw=1.2)
        axA.text(N - n1, axA.get_ylim()[1], f"  top {n1} (highest energy)",
                 ha="left", va="top", fontsize=8)
        axA.set_xlabel("trials sorted by energy  (low → high)")
        axA.set_ylabel(r"energy $\Sigma x^2$ (log)")
        axA.set_title(f"{tag} — {T}:  AUC = {auc:.2f}\n"
                      f"of the {n1} highest-energy trials, {pct_top:.0f}% are {T} "
                      f"(if corr.=1 → 100%)", fontsize=10)
        axA.legend(handles=[Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=CLASS_COL[t], markeredgecolor="none", label=t)
                   for t in tasks], loc="upper left", fontsize=8)
        axA.grid(axis="y", alpha=0.2, which="both")

        # ── Row B: distribution by task ────────────────────────────────────
        axB = axes[1][col]
        rng = np.random.default_rng(0)
        for ti, t in enumerate(tasks):
            v = E[classes == t]
            axB.scatter(rng.normal(ti, 0.07, len(v)), v,
                        c=CLASS_COL.get(t, "gray"), s=18, alpha=0.7,
                        edgecolors="white", linewidths=0.3, zorder=3)
            axB.hlines(np.median(v), ti - 0.28, ti + 0.28, color="black",
                       lw=2.5, zorder=5)
        axB.set_yscale("log")
        axB.set_xticks(range(len(tasks)))
        axB.set_xticklabels([f"{t}\n(n={int((classes==t).sum())})" for t in tasks])
        axB.set_ylabel(r"energy $\Sigma x^2$ (log)")
        axB.set_title(f"{tag} — energy by task (median = bar)", fontsize=10)
        axB.grid(axis="y", alpha=0.2, which="both")

    fig.suptitle("Eye vs. metric — sorted by energy (top) and distribution (bottom)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = "C:/Users/aymen/Desktop/Convince_Eye_vs_Metric.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()

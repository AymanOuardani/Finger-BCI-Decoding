"""_gen_clean_figs.py — TEMP. Clean, supervisor-ready diagnostic figures.

For ONE ICA source of ONE recording, save THREE separate clean figures into
C:/Users/aymen/Desktop/S{subj}/ :
  1_energy_per_trial   — Fullband Sigma x^2 per trial (chronological), by class
  2_fullband_by_class  — Fullband Sigma x^2 grouped by task (median bar)
  3_beta_by_class      — Beta (13-30 Hz) Sigma x^2 grouped by task (median bar)

Usage:
    python _gen_clean_figs.py <subj> <sess> <nclass> <model> <src>
"""
import os, sys, glob, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import corr_heatmap as ch
import fill_energy_table as fet
from Functions import build_raw_from_mat_files, get_folder
from config import TASK_LABELS

# Print-friendly, consistent class colours.
CLASS_COL = {"Thumb": "#1f77b4", "Index": "#ff7f0e",
             "Middle": "#9467bd", "Pinky": "#d62728"}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 150,
})


def per_trial_energy(sig1d, trials):
    return np.array([np.sum(sig1d[s:e] ** 2) for (_c, s, e) in trials])


def _strip_by_class(E, classes, tasks, title, ylabel, save_path, ylim=None):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for ti, T in enumerate(tasks):
        v = E[classes == T]
        x = rng.normal(ti, 0.07, len(v))
        ax.scatter(x, v, c=CLASS_COL.get(T, "gray"), s=22, alpha=0.75,
                   edgecolors="white", linewidths=0.3, zorder=3)
        med = np.median(v)
        ax.hlines(med, ti - 0.28, ti + 0.28, color="black", lw=2.5, zorder=5)
    ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([f"{T}\n(n={int((classes == T).sum())})" for T in tasks])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.margins(x=0.12)
    handle = Line2D([0], [0], color="black", lw=2.5, label="median")
    ax.legend(handles=[handle], loc="upper right", framealpha=0.9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}")


def _per_trial_fig(E, classes, tasks, title, save_path, ylim=None):
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for T in tasks:
        idx = np.where(classes == T)[0]
        ax.scatter(idx, E[idx], c=CLASS_COL.get(T, "gray"), s=22, alpha=0.8,
                   edgecolors="white", linewidths=0.3, label=T, zorder=3)
    ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Trial (chronological order)")
    ax.set_ylabel(r"Energy  $\Sigma x^2$  (log scale)")
    ax.set_title(title)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(loc="upper right", ncol=len(tasks), framealpha=0.9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}")


def main():
    subj = int(sys.argv[1]); sess = int(sys.argv[2]); ncl = int(sys.argv[3])
    model = sys.argv[4]; src = int(sys.argv[5]); task = "MI"

    out_dir = f"C:/Users/aymen/Desktop/S{subj:02d}"
    tag = f"S{subj:02d}_src{src}_Sess{sess:02d}_{ncl}class_{model}"
    print(f"\n=== {tag} -> {out_dir} ===")

    folder = get_folder(subj, task, sess, ncl, model)
    mats = sorted(glob.glob(os.path.join(folder, "*.mat")))
    raw, _, _ = build_raw_from_mat_files(mats)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)

    ica = ch._load_offline_ica(subj, task)
    src_raw = ica.get_sources(raw)
    one = src_raw.copy().pick([src_raw.ch_names[src]])
    sig_full = one.get_data()[0]
    sig_beta = one.copy().filter(13, 30, verbose=False).get_data()[0]

    trials = fet._chronological_trials(raw)
    classes = np.array([c for c, _s, _e in trials])
    tasks = [t for t in TASK_LABELS if t in set(classes)]

    Ef = per_trial_energy(sig_full, trials)
    Eb = per_trial_energy(sig_beta, trials)

    # Shared ENERGY (y) axis across all three figures so fullband / beta /
    # per-trial are directly comparable on the same scale.
    allE = np.concatenate([Ef[Ef > 0], Eb[Eb > 0]])
    ylim = (allE.min() * 0.7, allE.max() * 1.4)
    print(f"  shared energy axis: {ylim[0]:.2e} .. {ylim[1]:.2e}")

    head = f"S{subj:02d} · ICA source {src}"
    _per_trial_fig(
        Ef, classes, tasks,
        f"{head} · Fullband energy per trial",
        os.path.join(out_dir, f"{tag}_1_energy_per_trial.png"), ylim=ylim)
    _strip_by_class(
        Ef, classes, tasks,
        f"{head} · Fullband energy by task",
        r"Energy  $\Sigma x^2$  (log scale)",
        os.path.join(out_dir, f"{tag}_2_fullband_by_class.png"), ylim=ylim)
    _strip_by_class(
        Eb, classes, tasks,
        f"{head} · Beta-band (13–30 Hz) energy by task",
        r"Energy  $\Sigma x^2$  (log scale)",
        os.path.join(out_dir, f"{tag}_3_beta_by_class.png"), ylim=ylim)

    del raw, src_raw, one
    gc.collect()
    print("  done.")


if __name__ == "__main__":
    main()

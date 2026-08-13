"""
viz_source_energy.py

Diagnostic figures for ONE ICA source of ONE recording — the visual counterpart
of the numbers produced by fill_source_metrics.py.

The point of these figures is to show whether a task association is carried by
the bulk of the trials or by a few high-energy ones, which a single correlation
coefficient cannot tell you.

Three modes:

  trials   (default)  Three figures on a SHARED log energy axis, so fullband and
                      band are directly comparable:
                        1_energy_per_trial   energy per trial, chronological, by class
                        2_fullband_by_class  fullband energy grouped by finger
                        3_band_by_class      band energy grouped by finger

  bands               The same by-class strip for every frequency band, on one
                      common energy axis — where in the spectrum the association
                      lives.

  sorted              Trials sorted by energy and coloured by class, with the AUC
                      and the share of the top-n highest-energy trials that
                      belong to the task. Makes "what the eye sees" and "what the
                      AUC measures" the same picture.

Usage:
    python viz_source_energy.py <subj> <sess> <nclass> <model> <src> [mode] [band] [task]

    python viz_source_energy.py 4 5 3 Orig 80              # trials, Beta
    python viz_source_energy.py 4 5 3 Orig 80 bands        # every band
    python viz_source_energy.py 9 1 3 Orig 15 sorted       # sorted view

Figures -> RESULTS_ROOT/Source Diagnostics/S{XX}/
"""

import os
import sys
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import energy_stats as es
from config import SOURCE_FIGS_DIR


plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 150,
})

BAND_ORDER = ["Fullband", "Delta", "Theta", "Alpha", "Beta", "Gamma"]


def _save(fig, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.basename(save_path)}")


def strip_by_class(energy, classes, fingers, title, ylabel, save_path, ylim=None):
    """Per-finger strip plot of the trial energies, with the median marked."""
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for fi, finger in enumerate(fingers):
        vals = energy[classes == finger]
        ax.scatter(rng.normal(fi, 0.07, len(vals)), vals,
                   c=es.CLASS_COLORS.get(finger, "gray"), s=22, alpha=0.75,
                   edgecolors="white", linewidths=0.3, zorder=3)
        ax.hlines(np.median(vals), fi - 0.28, fi + 0.28,
                  color="black", lw=2.5, zorder=5)
    ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xticks(range(len(fingers)))
    ax.set_xticklabels([f"{f}\n(n={int((classes == f).sum())})" for f in fingers])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.margins(x=0.12)
    ax.legend(handles=[Line2D([0], [0], color="black", lw=2.5, label="median")],
              loc="upper right", framealpha=0.9)
    fig.tight_layout()
    _save(fig, save_path)


def energy_per_trial(energy, classes, fingers, title, save_path, ylim=None):
    """Trial energy in chronological order, coloured by finger."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for finger in fingers:
        idx = np.where(classes == finger)[0]
        ax.scatter(idx, energy[idx], c=es.CLASS_COLORS.get(finger, "gray"),
                   s=22, alpha=0.8, edgecolors="white", linewidths=0.3,
                   label=finger, zorder=3)
    ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Trial (chronological order)")
    ax.set_ylabel(r"Energy  $\Sigma x^2$  (log scale)")
    ax.set_title(title)
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(loc="upper right", ncol=len(fingers), framealpha=0.9)
    fig.tight_layout()
    _save(fig, save_path)


def energy_sorted(energy, classes, fingers, finger, title, save_path):
    """Trials ranked by energy, coloured by class, annotated with the AUC.

    The right-hand end holds the highest-energy trials — the ones the eye picks
    out. The annotation states how many of them actually belong to the finger,
    next to the AUC that summarises the whole ranking.
    """
    order = np.argsort(energy)
    ranked_energy = energy[order]
    ranked_classes = classes[order]
    n_in = int((classes == finger).sum())

    inside = classes == finger
    auc = es.auc_matrix(energy.reshape(-1, 1), classes, [finger])[0, 0]
    top_hits = int((ranked_classes[-n_in:] == finger).sum()) if n_in else 0

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for f in fingers:
        mask = ranked_classes == f
        ax.scatter(np.where(mask)[0], ranked_energy[mask],
                   c=es.CLASS_COLORS.get(f, "gray"), s=22, alpha=0.8,
                   edgecolors="white", linewidths=0.3, label=f, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Trials sorted by energy (ascending)")
    ax.set_ylabel(r"Energy  $\Sigma x^2$  (log scale)")
    ax.set_title(title)
    if n_in:
        ax.axvline(len(energy) - n_in, color="gray", linestyle="--", alpha=0.6)
        ax.text(len(energy) - n_in, ax.get_ylim()[1],
                f"  top {n_in} (highest energy)", va="top", fontsize=8,
                color="dimgray")
    ax.grid(axis="y", which="both", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], color="none",
               label=f"AUC = {auc:.3f}   P(energy higher during {finger})"),
        Line2D([], [], color="none",
               label=f"top {n_in} trials: {top_hits} are {finger} "
                     f"({100 * top_hits / n_in:.0f}%)" if n_in else ""),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9, fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return auc, inside.sum()


def main():
    if len(sys.argv) < 6:
        print(__doc__)
        sys.exit(1)

    subj = int(sys.argv[1])
    sess = int(sys.argv[2])
    nclass = int(sys.argv[3])
    model = sys.argv[4]
    src = int(sys.argv[5])
    mode = sys.argv[6].lower() if len(sys.argv) > 6 else "trials"
    band = sys.argv[7] if len(sys.argv) > 7 else "Beta"
    task = sys.argv[8].upper() if len(sys.argv) > 8 else "MI"

    if mode not in ("trials", "bands", "sorted"):
        print(f"unknown mode {mode!r}; expected trials, bands or sorted")
        sys.exit(1)
    if band not in es.BANDS:
        print(f"unknown band {band!r}; expected one of {sorted(es.BANDS)}")
        sys.exit(1)

    out_dir = os.path.join(SOURCE_FIGS_DIR, f"S{subj:02d}")
    tag = f"S{subj:02d}_src{src}_Sess{sess:02d}_{nclass}class_{model}"
    head = f"S{subj:02d} · {task} · Sess{sess:02d} {nclass}class {model} · source {src}"
    print(f"\n=== {tag}  [{mode}] -> {out_dir} ===")

    E_full, classes = es.trial_energy(subj, sess, nclass, model, task, "Fullband")
    energy_full = E_full[:, src]
    fingers = es.present_fingers(classes, nclass)
    del E_full
    gc.collect()

    if mode == "bands":
        # every band on one shared axis, so the bands are comparable
        per_band = {}
        for b in BAND_ORDER:
            E_b, _ = es.trial_energy(subj, sess, nclass, model, task, b)
            per_band[b] = E_b[:, src]
            del E_b
            gc.collect()
        stacked = np.concatenate([v[v > 0] for v in per_band.values()])
        ylim = (stacked.min() * 0.7, stacked.max() * 1.4)
        print(f"  shared energy axis: {ylim[0]:.2e} .. {ylim[1]:.2e}")
        for b, vals in per_band.items():
            strip_by_class(vals, classes, fingers,
                           f"{head} · {b}", rf"{b}  $\Sigma x^2$",
                           os.path.join(out_dir, f"{tag}_band_{b}.png"), ylim)

    elif mode == "sorted":
        for finger in fingers:
            energy_sorted(energy_full, classes, fingers, finger,
                          f"{head} · Fullband · sorted by energy · {finger}",
                          os.path.join(out_dir, f"{tag}_sorted_{finger}.png"))

    else:  # trials
        E_band, _ = es.trial_energy(subj, sess, nclass, model, task, band)
        energy_band = E_band[:, src]
        del E_band
        gc.collect()

        stacked = np.concatenate([energy_full[energy_full > 0],
                                  energy_band[energy_band > 0]])
        ylim = (stacked.min() * 0.7, stacked.max() * 1.4)
        print(f"  shared energy axis: {ylim[0]:.2e} .. {ylim[1]:.2e}")

        energy_per_trial(energy_full, classes, fingers,
                         f"{head} · Fullband · per trial",
                         os.path.join(out_dir, f"{tag}_1_energy_per_trial.png"), ylim)
        strip_by_class(energy_full, classes, fingers,
                       f"{head} · Fullband · by class",
                       r"Fullband  $\Sigma x^2$",
                       os.path.join(out_dir, f"{tag}_2_fullband_by_class.png"), ylim)
        strip_by_class(energy_band, classes, fingers,
                       f"{head} · {band} · by class",
                       rf"{band}  $\Sigma x^2$",
                       os.path.join(out_dir, f"{tag}_3_{band}_by_class.png"), ylim)

    print(f"\nDone -> {out_dir}")


if __name__ == "__main__":
    main()

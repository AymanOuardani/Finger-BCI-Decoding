"""
corr_distributions.py

Distribution, over the 128 ICA sources, of the per-source energy <-> task
correlation, split Artefact vs Non-Artefact. One image per
(session, model, nClass, finger, artefact definition).

The question the figure answers: is the task-related energy carried by the
artefactual sources (EOG/EMG) or by the neural ones? If the two densities sit on
top of each other, the decoder is not reading artefacts. The separation between
the two groups is quantified on the figure itself — AUC on |r|, Cohen's d and
the Mann-Whitney p-value — and the same numbers are written to Excel by
fill_separation_metrics.py.

Two artefact definitions are plotted when both are available:
    AUTO     EOG/EMG auto-detection on the offline ICA
    MANUEL   the components inspected by hand for that subject

Usage:
    python corr_distributions.py <subj> [metric] [band] [essai] [task]

    python corr_distributions.py 9                     # Pearson, Fullband, AUTO+MANUEL
    python corr_distributions.py 9 pearson Beta        # Beta band
    python corr_distributions.py 9 rankbiserial        # rank-biserial instead
    python corr_distributions.py 9 pearson Fullband auto     # AUTO only

Figures -> RESULTS_ROOT/Correlation Distributions/S{XX}/{Band}/
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
from scipy.stats import gaussian_kde

import energy_stats as es
from config import CORR_DIST_DIR, CORRUPTED_DATA


_COLORS = {
    "Artefact":     ("crimson", "darkred", "firebrick"),
    "Non-Artefact": ("steelblue", "navy", "cornflowerblue"),
}
_DOT_Y = {"Artefact": -0.15, "Non-Artefact": -0.05}


def _separation_handles(values, artifacts, n_components):
    """Legend rows carrying the Artefact-vs-Non-Artefact separation metrics."""
    stats = es.separation(values, artifacts, n_components)
    if not np.isfinite(stats["auc"]):
        return []
    sig = "significatif" if stats["significant"] == "YES" else "non sign."
    return [
        Line2D([], [], color="none", label="—— Séparation |r| (Art vs Non-Art) ——"),
        Line2D([], [], color="none",
               label=f"AUC = {stats['auc']:.3f}     Cohen's d = {stats['cohens_d']:+.2f}"),
        Line2D([], [], color="none", label=f"p = {stats['p_value']:.3g}   ({sig})"),
    ]


def plot_distribution(values, artifacts, n_components, subj, task, sess, nclass,
                      model, finger, metric, band, art_label, save_path):
    metric_label, _fn, (vmin, vmax) = es.METRICS[metric]
    values = np.asarray(values, float)
    art_idx = [c for c in range(n_components) if c in artifacts]
    non_idx = [c for c in range(n_components) if c not in artifacts]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"S{subj:02d} · {task} · {model} · Sess{sess:02d} {nclass}-class · "
        f"{band} · {finger} · {metric_label}",
        fontsize=12, fontweight="bold")
    fig.text(0.5, 0.905, f"Artefacts = {art_label}",
             ha="center", va="top", fontsize=9, color="dimgray")

    pad = 0.05 * (vmax - vmin)
    x_grid = np.linspace(vmin - pad, vmax + pad, 800)
    handles = []

    for group, idx in [("Artefact", art_idx), ("Non-Artefact", non_idx)]:
        col_kde, col_mean, col_med = _COLORS[group]
        dot_y = _DOT_Y[group]
        vals = values[idx]

        if len(vals) < 2:
            ax.scatter(vals, np.full_like(vals, dot_y), color=col_kde,
                       alpha=0.6, s=20, zorder=5)
            handles.append(Line2D([0], [0], marker="o", linestyle="none",
                                  markerfacecolor=col_kde, markersize=5,
                                  label=f"{group}  (n={len(vals)}, too few for KDE)"))
            continue

        density = gaussian_kde(vals)(x_grid)
        ax.plot(x_grid, density, color=col_kde, linewidth=2.0)
        ax.fill_between(x_grid, density, alpha=0.18, color=col_kde)
        ax.axvline(vals.mean(), color=col_mean, linestyle="--", linewidth=1.5)
        ax.axvline(np.median(vals), color=col_med, linestyle=":", linewidth=1.5)
        ax.scatter(vals, np.full_like(vals, dot_y), color=col_kde,
                   alpha=0.55, s=16, zorder=5)
        handles += [
            Line2D([0], [0], color=col_kde, linewidth=2.0,
                   label=f"{group}  (n={len(vals)})"),
            Line2D([0], [0], color=col_mean, linestyle="--", linewidth=1.5,
                   label=f"  mean   = {vals.mean():.3f}"),
            Line2D([0], [0], color=col_med, linestyle=":", linewidth=1.5,
                   label=f"  median = {np.median(vals):.3f}"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=col_kde,
                   markersize=5, alpha=0.6, label="  individual sources"),
        ]

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
    ax.axvline(0 if vmin < 0 else 0.5, color="gray", linewidth=0.8,
               linestyle="--", alpha=0.3)
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_ylim(bottom=min(_DOT_Y.values()) - 0.03)
    ax.set_xlabel(f"{metric_label}  (energy ↔ task)")
    ax.set_ylabel("Density")
    handles += _separation_handles(values, artifacts, n_components)
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(axis="y", alpha=0.22)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    saved: {os.path.basename(save_path)}", flush=True)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    subj = int(sys.argv[1])
    metric = sys.argv[2].lower() if len(sys.argv) > 2 else "pearson"
    band = sys.argv[3] if len(sys.argv) > 3 else "Fullband"
    essai = sys.argv[4].lower() if len(sys.argv) > 4 else "all"
    task = sys.argv[5].upper() if len(sys.argv) > 5 else "MI"

    if metric not in es.METRICS:
        print(f"unknown metric {metric!r}; expected one of {sorted(es.METRICS)}")
        sys.exit(1)
    if band not in es.BANDS:
        print(f"unknown band {band!r}; expected one of {sorted(es.BANDS)}")
        sys.exit(1)

    splits = es.artifact_splits(subj, task, essai)
    if not splits:
        print(f"S{subj:02d}: nothing to do for essai '{essai}' "
              f"(no manual selection configured?)")
        return
    for tag, (label, art) in splits.items():
        print(f"S{subj:02d}: {tag} artefacts = {len(art)}  ({label})")

    out_dir = os.path.join(CORR_DIST_DIR, f"S{subj:02d}", band)
    groups = es.available_recordings(subj, task)
    n_saved = 0

    for (sess, nclass), models in sorted(groups.items()):
        for model in models:
            if (subj, task, sess, nclass, model) in CORRUPTED_DATA:
                continue
            print(f"\n=== S{subj:02d} Sess{sess:02d} {model} {nclass}class ===", flush=True)
            try:
                E, classes = es.trial_energy(subj, sess, nclass, model, task, band)
            except Exception as exc:
                print(f"  [SKIP] {exc}", flush=True)
                continue

            n_comp = E.shape[1]
            fingers = es.present_fingers(classes, nclass)
            print(f"  n_trials={len(classes)} ncomp={n_comp}", flush=True)
            matrix = es.correlation_matrix(E, classes, fingers, metric)
            tag_model = es.MODEL_TAG[model]

            for fi, finger in enumerate(fingers):
                for tag, (label, artifacts) in splits.items():
                    name = (f"dist_{metric}_S{subj:02d}_{task}_Sess{sess:02d}_"
                            f"{nclass}class_{tag_model}_{band}_{finger}_{tag}.png")
                    plot_distribution(matrix[fi], artifacts, n_comp, subj, task,
                                      sess, nclass, model, finger, metric, band,
                                      label, os.path.join(out_dir, name))
                    n_saved += 1

            del E
            gc.collect()

    print(f"\nS{subj:02d} done. {n_saved} figure(s) -> {out_dir}")


if __name__ == "__main__":
    main()

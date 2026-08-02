"""_gen_rb_dist.py — TEMP. Rank-biserial correlation DISTRIBUTION plots for S09.

Same idea as Analysis/Distribution_Plot_Correlations.py (distribution over the
128 sources of the per-source association value, split Artefact vs Non-Artefact,
one image per finger task), but the metric is the RANK-BISERIAL r_rb = 2*AUC-1
(Fullband), computed directly (not read from the ODS Pearson sheet).

Output: C:/Users/aymen/Desktop/S09_Maps_RankBiserial (same folder as the maps).
"""
import os, sys, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

import fill_energy_table as fet
import _gen_auc_maps as g            # reuse fullband_energy / auc_matrix / offline_artifacts
from config import TASK_LABELS

OUT = "C:/Users/aymen/Desktop/S09_Maps_RankBiserial"

_COLORS = {
    "Artefact":     ("crimson",   "darkred",       "firebrick"),
    "Non-Artefact": ("steelblue", "navy",          "cornflowerblue"),
}
_DOT_Y = {"Artefact": -0.15, "Non-Artefact": -0.05}


def plot_dist(vals_art, vals_non, finger, sess, ncl, model, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"S{g.SUBJ:02d} · {g.TASK} · {model} · Sess{sess:02d} {ncl}-class · "
        f"Fullband · {finger} · Rank-biserial",
        fontsize=12, fontweight="bold")

    x_grid = np.linspace(-1.05, 1.05, 800)
    handles = []
    for grp_label, vals in [("Artefact", vals_art), ("Non-Artefact", vals_non)]:
        col_kde, col_mean, col_med = _COLORS[grp_label]
        dot_y = _DOT_Y[grp_label]
        vals = np.asarray(vals, float)
        if len(vals) < 2:
            ax.scatter(vals, np.full_like(vals, dot_y), color=col_kde,
                       alpha=0.6, s=20, zorder=5)
            handles.append(Line2D([0], [0], marker="o", linestyle="none",
                           markerfacecolor=col_kde, markersize=5,
                           label=f"{grp_label}  (n={len(vals)}, too few for KDE)"))
            continue
        kde = gaussian_kde(vals)
        y = kde(x_grid)
        ax.plot(x_grid, y, color=col_kde, linewidth=2.0)
        ax.fill_between(x_grid, y, alpha=0.18, color=col_kde)
        ax.axvline(vals.mean(),     color=col_mean, linestyle="--", linewidth=1.5)
        ax.axvline(np.median(vals), color=col_med,  linestyle=":",  linewidth=1.5)
        ax.scatter(vals, np.full_like(vals, dot_y), color=col_kde,
                   alpha=0.55, s=16, zorder=5)
        handles += [
            Line2D([0], [0], color=col_kde, linewidth=2.0,
                   label=f"{grp_label}  (n={len(vals)})"),
            Line2D([0], [0], color=col_mean, linestyle="--", linewidth=1.5,
                   label=f"  mean   = {vals.mean():.3f}"),
            Line2D([0], [0], color=col_med,  linestyle=":",  linewidth=1.5,
                   label=f"  median = {np.median(vals):.3f}"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=col_kde,
                   markersize=5, alpha=0.6, label="  individual sources"),
        ]

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(bottom=min(_DOT_Y.values()) - 0.03)
    ax.set_xlabel("Rank-biserial   r_rb = 2·AUC − 1")
    ax.set_ylabel("Density")
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(axis="y", alpha=0.22)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    groups = fet._detect_groups(g.SUBJ, g.TASK)
    print(f"S{g.SUBJ:02d}: {len(groups)} group(s) -> {OUT}")
    art = g.offline_artifacts()
    print(f"Offline artifact sources ({len(art)})")

    n_saved = 0
    for (sess, ncl), models in sorted(groups.items()):
        for model in models:
            print(f"  Sess{sess:02d} {ncl}class {model} ...", flush=True)
            try:
                E, classes = g.fullband_energy(sess, ncl, model)
            except Exception as e:
                print(f"    [SKIP] {e}", flush=True)
                continue
            tasks = [t for t in TASK_LABELS if t in set(classes)]
            rb = 2.0 * g.auc_matrix(E, classes, tasks) - 1.0      # (n_tasks, n_comp)
            ncomp = rb.shape[1]
            art_idx = [c for c in range(ncomp) if c in art]
            non_idx = [c for c in range(ncomp) if c not in art]
            tag = "1_Orig" if model == "Orig" else "2_Finetune"
            for ti, finger in enumerate(tasks):
                save = os.path.join(
                    OUT, f"DistRB_S{g.SUBJ:02d}_{g.TASK}_Sess{sess:02d}_"
                    f"{ncl}class_{tag}_Fullband_{finger}.png")
                plot_dist(rb[ti, art_idx], rb[ti, non_idx],
                          finger, sess, ncl, model, save)
                n_saved += 1
            print(f"    saved {len(tasks)} task plot(s)", flush=True)
            del E
            gc.collect()
    print(f"\nDone. {n_saved} distribution plot(s) -> {OUT}")


if __name__ == "__main__":
    main()

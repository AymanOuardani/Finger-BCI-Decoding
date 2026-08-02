"""_gen_pearson_iqr_dist.py — TEMP. Pearson (0.5xIQR-cleaned) DISTRIBUTION plots.

Same layout as _gen_rb_dist.py / Analysis/Distribution_Plot_Correlations.py
(distribution over the 128 ICA sources of the per-source correlation, split
Artefact vs Non-Artefact, one image per finger task), BUT:

  * metric = SIGNED Pearson r in [-1, 1] between a source's per-trial Fullband
    energy and the finger's one-vs-rest membership;
  * per source, OUTLIER TRIALS are removed first with a 0.5xIQR Tukey fence on
    that source's energy (keep E in [Q1 - 0.5*IQR, Q3 + 0.5*IQR]), THEN Pearson
    is computed on the surviving trials.

Only S09 / MI / Sess01 / 3-class / Orig / Fullband. Three fingers: Pinky,
Thumb, Index. Output PNGs saved directly to the Desktop.
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

import _gen_auc_maps as g            # reuse fullband_energy / offline_artifacts / SUBJ / TASK

# CLI: <subj> <sess> <nclass> <model>   (defaults: S09 Sess01 3class Orig)
SUBJ  = int(sys.argv[1]) if len(sys.argv) > 1 else 9
SESS  = int(sys.argv[2]) if len(sys.argv) > 2 else 1
NCL   = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MODEL = sys.argv[4] if len(sys.argv) > 4 else "Orig"
g.SUBJ = SUBJ                          # fullband_energy / offline_artifacts read this

OUT = f"C:/Users/aymen/Desktop/S{SUBJ:02d}_Distibution"
IQR_K = 0.5
FINGERS = ["Pinky", "Thumb", "Index"]

# Known presentation values (3-class, Fullband, 0.5xIQR) for the anomalous source
# of each subject -> used as a sanity check that the energy/IQR pipeline matches.
VALID = {
    9: (15, {"Thumb": -0.7120, "Index": +0.1984, "Pinky": +0.5265}),
    4: (80, {"Thumb": +0.8993, "Index": -0.4318, "Pinky": -0.4482}),
}

_COLORS = {
    "Artefact":     ("crimson",   "darkred",       "firebrick"),
    "Non-Artefact": ("steelblue", "navy",          "cornflowerblue"),
}
_DOT_Y = {"Artefact": -0.15, "Non-Artefact": -0.05}


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def iqr_keep_mask(x, k=IQR_K):
    """Tukey fence with multiplier k: keep x in [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x >= q1 - k * iqr) & (x <= q3 + k * iqr)


def cleaned_pearson_per_source(E, classes, finger):
    """For every source: drop 0.5xIQR outlier trials (on that source's energy),
    then signed Pearson(energy, one-vs-rest membership) over the survivors."""
    ncomp = E.shape[1]
    r = np.empty(ncomp)
    for c in range(ncomp):
        keep = iqr_keep_mask(E[:, c])
        m = (classes[keep] == finger).astype(float)
        r[c] = pearson(E[keep, c], m)
    return r


def plot_dist(vals_art, vals_non, finger, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"S{g.SUBJ:02d} · {g.TASK} · {MODEL} · Sess{SESS:02d} "
        f"{NCL}-class · Fullband · {finger} · "
        f"Pearson (0.5×IQR-cleaned)",
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
    ax.set_xlabel("Signed Pearson r  (energy ↔ task, 0.5×IQR-cleaned)")
    ax.set_ylabel("Density")
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(axis="y", alpha=0.22)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {save_path}", flush=True)


def main():
    art = g.offline_artifacts()
    print(f"Offline artifact sources ({len(art)}): {sorted(art)}", flush=True)

    print(f"Loading S{g.SUBJ:02d} {g.TASK} Sess{SESS:02d} {NCL}class {MODEL} ...",
          flush=True)
    E, classes = g.fullband_energy(SESS, NCL, MODEL)
    ncomp = E.shape[1]
    print(f"  n_trials={len(classes)}  ncomp={ncomp}", flush=True)

    # --- validation against the presentation's known values (if available) ---
    if SUBJ in VALID and NCL == 3:
        ref_src, expected = VALID[SUBJ]
        exp_str = ", ".join(f"{k} {v:+.4f}" for k, v in expected.items())
        print(f"\nValidation S{SUBJ:02d} src{ref_src} (3-class, Fullband, 0.5xIQR) "
              f"[presentation: {exp_str}]:", flush=True)
        keepr = iqr_keep_mask(E[:, ref_src])
        for f in ["Thumb", "Index", "Pinky"]:
            if f not in set(classes):
                continue
            r = pearson(E[keepr, ref_src], (classes[keepr] == f).astype(float))
            print(f"  src{ref_src} {f:6s}: r={r:+.4f}  "
                  f"(kept {int(keepr.sum())}/{len(classes)} trials)", flush=True)

    art_idx = [c for c in range(ncomp) if c in art]
    non_idx = [c for c in range(ncomp) if c not in art]

    print("", flush=True)
    for finger in FINGERS:
        r = cleaned_pearson_per_source(E, classes, finger)
        save = os.path.join(
            OUT, f"DistPearson05IQR_S{g.SUBJ:02d}_{g.TASK}_Sess{SESS:02d}_"
            f"{NCL}class_1_{MODEL}_Fullband_{finger}.png")
        plot_dist(r[art_idx], r[non_idx], finger, save)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

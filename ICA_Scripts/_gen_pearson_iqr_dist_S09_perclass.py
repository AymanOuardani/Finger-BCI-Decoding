"""_gen_pearson_iqr_dist_S09_perclass.py — TEMP.

Pearson (0.5xIQR-cleaned, PER-CLASS) energy<->task correlation DISTRIBUTION
plots for S09 / MI / Sess01 / Orig / Fullband, over the 128 ICA sources, split
Artefact vs Non-Artefact. Generates BOTH 2-class and 3-class.

Artefact definition: AUTO only = offline EOG/EMG auto-detection (offline ICA).
(No manually-confirmed exclusion list exists for S09.)

Outlier removal = PER-CLASS 0.5xIQR (Tukey fence computed WITHIN each finger
class on that source's energy), then signed Pearson(energy, one-vs-rest) over
the survivors — same method validated on S04.

Figures -> C:/Users/aymen/Desktop/S09 :
  DistPearson05IQR_S09_MI_Sess01_{n}class_1_Orig_Fullband_{finger}_AUTO.png
"""
import os, sys
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

import _gen_auc_maps as g   # reuse fullband_energy / offline_artifacts / SUBJ / TASK

SUBJ, SESS, MODEL = 9, 1, "Orig"
g.SUBJ = SUBJ
OUT = "C:/Users/aymen/Desktop/S09"
IQR_K = 0.5
NCLASSES = [2, 3]
FINGERS_BY_NCL = {2: ["Thumb", "Pinky"], 3: ["Thumb", "Index", "Pinky"]}

# Sanity values (src15). Computed = per-class method; presentation = target table.
VALID_SRC = 15
VALID_PRES = {
    2: {"Thumb": -0.6553, "Pinky": +0.6553},
    3: {"Thumb": -0.7120, "Index": +0.1984, "Pinky": +0.5265},
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


def per_class_keep(e, classes):
    """Per-class 0.5xIQR keep mask: clean energy WITHIN each finger class."""
    keep = np.zeros(len(e), bool)
    for cl in np.unique(classes):
        idx = np.where(classes == cl)[0]
        keep[idx[iqr_keep_mask(e[idx])]] = True
    return keep


def cleaned_pearson_all_fingers(E, classes, fingers):
    """Per source: per-class 0.5xIQR outlier removal, then signed Pearson with
    each finger's one-vs-rest membership over the survivors."""
    ncomp = E.shape[1]
    R = {f: np.empty(ncomp) for f in fingers}
    for c in range(ncomp):
        keep = per_class_keep(E[:, c], classes)
        cc, ec = classes[keep], E[keep, c]
        for f in fingers:
            R[f][c] = pearson(ec, (cc == f).astype(float))
    return R


def plot_dist(vals_art, vals_non, finger, ncl, save_path, art_def):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"S{g.SUBJ:02d} · {g.TASK} · {MODEL} · Sess{SESS:02d} "
        f"{ncl}-class · Fullband · {finger} · "
        f"Pearson (0.5×IQR-cleaned)",
        fontsize=12, fontweight="bold")
    fig.text(0.5, 0.905, f"Artefacts = {art_def}",
             ha="center", va="top", fontsize=9, color="dimgray")

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
    ax.set_xlabel("Signed Pearson r  (energy ↔ task, 0.5×IQR-cleaned, per-class)")
    ax.set_ylabel("Density")
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(axis="y", alpha=0.22)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.basename(save_path)}", flush=True)


def main():
    art_auto = g.offline_artifacts()
    print(f"AUTO artefacts (offline EOG/EMG): {len(art_auto)} -> {sorted(art_auto)}",
          flush=True)

    art_def = "auto offline ICA  (EOG/EMG)"
    tag = "AUTO"

    for ncl in NCLASSES:
        fingers = FINGERS_BY_NCL[ncl]
        print(f"\n===== {ncl}-class =====", flush=True)
        E, classes = g.fullband_energy(SESS, ncl, MODEL)
        ncomp = E.shape[1]
        print(f"  n_trials={len(classes)} ncomp={ncomp} "
              f"classes={sorted(set(classes))}", flush=True)

        # validation: per-class src15 vs presentation table
        keep15 = per_class_keep(E[:, VALID_SRC], classes)
        print(f"  validation src{VALID_SRC} (per-class 0.5xIQR, "
              f"kept {int(keep15.sum())}/{len(classes)}):", flush=True)
        for f in fingers:
            r = pearson(E[keep15, VALID_SRC], (classes[keep15] == f).astype(float))
            t = VALID_PRES[ncl][f]
            print(f"    {f:6s}: r={r:+.4f}  (presentation {t:+.4f}, "
                  f"diff={r - t:+.4f})", flush=True)

        R = cleaned_pearson_all_fingers(E, classes, fingers)
        art_idx = [c for c in range(ncomp) if c in art_auto]
        non_idx = [c for c in range(ncomp) if c not in art_auto]
        for finger in fingers:
            r = R[finger]
            save = os.path.join(
                OUT, f"DistPearson05IQR_S{g.SUBJ:02d}_{g.TASK}_Sess{SESS:02d}_"
                f"{ncl}class_1_{MODEL}_Fullband_{finger}_{tag}.png")
            plot_dist(r[art_idx], r[non_idx], finger, ncl, save, art_def)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

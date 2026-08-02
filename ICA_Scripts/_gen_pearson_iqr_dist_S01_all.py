"""_gen_pearson_iqr_dist_S01_all.py — TEMP.

Per-class 0.5xIQR Pearson energy<->task correlation DISTRIBUTION plots for S01 /
MI, ALL available online sessions and BOTH models. Distribution over the 128 ICA
sources, split Artefact vs Non-Artefact, one PNG per (session, model, nclass,
finger, essai).

Artefact partition is FIXED across sessions/models (offline ICA shared):
  * AUTO   : offline EOG/EMG auto-detection (offline ICA).
  * MANUEL : every component NOT in the user-provided KEPT list (29 kept -> 99 art).

Output: flat staging dir (organised into the Sujet01 tree afterwards):
  DistPearson05IQR_S01_MI_Sess{SS}_{n}class_{1_Orig|2_Finetune}_Fullband_{finger}_{AUTO|MANUEL}.png
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

import _gen_auc_maps as g

SUBJ = 1
g.SUBJ = SUBJ
OUT = "C:/Users/aymen/Desktop/_S01_staging"
IQR_K = 0.5
SESSIONS = [1, 2]
MODELS = ["Orig", "Finetune"]
NCLASSES = [2, 3]
FINGERS_BY_NCL = {2: ["Thumb", "Pinky"], 3: ["Thumb", "Index", "Pinky"]}

KEPT = [5, 6, 7, 8, 9, 17, 20, 21, 24, 25, 53, 54, 65, 66, 69, 73, 74, 80, 81,
        85, 87, 90, 91, 96, 98, 99, 100, 105, 108]

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
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x >= q1 - k * iqr) & (x <= q3 + k * iqr)


def per_class_keep(e, classes):
    keep = np.zeros(len(e), bool)
    for cl in np.unique(classes):
        idx = np.where(classes == cl)[0]
        keep[idx[iqr_keep_mask(e[idx])]] = True
    return keep


def cleaned_pearson_all_fingers(E, classes, fingers):
    ncomp = E.shape[1]
    R = {f: np.empty(ncomp) for f in fingers}
    for c in range(ncomp):
        keep = per_class_keep(E[:, c], classes)
        cc, ec = classes[keep], E[keep, c]
        for f in fingers:
            R[f][c] = pearson(ec, (cc == f).astype(float))
    return R


def plot_dist(vals_art, vals_non, finger, ncl, sess, model, save_path, art_def):
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"S{g.SUBJ:02d} · {g.TASK} · {model} · Sess{sess:02d} "
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
    os.makedirs(OUT, exist_ok=True)
    kept = set(KEPT)
    manual_art = set(range(128)) - kept
    art_auto = g.offline_artifacts()
    print(f"MANUEL artefacts: {len(manual_art)} excluded / {len(kept)} kept", flush=True)
    print(f"AUTO artefacts (offline EOG/EMG): {len(art_auto)}", flush=True)

    splits = {
        "auto offline ICA  (EOG/EMG)":           ("AUTO",   art_auto),
        f"manual selection  ({len(kept)} kept)": ("MANUEL", manual_art),
    }

    for sess in SESSIONS:
        for model in MODELS:
            mtag = "1_Orig" if model == "Orig" else "2_Finetune"
            for ncl in NCLASSES:
                fingers = FINGERS_BY_NCL[ncl]
                print(f"\n=== Sess{sess:02d} {model} {ncl}class ===", flush=True)
                try:
                    E, classes = g.fullband_energy(sess, ncl, model)
                except Exception as exc:
                    print(f"  [SKIP] {exc}", flush=True)
                    continue
                ncomp = E.shape[1]
                print(f"  n_trials={len(classes)} ncomp={ncomp}", flush=True)
                R = cleaned_pearson_all_fingers(E, classes, fingers)
                for finger in fingers:
                    r = R[finger]
                    for art_def, (tag, art_set) in splits.items():
                        art_idx = [c for c in range(ncomp) if c in art_set]
                        non_idx = [c for c in range(ncomp) if c not in art_set]
                        save = os.path.join(
                            OUT, f"DistPearson05IQR_S{g.SUBJ:02d}_{g.TASK}_"
                            f"Sess{sess:02d}_{ncl}class_{mtag}_Fullband_"
                            f"{finger}_{tag}.png")
                        plot_dist(r[art_idx], r[non_idx], finger, ncl, sess,
                                  model, save, art_def)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()

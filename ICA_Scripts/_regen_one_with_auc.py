"""_regen_one_with_auc.py — TEMP. Regenerate ONE distribution figure WITH the
separation metric box (AUC on |r|, Cohen's d, Mann-Whitney p) drawn on it.

Saves IN PLACE into the organised Sujet tree (overwrites the existing PNG).

Usage: python _regen_one_with_auc.py <subj> <sess> <ncl> <model> <finger> <AUTO|MANUEL>
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
from scipy.stats import gaussian_kde, mannwhitneyu
import _gen_auc_maps as g

KEPT = {
    1: [5,6,7,8,9,17,20,21,24,25,53,54,65,66,69,73,74,80,81,85,87,90,91,96,98,99,100,105,108],
    2: [3,5,7,10,23,48,52,53,64,69,77,83,84,85,86,88,89,90,91,4,8,21,42],
    9: [7,17,20,23,26,30,33,34,44,45,49,50,55,59,63,66,67,69,70,82,83,84,85,86,91,93,97,105,107],
}
EXCLUDED = {
    4: [0,1,2,3,4,5,6,7,8,9,10,11,15,18,21,22,23,24,29,30,31,32,33,34,35,36,37,38,39,40,41,
        42,43,44,45,46,47,48,49,50,51,53,54,55,57,59,60,62,63,64,65,67,68,69,70,71,74,75,77,
        79,80,81,85,86,88,89,90,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,
        109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127],
}
ROOT = "C:/Users/aymen/Desktop/Corrélations_IQR_Manuel_Auto"
_COLORS = {"Artefact": ("crimson","darkred","firebrick"),
           "Non-Artefact": ("steelblue","navy","cornflowerblue")}
_DOT_Y = {"Artefact": -0.15, "Non-Artefact": -0.05}


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def iqr_keep_mask(x, k=0.5):
    q1, q3 = np.percentile(x, [25, 75]); iqr = q3 - q1
    return (x >= q1 - k*iqr) & (x <= q3 + k*iqr)


def per_class_keep(e, classes):
    keep = np.zeros(len(e), bool)
    for cl in np.unique(classes):
        idx = np.where(classes == cl)[0]
        keep[idx[iqr_keep_mask(e[idx])]] = True
    return keep


def cleaned_r(E, classes, finger):
    ncomp = E.shape[1]; r = np.empty(ncomp)
    for c in range(ncomp):
        keep = per_class_keep(E[:, c], classes)
        r[c] = pearson(E[keep, c], (classes[keep] == finger).astype(float))
    return r


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / sp if sp > 0 else 0.0


def main():
    subj, sess, ncl = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    model, finger, essai = sys.argv[4], sys.argv[5], sys.argv[6].upper()
    g.SUBJ = subj
    art_def = ("auto offline ICA  (EOG/EMG)" if essai == "AUTO"
               else f"manual selection")
    if essai == "AUTO":
        art = g.offline_artifacts()
    elif subj in KEPT:
        art = set(range(128)) - set(KEPT[subj])
    else:
        art = set(EXCLUDED[subj])

    E, classes = g.fullband_energy(sess, ncl, model)
    ncomp = E.shape[1]
    r = cleaned_r(E, classes, finger)
    art_idx = [c for c in range(ncomp) if c in art]
    non_idx = [c for c in range(ncomp) if c not in art]
    r_art, r_non = r[art_idx], r[non_idx]
    a_art, a_non = np.abs(r_art), np.abs(r_non)

    U, p = mannwhitneyu(a_art, a_non, alternative="two-sided")
    auc = U / (len(a_art) * len(a_non))
    d = cohens_d(a_art, a_non)
    sig = "significatif" if p < 0.05 else "non sign."
    print(f"AUC={auc:.4f}  d={d:+.4f}  p={p:.3g} ({sig})", flush=True)

    # ── plot ──
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"S{subj:02d} · {g.TASK} · {model} · Sess{sess:02d} "
        f"{ncl}-class · Fullband · {finger} · Pearson (0.5×IQR-cleaned)",
        fontsize=12, fontweight="bold")
    fig.text(0.5, 0.905, f"Artefacts = {art_def}",
             ha="center", va="top", fontsize=9, color="dimgray")

    x_grid = np.linspace(-1.05, 1.05, 800)
    handles = []
    for grp, vals in [("Artefact", r_art), ("Non-Artefact", r_non)]:
        ck, cm, cmd = _COLORS[grp]; dy = _DOT_Y[grp]
        vals = np.asarray(vals, float)
        kde = gaussian_kde(vals); y = kde(x_grid)
        ax.plot(x_grid, y, color=ck, linewidth=2.0)
        ax.fill_between(x_grid, y, alpha=0.18, color=ck)
        ax.axvline(vals.mean(), color=cm, linestyle="--", linewidth=1.5)
        ax.axvline(np.median(vals), color=cmd, linestyle=":", linewidth=1.5)
        ax.scatter(vals, np.full_like(vals, dy), color=ck, alpha=0.55, s=16, zorder=5)
        handles += [
            Line2D([0],[0], color=ck, linewidth=2.0, label=f"{grp}  (n={len(vals)})"),
            Line2D([0],[0], color=cm, linestyle="--", linewidth=1.5, label=f"  mean   = {vals.mean():.3f}"),
            Line2D([0],[0], color=cmd, linestyle=":", linewidth=1.5, label=f"  median = {np.median(vals):.3f}"),
            Line2D([0],[0], marker="o", linestyle="none", markerfacecolor=ck, markersize=5, alpha=0.6, label="  individual sources"),
        ]

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(bottom=min(_DOT_Y.values()) - 0.03)
    ax.set_xlabel("Signed Pearson r  (energy ↔ task, 0.5×IQR-cleaned, per-class)")
    ax.set_ylabel("Density")
    # separation metrics added as legend rows (no separate box)
    handles += [
        Line2D([], [], color="none", label="—— Séparation |r| (Art vs Non-Art) ——"),
        Line2D([], [], color="none", label=f"AUC = {auc:.3f}     Cohen's d = {d:+.2f}"),
        Line2D([], [], color="none", label=f"p = {p:.3g}   ({sig})"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(axis="y", alpha=0.22)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    essai_folder = "Auto" if essai == "AUTO" else "Manuel"
    mtag = "1_Orig" if model == "Orig" else "2_Finetune"
    fname = (f"DistPearson05IQR_S{subj:02d}_{g.TASK}_Sess{sess:02d}_{ncl}class_"
             f"{mtag}_Fullband_{finger}_{essai}.png")
    save = os.path.join(ROOT, f"Sujet{subj:02d}", f"{ncl}Class", essai_folder, finger, fname)
    os.makedirs(os.path.dirname(save), exist_ok=True)
    fig.savefig(save, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {save}", flush=True)


if __name__ == "__main__":
    main()

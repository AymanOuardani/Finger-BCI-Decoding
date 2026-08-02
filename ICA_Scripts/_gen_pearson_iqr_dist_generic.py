"""_gen_pearson_iqr_dist_generic.py — TEMP.

Per-class 0.5xIQR Pearson energy<->task correlation DISTRIBUTION plots for one
subject (MI), ALL available online sessions (Sess01..05) and BOTH models
(Orig/Finetune). Distribution over the 128 ICA sources, split Artefact vs
Non-Artefact, one PNG per (session, model, nclass, finger, essai).

Artefact partition is FIXED across sessions/models (offline ICA shared):
  * AUTO   : offline EOG/EMG auto-detection (offline ICA).
  * MANUEL : user-defined component selection (subject-level). Provided either as
             a KEPT list (artefact = complement) or an EXCLUDED list.

Usage:  python _gen_pearson_iqr_dist_generic.py <subj>
Output: flat per-subject staging dir  C:/Users/aymen/Desktop/_S{XX}_staging
  DistPearson05IQR_S{XX}_MI_Sess{SS}_{n}class_{1_Orig|2_Finetune}_Fullband_{finger}_{AUTO|MANUEL}.png
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

# ── per-subject manual selection (subject level) ──────────────────────────────
# "kept": artefact = 128-complement.  "excluded": artefact = these.  None = AUTO only.
S04_EXCLUDED = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 18, 21, 22, 23, 24, 29, 30, 31, 32,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
    53, 54, 55, 57, 59, 60, 62, 63, 64, 65, 67, 68, 69, 70, 71, 74, 75, 77, 79,
    80, 81, 85, 86, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102,
    103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117,
    118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
]
S01_KEPT = [5, 6, 7, 8, 9, 17, 20, 21, 24, 25, 53, 54, 65, 66, 69, 73, 74, 80, 81,
            85, 87, 90, 91, 96, 98, 99, 100, 105, 108]
S02_KEPT = [3, 5, 7, 10, 23, 48, 52, 53, 64, 69, 77, 83, 84, 85, 86, 88, 89, 90, 91,
            4, 8, 21, 42]
S09_KEPT = [7, 17, 20, 23, 26, 30, 33, 34, 44, 45, 49, 50, 55, 59, 63, 66, 67, 69,
            70, 82, 83, 84, 85, 86, 91, 93, 97, 105, 107]
S03_KEPT = [2, 4, 6, 7, 8, 10, 11, 13, 14, 15, 18, 25, 29, 32, 33, 40, 42, 44, 46,
            50, 54, 55, 56, 61, 62, 63, 67, 72, 74, 75, 77, 79, 80, 84, 87, 89, 93,
            95, 96, 98, 102, 106, 113]
S06_KEPT = [3, 4, 6, 8, 14, 15, 18, 23, 24, 26, 35, 38, 41, 43, 46, 49, 52, 53, 58,
            59, 63, 71, 78, 81, 82, 83, 85, 88, 89, 92, 93, 94, 95, 100, 101, 103,
            104, 105, 107, 108]

SUBJ_CONFIG = {
    1: {"kept": S01_KEPT},          # AUTO + MANUEL
    2: {"kept": S02_KEPT},          # AUTO + MANUEL
    3: {"kept": S03_KEPT},          # AUTO + MANUEL
    4: {"excluded": S04_EXCLUDED},  # AUTO + MANUEL
    6: {"kept": S06_KEPT},          # AUTO + MANUEL
    9: {"kept": S09_KEPT},          # AUTO + MANUEL
}

SESSIONS = [1, 2, 3, 4, 5]
MODELS = ["Orig", "Finetune"]
NCLASSES = [2, 3]
FINGERS_BY_NCL = {2: ["Thumb", "Pinky"], 3: ["Thumb", "Index", "Pinky"]}
IQR_K = 0.5

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


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    sp = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / sp if sp > 0 else 0.0


def sep_metric_handles(vals_art, vals_non):
    """Legend rows for Artefact-vs-Non-Artefact separation (AUC on |r|, d, p)."""
    a, b = np.abs(np.asarray(vals_art, float)), np.abs(np.asarray(vals_non, float))
    if len(a) < 2 or len(b) < 2:
        return []
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    auc = U / (len(a) * len(b))
    d = cohens_d(a, b)
    sig = "significatif" if p < 0.05 else "non sign."
    return [
        Line2D([], [], color="none", label="—— Séparation |r| (Art vs Non-Art) ——"),
        Line2D([], [], color="none", label=f"AUC = {auc:.3f}     Cohen's d = {d:+.2f}"),
        Line2D([], [], color="none", label=f"p = {p:.3g}   ({sig})"),
    ]


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
    handles += sep_metric_handles(vals_art, vals_non)
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(axis="y", alpha=0.22)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {os.path.basename(save_path)}", flush=True)


def main():
    subj = int(sys.argv[1])
    # optional essai filter: "all" (default) | "auto" | "manuel"
    essai_filter = sys.argv[2].lower() if len(sys.argv) > 2 else "all"
    g.SUBJ = subj
    cfg = SUBJ_CONFIG.get(subj, {})
    out = f"C:/Users/aymen/Desktop/_S{subj:02d}_staging"
    os.makedirs(out, exist_ok=True)

    # Build the essai splits (AUTO always; MANUEL only if a manual list is configured),
    # honouring the essai_filter so we can (re)generate just one essai.
    splits = {}
    if essai_filter in ("all", "auto"):
        art_auto = g.offline_artifacts()
        splits["auto offline ICA  (EOG/EMG)"] = ("AUTO", art_auto)
        print(f"S{subj:02d}: AUTO artefacts = {len(art_auto)}", flush=True)
    if essai_filter in ("all", "manuel"):
        if "kept" in cfg:
            kept = set(cfg["kept"]); manual = set(range(128)) - kept
            splits[f"manual selection  ({len(kept)} kept)"] = ("MANUEL", manual)
            print(f"S{subj:02d}: MANUEL artefacts = {len(manual)} (kept {len(kept)})", flush=True)
        elif "excluded" in cfg:
            manual = set(cfg["excluded"])
            splits[f"manual selection  ({128 - len(manual)} kept)"] = ("MANUEL", manual)
            print(f"S{subj:02d}: MANUEL artefacts = {len(manual)} (kept {128 - len(manual)})", flush=True)
        else:
            print(f"S{subj:02d}: no manual list configured", flush=True)
    if not splits:
        print(f"S{subj:02d}: nothing to do for filter '{essai_filter}'", flush=True)
        return

    for sess in SESSIONS:
        for model in MODELS:
            mtag = "1_Orig" if model == "Orig" else "2_Finetune"
            for ncl in NCLASSES:
                fingers = FINGERS_BY_NCL[ncl]
                print(f"\n=== S{subj:02d} Sess{sess:02d} {model} {ncl}class ===", flush=True)
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
                            out, f"DistPearson05IQR_S{subj:02d}_{g.TASK}_"
                            f"Sess{sess:02d}_{ncl}class_{mtag}_Fullband_"
                            f"{finger}_{tag}.png")
                        plot_dist(r[art_idx], r[non_idx], finger, ncl, sess,
                                  model, save, art_def)

    print(f"\nS{subj:02d} done.", flush=True)


if __name__ == "__main__":
    main()

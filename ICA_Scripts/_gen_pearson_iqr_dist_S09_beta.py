"""_gen_pearson_iqr_dist_S09_beta.py — TEMP.

Same as _gen_pearson_iqr_dist_generic.py but the per-source energy is computed on
the BETA-band (13-30 Hz) filtered ICA source instead of Fullband. For S09 / MI,
all sessions × {Orig, Finetune}, AUTO + MANUEL essais.

Beta energy: load raw -> notch -> offline ICA sources -> band-pass 13-30 Hz (all
sources) -> per-trial sum of squares. (Same recipe as _gen_band_figs.py.)

Output staging -> C:/Users/aymen/Desktop/_S09_beta_staging :
  DistPearson05IQR_S09_MI_Sess{SS}_{n}class_{1_Orig|2_Finetune}_Beta_{finger}_{AUTO|MANUEL}.png
"""
import os, sys, glob
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

import _gen_auc_maps as g   # reuse build_raw/get_folder/_load_offline_ica/_chronological_trials

SUBJ, TASK = 9, "MI"
g.SUBJ = SUBJ
OUT = "C:/Users/aymen/Desktop/_S09_beta_staging"
BAND = ("Beta", 13.0, 30.0)
IQR_K = 0.5
SESSIONS = [1, 2, 3, 4, 5]
MODELS = ["Orig", "Finetune"]
NCLASSES = [2, 3]
FINGERS_BY_NCL = {2: ["Thumb", "Pinky"], 3: ["Thumb", "Index", "Pinky"]}

S09_KEPT = [7, 17, 20, 23, 26, 30, 33, 34, 44, 45, 49, 50, 55, 59, 63, 66, 67, 69,
            70, 82, 83, 84, 85, 86, 91, 93, 97, 105, 107]

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


def band_energy(sess, ncl, model, lo, hi):
    """Per-trial sum-of-squares energy of every ICA source, band-passed [lo,hi]."""
    folder = g.get_folder(SUBJ, TASK, sess, ncl, model)
    mats = sorted(glob.glob(os.path.join(folder, "*.mat")))
    raw, _, _ = g.build_raw_from_mat_files(mats)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)
    ica = g.ch._load_offline_ica(SUBJ, TASK)
    src = ica.get_sources(raw)
    src.filter(lo, hi, picks="all", verbose=False)     # band-pass all sources
    D = src.get_data()
    trials = g.fet._chronological_trials(raw)
    classes = np.array([c for c, _s, _e in trials])
    ncomp = D.shape[0]
    E = np.empty((len(trials), ncomp))
    for ti, (_c, s, e) in enumerate(trials):
        E[ti] = np.sum(D[:, s:e] ** 2, axis=1)
    del raw, src, D
    g.gc.collect()
    return E, classes


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
    band_lbl = BAND[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"S{SUBJ:02d} · {TASK} · {model} · Sess{sess:02d} "
        f"{ncl}-class · {band_lbl} ({BAND[1]:g}–{BAND[2]:g} Hz) · {finger} · "
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
    ax.set_xlabel(f"Signed Pearson r  (energy ↔ task, 0.5×IQR-cleaned, per-class, {band_lbl})")
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
    os.makedirs(OUT, exist_ok=True)
    band_lbl, lo, hi = BAND
    kept = set(S09_KEPT); manual = set(range(128)) - kept
    art_auto = g.offline_artifacts()
    splits = {
        "auto offline ICA  (EOG/EMG)":           ("AUTO",   art_auto),
        f"manual selection  ({len(kept)} kept)": ("MANUEL", manual),
    }
    print(f"S09 Beta: AUTO={len(art_auto)}  MANUEL={len(manual)} (kept {len(kept)})",
          flush=True)

    for sess in SESSIONS:
        for model in MODELS:
            mtag = "1_Orig" if model == "Orig" else "2_Finetune"
            for ncl in NCLASSES:
                fingers = FINGERS_BY_NCL[ncl]
                print(f"\n=== S09 Sess{sess:02d} {model} {ncl}class [{band_lbl}] ===",
                      flush=True)
                try:
                    E, classes = band_energy(sess, ncl, model, lo, hi)
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
                            OUT, f"DistPearson05IQR_S{SUBJ:02d}_{TASK}_"
                            f"Sess{sess:02d}_{ncl}class_{mtag}_{band_lbl}_"
                            f"{finger}_{tag}.png")
                        plot_dist(r[art_idx], r[non_idx], finger, ncl, sess,
                                  model, save, art_def)

    print("\nS09 Beta done.", flush=True)


if __name__ == "__main__":
    main()

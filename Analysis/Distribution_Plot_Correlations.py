"""
Distribution_Plot_Correlations.py

For one subject/session/nClass/model: read the 'Corr' sheet from
Ressources/Sujet_XX.ods (built by fill_energy_table.py), split ICA sources
into Artefact / Non-Artefact groups (from offline artifact detection), and
plot the distribution of signed energy<->task correlations for both groups.
One image per (finger task × frequency band). Saves PNGs and shows them
interactively.

The Corr sheet columns:
    Session | nClass | Model | Band | Task | ICA_Source | Corr_Energy | Artefact ?

    - Task   = finger (Thumb / Index / Pinky / Middle), not the modality.
    - Artefact ? = "Yes" if that source was flagged EOG/EMG from the offline
                  recording, "No" otherwise.
    - Corr_Energy = signed Pearson in [-1, 1] (energy vs one-vs-rest membership).

Usage:
    python Distribution_Plot_Correlations.py <subj> <sess> <nclass> <task> <model>

    <task>  = MI or ME
    <model> = Orig or Finetune

Example:
    python Distribution_Plot_Correlations.py 9 1 3 MI Orig
"""

import os
import sys
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Import fill_energy_table BEFORE pyplot so its matplotlib.use("TkAgg") call
# (triggered via corr_heatmap) runs before the backend is locked in.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ICA_Scripts")))

import fill_energy_table as fet   # noqa: E402 — must precede pyplot

import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from config import EXCEL_DIR, TASK_LABELS

# Save alongside the energy heatmaps, under the same Week 8 root.
DIST_CORR_DIR = os.path.join(os.path.dirname(fet.ENERGY_CORR_DIR),
                              "Distribution Correlation Maps")

# Bands to generate. Edit this list to restrict output.
# Available: "Fullband", "Alpha", "Beta", "Beta+2"
BANDS = ["Fullband", "Alpha", "Beta", "Beta+2"]

# (KDE fill/line colour,  mean line colour,  median line colour)
_COLORS = {
    "Artefact":     ("crimson",   "darkred",       "firebrick"),
    "Non-Artefact": ("steelblue", "navy",          "cornflowerblue"),
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_corr_sheet(subj_id):
    """Return the 'Corr' DataFrame from Ressources/Sujet_XX.ods."""
    path = os.path.join(EXCEL_DIR, f"Sujet_{subj_id:02d}.ods")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Workbook not found: {path}\n"
            f"Run first:  python ICA_Scripts/fill_energy_table.py {subj_id}")
    df = pd.read_excel(path, sheet_name="Corr", engine="odf")
    df["Session"]    = df["Session"].astype(int)
    df["nClass"]     = df["nClass"].astype(int)
    df["ICA_Source"] = df["ICA_Source"].astype(int)
    return df


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_one_task(task_df, finger, subj_id, task,
                  sess, nclass, model, band_label,
                  save_path=None, show=True):
    """One figure for a single (finger task × frequency band) combination.

    task_df: Corr sheet already filtered to (sess, nclass, model, band_label, finger).
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"S{subj_id:02d} | {task} | {model} | "
        f"Sess{sess:02d} {nclass}-class | {band_label} | {finger}",
        fontsize=12, fontweight="bold",
    )

    x_grid = np.linspace(-1.05, 1.05, 800)

    # Fixed dot-strip y positions: artefact below, non-artefact above.
    _DOT_Y = {"Artefact": -0.15, "Non-Artefact": -0.05}

    groups = [
        ("Artefact",     task_df[task_df["Artefact ?"] == "Yes"]["Corr_Energy"].dropna().values),
        ("Non-Artefact", task_df[task_df["Artefact ?"] == "No"]["Corr_Energy"].dropna().values),
    ]

    handles = []

    for grp_label, vals in groups:
        col_kde, col_mean, col_med = _COLORS[grp_label]
        dot_y = _DOT_Y[grp_label]

        if len(vals) < 2:
            # gaussian_kde requires at least 2 points to estimate a bandwidth;
            # fall back to plotting the raw point(s) without a density curve.
            ax.scatter(vals, np.full_like(vals, dot_y),
                       color=col_kde, alpha=0.6, s=20, zorder=5)
            handles.append(
                Line2D([0], [0], marker="o", linestyle="none",
                       markerfacecolor=col_kde, markersize=5,
                       label=f"{grp_label}  (n={len(vals)}, too few for KDE)"))
            continue

        kde    = gaussian_kde(vals)
        y_full = kde(x_grid)

        ax.plot(x_grid, y_full, color=col_kde, linewidth=2.0)
        ax.fill_between(x_grid, y_full, alpha=0.18, color=col_kde)
        ax.axvline(vals.mean(),     color=col_mean, linestyle="--", linewidth=1.5)
        ax.axvline(np.median(vals), color=col_med,  linestyle=":",  linewidth=1.5)
        ax.scatter(vals, np.full_like(vals, dot_y),
                   color=col_kde, alpha=0.55, s=16, zorder=5)

        handles += [
            Line2D([0], [0], color=col_kde, linewidth=2.0,
                   label=f"{grp_label}  (n={len(vals)})"),
            Line2D([0], [0], color=col_mean, linestyle="--", linewidth=1.5,
                   label=f"  mean   = {vals.mean():.3f}"),
            Line2D([0], [0], color=col_med,  linestyle=":",  linewidth=1.5,
                   label=f"  median = {np.median(vals):.3f}"),
            Line2D([0], [0], marker="o", linestyle="none",
                   markerfacecolor=col_kde, markersize=5, alpha=0.6,
                   label="  individual sources"),
        ]

    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.3)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(bottom=min(_DOT_Y.values()) - 0.03)
    ax.set_xlabel("Signed Pearson Correlation")
    ax.set_ylabel("Density")
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.85)
    ax.grid(axis="y", alpha=0.22)

    plt.tight_layout(rect=[0, 0, 1, 0.88])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ── public entry point ────────────────────────────────────────────────────────

def run(subj_id, sess, nclass, task="MI", model="Orig",
        show=True, save=True, bands=None):
    """Generate distribution plots for one subject × condition.

    Produces one image per (finger task × frequency band). Figures are saved
    and/or shown depending on the `save` / `show` flags.
    `bands` overrides the module-level BANDS list when provided.
    """
    bands_to_use = bands if bands is not None else BANDS

    df   = load_corr_sheet(subj_id)
    cond = df[(df["Session"] == sess) &
              (df["nClass"]  == nclass) &
              (df["Model"]   == model)]

    if cond.empty:
        print(f"  [skip] S{subj_id:02d} Sess{sess:02d} {nclass}class {model}: "
              "no rows in Corr sheet.")
        return

    tasks_ordered = [t for t in TASK_LABELS if t in cond["Task"].unique()]
    model_tag     = "1_Orig" if model == "Orig" else "2_Finetune"

    for band_label in bands_to_use:
        band_df = cond[cond["Band"] == band_label]
        if band_df.empty:
            continue

        for finger in tasks_ordered:
            task_df = band_df[band_df["Task"] == finger]
            if task_df.empty:
                continue

            save_path = None
            if save:
                save_path = os.path.join(
                    DIST_CORR_DIR, f"S{subj_id:02d}", band_label,
                    f"DistCorr_S{subj_id:02d}_{task}_Sess{sess:02d}"
                    f"_{nclass}class_{model_tag}_{band_label}_{finger}.png")

            plot_one_task(task_df, finger, subj_id, task,
                          sess, nclass, model, band_label,
                          save_path=save_path, show=show)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 6:
        print("Usage: python Distribution_Plot_Correlations.py "
              "<subj> <sess> <nclass> <task> <model>")
        print("  <task>  = MI | ME")
        print("  <model> = Orig | Finetune")
        sys.exit(1)
    subj_id = int(sys.argv[1])
    sess    = int(sys.argv[2])
    nclass  = int(sys.argv[3])
    task    = sys.argv[4]
    model   = sys.argv[5]
    run(subj_id, sess, nclass, task, model, show=True, save=True)


if __name__ == "__main__":
    main()

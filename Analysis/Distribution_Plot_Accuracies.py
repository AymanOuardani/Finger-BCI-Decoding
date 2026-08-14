"""
Distribution_Plot_Accuracies.py

Utility module for plotting and comparing EEGNet accuracy distributions.
Provides kl_divergence() and plot_distribution() — used in subject reports
and cross-condition comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D


def kl_divergence(p, q):
    """Kullback-Leibler divergence KL(p || q) between two discretized distributions p, q
    (assumed to already sum to 1 over the same support)."""
    # Clip away zeros to avoid log(0) / division-by-zero in regions where one
    # distribution has (near) no density.
    p = np.clip(p, 1e-10, None)
    q = np.clip(q, 1e-10, None)
    return np.sum(p * np.log(p / q))


def plot_distribution(values1, values2,
                      label1="Our Results", label2="Article's Results",
                      values3=None, label3="ICA Results",
                      title="Accuracy Distribution", xlabel="Accuracy (%)",
                      save_path=None):
    """Plot overlaid KDE distributions (with mean/median lines and individual score
    dots) for two or three accuracy series, annotated with pairwise KL divergences."""

    values1 = np.array(values1)
    values2 = np.array(values2)

    colors = [("steelblue",    "crimson",    "darkorange"),
              ("seagreen",     "navy",       "darkolivegreen"),
              ("mediumpurple", "darkviolet", "orchid")]

    all_values = [values1, values2]
    all_labels = [label1, label2]
    if values3 is not None:
        all_values.append(np.array(values3))
        all_labels.append(label3)

    x = np.linspace(20, 90, 1000)

    kdes = [gaussian_kde(v) for v in all_values]
    # Renormalize each KDE curve to sum to 1 over the sampled grid so it can be
    # treated as a discrete probability distribution for the KL-divergence calc.
    kde_norm = [k(x) / k(x).sum() for k in kdes]

    kl_pq = kl_divergence(kde_norm[0], kde_norm[1])
    kl_qp = kl_divergence(kde_norm[1], kde_norm[0])

    fig, ax = plt.subplots(figsize=(12 if values3 is not None else 11, 6))

    dot_step   = -0.0015
    dot_y_positions = [dot_step * (i + 1) for i in range(len(all_values))]

    for idx, (values, label, (col_kde, col_mean, col_median)) in enumerate(zip(
            all_values, all_labels, colors)):

        kde = gaussian_kde(values)
        y   = kde(x)
        ax.plot(x, y, color=col_kde, linewidth=2.5)
        ax.fill_between(x, y, alpha=0.20, color=col_kde)

        ax.axvline(values.mean(),     color=col_mean,   linestyle="--", linewidth=1.5)
        ax.axvline(np.median(values), color=col_median, linestyle=":",  linewidth=1.5)

        y_dot = dot_y_positions[idx]
        ax.scatter(values, np.full_like(values, y_dot),
                   color=col_kde, alpha=0.7, s=30, zorder=5)

    ax.set_xlim(20, 90)
    ax.set_ylim(bottom=min(dot_y_positions) - 0.001)
    ax.axhline(0, color='gray', linewidth=0.6, linestyle='-', alpha=0.4)

    legend_pairs = []
    for (col_kde, col_mean, col_median), label, values in zip(colors, all_labels, all_values):
        legend_pairs += [
            (Line2D([0],[0], color=col_kde, linewidth=2.5),
             f"{label} — KDE"),
        ]
    for (col_kde, col_mean, col_median), label, values in zip(colors, all_labels, all_values):
        legend_pairs += [
            (Line2D([0],[0], color=col_mean, linestyle='--', linewidth=1.5),
             f"{label} — Mean = {np.array(values).mean():.2f}%"),
        ]
    for (col_kde, col_mean, col_median), label, values in zip(colors, all_labels, all_values):
        legend_pairs += [
            (Line2D([0],[0], color=col_median, linestyle=':', linewidth=1.5),
             f"{label} — Median = {np.median(values):.2f}%"),
        ]
    for (col_kde, col_mean, col_median), label, values in zip(colors, all_labels, all_values):
        legend_pairs += [
            (Line2D([0],[0], marker='o', linestyle='none',
                    markerfacecolor=col_kde, markersize=6, alpha=0.7),
             f"{label} — Individual scores"),
        ]

    legend_pairs += [
        (Line2D([0],[0], linestyle='none'),
         f"KL({label1} ‖ {label2}) = {kl_pq:.4f}"),
        (Line2D([0],[0], linestyle='none'),
         f"KL({label2} ‖ {label1}) = {kl_qp:.4f}"),
    ]

    if values3 is not None:
        v3 = np.array(values3)
        kl_ica_art = kl_divergence(kde_norm[2], kde_norm[1])
        kl_art_ica = kl_divergence(kde_norm[1], kde_norm[2])
        legend_pairs += [
            (Line2D([0],[0], linestyle='none'),
             f"KL({label3} ‖ {label2}) = {kl_ica_art:.4f}"),
            (Line2D([0],[0], linestyle='none'),
             f"KL({label2} ‖ {label3}) = {kl_art_ica:.4f}"),
        ]

    handles, labels = zip(*legend_pairs)
    ax.legend(handles, labels, fontsize=9, loc='upper left',
              bbox_to_anchor=(1, 1))

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import math
    import sys
    import pandas as pd

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import (
        TRACKING_ODS, EXCEL_DATA, SAVE_FOLDER,
        ALL_SUBJECTS, ROW_SUBJECT_OFFSET, COL_ONLINE_PERF, ICA_ONLINE_PERF,
    )

    SHOW_ICA = "ICA" in [a.upper() for a in sys.argv[1:]]

    # -- Configuration — edit these to plot a different combo -------
    SESSION_NUM = 1
    NCLASS      = 3
    TASK        = "MI"
    MODELTYPE   = "Finetune"   # "Orig" or "Finetune"
    SUBJECTS    = ALL_SUBJECTS

    col_perf     = COL_ONLINE_PERF[MODELTYPE]
    col_ica_perf = ICA_ONLINE_PERF[MODELTYPE]
    sheet_name   = f"{TASK}_Sess{SESSION_NUM:02}_{NCLASS}Class"

    # -- Read our Online Perf Acc from the per-condition sheet ------
    tracking   = pd.read_excel(TRACKING_ODS, engine="odf",
                               sheet_name=sheet_name, header=None)
    our_values = [float(tracking.iloc[subj_id + ROW_SUBJECT_OFFSET, col_perf])
                  for subj_id in SUBJECTS]

    ica_values = None
    if SHOW_ICA:
        ica_values = [float(tracking.iloc[subj_id + ROW_SUBJECT_OFFSET, col_ica_perf])
                      for subj_id in SUBJECTS]

    GROUP_LABEL = {"Orig": "Base", "Finetune": "Fine-tuned"}[MODELTYPE]

    # -- Read article values from Excel DATA.xlsx -------------------
    subj_labels = [f"S{s:02d}" for s in SUBJECTS]

    if TASK == "MI":
        df = pd.read_excel(EXCEL_DATA, sheet_name="Fig. 1AB,EF",
                           header=None, skiprows=1)
        c_subj, c_sess, c_grp, c_val = (0, 1, 4, 5) if NCLASS == 2 else (9, 10, 13, 14)
        df_f = df[
            (df[c_sess] == f"Session {SESSION_NUM}") &
            (df[c_grp]  == GROUP_LABEL)
        ]
        df_f = df_f[df_f[c_subj].isin(subj_labels)].set_index(c_subj).loc[subj_labels]
        article_values = df_f[c_val].astype(float).tolist()

    else:  # ME — Fig. 3AB, EF
        df = pd.read_excel(EXCEL_DATA, sheet_name="Fig. 3AB, EF",
                           header=None, skiprows=2)   # skip 2 header rows

        # Two side-by-side blocks:
        #   2-class : cols 0=subject, 2=session_num, 5=group, 6=value
        #   3-class : cols 8=subject, 10=session_num, 13=group, 14=value
        if NCLASS == 2:
            c_subj, c_sess, c_grp, c_val = 0, 2, 5, 6
        else:
            c_subj, c_sess, c_grp, c_val = 8, 10, 13, 14

        df_f = df[
            (df[c_sess] == SESSION_NUM) &
            (df[c_grp]  == GROUP_LABEL)
        ]
        df_f = df_f[df_f[c_subj].isin(subj_labels)].set_index(c_subj).loc[subj_labels]
        article_values = df_f[c_val].astype(float).tolist()

    # -- Drop subjects with missing data (NaN in any active series) -
    def _ok(o, a, i):
        """True if none of the (our, article, ICA) values for a subject is NaN (ICA optional)."""
        return not (math.isnan(o) or math.isnan(a) or
                    (i is not None and math.isnan(i)))

    if SHOW_ICA:
        rows = [(o, a, i) for o, a, i in zip(our_values, article_values, ica_values)
                if _ok(o, a, i)]
        if not rows:
            raise RuntimeError("No valid data to plot — all values are NaN.")
        our_values, article_values, ica_values = zip(*rows)
    else:
        rows = [(o, a) for o, a in zip(our_values, article_values)
                if not (math.isnan(o) or math.isnan(a))]
        if not rows:
            raise RuntimeError("No valid data to plot — all our_values are NaN.")
        our_values, article_values = zip(*rows)

    n_plotted = len(our_values)
    print(f"Plotting {n_plotted} subjects "
          f"(skipped {len(SUBJECTS) - n_plotted} with missing data)"
          + (" [ICA enabled]" if SHOW_ICA else ""))

    # -- Plot -------------------------------------------------------
    ica_suffix = "_ICA" if SHOW_ICA else ""
    title = (f"Accuracy Distribution — Our Results vs Article's Results"
             f"{' vs ICA Results' if SHOW_ICA else ''}\n"
             f"({NCLASS}-class {TASK}, Session {SESSION_NUM}, {MODELTYPE} model)")

    plot_save_path = os.path.join(
        SAVE_FOLDER,
        f"distribution_{TASK}_{NCLASS}class_Sess{SESSION_NUM:02}_{MODELTYPE}{ica_suffix}.png"
    )

    plot_distribution(
        our_values, article_values,
        label1="Our Results",
        label2="Article's Results",
        values3=ica_values,
        label3="ICA Results",
        title=title,
        save_path=plot_save_path,
    )
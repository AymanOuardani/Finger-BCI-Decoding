"""
Plot_Accuracy_Evolution.py

Plots accuracy evolution across conditions for a given subject,
with one line per frequency band (Full, Alpha, Beta).

X-axis: Session 1 Base → Session 1 Fine-tuned → Session 2 Base → Session 2 Fine-tuned
Y-axis: Accuracy (%)

Usage:
    python Plot_Accuracy_Evolution.py <subj_id> <task> <nclass>
    python Plot_Accuracy_Evolution.py 9 MI 2
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import TRACKING_ODS, RESULTS_ROOT, ROW_OFFSETS, COL_ONLINE_PERF

# ── The 4 conditions in order ─────────────────────────────────────────────────
CONDITIONS = [
    ("Sess 1\nBase",       1, "Orig"),
    ("Sess 1\nFine-tuned", 1, "Finetune"),
    ("Sess 2\nBase",       2, "Orig"),
    ("Sess 2\nFine-tuned", 2, "Finetune"),
]

# ── Band styling ──────────────────────────────────────────────────────────────
BAND_STYLE = {
    "Full":  {"color": "steelblue",  "marker": "o", "lw": 2.2,
              "label": "Full Band [4–40 Hz]"},
    "Alpha": {"color": "darkorange", "marker": "s", "lw": 2.2,
              "label": "Alpha Band [8–13 Hz]"},
    "Beta":  {"color": "seagreen",   "marker": "^", "lw": 2.2,
              "label": "Beta Band [13–30 Hz]"},
}


# ── Read one value from ODS ───────────────────────────────────────────────────
def read_value(df, row_idx, col_idx):
    """Read a single numeric cell from `df`, returning None if it's missing, NaN, or unreadable."""
    try:
        val = df.iloc[row_idx, col_idx]
        fval = float(val)
        return None if math.isnan(fval) else fval
    except Exception:
        return None


# ── Load all values for one subject / task / nclass ──────────────────────────
def load_data(subj_id, task, nclass):
    """
    Returns dict: band_name → list of 4 values (one per condition).
    Missing values are None.
    """
    sheet_name = f"{task}_Sess{{:02}}_{nclass}Class"

    xl     = pd.ExcelFile(TRACKING_ODS, engine="odf")
    sheets = {
        name: pd.read_excel(TRACKING_ODS, engine="odf",
                            sheet_name=name, header=None)
        for name in xl.sheet_names
    }

    data = {}
    for band_name, row_offset in ROW_OFFSETS.items():
        values = []
        for label, session, modeltype in CONDITIONS:
            sname = f"{task}_Sess{session:02}_{nclass}Class"
            if sname not in sheets:
                values.append(None)
                continue
            row_idx = subj_id + row_offset
            col_idx = COL_ONLINE_PERF[modeltype]
            values.append(read_value(sheets[sname], row_idx, col_idx))
        data[band_name] = values

    return data


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_evolution(subj_id, task, nclass, data, save_path=None):
    """Plot per-band accuracy lines across the 4 conditions (session x base/fine-tuned),
    with value annotations, a chance-level reference line, and a session-boundary marker."""
    x_labels = [c[0] for c in CONDITIONS]
    x        = np.arange(len(CONDITIONS))

    fig, ax = plt.subplots(figsize=(9, 6))

    for band_name, values in data.items():
        style = BAND_STYLE[band_name]

        # ── split into segments at None gaps so lines don't cross missing pts ──
        xs_seg, ys_seg = [], []
        xs_cur, ys_cur = [], []
        for xi, val in zip(x, values):
            if val is not None:
                xs_cur.append(xi)
                ys_cur.append(val)
            else:
                if xs_cur:
                    xs_seg.append((xs_cur[:], ys_cur[:]))
                xs_cur, ys_cur = [], []
        if xs_cur:
            xs_seg.append((xs_cur, ys_cur))

        first = True
        for xs, ys in xs_seg:
            ax.plot(xs, ys,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=style["lw"],
                    markersize=8,
                    label=style["label"] if first else "_nolegend_",
                    zorder=3)
            first = False

        # ── annotate each point with its value ────────────────────────────────
        for xi, val in zip(x, values):
            if val is not None:
                ax.annotate(
                    f"{val:.2f}%",
                    xy=(xi, val),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8,
                    color=style["color"],
                    fontweight="bold",
                )

    # ── chance line ───────────────────────────────────────────────────────────
    chance = 100 / nclass
    ax.axhline(chance, color="gray", linestyle="--", linewidth=1,
               alpha=0.6, label=f"Chance level ({chance:.1f}%)")

    # ── session separator ─────────────────────────────────────────────────────
    ax.axvline(1.5, color="black", linestyle=":", linewidth=1, alpha=0.4)
    ax.text(1.5, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 30,
            "  Session boundary", fontsize=8, color="gray",
            va="bottom", ha="left")

    # ── formatting ────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("Online Performance Accuracy (%)", fontsize=11)
    ax.set_xlabel("Condition", fontsize=11)
    ax.set_title(
        f"Accuracy Evolution — Subject S{subj_id:02}\n"
        f"{task}  |  {nclass}-class  |  Full vs Alpha vs Beta band",
        fontsize=12, fontweight="bold"
    )
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.grid(axis="x", alpha=0.15)
    ax.set_ylim(bottom=max(0, min(
        v for vals in data.values() for v in vals if v is not None
    ) - 10))
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Plot accuracy evolution across sessions and bands",
        usage="python Plot_Accuracy_Evolution.py <subj_id> <task> <nclass>"
    )
    p.add_argument("subj_id", type=int,
                   help="Subject ID (1-21)")
    p.add_argument("task",    type=str, choices=["MI", "ME"],
                   help="Task type: MI or ME")
    p.add_argument("nclass",  type=int, choices=[2, 3],
                   help="Number of classes")
    p.add_argument("--no-save", action="store_true",
                   help="Do not save the figure")
    return p.parse_args()


def main():
    """CLI entry point: parses arguments, loads accuracy data, prints a summary table, and plots it."""
    args     = parse_args()
    subj_id  = args.subj_id
    task     = args.task
    nclass   = args.nclass

    print(f"Loading data for S{subj_id:02} | {task} {nclass}-class ...")
    data = load_data(subj_id, task, nclass)

    print(f"\n  {'Condition':<22} {'Full':>8}  {'Alpha':>8}  {'Beta':>8}")
    print(f"  {'─'*50}")
    for i, (label, session, modeltype) in enumerate(CONDITIONS):
        lbl = label.replace('\n', ' ')
        f   = data['Full'][i];  a = data['Alpha'][i];  b = data['Beta'][i]
        fstr = f"{f:.2f}%" if f is not None else "XX"
        astr = f"{a:.2f}%" if a is not None else "XX"
        bstr = f"{b:.2f}%" if b is not None else "XX"
        print(f"  {lbl:<22} {fstr:>8}  {astr:>8}  {bstr:>8}")

    save_path = None
    if not args.no_save:
        subj_folder = os.path.join(RESULTS_ROOT, f"Sujet {subj_id}")
        os.makedirs(subj_folder, exist_ok=True)
        save_path = os.path.join(
            subj_folder,
            f"AccuracyEvolution_S{subj_id:02}_{task}_{nclass}class.png"
        )

    plot_evolution(subj_id, task, nclass, data, save_path=save_path)


if __name__ == "__main__":
    main()
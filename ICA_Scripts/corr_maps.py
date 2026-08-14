"""
corr_maps.py

Energy <-> task heatmaps: tasks (rows) x ICA sources (columns), one image per
recording. Answers "which source carries energy that tracks which finger".

Three metrics are available and share the same layout, so the images are
directly comparable:

    pearson       signed Pearson r, per-class 0.5xIQR-cleaned   [-1, +1]
    auc           P(energy higher during the task)              [0, 1], 0.5 = chance
    rankbiserial  2*AUC - 1, the signed form of the same        [-1, +1], 0 = chance

Artefact source indices (offline EOG/EMG detection) are drawn in red on the
x-axis. The offline ICA is reused for every session, so a column means the same
component throughout.

Usage:
    python corr_maps.py <subj> [metric] [band] [task]

    python corr_maps.py 9                    # Pearson, Fullband, MI
    python corr_maps.py 9 auc                # AUC maps
    python corr_maps.py 9 rankbiserial       # rank-biserial maps
    python corr_maps.py 9 pearson Beta       # Beta band

Figures -> RESULTS_ROOT/Correlation Maps/S{XX}/{Band}/
"""

import os
import sys
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import energy_stats as es
from config import CORR_MAPS_DIR, CORRUPTED_DATA


def plot_map(matrix, fingers, subj, task, sess, nclass, model,
             metric, band, artifacts, save_path):
    """Render and save one tasks x ICA-sources correlation heatmap (fingers on
    the y-axis, sources on the x-axis, artefact source labels drawn in red)."""
    label, _fn, (vmin, vmax) = es.METRICS[metric]
    cmap = "RdBu_r"
    artifacts = set(artifacts or [])
    n_fingers, n_comp = matrix.shape

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(label)

    ax.set_yticks(range(n_fingers))
    ax.set_yticklabels(fingers)
    ax.set_xticks(range(n_comp))
    labels = ax.set_xticklabels([str(c) for c in range(n_comp)],
                                fontsize=6, rotation=90)
    for c, lbl in zip(range(n_comp), labels):
        if c in artifacts:
            lbl.set_color("red")
    ax.set_xlabel("ICA Source   (red = EOG/EMG artefact)")
    ax.set_ylabel("Task")
    ax.set_title(f"S{subj:02d} | {task} | Sess{sess:02d} {nclass}class {model} | "
                 f"{band} | Energy↔Task {metric}")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    """CLI entry point: parse argv, then generate one correlation map per
    (session, nClass, model) recording available for the subject."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    subj = int(sys.argv[1])
    metric = sys.argv[2].lower() if len(sys.argv) > 2 else "pearson"
    band = sys.argv[3] if len(sys.argv) > 3 else "Fullband"
    task = sys.argv[4].upper() if len(sys.argv) > 4 else "MI"

    if metric not in es.METRICS:
        print(f"unknown metric {metric!r}; expected one of {sorted(es.METRICS)}")
        sys.exit(1)
    if band not in es.BANDS:
        print(f"unknown band {band!r}; expected one of {sorted(es.BANDS)}")
        sys.exit(1)

    out_dir = os.path.join(CORR_MAPS_DIR, f"S{subj:02d}", band)
    groups = es.available_recordings(subj, task)
    print(f"S{subj:02d} {task}: {len(groups)} (session, nClass) group(s)")
    print(f"metric={metric}  band={band}  ->  {out_dir}")

    # Artefact indices come from the offline ICA and are reused for every
    # session/model of the subject, so a column means the same source throughout.
    artifacts = es.auto_artifacts(subj, task)
    print(f"Offline artefact sources ({len(artifacts)}): {sorted(artifacts)}")

    n_saved = 0
    for (sess, nclass), models in sorted(groups.items()):
        for model in models:
            if (subj, task, sess, nclass, model) in CORRUPTED_DATA:
                print(f"  Sess{sess:02d} {nclass}class {model} ... [corrupted, skipped]")
                continue
            print(f"  Sess{sess:02d} {nclass}class {model} ...", flush=True)
            try:
                E, classes = es.trial_energy(subj, sess, nclass, model, task, band)
            except Exception as exc:
                print(f"    [SKIP] {exc}", flush=True)
                continue

            fingers = es.present_fingers(classes, nclass)
            matrix = es.correlation_matrix(E, classes, fingers, metric)
            tag = es.MODEL_TAG[model]
            name = (f"{metric}_S{subj:02d}_{task}_Sess{sess:02d}_"
                    f"{nclass}class_{tag}_{band}.png")
            plot_map(matrix, fingers, subj, task, sess, nclass, model,
                     metric, band, artifacts, os.path.join(out_dir, name))
            print(f"    saved: {name}", flush=True)
            n_saved += 1

            del E
            gc.collect()

    print(f"\nDone. {n_saved} map(s) -> {out_dir}")


if __name__ == "__main__":
    main()

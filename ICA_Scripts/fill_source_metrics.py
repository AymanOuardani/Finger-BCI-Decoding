"""
fill_source_metrics.py

Per-source metric suite for ONE recording, written to Excel.

Where fill_separation_metrics.py answers "artefact or neural?" at the level of
the whole source population, this script goes the other way: it lists every ICA
source individually, with several measures of its energy<->task association
side by side, so that a suspicious correlation can be diagnosed.

Why several measures rather than one: per-trial energy is heavy-tailed, so a
raw Pearson r can be produced almost entirely by a handful of high-energy
trials. Comparing the raw r against the log-energy, trimmed and rank-based
variants shows immediately whether an association is carried by the bulk of the
trials or by its tail. The kurtosis and Max/Median columns quantify that tail.

Columns per (source, finger):
    r_raw        Pearson on raw energy
    r_log        Pearson on log10(energy)          — tail-insensitive
    r_trimmed    Pearson after dropping the top 5% — tail-insensitive
    spearman     rank correlation                  — tail-insensitive
    AUC          P(energy higher during the task)
    rank_biserial 2*AUC - 1
    r_band       Pearson on the band-filtered energy (default Beta)
    Artefact     Yes/No, offline EOG/EMG detection
    kurtosis     tail weight of the energy distribution
    max_median   max energy / median energy

Usage:
    python fill_source_metrics.py <subj> <sess> <nclass> <model> [band] [task]

    python fill_source_metrics.py 9 1 3 Orig            # Beta as comparison band
    python fill_source_metrics.py 4 5 3 Orig Alpha      # Alpha instead

Excel -> Ressources/Source_Metrics.xlsx, one sheet per recording.
"""

import os
import sys
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from scipy.stats import spearmanr, kurtosis, mannwhitneyu

import energy_stats as es
import excel_io
from config import SOURCE_METRICS_XLSX


HEADER = ["ICA_Source", "Finger", "Artefact",
          "r_raw", "r_log", "r_trimmed", "spearman",
          "AUC", "rank_biserial", "r_band", "Band",
          "kurtosis", "max_median"]

NUMBER_FORMAT = {
    "r_raw": "0.000", "r_log": "0.000", "r_trimmed": "0.000",
    "spearman": "0.000", "AUC": "0.000", "rank_biserial": "0.000",
    "r_band": "0.000", "kurtosis": "0.0", "max_median": "0.0",
}


def source_metrics(energy, membership):
    """The metric suite for one source against one finger's one-vs-rest vector."""
    energy = np.asarray(energy, float)
    membership = np.asarray(membership, float)
    n = len(energy)

    r_raw = es.pearson(energy, membership)
    r_log = es.pearson(np.log10(energy + 1e-30), membership)

    # drop the top 5% highest-energy trials, the ones a raw r leans on
    k = max(1, int(round(0.05 * n)))
    keep = np.argsort(energy)[:-k]
    r_trimmed = es.pearson(energy[keep], membership[keep])

    rho = spearmanr(energy, membership)[0]
    rho = float(rho) if np.isfinite(rho) else 0.0

    inside = membership > 0
    n1, n0 = int(inside.sum()), int((~inside).sum())
    if n1 and n0:
        U = mannwhitneyu(energy[inside], energy[~inside],
                         alternative="two-sided").statistic
        auc = float(U / (n1 * n0))
    else:
        auc = 0.5

    return {
        "r_raw": r_raw, "r_log": r_log, "r_trimmed": r_trimmed,
        "spearman": rho, "auc": auc, "rank_biserial": 2 * auc - 1,
        "kurtosis": float(kurtosis(energy)),
        "max_median": float(energy.max() / np.median(energy))
        if np.median(energy) > 0 else float("nan"),
    }


def main():
    """CLI entry point: parse args, compute per-source metrics for the recording,
    and write them to the Source_Metrics.xlsx workbook."""
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)

    subj = int(sys.argv[1])
    sess = int(sys.argv[2])
    nclass = int(sys.argv[3])
    model = sys.argv[4]
    band = sys.argv[5] if len(sys.argv) > 5 else "Beta"
    task = sys.argv[6].upper() if len(sys.argv) > 6 else "MI"

    if band not in es.BANDS:
        print(f"unknown band {band!r}; expected one of {sorted(es.BANDS)}")
        sys.exit(1)
    if model not in es.MODEL_TAG:
        print(f"unknown model {model!r}; expected Orig or Finetune")
        sys.exit(1)

    print(f"S{subj:02d} {task} Sess{sess:02d} {nclass}class {model} "
          f"(comparison band: {band})")

    E, classes = es.trial_energy(subj, sess, nclass, model, task, "Fullband")
    E_band, _ = es.trial_energy(subj, sess, nclass, model, task, band)
    artifacts = es.auto_artifacts(subj, task)

    n_comp = E.shape[1]
    fingers = es.present_fingers(classes, nclass)
    print(f"  n_trials={len(classes)}  ncomp={n_comp}  "
          f"artefacts={len(artifacts)}  fingers={fingers}")

    rows = []
    for src in range(n_comp):
        for finger in fingers:
            membership = (classes == finger).astype(float)
            m = source_metrics(E[:, src], membership)
            r_band = es.pearson(E_band[:, src], membership)
            rows.append([
                src, finger, "Yes" if src in artifacts else "No",
                round(m["r_raw"], 4), round(m["r_log"], 4),
                round(m["r_trimmed"], 4), round(m["spearman"], 4),
                round(m["auc"], 4), round(m["rank_biserial"], 4),
                round(r_band, 4), band,
                round(m["kurtosis"], 2), round(m["max_median"], 2),
            ])

    # Trial-energy arrays are (n_trials, n_components) and can be large; free
    # them explicitly once every row has been extracted.
    del E, E_band
    gc.collect()

    sheet = f"S{subj:02d}_S{sess:02d}_{nclass}c_{model}"
    excel_io.write_sheet(SOURCE_METRICS_XLSX, sheet, HEADER, rows,
                         number_format=NUMBER_FORMAT)
    print(f"\nDone -> {SOURCE_METRICS_XLSX}")
    print(f"  sheet '{sheet}' ({len(rows)} rows)")


if __name__ == "__main__":
    main()

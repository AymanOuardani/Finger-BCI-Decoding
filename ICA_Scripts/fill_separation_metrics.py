"""
fill_separation_metrics.py

Does the task-related energy sit in the artefactual sources or in the neural
ones? This script answers that question in numbers, for every recording of a
subject, and writes the answer to Excel.

For each (session, model, nClass, finger, artefact definition) it computes the
per-source energy<->task correlation, splits the 128 sources into Artefact and
Non-Artefact, and measures how far apart the two groups are:

    AUC        Mann-Whitney on |r|; 0.5 = the two groups are indistinguishable,
               > 0.5 = artefactual sources are more task-correlated
    Cohen's d  the same gap in pooled standard deviations
    p          two-sided p-value of the Mann-Whitney test

These are the numbers drawn on the figures of corr_distributions.py, gathered
here in a form that can be sorted and cited.

Usage:
    python fill_separation_metrics.py <subj> [band] [essai] [task]
    python fill_separation_metrics.py all [band] [essai] [task]

    python fill_separation_metrics.py 9                 # S09, Fullband, AUTO+MANUEL
    python fill_separation_metrics.py 9 Beta            # Beta band
    python fill_separation_metrics.py all Fullband auto # every subject, AUTO only

Excel -> Ressources/Separation_Metrics.xlsx, one sheet per (subject, band).
"""

import os
import sys
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import energy_stats as es
import excel_io
from config import SEPARATION_XLSX, CORRUPTED_DATA


HEADER = ["Subject", "Task", "Session", "nClass", "Model", "Band", "Finger",
          "Artefact_Def", "N_Artefact", "N_Non_Artefact",
          "Mean_r_Artefact", "Mean_r_Non_Artefact",
          "AUC", "Cohens_d", "p_value", "Significant"]

NUMBER_FORMAT = {
    "Mean_r_Artefact": "0.000", "Mean_r_Non_Artefact": "0.000",
    "AUC": "0.000", "Cohens_d": "+0.00;-0.00", "p_value": "0.000E+00",
}


def subject_rows(subj, band, essai, task, metric="pearson"):
    """One row per (session, model, nClass, finger, artefact definition)."""
    splits = es.artifact_splits(subj, task, essai)
    if not splits:
        print(f"S{subj:02d}: no artefact definition available for '{essai}'")
        return []
    for tag, (label, art) in splits.items():
        print(f"S{subj:02d}: {tag} artefacts = {len(art)}  ({label})")

    rows = []
    for (sess, nclass), models in sorted(es.available_recordings(subj, task).items()):
        for model in models:
            if (subj, task, sess, nclass, model) in CORRUPTED_DATA:
                print(f"  Sess{sess:02d} {nclass}class {model} [corrupted, skipped]")
                continue
            print(f"  Sess{sess:02d} {nclass}class {model} ...", flush=True)
            try:
                E, classes = es.trial_energy(subj, sess, nclass, model, task, band)
            except Exception as exc:
                print(f"    [SKIP] {exc}", flush=True)
                continue

            n_comp = E.shape[1]
            fingers = es.present_fingers(classes, nclass)
            matrix = es.correlation_matrix(E, classes, fingers, metric)

            for fi, finger in enumerate(fingers):
                for tag, (_label, artifacts) in splits.items():
                    st = es.separation(matrix[fi], artifacts, n_comp)
                    rows.append([
                        f"S{subj:02d}", task, sess, nclass, model, band, finger, tag,
                        st["n_artefact"], st["n_non_artefact"],
                        round(st["mean_artefact"], 4),
                        round(st["mean_non_artefact"], 4),
                        round(st["auc"], 4), round(st["cohens_d"], 4),
                        st["p_value"], st["significant"],
                    ])
                    print(f"    {finger:6s} {tag:6s} "
                          f"AUC={st['auc']:.3f} d={st['cohens_d']:+.2f} "
                          f"p={st['p_value']:.3g} {st['significant']}", flush=True)

            del E
            gc.collect()
    return rows


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]
    band = sys.argv[2] if len(sys.argv) > 2 else "Fullband"
    essai = sys.argv[3].lower() if len(sys.argv) > 3 else "all"
    task = sys.argv[4].upper() if len(sys.argv) > 4 else "MI"

    if band not in es.BANDS:
        print(f"unknown band {band!r}; expected one of {sorted(es.BANDS)}")
        sys.exit(1)

    subjects = (es.SUBJECTS_WITH_MANUAL if target.lower() == "all"
                else [int(target)])

    written = []
    for subj in subjects:
        print(f"\n=== S{subj:02d} | {task} | {band} ===", flush=True)
        rows = subject_rows(subj, band, essai, task)
        if not rows:
            print(f"S{subj:02d}: nothing written")
            continue
        sheet = f"S{subj:02d}_{band}"
        excel_io.write_sheet(SEPARATION_XLSX, sheet, HEADER, rows,
                             number_format=NUMBER_FORMAT)
        written.append(f"{sheet} ({len(rows)} rows)")
        print(f"S{subj:02d}: sheet '{sheet}' written ({len(rows)} rows)")

    if written:
        print(f"\nDone -> {SEPARATION_XLSX}")
        for w in written:
            print(f"  {w}")
    else:
        print("\nNothing written.")


if __name__ == "__main__":
    main()

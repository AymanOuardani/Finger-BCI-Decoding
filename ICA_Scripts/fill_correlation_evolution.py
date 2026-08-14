"""
fill_correlation_evolution.py

For one subject, fill its sheet in Ressources/Correlation_Evolution.ods.

For every band (Alpha, Beta, Beta+2, Broadband) and every available online
session, this computes — **only over the ICA sources auto-detected as EOG/EMG
artifacts** (same detection as clean_ICA.py) — the SUM of absolute correlations
with each task, separately for the 2-class and 3-class recordings, and writes
them to the matching cells of sheet "Sujet<id>".

The number of available sessions is detected per n-class before filling, so
subjects with 2 sessions and subjects with 5 sessions are both handled.

Usage:
    python fill_correlation_evolution.py <subj>            # MI / Orig
    python fill_correlation_evolution.py <subj> <task> <model>
"""

import os
import sys
import glob
from collections import defaultdict

import numpy as np
from scipy.stats import zscore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Functions import (
    build_raw_from_mat_files, build_task_vectors,
    get_folder, get_ica_fif_path, fit_or_load_ica,
    compute_exclusions, write_to_ods,
)
from config import EXCEL_DIR, EOG_THRESHOLD, EMG_SLOPE_THRESH

import corr_heatmap as ch   # reuse _compute_corr_matrix / _load_offline_ica

ODS_PATH = os.path.join(EXCEL_DIR, "Correlation_Evolution.ods")
MAX_SESSIONS = 5

# (band label, l_freq, h_freq, sheet row of "Session 1"). Session N -> base + N-1.
BANDS = [
    ("Alpha",     8.0,  13.0, 3),
    ("Beta",      13.0, 30.0, 13),
    ("Beta+2",    30.0, 40.0, 23),
    ("Broadband", None, None, 32),
]

# task name -> column index, per n-class (Pouce=Thumb, Index, Auriculaire=Pinky)
COL_MAP = {
    2: {"Thumb": 1, "Pinky": 2},
    3: {"Thumb": 4, "Index": 5, "Pinky": 6},
}


def _detect_sessions(subj_id, task, nclass, model):
    """Return the list of session numbers (1..MAX_SESSIONS) that have data."""
    found = []
    for sess in range(1, MAX_SESSIONS + 1):
        try:
            folder = get_folder(subj_id, task, sess, nclass, model)
        except Exception:
            continue
        if os.path.isdir(folder) and glob.glob(os.path.join(folder, "*.mat")):
            found.append(sess)
    return found


def _process_recording(subj_id, task, session, nclass, model):
    """Load one recording, detect EOG/EMG artifact sources, and return
    {band_label: {task_name: sum_of_abs_corr_over_artifact_sources}}.

    Returns (results, artifact_comps) or (None, None) if there are no tasks."""
    folder = get_folder(subj_id, task, session, nclass, model)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    raw, _, _ = build_raw_from_mat_files(mat_files)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)

    # Reuse the subject's offline ICA (fit once, applied to every session).
    ica = ch._load_offline_ica(subj_id, task)

    src_raw      = ica.get_sources(raw)
    sfreq        = float(raw.info["sfreq"])
    task_vectors = build_task_vectors(raw)
    if not task_vectors:
        return None, None

    tasks_list = list(task_vectors.keys())
    n_tasks    = len(tasks_list)
    n_samples  = src_raw.n_times
    tasks_z    = np.zeros((n_tasks, n_samples))
    for t_idx, vec in enumerate(task_vectors.values()):
        if np.std(vec) > 0:
            tasks_z[t_idx] = zscore(vec)

    # Auto EOG/EMG artifact detection (same thresholds/logic as clean_ICA.py).
    artifact_comps = set()
    try:
        raw_det = raw.copy().filter(1.0, None, verbose=False)
        _, eog_idx, emg_idx = compute_exclusions(
            ica, raw_det, muscle_thresh=EMG_SLOPE_THRESH,
            eog_thresh=EOG_THRESHOLD, ch_names=raw_det.info["ch_names"],
        )
        artifact_comps = set(eog_idx) | set(emg_idx)
    except Exception as e:
        print(f"      [warn] EOG/EMG detection failed: {e}")

    artifact_comps = sorted(c for c in artifact_comps if 0 <= c < src_raw.ch_names.__len__())
    print(f"      artifact sources (EOG/EMG): {artifact_comps}")

    results = {}
    for band_label, l_freq, h_freq, _row in BANDS:
        if l_freq is None:
            band_sources = src_raw.get_data()
        else:
            band_sources = src_raw.copy().filter(l_freq, h_freq, verbose=False).get_data()
        corr = ch._compute_corr_matrix(band_sources, sfreq, tasks_z)   # (n_tasks, n_comp)
        per_task = {}
        for t_idx, tname in enumerate(tasks_list):
            # Sum |correlation| across ONLY the artefact-flagged sources: a
            # high sum means the task signal is well explained by artefacts
            # rather than by neural activity.
            per_task[tname] = float(sum(corr[t_idx, c] for c in artifact_comps))
        results[band_label] = per_task
    return results, artifact_comps


def fill_subject(subj_id, task="MI", model="Orig"):
    """Compute and write all cells of the subject's sheet in
    Correlation_Evolution.ods: for every available session, band and task,
    the sum of |correlation| over the auto-detected EOG/EMG artefact sources.
    """
    sheet = f"Sujet{subj_id}"
    print(f"\n=== Filling '{sheet}' in {ODS_PATH} ===")

    sessions = {ncl: _detect_sessions(subj_id, task, ncl, model) for ncl in (2, 3)}
    for ncl in (2, 3):
        print(f"  {ncl}-class sessions found: {sessions[ncl] or '(none)'}")

    if not sessions[2] and not sessions[3]:
        print("  Nothing to fill — no session data for this subject.")
        return

    # Collect writes grouped by sheet row so each row is written once.
    row_writes = defaultdict(list)
    for ncl in (2, 3):
        col_map = COL_MAP[ncl]
        for sess in sessions[ncl]:
            print(f"\n  -- {ncl}-class | Session {sess} --", flush=True)
            results, _ = _process_recording(subj_id, task, sess, ncl, model)
            if results is None:
                print("      [skip] no task annotations.")
                continue
            for band_label, _l, _h, base_row in BANDS:
                # Sheet layout: within a band's block, "Session 1" sits at
                # base_row, and each following session is one row further down.
                row = base_row + (sess - 1)
                per_task = results[band_label]
                for tname, col in col_map.items():
                    if tname in per_task:
                        row_writes[row].append((col, per_task[tname]))
                        print(f"      {band_label:10s} {tname:6s} -> "
                              f"R{row} C{col} = {per_task[tname]:.4f}")

    print(f"\n  Writing {sum(len(v) for v in row_writes.values())} value(s) "
          f"into '{sheet}' ...")
    for row in sorted(row_writes):
        write_to_ods(ODS_PATH, row, row_writes[row], table_name=sheet)
    print("  Done.")


def main():
    """CLI entry point: parse argv and fill the requested subject's sheet."""
    if len(sys.argv) < 2:
        print("Usage: python fill_correlation_evolution.py <subj> [<task> <model>]")
        sys.exit(1)
    subj_id = int(sys.argv[1])
    task    = sys.argv[2] if len(sys.argv) > 2 else "MI"
    model   = sys.argv[3] if len(sys.argv) > 3 else "Orig"
    fill_subject(subj_id, task, model)


if __name__ == "__main__":
    main()

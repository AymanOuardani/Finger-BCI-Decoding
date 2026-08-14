"""
run_all_clean_ica.py

Batch ICA cleaning using the fixed thresholds defined in config.py.
No interactive UI — EOG_THRESHOLD and EMG_SLOPE_THRESH are applied
automatically for every subject.

For interactive threshold tuning use clean_ICA.py instead.
The fitted ICA is loaded from cache when available (see get_ica_fif_path).

Usage:
    python run_all_clean_ica.py
    (edit the Configuration block below before running)
"""

import sys
import os
import glob
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions import (
    build_raw_from_mat_files,
    get_folder, get_offline_folder, get_ica_folder,
    get_ica_fif_path, fit_or_load_ica,
    compute_exclusions, save_ica_cleaned_mat_files,
)
from config import EOG_THRESHOLD, EMG_SLOPE_THRESH

# ── Configuration — edit these before running ────────────────────────────────
IS_OFFLINE  = True
SESSION_NUM = 1
NCLASS      = 2
TASK        = 'MI'
MODELTYPE   = 'Orig'    # ignored when IS_OFFLINE = True
SUBJECTS    = list(range(13, 17))
# ─────────────────────────────────────────────────────────────────────────────


def clean_subject(subj_id):
    """Run ICA cleaning for one subject: load raw .mat files, fit/load ICA,
    auto-detect EOG/EMG components to exclude, and save cleaned .mat files.

    Returns (n_excluded, eog_idx, emg_idx).
    """
    if IS_OFFLINE:
        folder = get_offline_folder(subj_id, TASK)
    else:
        folder = get_folder(subj_id, TASK, SESSION_NUM, NCLASS, MODELTYPE)

    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {folder}")

    print(f"  Loading {len(mat_files)} file(s) from:\n    {folder}")
    raw, _, _ = build_raw_from_mat_files(mat_files)

    if IS_OFFLINE:
        raw.notch_filter(np.arange(60, 501, 60), verbose=False)

    raw_filt = raw.copy().filter(1.0, None, verbose=False)

    if IS_OFFLINE:
        ica_cache_path = get_ica_fif_path(subj_id, TASK)
    else:
        ica_cache_path = get_ica_fif_path(subj_id, TASK, SESSION_NUM, NCLASS, MODELTYPE)

    ica = fit_or_load_ica(raw_filt, ica_cache_path)

    excluded, eog_idx, emg_idx = compute_exclusions(
        ica, raw_filt,
        muscle_thresh=EMG_SLOPE_THRESH,
        eog_thresh=EOG_THRESHOLD,
        ch_names=raw_filt.info['ch_names'],
    )
    ica.exclude = excluded

    print(f"  EOG : {eog_idx}")
    print(f"  EMG : {emg_idx}")
    print(f"  Total excluded : {len(excluded)} — {sorted(excluded)}")

    save_folder = get_ica_folder(folder)
    save_ica_cleaned_mat_files(ica, raw, mat_files, save_folder)
    print(f"  Saved to: {save_folder}")

    return len(excluded), eog_idx, emg_idx


def main():
    """Loop clean_subject() over all configured SUBJECTS and print a summary."""
    mode = ("Offline" if IS_OFFLINE
            else f"Sess{SESSION_NUM:02} {NCLASS}-class {MODELTYPE}")
    print(f"Batch ICA Cleaning — {TASK} | {mode}")
    print(f"EOG threshold : {EOG_THRESHOLD}  |  EMG slope threshold : {EMG_SLOPE_THRESH}")
    print(f"Subjects      : {SUBJECTS}")
    print()

    ok_count  = 0
    err_count = 0

    for subj_id in SUBJECTS:
        print(f'\n{"="*60}')
        print(f'Subject S{subj_id:02}')
        print(f'{"="*60}')

        try:
            n_excl, eog_idx, emg_idx = clean_subject(subj_id)
            status = f'OK — {n_excl} component(s) removed'
            ok_count += 1
        except Exception as e:
            status = f'Error — {e}'
            print(f'  [ERROR] {e}')
            err_count += 1

        print(f'Status : {status}')

    print(f'\n{"="*60}')
    print(f'Done — {ok_count} OK  |  {err_count} errors')


if __name__ == '__main__':
    main()

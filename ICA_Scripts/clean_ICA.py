import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions import (
    build_raw_from_mat_files,
    get_folder, get_offline_folder,
    save_ica_cleaned_mat_files,
    get_ica_folder,
    get_standard_args, get_offline_args,
    apply_ica_to_raw,
    get_ica_fif_path,
)

"""
clean_ICA.py

Interactive ICA artifact removal for both ONLINE and OFFLINE data.
The fitted ICA object is cached as a .fif file (see get_ica_fif_path in
Functions.py); subsequent runs reload it and skip the slow fitting step.

Flow:
    1. ICA fit loaded from cache or computed and saved.
    2. Adjust EOG / EMG thresholds with sliders.
    3. Click CONFIRM to lock in the exclusion list.
    4. Inspect Raw vs Cleaned side-by-side.
    5. Save *_ICA.mat files after closing the viewer.

Usage:
    Online:  python clean_ICA.py <subj> <sess> <nclass> <task> <model>
    Offline: python clean_ICA.py <subj> <task> OFF
"""


def clean_online(subj, sess, ncl, task, model):
    """Run the interactive ICA cleaning flow for one online subject/session:
    load the raw .mat files, fit/reload ICA, let the user validate exclusions
    interactively, then save the ICA-cleaned trials as *_ICA.mat files."""
    print("=" * 60)
    print(f"  Interactive Online ICA: S{subj:02} Sess{sess:02} | {task} {ncl}-class | {model}")
    print("=" * 60)

    folder    = get_folder(subj, task, sess, ncl, model)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {folder}")

    print(f"\nLoading {len(mat_files)} online file(s) from:\n  {folder}")
    raw, _, _ = build_raw_from_mat_files(mat_files)

    ica_cache_path = get_ica_fif_path(subj, task, sess, ncl, model)
    raw_clean, ica, _, _ = apply_ica_to_raw(
        raw,
        subj_id=subj, task=task, session=sess, nclass=ncl, model_type=model,
        ica_cache_path=ica_cache_path,
    )

    # Persist the validated exclusions into the ICA .fif so they are kept
    # permanently and reused by later steps (no need to re-validate).
    ica.save(ica_cache_path, overwrite=True)
    print(f"  Saved ICA with {len(ica.exclude)} excluded component(s) "
          f"{sorted(ica.exclude)} to:\n  {ica_cache_path}")

    save_folder = get_ica_folder(folder)
    print(f"\n[CONFIRM] Saving {len(ica.exclude)} component(s) removed to:\n  {save_folder}")
    save_ica_cleaned_mat_files(ica, raw, mat_files, save_folder)
    print("Done.")


def clean_offline(subj_id, task):
    """Run the interactive ICA cleaning flow for one offline subject/task:
    load the raw .mat files, notch-filter line noise, fit/reload ICA, let the
    user validate exclusions interactively, then save the cleaned trials."""
    print("=" * 60)
    print(f"  Interactive Offline ICA: S{subj_id:02} | {task}")
    print("=" * 60)

    folder    = get_offline_folder(subj_id, task)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {folder}")

    print(f"\nLoading {len(mat_files)} offline file(s) from:\n  {folder}")
    raw, _, _ = build_raw_from_mat_files(mat_files)
    # Notch out 60 Hz US mains hum and its harmonics (120, 180, ... 480 Hz)
    # before ICA fitting; the online path skips this since its data is
    # already preprocessed upstream.
    raw.notch_filter(np.arange(60, 501, 60))

    ica_cache_path = get_ica_fif_path(subj_id, task)  # session=0 → Offline
    raw_clean, ica, _, _ = apply_ica_to_raw(
        raw,
        subj_id=subj_id, task=task, session=0, nclass=0, model_type="Offline",
        ica_cache_path=ica_cache_path,
    )

    # Persist the validated exclusions into the ICA .fif so they are kept
    # permanently and reused by later steps (no need to re-validate).
    ica.save(ica_cache_path, overwrite=True)
    print(f"  Saved ICA with {len(ica.exclude)} excluded component(s) "
          f"{sorted(ica.exclude)} to:\n  {ica_cache_path}")

    save_folder = get_ica_folder(folder)
    print(f"\n[CONFIRM] Saving {len(ica.exclude)} component(s) removed to:\n  {save_folder}")
    save_ica_cleaned_mat_files(ica, raw, mat_files, save_folder)
    print("Done.")


def main():
    """CLI entry point: parses sys.argv to dispatch between offline and
    online interactive ICA cleaning."""
    try:
        if len(sys.argv) > 1 and sys.argv[-1].upper() == "OFF":
            sys.argv.pop()
            subj_id, task = get_offline_args(description="Interactive ICA - Offline")
            clean_offline(subj_id, task)
        else:
            subj, sess, ncl, task, model = get_standard_args(description="Interactive ICA - Online")
            clean_online(subj, sess, ncl, task, model)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

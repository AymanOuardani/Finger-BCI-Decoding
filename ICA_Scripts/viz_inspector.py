"""
viz_inspector.py

Interactive ICA inspector for online and offline sessions.
Loads (or fits) ICA, opens the slider UI to tune EOG/EMG thresholds,
then launches the Raw vs Cleaned side-by-side viewer on CONFIRM.
Does NOT save *_ICA.mat files to disk.

Usage:
    Online:  python viz_inspector.py <subj> <sess> <nclass> <task> <model>
    Offline: python viz_inspector.py <subj> <task> OFF
"""

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
    get_standard_args, get_offline_args,
    apply_ica_to_raw, get_ica_fif_path,
)


def inspect_online(subj, sess, ncl, task, model):
    """Load the online recording and open the interactive ICA threshold
    inspector (fit/load ICA, then apply_ica_to_raw drives the UI)."""
    print("=" * 60)
    print(f"  ICA Inspector: S{subj:02} Sess{sess:02} | {task} {ncl}-class | {model}")
    print("=" * 60)

    folder = get_folder(subj, task, sess, ncl, model)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {folder}")

    print(f"\nLoading online files from: {folder}")
    raw, _, _ = build_raw_from_mat_files(mat_files)

    ica_cache_path = get_ica_fif_path(subj, task, sess, ncl, model)
    apply_ica_to_raw(
        raw,
        subj_id=subj, task=task, session=sess, nclass=ncl, model_type=model,
        ica_cache_path=ica_cache_path,
    )

    print("\nInspector closed. No files were saved.")


def inspect_offline(subj_id, task):
    """Load the offline recording and open the interactive ICA threshold
    inspector (fit/load ICA, then apply_ica_to_raw drives the UI)."""
    print("=" * 60)
    print(f"  ICA Inspector: S{subj_id:02} | {task} (Offline)")
    print("=" * 60)

    folder = get_offline_folder(subj_id, task)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {folder}")

    print(f"\nLoading offline files from: {folder}")
    raw, _, _ = build_raw_from_mat_files(mat_files)
    raw.notch_filter(np.arange(60, 501, 60))

    ica_cache_path = get_ica_fif_path(subj_id, task)
    apply_ica_to_raw(
        raw,
        subj_id=subj_id, task=task, session=0, nclass=0, model_type="Offline",
        ica_cache_path=ica_cache_path,
    )

    print("\nInspector closed. No files were saved.")


def main():
    """CLI entry point: dispatch to the offline or online inspector based on
    the trailing 'OFF' argument."""
    try:
        if len(sys.argv) > 1 and sys.argv[-1].upper() == "OFF":
            sys.argv.remove(sys.argv[-1])  # drop "OFF" before argparse sees it
            subj_id, task = get_offline_args(description="ICA Inspector - Offline")
            inspect_offline(subj_id, task)
        else:
            subj, sess, ncl, task, model = get_standard_args(
                description="ICA Inspector - Online"
            )
            inspect_online(subj, sess, ncl, task, model)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

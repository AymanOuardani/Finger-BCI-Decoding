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
    fit_or_load_ica,
)

"""
clean_ica_offline.py

Same interactive ICA artifact removal as clean_ICA.py, but the ICA mixing
matrix is ALWAYS the subject's OFFLINE ICA, for any session (online or offline).
The offline ICA is loaded from its cache, or fitted ONCE from the offline
recording (60 Hz notch + 1 Hz highpass) if it does not exist yet, then applied
to the chosen recording. Exclusions are still chosen interactively per session
and saved to that recording's *_ICA.mat files.

For an ONLINE session the shared offline ICA .fif is NOT overwritten (the
mixing matrix is reused as-is). For the OFFLINE recording itself, the validated
exclusions are persisted into the offline .fif.

Usage:
    Online:  python clean_ica_offline.py <subj> <sess> <nclass> <task> <model>
    Offline: python clean_ica_offline.py <subj> <task> OFF
"""


def ensure_offline_ica(subj, task):
    """Return the path to the subject's OFFLINE ICA .fif, fitting it once from
    the offline recording (notch + 1 Hz highpass) if the cache is absent."""
    path = get_ica_fif_path(subj, task)          # offline filename form
    if os.path.exists(path):
        return path
    print("  Offline ICA not cached — fitting it once from the offline recording ...")
    folder = get_offline_folder(subj, task)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(
            f"No offline .mat files in {folder} to fit the reusable ICA.")
    raw, _, _ = build_raw_from_mat_files(mat_files)
    raw.notch_filter(np.arange(60, 501, 60))
    raw_filt = raw.copy().filter(1.0, None, verbose=False)
    fit_or_load_ica(raw_filt, path)              # fits and saves to `path`
    return path


def clean_online(subj, sess, ncl, task, model):
    print("=" * 60)
    print(f"  Interactive Online ICA (OFFLINE ICA): "
          f"S{subj:02} Sess{sess:02} | {task} {ncl}-class | {model}")
    print("=" * 60)

    folder    = get_folder(subj, task, sess, ncl, model)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {folder}")

    print(f"\nLoading {len(mat_files)} online file(s) from:\n  {folder}")
    raw, _, _ = build_raw_from_mat_files(mat_files)

    # ALWAYS use the subject's offline ICA mixing matrix.
    ica_cache_path = ensure_offline_ica(subj, task)
    print(f"  Using OFFLINE ICA: {ica_cache_path}")
    raw_clean, ica, _, _ = apply_ica_to_raw(
        raw,
        subj_id=subj, task=task, session=sess, nclass=ncl, model_type=model,
        ica_cache_path=ica_cache_path,
        erd_vlim=(-1, 1),          # ERD topo colorbar on [-1, 1] (this script only)
        erd_cmap="RdBu_r",         # +1 red, -1 blue
    )
    # NOTE: the shared offline .fif is intentionally NOT overwritten here — the
    # per-session exclusions are saved only to this session's *_ICA.mat files.

    save_folder = get_ica_folder(folder)
    print(f"\n[CONFIRM] Saving {len(ica.exclude)} component(s) removed to:\n  {save_folder}")
    save_ica_cleaned_mat_files(ica, raw, mat_files, save_folder)
    print("Done.")


def clean_offline(subj_id, task):
    print("=" * 60)
    print(f"  Interactive Offline ICA (OFFLINE ICA): S{subj_id:02} | {task}")
    print("=" * 60)

    folder    = get_offline_folder(subj_id, task)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {folder}")

    print(f"\nLoading {len(mat_files)} offline file(s) from:\n  {folder}")
    raw, _, _ = build_raw_from_mat_files(mat_files)
    raw.notch_filter(np.arange(60, 501, 60))

    ica_cache_path = get_ica_fif_path(subj_id, task)  # offline ICA itself
    raw_clean, ica, _, _ = apply_ica_to_raw(
        raw,
        subj_id=subj_id, task=task, session=0, nclass=0, model_type="Offline",
        ica_cache_path=ica_cache_path,
        erd_vlim=(-1, 1),          # ERD topo colorbar on [-1, 1] (this script only)
        erd_cmap="RdBu_r",         # +1 red, -1 blue
    )

    # Cleaning the offline recording itself: persist the validated exclusions
    # into the offline .fif.
    ica.save(ica_cache_path, overwrite=True)
    print(f"  Saved offline ICA with {len(ica.exclude)} excluded component(s) "
          f"{sorted(ica.exclude)} to:\n  {ica_cache_path}")

    save_folder = get_ica_folder(folder)
    print(f"\n[CONFIRM] Saving {len(ica.exclude)} component(s) removed to:\n  {save_folder}")
    save_ica_cleaned_mat_files(ica, raw, mat_files, save_folder)
    print("Done.")


def main():
    try:
        if len(sys.argv) > 1 and sys.argv[-1].upper() == "OFF":
            sys.argv.pop()
            subj_id, task = get_offline_args(description="Interactive ICA (offline ICA) - Offline")
            clean_offline(subj_id, task)
        else:
            subj, sess, ncl, task, model = get_standard_args(
                description="Interactive ICA (offline ICA) - Online")
            clean_online(subj, sess, ncl, task, model)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

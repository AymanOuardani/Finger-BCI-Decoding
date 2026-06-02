"""
ERD_Topo_Group.py

Group-averaged per-finger ERD topography across all 21 subjects.
Shows Before ICA vs After ICA comparison (same layout as ERD_Topo.py).
Reuses core functions from ERD_Topo.py.

Usage:
    Online:  python ERD_Topo_Group.py <sess> <nclass> <task> <model> <band>
    Offline: python ERD_Topo_Group.py <task> <band> OFF
    band: Alpha | Beta | Fullband

    Examples:
        python ERD_Topo_Group.py 1 2 MI Orig Fullband
        python ERD_Topo_Group.py MI Fullband OFF
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import mne
mne.set_log_level("WARNING")

from ERD_Topo import (
    load_and_preprocess, load_and_preprocess_ica,
    load_and_preprocess_offline, load_and_preprocess_ica_offline,
    build_epochs, build_epochs_no_reject, _compute_erd_band,
    plot_erd_comparison,
    BANDS_CONFIG, FINGERS, FINGERS_OFFLINE,
    TMAX_ONLINE, TMAX_OFFLINE, TASK_WIN_ONLINE, TASK_WIN_OFFLINE,
)
from Functions import write_to_ods
from config import RESULTS_ROOT, CORRUPTED_DATA, TRACKING_ODS, GROUP_ERD_SUBJECTS

ALL_SUBJECTS = GROUP_ERD_SUBJECTS


def _run_group(band_raw, finger_pairs, load_fn, load_ica_fn,
               title_prefix, save_name, task,
               tmax=TMAX_ONLINE, task_win=TASK_WIN_ONLINE):
    band = next((k for k in BANDS_CONFIG if k.lower() == band_raw.lower()), band_raw)
    if band not in BANDS_CONFIG:
        print(f"[ERROR] band must be one of: {list(BANDS_CONFIG.keys())}")
        sys.exit(1)

    ODS_SECTIONS = {
        ("ME", "Alpha"):    {1: "Thumb", 2: "Index", 3: "Middle", 4: "Pinky"},
        ("ME", "Beta"):     {8: "Thumb", 9: "Index", 10: "Middle", 11: "Pinky"},
        ("MI", "Alpha"):    {15: "Thumb", 16: "Index", 17: "Middle", 18: "Pinky"},
        ("MI", "Beta"):     {22: "Thumb", 23: "Index", 24: "Middle", 25: "Pinky"},
        ("MI", "Fullband"): {28: "Thumb", 29: "Index", 30: "Middle", 31: "Pinky"},
        ("ME", "Fullband"): {35: "Thumb", 36: "Index", 37: "Middle", 38: "Pinky"},
    }

    bands        = BANDS_CONFIG[band]
    finger_names = [name for name, _ in finger_pairs]

    raw_collection = {bl: {fn: [] for fn, _ in finger_pairs} for _, _, bl in bands}
    ica_collection = {bl: {fn: [] for fn, _ in finger_pairs} for _, _, bl in bands}
    missing_ica    = []

    n_ok = 0
    for subj in ALL_SUBJECTS:
        print(f"\nS{subj:02}...", end=" ", flush=True)

        try:
            raw    = load_fn(subj)
            # Apply the same 20 µV noisy-trial rejection as the After-ICA
            # path so the Before / After topomaps are computed on data of
            # comparable cleanliness.
            epochs = build_epochs(raw, finger_pairs, tmax=tmax)
        except Exception as e:
            print(f"skipped ({e})")
            continue

        if len(epochs) == 0:
            print("skipped (no epochs left after 20 µV rejection)")
            continue

        subj_ok = False
        for fmin, fmax, band_label in bands:
            erd_raw = _compute_erd_band(epochs, finger_pairs, fmin, fmax, task_win)
            for fname in finger_names:
                if erd_raw[fname] is not None:
                    raw_collection[band_label][fname].append(erd_raw[fname])
                    subj_ok = True

        try:
            raw_ica    = load_ica_fn(subj)
            epochs_ica = build_epochs(raw_ica, finger_pairs, tmax=tmax)
            if len(epochs_ica) > 0:
                for fmin, fmax, band_label in bands:
                    erd_ica = _compute_erd_band(epochs_ica, finger_pairs, fmin, fmax, task_win)
                    for fname in finger_names:
                        if erd_ica[fname] is not None:
                            ica_collection[band_label][fname].append(erd_ica[fname])

                    finger_rows = ODS_SECTIONS.get((task, band), {})
                    col = subj
                    for row_idx, fname in finger_rows.items():
                        if fname in finger_names and erd_ica.get(fname) is not None:
                            try:
                                write_to_ods(TRACKING_ODS, row_idx,
                                             [(col, float(np.mean(erd_ica[fname])))],
                                             "ERD Group Plots")
                            except Exception as e:
                                print(f"  [ODS] write failed S{subj:02} ICA {fname}: {e}")
        except FileNotFoundError:
            missing_ica.append(subj)
            print(f"  [!] ICA data missing — run clean_ICA.py for S{subj:02} first")

        if subj_ok:
            n_ok += 1
            print("OK")

    print(f"\n{'='*55}")
    print(f"  Subjects included: {n_ok} / {len(ALL_SUBJECTS)}")

    if missing_ica:
        print(f"\n[ACTION REQUIRED] ICA data missing for {len(missing_ica)} subject(s):")
        for s in missing_ica:
            print(f"  S{s:02}")
        print("\nRun clean_ICA.py for each subject listed above, then re-run this script:")
        print("  Online:  python ICA_Scripts/clean_ICA.py <subj> <sess> <nclass> <task> <model>")
        print("  Offline: python ICA_Scripts/clean_ICA.py <subj> <task> OFF")
        sys.exit(1)

    if n_ok == 0:
        print("[ERROR] No valid subjects — check paths and arguments.")
        sys.exit(1)

    # Get channel info from first valid subject.
    # We only need ep.info (channel layout) here, so use the no-rejection
    # path — this only affects which subject we *find* the info from, not the
    # info itself, and it makes the lookup robust even if a particular
    # subject's pre-ICA trials all exceed the 20 µV rule.
    info_subj = None
    for subj in ALL_SUBJECTS:
        try:
            raw = load_fn(subj)
            ep  = build_epochs_no_reject(raw, finger_pairs, tmax=tmax)
            if len(ep) > 0:
                info_subj = ep.info
                break
        except Exception:
            continue

    if info_subj is None:
        print("[ERROR] Could not obtain channel info.")
        sys.exit(1)

    for fmin, fmax, band_label in bands:
        erd_raw_avg = {}
        erd_ica_avg = {}
        for fname in finger_names:
            maps_raw = raw_collection[band_label][fname]
            maps_ica = ica_collection[band_label][fname]
            erd_raw_avg[fname] = np.mean(maps_raw, axis=0) if maps_raw else None
            erd_ica_avg[fname] = np.mean(maps_ica, axis=0) if maps_ica else None
            print(f"  {band_label} | {fname}: "
                  f"{len(maps_raw)} raw, {len(maps_ica)} ICA subject(s)")

        title     = f"{title_prefix}  |  {band_label}  (n={n_ok})"
        save_path = os.path.join(RESULTS_ROOT, "Group", save_name)

        plot_erd_comparison(erd_raw_avg, erd_ica_avg, band_label, finger_names,
                            info_subj, title, save_path)


def main():
    try:
        if len(sys.argv) > 1 and sys.argv[-1].upper() == "OFF":
            sys.argv.pop()  # strip "OFF"
            if len(sys.argv) < 3:
                print("Usage: python ERD_Topo_Group.py <task> <band> OFF")
                sys.exit(1)
            band_raw = sys.argv.pop()  # strip band
            task     = sys.argv[1]

            print("=" * 55)
            print(f"   Group ERD Topography  (n={len(ALL_SUBJECTS)} subjects)  [Offline]")
            print(f"   {task} | {band_raw}")
            print("=" * 55)

            _run_group(
                band_raw     = band_raw,
                finger_pairs = FINGERS_OFFLINE,
                load_fn      = lambda subj: load_and_preprocess_offline(subj, task),
                load_ica_fn  = lambda subj: load_and_preprocess_ica_offline(subj, task),
                title_prefix = f"Group ERD  |  {task}  |  Offline",
                save_name    = f"ERD_Group_{task}_Offline_{band_raw}_vs_ICA.png",
                task         = task,
                tmax         = TMAX_OFFLINE,
                task_win     = TASK_WIN_OFFLINE,
            )

        else:
            if len(sys.argv) < 6:
                print("Usage: python ERD_Topo_Group.py <sess> <nclass> <task> <model> <band>")
                print("  band: Alpha | Beta | Fullband")
                sys.exit(1)

            try:
                sess     = int(sys.argv[1])
                ncl      = int(sys.argv[2])
                task     = sys.argv[3]
                model    = sys.argv[4]
                band_raw = sys.argv[5]
            except (ValueError, IndexError):
                print("[ERROR] Invalid arguments.")
                sys.exit(1)

            if ncl not in FINGERS:
                print("[ERROR] nclass must be 2 or 3")
                sys.exit(1)

            model_label = "Base" if model == "Orig" else "Fine-tuned"

            print("=" * 55)
            print(f"   Group ERD Topography  (n={len(ALL_SUBJECTS)} subjects)")
            print(f"   Sess{sess:02} | {task} {ncl}-class | {model} | {band_raw}")
            print("=" * 55)

            def online_load(subj):
                if (subj, task, sess, ncl, model) in CORRUPTED_DATA:
                    raise RuntimeError("corrupted")
                return load_and_preprocess(subj, sess, ncl, task, model)

            _run_group(
                band_raw     = band_raw,
                finger_pairs = FINGERS[ncl],
                load_fn      = online_load,
                load_ica_fn  = lambda subj: load_and_preprocess_ica(subj, sess, ncl, task, model),
                title_prefix = (f"Group ERD  |  Sess{sess:02}  |  "
                                f"{task} {ncl}-class  |  {model_label}"),
                save_name    = (f"ERD_Group_Sess{sess:02}_{task}_{ncl}class_"
                                f"{model}_{band_raw}_vs_ICA.png"),
                task         = task,
            )

    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

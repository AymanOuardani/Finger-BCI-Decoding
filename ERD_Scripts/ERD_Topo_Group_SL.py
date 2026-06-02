"""
ERD_Topo_Group_SL.py

Group-averaged per-finger ERD topography — Sans Limite variant.
Same computation as ERD_Topo_Group.py; dynamic color scale via
plot_erd_comparison_sl (imported from ERD_Topo_SL.py).

Usage:
    Online:  python ERD_Topo_Group_SL.py <sess> <nclass> <task> <model> <band>
    Offline: python ERD_Topo_Group_SL.py <task> <band> OFF
    band: Alpha | Beta | Fullband

    Examples:
        python ERD_Topo_Group_SL.py 1 2 MI Orig Fullband
        python ERD_Topo_Group_SL.py MI Fullband OFF
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
    BANDS_CONFIG, FINGERS, FINGERS_OFFLINE,
    TMAX_ONLINE, TMAX_OFFLINE, TASK_WIN_ONLINE, TASK_WIN_OFFLINE,
)
from ERD_Topo_SL import plot_erd_comparison_sl
from config import RESULTS_ROOT, CORRUPTED_DATA, GROUP_ERD_SUBJECTS

ALL_SUBJECTS = GROUP_ERD_SUBJECTS


def _run_group(band_raw, finger_pairs, load_fn, load_ica_fn,
               title_prefix, save_name,
               tmax=TMAX_ONLINE, task_win=TASK_WIN_ONLINE):
    band = next((k for k in BANDS_CONFIG if k.lower() == band_raw.lower()), band_raw)
    if band not in BANDS_CONFIG:
        print(f"[ERROR] band must be one of: {list(BANDS_CONFIG.keys())}")
        sys.exit(1)

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
            # Apply the same 20 µV rejection to the Before-ICA path so the
            # Before / After topomaps are computed on comparable trial sets.
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

    info_subj = None
    for subj in ALL_SUBJECTS:
        try:
            raw = load_fn(subj)
            ep  = build_epochs(raw, finger_pairs, tmax=tmax)
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

        title     = f"{title_prefix}  |  {band_label}  (n={n_ok})  [Sans Limite]"
        save_path = os.path.join(RESULTS_ROOT, "Group", save_name)

        plot_erd_comparison_sl(erd_raw_avg, erd_ica_avg, band_label, finger_names,
                               info_subj, title, save_path)


def main():
    try:
        if len(sys.argv) > 1 and sys.argv[-1].upper() == "OFF":
            sys.argv.pop()
            if len(sys.argv) < 3:
                print("Usage: python ERD_Topo_Group_SL.py <task> <band> OFF")
                sys.exit(1)
            band_raw = sys.argv.pop()
            task     = sys.argv[1]

            print("=" * 55)
            print(f"   Group ERD Topography SL  (n={len(ALL_SUBJECTS)} subjects)  [Offline]")
            print(f"   {task} | {band_raw}")
            print("=" * 55)

            _run_group(
                band_raw     = band_raw,
                finger_pairs = FINGERS_OFFLINE,
                load_fn      = lambda subj: load_and_preprocess_offline(subj, task),
                load_ica_fn  = lambda subj: load_and_preprocess_ica_offline(subj, task),
                title_prefix = f"Group ERD  |  {task}  |  Offline",
                save_name    = f"ERD_Group_SL_{task}_Offline_{band_raw}_vs_ICA.png",
                tmax         = TMAX_OFFLINE,
                task_win     = TASK_WIN_OFFLINE,
            )

        else:
            if len(sys.argv) < 6:
                print("Usage: python ERD_Topo_Group_SL.py <sess> <nclass> <task> <model> <band>")
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
            print(f"   Group ERD Topography SL  (n={len(ALL_SUBJECTS)} subjects)")
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
                save_name    = (f"ERD_Group_SL_Sess{sess:02}_{task}_{ncl}class_"
                                f"{model}_{band_raw}_vs_ICA.png"),
            )

    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

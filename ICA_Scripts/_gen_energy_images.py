"""
_gen_energy_images.py  —  temp script
Deletes existing energy-correlation heatmaps and regenerates them.
Images only — no ODS writing, no per-recording artifact detection.
Artifact sources are fixed from the offline session for each subject.

Naming: EnergyCorr_S{XX}_{task}_Sess{SS}_{n}class_1_Orig_{Band}.png
        EnergyCorr_S{XX}_{task}_Sess{SS}_{n}class_2_Finetune_{Band}.png
"""

import os, sys, glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import corr_heatmap as ch
import fill_energy_table as fet
from Functions import (build_raw_from_mat_files, get_folder,
                       get_offline_folder, compute_exclusions)
from config import (TASK_LABELS, EOG_THRESHOLD, EMG_SLOPE_THRESH, CORRUPTED_DATA)

SUBJECTS = [2,3,4,6,7,8,9,10,11,13,14,15,16,17,18,19]
TASK     = "MI"

ENERGY_CORR_DIR = fet.ENERGY_CORR_DIR


# ── 1. Show current structure then delete all .png ────────────────────────────

def _purge_images():
    if not os.path.isdir(ENERGY_CORR_DIR):
        print(f"  [info] folder does not exist yet: {ENERGY_CORR_DIR}")
        return
    pngs = sorted(glob.glob(os.path.join(ENERGY_CORR_DIR, "**", "*.png"),
                            recursive=True))
    print(f"Current images ({len(pngs)}):")
    for p in pngs:
        print(f"  {p}")
    print(f"\nDeleting {len(pngs)} image(s) …")
    for p in pngs:
        os.remove(p)
    for root, dirs, files in os.walk(ENERGY_CORR_DIR, topdown=False):
        if not os.listdir(root) and root != ENERGY_CORR_DIR:
            os.rmdir(root)
    print("  Done.\n")


# ── 2. Offline artifacts — computed once per subject ──────────────────────────

def _offline_artifacts(subj_id, task):
    try:
        off_mats = sorted(glob.glob(
            os.path.join(get_offline_folder(subj_id, task), "*.mat")))
        if not off_mats:
            return set()
        raw_off, _, _ = build_raw_from_mat_files(off_mats)
        raw_off.notch_filter(np.arange(60, 501, 60), verbose=False)
        ica_off  = ch._load_offline_ica(subj_id, task)
        raw_filt = raw_off.copy().filter(1.0, None, verbose=False)
        _, eog, emg = compute_exclusions(
            ica_off, raw_filt, muscle_thresh=EMG_SLOPE_THRESH,
            eog_thresh=EOG_THRESHOLD, ch_names=raw_filt.info["ch_names"],
        )
        art = set(eog) | set(emg)
        print(f"  Offline artifacts ({len(art)}): {sorted(art)}")
        return art
    except Exception as e:
        print(f"  [warn] offline artifact detection failed: {e}")
        return set()


# ── 3. Load energies for one recording (no artifact detection) ────────────────

def _load_energies(subj_id, task, sess, ncl, model):
    """Return (E_per_band, trial_classes, n_comp) or raise."""
    folder    = get_folder(subj_id, task, sess, ncl, model)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    raw, _, _ = build_raw_from_mat_files(mat_files)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)

    ica     = ch._load_offline_ica(subj_id, task)
    src_raw = ica.get_sources(raw)
    n_comp  = len(src_raw.ch_names)

    trials = fet._chronological_trials(raw)
    if not trials:
        return None

    band_data = []
    for _lbl, lo, hi in fet.BANDS:
        if lo is None:
            band_data.append(src_raw.get_data())
        else:
            band_data.append(src_raw.copy().filter(lo, hi, verbose=False).get_data())

    n_trials      = len(trials)
    trial_classes = [cls for cls, _s, _e in trials]
    E_per_band    = [np.empty((n_trials, n_comp)) for _ in fet.BANDS]

    for t_idx, (cls, s, e) in enumerate(trials):
        for bi, bd in enumerate(band_data):
            E_per_band[bi][t_idx] = np.sum(bd[:, s:e] ** 2, axis=1)

    return E_per_band, trial_classes, n_comp


# ── 4. Generate heatmaps for one subject ─────────────────────────────────────

def _gen_subject(subj_id, task):
    print(f"\n{'='*55}\n  S{subj_id:02d}\n{'='*55}")
    groups = fet._detect_groups(subj_id, task)
    if not groups:
        print("  No data found, skipping.")
        return

    art = _offline_artifacts(subj_id, task)

    for (sess, ncl), models in sorted(groups.items()):
        print(f"\n  Sess{sess:02d}  {ncl}class  ({', '.join(models)})")
        for model in models:
            if (subj_id, task, sess, ncl, model) in CORRUPTED_DATA:
                continue
            print(f"    -- {model} --", flush=True)
            try:
                result = _load_energies(subj_id, task, sess, ncl, model)
            except Exception as ex:
                print(f"      [SKIP] {ex}", flush=True)
                continue
            if result is None:
                print("      [SKIP] no trials", flush=True)
                continue

            E_per_band, trial_classes, n_comp = result
            tasks     = [t for t in TASK_LABELS if t in set(trial_classes)]
            model_tag = "1_Orig" if model == "Orig" else "2_Finetune"
            mode      = f"Sess{sess:02d} {ncl}class {model} | EnergyCorr"

            for bi, (band_label, _lo, _hi) in enumerate(fet.BANDS):
                mat  = fet._energy_task_corr(E_per_band[bi], trial_classes, tasks)
                path = os.path.join(
                    ENERGY_CORR_DIR, f"S{subj_id:02d}", band_label,
                    f"{ncl}class",
                    f"EnergyCorr_S{subj_id:02d}_{task}_Sess{sess:02d}"
                    f"_{ncl}class_{model_tag}_{band_label}.png")
                try:
                    fet._plot_signed_heatmap(
                        mat, tasks, subj_id, task, mode, band_label,
                        path, excluded=art)
                    print(f"      saved {band_label}: {os.path.basename(path)}")
                except Exception as ex:
                    print(f"      [warn] {band_label}: {ex}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _purge_images()
    for subj in SUBJECTS:
        _gen_subject(subj, TASK)
    print("\n\nAll done.")

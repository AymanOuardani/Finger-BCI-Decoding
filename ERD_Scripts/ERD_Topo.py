"""
ERD_Topo.py

Single-subject per-finger ERD topography using Morlet wavelets.
Reproduces the per-finger topomap layout of Fig. 1C/D from Ding et al. (2025).

Method (paper-exact):
  Preprocessing : notch 60 Hz → CAR → bandpass 2-45 Hz → downsample 100 Hz
  Segmentation  : -2 s to +3 s (online) or -2 s to +5 s (offline)
  Rejection     : global std (channels × time) > 20 µV
  Power         : Morlet wavelets, per-channel, per-trial
  Baseline      : [-1, 0] s before onset
  Task window   : [+0.5, +3] s (online) or [+0.5, +5] s (offline)
  ERD           : -(P_task - P_base) / P_base   (fraction; positive = desynchronization)

Usage:
    Online:        python ERD_Topo.py <subj> <sess> <nclass> <task> <model> <band>
    Offline:       python ERD_Topo.py <subj> <task> <band> OFF
    Offline group: python ERD_Topo.py <task> <band> OFF GROUP
    band: Alpha | Beta | Fullband

    Examples:
        python ERD_Topo.py 9 1 2 MI Orig Fullband
        python ERD_Topo.py 9 MI Fullband OFF
        python ERD_Topo.py MI Beta OFF GROUP

GROUP flag: can appear anywhere in argv; loops over ALL_SUBJECTS from config.py,
saves each figure to Itération 4/Week 7/W7D2 - 12-05 as ERD_<task>_Offline_<band>_<subj>.png,
skips subjects whose data folder is missing, does not call plt.show().
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import mne
mne.set_log_level("WARNING")
from mne.time_frequency import tfr_array_morlet

from Functions import (
    build_raw_from_mat_files,
    get_folder, get_offline_folder,
    to_ica_path,
    get_standard_args, get_offline_args,
)
from config import RESULTS_ROOT, MORLET_N_CYCLES, ALL_SUBJECTS


# ── Constants ─ paper: Ding et al. 2025, "Electrophysiological analysis" ─────
# Pipeline (per the paper):
#   CAR → resample 100 Hz → bandpass 2–30 Hz → ICA → segment −2 s..end
#   → reject trials with std > 20 µV → Morlet (alpha 8–13, beta 13–30)
#   → task window 0.5..end, baseline −1..0, ERD = (Pc − Rc) / Rc × 100 %
SFREQ_DS   = 100
FILT_LO    = 2.0
FILT_HI    = 30.0   # paper-exact: bandpass 2–30 Hz
TMIN             = -2.0
TMAX_ONLINE      = 3.0
TMAX_OFFLINE     = 5.0
BASELINE         = (-1.0, 0.0)
TASK_WIN_ONLINE  = (0.5, 3.0)
TASK_WIN_OFFLINE = (0.5, 5.0)
STD_THRESH       = 20e-6   # 20 µV — single trial-level std (all channels × all samples)

FINGERS = {
    2: [("Thumb", 1), ("Pinky", 4)],
    3: [("Thumb", 1), ("Index", 2), ("Pinky", 4)],
}
FINGERS_OFFLINE = [("Thumb", 1), ("Index", 2), ("Middle", 3), ("Pinky", 4)]

# Fullband is an analysis-side extension not present in the paper. Its upper
# bound is clamped to 30 Hz so it stays inside the paper-exact 2–30 Hz filter.
BANDS_CONFIG = {
    "Fullband": [(4,  30, "Fullband (4-30 Hz)")],
    "Alpha":    [(8,  13, "Alpha (8-13 Hz)")],
    "Beta":     [(13, 30, "Beta (13-30 Hz)")],
}


# ── Data pipeline ─────────────────────────────────────────────────────────────

def _preprocess_raw(raw):
    raw = raw.copy()
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    raw.filter(FILT_LO, FILT_HI, verbose=False)
    raw.resample(SFREQ_DS, verbose=False)
    return raw


def load_and_preprocess(subj, sess, ncl, task, model):
    folder    = get_folder(subj, task, sess, ncl, model)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in:\n  {folder}")
    print(f"\nLoading {len(mat_files)} file(s) from:\n  {folder}")

    raw, _, _ = build_raw_from_mat_files(mat_files)
    return _preprocess_raw(raw)


def load_and_preprocess_ica(subj, sess, ncl, task, model):
    folder_ica = to_ica_path(get_folder(subj, task, sess, ncl, model))
    mat_files  = sorted(glob.glob(os.path.join(folder_ica, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(
            f"No ICA .mat files in:\n  {folder_ica}\n  Run clean_ICA.py first.")
    print(f"\nLoading {len(mat_files)} ICA file(s) from:\n  {folder_ica}")
    raw, _, _ = build_raw_from_mat_files(mat_files)
    return _preprocess_raw(raw)


def load_and_preprocess_offline(subj_id, task):
    folder    = get_offline_folder(subj_id, task)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in:\n  {folder}")
    print(f"\nLoading {len(mat_files)} offline file(s) from:\n  {folder}")
    raw, _, _ = build_raw_from_mat_files(mat_files)
    return _preprocess_raw(raw)


def load_and_preprocess_ica_offline(subj_id, task):
    folder_ica = to_ica_path(get_offline_folder(subj_id, task))
    mat_files  = sorted(glob.glob(os.path.join(folder_ica, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(
            f"No ICA .mat files in:\n  {folder_ica}\n  Run clean_ICA.py first.")
    print(f"\nLoading {len(mat_files)} ICA offline file(s) from:\n  {folder_ica}")
    raw, _, _ = build_raw_from_mat_files(mat_files)
    return _preprocess_raw(raw)


def build_epochs_no_reject(raw, finger_pairs, tmax=TMAX_ONLINE):
    event_id = {name: eid for name, eid in finger_pairs}
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    if len(events) == 0:
        raise RuntimeError(
            "No matching events found in annotations. "
            "Check that nclass/task match the loaded folder.")
    return mne.Epochs(
        raw, events, event_id=event_id,
        tmin=TMIN, tmax=tmax,
        baseline=None, preload=True, verbose=False,
    )


def build_epochs(raw, finger_pairs, tmax=TMAX_ONLINE):
    event_id = {name: eid for name, eid in finger_pairs}

    # Events extracted from annotations at the resampled sfreq
    events, _ = mne.events_from_annotations(
        raw, event_id=event_id, verbose=False)

    if len(events) == 0:
        raise RuntimeError(
            "No matching events found in annotations. "
            "Check that nclass/task match the loaded folder.")

    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=TMIN, tmax=tmax,
        baseline=None, preload=True, verbose=False,
    )

    # Trial rejection — paper-literal (Ding et al. 2025):
    #   "Trials with a standard deviation above 20 µV were excluded"
    # Singular "a standard deviation" → one number per trial = the std of
    # the whole trial (all channels × all time samples, flattened). Any
    # trial whose overall std exceeds 20 µV is dropped, exactly as the
    # paper describes — no per-channel aggregation, no extra thresholds.
    data       = epochs.get_data()                          # (n_epochs, n_chan, n_times)
    trial_std  = data.reshape(len(data), -1).std(axis=1)    # one std per trial
    bad        = trial_std > STD_THRESH
    if bad.sum():
        worst = float(trial_std.max()) * 1e6
        print(f"  Rejected {int(bad.sum())} noisy trial(s) "
              f"(trial std > {STD_THRESH * 1e6:.0f} µV; worst trial std={worst:.1f} µV)"
              f"  -> {len(epochs) - int(bad.sum())} remaining")
    else:
        worst = float(trial_std.max()) * 1e6 if len(trial_std) else 0.0
        print(f"  No trials rejected "
              f"(worst trial std = {worst:.1f} µV, threshold = "
              f"{STD_THRESH * 1e6:.0f} µV)")
    return epochs[~bad]


def compute_band_erd(epoch_data, times, fmin, fmax, task_win=TASK_WIN_ONLINE):
    """
    Morlet TFR → per-finger ERD averaged over trials and band frequencies.
    Returns (n_chan,) array as fraction (positive = desynchronization).
    """
    freqs    = np.arange(fmin, fmax + 1, dtype=float)
    n_cycles = MORLET_N_CYCLES   # 7 cycles — paper-exact (Ding et al. 2025)

    # power: (n_epochs, n_chan, n_freqs, n_times)
    power = tfr_array_morlet(
        epoch_data, sfreq=SFREQ_DS, freqs=freqs,
        n_cycles=n_cycles, output="power", verbose=False,
    )

    bl_mask   = (times >= BASELINE[0])  & (times <  BASELINE[1])
    task_mask = (times >= task_win[0])  & (times <= task_win[1])

    Rc  = power[:, :, :, bl_mask  ].mean(axis=-1)   # (n_epochs, n_chan, n_freqs)
    Pc  = power[:, :, :, task_mask].mean(axis=-1)

    ERD = (Pc - Rc) / (Rc + 1e-12)   # fraction, negative = ERD
    print(f"    Rc mean={Rc.mean():.4f}  Pc mean={Pc.mean():.4f}  ratio={Pc.mean()/Rc.mean():.3f}")

    return ERD.mean(axis=(0, 2))   # negative = desynchronization → (n_chan,)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_erd_topo(erd_per_band, finger_names, info, title, save_path=None):
    """
    erd_per_band : {band_label: {finger_name: (n_chan,) array or None}}
    """
    band_labels = list(erd_per_band.keys())
    n_rows = len(band_labels)
    n_cols = len(finger_names)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols + 1.2, 2.8 * n_rows + 0.6),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.88, right=0.88)

    vmin, vmax = -0.5, 0.0

    for r, band_label in enumerate(band_labels):
        for c, fname in enumerate(finger_names):
            ax  = axes[r, c]
            erd = erd_per_band[band_label].get(fname)

            if r == 0:
                ax.set_title(fname, fontsize=10, fontweight="bold", pad=4)
            if c == 0:
                ax.set_ylabel(band_label, fontsize=9, labelpad=6)

            if erd is None:
                ax.axis("off")
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                continue

            mne.viz.plot_topomap(
                erd, info,
                axes=ax,
                cmap="Blues_r",          # dark blue at vmin, white at vmax
                vlim=(vmin, vmax),
                contours=0,
                extrapolate="head",
                sphere=(0., 0., 0., 0.095),
                outlines="head",
                show=False,
                sensors=False,
            )

    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.97)

    # Shared colorbar — fixed position so it doesn't fight tight_layout
    cbar_ax = fig.add_axes([0.91, 0.15, 0.025, 0.65])
    sm = plt.cm.ScalarMappable(
        cmap="Blues_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, label="ERD (fraction)")
    ticks = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {save_path}")

    plt.show(block=True)


def _compute_erd_band(epochs, finger_pairs, fmin, fmax, task_win=TASK_WIN_ONLINE):
    """Compute per-finger ERD for one band. Returns {finger_name: (n_chan,) or None}."""
    result = {}
    for fname, feid in finger_pairs:
        mask = epochs.events[:, 2] == feid
        fepo = epochs[mask]
        if len(fepo) == 0:
            print(f"  {fname}: no epochs — skipping")
            result[fname] = None
            continue
        print(f"  {fname}: {len(fepo)} trial(s) -> Morlet TFR...", end=" ", flush=True)
        erd = compute_band_erd(fepo.get_data(), fepo.times, fmin, fmax, task_win)
        result[fname] = erd
        print(f"ERD [{erd.min():.3f}, {erd.max():.3f}]")
    return result


def plot_erd_comparison(erd_raw, erd_ica, band_label, finger_names, info,
                        title, save_path=None, show=True):
    """
    Two-row comparison figure: Before ICA (top) vs After ICA (bottom).
    erd_raw, erd_ica : {finger_name: (n_chan,) array or None}
    """
    n_cols = len(finger_names)

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(2.5 * n_cols + 1.2, 5.6 + 0.6),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.88, right=0.88, hspace=0.1)

    vmin, vmax = -0.5, 0.0

    row_labels = [f"Before ICA\n{band_label}", f"After ICA\n{band_label}"]
    row_dicts  = [erd_raw, erd_ica]

    for r, (row_label, erd_dict) in enumerate(zip(row_labels, row_dicts)):
        for c, fname in enumerate(finger_names):
            ax  = axes[r, c]
            erd = erd_dict.get(fname)

            if r == 0:
                ax.set_title(fname, fontsize=10, fontweight="bold", pad=4)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=9, labelpad=6)

            if erd is None:
                ax.axis("off")
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                continue

            mne.viz.plot_topomap(
                erd, info,
                axes=ax,
                cmap="Blues_r",          # dark blue at vmin, white at vmax
                vlim=(vmin, vmax),
                contours=0,
                extrapolate="head",
                sphere=(0., 0., 0., 0.095),
                outlines="head",
                show=False,
                sensors=False,
            )

    # Dashed separator between the two rows
    fig.add_artist(plt.Line2D(
        [0.07, 0.88], [0.5, 0.5],
        transform=fig.transFigure,
        color="gray", linewidth=0.8, linestyle="--",
    ))

    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.97)

    cbar_ax = fig.add_axes([0.91, 0.15, 0.025, 0.65])
    sm = plt.cm.ScalarMappable(
        cmap="Blues_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax, label="ERD (fraction)")
    ticks = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {save_path}")

    if show:
        plt.show(block=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def _run(band_raw, finger_pairs, load_fn, load_ica_fn, title_prefix, save_name,
         tmax=TMAX_ONLINE, task_win=TASK_WIN_ONLINE, show=True):
    """Shared ERD computation + comparison plot for both online and offline."""
    band = next((k for k in BANDS_CONFIG if k.lower() == band_raw.lower()), band_raw)
    if band not in BANDS_CONFIG:
        print(f"[ERROR] band must be one of: {list(BANDS_CONFIG.keys())}")
        sys.exit(1)

    finger_names = [name for name, _ in finger_pairs]

    try:
        raw    = load_fn()
        # Apply the same 20 µV noisy-trial rejection on the Before-ICA path
        # so that Before-vs-After topomaps are computed on the same kind of
        # filtered trials (otherwise pre-ICA outliers leak into the topomap).
        epochs = build_epochs(raw, finger_pairs, tmax=tmax)
        if len(epochs) == 0:
            print("\n[ERROR] All Before-ICA trials rejected (mean channel std > 20 µV).")
            sys.exit(1)
        data = epochs.get_data()
        print(f"  Signal: mean_abs={np.abs(data).mean()*1e6:.2f} µV  std={data.std()*1e6:.2f} µV")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print(f"\n  Epochs (Before ICA, with rejection): {len(epochs)}")
    for eid, name in {eid: n for n, eid in finger_pairs}.items():
        print(f"    {name}: {(epochs.events[:, 2] == eid).sum()}")

    epochs_ica = None
    try:
        raw_ica    = load_ica_fn()
        epochs_ica = build_epochs(raw_ica, finger_pairs, tmax=tmax)
        if len(epochs_ica) == 0:
            print("\n  [WARN] All ICA epochs rejected — After-ICA row will show 'no data'")
            epochs_ica = None
        else:
            print(f"\n  Epochs (After ICA): {len(epochs_ica)}")
    except FileNotFoundError:
        print("\n  [INFO] ICA data not found — After-ICA row will show 'no data'")
        print("         Run ICA_Scripts/clean_ICA.py first.")

    for fmin, fmax, band_label in BANDS_CONFIG[band]:
        print(f"\nComputing {band_label} (Before ICA)...")
        erd_raw = _compute_erd_band(epochs, finger_pairs, fmin, fmax, task_win)

        if epochs_ica is not None:
            print(f"Computing {band_label} (After ICA)...")
            erd_ica = _compute_erd_band(epochs_ica, finger_pairs, fmin, fmax, task_win)
        else:
            erd_ica = {fname: None for fname, _ in finger_pairs}

        title     = f"{title_prefix}  |  {band_label}"
        save_path = os.path.join(RESULTS_ROOT, save_name)

        plot_erd_comparison(erd_raw, erd_ica, band_label, finger_names, epochs.info,
                            title, save_path, show=show)
        if not show:
            plt.close('all')


def main():
    group_mode = 'GROUP' in [a.upper() for a in sys.argv]
    if group_mode:
        sys.argv = [a for a in sys.argv if a.upper() != 'GROUP']

    GROUP_SAVE_DIR = (
        r"C:\Users\aymen\Desktop\5 Years Of Engineering\DATASIM\DATASIM - Ouardani"
        r"\15 - STING (Début 30-03 et Fin 14-08)\Suivi du Stage"
        r"\Itération 4\Week 7\W7D2 - 12-05"
    )

    try:
        if len(sys.argv) > 1 and sys.argv[-1].upper() == "OFF":
            sys.argv.pop()
            if group_mode:
                if len(sys.argv) < 3:
                    print("Usage: python ERD_Topo.py <task> <band> OFF GROUP")
                    sys.exit(1)
                band_raw = sys.argv.pop()
                task = sys.argv[1]
                for subj_id in ALL_SUBJECTS:
                    print(f"\n{'='*55}")
                    print(f"   ERD Topography  (Before vs After ICA)  [Offline] [GROUP]")
                    print(f"   S{subj_id:02} | {task} | {band_raw}")
                    print(f"{'='*55}")
                    save_path = os.path.join(
                        GROUP_SAVE_DIR,
                        f"ERD_{task}_Offline_{band_raw}_{subj_id:02d}.png",
                    )
                    try:
                        _run(
                            band_raw=band_raw,
                            finger_pairs=FINGERS_OFFLINE,
                            load_fn=lambda s=subj_id: load_and_preprocess_offline(s, task),
                            load_ica_fn=lambda s=subj_id: load_and_preprocess_ica_offline(s, task),
                            title_prefix=f"ERD — S{subj_id:02}  |  {task}  |  Offline",
                            save_name=save_path,
                            tmax=TMAX_OFFLINE,
                            task_win=TASK_WIN_OFFLINE,
                            show=False,
                        )
                    except SystemExit:
                        print(f"  [WARN] S{subj_id:02} skipped.")
                        plt.close('all')
                return

            if len(sys.argv) < 4:
                print("Usage: python ERD_Topo.py <subj> <task> <band> OFF")
                sys.exit(1)
            band_raw = sys.argv.pop()            # strip band (now last positional)
            subj_id, task = get_offline_args(description="ERD Topography - Offline")

            print("=" * 55)
            print(f"   ERD Topography  (Before vs After ICA)  [Offline]")
            print(f"   S{subj_id:02} | {task} | {band_raw}")
            print("=" * 55)

            _run(
                band_raw      = band_raw,
                finger_pairs  = FINGERS_OFFLINE,  # offline: all 4 fingers (Thumb/Index/Middle/Pinky)
                load_fn       = lambda: load_and_preprocess_offline(subj_id, task),
                load_ica_fn   = lambda: load_and_preprocess_ica_offline(subj_id, task),
                title_prefix  = f"ERD — S{subj_id:02}  |  {task}  |  Offline",
                save_name     = os.path.join(
                    GROUP_SAVE_DIR,
                    f"ERD_{task}_Offline_{band_raw}_{subj_id:02d}.png",
                ),
                tmax          = TMAX_OFFLINE,
                task_win      = TASK_WIN_OFFLINE,
            )

        else:
            subj, sess, ncl, task, model = get_standard_args(
                description="ERD Topography - Online")

            if ncl not in FINGERS:
                print("[ERROR] nclass must be 2 or 3")
                sys.exit(1)
            if len(sys.argv) < 7:
                print("Usage: python ERD_Topo.py <subj> <sess> <nclass> <task> <model> <band>")
                sys.exit(1)
            band_raw = sys.argv[6]

            model_label = "Base" if model == "Orig" else "Fine-tuned"

            print("=" * 55)
            print(f"   ERD Topography  (Before vs After ICA)")
            print(f"   S{subj:02} | Sess{sess:02} | {task} {ncl}-class | {model} | {band_raw}")
            print("=" * 55)

            _run(
                band_raw      = band_raw,
                finger_pairs  = FINGERS[ncl],
                load_fn       = lambda: load_and_preprocess(subj, sess, ncl, task, model),
                load_ica_fn   = lambda: load_and_preprocess_ica(subj, sess, ncl, task, model),
                title_prefix  = (f"ERD — S{subj:02}  |  Sess{sess:02}  |  "
                                 f"{task} {ncl}-class  |  {model_label}"),
                save_name     = os.path.join(
                    f"Sujet {subj}",
                    f"ERD_S{subj:02}_Sess{sess:02}_{task}_{ncl}class_{model}_{band_raw}_vs_ICA.png",
                ),
            )

    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

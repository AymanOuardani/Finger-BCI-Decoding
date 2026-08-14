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

# Paper-defined bands (Ding et al. 2025, Sec. "Electrophysiological analysis"):
#   "Morlet wavelets were used to extract the average power in the alpha
#    (8–13 Hz) and beta (13–30 Hz) bands"
# `Fullband` is an analysis-side extension kept for convenience (e.g. the
# SHOW ERD button in clean_ICA.py uses it). Its upper bound is clamped to
# 30 Hz so it stays inside the paper-exact 2–30 Hz bandpass.
BANDS_CONFIG = {
    "Alpha":    [(8,  13, "Alpha (8-13 Hz)")],   # paper
    "Beta":     [(13, 30, "Beta (13-30 Hz)")],   # paper
    "Fullband": [(4,  30, "Fullband (4-30 Hz)")],  # extension (not in paper)
}


# ── Data pipeline ─────────────────────────────────────────────────────────────

def _preprocess_raw(raw):
    """
    Paper-exact preprocessing (Ding et al. 2025):

      "The raw EEG data were re-referenced to the common average,
       downsampled to 100 Hz, and bandpass filtered between 2 and 30 Hz."

    Order is taken literally from the text: CAR, then downsample, then
    bandpass. No notch filter — the paper does not mention one, and the
    2–30 Hz bandpass already removes any 60 Hz line noise.
    """
    raw = raw.copy()
    raw.set_eeg_reference("average", verbose=False)   # 1. common average reference
    raw.resample(SFREQ_DS, verbose=False)             # 2. downsample to 100 Hz
    raw.filter(FILT_LO, FILT_HI, verbose=False)       # 3. bandpass 2–30 Hz
    return raw


def load_and_preprocess(subj, sess, ncl, task, model):
    """Load raw (non-ICA) online .mat files for one subject/session and run
    the paper-exact preprocessing pipeline (CAR -> downsample -> bandpass)."""
    folder    = get_folder(subj, task, sess, ncl, model)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in:\n  {folder}")
    print(f"\nLoading {len(mat_files)} file(s) from:\n  {folder}")

    raw, _, _ = build_raw_from_mat_files(mat_files)
    return _preprocess_raw(raw)


def load_and_preprocess_ica(subj, sess, ncl, task, model):
    """Load the ICA-cleaned online .mat files (produced by clean_ICA.py) for
    one subject/session and run the paper-exact preprocessing pipeline."""
    folder_ica = to_ica_path(get_folder(subj, task, sess, ncl, model))
    mat_files  = sorted(glob.glob(os.path.join(folder_ica, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(
            f"No ICA .mat files in:\n  {folder_ica}\n  Run clean_ICA.py first.")
    print(f"\nLoading {len(mat_files)} ICA file(s) from:\n  {folder_ica}")
    raw, _, _ = build_raw_from_mat_files(mat_files)
    return _preprocess_raw(raw)


def load_and_preprocess_offline(subj_id, task):
    """Load raw (non-ICA) offline .mat files for one subject/task and run
    the paper-exact preprocessing pipeline."""
    folder    = get_offline_folder(subj_id, task)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in:\n  {folder}")
    print(f"\nLoading {len(mat_files)} offline file(s) from:\n  {folder}")
    raw, _, _ = build_raw_from_mat_files(mat_files)
    return _preprocess_raw(raw)


def load_and_preprocess_ica_offline(subj_id, task):
    """Load the ICA-cleaned offline .mat files for one subject/task and run
    the paper-exact preprocessing pipeline."""
    folder_ica = to_ica_path(get_offline_folder(subj_id, task))
    mat_files  = sorted(glob.glob(os.path.join(folder_ica, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(
            f"No ICA .mat files in:\n  {folder_ica}\n  Run clean_ICA.py first.")
    print(f"\nLoading {len(mat_files)} ICA offline file(s) from:\n  {folder_ica}")
    raw, _, _ = build_raw_from_mat_files(mat_files)
    return _preprocess_raw(raw)


def build_epochs_no_reject(raw, finger_pairs, tmax=TMAX_ONLINE):
    """Build per-finger epochs from continuous data WITHOUT the 20 uV
    trial-rejection step (used only when the caller just needs channel/info
    metadata and doesn't care whether trials would otherwise be dropped)."""
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
    """Build per-finger epochs from continuous data and apply the paper's
    20 uV trial-rejection rule (see inline comment below for the exact rule)."""
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


# NOTE: the old per-finger `compute_band_erd(...)` helper has been folded
# directly into `_compute_erd_band(...)` below so we can share a single
# session-level baseline Rc across all fingers, which is what the paper
# (Ding et al. 2025, Eq. 3) actually says — see the comment in the new
# function for the audit trail.


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
    """
    Compute per-finger ERD topomaps for one frequency band — paper-exact
    (Ding et al. 2025, Eq. 3).

        ERDc = (Pc - Rc) / Rc × 100 %

    Where:
      * Pc = average power in the alpha/beta band during the task window
             — computed per FINGER (each finger's own trials).
      * Rc = average baseline power **within each session**
             — computed ONCE across ALL trials of ALL fingers in the
               session, giving a single stable reference per (channel,
               frequency). This is the crucial "within each session"
               wording from the paper.

    Pipeline executed here:
      1. Morlet TFR computed ONCE on every trial in the session
         (memory-cheaper than per-finger Morlet × 4).
      2. Rc = average of power over (all trials, baseline time).
      3. For each finger, Pc = average of power over (this finger's
         trials, task-window time).
      4. ERD = (Pc − Rc) / Rc, then averaged across the band's frequencies.

    Returns
    -------
    {finger_name: ndarray(n_chan,) or None}
    """
    freqs    = np.arange(fmin, fmax + 1, dtype=float)
    # Adaptive n_cycles: scale with frequency so low-freq wavelets (4–8 Hz)
    # use shorter kernels instead of the 1.75 s window that fixed n_cycles=7
    # produces at 4 Hz. Minimum of 3 keeps frequency resolution acceptable.
    n_cycles = np.maximum(freqs / 2, 3.0)

    all_data = epochs.get_data()                  # (n_total, n_chan, n_times)
    times    = epochs.times

    bl_mask   = (times >= BASELINE[0]) & (times <  BASELINE[1])
    task_mask = (times >= task_win[0]) & (times <= task_win[1])

    print(f"  Morlet TFR on {len(all_data)} epoch(s) "
          f"({len(freqs)} freqs, {fmin}-{fmax} Hz) ...", flush=True)
    power = tfr_array_morlet(
        all_data, sfreq=SFREQ_DS, freqs=freqs,
        n_cycles=n_cycles, output="power", verbose=False,
    )
    # power shape: (n_total_epochs, n_chan, n_freqs, n_times)

    # ── Session-level baseline reference Rc ─────────────────────────────────
    # Average over ALL trials (axis 0) AND baseline-time (axis -1).
    # This is the paper's "average baseline power within each session".
    Rc = power[:, :, :, bl_mask].mean(axis=(0, -1))    # (n_chan, n_freqs)

    result = {}
    for fname, feid in finger_pairs:
        mask     = epochs.events[:, 2] == feid
        n_trials = int(mask.sum())
        if n_trials == 0:
            print(f"    {fname:8s}: no epochs — skipping")
            result[fname] = None
            continue

        # Per-finger task-window power
        Pc = power[mask][:, :, :, task_mask].mean(axis=(0, -1))  # (n_chan, n_freqs)

        # ERD vs the session-level Rc, then average across the band freqs
        ERD        = (Pc - Rc) / (Rc + 1e-12)
        erd_per_ch = ERD.mean(axis=-1)                 # (n_chan,)

        # Use the mean of (Pc / Rc) instead of Pc.mean() / Rc.mean() to surface
        # the ratio that actually drives ERD, in a scale that's readable
        # (raw power values are ~1e-12 V² and round to 0.0000).
        ratio = (Pc / (Rc + 1e-30)).mean()
        print(f"    {fname:8s}: {n_trials:3d} trial(s)  "
              f"Pc̄={Pc.mean():.3e}  Rc̄={Rc.mean():.3e}  "
              f"Pc/Rc̄={ratio:+.3f}  "
              f"ERD=[{erd_per_ch.min():+.3f}, median={np.median(erd_per_ch):+.3f}, "
              f"{erd_per_ch.max():+.3f}]")

        # ── diagnostic: flag outlier channels (|ERD| > 1) ───────────────────
        extreme = np.where(np.abs(erd_per_ch) > 1.0)[0]
        if extreme.size > 0:
            worst_pos = np.argsort(erd_per_ch)[-3:][::-1]
            worst_neg = np.argsort(erd_per_ch)[:3]
            print(f"      [diag] {extreme.size} channel(s) with |ERD|>1")
            print(f"             top +ERS: " + ", ".join(
                f"ch{int(i)}={erd_per_ch[i]:+.2f}" for i in worst_pos))
            print(f"             top -ERD: " + ", ".join(
                f"ch{int(i)}={erd_per_ch[i]:+.2f}" for i in worst_neg))

        result[fname] = erd_per_ch

    return result


def plot_erd_comparison(erd_raw, erd_ica, band_label, finger_names, info,
                        title, save_path=None, show=True, vlim=None, cmap=None):
    """
    Two-row comparison figure: Before ICA (top) vs After ICA (bottom).
    erd_raw, erd_ica : {finger_name: (n_chan,) array or None}
    vlim : (vmin, vmax) colorbar range; defaults to (-0.5, 0.0) when None.
    cmap : matplotlib colormap name; defaults to "Blues_r" when None.
    """
    n_cols = len(finger_names)

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(2.5 * n_cols + 1.2, 5.6 + 0.6),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.88, right=0.88, hspace=0.1)

    vmin, vmax = vlim if vlim is not None else (-0.5, 0.0)
    cmap = cmap or "Blues_r"

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
                cmap=cmap,               # default Blues_r; RdBu_r for signed [-1,1]
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
        cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
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
    """CLI entry point: parses sys.argv to dispatch between online / offline
    / offline-group modes and drives `_run` (or a per-subject loop in GROUP
    mode) to produce the Before-vs-After-ICA ERD topomap figure(s)."""
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

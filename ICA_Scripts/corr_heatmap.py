"""
corr_heatmap.py

Computes ICA component correlations with task signals and saves a heatmap
(tasks x components, fixed scale 0-0.6) for one or more subjects.

Usage:
    Offline: python corr_heatmap.py <s1> [s2 ...] <task> OFF
    Online:  python corr_heatmap.py <s1> [s2 ...] <sess> <nclass> <task> <model>

Examples:
    python corr_heatmap.py 1 2 9 MI OFF
    python corr_heatmap.py 6 MI OFF
    python corr_heatmap.py 1 2 9 1 2 MI Orig
"""

import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.stats import zscore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mne

# Force the matplotlib browser backend (not mne-qt-browser) so the native
# source browser shares the same Tk event loop as the heatmap / properties
# windows — otherwise the two backends fight over the event loop.
try:
    mne.viz.set_browser_backend('matplotlib')
except Exception as _e:
    print(f"  [info] could not set browser backend: {_e}")

from Functions import (
    get_folder, get_offline_folder,
    build_raw_from_mat_files,
    build_task_vectors, compute_envelope,
    get_ica_fif_path, fit_or_load_ica,
    compute_exclusions,
)
from config import EOG_THRESHOLD, EMG_SLOPE_THRESH

SAVE_DIR = (
    r"C:\Users\aymen\Desktop\5 Years Of Engineering\DATASIM\DATASIM - Ouardani"
    r"\15 - STING (Début 30-03 et Fin 14-08)\Suivi du Stage"
    r"\Itération 4\Week 8\Correlations Maps"
)

_HIST_COLORS = ["#4CAF50", "#2196F3", "#FF9800", "#F44336",
                "#9C27B0", "#795548", "#E91E63", "#009688"]

# (label, l_freq, h_freq, filename_suffix, subfolder). l_freq=None -> no
# band-pass (broadband). Each band's PNG is saved under S<id>/<subfolder>/.
BANDS = [
    ("Broadband",         None, None, "_Broadband", "FullBand"),
    ("Alpha (8-13 Hz)",   8.0,  13.0, "_Alpha",     "Alpha"),
    ("Beta (13-30 Hz)",   13.0, 30.0, "_Beta",      "Beta"),
    ("Beta+2 (30-40 Hz)", 30.0, 40.0, "_BetaPlus2", "Beta+2"),
]


def _with_suffix(path, suffix):
    """Insert a suffix before the file extension."""
    base, ext = os.path.splitext(path)
    return f"{base}{suffix}{ext}"


def _compute_corr_matrix(sources_data, sfreq, tasks_z):
    """Absolute Pearson correlation between each task vector and each ICA
    source envelope. Returns (n_tasks, n_comp)."""
    n_comp = sources_data.shape[0]

    envelopes = np.array([compute_envelope(sources_data[c], sfreq) for c in range(n_comp)])

    # Z-score envelopes row-wise; leave zero rows where std == 0
    env_stds  = envelopes.std(axis=1)
    env_means = envelopes.mean(axis=1)
    envelopes_z = np.where(
        env_stds[:, np.newaxis] > 0,
        (envelopes - env_means[:, np.newaxis]) / env_stds[:, np.newaxis],
        0.0,
    )

    # Vectorized Pearson r: normalise rows to unit L2, then dot product
    env_norms  = np.linalg.norm(envelopes_z, axis=1)
    task_norms = np.linalg.norm(tasks_z,     axis=1)
    env_norms  = np.where(env_norms  > 0, env_norms,  1.0)
    task_norms = np.where(task_norms > 0, task_norms, 1.0)

    envelopes_z_unit = envelopes_z / env_norms[:, np.newaxis]
    tasks_z_unit     = tasks_z     / task_norms[:, np.newaxis]

    return np.abs(tasks_z_unit @ envelopes_z_unit.T)    # (n_tasks, n_comp)


def _open_component_properties(ica, raw, comp, band_label, band_range=None):
    """Open MNE's ica.plot_properties() window for one component — the same
    view that a right-click on a source opens in viz_sources.py.

    `raw` is the band-filtered recording for the heatmap that was clicked, so
    the source trace / PSD / epochs image reflect that band. `band_range`
    focuses the PSD x-axis on the band (None = broadband, full range)."""
    print(f"  [{band_label}] ICA{comp:03d}: opening MNE properties ...", flush=True)
    psd_args = {}
    if band_range is not None and band_range[0] is not None:
        lo, hi = band_range
        psd_args = {"fmin": max(0.0, lo - 2.0), "fmax": hi + 5.0}
    try:
        ica.plot_properties(raw, picks=[comp], show=True, psd_args=psd_args)
    except Exception as e:
        print(f"  [warn] plot_properties failed for ICA{comp:03d}: {e}")


def _open_source_window(ica, raw, comp, band_label):
    """Open MNE's native scrollable browser showing the one clicked ICA source
    trace, with the task periods overlaid as colored annotation spans.

    This is the same native viewer the other scripts use, so it has the usual
    MNE navigation: scroll through time, zoom the time axis (Home/End or the
    arrow keys), and rescale the amplitude (+/-). `raw` is the band-filtered
    recording so the source reflects the band of the clicked heatmap."""
    print(f"  [{band_label}] ICA{comp:03d}: opening source browser ...", flush=True)
    try:
        sources = ica.get_sources(raw)              # Raw of ICA source channels
        ch_name = sources.ch_names[comp]
        src_one = sources.copy().pick([ch_name])

        # Carry the task annotations over so they render as labeled, colored
        # spans (the "task lines"). get_sources usually copies them already,
        # but set them explicitly to be robust across MNE versions.
        if raw.annotations is not None and len(raw.annotations):
            try:
                src_one.set_annotations(raw.annotations)
            except Exception as e:
                print(f"  [info] could not attach task annotations: {e}")

        fig = src_one.plot(
            duration=20.0, start=0.0, scalings="auto",
            title=f"ICA{comp:03d} source — {band_label}  (task periods shaded)",
            block=False, show=True,
        )
        try:
            fig.canvas.manager.set_window_title(
                f"ICA{comp:03d} source — {band_label}"
            )
        except Exception:
            pass
    except Exception as e:
        print(f"  [warn] source window failed for ICA{comp:03d}: {e}")


def _ask_yes_no(prompt):
    """Prompt until the user answers y/yes or n/no. Returns True for yes.
    Treats EOF (no stdin) as 'no' so non-interactive runs never hang."""
    while True:
        try:
            ans = input(prompt).strip().lower()
        except EOFError:
            return False
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("  Please answer 'y' (yes) or 'n' (no).")


def _plot_corr_histogram(corr_matrix, tasks_list, subj_id, task, mode, band_label,
                         excluded=None):
    """Bar chart of the absolute task<->component correlation for a single band:
    each ICA source on the x-axis, its correlation value on the y-axis, one
    subplot per task.

    Excluded ICA sources (from the saved ICA) get a red x-axis label.

    Returned non-shown; the caller renders it non-blocking so it stays visible
    while moving on to the next band."""
    excluded = set(excluded or [])
    n_tasks, n_comp = corr_matrix.shape
    ncols   = min(n_tasks, 2)
    nrows   = int(np.ceil(n_tasks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 3.2 * nrows),
                             squeeze=False)
    try:
        fig.canvas.manager.set_window_title(f"Correlation histograms — {band_label}")
    except Exception:
        pass

    comps = np.arange(n_comp)
    for t_idx, task_name in enumerate(tasks_list):
        ax   = axes[t_idx // ncols][t_idx % ncols]
        vals = corr_matrix[t_idx]
        col  = _HIST_COLORS[t_idx % len(_HIST_COLORS)]
        ax.bar(comps, vals, color=col, alpha=0.85, edgecolor="white")
        ax.set_title(task_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("ICA source")
        ax.set_ylabel("Absolute correlation")
        ax.set_xticks(comps)
        tick_labels = ax.set_xticklabels([f"{c}" for c in comps],
                                         fontsize=6, rotation=90)
        for c, lbl in zip(comps, tick_labels):
            if c in excluded:
                lbl.set_color("red")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

    # Hide any unused subplot cells.
    for k in range(n_tasks, nrows * ncols):
        axes[k // ncols][k % ncols].set_axis_off()

    fig.suptitle(
        f"S{subj_id:02} | {task} | {mode} | {band_label}\n"
        f"Task <-> component correlation per ICA source",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def _plot_heatmap(corr_matrix, tasks_list, subj_id, task, mode, band_label,
                  save_path, ica=None, raw=None, interactive=False,
                  band_range=None, excluded=None):
    """Render one correlation heatmap. Saves the PNG if save_path is given.

    `excluded` is the set of ICA components flagged as artifacts by the
    automatic EOG/EMG detection; their x-axis labels are drawn in red.

    When interactive (or when no save_path), the figure is kept open and a
    left-click on any cell opens, for that ICA source (the heatmap column →
    component index), both the MNE component-properties window and a native
    scrollable source browser with the task periods shaded."""
    n_tasks, n_comp = corr_matrix.shape
    fig = plt.figure(figsize=(12, 6))
    ax  = fig.add_subplot(111)
    im  = ax.imshow(corr_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, label="Absolute Correlation")
    ax.set_yticks(range(n_tasks)); ax.set_yticklabels(tasks_list)
    ax.set_xticks(range(n_comp))
    _excluded = set(excluded or [])
    _xt_labels = ax.set_xticklabels([str(c) for c in range(n_comp)],
                                    fontsize=6, rotation=90)
    for _c, _lbl in zip(range(n_comp), _xt_labels):
        if _c in _excluded:
            _lbl.set_color("red")
    ax.set_xlabel("ICA Components")
    ax.set_ylabel("Tasks")
    ax.set_title(
        f"S{subj_id:02} | {task} | {mode} | {band_label}"
    )
    fig.tight_layout()

    keep_open = interactive or (save_path is None)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    if keep_open and ica is not None and raw is not None:
        def _on_click(event, _ax=ax, _ica=ica, _raw=raw, _n=n_comp,
                      _band=band_label, _range=band_range):
            if event.inaxes is not _ax or event.button != 1 or event.xdata is None:
                return
            comp = int(round(event.xdata))
            if 0 <= comp < _n:
                _open_component_properties(_ica, _raw, comp, _band, _range)
                _open_source_window(_ica, _raw, comp, _band)
        # Keep a reference so the callback isn't garbage-collected.
        fig._corr_click_cid = fig.canvas.mpl_connect("button_press_event", _on_click)
    elif not keep_open:
        plt.close(fig)

    return fig


def _load_offline_ica(subj_id, task):
    """Return the subject's OFFLINE ICA for `task`, reused across every online
    session of that task so the ICA is fitted only once.

    If the offline ICA is already cached it is just loaded. Otherwise it is
    fitted once from the offline recording (same preprocessing as the offline
    run: 60 Hz notch + 1 Hz highpass) and saved to its offline cache path."""
    ica_path = get_ica_fif_path(subj_id, task)        # offline filename form
    if os.path.exists(ica_path):
        print(f"  Reusing cached OFFLINE ICA for online session: {ica_path}")
        # Load directly (NOT via fit_or_load_ica, which wipes exclude) so the
        # exclusions validated/saved by clean_ICA.py are kept.
        ica = mne.preprocessing.read_ica(ica_path)
        if ica.exclude:
            print(f"  -> keeping {len(ica.exclude)} saved exclusion(s): "
                  f"{sorted(ica.exclude)}")
        return ica

    print("  Offline ICA not cached — fitting it once from the offline recording ...")
    off_folder = get_offline_folder(subj_id, task)
    mat_files  = sorted(glob.glob(os.path.join(off_folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(
            f"No offline .mat files in {off_folder} to fit the reusable ICA.")
    off_raw, _, _ = build_raw_from_mat_files(mat_files)
    off_raw.notch_filter(np.arange(60, 501, 60))
    off_filt = off_raw.copy().filter(1.0, None, verbose=False)
    return fit_or_load_ica(off_filt, ica_path)


def run_heatmap(subj_id, task, session=1, nclass=2,
                model_type="Orig", is_offline=False, save_path=None,
                interactive=False, reuse_offline_ica=True):
    """Load one recording, run/reuse its ICA, and compute+render the task<->
    component correlation heatmap for every frequency band in BANDS.

    In interactive mode (or when save_path is None) the figures stay open and
    left-clicking a heatmap cell opens that ICA source's properties window and
    a scrollable source browser; otherwise the PNGs are just saved to disk.
    """
    if is_offline:
        folder = get_offline_folder(subj_id, task)
    else:
        folder = get_folder(subj_id, task, session, nclass, model_type)

    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No files in {folder}")

    print(f"\nLoading data from: {folder}")
    raw, _, _ = build_raw_from_mat_files(mat_files)

    print("  Preprocessing: notch filter at 60 Hz harmonics")
    raw.notch_filter(np.arange(60, 501, 60))

    if not is_offline and reuse_offline_ica:
        # Reuse the subject's offline ICA for this online session (fit once,
        # apply to every session) instead of fitting a per-session ICA.
        ica = _load_offline_ica(subj_id, task)
    else:
        ica_cache_path = (get_ica_fif_path(subj_id, task) if is_offline
                          else get_ica_fif_path(subj_id, task, session, nclass, model_type))

        # Skip the 1 Hz highpass filter when the ICA cache already exists —
        # raw_filt is only needed for fitting, not for loading.
        raw_filt = (None if (ica_cache_path and os.path.exists(ica_cache_path))
                    else raw.copy().filter(1.0, None, verbose=False))
        ica = fit_or_load_ica(raw_filt, ica_cache_path)

    print("Computing task <-> ICA correlations ...")
    src_raw      = ica.get_sources(raw)              # Raw with the ICA source channels
    sfreq        = float(raw.info["sfreq"])
    # One binary/continuous vector per task, sampled at sfreq, marking when
    # that task was active during the recording.
    task_vectors = build_task_vectors(raw)

    if not task_vectors:
        print("\n[ERROR] No task annotations found. Cannot compute correlations.")
        return

    tasks_list = list(task_vectors.keys())
    n_tasks    = len(tasks_list)
    n_samples  = src_raw.n_times

    # Z-score task vectors once; leave zeros where std == 0
    tasks_z = np.zeros((n_tasks, n_samples))
    for t_idx, vec in enumerate(task_vectors.values()):
        if np.std(vec) > 0:
            tasks_z[t_idx] = zscore(vec)

    mode = "Offline" if is_offline else f"Sess{session:02} | {nclass}-class | {model_type}"

    # Auto-detect EOG/EMG artifact components (same thresholds/logic as
    # clean_ICA.py) so their source index is drawn in red on every heatmap and
    # bar chart. Failure here (e.g. no EOG channels) just leaves nothing red.
    artifact_comps = set()
    try:
        raw_det = raw.copy().filter(1.0, None, verbose=False)
        _, eog_idx, emg_idx = compute_exclusions(
            ica, raw_det, muscle_thresh=EMG_SLOPE_THRESH,
            eog_thresh=EOG_THRESHOLD, ch_names=raw_det.info["ch_names"],
        )
        artifact_comps = set(eog_idx) | set(emg_idx)
        print(f"  Auto artifact detection: EOG={sorted(eog_idx)}  "
              f"EMG={sorted(emg_idx)}  -> red labels: {sorted(artifact_comps)}")
    except Exception as e:
        print(f"  [warn] EOG/EMG detection failed (no red labels): {e}")

    # One heatmap per frequency band: broadband, alpha (8-13), beta (13-30),
    # beta+2 (30-40). We build ALL heatmaps first (and render them now in
    # interactive mode) so every correlation map is on screen before we start
    # asking, per band, whether to also pop a histogram window.
    prompt_hist = interactive or (save_path is None)
    figs = []
    band_results = []          # (band_label, corr_matrix) for the histogram pass
    for band_label, l_freq, h_freq, fname_suffix, subfolder in BANDS:
        if l_freq is None:
            band_sources = src_raw.get_data()
            raw_band     = raw                 # broadband: full recording
        else:
            print(f"  Band-pass filtering sources: {band_label}")
            band_sources = src_raw.copy().filter(l_freq, h_freq, verbose=False).get_data()
            # Band-filtered recording so clicking a cell shows the band-filtered
            # source in plot_properties (trace / PSD / epochs image).
            raw_band     = raw.copy().filter(l_freq, h_freq, verbose=False)

        print(f"  [{band_label}] computing envelopes for {band_sources.shape[0]} components ...",
              flush=True)
        corr_matrix = _compute_corr_matrix(band_sources, sfreq, tasks_z)

        print(f"\n=== {band_label} ===")
        for t_idx, task_name in enumerate(tasks_list):
            print(f"Task: {task_name}  ->", np.round(corr_matrix[t_idx], 3))

        # Save under  SAVE_DIR/S<id>/<subfolder>/<basename>_<suffix>.png
        if save_path:
            band_dir  = os.path.join(SAVE_DIR, f"S{subj_id:02d}", subfolder)
            band_save = os.path.join(band_dir,
                                     os.path.basename(_with_suffix(save_path, fname_suffix)))
        else:
            band_save = None
        figs.append(_plot_heatmap(corr_matrix, tasks_list, subj_id, task,
                                  mode, band_label, band_save,
                                  ica=ica, raw=raw_band, interactive=interactive,
                                  band_range=(l_freq, h_freq),
                                  excluded=artifact_comps))
        band_results.append((band_label, corr_matrix))

    # Make sure every correlation heatmap is painted on screen before asking.
    if prompt_hist:
        plt.show(block=False)
        plt.pause(0.1)

        # Now ask, per band, whether to pop a histogram window. Answering 'y'
        # shows it non-blocking and immediately moves on to the next band —
        # earlier histogram windows stay open meanwhile.
        for band_label, corr_matrix in band_results:
            if _ask_yes_no(
                    f"  Show correlation histograms for {band_label}? (y/n): "):
                hist_fig = _plot_corr_histogram(corr_matrix, tasks_list, subj_id,
                                                task, mode, band_label,
                                                excluded=artifact_comps)
                figs.append(hist_fig)
                hist_fig.show()       # non-blocking: render now, keep iterating
                plt.pause(0.1)        # let the Tk event loop actually paint it

        print("\n  Left-click a heatmap cell to open that ICA source's properties "
              "and a scrollable source browser (task periods shaded).")
        plt.show(block=True)


def main():
    """CLI entry point: dispatch to offline or online argument parsing based on
    a trailing "OFF" token, then run the heatmap for every requested subject."""
    offline = len(sys.argv) > 1 and sys.argv[-1].upper() == "OFF"

    if offline:
        sys.argv.pop()
        # Remaining argv: <s1> [s2 ...] <task>
        if len(sys.argv) < 3:
            print("Usage: python corr_heatmap.py <s1> [s2 ...] <task> OFF")
            sys.exit(1)
        task     = sys.argv[-1]
        subjects = [int(a) for a in sys.argv[1:-1]]
        # A single subject opens interactive (clickable) windows; multiple
        # subjects run as a non-blocking batch that just saves the PNGs.
        interactive = len(subjects) == 1
        for subj_id in subjects:
            save_path = os.path.join(
                SAVE_DIR,
                f"Correlation_Heatmap_S{subj_id:02d}_{task}_Offline.png",
            )
            try:
                run_heatmap(subj_id, task, is_offline=True, save_path=save_path,
                            interactive=interactive)
            except Exception as e:
                print(f"  [ERROR] S{subj_id:02d}: {e}")

    else:
        # Remaining argv: <s1> [s2 ...] <sess> <nclass> <task> <model>
        if len(sys.argv) < 6:
            print("Usage: python corr_heatmap.py <s1> [s2 ...] <sess> <nclass> <task> <model>")
            sys.exit(1)
        model    = sys.argv[-1]
        task     = sys.argv[-2]
        nclass   = int(sys.argv[-3])
        session  = int(sys.argv[-4])
        subjects = [int(a) for a in sys.argv[1:-4]]
        # A single subject opens interactive (clickable) windows; multiple
        # subjects run as a non-blocking batch that just saves the PNGs.
        interactive = len(subjects) == 1
        for subj_id in subjects:
            save_path = os.path.join(
                SAVE_DIR,
                f"Correlation_Heatmap_S{subj_id:02d}_{task}_Sess{session:02d}_{nclass}class_{model}.png",
            )
            try:
                run_heatmap(subj_id, task, session=session, nclass=nclass,
                            model_type=model, is_offline=False, save_path=save_path,
                            interactive=interactive)
            except Exception as e:
                print(f"  [ERROR] S{subj_id:02d}: {e}")


if __name__ == "__main__":
    main()

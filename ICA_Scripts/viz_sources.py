"""
viz_sources.py

Plot ICA sources (ica.plot_sources) from a cached ICA .fif file.
Double-click on any source trace to open its properties (spectrum, ERD, topomap).
Requires the ICA to have been fitted first (viz_inspector.py or clean_ICA.py).

Usage:
    Online:  python viz_sources.py <subj> <sess> <nclass> <task> <model>
    Offline: python viz_sources.py <subj> <task> OFF
"""

import sys
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mne
import matplotlib.pyplot as plt
from Functions import (
    build_raw_from_mat_files,
    get_folder, get_offline_folder,
    get_standard_args, get_offline_args,
    get_ica_fif_path,
)


def _load_raw(folder):
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {folder}")
    raw, _, _ = build_raw_from_mat_files(mat_files)
    return raw.filter(1.0, None, verbose=False)


def _connect_dblclick(fig, ica, raw_filt):
    """Double-click on a source trace → opens plot_properties for that component."""
    n_comp = ica.n_components_

    def on_click(event):
        if not event.dblclick or event.button != 1 or event.inaxes is None:
            return
        ax = event.inaxes
        ytick_pos    = ax.get_yticks()
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        if event.ydata is None or len(ytick_pos) == 0:
            return
        closest = int(np.argmin(np.abs(np.array(ytick_pos, dtype=float) - event.ydata)))
        label = ytick_labels[closest] if closest < len(ytick_labels) else ""
        m = re.search(r'\d+', label)
        if m:
            comp_idx = int(m.group())
            if 0 <= comp_idx < n_comp:
                print(f"  Opening properties for ICA{comp_idx:03d} ...")
                ica.plot_properties(raw_filt, picks=[comp_idx])

    fig.canvas.mpl_connect('button_press_event', on_click)


def sources_online(subj, sess, ncl, task, model):
    print("=" * 60)
    print(f"  ICA Sources: S{subj:02} Sess{sess:02} | {task} {ncl}-class | {model}")
    print("=" * 60)
    raw_filt = _load_raw(get_folder(subj, task, sess, ncl, model))
    ica_path = get_ica_fif_path(subj, task, sess, ncl, model)
    if not os.path.exists(ica_path):
        raise FileNotFoundError(f"No cached ICA at {ica_path}\nRun viz_inspector.py first.")
    ica = mne.preprocessing.read_ica(ica_path)
    print(f"  Loaded {ica.n_components_} components from {ica_path}")
    print("  Double-click on a source trace to open its properties.")
    fig = ica.plot_sources(raw_filt, show=True, block=False)
    _connect_dblclick(fig, ica, raw_filt)
    plt.show(block=True)


def sources_offline(subj, task):
    print("=" * 60)
    print(f"  ICA Sources: S{subj:02} | {task} (Offline)")
    print("=" * 60)
    raw_filt = _load_raw(get_offline_folder(subj, task))
    raw_filt.notch_filter(np.arange(60, 501, 60))
    ica_path = get_ica_fif_path(subj, task)
    if not os.path.exists(ica_path):
        raise FileNotFoundError(f"No cached ICA at {ica_path}\nRun viz_inspector.py first.")
    ica = mne.preprocessing.read_ica(ica_path)
    print(f"  Loaded {ica.n_components_} components from {ica_path}")
    print("  Double-click on a source trace to open its properties.")
    fig = ica.plot_sources(raw_filt, show=True, block=False)
    _connect_dblclick(fig, ica, raw_filt)
    plt.show(block=True)


def main():
    try:
        if len(sys.argv) > 1 and sys.argv[-1].upper() == "OFF":
            sys.argv.remove(sys.argv[-1])
            subj, task = get_offline_args(description="ICA Sources - Offline")
            sources_offline(subj, task)
        else:
            subj, sess, ncl, task, model = get_standard_args(description="ICA Sources - Online")
            sources_online(subj, sess, ncl, task, model)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
viz_band_source.py

Interactive MNE viewer for ONE ICA source band-filtered to a chosen frequency
band, with the task periods overlaid as coloured annotation spans. Uses the
subject's OFFLINE ICA (same component index as the energy/correlation analyses).

Scroll/zoom with the usual MNE browser controls (←/→ scroll, Home/End, +/-
rescale, etc.).

Usage:
    python viz_band_source.py <subj> <sess> <nclass> <task> <model> <src> [lo] [hi]

Defaults: band = delta (1-4 Hz).

Examples:
    python viz_band_source.py 4 5 3 MI Orig 80          # S04 src80, delta 1-4 Hz
    python viz_band_source.py 4 5 3 MI Orig 80 8 13      # same source, alpha band
"""
import os
import sys
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mne

# Importing corr_heatmap sets matplotlib's TkAgg backend and the MNE matplotlib
# browser backend (so the interactive viewer shares one event loop) and gives
# us the offline-ICA loader reused across the analyses.
import corr_heatmap as ch
from Functions import build_raw_from_mat_files, get_folder


def main():
    """CLI entry point: load the recording, band-filter the selected ICA
    source, overlay task annotations, and open the interactive MNE viewer."""
    if len(sys.argv) < 7:
        print("Usage: python viz_band_source.py <subj> <sess> <nclass> <task> "
              "<model> <src> [lo] [hi]   (default band: delta 1-4 Hz)")
        sys.exit(1)

    subj  = int(sys.argv[1])
    sess  = int(sys.argv[2])
    nclass = int(sys.argv[3])
    task  = sys.argv[4]
    model = sys.argv[5]
    src   = int(sys.argv[6])
    lo    = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
    hi    = float(sys.argv[8]) if len(sys.argv) > 8 else 4.0

    folder = get_folder(subj, task, sess, nclass, model)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in:\n  {folder}")

    print(f"Loading {len(mat_files)} file(s) from:\n  {folder}")
    raw, _, _ = build_raw_from_mat_files(mat_files)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)

    ica = ch._load_offline_ica(subj, task)
    sources = ica.get_sources(raw)      # unmix raw into one channel per ICA component
    ch_name = sources.ch_names[src]
    one = sources.copy().pick([ch_name])   # isolate the single source of interest

    print(f"Filtering ICA{src:03d} to {lo:g}-{hi:g} Hz ...")
    one.filter(lo, hi, verbose=False)

    # Carry the task annotations so they render as coloured, labelled spans.
    if raw.annotations is not None and len(raw.annotations):
        one.set_annotations(raw.annotations)

    band = ("delta" if (lo, hi) == (1.0, 4.0) else f"{lo:g}-{hi:g} Hz")
    title = (f"S{subj:02d} | {task} | Sess{sess:02d} {nclass}class {model} | "
             f"ICA{src:03d} | {band} ({lo:g}-{hi:g} Hz) | task periods shaded")
    print("Opening interactive MNE viewer "
          "(scroll: ←/→, zoom time: Home/End, rescale: +/-).")
    one.plot(duration=20.0, start=0.0, scalings="auto", block=True,
             title=title, show=True)


if __name__ == "__main__":
    main()

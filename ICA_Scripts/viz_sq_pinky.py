"""
viz_sq_pinky.py

Interactive MNE viewer showing, for ONE ICA source, both signals SUPERPOSED
on the same axes (MNE butterfly mode):
  - the source SQUARED  (x^2, instantaneous power) -- what gets summed
    per trial to form the energy,
  - the task one-vs-rest signal (scaled, 1 during trials of <task>, else 0) --
    exactly the vector the energy<->task correlation is computed against,
and EVERY task trial drawn as a shaded span, one fixed colour per class
(Thumb=yellow, Index=blue, Pinky=green; see TASK_COLORS).

Press 'b' in the viewer to toggle butterfly (superposed) vs separate-row layout.

Uses the subject's OFFLINE ICA (same component index as the analyses).

Usage:
    python viz_sq_pinky.py <subj> <sess> <nclass> <task> <model> <src> [finger]
    default: 9 1 3 MI Orig 15 Pinky
"""
import os
import sys
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mne

# corr_heatmap sets matplotlib TkAgg + the MNE matplotlib browser backend, and
# provides the offline-ICA loader reused by all the analyses.
import corr_heatmap as ch
import fill_energy_table as fet
from Functions import build_raw_from_mat_files, get_folder

# One fixed colour per task class so the shading is consistent across sessions.
# Online paradigms only ever use Thumb / Index / Pinky (Middle is offline-only).
TASK_COLORS = {
    "Thumb":  "#FFD500",   # yellow
    "Index":  "#1F77B4",   # blue
    "Pinky":  "#2CA02C",   # green
    "Middle": "#9467BD",   # purple (fallback; not used in the online paradigms)
}


def main():
    """Parse CLI args, build the squared-source / task-membership overlay, and open the MNE viewer."""
    subj   = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    sess   = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    nclass = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    task   = sys.argv[4] if len(sys.argv) > 4 else "MI"
    model  = sys.argv[5] if len(sys.argv) > 5 else "Orig"
    src    = int(sys.argv[6]) if len(sys.argv) > 6 else 15
    finger = sys.argv[7] if len(sys.argv) > 7 else "Pinky"

    folder = get_folder(subj, task, sess, nclass, model)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in:\n  {folder}")

    print(f"Loading {len(mat_files)} file(s) from:\n  {folder}")
    raw, _, _ = build_raw_from_mat_files(mat_files)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)
    sfreq = float(raw.info["sfreq"])

    ica = ch._load_offline_ica(subj, task)
    sources = ica.get_sources(raw)
    sig = sources.get_data(picks=src)[0]
    sq = sig ** 2                                  # instantaneous power

    # Per-trial windows (exactly those used for the energy). Shade EVERY task
    # trial (Thumb / Index / Pinky / ... -- anything but TrialEnd, which is not
    # returned here) with one colour per class; also build the one-vs-rest step
    # signal for the chosen `finger` (the vector the energy<->task corr uses).
    trials = fet._chronological_trials(raw)
    member = np.zeros(raw.n_times, dtype=float)
    onsets, durs, descs = [], [], []
    n_trials_f = 0
    for cls, s, e in trials:
        onsets.append(s / sfreq)
        durs.append((e - s) / sfreq)
        descs.append(cls)
        if cls == finger:
            member[s:e] = 1.0
            n_trials_f += 1
    classes_present = sorted(set(descs), key=lambda c: list(TASK_COLORS).index(c)
                             if c in TASK_COLORS else 99)
    legend = ", ".join(f"{c}={TASK_COLORS.get(c, '#7F7F7F')}" for c in classes_present)
    print(f"ICA{src:03d}: {len(trials)} trials over classes {classes_present}; "
          f"{n_trials_f} are {finger}.  Shading colours: {legend}")

    # Both traces share ONE baseline at data value 0 (MNE butterfly + remove_dc=False
    # below); the axis is cropped after plotting so that baseline sits at the BOTTOM
    # (see the set_ylim call). 'hi' is a robust power level (95th pct) used as the
    # display scale; the rare extreme spikes clip on purpose -- otherwise they squash
    # everything and the task box becomes invisible.
    hi = float(np.percentile(sq, 95))
    if hi <= 0:
        hi = float(np.max(sq)) or 1.0
    sq_disp = sq                                   # power: floor at 0 (its lowest value)
    member_disp = member * (1.6 * hi)              # task box steps from 0 up to ~80% height
    # power floor (0) and task "off" level (0) coincide: both sit on the y=0 baseline.

    info = mne.create_info([f"ICA{src:03d}_squared", f"{finger}_task"],
                           sfreq, ["misc", "misc"])
    raw2 = mne.io.RawArray(np.vstack([sq_disp, member_disp]), info, verbose=False)
    raw2.set_annotations(mne.Annotations(onsets, durs, descs))

    title = (f"S{subj:02d} | {task} | Sess{sess:02d} {nclass}class {model} | "
             f"ICA{src:03d} squared  +  {finger} task signal SUPERPOSED "
             f"(trials shaded by class: " +
             ", ".join(classes_present) + ")")
    print("Opening interactive MNE viewer  (scroll: ←/→ , rescale: +/- , "
          "zoom time: Home/End , toggle superposed: b).")
    # butterfly=True   -> both 'misc' channels on one shared axis (superposed).
    # remove_dc=False  -> keep the common zero baseline (no per-channel mean
    #                     removal), so the squared signal's floor stays at value 0.
    # scalings=dict(misc=hi) -> fixed robust scale (not driven by the spikes).
    fig = raw2.plot(duration=30.0, start=0.0, scalings=dict(misc=hi), block=False,
                    butterfly=True, remove_dc=False, title=title, show=False)
    # By default MNE centres the data==0 ("misc") baseline in the middle of the axis,
    # leaving the whole lower half empty (the squared signal is >= 0). Crop the axis
    # to its upper half so the y=0 / misc baseline sits at the BOTTOM edge and the
    # signal rises upward from it. MNE's trace coords put the baseline at y=0 and the
    # top of the band at y=-0.5. (Horizontal scroll, Home/End and +/- all keep this;
    # only toggling butterfly with 'b' would reset it.)
    fig.mne.ax_main.set_ylim(0.0, -0.5)            # 0 = misc baseline -> bottom edge
    # Force one fixed colour per task class for the shaded spans (MNE otherwise
    # auto-assigns colours from a cycle). _setup_annotation_colors() keeps any key
    # already present in this dict, so pre-seeding it pins our choices.
    fig.mne.annotation_segment_colors = {
        d: TASK_COLORS.get(d, "#7F7F7F") for d in set(descs)
    }
    fig._redraw(annotations=True)                  # redraw spans with our colours
    import matplotlib.pyplot as plt
    plt.show(block=True)


if __name__ == "__main__":
    main()

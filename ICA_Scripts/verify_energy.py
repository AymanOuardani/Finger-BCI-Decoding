"""
verify_energy.py

Audit of the trial-window extraction and of the energy computation, on real
data. Every energy figure in the analyses rests on two assumptions — that a
trial window is exactly what the .mat markers say, and that the ICA source is
time-aligned with the raw used to cut those windows. This script checks both
rather than trusting them.

Six checks:
  1. the raw Target/TrialEnd markers of the first run, straight from the .mat
     (MATLAB 1-based samples)
  2. the annotations rebuilt by the pipeline
  3. the trials returned by _chronological_trials, their count and durations
  4. run-1 alignment: pipeline window starts == .mat Target samples, 0-based
  5. a trial's end == the next TrialEnd marker
  6. energy = sum of squared samples over [start, end), recomputed with an
     explicit loop to rule out a vectorisation bug

Expected trial counts (see CLAUDE.md): 8 runs per recording, 80 trials per
finger — 160 trials in 2-class, 240 in 3-class.

Usage:
    python verify_energy.py [subj] [sess] [nclass] [model] [src] [task]

    python verify_energy.py                    # S04 Sess05 3class Orig src80
    python verify_energy.py 9 1 3 Orig 15
"""

import os
import sys
import glob
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import scipy.io

import corr_heatmap as ch
import fill_energy_table as fet
from Functions import build_raw_from_mat_files, get_folder
from config import TASK_LABELS, FINGER_LABEL


def main():
    """CLI entry point: run the six alignment/energy sanity checks described
    in the module docstring for one recording and print the results."""
    subj = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    sess = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    nclass = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    model = sys.argv[4] if len(sys.argv) > 4 else "Orig"
    src = int(sys.argv[5]) if len(sys.argv) > 5 else 80
    task = sys.argv[6].upper() if len(sys.argv) > 6 else "MI"

    folder = get_folder(subj, task, sess, nclass, model)
    mats = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mats:
        print(f"no .mat found in {folder}")
        sys.exit(1)
    print(f"=== S{subj:02d} Sess{sess:02d} {nclass}class {model} | "
          f"{len(mats)} runs ===\n")

    # ── 1. Raw markers in the FIRST .mat (ground truth) ──────────────────────
    mat = scipy.io.loadmat(mats[0])
    eeg, event = mat["eeg"], mat["event"]
    # scipy.io.loadmat wraps MATLAB structs in nested 1x1 object arrays, hence
    # the repeated [0][0] indexing to reach the scalar/array field values.
    fs_mat = int(eeg["fsample"][0][0][0][0])
    n_samples_run1 = eeg["data"][0][0].shape[1]
    print(f"[1] run1 .mat: fsample={fs_mat} Hz, nSamples={n_samples_run1}")
    print("    first 8 events (MATLAB, sample is 1-based):")

    mat_targets = []
    for i in range(event.shape[1]):
        evt = event[0, i]
        etype = str(evt["type"][0]).strip()
        sample = int(evt["sample"][0][0])                  # 1-based
        # MATLAB event timestamps are 1-based sample indices; subtracting 1
        # below converts to the 0-based indexing used by the Python pipeline.
        if etype == "Target":
            value = int(evt["value"][0][0])
            mat_targets.append((FINGER_LABEL.get(value, f"?{value}"), sample - 1))
            if len(mat_targets) <= 8:
                print(f"      Target  value={value} "
                      f"({FINGER_LABEL.get(value, '?'):6s}) sample={sample} "
                      f"-> 0-based {sample - 1}  t={(sample - 1) / fs_mat:.4f}s")

    # ── 2. Pipeline raw + annotations ────────────────────────────────────────
    raw, _, _ = build_raw_from_mat_files(mats)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)
    sfreq = float(raw.info["sfreq"])
    descriptions = np.asarray(raw.annotations.description)
    print(f"\n[2] pipeline raw: sfreq={sfreq} Hz, n_times={raw.n_times}")
    print("    annotations: " +
          ", ".join(f"{t}={int((descriptions == t).sum())}" for t in TASK_LABELS
                    if (descriptions == t).sum()) +
          f", TrialEnd={int((descriptions == 'TrialEnd').sum())}")

    # ── 3. Trials from the pipeline ──────────────────────────────────────────
    trials = fet._chronological_trials(raw)
    classes = [c for c, _s, _e in trials]
    expected = 80 * nclass
    print(f"\n[3] _chronological_trials -> n_trials={len(trials)}  "
          f"per class={dict(Counter(classes))}")
    print(f"    expected ({nclass}-class, 8 runs): {expected} total, 80/task  "
          f"-> {'OK' if len(trials) == expected else 'MISMATCH'}")
    durations = np.array([e - s for _c, s, e in trials])
    print(f"    window length (samples): min={durations.min()} "
          f"max={durations.max()} -> {durations.min() / sfreq:.4f}"
          f"-{durations.max() / sfreq:.4f}s")
    print("    first 6 windows:")
    for cls, start, end in trials[:6]:
        print(f"      {cls:6s} start={start:7d} end={end:7d} dur={end - start} "
              f"({(end - start) / sfreq:.4f}s)  onset={start / sfreq:.4f}s")

    # ── 4. run1 pipeline starts == .mat Target samples (0-based) ─────────────
    pipeline_run1 = [(c, s) for c, s, _e in trials if s < n_samples_run1]
    print(f"\n[4] run1 alignment check ({len(pipeline_run1)} trials in run1):")
    print("      .mat Target (label, 0-based sample) | pipeline (label, start)")
    aligned = True
    for (mat_cls, mat_s), (pipe_cls, pipe_s) in zip(
            mat_targets[:len(pipeline_run1)], pipeline_run1):
        same = (mat_cls == pipe_cls) and (mat_s == pipe_s)
        aligned = aligned and same
        print(f"      {mat_cls:6s} {mat_s:7d}              | "
              f"{pipe_cls:6s} {pipe_s:7d}   {'OK' if same else '  <-- MISMATCH'}")
    print(f"    => run1 windows match raw markers exactly: {aligned}")

    # ── 5. Window end == next TrialEnd marker ────────────────────────────────
    trial_ends = np.sort(np.asarray(raw.annotations.onset)[descriptions == "TrialEnd"])
    _c0, s0, e0 = trials[0]
    next_end = trial_ends[trial_ends > s0 / sfreq][0]
    print(f"\n[5] trial0 end check: end_sample={e0}  next TrialEnd onset="
          f"{next_end:.4f}s -> sample {int(round(next_end * sfreq))}  "
          f"(match: {e0 == int(round(next_end * sfreq))})")

    # ── 6. Energy reproduction for the chosen source ─────────────────────────
    ica = ch._load_offline_ica(subj, task)
    sources = ica.get_sources(raw)
    print(f"\n[6] ICA sources: n_times={sources.n_times} "
          f"(aligned with raw: {sources.n_times == raw.n_times})")
    signal = sources.get_data(picks=src)[0]
    print(f"    energy = sum x^2 of source {src} over [start, end):")
    for cls, start, end in trials[:3]:
        vectorised = float(np.sum(signal[start:end] ** 2))
        explicit = float(sum(float(v) * float(v) for v in signal[start:end]))
        print(f"      {cls:6s} [{start}:{end}]  sum x^2={vectorised:.6e}  "
              f"(explicit-loop={explicit:.6e}, "
              f"diff={abs(vectorised - explicit):.2e})")

    print("\n=== verification done ===")


if __name__ == "__main__":
    main()

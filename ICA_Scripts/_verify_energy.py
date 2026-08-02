"""_verify_energy.py — TEMP. Audit of trial-window extraction and energy.

Proves, on real data (S04 Sess05 3class Orig, src80), that:
  1. trial windows = exactly the .mat Target/TrialEnd markers (1-based -> 0-based),
  2. trial count matches the protocol (240 = 80/task in 3-class),
  3. energy = sum of squared samples of the ICA source over [start, end),
  4. the ICA source is time-aligned with the raw used for the windows.
"""
import os, sys, glob
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import scipy.io
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import corr_heatmap as ch
import fill_energy_table as fet
from Functions import build_raw_from_mat_files, get_folder
from config import TASK_LABELS, FINGER_LABEL

SUBJ, SESS, NCL, MODEL, SRC, TASK = 4, 5, 3, "Orig", 80, "MI"


def main():
    folder = get_folder(SUBJ, TASK, SESS, NCL, MODEL)
    mats = sorted(glob.glob(os.path.join(folder, "*.mat")))
    print(f"=== S{SUBJ:02d} Sess{SESS:02d} {NCL}class {MODEL} | {len(mats)} runs ===\n")

    # ── 1. Raw markers in the FIRST .mat (ground truth) ───────────────────
    mat = scipy.io.loadmat(mats[0])
    eeg, event = mat["eeg"], mat["event"]
    fs_mat = int(eeg["fsample"][0][0][0][0])
    nsamp_run1 = eeg["data"][0][0].shape[1]
    print(f"[1] run1 .mat: fsample={fs_mat} Hz, nSamples={nsamp_run1}")
    print("    first 8 events (MATLAB, sample is 1-based):")
    mt_targets = []
    trialends_run1 = []
    for i in range(event.shape[1]):
        evt = event[0, i]
        etype = str(evt["type"][0]).strip()
        samp = int(evt["sample"][0][0])             # 1-based
        if etype == "Target":
            val = int(evt["value"][0][0])
            mt_targets.append((FINGER_LABEL.get(val, f"?{val}"), samp - 1))  # 0-based
            if len(mt_targets) <= 8:
                print(f"      Target  value={val} ({FINGER_LABEL.get(val,'?'):6s}) "
                      f"sample={samp} -> 0-based {samp-1}  t={(samp-1)/fs_mat:.4f}s")
        elif etype == "TrialEnd":
            trialends_run1.append(samp - 1)

    # ── 2. Pipeline raw + annotations ─────────────────────────────────────
    raw, _, _ = build_raw_from_mat_files(mats)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)
    sfreq = float(raw.info["sfreq"])
    descs = np.asarray(raw.annotations.description)
    print(f"\n[2] pipeline raw: sfreq={sfreq} Hz, n_times={raw.n_times}")
    print(f"    annotations: " +
          ", ".join(f"{T}={int((descs==T).sum())}" for T in TASK_LABELS
                    if (descs == T).sum()) +
          f", TrialEnd={int((descs=='TrialEnd').sum())}")

    # ── 3. Trials from the pipeline ───────────────────────────────────────
    trials = fet._chronological_trials(raw)
    classes = [c for c, _s, _e in trials]
    print(f"\n[3] _chronological_trials -> n_trials={len(trials)}  "
          f"per class={dict(Counter(classes))}")
    print(f"    expected (3-class, 8 runs): 240 total, 80/task")
    durs = np.array([e - s for _c, s, e in trials])
    print(f"    window length (samples): min={durs.min()} max={durs.max()} "
          f"-> {durs.min()/sfreq:.4f}-{durs.max()/sfreq:.4f}s")
    print("    first 6 windows:")
    for c, s, e in trials[:6]:
        print(f"      {c:6s} start={s:7d} end={e:7d} dur={e-s} "
              f"({(e-s)/sfreq:.4f}s)  onset={s/sfreq:.4f}s")

    # ── 4. Cross-check: run1 pipeline starts == .mat Target samples (0-based)
    pipe_run1 = [(c, s) for c, s, _e in trials if s < nsamp_run1]
    print(f"\n[4] run1 alignment check ({len(pipe_run1)} trials in run1):")
    print("      .mat Target (label, 0-based sample) | pipeline (label, start)")
    ok = True
    for (mc, ms), (pc, ps) in zip(mt_targets[:len(pipe_run1)], pipe_run1):
        same = (mc == pc) and (ms == ps)
        ok = ok and same
        flag = "OK" if same else "  <-- MISMATCH"
        print(f"      {mc:6s} {ms:7d}              | {pc:6s} {ps:7d}   {flag}")
    print(f"    => run1 windows match raw markers exactly: {ok}")

    # ── 5. Window end == next TrialEnd marker ─────────────────────────────
    te = np.sort(np.asarray(raw.annotations.onset)[descs == "TrialEnd"])
    c0, s0, e0 = trials[0]
    next_te = te[te > s0 / sfreq][0]
    print(f"\n[5] trial0 end check: end_sample={e0}  next TrialEnd onset="
          f"{next_te:.4f}s -> sample {int(round(next_te*sfreq))}  "
          f"(match: {e0 == int(round(next_te*sfreq))})")

    # ── 6. Energy reproduction for src80 ──────────────────────────────────
    ica = ch._load_offline_ica(SUBJ, TASK)
    src_raw = ica.get_sources(raw)
    aligned = (src_raw.n_times == raw.n_times)
    sig = src_raw.get_data(picks=SRC)[0]
    print(f"\n[6] ICA sources: n_times={src_raw.n_times} (aligned with raw: {aligned})")
    print(f"    energy = sum x^2 of source {SRC} over [start, end):")
    for c, s, e in trials[:3]:
        E = float(np.sum(sig[s:e] ** 2))
        # independent recomputation, explicit loop, to rule out a vectorization bug
        E_loop = float(sum(float(v) * float(v) for v in sig[s:e]))
        print(f"      {c:6s} [{s}:{e}]  sum x^2={E:.6e}  (explicit-loop={E_loop:.6e}, "
              f"diff={abs(E-E_loop):.2e})")

    print("\n=== verification done ===")


if __name__ == "__main__":
    main()

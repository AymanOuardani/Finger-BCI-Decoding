"""
viz_dual_sources.py

Interactive viewer for two ICA sources from different subjects / sessions.
Each row shows the source signal with coloured trial spans; the per-trial
full-band energy (Σx²) is printed next to the task label at the top of each span.

Controls
--------
  ← / →      scroll left / right  (SCROLL_STEP seconds)
  ↑ / ↓      zoom in / out
  scroll      zoom in / out
  Home / End  jump to start / end

Hard-coded targets (edit SOURCES below):
    ICA015 (comp=15, 0-indexed) — S09, Sess01, 3-class, MI, Orig
    ICA080 (comp=80, 0-indexed) — S04, Sess05, 3-class, MI, Orig

Usage
-----
    python viz_dual_sources.py
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne
mne.set_log_level("WARNING")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Functions import get_folder, get_ica_fif_path, build_raw_from_mat_files

# ── Hard-coded targets ─────────────────────────────────────────────────────────
# comp is 0-indexed, exactly like MNE's ICA viewer (ICA000 = comp=0, ICA015 = comp=15).
# This matches the ICA_Source column in the ODS (also 0-indexed).
SOURCES = [
    dict(
        subj=9,  sess=1, ncl=3, task="MI", model="Orig",
        comp=15,
        label="S09  |  Sess01  |  3-class  |  MI  |  Orig  —  ICA015  (comp=15)",
    ),
    dict(
        subj=4,  sess=5, ncl=3, task="MI", model="Orig",
        comp=80,
        label="S04  |  Sess05  |  3-class  |  MI  |  Orig  —  ICA080  (comp=80)",
    ),
]

TASK_COLORS = {
    "Thumb":  "#2ecc71",
    "Index":  "#3498db",
    "Middle": "#e67e22",
    "Pinky":  "#e74c3c",
}
TASK_NAMES  = set(TASK_COLORS)

WINDOW_SEC  = 30.0    # initial view width (seconds)
SCROLL_STEP = 10.0    # seconds per ← → keypress


# ── Loading helpers ────────────────────────────────────────────────────────────

def _load_source(cfg: dict):
    """
    Return (signal, times, sfreq, trials, label) for one config entry.

    signal : 1-D ndarray — ICA source time series (raw amplitude, a.u.)
    times  : 1-D ndarray — time in seconds
    sfreq  : float
    trials : list of dicts {task, onset, duration, energy}
    label  : str for the subplot title
    """
    folder    = get_folder(cfg["subj"], cfg["task"], cfg["sess"], cfg["ncl"], cfg["model"])
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in:\n  {folder}")

    print(f"\n[{cfg['label']}]")
    print(f"  Loading {len(mat_files)} file(s) from:\n    {folder}")
    raw, _, _ = build_raw_from_mat_files(mat_files)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)

    ica_path = get_ica_fif_path(cfg["subj"], cfg["task"])
    if not os.path.exists(ica_path):
        raise FileNotFoundError(
            f"Offline ICA not found:\n  {ica_path}\n"
            "Run  clean_ICA.py  or  clean_ica_offline.py  for this subject first."
        )
    print(f"  Using offline ICA: {ica_path}")
    ica = mne.preprocessing.read_ica(ica_path, verbose=False)

    sources_raw = ica.get_sources(raw)
    signal = sources_raw.get_data(picks=cfg["comp"])[0]   # shape (n_samples,)
    times  = sources_raw.times
    sfreq  = raw.info["sfreq"]

    trials = _extract_trials(raw, signal, sfreq)
    print(f"  ICA{cfg['comp']:03d} (comp={cfg['comp']}): {len(trials):3d} trials, "
          f"duration {times[-1]:.1f} s")

    return signal, times, sfreq, trials, cfg["label"]


def _extract_trials(raw, signal: np.ndarray, sfreq: float) -> list:
    """Return list of dicts: task, onset, duration, energy (Σx² over window)."""
    trials = []
    annots = raw.annotations
    n      = len(annots)

    for i in range(n):
        desc = annots[i]["description"]
        if desc not in TASK_NAMES:
            continue

        onset    = float(annots[i]["onset"])
        duration = 3.0   # default if no TrialEnd found

        for j in range(i + 1, n):
            nd = annots[j]["description"]
            if nd == "TrialEnd":
                duration = float(annots[j]["onset"]) - onset
                break
            if nd in TASK_NAMES:
                break

        s0     = int(onset * sfreq)
        s1     = min(int((onset + duration) * sfreq), len(signal))
        energy = float(np.sum(signal[s0:s1] ** 2))

        trials.append(dict(task=desc, onset=onset, duration=duration, energy=energy))

    return trials


# ── Energy detail window ───────────────────────────────────────────────────────

def _show_energy_detail_window(signal: np.ndarray, times: np.ndarray,
                                sfreq: float, trials: list, source_label: str):
    """
    Open a figure (non-blocking) showing, for each task, the first trial's
    signal and the full energy-calculation breakdown:
      - signal plot over time (allure)
      - first / last N sample values of the vector
      - number of samples
      - formula E = sum(x_i^2) with explicit first/last terms
      - final energy value
    """
    N_EDGE  = 5   # values shown at each end of the vector
    N_SQ    = 3   # squared terms shown explicitly in the formula

    # First trial per task
    first_trial: dict = {}
    for tr in trials:
        if tr["task"] not in first_trial:
            first_trial[tr["task"]] = tr

    present = [t for t in TASK_COLORS if t in first_trial]
    n_tasks = len(present)
    if n_tasks == 0:
        return

    fig = plt.figure(figsize=(6.5 * n_tasks, 9))
    fig.patch.set_facecolor("#0f172a")
    fig.canvas.manager.set_window_title(f"Energy Detail  --  {source_label}")

    gs = fig.add_gridspec(
        2, n_tasks,
        height_ratios=[2.2, 2.0],
        hspace=0.55, wspace=0.35,
        left=0.05, right=0.97, top=0.90, bottom=0.04,
    )

    for col, task in enumerate(present):
        tr    = first_trial[task]
        color = TASK_COLORS[task]

        s0  = int(tr["onset"] * sfreq)
        s1  = min(int((tr["onset"] + tr["duration"]) * sfreq), len(signal))
        seg = signal[s0:s1]
        t_r = times[s0:s1] - times[s0]   # relative time from 0
        n   = len(seg)
        E   = float(np.sum(seg ** 2))
        n_e = min(N_EDGE, n // 2)          # guard against very short windows

        # ── Signal plot ───────────────────────────────────────────────────────
        ax_sig = fig.add_subplot(gs[0, col])
        ax_sig.plot(t_r, seg**2, color=color, lw=0.8)
        ax_sig.fill_between(t_r, seg**2, alpha=0.12, color=color)
        ax_sig.axhline(0, color="#374151", lw=0.5, ls="--")
        ax_sig.set_facecolor("#1e293b")
        ax_sig.set_title(f"{task}  (1st trial)", color=color,
                         fontsize=11, fontweight="bold", pad=5)
        ax_sig.set_xlabel("t (s)", color="#94a3b8", fontsize=8)
        ax_sig.set_ylabel("x²  (a.u.)", color="#94a3b8", fontsize=8)
        ax_sig.tick_params(colors="#94a3b8", labelsize=7)
        for sp in ax_sig.spines.values():
            sp.set_edgecolor("#334155")

        # ── Formula panel ────────────────────────────────────────────────────
        ax_txt = fig.add_subplot(gs[1, col])
        ax_txt.set_facecolor("#1e293b")
        ax_txt.set_xlim(0, 1)
        ax_txt.set_ylim(0, 1)
        ax_txt.axis("off")
        for sp in ax_txt.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor("#334155")
            sp.set_linewidth(0.8)

        # --- vector preview --------------------------------------------------
        head_vals = "  ".join(f"{v: .4f}" for v in seg[:n_e])
        tail_vals = "  ".join(f"{v: .4f}" for v in seg[-n_e:])
        vec_line  = f"[ {head_vals}  ...  {tail_vals} ]"

        # --- squared terms for formula display --------------------------------
        n_sq  = min(N_SQ, n // 2)
        sq_h  = " + ".join(f"({v:.4f})^2" for v in seg[:n_sq])
        sq_t  = " + ".join(f"({v:.4f})^2" for v in seg[-2:])

        dur_s = tr["duration"]
        txt_body = (
            f"n = {n} samples   "
            f"({dur_s:.3f} s  @  {sfreq:.0f} Hz)\n"
            "\n"
            "x  =\n"
            f"  {vec_line}\n"
            "\n"
            f"E  =  x[0]^2 + x[1]^2 + ... + x[{n-1}]^2\n"
            "\n"
            f"   = {sq_h}\n"
            f"     + ... + {sq_t}\n"
        )
        ax_txt.text(
            0.04, 0.98, txt_body,
            transform=ax_txt.transAxes,
            ha="left", va="top",
            fontsize=7.2, color="#cbd5e1",
            fontfamily="monospace",
            linespacing=1.55,
            clip_on=True,
        )

        # --- final energy value (larger, coloured) ---------------------------
        ax_txt.text(
            0.04, 0.08,
            f"E  =  {E:.6e}",
            transform=ax_txt.transAxes,
            ha="left", va="bottom",
            fontsize=11, color=color, fontweight="bold",
            fontfamily="monospace",
        )

    fig.suptitle(source_label, color="#94a3b8", fontsize=9, y=0.96)
    fig.canvas.draw()


def _show_trial_detail_window(signal: np.ndarray, times: np.ndarray,
                               sfreq: float, trial: dict, source_label: str,
                               trial_idx: int, win_offset: int = 0):
    """
    Open a detail window for ONE specific trial (triggered by a click in the
    main viewer).  Shows: signal plot, vector preview, sample count, energy
    formula and final value.

    win_offset: pixel offset so successive click-windows don't stack exactly.
    """
    N_EDGE = 5
    N_SQ   = 3

    color = TASK_COLORS.get(trial["task"], "#ffffff")
    s0    = int(trial["onset"] * sfreq)
    s1    = min(int((trial["onset"] + trial["duration"]) * sfreq), len(signal))
    seg   = signal[s0:s1]
    t_rel = times[s0:s1] - times[s0]
    n     = len(seg)
    E     = float(np.sum(seg ** 2))
    n_e   = min(N_EDGE, n // 2)
    n_sq  = min(N_SQ, n // 2)

    title = (f"Trial #{trial_idx}  --  {trial['task']}"
             f"  |  onset = {trial['onset']:.2f} s"
             f"  |  {source_label}")

    fig = plt.figure(figsize=(9, 7))
    fig.patch.set_facecolor("#0f172a")
    fig.canvas.manager.set_window_title(title)

    # Shift window slightly so stacked clicks don't overlap perfectly.
    try:
        mng = fig.canvas.manager
        geom = mng.window.geometry()          # Tk geometry string "WxH+X+Y"
        parts = geom.replace("-", "+-").split("+")
        x = int(parts[1]) + win_offset * 30
        y = int(parts[2]) + win_offset * 30
        mng.window.geometry(f"{parts[0]}+{x}+{y}")
    except Exception:
        pass

    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 2.0],
                          hspace=0.45, left=0.08, right=0.96,
                          top=0.88, bottom=0.05)

    # ── Signal plot ───────────────────────────────────────────────────────────
    ax_sig = fig.add_subplot(gs[0])
    ax_sig.plot(t_rel, seg**2, color=color, lw=0.9)
    ax_sig.fill_between(t_rel, seg**2, alpha=0.14, color=color)
    ax_sig.axhline(0, color="#374151", lw=0.5, ls="--")
    ax_sig.set_facecolor("#1e293b")
    ax_sig.set_xlabel("t (s)", color="#94a3b8", fontsize=9)
    ax_sig.set_ylabel("x²  (a.u.)", color="#94a3b8", fontsize=9)
    ax_sig.set_title(f"{trial['task']}  —  Trial #{trial_idx}"
                     f"  (onset = {trial['onset']:.2f} s)",
                     color=color, fontsize=11, fontweight="bold", pad=5)
    ax_sig.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ax_sig.spines.values():
        sp.set_edgecolor("#334155")

    # ── Formula panel ─────────────────────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[1])
    ax_txt.set_facecolor("#1e293b")
    ax_txt.set_xlim(0, 1)
    ax_txt.set_ylim(0, 1)
    ax_txt.axis("off")
    for sp in ax_txt.spines.values():
        sp.set_visible(True)
        sp.set_edgecolor("#334155")
        sp.set_linewidth(0.8)

    head_vals = "  ".join(f"{v: .4f}" for v in seg[:n_e])
    tail_vals = "  ".join(f"{v: .4f}" for v in seg[-n_e:])
    sq_h      = " + ".join(f"({v:.4f})^2" for v in seg[:n_sq])
    sq_t      = " + ".join(f"({v:.4f})^2" for v in seg[-2:])

    body = (
        f"n = {n} samples   "
        f"({trial['duration']:.3f} s  @  {sfreq:.0f} Hz)\n"
        "\n"
        "x  =\n"
        f"  [ {head_vals}  ...  {tail_vals} ]\n"
        "\n"
        f"E  =  x[0]^2 + x[1]^2 + ... + x[{n-1}]^2\n"
        "\n"
        f"   = {sq_h}\n"
        f"     + ... + {sq_t}\n"
    )
    ax_txt.text(0.04, 0.98, body,
                transform=ax_txt.transAxes,
                ha="left", va="top",
                fontsize=8, color="#cbd5e1",
                fontfamily="monospace",
                linespacing=1.6, clip_on=True)
    ax_txt.text(0.04, 0.07, f"E  =  {E:.6e}",
                transform=ax_txt.transAxes,
                ha="left", va="bottom",
                fontsize=12, color=color, fontweight="bold",
                fontfamily="monospace")

    fig.suptitle(source_label, color="#64748b", fontsize=8, y=0.96)
    fig.canvas.draw()
    plt.pause(0.01)   # let the Tk event loop render the new window


# ── Interactive viewer ─────────────────────────────────────────────────────────

class DualSourceViewer:
    """
    Two-row matplotlib viewer that behaves like a minimal MNE browser.

    Each row shows one ICA source signal with:
      - coloured vertical span per trial (Thumb=green, Index=blue,
        Middle=orange, Pinky=red)
      - task name + Σx² energy annotated at the top of each span
    Navigation: ← / → scroll, ↑ / ↓ zoom, scroll-wheel zoom, Home/End.
    """

    def __init__(self, records: list):
        """records: list of (signal, times, sfreq, trials, label)."""
        self.records = records
        self.t_start = 0.0
        self.win_sec = WINDOW_SEC
        self.t_max   = max(r[1][-1] for r in records)
        # Fixed y-axis per row: [0, global_max(x²)]; vertical zoom adjusts the ceiling
        self.y_lims  = [(0.0, float(np.max(sig**2))) for sig, *_ in records]
        self.y_zoom  = 1.0   # Shift+↑ increases, Shift+↓ decreases

        n = len(records)
        self.fig, axes = plt.subplots(
            n, 1, figsize=(18, 4 * n), squeeze=False
        )
        self.axes = axes[:, 0]

        self.fig.canvas.manager.set_window_title("ICA Dual Source Viewer")
        self.fig.patch.set_facecolor("#111827")
        for ax in self.axes:
            ax.set_facecolor("#1f2937")

        self.fig.canvas.mpl_connect("key_press_event",  self._on_key)
        self.fig.canvas.mpl_connect("scroll_event",     self._on_scroll)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

        self._click_count = 0   # offset successive click-windows

        self._draw()
        self.fig.tight_layout()
        plt.show()

    # ── draw ──────────────────────────────────────────────────────────────────

    def _draw(self):
        t0 = self.t_start
        t1 = t0 + self.win_sec

        for i, (ax, (signal, times, sfreq, trials, label)) in enumerate(
                zip(self.axes, self.records)):
            ax.clear()
            ax.set_facecolor("#1f2937")

            at0 = max(t0, 0.0)
            at1 = min(t1, times[-1])

            # ── Signal ────────────────────────────────────────────────────────
            s0 = int(np.searchsorted(times, at0))
            s1 = int(np.searchsorted(times, at1))
            ax.plot(times[s0:s1], signal[s0:s1]**2,
                    color="#38bdf8", lw=0.5, alpha=0.85)

            # ── Trial spans + energy labels ────────────────────────────────────
            for tr in trials:
                on  = tr["onset"]
                off = on + tr["duration"]
                if off < at0 or on > at1:
                    continue

                col = TASK_COLORS.get(tr["task"], "#ffffff")

                # Shaded span
                ax.axvspan(on, off, alpha=0.13, color=col, linewidth=0)
                # Onset marker
                ax.axvline(on, color=col, lw=0.7, alpha=0.6)

                # Label at top: "TaskName\n2.31e+04"
                mid   = (on + off) / 2.0
                e_str = f"{tr['task']}\n{tr['energy']:.2e}"
                ax.text(
                    mid, 0.97, e_str,
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top",
                    fontsize=6.5, color=col, fontweight="bold",
                    clip_on=True,
                )

            # ── Axis styling ───────────────────────────────────────────────────
            ax.set_xlim(at0, at1)
            # Fixed y-axis: global max divided by current vertical zoom
            ax.set_ylim(0.0, self.y_lims[i][1] / self.y_zoom)
            ax.set_title(label, color="#e2e8f0", fontsize=10, pad=5)
            ax.set_ylabel("x²  (a.u.)", color="#94a3b8", fontsize=8)
            ax.tick_params(colors="#94a3b8", labelsize=7)
            for sp in ax.spines.values():
                sp.set_edgecolor("#374151")

        self.axes[-1].set_xlabel(
            f"Time (s)   [window: {self.win_sec:.0f} s  |  "
            f"{self.t_start:.1f} – {self.t_start + self.win_sec:.1f} s]",
            color="#94a3b8", fontsize=8,
        )

        # Task colour legend — top subplot only
        patches = [
            mpatches.Patch(color=c, label=t, alpha=0.8)
            for t, c in TASK_COLORS.items()
        ]
        self.axes[0].legend(
            handles=patches, loc="upper right", fontsize=7,
            facecolor="#1f2937", edgecolor="#374151", labelcolor="#e2e8f0",
        )

        # Navigation hint — bottom
        self.fig.text(
            0.01, 0.005,
            "← / →  scroll  │  ↑ / ↓  h-zoom  │  Shift+↑ / Shift+↓  v-zoom  │  scroll-wheel  h-zoom  │  Home / End  jump",
            color="#6b7280", fontsize=7, va="bottom",
        )

        self.fig.canvas.draw_idle()

    # ── navigation ────────────────────────────────────────────────────────────

    def _on_key(self, ev):
        k = ev.key
        if k == "right":
            self.t_start = min(
                self.t_start + SCROLL_STEP,
                max(self.t_max - self.win_sec, 0.0),
            )
        elif k == "left":
            self.t_start = max(self.t_start - SCROLL_STEP, 0.0)
        elif k == "up":
            self.win_sec = max(self.win_sec / 1.5, 2.0)
        elif k == "down":
            self.win_sec = min(self.win_sec * 1.5, self.t_max)
        elif k == "shift+up":
            self.y_zoom = min(self.y_zoom * 1.5, 1000.0)
        elif k == "shift+down":
            self.y_zoom = max(self.y_zoom / 1.5, 0.1)
        elif k == "home":
            self.t_start = 0.0
        elif k == "end":
            self.t_start = max(self.t_max - self.win_sec, 0.0)
        else:
            return
        self._draw()

    def _on_scroll(self, ev):
        factor = 1.0 / 1.3 if ev.step > 0 else 1.3
        self.win_sec = float(np.clip(self.win_sec * factor, 2.0, self.t_max))
        self._draw()

    def _on_click(self, ev):
        """Left-click inside a trial span → open a detail window for that trial."""
        if ev.button != 1 or ev.inaxes is None or ev.xdata is None:
            return
        t_click = ev.xdata
        for ax, (signal, times, sfreq, trials, label) in zip(self.axes, self.records):
            if ev.inaxes is not ax:
                continue
            for idx, tr in enumerate(trials):
                if tr["onset"] <= t_click <= tr["onset"] + tr["duration"]:
                    _show_trial_detail_window(
                        signal, times, sfreq, tr, label,
                        trial_idx=idx + 1,
                        win_offset=self._click_count,
                    )
                    self._click_count += 1
                    break
            break


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ICA Dual Source Viewer")
    print("=" * 60)

    records = [_load_source(cfg) for cfg in SOURCES]

    # Open one energy-detail window per source (non-blocking — all stay open
    # alongside the main viewer until it is closed).
    for signal, times, sfreq, trials, label in records:
        _show_energy_detail_window(signal, times, sfreq, trials, label)

    DualSourceViewer(records)


if __name__ == "__main__":
    main()

"""
viz_correlations.py

Visualizes ICA component correlations with task signals.

ICA fitting is skipped if a cached .fif file already exists (see get_ica_fif_path).

Four figure types are opened simultaneously:
  1. Heatmap            — all tasks × all components (fixed scale 0–0.6).
  2. Per-task bar chart — one figure per task, same fixed scale.
  3. plot_sources_with_task  — scrollable page viewer: all ICA source traces
                               with task-signal subplots and task-period shading.
  4. plot_source_inspector   — per-component detail: raw source + Hilbert
                               envelope, z-scored task overlay, correlation
                               scores in the title, Prev/Next buttons.

Usage:
    Online:  python viz_correlations.py <subj> <sess> <nclass> <task> <model>
    Offline: python viz_correlations.py <subj> <task> OFF
"""

import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.transforms
from scipy.stats import zscore

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions import (
    get_folder, get_offline_folder,
    build_raw_from_mat_files,
    build_task_vectors, compute_envelope,
    get_standard_args, get_offline_args,
    get_ica_fif_path, fit_or_load_ica,
)

_TASK_COLORS = ["green", "blue", "orange", "red", "gray",
                "purple", "brown", "pink"]
_BG_COLORS   = ["#4CAF50", "#2196F3", "#FF9800", "#F44336",
                "#9C27B0", "#795548", "#E91E63", "#009688"]


# =============================================================================
# INTERACTIVE VIEWER 2 — per-component source inspector
# =============================================================================

def plot_source_inspector(ica, raw, corr_matrix, tasks_list):
    """
    Dedicated interactive figure for single ICA component inspection.

    Shows the raw ICA source signal and its Hilbert envelope side-by-side
    in time, with task-period background shading and correlation scores in
    the title.

    Keyboard / button controls:
      n / →             next component
      p / ←             previous component
      + / ↑             amplitude scale up
      - / ↓             amplitude scale down
      Home              zoom in  (×1.5)
      End               zoom out (×1.5)
      Shift+← / →       scroll backward / forward (25 % of window)
      RadioButtons      select which task signal to overlay
    """
    from matplotlib.widgets import Button, RadioButtons
    from matplotlib.patches import Patch

    sources_data = ica.get_sources(raw).get_data()   # (n_comp, n_samples)
    n_comp  = sources_data.shape[0]
    times   = raw.times
    sfreq   = float(raw.info["sfreq"])
    n_tasks = len(tasks_list)

    envelopes = np.array([compute_envelope(src, sfreq) for src in sources_data])

    comp_ptp = np.array([
        np.ptp(src) if np.ptp(src) > 1e-15 else 1.0
        for src in sources_data
    ])

    BASE_YLIM   = 0.65
    TASK_HEIGHT = 0.50

    task_vectors    = build_task_vectors(raw)
    _task_names     = list(task_vectors.keys())
    task_vectors_z  = {}
    task_z_peaks    = {}
    for _name, _vec in task_vectors.items():
        _vz = zscore(_vec) if np.std(_vec) > 0 else _vec
        task_vectors_z[_name] = _vz
        _peak = float(np.max(np.abs(_vz))) if np.size(_vz) else 1.0
        task_z_peaks[_name]   = _peak if _peak > 1e-12 else 1.0

    state = {
        "comp":      0,
        "t_start":   float(times[0]),
        "t_end":     float(times[-1]),
        "amp_scale": 1.0,
    }

    fig = plt.figure(figsize=(18, 5))
    fig.patch.set_facecolor("#F0F4F8")
    try:
        fig.canvas.manager.set_window_title("ICA Source Inspector")
    except Exception:
        pass

    fig.subplots_adjust(left=0.06, right=0.85, top=0.80, bottom=0.22)
    ax = fig.add_subplot(111)
    ax.set_facecolor("#FAFAFA")
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")
    ax.tick_params(colors="#555555")
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("Amplitude (normalized)", fontsize=9)
    ax.axhline(0, color="#DDDDDD", linewidth=0.5, zorder=0)

    for t_idx, (task_name, vec) in enumerate(task_vectors.items()):
        bg_col  = _BG_COLORS[t_idx % len(_BG_COLORS)]
        active  = (vec > 0.5).astype(int)
        changes = np.diff(np.concatenate([[0], active, [0]]))
        s_idx   = np.where(changes ==  1)[0]
        e_idx   = np.where(changes == -1)[0]
        for s, e in zip(s_idx, e_idx):
            ts = times[min(s,     len(times) - 1)]
            te = times[min(e - 1, len(times) - 1)]
            ax.axvspan(ts, te, color=bg_col, alpha=0.07, linewidth=0, zorder=0)

    line_sig, = ax.plot([], [], color="#1565C0", linewidth=0.9,
                        label="ICA source", zorder=2, alpha=0.85)
    line_env, = ax.plot([], [], color="#B71C1C", linewidth=1.5,
                        label="Envelope  |Hilbert|", zorder=3)
    fill_ref  = [None]

    task_lines = {}
    for t_idx, task_name in enumerate(_task_names):
        col = _BG_COLORS[t_idx % len(_BG_COLORS)]
        ln, = ax.plot([], [], color=col, linewidth=1.4, alpha=0.9,
                      linestyle="--", drawstyle="steps-post",
                      label="{} (z-task)".format(task_name), zorder=1.5,
                      visible=(t_idx == 0))
        task_lines[task_name] = ln

    _task_patches = [
        Patch(facecolor=_BG_COLORS[t_idx % len(_BG_COLORS)],
              edgecolor=_BG_COLORS[t_idx % len(_BG_COLORS)],
              alpha=0.55, label=task_name)
        for t_idx, task_name in enumerate(task_vectors.keys())
    ]
    ax.legend(
        handles=[line_sig, line_env] + _task_patches,
        loc="upper right", fontsize=9, framealpha=0.85,
        title="Signals & Tasks", title_fontsize=8,
        borderpad=0.8, labelspacing=0.4,
    )

    _ax_prev  = fig.add_axes([0.36, 0.05, 0.08, 0.09])
    _ax_next  = fig.add_axes([0.56, 0.05, 0.08, 0.09])
    _ax_label = fig.add_axes([0.44, 0.05, 0.12, 0.09])
    _ax_label.set_axis_off()

    btn_prev = Button(_ax_prev, "◀  Prev", color="#D0D8E8", hovercolor="#B0C0D8")
    btn_next = Button(_ax_next, "Next  ▶", color="#D0D8E8", hovercolor="#B0C0D8")
    for _b in (btn_prev, btn_next):
        _b.label.set_fontsize(10)

    comp_text = _ax_label.text(
        0.5, 0.5, "", ha="center", va="center",
        fontsize=13, fontweight="bold",
        transform=_ax_label.transAxes,
    )

    _ax_radio = fig.add_axes([0.865, 0.22, 0.12, 0.58])
    _ax_radio.set_facecolor("#FAFAFA")
    for spine in _ax_radio.spines.values():
        spine.set_color("#CCCCCC")
    _ax_radio.set_title("Task signal", fontsize=9, fontweight="bold",
                         color="#333333", pad=6)

    _label_colors = [_BG_COLORS[i % len(_BG_COLORS)] for i in range(len(_task_names))]
    try:
        radio_buttons = RadioButtons(
            _ax_radio, labels=_task_names, active=0,
            label_props={"color": _label_colors, "fontweight": ["bold"] * len(_task_names)},
            radio_props={"facecolor": _label_colors, "edgecolor": _label_colors},
        )
    except (TypeError, ValueError):
        radio_buttons = RadioButtons(_ax_radio, labels=_task_names, active=0)
        for _lbl, _col in zip(radio_buttons.labels, _label_colors):
            _lbl.set_color(_col); _lbl.set_fontweight("bold")

    def _on_radio(selected):
        for _name, _ln in task_lines.items():
            _ln.set_visible(_name == selected)
        fig.canvas.draw_idle()

    radio_buttons.on_clicked(_on_radio)

    def _update():
        c_idx  = state["comp"]
        t0, t1 = state["t_start"], state["t_end"]
        scale  = state["amp_scale"]

        mask = (times >= t0) & (times <= t1)
        if not np.any(mask):
            return

        t_win   = times[mask]
        sig_win = sources_data[c_idx][mask]
        env_win = envelopes[c_idx][mask]

        ptp_c    = comp_ptp[c_idx]
        sig_plot = sig_win / ptp_c
        env_plot = env_win / ptp_c

        line_sig.set_data(t_win, sig_plot)
        line_env.set_data(t_win, env_plot)

        if fill_ref[0] is not None:
            fill_ref[0].remove()
        fill_ref[0] = ax.fill_between(
            t_win, 0, env_plot, color="#EF9A9A", alpha=0.25, zorder=1)

        for task_name in _task_names:
            ln    = task_lines[task_name]
            vec_z = task_vectors_z[task_name][mask]
            ln.set_data(t_win, (vec_z / task_z_peaks[task_name]) * TASK_HEIGHT)

        ax.set_xlim(t0, t1)
        ylim_half = BASE_YLIM / max(scale, 1e-6)
        ax.set_ylim(-ylim_half, ylim_half)

        is_excl = c_idx in ica.exclude
        status  = "EXCLUDED" if is_excl else "KEPT"
        s_color = "#C62828" if is_excl else "#2E7D32"

        corr_str = (
            "   |   ".join(
                "{}: {:.3f}".format(tasks_list[t], corr_matrix[t, c_idx])
                for t in range(n_tasks)
            ) if n_tasks else "—"
        )
        fig.suptitle(
            "ICA {:03d} / {:d}   [{}]   —   Correlations:  {}\n"
            "n/→: next   p/←: prev   +/↑: amp+   -/↓: amp−   "
            "Home: zoom in   End: zoom out   Shift+←/→: scroll".format(
                c_idx, n_comp - 1, status, corr_str),
            fontsize=9, fontweight="bold", color="#222222", y=0.97,
        )
        comp_text.set_text("ICA {:03d}".format(c_idx))
        comp_text.set_color(s_color)
        fig.canvas.draw_idle()

    def _on_key(event):
        k = event.key
        if   k in ("n", "right"):  state["comp"] = min(n_comp - 1, state["comp"] + 1)
        elif k in ("p", "left"):   state["comp"] = max(0, state["comp"] - 1)
        elif k in ("+", "=", "up"):   state["amp_scale"] *= 1.5
        elif k in ("-", "down"):      state["amp_scale"] = max(state["amp_scale"] / 1.5, 1e-3)
        elif k == "home":
            mid  = (state["t_start"] + state["t_end"]) / 2
            half = (state["t_end"] - state["t_start"]) / (2 * 1.5)
            state["t_start"] = max(float(times[0]),  mid - half)
            state["t_end"]   = min(float(times[-1]), mid + half)
        elif k == "end":
            mid  = (state["t_start"] + state["t_end"]) / 2
            half = (state["t_end"] - state["t_start"]) / 2 * 1.5
            state["t_start"] = max(float(times[0]),  mid - half)
            state["t_end"]   = min(float(times[-1]), mid + half)
        elif k == "shift+right":
            window = state["t_end"] - state["t_start"]
            shift  = window * 0.25
            new_end          = min(float(times[-1]), state["t_end"] + shift)
            state["t_end"]   = new_end
            state["t_start"] = max(float(times[0]), new_end - window)
        elif k == "shift+left":
            window = state["t_end"] - state["t_start"]
            shift  = window * 0.25
            new_start        = max(float(times[0]), state["t_start"] - shift)
            state["t_start"] = new_start
            state["t_end"]   = min(float(times[-1]), new_start + window)
        else:
            return
        _update()

    def _on_prev(_): state["comp"] = max(0, state["comp"] - 1); _update()
    def _on_next(_): state["comp"] = min(n_comp - 1, state["comp"] + 1); _update()

    btn_prev.on_clicked(_on_prev)
    btn_next.on_clicked(_on_next)
    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig._inspector_buttons = (btn_prev, btn_next, radio_buttons)   # prevent GC

    _update()
    return fig


# =============================================================================
# INTERACTIVE VIEWER 1 — scrollable page viewer (all sources + task subplots)
# =============================================================================

def plot_sources_with_task(ica, raw_filt, raw, corr_matrix, tasks_list, n_per_page=10):
    """
    MNE-style scrollable viewer showing all ICA source traces in vertical-offset
    layout, with task-signal subplots below and task-period background shading.

    Opens non-blocking alongside the source inspector (both shown simultaneously).

    Keyboard controls:
      Left / PgUp       previous page
      Right / PgDn      next page
      + / ↑             amplitude scale up
      - / ↓             amplitude scale down
      Home              zoom in on time axis  (×1.5)
      End               zoom out on time axis (×1.5)
      Shift+← / →       scroll backward / forward (20 % of window)
    """
    sources_data = ica.get_sources(raw).get_data()   # (n_comp, n_samples)
    n_comp       = sources_data.shape[0]
    times        = raw.times
    n_pages      = max(1, int(np.ceil(n_comp / n_per_page)))
    n_tasks      = len(tasks_list)

    task_vectors = build_task_vectors(raw)
    tvec_ds      = list(task_vectors.values())

    state = {
        "page":    0,
        "scale":   1.0,
        "t_start": float(times[0]),
        "t_end":   float(times[-1]),
    }
    spacing = 1.0

    fig = plt.figure(figsize=(18, max(n_per_page * 0.9 + n_tasks * 1.2, 8)))
    fig.patch.set_facecolor("#F5F5F5")

    gs = plt.GridSpec(
        1 + n_tasks, 1,
        height_ratios=[n_per_page] + [1] * n_tasks,
        hspace=0.06,
    )
    ax_ica   = fig.add_subplot(gs[0, 0])
    ax_tasks = [fig.add_subplot(gs[i + 1, 0], sharex=ax_ica) for i in range(n_tasks)]

    for ax in [ax_ica] + ax_tasks:
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax_ica.set_xlim(times[0], times[-1])
    ax_ica.tick_params(axis="x", labelbottom=False)

    blank     = np.zeros(len(times))
    trans     = ax_ica.get_yaxis_transform()
    ref_lines, sig_lines, label_texts = [], [], []
    for _ in range(n_per_page):
        rl, = ax_ica.plot(times, blank, color="#E0E0E0", linewidth=0.4, zorder=0)
        sl, = ax_ica.plot(times, blank, linewidth=0.6, alpha=0.85, zorder=1)
        lt  = ax_ica.text(
            0.002, 0, "", transform=trans,
            fontsize=7, va="center", ha="left", fontweight="bold",
            clip_on=True, zorder=2,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
        )
        rl.set_visible(False); sl.set_visible(False); lt.set_visible(False)
        ref_lines.append(rl); sig_lines.append(sl); label_texts.append(lt)

    for t_idx, (task_name, vec) in enumerate(zip(task_vectors.keys(), tvec_ds)):
        ax    = ax_tasks[t_idx]
        color = _TASK_COLORS[t_idx % len(_TASK_COLORS)]
        ax.fill_between(times, 0, vec, color=color, alpha=0.35, linewidth=0)
        ax.plot(times, vec, color=color, linewidth=0.9)
        ax.set_xlim(times[0], times[-1])
        ax.set_ylim(-0.1, 1.3)
        ax.set_yticks([0, 1])
        ax.set_ylabel(task_name, fontsize=8, rotation=0,
                      labelpad=100, va="center", color=color, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(False)
    if ax_tasks:
        ax_tasks[-1].set_xlabel("Time (s)", fontsize=9)

    ann_trans = matplotlib.transforms.blended_transform_factory(
        ax_ica.transData, ax_ica.transAxes)
    trial_end_onsets = sorted(
        float(a["onset"]) for a in raw.annotations if a["description"] == "TrialEnd"
    )
    for ann in raw.annotations:
        desc = ann["description"]
        if desc not in task_vectors:
            continue
        color  = _TASK_COLORS[list(task_vectors.keys()).index(desc) % len(_TASK_COLORS)]
        onset  = float(ann["onset"])
        dur    = float(ann["duration"])
        if dur <= 0:
            future = [t for t in trial_end_onsets if t > onset]
            dur = (future[0] - onset) if future else 3.0
        ax_ica.axvspan(onset, onset + dur, color=color, alpha=0.10, linewidth=0, zorder=0)
        ax_ica.text(
            onset + dur * 0.5, 0.98, desc,
            transform=ann_trans,
            color=color, fontsize=6.5, fontweight="bold",
            ha="center", va="top", clip_on=True,
        )

    def draw_page():
        page      = state["page"]
        scale     = state["scale"]
        start     = page * n_per_page
        end       = min(start + n_per_page, n_comp)
        n_on_page = end - start

        fig.suptitle(
            "ICA Sources + Task signals  —  Page {}/{}  "
            "[Left/PgUp: prev   Right/PgDn: next   +/↑: amp+   -/↓: amp−   "
            "Home: zoom in   End: zoom out   Shift+←/→: scroll]".format(
                page + 1, n_pages),
            fontsize=10, fontweight="bold",
        )

        ytick_pos, ytick_labels = [], []

        for row in range(n_per_page):
            rl = ref_lines[row]; sl = sig_lines[row]; lt = label_texts[row]
            if row < n_on_page:
                c_idx    = start + row
                offset   = (n_on_page - 1 - row) * spacing
                sig      = sources_data[c_idx]
                sig_norm = (sig / (np.ptp(sig) + 1e-12)) * 0.45 * scale
                is_excl  = c_idx in ica.exclude
                color    = "#AAAAAA" if is_excl else "#2C7BB6"

                rl.set_ydata(np.full(len(times), offset)); rl.set_visible(True)
                sl.set_ydata(sig_norm + offset); sl.set_color(color); sl.set_visible(True)
                lt.set_position((0.002, offset)); lt.set_color(color); lt.set_visible(True)

                corr_str = "  ".join(
                    "{}: {:.2f}".format(tasks_list[t], corr_matrix[t, c_idx])
                    for t in range(n_tasks)
                )
                ytick_pos.append(offset)
                ytick_labels.append(
                    "ICA{:03d}{}\n{}".format(c_idx, " [X]" if is_excl else "", corr_str)
                )
            else:
                rl.set_visible(False); sl.set_visible(False); lt.set_visible(False)

        ax_ica.set_yticks(ytick_pos)
        ax_ica.set_yticklabels(ytick_labels, fontsize=6)
        ax_ica.set_ylim(-spacing * 0.6, n_on_page * spacing - spacing * 0.4)
        ax_ica.set_xlim(state["t_start"], state["t_end"])
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in ("pagedown", "right"):
            if state["page"] < n_pages - 1: state["page"] += 1; draw_page()
        elif event.key in ("pageup", "left"):
            if state["page"] > 0: state["page"] -= 1; draw_page()
        elif event.key in ("+", "=", "up"):
            state["scale"] *= 1.5; draw_page()
        elif event.key in ("-", "down"):
            state["scale"] = max(state["scale"] / 1.5, 1e-3); draw_page()
        elif event.key == "home":
            mid  = (state["t_start"] + state["t_end"]) / 2
            half = (state["t_end"] - state["t_start"]) / (2 * 1.5)
            state["t_start"] = max(float(times[0]), mid - half)
            state["t_end"]   = min(float(times[-1]), mid + half)
            ax_ica.set_xlim(state["t_start"], state["t_end"]); fig.canvas.draw_idle()
        elif event.key == "end":
            mid  = (state["t_start"] + state["t_end"]) / 2
            half = (state["t_end"] - state["t_start"]) / 2 * 1.5
            state["t_start"] = max(float(times[0]), mid - half)
            state["t_end"]   = min(float(times[-1]), mid + half)
            ax_ica.set_xlim(state["t_start"], state["t_end"]); fig.canvas.draw_idle()
        elif event.key == "shift+right":
            window = state["t_end"] - state["t_start"]; shift = window * 0.20
            new_end = min(float(times[-1]), state["t_end"] + shift)
            state["t_end"] = new_end; state["t_start"] = new_end - window
            ax_ica.set_xlim(state["t_start"], state["t_end"]); fig.canvas.draw_idle()
        elif event.key == "shift+left":
            window = state["t_end"] - state["t_start"]; shift = window * 0.20
            new_start = max(float(times[0]), state["t_start"] - shift)
            state["t_start"] = new_start; state["t_end"] = new_start + window
            ax_ica.set_xlim(state["t_start"], state["t_end"]); fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    draw_page()
    plt.show(block=False)


# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def run_correlation_analysis(subj_id, task, session=1, nclass=2,
                              model_type="Orig", is_offline=False):
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

    # 1. Fit ICA (or load from cache)
    ica_cache_path = (get_ica_fif_path(subj_id, task) if is_offline
                      else get_ica_fif_path(subj_id, task, session, nclass, model_type))
    raw_filt = raw.copy().filter(1.0, None, verbose=False)
    ica = fit_or_load_ica(raw_filt, ica_cache_path)

    # 2. Compute correlation matrix
    print("Computing task <-> ICA correlations ...")
    sources_data = ica.get_sources(raw).get_data()
    sfreq        = float(raw.info["sfreq"])
    task_vectors = build_task_vectors(raw)

    if not task_vectors:
        print("\n[ERROR] No task annotations found. Cannot compute correlations.")
        return

    tasks_list   = list(task_vectors.keys())
    n_tasks      = len(tasks_list)
    n_comp       = sources_data.shape[0]
    corr_matrix  = np.zeros((n_tasks, n_comp))

    for t_idx, task_name in enumerate(tasks_list):
        vec   = task_vectors[task_name]
        vec_z = zscore(vec) if np.std(vec) > 0 else vec
        for c_idx in range(n_comp):
            comp_env = compute_envelope(sources_data[c_idx], sfreq)
            if np.std(comp_env) > 0 and np.std(vec_z) > 0:
                r = np.corrcoef(vec_z, zscore(comp_env))[0, 1]
                corr_matrix[t_idx, c_idx] = 0.0 if np.isnan(r) else abs(r)

    # 3. Print summary
    for t_idx, task_name in enumerate(tasks_list):
        print(f"\nTask: {task_name}")
        print("Correlations:", np.round(corr_matrix[t_idx], 3))

    # 4. Static heatmap (non-blocking, stays open alongside interactive viewers)
    plt.figure(figsize=(12, 6))
    plt.imshow(corr_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=0.6)
    plt.colorbar(label="Absolute Correlation")
    plt.yticks(range(n_tasks), tasks_list)
    plt.xticks(range(n_comp))
    plt.xlabel("ICA Components")
    plt.ylabel("Tasks")
    plt.title(f"Task vs ICA Correlation Heatmap — S{subj_id:02} | {task}")
    plt.tight_layout()
    plt.show(block=False)

    # 4b. Per-task bar charts (one figure per task, same fixed scale)
    for t_idx, task_name in enumerate(tasks_list):
        plt.figure(figsize=(10, 4))
        plt.bar(range(n_comp), corr_matrix[t_idx],
                color=_BG_COLORS[t_idx % len(_BG_COLORS)], alpha=0.8)
        plt.axhline(0.6, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
        plt.title(f"ICA Correlation with Task: {task_name} — S{subj_id:02} | {task}")
        plt.xlabel("ICA Component")
        plt.ylabel("Absolute Correlation")
        plt.ylim(0, 0.6)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show(block=False)

    # 5. Interactive viewers — all opened simultaneously, single blocking show at end
    plot_sources_with_task(ica, raw_filt, raw, corr_matrix, tasks_list)
    plot_source_inspector(ica, raw, corr_matrix, tasks_list)
    plt.show(block=True)


def main():
    if len(sys.argv) > 1 and sys.argv[-1].upper() == "OFF":
        sys.argv.pop()
        subj_id, task = get_offline_args(description="ICA Correlation - Offline")
        run_correlation_analysis(subj_id, task, is_offline=True)
    else:
        subj, sess, ncl, task, model = get_standard_args(description="ICA Correlation - Online")
        run_correlation_analysis(subj, task, sess, ncl, model)


if __name__ == "__main__":
    main()

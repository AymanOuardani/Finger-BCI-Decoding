"""
EEG_Visualizer.py

Interactive EEG signal viewer for the STING dataset.
Click on a channel name in the viewer → highlights it on the topomap.

Usage:
    Online:  python EEG_Visualizer.py <subj> <sess> <nclass> <task> <model> [band]
    Offline: python EEG_Visualizer.py <subj> <task> OFF [band]

    [band] (optional, last argument) selects the band-pass applied before viewing:
        Alpha (8-13 Hz), Beta (13-30 Hz), Delta (1-4 Hz), Fullband (default, no band-pass).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import numpy as np
from collections import Counter
import scipy.io
import scipy.signal
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import mne
mne.set_log_level("WARNING")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from Functions import (
    make_info, get_folder, get_offline_folder,
    build_raw_from_mat_files,
    get_standard_args, get_offline_args,
)
from config import SRATE


EVENT_ID = {"Thumb": 1, "Index": 2, "Middle": 3, "Pinky": 4, "TrialEnd": 9}

# Band-pass ranges (Hz) selectable from the command line. Fullband = no band-pass.
BANDS = {
    "Delta":    (1.0, 4.0),
    "Alpha":    (8.0, 13.0),
    "Beta":     (13.0, 30.0),
    "Fullband": None,
}


def apply_band_filter(raw, band):
    """Band-pass filter `raw` in place. 'Fullband' leaves the signal untouched."""
    freqs = BANDS.get(band)
    if freqs is None:
        print(f"  Band           : {band} (no band-pass applied)")
        return raw
    l_freq, h_freq = freqs
    print(f"  Band           : {band} ({l_freq:g}-{h_freq:g} Hz band-pass)")
    raw.filter(l_freq, h_freq, verbose=False)
    return raw


# ── Data loading ──────────────────────────────────────────────────────────────

def load_online_raw(subj_id, task, session, nclass, model_type, band="Fullband"):
    """Load and concatenate all .mat trial files for one online (BCI) session into an MNE Raw object,
    resampling to SRATE if needed and attaching event annotations for viewing."""
    folder    = get_folder(subj_id, task, session, nclass, model_type)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in:\n  {folder}")

    print(f"\nLoading {len(mat_files)} .mat file(s) from:\n  {folder}")

    id_to_label  = {v: k for k, v in EVENT_ID.items()}
    finger_label = {1: "Thumb", 2: "Index", 3: "Middle", 4: "Pinky"}
    all_signals, all_events, sample_offset = [], [], 0

    for fpath in mat_files:
        mat     = scipy.io.loadmat(fpath)
        eeg     = mat["eeg"]
        signals = eeg["data"][0][0].astype(float)
        srate   = int(eeg["fsample"][0][0][0][0])
        event   = mat["event"]

        if srate != SRATE:
            # Recording sampling rate differs from the project-wide SRATE: resample
            # the signal (Fourier-based) so all sessions share a common time base.
            n_new   = int(signals.shape[1] * SRATE / srate)
            signals = scipy.signal.resample(signals, n_new, axis=1)

        for i in range(event.shape[1]):
            evt   = event[0, i]
            etype = str(evt["type"][0]).strip()
            samp  = int(evt["sample"][0][0]) - 1
            if etype == "Target":
                try:
                    val = int(evt["value"][0][0])
                except Exception:
                    val = 0
                label   = finger_label.get(val, f"Target{val}")
                evt_int = EVENT_ID.get(label, 99)
                all_events.append([samp + sample_offset, 0, evt_int])
            elif etype == "TrialEnd":
                all_events.append([samp + sample_offset, 0, EVENT_ID["TrialEnd"]])

        all_signals.append(signals)
        sample_offset += signals.shape[1]

    data   = np.concatenate(all_signals, axis=1)
    info, _ = make_info()
    # Source data is in microvolts; MNE expects SI units (volts) internally.
    raw    = mne.io.RawArray(data * 1e-6, info, verbose=False)

    if all_events:
        events_arr = np.array(all_events, dtype=int)
        annot = mne.annotations_from_events(
            events_arr, sfreq=SRATE, event_desc=id_to_label, verbose=False,
        )
        raw.set_annotations(annot)

    # Remove 60 Hz mains-power line noise and its harmonics up to 500 Hz.
    raw.notch_filter(np.arange(60, 501, 60))
    apply_band_filter(raw, band)

    counts = Counter(id_to_label.get(e[2], str(e[2])) for e in all_events)
    print(f"  Total duration : {raw.times[-1]:.1f} s")
    print(f"  Channels       : {raw.info['nchan']}")
    print(f"  Events         : {dict(counts)}")
    return raw


def load_offline_raw(subj_id, task, band="Fullband"):
    """Load and concatenate all .mat trial files for one offline (non-BCI) recording into an MNE Raw object."""
    folder    = get_offline_folder(subj_id, task)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found in:\n  {folder}")

    print(f"\nLoading {len(mat_files)} offline .mat file(s) from:\n  {folder}")
    raw, all_events, id_to_label = build_raw_from_mat_files(mat_files)
    raw.notch_filter(np.arange(60, 501, 60))
    apply_band_filter(raw, band)

    counts = Counter(id_to_label.get(e[2], str(e[2])) for e in all_events)
    print(f"  Total duration : {raw.times[-1]:.1f} s")
    print(f"  Channels       : {raw.info['nchan']}")
    print(f"  Events         : {dict(counts)}")
    return raw


# ── Topomap ───────────────────────────────────────────────────────────────────

class TopoHighlighter:
    """Persistent topomap. Call .highlight(ch_name) to mark a channel in red."""

    def __init__(self, info, ch_names):
        """Create the topomap figure and draw the unhighlighted (base) sensor layout."""
        self.info     = info
        self.ch_names = ch_names
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.patch.set_facecolor("white")
        self.fig.canvas.manager.set_window_title("Electrode Map")
        self._draw_base()
        plt.show(block=False)
        self.fig.canvas.draw()

    def _draw_base(self, highlighted_idx=None):
        """Redraw the topomap; if `highlighted_idx` is given, mark that electrode in red with a label."""
        self.ax.clear()
        # A dummy value map where only the selected channel is "hot" (1.0) lets us
        # reuse plot_topomap purely as a way to render the sensor layout / highlight.
        values = np.zeros(len(self.ch_names))
        if highlighted_idx is not None:
            values[highlighted_idx] = 1.0

        mne.viz.plot_topomap(
            values, self.info,
            axes=self.ax,
            cmap="Reds",
            vlim=(0, 1),
            contours=0,
            extrapolate="head",
            sphere=(0., 0., 0., 0.095),
            outlines="head",
            show=False,
            sensors=True,
        )

        if highlighted_idx is not None:
            ch_name = self.ch_names[highlighted_idx]
            try:
                pos2d = mne.channels.layout._find_topomap_coords(
                    self.info, picks=[highlighted_idx]
                )
                x, y = pos2d[0]
                self.ax.plot(x, y, "o",
                             color="red", markersize=14,
                             markeredgecolor="darkred", markeredgewidth=2,
                             zorder=10)
                self.ax.annotate(
                    ch_name,
                    xy=(x, y), xytext=(x + 0.02, y + 0.02),
                    fontsize=11, fontweight="bold", color="darkred",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", ec="darkred", alpha=0.85),
                    zorder=11,
                )
            except Exception:
                pass
            self.ax.set_title(
                f"Selected electrode: {ch_name}",
                fontsize=12, fontweight="bold", color="darkred"
            )
        else:
            self.ax.set_title(
                "Click a channel in the EEG viewer",
                fontsize=11, color="gray"
            )

    def highlight(self, ch_name):
        """Redraw the topomap with the given channel name highlighted, if it exists."""
        ch_name = ch_name.strip()
        if ch_name not in self.ch_names:
            return
        idx = self.ch_names.index(ch_name)
        print(f"  → Electrode highlighted: {ch_name}  (index {idx})")
        self._draw_base(highlighted_idx=idx)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ── Viewer ────────────────────────────────────────────────────────────────────

def launch_viewer(raw, subj_id, task, session, nclass, model_type, band="Fullband"):
    """Open the MNE signal browser + topomap. Clicking a channel highlights it."""
    info, ch_names = make_info()

    if model_type == "Offline":
        title = f"S{subj_id:02}  |  {task}  |  Offline"
    else:
        model_label = "Base Model" if model_type == "Orig" else "Fine-tuned Model"
        title = (
            f"S{subj_id:02}  |  {task} {nclass}-class  |  "
            f"Session {session}  |  {model_label}"
        )
    title += f"  |  {band}"

    print(f"\nOpening MNE viewer — {title}")
    print("Controls:")
    print("  ← / →            scroll in time")
    print("  scroll wheel     scroll in time")
    print("  + / -            scale amplitude")
    print("  Home / End       jump to start / end")
    print("  Page Up/Down     jump ±10s")
    print("  click channel    highlight on topomap")
    print("  close window     exit\n")

    event_color = {
        EVENT_ID["Thumb"]:    "green",
        EVENT_ID["Index"]:    "blue",
        EVENT_ID["Middle"]:   "orange",
        EVENT_ID["Pinky"]:    "red",
        EVENT_ID["TrialEnd"]: "gray",
    }

    topo    = TopoHighlighter(info, ch_names)
    fig_raw = raw.plot(
        n_channels=20,
        duration=10.0,
        scalings=dict(eeg=50e-6),
        title=title,
        show_scrollbars=True,
        show_options=True,
        block=False,
        overview_mode="channels",
        color=dict(eeg="steelblue"),
        event_color=event_color,
    )

    def on_pick(event):
        artist = event.artist
        if hasattr(artist, "get_text"):
            text = artist.get_text().strip()
            if text in ch_names:
                topo.highlight(text)

    def on_click(event):
        # Fallback hit-test: MNE's channel-name labels aren't always pickable
        # artists, so scan text artists in the clicked axes and match by
        # vertical position (channel row) when the click lands left of x=0
        # (i.e. in the channel-name margin rather than the signal trace).
        if event.inaxes is None:
            return
        for artist in event.inaxes.get_children():
            if hasattr(artist, "get_text") and hasattr(artist, "get_position"):
                try:
                    tx, ty = artist.get_position()
                    if abs(event.ydata - ty) < 0.5 and event.xdata < 0:
                        text = artist.get_text().strip()
                        if text in ch_names:
                            topo.highlight(text)
                            break
                except Exception:
                    pass

    fig_raw.canvas.mpl_connect("pick_event", on_pick)
    fig_raw.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)


# ── Entry points ──────────────────────────────────────────────────────────────

def view_online(subj, sess, ncl, task, model, band="Fullband"):
    """Load one online-session recording and launch the interactive viewer for it."""
    print("=" * 55)
    print("   STING EEG Signal Viewer")
    print("=" * 55)
    print(f"\nSelected: S{subj:02} | {task} {ncl}-class | Session {sess} | "
          f"{'Base' if model == 'Orig' else 'Fine-tuned'} | {band}")
    raw = load_online_raw(subj, task, sess, ncl, model, band)
    launch_viewer(raw, subj, task, sess, ncl, model, band)


def view_offline(subj_id, task, band="Fullband"):
    """Load one offline recording and launch the interactive viewer for it."""
    print("=" * 55)
    print("   STING EEG Signal Viewer - OFFLINE")
    print("=" * 55)
    print(f"\nSelected: S{subj_id:02} | {task} | Offline | {band}")
    raw = load_offline_raw(subj_id, task, band)
    launch_viewer(raw, subj_id, task, 0, 0, "Offline", band)


def main():
    """CLI entry point: parses arguments (online vs offline mode, optional band) and launches the viewer."""
    try:
        # Optional last argument selects the band-pass (default Fullband).
        band = "Fullband"
        if len(sys.argv) > 1:
            band_lookup = {b.lower(): b for b in BANDS}
            cand = sys.argv[-1].strip().lower()
            if cand in band_lookup:
                band = band_lookup[cand]
                sys.argv.pop()  # strip band so the standard parsers don't see it

        if len(sys.argv) > 1 and sys.argv[-1].upper() == "OFF":
            sys.argv.pop()  # strip "OFF" so argparse sees only <subj> <task>
            subj_id, task = get_offline_args(description="EEG Viewer - Offline")
            view_offline(subj_id, task, band)
        else:
            subj, sess, ncl, task, model = get_standard_args(description="EEG Viewer - Online")
            view_online(subj, sess, ncl, task, model, band)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

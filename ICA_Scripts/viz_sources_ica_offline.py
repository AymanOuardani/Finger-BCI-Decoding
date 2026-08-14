"""
viz_sources_ica_offline.py

Same interactive two-window ICA inspector as viz_sources.py, but it ALWAYS uses
the subject's OFFLINE ICA to generate the sources — for any session, online or
offline. The offline mixing matrix is loaded from cache (or fitted once from the
offline recording if absent) and applied to the chosen recording, exactly like
the offline-ICA reuse used for the correlation maps.

  Sources window : MNE ica.plot_sources() — scrollable native browser.
  Signal window  : MNE raw.plot()         — same native browser, updated in-place.

Left-click  a source : toggle exclusion (line turns gray, signal updates instantly).
Right-click a source : open ica.plot_properties() for that component.

Usage
-----
  Online:  python viz_sources_ica_offline.py <subj> <sess> <nclass> <task> <model>
  Offline: python viz_sources_ica_offline.py <subj> <task> OFF
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
    get_ica_folder,
    save_ica_cleaned_mat_files,
    fit_or_load_ica,
)


def load_offline_ica(subj, task):
    """Return the subject's OFFLINE ICA, reused for any session.

    Loads the cached offline ICA directly (preserving any saved exclusions);
    if it does not exist yet, fits it once from the offline recording (same
    preprocessing as the offline run: 60 Hz notch + 1 Hz highpass) and caches it.
    """
    ica_path = get_ica_fif_path(subj, task)        # offline filename form
    if os.path.exists(ica_path):
        print(f"  Reusing cached OFFLINE ICA: {ica_path}")
        return mne.preprocessing.read_ica(ica_path)

    print("  Offline ICA not cached — fitting it once from the offline recording ...")
    off_folder = get_offline_folder(subj, task)
    mat_files  = sorted(glob.glob(os.path.join(off_folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No offline .mat files in {off_folder} to fit ICA.")
    off_raw, _, _ = build_raw_from_mat_files(mat_files)
    off_raw.notch_filter(np.arange(60, 501, 60), verbose=False)
    off_filt = off_raw.copy().filter(1.0, None, verbose=False)
    return fit_or_load_ica(off_filt, ica_path)

# Force the matplotlib browser backend (not mne-qt-browser): the in-place
# redraw trick below relies on the matplotlib MNEBrowseFigure._redraw API.
try:
    mne.viz.set_browser_backend('matplotlib')
except Exception as _e:
    print(f"  [info] could not set browser backend: {_e}")

# ── tuning constants ──────────────────────────────────────────────────────────
_PREVIEW_SECS   = 10.0   # seconds shown in signal window — short = fast ICA apply
_N_SIG_CHANNELS = 20     # channels visible in MNE's signal browser
_SPECTRUM_FMAX  = 60.0  # upper edge of the dedicated spectrum window (Hz)
# ─────────────────────────────────────────────────────────────────────────────


def _load_raw(folder):
    """Return (filtered_raw, mat_files) for the given source folder."""
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files in {folder}")
    raw, _, _ = build_raw_from_mat_files(mat_files)
    raw_filt = raw.filter(1.0, None, verbose=False)
    return raw_filt, mat_files


def _is_trace_axes(ax):
    """True if `ax` is the ICA component-trace axis (labels start with 'ICA')."""
    if ax is None:
        return False
    labels = [t.get_text() for t in ax.get_yticklabels()]
    return any(lbl.startswith('ICA') or lbl.startswith('ica') for lbl in labels)


_LABEL_MARGIN_PX = 100   # pixels left of the trace axis where labels live


def _component_from_ydata(ax, y_data, n_comp):
    """Map a data-coordinate y value on `ax` to an ICA component index."""
    ytick_pos    = np.asarray(ax.get_yticks(), dtype=float)
    ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
    if ytick_pos.size == 0 or not any(ytick_labels):
        return None
    n_ticks = min(len(ytick_pos), len(ytick_labels))
    if n_ticks == 0:
        return None
    closest = int(np.argmin(np.abs(ytick_pos[:n_ticks] - y_data)))
    label   = ytick_labels[closest]
    if not (label.startswith('ICA') or label.startswith('ica')):
        return None
    m = re.search(r'\d+', label)
    if m:
        idx = int(m.group())
        if 0 <= idx < n_comp:
            return idx
    return None


def _click_to_component(event, trace_ax, n_comp):
    """
    Resolve a mouse click to an ICA component index.

    Accepts clicks anywhere on the trace axis (on the signal itself) **and**
    clicks in the y-tick label region just to the left of the axis, so that
    clicking the component name (e.g. "ICA002") also counts as selecting it.

    Returns None if the click is somewhere else (scrollbar, slider, toolbar…).
    """
    if trace_ax is None:
        return None

    # Case 1: click on the signal/trace area itself
    if event.inaxes is trace_ax and event.ydata is not None:
        return _component_from_ydata(trace_ax, event.ydata, n_comp)

    # Case 2: click in the y-tick label margin to the LEFT of the trace axis
    # (event.inaxes is usually None there — labels live outside the data area).
    if event.x is None or event.y is None:
        return None
    try:
        ax_bbox = trace_ax.get_window_extent()
    except Exception:
        return None

    # Must be vertically within the axis bounds and just to its left
    in_y_band  = ax_bbox.y0 <= event.y <= ax_bbox.y1
    in_label_x = (ax_bbox.x0 - _LABEL_MARGIN_PX) <= event.x < ax_bbox.x0
    if not (in_y_band and in_label_x):
        return None

    # Convert pixel y → data y in the trace axis, then map to a component
    try:
        _, y_data = trace_ax.transData.inverted().transform((event.x, event.y))
    except Exception:
        return None
    return _component_from_ydata(trace_ax, y_data, n_comp)


def _mne_force_redraw(fig):
    """
    Force MNE's matplotlib browser to re-read its `inst` and repaint.

    `fig._redraw(update_data=True)` is private API but stable across MNE 1.x.
    The fallbacks cover older signatures. We always finish with a synchronous
    `canvas.draw()` so the paint is visible immediately, not on the next
    event-loop idle tick.
    """
    used = None
    try:
        fig._redraw(update_data=True)
        used = '_redraw(update_data=True)'
    except (AttributeError, TypeError):
        try:
            if hasattr(fig, '_update_data'):
                fig._update_data()
                used = '_update_data + _draw_traces'
            if hasattr(fig, '_draw_traces'):
                fig._draw_traces()
        except Exception as e:
            print(f"  [warn] MNE redraw fallback failed: {e}")

    # Force a synchronous paint — this is what makes the update *visible* now,
    # not after the click callback returns.
    try:
        fig.canvas.draw()
    except Exception:
        fig.canvas.draw_idle()
    fig.canvas.flush_events()
    return used


class ICAInspector:
    """Two-window ICA inspector. Both windows are MNE native browsers."""

    def __init__(self, ica: mne.preprocessing.ICA, raw: mne.io.BaseRaw,
                 mat_files=None, save_folder=None):
        self.ica         = ica
        self.raw         = raw
        self.n_comp      = ica.n_components_
        self.mat_files   = mat_files
        self.save_folder = save_folder
        self._saved      = False

        # Preview slice: short enough that ica.apply() runs in a few tens of ms
        t_end = min(raw.times[-1], _PREVIEW_SECS)
        self.raw_preview = raw.copy().crop(tmax=t_end)

        # raw_clean: shown in the signal browser. We update its _data in place
        # on each toggle and ask MNE to re-read from this same object.
        self.raw_clean = self.raw_preview.copy()
        self.ica.apply(self.raw_clean, verbose=False)

        # ── 1. MNE sources browser ──────────────────────────────────────────
        self.src_fig = ica.plot_sources(raw, show=False, block=False)
        self.src_fig.canvas.manager.set_window_title(
            'ICA Sources — left-click: exclude/include  |  right-click: properties'
        )

        # Locate the component-trace axis once — every click handler must
        # verify that the click happened on THIS axis (and not on scrollbar
        # or time-slider axes, which would otherwise leak through).
        self._trace_ax = None
        for ax in self.src_fig.axes:
            if _is_trace_axes(ax):
                self._trace_ax = ax
                break
        if self._trace_ax is None:
            print("  [warn] could not locate ICA trace axis — clicks may misfire.")

        # ── 2. MNE signal browser (cleaned EEG) ─────────────────────────────
        n_ch = min(_N_SIG_CHANNELS, len(raw.ch_names))
        self.sig_fig = self.raw_clean.plot(
            n_channels = n_ch,
            duration   = _PREVIEW_SECS,
            start      = 0.0,
            scalings   = 'auto',
            show       = False,
            block      = False,
            title      = 'EEG — ICA Cleaned',
        )
        self._update_sig_title()

        # Track exclusion state so we can detect whether MNE's own click
        # handler already toggled the component (newer MNE versions do this
        # natively in plot_sources). Without this check we would double-toggle.
        self._last_seen_excl = sorted(self.ica.exclude)

        # ── 3. Wire events on the sources figure ────────────────────────────
        self.src_fig.canvas.mpl_connect('button_press_event', self._on_press)

        # Save the *_ICA.mat files when either window is closed.
        self.src_fig.canvas.mpl_connect('close_event', self._on_close)
        self.sig_fig.canvas.mpl_connect('close_event', self._on_close)

        plt.show(block=True)

    # ── save on close ─────────────────────────────────────────────────────────

    def _on_close(self, event):
        """Save cleaned .mat files the first time any window is closed."""
        if self._saved:
            return
        if not self.mat_files or not self.save_folder:
            return
        self._saved = True
        print(f"\n  [CONFIRM] Saving {len(self.ica.exclude)} component(s) removed "
              f"({self.ica.exclude}) to:\n    {self.save_folder}")
        try:
            save_ica_cleaned_mat_files(
                self.ica, self.raw, self.mat_files, self.save_folder
            )
            print("  Done.")
        except Exception as e:
            print(f"  [ERROR] Save failed: {e}")

    # ── signal browser title ──────────────────────────────────────────────────

    def _update_sig_title(self):
        n_exc = len(self.ica.exclude)
        label = (': ' + ', '.join(f'ICA{i:03d}' for i in sorted(self.ica.exclude))
                 if n_exc else '')
        title = f'EEG — {n_exc} component(s) removed{label}'
        try:
            self.sig_fig.canvas.manager.set_window_title(title)
        except Exception:
            pass

    # ── click handler ─────────────────────────────────────────────────────────

    def _on_press(self, event):
        # Resolve the click to a component — accepts clicks on the trace AND
        # on the y-tick label region (the component name to the left).
        # Returns None for scrollbar / time-slider / toolbar clicks.
        comp = _click_to_component(event, self._trace_ax, self.n_comp)
        if comp is None:
            return

        if event.button == 3:
            # Right-click → open MNE properties panel for this component
            print(f'  Opening properties for ICA{comp:03d} …')
            self._open_properties(comp)
            return

        if event.button == 1:
            # Left-click → MNE may toggle natively. Defer our check briefly so
            # MNE's own handler runs first, then reconcile and refresh.
            timer = self.src_fig.canvas.new_timer(interval=80)
            timer.add_callback(self._after_left_click, comp, timer)
            timer.start()

    def _open_properties(self, comp):
        """
        Right-click on a source → open the dedicated spectrum window only.

        MNE's `plot_sources` figure has its own native right-click handler
        that already opens `ica.plot_properties()` for the clicked
        component. Calling it from here too would pop a second copy of
        that same window (the "duplicate original" the user saw). So we
        leave the properties view to MNE and add only the big spectrum
        plot that needs more vertical space than plot_properties gives.
        """
        try:
            self._open_spectrum_window(comp)
        except Exception as e:
            print(f"  [warn] spectrum window failed: {e}")

    def _open_spectrum_window(self, comp):
        """Pop a large standalone figure showing component `comp`'s PSD."""
        import scipy.signal as _sp_signal
        sources = self.ica.get_sources(self.raw)
        sfreq   = float(sources.info['sfreq'])
        src     = sources.get_data()[comp]                 # (n_times,)

        # Welch PSD — 2-second windows for clean low-frequency resolution
        nperseg     = int(min(sfreq * 2.0, len(src)))
        freqs, psd  = _sp_signal.welch(src, fs=sfreq, nperseg=nperseg)
        mask        = freqs <= _SPECTRUM_FMAX
        freqs, psd  = freqs[mask], psd[mask]

        fig, ax = plt.subplots(figsize=(13, 6))
        try:
            fig.canvas.manager.set_window_title(
                f'ICA{comp:03d} — Power Spectrum'
            )
        except Exception:
            pass

        # Raw power on a log-scaled y-axis: spreads the ~4-orders-of-magnitude
        # variation of EEG across the figure so alpha/beta peaks and the 1/f
        # background are both clearly readable.
        ax.plot(freqs, psd, color='#2E5984', linewidth=1.5)
        ax.set_yscale('log')
        ax.set_xlabel('Frequency (Hz)', fontsize=12)
        ax.set_ylabel('Power (V²/Hz)',  fontsize=12)
        ax.set_title(f'ICA{comp:03d} — Power Spectrum',
                     fontsize=13, fontweight='bold')
        ax.set_xlim(0, _SPECTRUM_FMAX)
        ax.grid(True, which='both', alpha=0.35, linestyle='--')
        ax.tick_params(labelsize=11)
        fig.tight_layout()

        # Force the new Tk window to actually paint inside the running event loop
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)
        except Exception:
            pass

    def _after_left_click(self, comp, timer):
        # Stop the (otherwise repeating) timer
        timer.stop()

        # If MNE didn't toggle anything (older MNE versions without native
        # plot_sources click handling), toggle ourselves.
        if sorted(self.ica.exclude) == self._last_seen_excl and comp is not None:
            excl = set(self.ica.exclude)
            if comp in excl:
                excl.discard(comp)
            else:
                excl.add(comp)
            self.ica.exclude = sorted(excl)

        new_excl = sorted(self.ica.exclude)
        if new_excl == self._last_seen_excl:
            return   # nothing actually changed (e.g. click missed)

        self._last_seen_excl = new_excl
        print(f"  → exclusion: {new_excl}  ({len(new_excl)} removed)")

        # MNE handles its own visual marking (red/gray on excluded components)
        # so we don't need to repaint the sources figure ourselves.
        self._refresh_signal()

    # ── signal browser refresh ────────────────────────────────────────────────

    def _refresh_signal(self):
        """
        Apply ICA to the preview slice, copy the result into raw_clean._data
        in-place, then ask MNE's browser to re-read from raw_clean and redraw.
        No window is closed/reopened, so the update is instant.
        """
        raw_tmp = self.raw_preview.copy()
        self.ica.apply(raw_tmp, verbose=False)
        new_data = raw_tmp.get_data()

        # In-place data update on raw_clean.  Also defensively poke the
        # browser's internal `inst` reference in case some MNE version stored
        # a separate copy when raw.plot() was first called.
        self.raw_clean._data[:] = new_data
        try:
            inst = self.sig_fig.mne.inst
            if inst is not self.raw_clean and hasattr(inst, '_data'):
                inst._data[:] = new_data
        except AttributeError:
            pass

        self._update_sig_title()
        used = _mne_force_redraw(self.sig_fig)
        print(f"  → signal refreshed (redraw via {used or 'fallback'})")


# ─────────────────────────────────────────────────────────────────────────────

def sources_online(subj, sess, ncl, task, model):
    """Open the interactive inspector for an online recording, always using the subject's OFFLINE ICA."""
    print("=" * 60)
    print(f"  ICA Sources (OFFLINE ICA): S{subj:02} Sess{sess:02} | {task} {ncl}-class | {model}")
    print("=" * 60)
    folder           = get_folder(subj, task, sess, ncl, model)
    raw, mat_files   = _load_raw(folder)
    raw.notch_filter(np.arange(60, 501, 60))
    save_folder      = get_ica_folder(folder)
    # Always use the subject's offline ICA (not a per-session one).
    ica = load_offline_ica(subj, task)
    print(f"  Using OFFLINE ICA ({ica.n_components_} components) for this online session.")
    print("  Left-click: exclude/include.  Right-click: open properties.")
    print(f"  Cleaned files will be saved on window close to:\n    {save_folder}")
    ICAInspector(ica, raw, mat_files=mat_files, save_folder=save_folder)


def sources_offline(subj, task):
    """Open the interactive inspector for an offline recording, using the subject's OFFLINE ICA."""
    print("=" * 60)
    print(f"  ICA Sources: S{subj:02} | {task} (Offline)")
    print("=" * 60)
    folder           = get_offline_folder(subj, task)
    raw, mat_files   = _load_raw(folder)
    raw.notch_filter(np.arange(60, 501, 60))
    save_folder      = get_ica_folder(folder)
    # Offline ICA (load from cache, or fit once from this offline recording).
    ica = load_offline_ica(subj, task)
    print(f"  Using OFFLINE ICA ({ica.n_components_} components).")
    print("  Left-click: exclude/include.  Right-click: open properties.")
    print(f"  Cleaned files will be saved on window close to:\n    {save_folder}")
    ICAInspector(ica, raw, mat_files=mat_files, save_folder=save_folder)


def main():
    """Parse CLI args and dispatch to the online or offline ICA inspector."""
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

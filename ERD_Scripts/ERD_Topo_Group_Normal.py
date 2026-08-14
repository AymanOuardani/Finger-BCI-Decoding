"""
ERD_Topo_Group_Normal.py

Same group-level ERD pipeline as ERD_Topo_Group.py, with one extra step:
each averaged topomap is *normalised* by subtracting its own maximum so
that the most-positive channel becomes 0 and every other channel is ≤ 0.
The colormap is then clamped to the range [-0.5, 0] to emphasise the
relative desynchronisation pattern within each map.

Usage:
    Online:  python ERD_Topo_Group_Normal.py <sess> <nclass> <task> <model> <band>
    Offline: python ERD_Topo_Group_Normal.py <task> <band> OFF
    band: Alpha | Beta | Fullband

    Examples:
        python ERD_Topo_Group_Normal.py 1 2 MI Orig Fullband
        python ERD_Topo_Group_Normal.py MI Beta OFF
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import mne
mne.set_log_level("WARNING")

from ERD_Topo import (
    load_and_preprocess, load_and_preprocess_ica,
    load_and_preprocess_offline, load_and_preprocess_ica_offline,
    build_epochs, build_epochs_no_reject, _compute_erd_band,
    BANDS_CONFIG, FINGERS, FINGERS_OFFLINE,
    TMAX_ONLINE, TMAX_OFFLINE, TASK_WIN_ONLINE, TASK_WIN_OFFLINE,
)
from config import RESULTS_ROOT, CORRUPTED_DATA, GROUP_ERD_SUBJECTS

ALL_SUBJECTS = GROUP_ERD_SUBJECTS

# Normalised display range — fixed across all maps so colours are comparable
VMIN, VMAX = -0.5, 0.0


# ── Normalisation helper ──────────────────────────────────────────────────────

def _normalize_to_zero_max(erd_map):
    """
    Shift `erd_map` so its global maximum sits at 0.

      erd_norm = erd - max(erd)

    The channel that was the most positive (or the least negative) ends up
    at 0; every other channel falls below 0. Returns the shifted array, or
    None if `erd_map` is None.
    """
    if erd_map is None:
        return None
    return erd_map - float(np.max(erd_map))


# ── Plotting (Before vs After ICA, [-0.5, 0] clipped) ─────────────────────────

def plot_normalized_comparison(erd_raw, erd_ica, band_label, finger_names,
                               info, title, save_path=None, show=True,
                               vmin=VMIN, vmax=VMAX):
    """Two-row Before-vs-After-ICA figure with a normalised [-0.5, 0] colormap."""
    n_cols = len(finger_names)

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(2.5 * n_cols + 1.2, 5.6 + 0.6),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.88, right=0.88, hspace=0.1)

    row_labels = [f"Before ICA\n{band_label}", f"After ICA\n{band_label}"]
    row_dicts  = [erd_raw, erd_ica]

    for r, (row_label, erd_dict) in enumerate(zip(row_labels, row_dicts)):
        for c, fname in enumerate(finger_names):
            ax  = axes[r, c]
            erd = erd_dict.get(fname)

            if r == 0:
                ax.set_title(fname, fontsize=10, fontweight="bold", pad=4)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=9, labelpad=6)

            if erd is None:
                ax.axis("off")
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="gray")
                continue

            mne.viz.plot_topomap(
                erd, info,
                axes=ax,
                cmap="Blues_r",          # dark blue at vmin, white at vmax
                vlim=(vmin, vmax),
                contours=0,
                extrapolate="head",
                sphere=(0., 0., 0., 0.095),
                outlines="head",
                show=False,
                sensors=False,
            )

    # Dashed separator between Before / After rows
    fig.add_artist(plt.Line2D(
        [0.07, 0.88], [0.5, 0.5],
        transform=fig.transFigure,
        color="gray", linewidth=0.8, linestyle="--",
    ))

    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.97)

    cbar_ax = fig.add_axes([0.91, 0.15, 0.025, 0.65])
    sm = plt.cm.ScalarMappable(
        cmap="Blues_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar  = plt.colorbar(sm, cax=cbar_ax, label="Normalised ERD  (max → 0)")
    ticks = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {save_path}")

    if show:
        plt.show(block=True)


# ── Group runner ──────────────────────────────────────────────────────────────

def _run_group(band_raw, finger_pairs, load_fn, load_ica_fn,
               title_prefix, save_name,
               tmax=TMAX_ONLINE, task_win=TASK_WIN_ONLINE):
    """Loop over ALL_SUBJECTS, average the per-subject Before/After-ICA ERD
    maps across the group, normalise each averaged map (max -> 0, see
    `_normalize_to_zero_max`), and save the comparison figure."""
    band = next((k for k in BANDS_CONFIG if k.lower() == band_raw.lower()), band_raw)
    if band not in BANDS_CONFIG:
        print(f"[ERROR] band must be one of: {list(BANDS_CONFIG.keys())}")
        sys.exit(1)

    bands        = BANDS_CONFIG[band]
    finger_names = [name for name, _ in finger_pairs]

    raw_collection = {bl: {fn: [] for fn, _ in finger_pairs} for _, _, bl in bands}
    ica_collection = {bl: {fn: [] for fn, _ in finger_pairs} for _, _, bl in bands}
    missing_ica    = []

    n_ok = 0
    for subj in ALL_SUBJECTS:
        print(f"\nS{subj:02}...", end=" ", flush=True)

        try:
            raw    = load_fn(subj)
            # Apply the same 20 µV trial-rejection rule (mean per-channel std)
            # as ERD_Topo_Group.py — both Before and After ICA paths.
            epochs = build_epochs(raw, finger_pairs, tmax=tmax)
        except Exception as e:
            print(f"skipped ({e})")
            continue

        if len(epochs) == 0:
            print("skipped (no epochs left after 20 µV rejection)")
            continue

        subj_ok = False
        for fmin, fmax, band_label in bands:
            erd_raw = _compute_erd_band(epochs, finger_pairs, fmin, fmax, task_win)
            for fname in finger_names:
                if erd_raw[fname] is not None:
                    raw_collection[band_label][fname].append(erd_raw[fname])
                    subj_ok = True

        try:
            raw_ica    = load_ica_fn(subj)
            epochs_ica = build_epochs(raw_ica, finger_pairs, tmax=tmax)
            if len(epochs_ica) > 0:
                for fmin, fmax, band_label in bands:
                    erd_ica = _compute_erd_band(epochs_ica, finger_pairs, fmin, fmax, task_win)
                    for fname in finger_names:
                        if erd_ica[fname] is not None:
                            ica_collection[band_label][fname].append(erd_ica[fname])
        except FileNotFoundError:
            missing_ica.append(subj)
            print(f"  [!] ICA data missing — run clean_ICA.py for S{subj:02} first")

        if subj_ok:
            n_ok += 1
            print("OK")

    print(f"\n{'='*55}")
    print(f"  Subjects included: {n_ok} / {len(ALL_SUBJECTS)}")

    if missing_ica:
        print(f"\n[ACTION REQUIRED] ICA data missing for {len(missing_ica)} subject(s):")
        for s in missing_ica:
            print(f"  S{s:02}")
        print("\nRun clean_ICA.py for each subject listed above, then re-run this script:")
        print("  Online:  python ICA_Scripts/clean_ICA.py <subj> <sess> <nclass> <task> <model>")
        print("  Offline: python ICA_Scripts/clean_ICA.py <subj> <task> OFF")
        sys.exit(1)

    if n_ok == 0:
        print("[ERROR] No valid subjects — check paths and arguments.")
        sys.exit(1)

    # Get channel info from first valid subject (rejection-free lookup,
    # info doesn't depend on rejection so this is just for robustness).
    info_subj = None
    for subj in ALL_SUBJECTS:
        try:
            raw = load_fn(subj)
            ep  = build_epochs_no_reject(raw, finger_pairs, tmax=tmax)
            if len(ep) > 0:
                info_subj = ep.info
                break
        except Exception:
            continue

    if info_subj is None:
        print("[ERROR] Could not obtain channel info.")
        sys.exit(1)

    for fmin, fmax, band_label in bands:
        erd_raw_norm = {}
        erd_ica_norm = {}
        for fname in finger_names:
            maps_raw = raw_collection[band_label][fname]
            maps_ica = ica_collection[band_label][fname]

            raw_avg = np.mean(maps_raw, axis=0) if maps_raw else None
            ica_avg = np.mean(maps_ica, axis=0) if maps_ica else None

            # ────────────── normalisation step (the whole point) ──────────────
            # Subtract the per-map maximum so the most-positive channel sits
            # at 0; every other channel ends up ≤ 0. We do this AFTER the
            # group average so each (band × finger × Before/After) map has
            # its own reference, and the colormap then clamps to [-0.5, 0].
            # ────────────────────────────────────────────────────────────────
            erd_raw_norm[fname] = _normalize_to_zero_max(raw_avg)
            erd_ica_norm[fname] = _normalize_to_zero_max(ica_avg)

            if raw_avg is not None:
                print(f"  {band_label} | {fname:6s} Before: "
                      f"raw_max={raw_avg.max():+.3f} -> 0, "
                      f"norm_min={erd_raw_norm[fname].min():+.3f}  "
                      f"({len(maps_raw)} subj)")
            if ica_avg is not None:
                print(f"  {band_label} | {fname:6s} After : "
                      f"raw_max={ica_avg.max():+.3f} -> 0, "
                      f"norm_min={erd_ica_norm[fname].min():+.3f}  "
                      f"({len(maps_ica)} subj)")

        title     = (f"{title_prefix}  |  {band_label}  (n={n_ok})"
                     f"  —  normalised, max=0, range [{VMIN}, {VMAX}]")
        save_path = os.path.join(RESULTS_ROOT, "Group", save_name)

        plot_normalized_comparison(
            erd_raw_norm, erd_ica_norm, band_label, finger_names,
            info_subj, title, save_path,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point: parses sys.argv to dispatch between online and
    offline group modes and drives `_run_group`."""
    try:
        if len(sys.argv) > 1 and sys.argv[-1].upper() == "OFF":
            sys.argv.pop()
            if len(sys.argv) < 3:
                print("Usage: python ERD_Topo_Group_Normal.py <task> <band> OFF")
                sys.exit(1)
            band_raw = sys.argv.pop()
            task     = sys.argv[1]

            print("=" * 55)
            print(f"   Group ERD (NORMALISED)  (n={len(ALL_SUBJECTS)} subjects)  [Offline]")
            print(f"   {task} | {band_raw}")
            print(f"   per-map max → 0, colormap range [{VMIN}, {VMAX}]")
            print("=" * 55)

            _run_group(
                band_raw     = band_raw,
                finger_pairs = FINGERS_OFFLINE,
                load_fn      = lambda subj: load_and_preprocess_offline(subj, task),
                load_ica_fn  = lambda subj: load_and_preprocess_ica_offline(subj, task),
                title_prefix = f"Group ERD  |  {task}  |  Offline",
                save_name    = f"ERD_GroupNorm_{task}_Offline_{band_raw}_vs_ICA.png",
                tmax         = TMAX_OFFLINE,
                task_win     = TASK_WIN_OFFLINE,
            )

        else:
            if len(sys.argv) < 6:
                print("Usage: python ERD_Topo_Group_Normal.py <sess> <nclass> <task> <model> <band>")
                print("  band: Alpha | Beta | Fullband")
                sys.exit(1)

            try:
                sess     = int(sys.argv[1])
                ncl      = int(sys.argv[2])
                task     = sys.argv[3]
                model    = sys.argv[4]
                band_raw = sys.argv[5]
            except (ValueError, IndexError):
                print("[ERROR] Invalid arguments.")
                sys.exit(1)

            if ncl not in FINGERS:
                print("[ERROR] nclass must be 2 or 3")
                sys.exit(1)

            model_label = "Base" if model == "Orig" else "Fine-tuned"

            print("=" * 55)
            print(f"   Group ERD (NORMALISED)  (n={len(ALL_SUBJECTS)} subjects)")
            print(f"   Sess{sess:02} | {task} {ncl}-class | {model} | {band_raw}")
            print(f"   per-map max → 0, colormap range [{VMIN}, {VMAX}]")
            print("=" * 55)

            def online_load(subj):
                if (subj, task, sess, ncl, model) in CORRUPTED_DATA:
                    raise RuntimeError("corrupted")
                return load_and_preprocess(subj, sess, ncl, task, model)

            _run_group(
                band_raw     = band_raw,
                finger_pairs = FINGERS[ncl],
                load_fn      = online_load,
                load_ica_fn  = lambda subj: load_and_preprocess_ica(subj, sess, ncl, task, model),
                title_prefix = (f"Group ERD  |  Sess{sess:02}  |  "
                                f"{task} {ncl}-class  |  {model_label}"),
                save_name    = (f"ERD_GroupNorm_Sess{sess:02}_{task}_{ncl}class_"
                                f"{model}_{band_raw}_vs_ICA.png"),
            )

    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

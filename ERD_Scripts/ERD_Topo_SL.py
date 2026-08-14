"""
ERD_Topo_SL.py

Version "Sans Limite" de ERD_Topo.py :
même pipeline, mêmes arguments, mais l'échelle de couleur est calculée
dynamiquement depuis les données (pas de limite fixe à [-0.5, 0]).
Colormap divergente (RdBu_r) centrée en 0 :
  Bleu  = désynchronisation (ERD, valeurs négatives)
  Rouge = synchronisation   (ERS, valeurs positives)

Usage:
    Online:  python ERD_Topo_SL.py <subj> <sess> <nclass> <task> <model> <band>
    Offline: python ERD_Topo_SL.py <subj> <task> <band> OFF
    band: Alpha | Beta | Fullband

    Examples:
        python ERD_Topo_SL.py 9 1 2 MI Orig Fullband
        python ERD_Topo_SL.py 9 MI Fullband OFF
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
from Functions import get_standard_args, get_offline_args
from config import RESULTS_ROOT


# ── Plot (dynamic scale) ───────────────────────────────────────────────────────

def plot_erd_comparison_sl(erd_raw, erd_ica, band_label, finger_names, info,
                           title, save_path=None):
    """
    Two-row comparison figure with dynamic color scale (no fixed limits).
    erd_raw, erd_ica : {finger_name: (n_chan,) array or None}
    """
    n_cols = len(finger_names)

    fig, axes = plt.subplots(
        2, n_cols,
        figsize=(2.5 * n_cols + 1.2, 5.6 + 0.6),
        squeeze=False,
    )
    fig.subplots_adjust(top=0.88, right=0.88, hspace=0.1)

    # Fixed colormap range [-0.5, 0]: 0 → white, -0.5 → dark blue.
    vmin, vmax = -0.5, 0.0

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

    # Dashed separator between the two rows
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
    cbar = plt.colorbar(sm, cax=cbar_ax, label="ERD (fraction)")
    ticks = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {save_path}")

    plt.show(block=True)


# ── Shared computation runner ──────────────────────────────────────────────────

def _run(band_raw, finger_pairs, load_fn, load_ica_fn, title_prefix, save_name,
         tmax=TMAX_ONLINE, task_win=TASK_WIN_ONLINE):
    """Load Before/After-ICA data for one subject, compute per-band ERD maps,
    and render/save the Sans Limite comparison figure via `plot_erd_comparison_sl`."""
    band = next((k for k in BANDS_CONFIG if k.lower() == band_raw.lower()), band_raw)
    if band not in BANDS_CONFIG:
        print(f"[ERROR] band must be one of: {list(BANDS_CONFIG.keys())}")
        sys.exit(1)

    finger_names = [name for name, _ in finger_pairs]

    try:
        raw    = load_fn()
        # Apply the 20 µV noisy-trial rejection to the Before-ICA path too so
        # the Before / After topomaps are computed on comparable trial sets.
        epochs = build_epochs(raw, finger_pairs, tmax=tmax)
        if len(epochs) == 0:
            print("\n[ERROR] All Before-ICA trials rejected (mean channel std > 20 µV).")
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    print(f"\n  Epochs (Before ICA, with rejection): {len(epochs)}")
    for eid, name in {eid: n for n, eid in finger_pairs}.items():
        print(f"    {name}: {(epochs.events[:, 2] == eid).sum()}")

    epochs_ica = None
    try:
        raw_ica    = load_ica_fn()
        epochs_ica = build_epochs(raw_ica, finger_pairs, tmax=tmax)
        if len(epochs_ica) == 0:
            print("\n  [WARN] All ICA epochs empty — After-ICA row will show 'no data'")
            epochs_ica = None
        else:
            print(f"\n  ICA epochs: {len(epochs_ica)}")
    except FileNotFoundError:
        print("\n  [INFO] ICA data not found — After-ICA row will show 'no data'")
        print("         Run ICA_Scripts/clean_ICA.py first.")

    for fmin, fmax, band_label in BANDS_CONFIG[band]:
        print(f"\nComputing {band_label} (Before ICA)...")
        erd_raw = _compute_erd_band(epochs, finger_pairs, fmin, fmax, task_win)

        if epochs_ica is not None:
            print(f"Computing {band_label} (After ICA)...")
            erd_ica = _compute_erd_band(epochs_ica, finger_pairs, fmin, fmax, task_win)
        else:
            erd_ica = {fname: None for fname, _ in finger_pairs}

        title     = f"{title_prefix}  |  {band_label}  [Sans Limite]"
        save_path = os.path.join(RESULTS_ROOT, save_name)

        plot_erd_comparison_sl(erd_raw, erd_ica, band_label, finger_names,
                               epochs.info, title, save_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    """CLI entry point: parses sys.argv to dispatch between online and
    offline single-subject modes and drives `_run`."""
    try:
        if len(sys.argv) > 1 and sys.argv[-1].upper() == "OFF":
            sys.argv.pop()
            if len(sys.argv) < 4:
                print("Usage: python ERD_Topo_SL.py <subj> <task> <band> OFF")
                sys.exit(1)
            band_raw = sys.argv.pop()
            subj_id, task = get_offline_args(description="ERD Topography SL - Offline")

            print("=" * 55)
            print(f"   ERD Topography (Sans Limite)  [Offline]")
            print(f"   S{subj_id:02} | {task} | {band_raw}")
            print("=" * 55)

            _run(
                band_raw     = band_raw,
                finger_pairs = FINGERS_OFFLINE,
                load_fn      = lambda: load_and_preprocess_offline(subj_id, task),
                load_ica_fn  = lambda: load_and_preprocess_ica_offline(subj_id, task),
                title_prefix = f"ERD — S{subj_id:02}  |  {task}  |  Offline",
                save_name    = os.path.join(
                    f"Sujet {subj_id}",
                    f"ERD_SL_S{subj_id:02}_{task}_Offline_{band_raw}_vs_ICA.png",
                ),
                tmax         = TMAX_OFFLINE,
                task_win     = TASK_WIN_OFFLINE,
            )

        else:
            subj, sess, ncl, task, model = get_standard_args(
                description="ERD Topography SL - Online")

            if ncl not in FINGERS:
                print("[ERROR] nclass must be 2 or 3")
                sys.exit(1)
            if len(sys.argv) < 7:
                print("Usage: python ERD_Topo_SL.py <subj> <sess> <nclass> <task> <model> <band>")
                sys.exit(1)
            band_raw = sys.argv[6]

            model_label = "Base" if model == "Orig" else "Fine-tuned"

            print("=" * 55)
            print(f"   ERD Topography (Sans Limite)  (Before vs After ICA)")
            print(f"   S{subj:02} | Sess{sess:02} | {task} {ncl}-class | {model} | {band_raw}")
            print("=" * 55)

            _run(
                band_raw     = band_raw,
                finger_pairs = FINGERS[ncl],
                load_fn      = lambda: load_and_preprocess(subj, sess, ncl, task, model),
                load_ica_fn  = lambda: load_and_preprocess_ica(subj, sess, ncl, task, model),
                title_prefix = (f"ERD — S{subj:02}  |  Sess{sess:02}  |  "
                                f"{task} {ncl}-class  |  {model_label}"),
                save_name    = os.path.join(
                    f"Sujet {subj}",
                    f"ERD_SL_S{subj:02}_Sess{sess:02}_{task}_{ncl}class_{model}_{band_raw}_vs_ICA.png",
                ),
            )

    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

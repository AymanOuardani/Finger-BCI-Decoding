"""_gen_auc_maps.py — TEMP. Fullband AUC energy<->task maps for S09, all sessions.

For subject 9 (MI), for every (session, nClass, model) recording present, compute
the Fullband per-trial energy of every ICA source (offline ICA reused), then the
AUC = P(E_in > E_out) of each source vs each task's one-vs-rest membership, and
save a tasks x sources heatmap to C:/Users/aymen/Desktop/S09_Maps.

AUC in [0,1]; 0.5 = chance (white); red = more energy during the task, blue = less.
EOG/EMG artifact source indices (offline detection) are drawn red on the x-axis.
"""
import os, sys, glob, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import corr_heatmap as ch
import fill_energy_table as fet
from Functions import (build_raw_from_mat_files, get_folder, get_offline_folder,
                       compute_exclusions)
from config import TASK_LABELS, EOG_THRESHOLD, EMG_SLOPE_THRESH

SUBJ = 9
TASK = "MI"
OUT = "C:/Users/aymen/Desktop/S09_Maps"


def auc_matrix(E, classes, tasks):
    """AUC = P(E_in > E_out) per (task, source). E: (n_trials, n_comp).
    Uses the Wilcoxon identity AUC = (R_in - n1(n1+1)/2) / (n1 n0), with
    average ranks (tie-corrected), matching scipy's Mann-Whitney U."""
    n, ncomp = E.shape
    R = np.apply_along_axis(rankdata, 0, E)        # average ranks per source column
    out = np.full((len(tasks), ncomp), 0.5)
    for ti, T in enumerate(tasks):
        inm = classes == T
        n1 = int(inm.sum()); n0 = n - n1
        if n1 == 0 or n0 == 0:
            continue
        U = R[inm].sum(axis=0) - n1 * (n1 + 1) / 2.0
        out[ti] = U / (n1 * n0)
    return out


def plot_auc(mat, tasks, sess, ncl, model, save_path, excluded):
    excluded = set(excluded or [])
    n_tasks, n_comp = mat.shape
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=1.0)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("AUC   P(E_in > E_out)   [0.5 = chance]")
    ax.set_yticks(range(n_tasks)); ax.set_yticklabels(tasks)
    ax.set_xticks(range(n_comp))
    xt = ax.set_xticklabels([str(c) for c in range(n_comp)], fontsize=6, rotation=90)
    for c, lbl in zip(range(n_comp), xt):
        if c in excluded:
            lbl.set_color("red")
    ax.set_xlabel("ICA Source")
    ax.set_ylabel("Task")
    ax.set_title(f"S{SUBJ:02d} | {TASK} | Sess{sess:02d} {ncl}class {model} | "
                 f"Fullband | Energy↔Task AUC")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fullband_energy(sess, ncl, model):
    folder = get_folder(SUBJ, TASK, sess, ncl, model)
    mats = sorted(glob.glob(os.path.join(folder, "*.mat")))
    raw, _, _ = build_raw_from_mat_files(mats)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)
    ica = ch._load_offline_ica(SUBJ, TASK)
    src = ica.get_sources(raw)
    D = src.get_data()
    trials = fet._chronological_trials(raw)
    classes = np.array([c for c, _s, _e in trials])
    ncomp = D.shape[0]
    E = np.empty((len(trials), ncomp))
    for ti, (_c, s, e) in enumerate(trials):
        E[ti] = np.sum(D[:, s:e] ** 2, axis=1)
    del raw, src, D
    gc.collect()
    return E, classes


def offline_artifacts():
    try:
        off = sorted(glob.glob(os.path.join(get_offline_folder(SUBJ, TASK), "*.mat")))
        raw, _, _ = build_raw_from_mat_files(off)
        raw.notch_filter(np.arange(60, 501, 60), verbose=False)
        ica = ch._load_offline_ica(SUBJ, TASK)
        rf = raw.copy().filter(1.0, None, verbose=False)
        _, eog, emg = compute_exclusions(
            ica, rf, muscle_thresh=EMG_SLOPE_THRESH, eog_thresh=EOG_THRESHOLD,
            ch_names=rf.info["ch_names"])
        del raw, rf
        gc.collect()
        return set(eog) | set(emg)
    except Exception as e:
        print(f"  [warn] offline artifact detection failed: {e}")
        return set()


def main():
    groups = fet._detect_groups(SUBJ, TASK)
    print(f"S{SUBJ:02d}: {len(groups)} (session,nClass) group(s) -> {OUT}")
    art = offline_artifacts()
    print(f"Offline artifact sources ({len(art)}): {sorted(art)}")

    n_saved = 0
    for (sess, ncl), models in sorted(groups.items()):
        for model in models:
            print(f"  Sess{sess:02d} {ncl}class {model} ...", flush=True)
            try:
                E, classes = fullband_energy(sess, ncl, model)
            except Exception as e:
                print(f"    [SKIP] {e}", flush=True)
                continue
            tasks = [t for t in TASK_LABELS if t in set(classes)]
            mat = auc_matrix(E, classes, tasks)
            tag = "1_Orig" if model == "Orig" else "2_Finetune"
            save = os.path.join(
                OUT, f"AUC_S{SUBJ:02d}_{TASK}_Sess{sess:02d}_{ncl}class_{tag}_Fullband.png")
            plot_auc(mat, tasks, sess, ncl, model, save, art)
            print(f"    saved: {os.path.basename(save)}", flush=True)
            n_saved += 1
            del E
            gc.collect()
    print(f"\nDone. {n_saved} map(s) saved to {OUT}")


if __name__ == "__main__":
    main()

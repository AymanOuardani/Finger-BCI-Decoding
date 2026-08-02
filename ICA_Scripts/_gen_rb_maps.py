"""_gen_rb_maps.py — TEMP. Fullband RANK-BISERIAL energy<->task maps for S09.

Same as _gen_auc_maps.py but the metric is the signed rank-biserial
r_rb = 2*AUC - 1 in [-1, 1] (0 = chance). Diverging RdBu_r centered at 0,
directly comparable to the old Pearson maps. Reuses the AUC computation.

Output: C:/Users/aymen/Desktop/S09_Maps_RankBiserial
"""
import os, sys, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fill_energy_table as fet
import _gen_auc_maps as g            # reuse fullband_energy / auc_matrix / offline_artifacts
from config import TASK_LABELS

OUT = "C:/Users/aymen/Desktop/S09_Maps_RankBiserial"


def plot_rb(mat, tasks, sess, ncl, model, save_path, excluded):
    excluded = set(excluded or [])
    n_tasks, n_comp = mat.shape
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Rank-biserial  r_rb = 2·AUC − 1   [0 = chance]")
    ax.set_yticks(range(n_tasks)); ax.set_yticklabels(tasks)
    ax.set_xticks(range(n_comp))
    xt = ax.set_xticklabels([str(c) for c in range(n_comp)], fontsize=6, rotation=90)
    for c, lbl in zip(range(n_comp), xt):
        if c in excluded:
            lbl.set_color("red")
    ax.set_xlabel("ICA Source")
    ax.set_ylabel("Task")
    ax.set_title(f"S{g.SUBJ:02d} | {g.TASK} | Sess{sess:02d} {ncl}class {model} | "
                 f"Fullband | Energy↔Task Rank-biserial")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    groups = fet._detect_groups(g.SUBJ, g.TASK)
    print(f"S{g.SUBJ:02d}: {len(groups)} group(s) -> {OUT}")
    art = g.offline_artifacts()
    print(f"Offline artifact sources ({len(art)}): {sorted(art)}")

    n_saved = 0
    for (sess, ncl), models in sorted(groups.items()):
        for model in models:
            print(f"  Sess{sess:02d} {ncl}class {model} ...", flush=True)
            try:
                E, classes = g.fullband_energy(sess, ncl, model)
            except Exception as e:
                print(f"    [SKIP] {e}", flush=True)
                continue
            tasks = [t for t in TASK_LABELS if t in set(classes)]
            mat = 2.0 * g.auc_matrix(E, classes, tasks) - 1.0        # rank-biserial
            tag = "1_Orig" if model == "Orig" else "2_Finetune"
            save = os.path.join(
                OUT, f"RB_S{g.SUBJ:02d}_{g.TASK}_Sess{sess:02d}_{ncl}class_{tag}_Fullband.png")
            plot_rb(mat, tasks, sess, ncl, model, save, art)
            print(f"    saved: {os.path.basename(save)}", flush=True)
            n_saved += 1
            del E
            gc.collect()
    print(f"\nDone. {n_saved} map(s) saved to {OUT}")


if __name__ == "__main__":
    main()

"""_table_one.py — TEMP. Per-figure summary table for ONE distribution image.

Recomputes the per-source per-class 0.5xIQR Pearson r (Fullband, one finger),
splits Artefact vs Non-Artefact for the chosen essai, and prints group stats +
separation metrics (Mann-Whitney AUC on |r|, Cohen's d, p-value).

Usage: python _table_one.py <subj> <sess> <ncl> <model> <finger> <AUTO|MANUEL>
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from scipy.stats import mannwhitneyu
import _gen_auc_maps as g

# manual selections (subject level) — same as the generic generator
KEPT = {
    1: [5,6,7,8,9,17,20,21,24,25,53,54,65,66,69,73,74,80,81,85,87,90,91,96,98,99,100,105,108],
    2: [3,5,7,10,23,48,52,53,64,69,77,83,84,85,86,88,89,90,91,4,8,21,42],
    9: [7,17,20,23,26,30,33,34,44,45,49,50,55,59,63,66,67,69,70,82,83,84,85,86,91,93,97,105,107],
}
EXCLUDED = {
    4: [0,1,2,3,4,5,6,7,8,9,10,11,15,18,21,22,23,24,29,30,31,32,33,34,35,36,37,38,39,40,41,
        42,43,44,45,46,47,48,49,50,51,53,54,55,57,59,60,62,63,64,65,67,68,69,70,71,74,75,77,
        79,80,81,85,86,88,89,90,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,
        109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127],
}


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def iqr_keep_mask(x, k=0.5):
    q1, q3 = np.percentile(x, [25, 75]); iqr = q3 - q1
    return (x >= q1 - k * iqr) & (x <= q3 + k * iqr)


def per_class_keep(e, classes):
    keep = np.zeros(len(e), bool)
    for cl in np.unique(classes):
        idx = np.where(classes == cl)[0]
        keep[idx[iqr_keep_mask(e[idx])]] = True
    return keep


def cleaned_r(E, classes, finger):
    ncomp = E.shape[1]; r = np.empty(ncomp)
    for c in range(ncomp):
        keep = per_class_keep(E[:, c], classes)
        r[c] = pearson(E[keep, c], (classes[keep] == finger).astype(float))
    return r


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    sp = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    return (np.mean(x) - np.mean(y)) / sp if sp > 0 else 0.0


def main():
    subj, sess, ncl = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    model, finger, essai = sys.argv[4], sys.argv[5], sys.argv[6].upper()
    g.SUBJ = subj

    if essai == "AUTO":
        art = g.offline_artifacts()
    elif subj in KEPT:
        art = set(range(128)) - set(KEPT[subj])
    elif subj in EXCLUDED:
        art = set(EXCLUDED[subj])
    else:
        raise SystemExit(f"no manual list for S{subj:02d}")

    E, classes = g.fullband_energy(sess, ncl, model)
    ncomp = E.shape[1]
    r = cleaned_r(E, classes, finger)

    art_idx = [c for c in range(ncomp) if c in art]
    non_idx = [c for c in range(ncomp) if c not in art]
    r_art, r_non = r[art_idx], r[non_idx]
    a_art, a_non = np.abs(r_art), np.abs(r_non)

    # Mann-Whitney on |r|: AUC = P(|r|_art > |r|_non)
    U, p = mannwhitneyu(a_art, a_non, alternative="two-sided")
    auc = U / (len(a_art) * len(a_non))
    d = cohens_d(a_art, a_non)

    print(f"\n=== S{subj:02d} {g.TASK} {model} Sess{sess:02d} {ncl}class Fullband "
          f"{finger} [{essai}] ===")
    print(f"trials={len(classes)}  sources=128\n")
    print(f"{'group':14s} {'n':>4s} {'mean r':>9s} {'med r':>9s} "
          f"{'mean|r|':>9s} {'med|r|':>9s} {'max|r|':>9s}")
    for lbl, rr in [("Artefact", r_art), ("Non-Artefact", r_non)]:
        aa = np.abs(rr)
        print(f"{lbl:14s} {len(rr):>4d} {rr.mean():>+9.4f} {np.median(rr):>+9.4f} "
              f"{aa.mean():>9.4f} {np.median(aa):>9.4f} {aa.max():>9.4f}")
    print(f"\nSeparation (Artefact vs Non-Artefact, on |r|):")
    print(f"  AUC (Mann-Whitney) = {auc:.4f}   [0.5 = no diff, >0.5 = artefacts more correlated]")
    print(f"  Cohen's d          = {d:+.4f}   [0.2 small / 0.5 medium / 0.8 large]")
    print(f"  p-value (MWU)      = {p:.3g}    [{'significant' if p < 0.05 else 'NOT significant'} at 0.05]")


if __name__ == "__main__":
    main()

"""_auc_table.py — TEMP. Compute Artefact-vs-Non-Artefact separation AUC
(Mann-Whitney on |Pearson r|, 0.5xIQR-cleaned, per-class) for all fingers,
reusing the EXACT logic of _gen_pearson_iqr_dist_generic (offline-ICA shared,
MANUEL artefact partition). Prints/writes RES lines:

  RES|S{XX}|{sess}|{model}|{finger}|{auc:.3f}|{p:.4g}|{YES/no}

Usage:
  python _auc_table.py test          # quick: S09 Sess01 Orig only
  python _auc_table.py               # full: S02,S03,S04,S06,S09 x Sess1-5 x {Orig,FT}
"""
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from scipy.stats import mannwhitneyu

import _gen_auc_maps as g
import _gen_pearson_iqr_dist_generic as G

SUBJECTS = [2, 3, 4, 6, 9]
SESSIONS = [1, 2, 3, 4, 5]
MODELS = ["Orig", "Finetune"]
FINGERS = ["Thumb", "Index", "Pinky"]
NCL = 3
OUT = ("C:/Users/aymen/AppData/Local/Temp/claude/"
       "C--Users-aymen-Desktop-Finger-BCI-Decoding/"
       "f6525d50-d38a-4b33-820d-41aab296ee59/scratchpad/auc_other_fingers.txt")


def art_set(subj, ncomp=128):
    cfg = G.SUBJ_CONFIG[subj]
    if "kept" in cfg:
        return set(range(ncomp)) - set(cfg["kept"])
    return set(cfg["excluded"])


def sep_auc(vals_art, vals_non):
    a = np.abs(np.asarray(vals_art, float))
    b = np.abs(np.asarray(vals_non, float))
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    U, p = mannwhitneyu(a, b, alternative="two-sided")
    return float(U / (len(a) * len(b))), float(p)


def run(subjects, fh=None):
    for subj in subjects:
        g.SUBJ = subj
        art = art_set(subj)
        for sess in SESSIONS:
            for model in MODELS:
                try:
                    E, classes = g.fullband_energy(sess, NCL, model)
                except Exception as exc:
                    line = f"SKIP|S{subj:02d}|{sess}|{model}|{exc}"
                    print(line, flush=True)
                    if fh: fh.write(line + "\n"); fh.flush()
                    continue
                ncomp = E.shape[1]
                R = G.cleaned_pearson_all_fingers(E, classes, FINGERS)
                art_idx = [c for c in range(ncomp) if c in art]
                non_idx = [c for c in range(ncomp) if c not in art]
                for finger in FINGERS:
                    r = R[finger]
                    auc, p = sep_auc(r[art_idx], r[non_idx])
                    sig = "YES" if (p == p and p < 0.05) else "no"
                    line = (f"RES|S{subj:02d}|{sess}|{model}|{finger}|"
                            f"{auc:.3f}|{p:.4g}|{sig}")
                    print(line, flush=True)
                    if fh: fh.write(line + "\n"); fh.flush()
                del E
                import gc; gc.collect()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        global SUBJECTS, SESSIONS, MODELS
        SUBJECTS, SESSIONS, MODELS = [9], [1], ["Orig"]
        run(SUBJECTS)
        return
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        run(SUBJECTS, fh)
    print(f"\nDONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

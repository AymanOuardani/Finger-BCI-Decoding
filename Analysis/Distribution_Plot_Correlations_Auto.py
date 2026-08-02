"""
Distribution_Plot_Correlations_Auto.py

Batch version of Distribution_Plot_Correlations.py. For every subject in
SUBJECTS, reads Ressources/Sujet_XX.ods and generates a correlation-distribution
plot (Artefact vs Non-Artefact) for every (session, nClass, model, band)
combination found in the Corr sheet. Saves PNGs under:

    Distribution Correlation Maps/S{XX}/{Band}/
        DistCorr_S{XX}_{task}_Sess{SS}_{n}class_{1_Orig|2_Finetune}_{Band}.png

No interactive display. No command-line arguments.
Edit SUBJECTS and TASK below to target a different set.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Make Distribution_Plot_Correlations importable from the same folder.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Distribution_Plot_Correlations as dpc   # noqa: E402

# ── configuration ─────────────────────────────────────────────────────────────

SUBJECTS = [2]
TASK     = "MI"

# Bands to generate. Edit this list to restrict output.
# Available: "Fullband", "Alpha", "Beta", "Beta+2"
# BANDS = ["Fullband", "Alpha", "Beta", "Beta+2"]
BANDS = ["Fullband"]


# ── per-subject runner ────────────────────────────────────────────────────────

def _run_subject(subj_id, task=TASK):
    print(f"\n{'='*55}\n  S{subj_id:02d}\n{'='*55}")

    try:
        df = dpc.load_corr_sheet(subj_id)
    except FileNotFoundError as exc:
        print(f"  [skip] {exc}")
        return

    # Discover every (session, nClass, model) combination present in the sheet.
    combos = (df[["Session", "nClass", "Model"]]
              .drop_duplicates()
              .sort_values(["Session", "nClass", "Model"]))

    if combos.empty:
        print("  [skip] Corr sheet is empty.")
        return

    for _, row in combos.iterrows():
        sess   = int(row["Session"])
        nclass = int(row["nClass"])
        model  = str(row["Model"])
        print(f"\n  Sess{sess:02d}  {nclass}class  {model}", flush=True)
        try:
            dpc.run(subj_id, sess, nclass,
                    task=task, model=model,
                    show=False, save=True, bands=BANDS)
        except Exception as exc:
            print(f"    [ERROR] {exc}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for subj in SUBJECTS:
        _run_subject(subj, TASK)
    print("\n\nAll done.")

"""
run_all_evaluation.py

Evaluates each subject, writes Online Val Accuracy + Online Performance Accuracy
into the matching per-condition sheet in Tracking.ods, and saves a confusion
matrix PNG for each subject under:
  .../Itération 2 - Résultats/Sujet {X}/S{XX}_Sess{NN}_{TASK}_{N}class_{MODEL}_CM.png

═══════════════════════════════════════════════════════════════
PER-CONDITION SHEET LAYOUT (e.g. MI_Sess02_2Class)

  Row 0 (index 0) : Task label (MI / ME)
  Row 1 (index 1) : Session label
  Row 2 (index 2) : Base Model | Fine Tune
  Row 3 (index 3) : 2Class / 3Class
  Row 4 (index 4) : Val_Accuracy | Online Val_Accuracy | Online Perf Accuracy
  Row 5 (index 5) : S01  →  subj_id + 4
  ...
  Row 25 (index 25): S21

  Base Model : cols 1(Val), 2(Online Val), 3(Online Perf)
  Fine Tune  : cols 4(Val), 5(Online Val), 6(Online Perf)
"""

import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Functions import (
    load_and_filter_data, generate_eval_paths, eval_model,
    write_to_ods,
)
from config import (
    DATA_FOLDER, SAVE_FOLDER, RESULTS_ROOT, TRACKING_ODS,
    MAXTRIAL_S, WINDOWLEN, BLOCK_SIZE, DOWNSRATE, BANDPASS_FILT,
    CLASS_NAMES, SHEET_MODEL_START, ROW_SUBJECT_OFFSET,
)

# ── Configuration — edit these to run a different combo ──────────────────────
SESSION_NUM = 2
NCLASS      = 3
TASK        = "ME"
MODELTYPE   = "Finetune"    # "Orig" = Base Model | "Finetune" = Fine Tune
SUBJECTS    = list(range(3, 22))


def get_row_idx(subj_id):
    return subj_id + ROW_SUBJECT_OFFSET


def get_per_sheet_name(task, session_num, nclass):
    return f"{task}_Sess{session_num:02}_{nclass}Class"


def get_cm_save_path(subj_id):
    subj_folder = os.path.join(RESULTS_ROOT, f"Sujet {subj_id}")
    os.makedirs(subj_folder, exist_ok=True)
    filename = (
        f"S{subj_id:02}_Sess{SESSION_NUM:02}_{TASK}_{NCLASS}class"
        f"_{MODELTYPE}_CM.png"
    )
    return os.path.join(subj_folder, filename)


def save_confusion_matrix(subj_id, label, trial_preds, accuracy_online):
    cm   = confusion_matrix(label, trial_preds, labels=list(range(1, NCLASS + 1)))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES[NCLASS],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(
        f"Confusion Matrix — S{subj_id:02} Sess{SESSION_NUM:02} "
        f"{TASK} {NCLASS}-class {MODELTYPE}\n"
        f"Accuracy: {accuracy_online:.2f}%"
    )
    plt.tight_layout()
    save_path = get_cm_save_path(subj_id)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix saved: {save_path}")


PARAMS = {
    "maxtriallen":   MAXTRIAL_S,
    "windowlen":     WINDOWLEN,
    "block_size":    BLOCK_SIZE,
    "downsrate":     DOWNSRATE,
    "bandpass_filt": BANDPASS_FILT,
    "nclass":        NCLASS,
}


def main():
    per_sheet_name  = get_per_sheet_name(TASK, SESSION_NUM, NCLASS)
    base_col        = SHEET_MODEL_START[MODELTYPE]
    col_online_val  = base_col + 1
    col_online_perf = base_col + 2

    print(f"Evaluation : {TASK}  {NCLASS}-class  {MODELTYPE}  Session {SESSION_NUM}")
    print(f"Per-sheet  : '{per_sheet_name}'")
    print(f"Columns    : Online Val_Acc -> col {col_online_val} | "
          f"Online Perf_Acc -> col {col_online_perf}")
    print(f"Row check  : S01 -> index {get_row_idx(1)} | "
          f"S21 -> index {get_row_idx(21)}")
    print(f"CM output  : {RESULTS_ROOT}/Sujet X/")
    print()

    for subj_id in SUBJECTS:
        print(f"{'='*55}")
        print(f"Subject S{subj_id:02}  (row index {get_row_idx(subj_id)})")

        try:
            data_paths = generate_eval_paths(
                subj_id, TASK, NCLASS, SESSION_NUM,
                model_type=MODELTYPE, data_folder=DATA_FOLDER
            )
            data, label, p = load_and_filter_data(data_paths, dict(PARAMS))

            model_path = os.path.join(
                SAVE_FOLDER,
                f"S{subj_id:02}_Sess{SESSION_NUM:02}_{TASK}_{NCLASS}class"
                f"_{MODELTYPE}.h5"
            )

            accuracy, _, _, trial_preds, accuracy_online = eval_model(
                data, label, model_path, p
            )

            print(f"  Online Val Acc  : {accuracy:.4f}%")
            print(f"  Online Perf Acc : {accuracy_online:.4f}%")

            write_to_ods(
                TRACKING_ODS,
                get_row_idx(subj_id),
                [(col_online_val, accuracy), (col_online_perf, accuracy_online)],
                table_name=per_sheet_name,
            )
            print(f"  -> [{per_sheet_name}] S{subj_id:02} "
                  f"col {col_online_val}={accuracy:.4f}% "
                  f"col {col_online_perf}={accuracy_online:.4f}%")

            save_confusion_matrix(subj_id, label, trial_preds, accuracy_online)

        except Exception as e:
            print(f"  ERROR for S{subj_id:02}: {e}")

    print(f"\n{'='*55}")
    print("Done.")


if __name__ == "__main__":
    main()

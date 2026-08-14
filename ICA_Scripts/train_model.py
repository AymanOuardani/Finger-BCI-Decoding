"""
train_model.py

Trains the EEGNet model using the official cleaned dataset (*_ICA.mat files).

Usage:
    python train_model.py <subj> <sess> <nclass> <task> <model>
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Functions import (
    load_and_filter_data, generate_paths, train_models,
    get_standard_args, to_ica_path, write_to_ods,
)
from config import (
    DATA_FOLDER, SAVE_FOLDER, TRACKING_ODS,
    MAXTRIAL_TRAINING_S, WINDOWLEN, BLOCK_SIZE, DOWNSRATE, BANDPASS_FILT,
    ICA_MODEL_START, ROW_SUBJECT_OFFSET,
)

import numpy as np

def main():
    """CLI entry point: locates the ICA-cleaned training folders for one
    subject/session/task/model, loads and filters the data, trains EEGNet
    via `train_models`, and logs the resulting validation accuracy to the
    tracking ODS spreadsheet."""
    subj_id, session_num, nclass, task, modeltype = get_standard_args(
        description="ICA Training"
    )

    params = {
        'maxtriallen':   MAXTRIAL_TRAINING_S,
        'windowlen':     WINDOWLEN,
        'block_size':    BLOCK_SIZE,
        'downsrate':     DOWNSRATE,
        'bandpass_filt': BANDPASS_FILT,
        'nclass':        nclass,
    }

    data_folder = DATA_FOLDER
    save_folder = SAVE_FOLDER
    os.makedirs(save_folder, exist_ok=True)

    print("=" * 60)
    print("  ICA Training")
    print("=" * 60)
    print(f"  Subject    : S{subj_id:02}")
    print(f"  Session    : {session_num}")
    print(f"  Task       : {task}  {nclass}-class")
    print(f"  Model type : {modeltype}")

    # Build the regular training-folder list, then mirror to the *_ICA siblings
    original_paths = generate_paths(
        subj_id, task, nclass, session_num,
        model_type=modeltype, data_folder=data_folder,
    )

    artifact_paths = [to_ica_path(p) for p in original_paths]

    missing = [p for p in artifact_paths if not os.path.exists(p)]
    if missing:
        print("\n[ERROR] ICA folder(s) not found:")
        for m in missing:
            print(f"  - {m}")
        print("\nRun clean_online.py or clean_offline.py first.")
        sys.exit(1)

    print(f"\n  Using {len(artifact_paths)} ICA training folder(s):")
    for p in artifact_paths:
        print(f"    {p}")

    # Load + filter the ICA data (CAR, bandpass, epoching - same as main_model_training)
    data, label, params = load_and_filter_data(artifact_paths, params)

    save_name = os.path.join(
        save_folder,
        f'S{subj_id:02}_Sess{session_num:02}_{task}_{nclass}class_{modeltype}_ICA.h5'
    )

    if modeltype == 'Finetune':
        params['modelpath'] = save_name.replace('Finetune', 'Orig')

    print(f"\n-- Training -----------------------------------------------------")
    print(f"  Saving to : {save_name}")

    save_name, best_val_acc = train_models(data, label, save_name, params)

    # Write best val_accuracy to the Val_Accuracy column of the per-condition sheet
    per_sheet = f"{task}_Sess{session_num:02}_{nclass}Class"
    col_val   = ICA_MODEL_START[modeltype]          # col 9 (Orig) or col 12 (Finetune)
    row_idx   = subj_id + ROW_SUBJECT_OFFSET

    write_to_ods(
        TRACKING_ODS,
        row_idx,
        [(col_val, best_val_acc)],
        table_name=per_sheet,
    )
    print(f"-> ODS [{per_sheet}] S{subj_id:02} col {col_val} = {best_val_acc:.4f}% (Val_Accuracy ICA)")
    print(f"\nDone. Model saved to: {save_name}")


if __name__ == "__main__":
    main()

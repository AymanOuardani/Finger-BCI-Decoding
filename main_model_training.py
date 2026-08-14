"""
main_model_training.py

Trains the EEGNet model on raw EEG data.
Reproduction of Ding et al. (Nature Communications 2025).

Usage:
    python main_model_training.py <subj> <sess> <nclass> <task> <model>
    python main_model_training.py 1 1 2 ME Orig
"""

# %%

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Functions import load_and_filter_data, generate_paths, train_models, write_to_ods
from config import (
    DATA_FOLDER, SAVE_FOLDER, TRACKING_ODS,
    MAXTRIAL_TRAINING_S, WINDOWLEN, BLOCK_SIZE, DOWNSRATE, BANDPASS_FILT,
    SHEET_MODEL_START, ROW_SUBJECT_OFFSET,
)

import numpy as np

# Read command-line arguments
subj_id = int(sys.argv[1])
session_num = int(sys.argv[2])
nclass = int(sys.argv[3])
task = sys.argv[4]
modeltype = sys.argv[5]

# validate inputs
if nclass not in (2, 3):
    raise ValueError("nclass must be either 2 or 3.")

if task not in ("MI", "ME"):
    raise ValueError("task must be either 'MI' (motor imagery) or 'ME' (motor execution).")

if modeltype not in ("Orig", "Finetune"):
    raise ValueError("modeltype must be either 'Orig' (pre-training) or 'Finetune' (fine-tuning).")

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

data_paths = generate_paths(subj_id, task, nclass, session_num, model_type=modeltype, data_folder=data_folder)
data, label, params = load_and_filter_data(data_paths, params)

save_name = os.path.join(save_folder, f'S{subj_id:02}_Sess{session_num:02}_{task}_{nclass}class_{modeltype}.h5')

if modeltype == 'Finetune':
    # Fine-tuning starts from the already-trained "Orig" model for the same
    # subject/session/task/class-count, so derive its path from the current one.
    params['modelpath'] = save_name.replace('Finetune', 'Orig')

save_name, best_val_acc = train_models(data, label, save_name, params)

# Write best val_accuracy to the Val_Accuracy column of the per-condition sheet
per_sheet = f"{task}_Sess{session_num:02}_{nclass}Class"
col_val   = SHEET_MODEL_START[modeltype]          # col 1 (Orig) or col 4 (Finetune)
# Rows are offset from the raw subject id to skip the sheet's header rows (see config.ROW_SUBJECT_OFFSET).
row_idx   = subj_id + ROW_SUBJECT_OFFSET

write_to_ods(
    TRACKING_ODS,
    row_idx,
    [(col_val, best_val_acc)],
    table_name=per_sheet,
)
print(f"-> ODS [{per_sheet}] S{subj_id:02} col {col_val} = {best_val_acc:.4f}% (Val_Accuracy)")

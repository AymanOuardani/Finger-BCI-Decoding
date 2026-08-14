"""
main_model_evaluation.py

Evaluates a trained EEGNet model on test data and writes Online Val Accuracy
+ Online Performance Accuracy to Tracking.ods plus a confusion matrix PNG.

Usage:
    python main_model_evaluation.py <subj> <sess> <nclass> <task> <model>
    python main_model_evaluation.py 1 1 2 ME Orig
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Functions import load_and_filter_data, generate_eval_paths, eval_model
from config import (
    DATA_FOLDER, SAVE_FOLDER, RESULTS_ROOT,
    MAXTRIAL_S, WINDOWLEN, BLOCK_SIZE, DOWNSRATE, BANDPASS_FILT,
)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

subj_id     = int(sys.argv[1])
session_num = int(sys.argv[2])
nclass      = int(sys.argv[3])
task        = sys.argv[4]
modeltype   = sys.argv[5]

if nclass not in (2, 3):
    raise ValueError("nclass must be either 2 or 3.")
if task not in ("MI", "ME"):
    raise ValueError("task must be either 'MI' or 'ME'.")
if modeltype not in ("Orig", "Finetune"):
    raise ValueError("modeltype must be either 'Orig' or 'Finetune'.")

params = {
    'maxtriallen':   MAXTRIAL_S,
    'windowlen':     WINDOWLEN,
    'block_size':    BLOCK_SIZE,
    'downsrate':     DOWNSRATE,
    'bandpass_filt': BANDPASS_FILT,
    'nclass':        nclass,
}

data_folder = DATA_FOLDER
save_folder = SAVE_FOLDER
cm_root     = RESULTS_ROOT

os.makedirs(save_folder, exist_ok=True)

# Evaluation
data_paths = generate_eval_paths(
    subj_id, task, nclass, session_num,
    model_type=modeltype, data_folder=data_folder
)

data, label, params = load_and_filter_data(data_paths, params)

model_path = os.path.join(
    save_folder,
    f"S{subj_id:02}_Sess{session_num:02}_{task}_{nclass}class_{modeltype}.h5"
)

accuracy, I_eval, probs, trial_preds, accuracy_online = eval_model(
    data, label, model_path, params
)

# Confusion matrix
class_names = {2: ["Thumb", "Pinky"], 3: ["Thumb", "Index", "Pinky"]}[nclass]

# Class labels are 1-indexed (1..nclass) in this dataset, not 0-indexed.
cm   = confusion_matrix(label, trial_preds, labels=list(range(1, nclass + 1)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=True, cmap="Blues")
ax.set_title(
    f"Confusion Matrix — S{subj_id:02} Sess{session_num:02} "
    f"{task} {nclass}-class {modeltype}\n"
    f"Accuracy: {accuracy_online:.2f}%"
)
plt.tight_layout()

# Save to .../Itération 2 - Résultats/Sujet {X}/
subj_folder = os.path.join(cm_root, f"Sujet {subj_id}")
os.makedirs(subj_folder, exist_ok=True)

cm_save_path = os.path.join(
    subj_folder,
    f"S{subj_id:02}_Sess{session_num:02}_{task}_{nclass}class_{modeltype}_CM.png"
)
plt.savefig(cm_save_path, dpi=150)
plt.close(fig)
print(f"Confusion matrix saved to {cm_save_path}")
print(f"RESULT:online_val={accuracy:.6f}:online_perf={accuracy_online:.6f}")
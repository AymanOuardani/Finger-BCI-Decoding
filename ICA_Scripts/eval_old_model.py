"""
eval_old_model.py

Evaluates the ORIGINAL model (trained on raw data) on the official
cleaned signals (*_ICA.mat).

Usage:
    python eval_old_model.py <subj> <sess> <nclass> <task> <model>
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from Functions import (
    load_and_filter_data, generate_eval_paths, eval_model,
    get_standard_args, to_ica_path
)
from config import (
    DATA_FOLDER, SAVE_FOLDER, RESULTS_ROOT,
    MAXTRIAL_S, WINDOWLEN, BLOCK_SIZE, DOWNSRATE, BANDPASS_FILT,
)


def main():
    subj_id, session_num, nclass, task, modeltype = get_standard_args(
        description="Evaluate Base model on ICA signals"
    )

    params = {
        'maxtriallen':   MAXTRIAL_S,
        'windowlen':     WINDOWLEN,
        'block_size':    BLOCK_SIZE,
        'downsrate':     DOWNSRATE,
        'bandpass_filt': BANDPASS_FILT,
        'nclass':        nclass,
    }

    print("=" * 60)
    print("  ICA EVALUATION (Robustness check with Base Model)")
    print("=" * 60)
    print(f"  Subject    : S{subj_id:02}")
    print(f"  Session    : {session_num}")
    print(f"  Task       : {task}  {nclass}-class")
    print(f"  Model type : {modeltype}")

    # 1. Paths
    orig_eval_paths = generate_eval_paths(subj_id, task, nclass, session_num, modeltype, DATA_FOLDER)
    artifact_eval_paths = [to_ica_path(p) for p in orig_eval_paths]

    missing = [p for p in artifact_eval_paths if not os.path.exists(p)]
    if missing:
        print("\n[ERROR] ICA evaluation folder(s) not found:")
        for m in missing: print(f"  - {m}")
        sys.exit(1)

    # 2. Load
    data, label, params = load_and_filter_data(artifact_eval_paths, params)

    # 3. Model (Use the ORIGINAL model, not the ICA-trained one)
    model_path = os.path.join(
        SAVE_FOLDER,
        f"S{subj_id:02}_Sess{session_num:02}_{task}_{nclass}class_{modeltype}.h5"
    )
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model not found: {model_path}")
        sys.exit(1)

    # 4. Eval
    accuracy, I_eval, probs, trial_preds, acc_online = eval_model(data, label, model_path, params)

    # 5. Plot
    class_names = {2: ["Thumb", "Pinky"], 3: ["Thumb", "Index", "Pinky"]}[nclass]
    cm = confusion_matrix(label, trial_preds, labels=list(range(1, nclass + 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"ICA Signal (Base Model) S{subj_id:02} S{session_num:02}\nAcc: {acc_online:.2f}%")
    plt.tight_layout()

    # 6. Save
    subj_folder = os.path.join(RESULTS_ROOT, f"Sujet {subj_id}")
    os.makedirs(subj_folder, exist_ok=True)
    save_path = os.path.join(subj_folder, f"S{subj_id:02}_Sess{session_num:02}_{task}_{nclass}class_{modeltype}_ICA_Signal_CM.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"\nEvaluation Complete. Accuracy: {acc_online:.2f}%")
    print(f"CM saved to: {save_path}")


if __name__ == "__main__":
    main()

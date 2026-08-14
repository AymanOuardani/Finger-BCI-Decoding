"""
eval_ica_model.py

Evaluates a model trained on cleaned data (via train_model.py)
using the official cleaned dataset (*_ICA.mat).

Usage:
    python eval_ica_model.py <subj> <sess> <nclass> <task> <model>
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Functions import (
    load_and_filter_data, generate_eval_paths, eval_model,
    get_standard_args, to_ica_path
)
from config import (
    DATA_FOLDER, SAVE_FOLDER, RESULTS_ROOT,
    MAXTRIAL_S, WINDOWLEN, BLOCK_SIZE, DOWNSRATE, BANDPASS_FILT,
)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    """CLI entry point: locates the ICA-cleaned evaluation folders and the
    corresponding trained model (from train_model.py), runs evaluation,
    builds/saves a confusion matrix figure, and prints the resulting
    online/offline accuracy metrics."""
    subj_id, session_num, nclass, task, modeltype = get_standard_args(
        description="ICA Evaluation"
    )

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

    print("=" * 60)
    print("  ICA Evaluation")
    print("=" * 60)
    print(f"  Subject    : S{subj_id:02}")
    print(f"  Session    : {session_num}")
    print(f"  Task       : {task}  {nclass}-class")
    print(f"  Model type : {modeltype}")

    # 1. Generate regular evaluation paths and mirror them to ICA versions
    original_eval_paths = generate_eval_paths(
        subj_id, task, nclass, session_num,
        model_type=modeltype, data_folder=data_folder
    )

    artifact_eval_paths = [to_ica_path(p) for p in original_eval_paths]

    # Check if ICA folders exist
    missing = [p for p in artifact_eval_paths if not os.path.exists(p)]
    if missing:
        print("\n[ERROR] ICA evaluation folder(s) not found:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    # 2. Load + filter the ICA data
    data, label, params = load_and_filter_data(artifact_eval_paths, params)

    # 3. Path to the model trained via train_model.py
    model_path = os.path.join(
        save_folder,
        f"S{subj_id:02}_Sess{session_num:02}_{task}_{nclass}class_{modeltype}_ICA.h5"
    )

    if not os.path.exists(model_path):
        print(f"\n[ERROR] ICA model not found:\n  {model_path}")
        sys.exit(1)

    # 4. Evaluate the model
    accuracy, I_eval, probs, trial_preds, accuracy_online = eval_model(
        data, label, model_path, params
    )

    # 5. Confusion matrix generation
    class_names = {2: ["Thumb", "Pinky"], 3: ["Thumb", "Index", "Pinky"]}[nclass]

    cm   = confusion_matrix(label, trial_preds, labels=list(range(1, nclass + 1)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(
        f"ICA CM — S{subj_id:02} Sess{session_num:02} "
        f"{task} {nclass}-class {modeltype}\n"
        f"Accuracy: {accuracy_online:.2f}%"
    )
    plt.tight_layout()

    # 6. Save results
    subj_folder = os.path.join(cm_root, f"Sujet {subj_id}")
    os.makedirs(subj_folder, exist_ok=True)

    cm_save_path = os.path.join(
        subj_folder,
        f"S{subj_id:02}_Sess{session_num:02}_{task}_{nclass}class_{modeltype}_ICA_CM.png"
    )

    plt.savefig(cm_save_path, dpi=150)
    plt.close(fig)

    print(f"\nEvaluation Complete.")
    print(f"Accuracy: {accuracy_online:.2f}%")
    print(f"Confusion matrix saved to: {cm_save_path}")
    print(f"RESULT:online_val={accuracy:.6f}:online_perf={accuracy_online:.6f}")


if __name__ == "__main__":
    main()

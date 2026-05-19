# ICA Scripts Workflow (Strictly Interactive)

This directory contains the refactored ICA pipeline. **All cleaning is strictly interactive.** No files are saved without first visualizing the signals and tuning thresholds.

## 1. The Standard Suffix: `_ICA`
We use a single suffix for all cleaned data: **`_ICA`**.
- Cleaning scripts generate folders like `OfflineImagery_ICA/`.
- Training and Evaluation scripts look for these `_ICA` files.

---

## 2. Step-by-Step Workflow

### Step 1: Clean your Data

The fitted ICA is cached as a `.fif` file — subsequent runs skip the slow fitting step.

**Option A — Interactive** (tune EOG/EMG thresholds manually):
```bash
python clean_ICA.py <subj> <sess> <nclass> <task> <model>   # online
python clean_ICA.py <subj> <task> OFF                        # offline
```
Tune sliders → Preview removals → Click **Confirm** → Close viewer → Files saved.

**Option B — Batch automatic** (uses `EOG_THRESHOLD` + `EMG_SLOPE_THRESH` from `config.py`):
```bash
python run_all_clean_ica.py
```
Configure `IS_OFFLINE / SESSION_NUM / NCLASS / TASK / MODELTYPE / SUBJECTS` at the top of the file.
Results logged to `cleaning_ica_log.txt`.

---

### Step 2: Train a Model

Single subject:
```bash
python train_model.py <subj> <sess> <nclass> <task> <model>
```

All subjects (configure `SESSION_NUM / NCLASS / TASK / MODELTYPE / SUBJECTS` at top of file):
```bash
python run_all_training_ica.py
```
Saves models as `S{subj}_Sess{sess}_{task}_{nclass}class_{model}_ICA.h5`.

---

### Step 3: Evaluate Performance

Single subject:
```bash
# Evaluate the ICA-trained model on cleaned data
python eval_ica_model.py <subj> <sess> <nclass> <task> <model>

# Robustness check: evaluate the base (raw-trained) model on cleaned data
python eval_old_model.py <subj> <sess> <nclass> <task> <model>
```

All subjects — writes results into the ICA table (cols 9–14) of `Tracking.ods`:
```bash
python run_all_evaluation_ica.py
```

---

## 3. Visualization Tools (No Save)

| Script | Purpose |
| :--- | :--- |
| `viz_inspector.py` | Tune EOG/EMG thresholds and preview Raw vs Cleaned — online or offline via `OFF` suffix (no saving). |
| `viz_compare.py` | Side-by-side Raw vs `*_ICA.mat` — online or offline via `OFF` suffix. |
| `viz_correlations.py` | Heatmap + bar charts of ICA-task correlations, scrollable source viewer, per-component envelope inspector — all simultaneously. |

**Argument format:**
- Online: `<subj> <sess> <nclass> <task> <model>`
- Offline: `<subj> <task> OFF`

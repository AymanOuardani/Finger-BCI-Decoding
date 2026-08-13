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

---

## 4. Energy ↔ Task Analyses

Does the energy the decoder relies on come from the brain or from the artefacts?
These scripts answer that, all built on `energy_stats.py`.

`energy_stats.py` owns the energy extraction, the per-class 0.5×IQR outlier
cleaning, the three association metrics and the Artefact / Non-Artefact
partition of the 128 sources. It is a library, not an entry point. The subject's
**offline ICA is reused for every session**, so a source index means the same
spatial component throughout.

### Metrics

| Metric | Range | Reading |
| :--- | :--- | :--- |
| `pearson` | −1 … +1 | signed Pearson r, per-class 0.5×IQR-cleaned (default) |
| `auc` | 0 … 1 | P(energy higher during the task); 0.5 = chance |
| `rankbiserial` | −1 … +1 | 2·AUC − 1, the signed form of the same |

Bands: `Fullband` `Delta` `Theta` `Alpha` `Beta` `Beta+2` `Gamma`.
Artefact definitions (`essai`): `auto` (offline EOG/EMG detection), `manuel`
(components inspected by hand), `all` (both).

### Figures → `RESULTS_ROOT`

```bash
# tasks x sources heatmaps, one per recording
python corr_maps.py <subj> [metric] [band] [task]
python corr_maps.py 9 auc

# distribution over the 128 sources, Artefact vs Non-Artefact
python corr_distributions.py <subj> [metric] [band] [essai] [task]
python corr_distributions.py 9 pearson Beta

# per-source diagnostics: trials | bands | sorted
python viz_source_energy.py <subj> <sess> <nclass> <model> <src> [mode] [band] [task]
python viz_source_energy.py 4 5 3 Orig 80 bands

# where in the spectrum the energy sits: task | full
python viz_source_spectrum.py <subj> <sess> <nclass> <model> <src> [mode] [task]
```

### Numbers → `Ressources/*.xlsx`

```bash
# Artefact-vs-Non-Artefact separation (AUC on |r|, Cohen's d, p)
# -> Ressources/Separation_Metrics.xlsx, one sheet per (subject, band)
python fill_separation_metrics.py <subj> [band] [essai] [task]
python fill_separation_metrics.py all

# per-source metric suite for one recording
# -> Ressources/Source_Metrics.xlsx, one sheet per recording
python fill_source_metrics.py <subj> <sess> <nclass> <model> [band] [task]
```

`fill_source_metrics.py` reports the raw, log, trimmed and rank-based variants
side by side on purpose: per-trial energy is heavy-tailed, so a raw Pearson r
can be produced almost entirely by a handful of high-energy trials. When the raw
r is large and the others are not, the association lives in the tail.

### Audit

```bash
# checks trial windows against the .mat markers and re-derives the energy
python verify_energy.py [subj] [sess] [nclass] [model] [src] [task]
```

---

## 5. Where results go

| Output | Location |
| :--- | :--- |
| Cleaned recordings | `<recording folder>_ICA/` |
| Accuracies | `Ressources/Tracking.ods` |
| Energy workbooks | `Ressources/Sujet_XX.ods`, `Correlation_Evolution.ods` |
| Separation metrics | `Ressources/Separation_Metrics.xlsx` |
| Per-source metrics | `Ressources/Source_Metrics.xlsx` |
| Per-subject recap | `Ressources/Subject_Reports.xlsx` |
| Figures | `RESULTS_ROOT/` (Correlation Maps, Correlation Distributions, Source Diagnostics) |

No script emits LaTeX any more; numeric results go to `.xlsx` and figures stay
PNG.

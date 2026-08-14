# Finger-BCI-Decoding (STING)

This repository is a fork of the code released alongside:

> Ding, Y., Udompanyawit, C., Zhang, Y., & He, B. (2025). *EEG-based brain-computer interface enables real-time robotic hand control at individual finger level.* Nature Communications, 16(1), 5401. https://doi.org/10.1038/s41467-025-61064-x

It was extended during an engineering internship to (1) reproduce the article's individual-finger EEGNet decoding pipeline offline, and (2) investigate what the decoder is actually learning — through ICA-based artefact/source analysis, ERD/ERS topography reproduction, and cross-session tracking of the correlation between per-source signal energy and the finger task. The internship work sits alongside the original online/offline pipeline rather than replacing it; see the [Attribution](#attribution--citation) section at the end for the original credits.

## Overview

The codebase covers three layers:

1. **Reproduction of the article's pipeline** — EEGNet training/evaluation on raw EEG (`main_model_training.py`, `main_model_evaluation.py`), for both paradigms (**MI**: motor imagery, **ME**: movement execution), both class settings (2-class: Thumb/Pinky, 3-class: Thumb/Index/Pinky), and both model stages (**Orig**/Base and **Finetune**).
2. **Independent-component analysis of what the decoder relies on** — ICA decomposition of the 128-channel recordings, interactive/automatic artefact (EOG/EMG) labelling, training and evaluating EEGNet on ICA-cleaned data, and per-source energy↔task correlation analysis to quantify how much of the decoder's discriminative signal comes from brain sources versus ocular/muscular artefacts (`ICA_Scripts/`).
3. **ERD/ERS topography reproduction** — per-finger event-related desynchronization/synchronization topomaps, single-subject and group-averaged, following the article's preprocessing method (`ERD_Scripts/`).

Supporting these are batch runners over all subjects, accuracy/correlation distribution plots (`Analysis/`), and a per-subject Excel recap generator (`Subjects Reports/`).

## Installation

- **Python**: developed against Python 3.9 (see `EEGModels_tf.py`'s own note that it targets TensorFlow 2.0–2.3; a Python 3.8–3.10 environment is recommended for compatibility with that TensorFlow range, though any environment providing a working `tensorflow` + `mne` stack should work).
- Install dependencies with pip:

```bash
pip install numpy scipy pandas tensorflow scikit-learn mne matplotlib openpyxl odfpy sympy
```

These are the packages actually imported across the pipeline:
- `numpy`, `scipy`, `pandas` — numerical/signal processing and tabular data.
- `tensorflow` (with `tensorflow.keras`) — EEGNet training/inference (`EEGModels_tf.py`, `Functions.py`).
- `scikit-learn` — confusion matrices (`sklearn.metrics`) in evaluation scripts.
- `mne` — EEG data structures, filtering, ICA, and Morlet time-frequency analysis (`mne.time_frequency.tfr_array_morlet`) used throughout `ICA_Scripts/` and `ERD_Scripts/`.
- `matplotlib` — all figures (confusion matrices, topomaps, correlation heatmaps, distribution plots, interactive viewers).
- `openpyxl` — writing `.xlsx` result workbooks (`ICA_Scripts/excel_io.py`, `Subjects Reports/generate_subject_report.py`).
- `odfpy` — reading/writing the `.ods` tracking files (`Tracking.ods`, `Sujet_XX.ods`, `Correlation_Evolution.ods`) via `Functions.write_to_ods` and `clean_ods_formulas.py`.
- `sympy` — used by one of the group-averaged ERD scripts.

`main_online_processing.py` additionally depends on `BCPy2000` (`from BCPy2000.GenericSignalProcessing import BciGenericSignalProcessing`), which must be installed separately from the BCI2000 source; it is only needed for real-time/online acquisition, not for any of the offline analysis described below.

### Configuring `config.py`

`config.py` is the single source of truth for every path and parameter — it is documented in detail at the top of the file and must be edited (not overridden per-script) before running anything. At minimum, update the paths section for your machine:

- `DATA_FOLDER` — root of the raw EEG dataset (see [Data Setup](#data-setup)).
- `SAVE_FOLDER` — where trained models (`.h5`) and the cached ICA fits (`SAVE_FOLDER/ICA_Fitted/`) are written.
- `RESULTS_ROOT` — where figures (confusion matrices, topographies, correlation heatmaps) are saved.
- `SCRIPT_DIR` — legacy path used by some batch runners; point it at your local checkout if needed.

`EXCEL_DIR`/`TRACKING_ODS`/`EXCEL_DATA` and the other `Ressources/*` paths are derived automatically from the repo's own location and normally don't need editing. Signal parameters (`SRATE`, `N_CHANNELS`, `DOWNSRATE`, `BANDPASS_FILT`, …), EEGNet hyperparameters, ICA thresholds, class/label mappings, and the ODS layout constants are all also centralized in this file — see its inline comments for what each one controls.

## Data Setup

The pipeline expects the EEG-BCI dataset from the original article, organized per subject under `config.DATA_FOLDER`:

```
S{XX}/
  OfflineImagery/                                     # offline motor-imagery recording (used to fit the reusable ICA)
  OfflineMovement/                                     # offline movement-execution recording
  OnlineImagery_Sess{NN}_{n}class_{Base|Finetune}/     # online MI recordings (one block of 8 runs)
  OnlineMovement_Sess{NN}_{n}class_{Base|Finetune}/    # online ME recordings
  ..._ICA/                                             # ICA-cleaned *_ICA.mat siblings, created by the cleaning scripts
```

Each recording is a set of `.mat` files with two top-level keys, `eeg` (128-channel data, sampling rate, per-sample predictions/probabilities) and `event` (trial markers). Subjects are numbered S01–S21 (`config.ALL_SUBJECTS`). Never hard-code these paths in a script — resolve them through the helpers in `Functions.py` (`get_folder`, `get_offline_folder`, `get_ica_folder`, `get_ica_fif_path`), which build them from `config.DATA_FOLDER`/`config.SAVE_FOLDER`.

## Pipeline Overview

### Raw EEGNet training/evaluation — reproducing the article

- `main_model_training.py` — trains EEGNet on raw EEG for one subject/session/class-count/task/model-type, saves the model as an `.h5` file, and logs the validation accuracy to `Tracking.ods`.
- `main_model_evaluation.py` — evaluates a trained model, writes online validation and online performance accuracy to `Tracking.ods`, and saves a confusion-matrix PNG.
- `run_all_training.py` / `run_all_evaluation.py` — batch wrappers that loop the above over all subjects for a fixed session/class-count/task/model-type (edit the constants at the top of each file).
- `main_online_processing.py` — the real-time BCI signal-processing loop (a modified BCPy2000 script), for online/live acquisition rather than offline analysis.

### `ICA_Scripts/` — what is the decoder actually using?

- **Cleaning**: `clean_ICA.py` (interactive, slider-tuned EOG/EMG thresholds) and `clean_ica_offline.py` / `run_all_clean_ica.py` (batch, using the thresholds in `config.py`) decompose recordings into 128 independent components, flag artefactual ones, and save cleaned `*_ICA.mat` files. The fitted ICA is cached as a `.fif` file so refitting is only needed once per subject/task.
- **Training/evaluation on cleaned data**: `train_model.py` trains EEGNet on the `_ICA`-cleaned recordings; `eval_ica_model.py` evaluates it; `eval_old_model.py` checks how a model trained on raw data performs on cleaned data (a robustness check). `run_all_training_ica.py` / `run_all_evaluation_ica.py` batch these over subjects and write into the dedicated ICA columns of `Tracking.ods`.
- **Energy↔task correlation analysis**: `energy_stats.py` is the shared library (per-band energy extraction, outlier cleaning, Pearson/AUC/rank-biserial association metrics, artefact-vs-non-artefact partitioning of the 128 sources) used by every script below. `corr_maps.py` and `corr_distributions.py` produce heatmap/distribution figures. `fill_energy_table.py`, `fill_energy_table_simple.py`, `fill_correlation_evolution.py`, `fill_separation_metrics.py`, and `fill_source_metrics.py` compute these metrics and write them into `.xlsx`/`.ods` result tables under `Ressources/`. The underlying question: does the signal the decoder relies on come from brain sources or from EOG/EMG artefact components?
- **Visualization**: `viz_inspector.py`, `viz_compare.py`, `viz_sources.py`, `viz_sources_ica_offline.py`, `viz_correlations.py`, `viz_source_energy.py`, `viz_source_spectrum.py`, `viz_dual_sources.py`, `viz_sq_pinky.py` — interactive inspection tools, none of which save files on their own.
- See `ICA_Scripts/README.md` for the exact interactive workflow (clean → train → evaluate → analyze) and full CLI syntax for every script in this folder.

### `ERD_Scripts/` — ERD/ERS topography reproduction

Reproduces the article's per-finger event-related desynchronization/synchronization topomaps (Morlet wavelet power, baseline-normalized, per-finger). `ERD_Topo.py` handles a single subject (online or offline); `ERD_Topo_Group.py`, `ERD_Topo_Group_Normal.py`, and `ERD_Topo_Group_SL.py` produce group-averaged variants (before/after ICA comparison, self-normalized scaling, and a dynamic/unclamped color scale, respectively) across the subjects listed in `config.GROUP_ERD_SUBJECTS`.

### `Analysis/` — distribution and evolution plots

`Distribution_Plot_Accuracies.py` and `Distribution_Plot_Correlations.py` (plus its batch variant `Distribution_Plot_Correlations_Auto.py`) plot accuracy and energy↔task correlation distributions, comparing artefact vs. non-artefact ICA sources. `Plot_Accuracy_Evolution.py` plots how accuracy evolves across the Base→Finetune, Session 1→2 conditions for a given subject.

### `Subjects Reports/` — per-subject Excel recap

`generate_subject_report.py` checks a given subject for missing confusion matrices, accuracies, and accuracy-evolution plots, generates whatever is missing, and writes a two-sheet recap (`S{XX}_Accuracy`, `S{XX}_Figures`) into `Ressources/Subject_Reports.xlsx`. `run_all_reports.py` batches this over a configurable list of subjects. (Earlier versions of this step emitted a LaTeX/PDF report per subject; the pipeline now writes structured `.xlsx` output instead — no script in the repository currently emits LaTeX or PDF.)

### `EEG_Visualizer.py` / `clean_ods_formulas.py`

`EEG_Visualizer.py` is an interactive channel/topomap viewer for raw or band-passed signals. `clean_ods_formulas.py` repairs a LibreOffice-specific formula-prefix corruption bug in the `.ods` tracking files, either as a one-shot fix or a `--watch` daemon that fixes files on every save.

## Usage

```bash
# Train / evaluate EEGNet on raw data (reproducing the article)
python main_model_training.py 1 1 2 ME Orig
python main_model_evaluation.py 1 1 2 ME Orig

# Batch over all subjects (edit SESSION_NUM/NCLASS/TASK/MODELTYPE at the top of the file first)
python run_all_training.py
python run_all_evaluation.py

# ICA cleaning (offline, interactive) then training/evaluation on cleaned data
cd ICA_Scripts
python clean_ICA.py 9 MI OFF
python train_model.py 9 1 3 MI Orig
python eval_ica_model.py 9 1 3 MI Orig

# Energy <-> task correlation analysis for one subject
python fill_energy_table.py 9
python corr_maps.py 9 auc
python corr_distributions.py 9 pearson Beta
python fill_separation_metrics.py 9

# ERD topography reproduction
cd ../ERD_Scripts
python ERD_Topo.py 9 1 2 MI Orig Fullband        # online, single subject
python ERD_Topo.py 9 MI Fullband OFF              # offline, single subject
python ERD_Topo_Group.py MI Fullband OFF          # offline, group-averaged

# Per-subject Excel recap
cd "../Subjects Reports"
python generate_subject_report.py --subj 9
```

Most scripts accept arguments either positionally (`<subj> <sess> <nclass> <task> <model>` for online conditions, `<subj> <task> OFF` for offline) or via `--subj/--session/--nclass/--task/--model` flags, and will prompt interactively for anything left unspecified. See each script's module docstring, or `ICA_Scripts/README.md`, for exact per-script argument order.

## Attribution & Citation

This repository builds on the scripts released for:

> Ding, Y., Udompanyawit, C., Zhang, Y., & He, B. (2025). EEG-based brain-computer interface enables real-time robotic hand control at individual finger level. Nature Communications, 16(1), 5401. https://doi.org/10.1038/s41467-025-61064-x

The original scripts include the online processing/decoding loop as well as offline deep-learning model training. EEGNet is used as the decoder, and the TensorFlow implementation (`EEGModels_tf.py`) is taken from the Army Research Laboratory (ARL) EEGModels project [1]:

https://github.com/vlawhern/arl-eegmodels

This work was supported by the National Institutes of Health via grants NS124564, NS131069, NS127849, and NS096761 to Dr. Bin He.

[1] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces. Journal of Neural Engineering, 15, 056013 (2018).

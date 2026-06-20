# CLAUDE.md — Finger-BCI-Decoding (STING)

Guidance for working in this repository. Read this before editing or running anything.

---

## 1. What this project is

EEG-based Brain-Computer Interface for **real-time robotic hand control at the individual finger level**,
reproducing and extending **Ding et al., Nature Communications 2025**
(*EEG-based brain-computer interface enables real-time robotic hand control at individual finger level*).

- Decoder: **EEGNet** (TensorFlow/Keras), implementation from the ARL EEGModels project (`EEGModels_tf.py`).
- 128-channel BioSemi cap, 21 subjects (S01–S21).
- Two paradigms: **MI** (Motor Imagery) and **ME** (Movement Execution).
- Two class settings: **2-class** (Thumb, Pinky) and **3-class** (Thumb, Index, Pinky).
- Two model types per condition: **Orig** (= "Base" model) and **Finetune** (= "Fine-tuned" model).
- The internship work (this repo's "Suivi du Stage") adds: **ICA cleaning/analysis**, **ERD topographies**,
  **per-source energy ↔ task correlation analysis**, accuracy tracking, and per-subject LaTeX reports.

Language note: code/comments are English; the user communicates in **French**.

---

## 2. Experimental protocol (important for indexing trials)

Each **online session** = **32 runs**, each run = **10 trials per task**, in randomized order.
The 32 runs are 4 consecutive blocks of 8 runs:

| Block | Runs  | Paradigm        | Model     |
|-------|-------|-----------------|-----------|
| 1     | 1–8   | ternary (3c)    | Base/Orig |
| 2     | 9–16  | binary (2c)     | Base/Orig |
| 3     | 17–24 | ternary (3c)    | Finetune  |
| 4     | 25–32 | binary (2c)     | Finetune  |

So a **recording folder** = one block = 8 runs. Trials per recording:
- 2-class: 8 runs × 10 × 2 tasks = **160 trials** (80/task)
- 3-class: 8 runs × 10 × 3 tasks = **240 trials** (80/task)
- Whole session (2c+3c, Base+Finetune) = **800 trials**.

Each trial = 3 s; feedback 1–3 s after onset; 2 s inter-trial interval.

---

## 3. Data layout (on disk)

Raw dataset root: `config.DATA_FOLDER` = `D:/EEG-BCI Dataset for Real-time Robotic Hand Control at Individual Finger Level`.

```
S{XX}/
  OfflineImagery/                              # offline MI (used to fit the reusable ICA)
  OfflineMovement/                             # offline ME
  OnlineImagery_Sess{NN}_{n}class_{Base|Finetune}/   # MI online recordings (one block of 8 runs)
  OnlineMovement_Sess{NN}_{n}class_{Base|Finetune}/  # ME online recordings
  ..._ICA/                                     # cleaned (*_ICA.mat) siblings (get_ica_folder)
```

Resolve paths via `Functions.py`, never hard-code:
- `get_folder(subj, task, session, nclass, model_type)` → online folder. `model_type="Finetune"`→`Finetune` suffix, else `Base`. `task="ME"`→`Movement`, else `Imagery`.
- `get_offline_folder(subj, task)` → offline folder.
- `get_ica_folder(source_folder)` → `<folder>_ICA` sibling for cleaned `.mat`.
- `get_ica_fif_path(subj, task[, session, nclass, model_type])` → cached ICA `.fif`. Offline form: `S{XX}_{task}_Offline-ica.fif`.

### `.mat` file structure
Each `*.mat` has two top-level keys: `eeg` and `event`.
- `eeg` is a struct: `data` (128 × N_samples, float64), `time` (1×N), `label` (128×1), `fsample`, `nChans`,
  `nSamples`, `prediction` (1×N), `prob_thumb`/`prob_pinky` (1×N), `prob_index` (N×1).
- `event` is a struct-array with fields `type, sample, value, offset, duration`. **`sample` is 1-based (MATLAB)**.
- Loaders read `mat["eeg"]["data"][0][0]` and `mat["event"]`. Columns of `eeg.data` = time samples.

---

## 4. `config.py` — single source of truth

Edit ONLY `config.py` for paths/parameters. Key values:
- Paths: `DATA_FOLDER`, `SAVE_FOLDER`, `RESULTS_ROOT`, `REPO_DIR`, `EXCEL_DIR` (= `Ressources/`),
  `TRACKING_ODS`, `ICA_FIF_FOLDER` (= `SAVE_FOLDER/ICA_Fitted`).
- Signal: `SRATE=1024`, `N_CHANNELS=128`, `DOWNSRATE=100`, `WINDOWLEN=1`, `BANDPASS_FILT=[4,40]`.
- EEGNet: `KERN_LENGTH, F1, D, F2, BATCH_SIZE, EPOCHS_ORIG=300, EPOCHS_FINETUNE=100`, etc.
- ICA: `ICA_N_COMPONENTS=128`, `ICA_METHOD="fastica"`, `ICA_ENVELOPE_SMOOTH_HZ=2.0`,
  `EOG_CH_INDICES`, `EOG_THRESHOLD=0.7`, `EMG_SLOPE_THRESH=-1` (**very permissive — flags many components**).
- Classes: `CLASS_NAMES={2:[Thumb,Pinky], 3:[Thumb,Index,Pinky]}`, `TASK_LABELS=(Thumb,Index,Middle,Pinky)`,
  `EVENT_ID` (Thumb=1, Index=2, Middle=3, Pinky=4, TrialEnd=9).
- `ALL_SUBJECTS = 1..21`; `CORRUPTED_DATA` = set of `(subj, task, sess, nclass, model)` to skip.
- ODS layout constants (see §7).

---

## 5. Repository map

```
Functions.py            Central shared library. Entry scripts are thin wrappers. (data load,
                        preprocessing, EEGNet train/eval, ICA helpers, MNE helpers, ODS I/O, CLI helpers)
EEGModels_tf.py         EEGNet (ARL). Don't edit unless changing the model.
config.py               All paths/params/ODS layout.

main_model_training.py      Train EEGNet:  python main_model_training.py <subj> <sess> <nclass> <task> <model>
main_model_evaluation.py    Eval + write accuracies to Tracking.ods + confusion-matrix PNG.
main_online_processing.py   Real-time BCI loop (BCPy2000) — research/online use.
run_all_training.py         Batch wrapper over subjects for training.
run_all_evaluation.py       Batch wrapper over subjects for evaluation (+ ODS + CM PNG).

ICA_Scripts/
  clean_ICA.py              Interactive ICA artifact removal (sliders); caches ICA .fif; persists
                            exclusions into the .fif; saves *_ICA.mat. (online + offline)
  run_all_clean_ica.py      Batch ICA cleaning with fixed EOG/EMG thresholds (no UI).
  viz_inspector.py          Interactive ICA inspector.
  viz_sources.py            Two-window MNE ICA inspector (sources + cleaned signal). Per-session ICA.
  viz_sources_ica_offline.py  Same, but ALWAYS uses the subject's OFFLINE ICA for any session.
  viz_correlations.py       ICA source ↔ task correlation viewers (envelope-based).
  viz_compare.py            Compare raw vs ICA-cleaned.
  corr_heatmap.py           Envelope-correlation heatmaps (tasks × sources) per band. Reuses offline ICA.
  train_model.py            Train EEGNet on cleaned (*_ICA) data.
  eval_ica_model.py         Eval EEGNet trained on cleaned data; prints RESULT line.
  run_all_training_ica.py / run_all_evaluation_ica.py   Batch wrappers (write to Tracking.ods ICA table).
  fill_correlation_evolution.py   Fills Ressources/Correlation_Evolution.ods (per-subject sheet).
  fill_energy_table.py      Builds Ressources/Sujet_XX.ods (per-trial energy + energy↔task corr + evolution).

ERD_Scripts/                Per-finger ERD topographies (Morlet), single-subject + group, paper method.
Analysis/                   Accuracy distribution / evolution plots.
Subjects Reports/           generate_subject_report.py + run_all_reports.py → per-subject LaTeX report.
clean_ods_formulas.py       Repairs LibreOffice formula corruption in .ods (see §7). One-shot or --watch.
Ressources/                 Tracking.ods, Sujet_XX.ods, Correlation_Evolution.ods, Excel DATA.xlsx.
```

---

## 6. Key conventions & shared helpers (Functions.py)

- **CLI args**: `get_standard_args()` → `<subj> <sess> <nclass> <task> <model>`; `get_offline_args()` → `<subj> <task>` (used with trailing `OFF`).
- **Raw build**: `build_raw_from_mat_files(mat_files)` → MNE Raw (µV→V), with annotations from events
  (`Thumb/Index/Middle/Pinky` + `TrialEnd`). Raises on empty/corrupt `.mat`.
- **Task vectors**: `build_task_vectors(raw)` → `{task_name: binary vector}`; a task annotation runs to its
  duration, else to the next `TrialEnd`, else 3 s.
- **Envelope**: `compute_envelope(signal, sfreq)` → smoothed Hilbert amplitude (low-pass `ICA_ENVELOPE_SMOOTH_HZ`).
- **ICA fit/load**: `fit_or_load_ica(raw_filt, ica_cache_path)` — loads cached `.fif` (and **resets `exclude=[]`**)
  or fits FastICA (128 comps) and saves. Preprocessing for fitting: 60 Hz notch + 1 Hz highpass.
- **Artifact detection**: `compute_exclusions(ica, raw_filt, muscle_thresh=EMG_SLOPE_THRESH, eog_thresh=EOG_THRESHOLD, ch_names=...)`
  → `(all_excl, eog_idx, emg_idx)`. EOG via correlation with `EOG_CH_INDICES`; EMG via spectral slope.
- **ODS write**: `write_to_ods(ods_path, row_idx, [(col, value), ...], table_name)` — one row per call (re-zips the file).

### Offline-ICA reuse (a core design choice)
For correlation/energy analyses and `viz_sources_ica_offline.py`, the subject's **offline ICA is reused for every
session** (fit once on the offline recording, applied to each online recording). This makes **ICA source index `c`
the same spatial component across sessions** — required for the evolution analysis. Helper:
`corr_heatmap._load_offline_ica(subj, task)` (loads the offline `.fif` directly, preserving saved `exclude`;
fits from the offline recording if absent).

---

## 7. Excel / ODS files (Ressources/)

### Tracking.ods — accuracy tracking
Per-condition sheets named `MI_Sess{NN}_{n}Class`. Rows 0–4 are headers; subject S{XX} → row `subj_id + ROW_SUBJECT_OFFSET (4)`.
- Original table cols: Orig `SHEET_MODEL_START["Orig"]=1` (Val/OnlineVal/OnlinePerf = 1/2/3), Finetune = 4/5/6.
- ICA table cols: Orig `ICA_MODEL_START["Orig"]=9` (9/10/11), Finetune = 12/13/14; subject col 8.
- Accuracy-plot band sections at row offsets `ROW_OFFSETS` (Full=4, Alpha=32, Beta=57, Beta+=82).

### ⚠️ LibreOffice formula corruption (recurring)
ODF formulas are stored as `of:=SUM(...)`. LibreOffice **duplicates the prefix on each save**
(`of:=of:=SUM...`, shown as `=of:=SUM(...)`). `write_to_ods` self-heals via `_normalize_formula_prefixes`
on every write. To clean a file on demand or live while editing: `python clean_ods_formulas.py [path] [--watch]`.

### Sujet_XX.ods — per-subject energy workbook (built by `fill_energy_table.py <subj>`)
- **10 energy sheets** `Sess_XX_nClass` (one per session × nClass, **Orig+Finetune merged**, `Model` column).
  Columns: `Sujet_id | Session | nClass | Model | Task | Class | Trial | ICA_Source | Energy_Fullband |
  Energy_Alpha | Energy_Beta | Energy_Beta+2 | Artefact ?`. Trials in **chronological** order (by onset), 128 sources/trial.
  Energy = Σ x² of the band-filtered ICA source over the trial window. `Artefact ?` = Yes if source flagged EOG/EMG.
- **`Corr` sheet** (long format): `Session | nClass | Band | Task | ICA_Source | Corr_Energy | Artefact ?`.
  `Corr_Energy` = **SIGNED** point-biserial Pearson in **[-1, 1]** between the source's per-trial energy and the
  task's one-vs-rest membership, over ALL trials of that session+nClass (Orig+FT combined). (Pearson, not rank-based,
  so energy magnitude matters; sign = direction.)
- **`Enrgy_Corr_Evolution` sheet** (3-class): 12 blocks (4 bands × 3 tasks), header `Energy_<band> (<task>, 3Class)`
  + `Source_1..128`, then one row per session `Sess0X` = the signed corr of each source at that session.
- Built from scratch (overwrites the file). One subject at a time. Heatmaps → see §8.

### Correlation_Evolution.ods — built by `fill_correlation_evolution.py <subj>`
21 sheets `Sujet1..Sujet21` (band blocks × 2/3-class × Session rows), filled with summed correlations of
auto-detected EOG/EMG artifact sources.

When editing ODS by hand (adding/clearing a sheet), parse `content.xml` with ElementTree, register namespaces
via `Functions._get_namespaces`, modify the target `<table>`, and rewrite the zip preserving every other entry.
**Always back up** (`*.ods.bak`) first.

---

## 8. Output locations (under SAVE_FOLDER's "Suivi du Stage" tree)

- Confusion matrices: `RESULTS_ROOT/Sujet {X}/S{XX}_Sess{NN}_{TASK}_{n}class_{MODEL}_CM.png`.
- ICA `.fif` cache: `SAVE_FOLDER/ICA_Fitted/`.
- Envelope-correlation heatmaps (`corr_heatmap.py`): `…/Week 8/Correlations Maps/S{XX}/{FullBand|Alpha|Beta|Beta+2}/`
  — viridis, scale **0–1** (|correlation|). Bands: Broadband/Alpha(8-13)/Beta(13-30)/Beta+2(30-40).
- Energy↔task heatmaps (`fill_energy_table.py`): `…/Week 8/Energy Correlation Maps/S{XX}/<Band>/`
  — **RdBu_r, signed, scale -1..+1** (+1 red, -1 blue), EOG/EMG artifact source indices in red on the x-axis.

---

## 9. Common commands (Windows / PowerShell)

```powershell
# Training / evaluation (raw)
python main_model_training.py 9 1 2 MI Orig
python main_model_evaluation.py 9 1 2 MI Orig

# ICA-cleaned model train/eval (run from ICA_Scripts)
python eval_ica_model.py 6 1 3 MI Orig

# ICA cleaning / inspection
python clean_ICA.py 9 MI OFF                  # offline, interactive
python viz_sources_ica_offline.py 9 1 3 MI Orig

# Correlation / energy analyses (one subject at a time)
python corr_heatmap.py 9 MI OFF
python fill_energy_table.py 9
python fill_correlation_evolution.py 9

# ODS formula repair
python clean_ods_formulas.py                  # one-shot on Tracking.ods
python clean_ods_formulas.py --watch          # auto-fix on every save
```

---

## 10. Agent guidance / gotchas

- **Platform**: Windows + PowerShell. Paths contain spaces and accents (é, è, û). Quote paths; expect mojibake in
  console output (cosmetic). The dataset is on `D:`, outputs on `C:`.
- **Run heavy jobs ONE AT A TIME.** Loading recordings + 128-component band filtering is memory-heavy; running two
  such Python processes in parallel has caused **OOM** (`std::bad_alloc` / "Unable to allocate ..."). Use background
  runs sequentially.
- **Offline recordings are huge** (e.g. ~5.8 M samples). The standard offline `corr_heatmap` band-filtering can OOM;
  process **per component** when needed (frugal loop) rather than filtering all 128 channels at once.
- **Sandbox false-positives**: piping a Python heredoc (`@'...'@ | python -`) through PowerShell can be wrongly
  flagged as a destructive command (e.g. on `del`, `as z:`). **Write a temp `.py` file and run it**, then delete it.
- **Temp scripts**: prefix with `_` (e.g. `_inspect.py`), run, then delete. Keep the repo clean.
- **Destructive edits** (truncating `.mat`, overwriting `.ods`): always create a `.bak` first and report what changed.
- **MATLAB 1-based vs Python 0-based**: `event.sample` and "columns" in `.mat` are 1-based; convert carefully.
- **Git**: default branch `main`; active work on `develop`. Commit/push only when asked; branch first if on `main`.
  End commit messages with the required `Co-Authored-By` trailer.
- **Known data issues**: `config.CORRUPTED_DATA`; `S03 OnlineImagery_Sess03_3class_Base_R08.mat` was manually
  truncated to 94483 samples (original in `..._R08.mat.bak`).

---

## 11. Analysis ideas the energy workbook enables
(See the per-source energy↔task correlation.) Evolution/learning across sessions (corr slope per source);
top task-discriminative sources + topography check; **artifact vs neural** contribution (fraction of task-correlated
energy carried by EOG/EMG sources — a validity check); band specificity (mu/beta ERD); Base vs Finetune;
per-finger separability (cross-ref confusion matrices); permutation significance vs chance.

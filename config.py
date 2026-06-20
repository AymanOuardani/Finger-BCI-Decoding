"""
config.py

Single source of truth for all paths, signal parameters, model hyperparameters,
ICA settings, and ODS layout constants used across the STING project.

Edit ONLY this file to update paths or experiment parameters.
"""

import os

# =============================================================================
# PATHS
# =============================================================================

DATA_FOLDER = (
    "D:/EEG-BCI Dataset for Real-time Robotic Hand Control"
    " at Individual Finger Level"
)
SAVE_FOLDER = (
    "C:/Users/aymen/Desktop/5 Years Of Engineering/DATASIM/DATASIM - Ouardani"
    "/15 - STING (Début 30-03 et Fin 14-08)/Suivi du Stage"
    "/Itération 1/Week 1/W1D1 - 30-03/SavePath"
)
RESULTS_ROOT = (
    "C:/Users/aymen/Desktop/5 Years Of Engineering/DATASIM/DATASIM - Ouardani"
    "/15 - STING (Début 30-03 et Fin 14-08)/Suivi du Stage"
    "/Itération 2 - Résultats"
)
SCRIPT_DIR = (
    "C:/Users/aymen/Desktop/5 Years Of Engineering/DATASIM/DATASIM - Ouardani"
    "/15 - STING (Début 30-03 et Fin 14-08)/Suivi du Stage"
    "/Itération 1/Week 1/W1D1 - 30-03/Codes/Finger-BCI-Decoding-main/Original_Codes"
)
# Excel/ODS tracking files live in the repo's own Ressources folder
# (the directory containing this config.py), not in SCRIPT_DIR.
REPO_DIR     = os.path.dirname(os.path.abspath(__file__))
EXCEL_DIR    = os.path.join(REPO_DIR, "Ressources")
TRACKING_ODS = os.path.join(EXCEL_DIR, "Tracking.ods")
EXCEL_DATA   = os.path.join(EXCEL_DIR, "Excel DATA.xlsx")
ICA_FIF_FOLDER = os.path.join(SAVE_FOLDER, "ICA_Fitted")

# =============================================================================
# EEG SIGNAL PARAMETERS
# =============================================================================

SRATE       = 1024    # raw acquisition sampling rate (Hz)
N_CHANNELS  = 128     # BioSemi 128-channel cap
DOWNSRATE   = 100     # downsampled rate used for training / evaluation (Hz)
WINDOWLEN   = 1       # sliding-window segment length (s)
BLOCK_SIZE  = 128     # online processing block size (samples)
BANDPASS_FILT = [4, 40]   # Butterworth bandpass range (Hz)

MAXTRIAL_S          = 3   # max trial duration for evaluation (s)
MAXTRIAL_TRAINING_S = 5   # max trial duration for training (s)

# =============================================================================
# EEGNET MODEL HYPERPARAMETERS
# =============================================================================

KERN_LENGTH      = 32
F1               = 8
D                = 2
F2               = 16
BATCH_SIZE       = 16
EPOCHS_ORIG      = 300
EPOCHS_FINETUNE  = 100
DROPOUT_ORIG     = 0.5
DROPOUT_FINETUNE = 0.65
LAYERS_FINE_TUNE = 12
TRAIN_PERCENT    = 0.8
STEP_SIZE        = 128   # sliding-window step size (samples)
PADDING_LENGTH   = 100   # zero-padding length for bandpass edge artefacts

# =============================================================================
# ICA PARAMETERS
# =============================================================================

ICA_N_COMPONENTS       = 128    # for individual evaluation / visualization
ICA_N_COMPONENTS_GROUP = 20    # for group ERD (article method)
ICA_METHOD             = "fastica"
ICA_ENVELOPE_SMOOTH_HZ = 2.0   # low-pass cutoff for smoothed Hilbert envelope
ICA_N_PER_PAGE         = 32    # number of ICA components shown per page in visualizers
EOG_CH_INDICES         = [79, 92, 81, 80, 78, 91, 93, 94, 72, 71, 70, 102]
EOG_THRESHOLD          = 0.7   # EOG correlation threshold for component removal
EOG_THRESHOLD_GROUP    = 3.0   # correlation threshold (group / article method)
EMG_SLOPE_THRESH       = -1    # spectral slope threshold for muscle component removal

# =============================================================================
# CLASS / SUBJECT CONFIGURATION
# =============================================================================

ALL_SUBJECTS = list(range(1, 22))   # subjects S01 to S21

# Subjects included in group ERD plots — edit this list to restrict the group.
# Set to ALL_SUBJECTS to include everyone.
GROUP_ERD_SUBJECTS = [2,3,4,5,6,7,8,9,11,12,13]

KEEP_LABELS = {
    2: [1, 4],        # 2-class: thumb, pinky
    3: [1, 2, 4],     # 3-class: thumb, index, pinky
}
NEW_LABELS = {
    2: {1: 1, 4: 2},
    3: {1: 1, 2: 2, 4: 3},
}
CLASS_NAMES = {
    2: ["Thumb", "Pinky"],
    3: ["Thumb", "Index", "Pinky"],
}
TASK_LABELS = ("Thumb", "Index", "Middle", "Pinky")
EVENT_ID = {
    "Thumb": 1,
    "Index": 2,
    "Middle": 3,
    "Pinky": 4,
    "TrialEnd": 9,
}
ID_TO_LABEL = {v: k for k, v in EVENT_ID.items()}
FINGER_LABEL = {1: "Thumb", 2: "Index", 3: "Middle", 4: "Pinky"}

# =============================================================================
# FREQUENCY BANDS
# =============================================================================

BANDS = {
    "alpha": (8,  13),
    "beta":  (13, 30),
    "mu":    (8,  13),   # alias for alpha
    "full":  (4,  40),
}

# =============================================================================
# GROUP ERD / ARTICLE PREPROCESSING PARAMETERS
# =============================================================================

PREPROC_BANDPASS = (2, 30)   # bandpass used in the article method
SD_THRESHOLD_UV  = 20.0      # trial exclusion: SD > 20 µV on any channel
SEGMENT_PRE_S    = 2.0       # full segment starts 2 s before onset
BASELINE_PRE_S   = 1.0       # baseline window: 1 s before onset
TASK_ONSET_S     = 0.5       # task period starts 0.5 s after onset
MORLET_N_CYCLES  = 7         # Morlet wavelet cycles
ALL_COMBOS       = [(1, 2), (1, 3), (2, 2), (2, 3)]   # (session, nclass)

# =============================================================================
# ODS TRACKING LAYOUT
# =============================================================================

# Per-condition sheets (e.g. MI_Sess02_2Class):
#   Rows 0-4 = headers; S01 → row subj_id + ROW_SUBJECT_OFFSET
#
#   Original table (cols 0–6):
#     Col 0        : Subject label
#     Orig model   : cols 1 (Val), 2 (Online Val), 3 (Online Perf)
#     Finetune     : cols 4 (Val), 5 (Online Val), 6 (Online Perf)
#
#   ICA table (cols 8–14):
#     Col 8        : Subject label
#     ICA Orig     : cols 9 (Val), 10 (Online Val), 11 (Online Perf)
#     ICA Finetune : cols 12 (Val), 13 (Online Val), 14 (Online Perf)

ROW_SUBJECT_OFFSET = 4
SHEET_MODEL_START  = {"Orig": 1,  "Finetune": 4}    # original table: first col per model block
ICA_MODEL_START    = {"Orig": 9,  "Finetune": 12}   # ICA table: first col per model block
ICA_SUBJECT_COL    = 8                               # ICA table: Subject column index

# Band-section row offsets used in Tracking.ods for accuracy plots
ROW_OFFSETS = {
    "Full":   4,
    "Alpha":  32,
    "Beta":   57,
    "Beta+":  82,   # Beta+ [13–40 Hz]
}
COL_ONLINE_VAL  = {"Orig": 2, "Finetune": 5}    # original table: Online Val_Accuracy
COL_ONLINE_PERF = {"Orig": 3, "Finetune": 6}    # original table: Online Perf_Accuracy
ICA_ONLINE_VAL  = {"Orig": 10, "Finetune": 13}  # ICA table: Online Val_Accuracy
ICA_ONLINE_PERF = {"Orig": 11, "Finetune": 14}  # ICA table: Online Perf_Accuracy

# =============================================================================
# CORRUPTED / MISSING DATA
# =============================================================================

CORRUPTED_DATA = {
    (10, "MI", 2, 2, "Finetune"),
    (10, "MI", 3, 2, "Orig"),
    (10, "MI", 3, 2, "Finetune"),
}

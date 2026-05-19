"""
Functions.py

Central shared library for the STING BCI pipeline.
All entry-point scripts are thin wrappers around functions defined here.

Key sections: data loading, preprocessing, model train/eval,
ICA helpers, MNE helpers, path helpers, CLI helpers.
See CLAUDE.md for the full function index.
"""

import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import glob
import re
import copy
import shutil
import zipfile
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import scipy
import scipy.io
import scipy.signal
import scipy.stats
import pandas as pd
import argparse
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from collections import Counter
from scipy.signal import resample
from EEGModels_tf import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
import mne
mne.set_log_level("WARNING")

from config import (
    N_CHANNELS, ICA_N_COMPONENTS, ICA_METHOD,
    EOG_CH_INDICES, EOG_THRESHOLD, EMG_SLOPE_THRESH,
    DATA_FOLDER, SRATE,
    TASK_LABELS, EVENT_ID, ID_TO_LABEL, FINGER_LABEL,
    ICA_ENVELOPE_SMOOTH_HZ, ICA_FIF_FOLDER,
)


# =============================================================================
# DATA SEGMENTING AND RELABELING FUNCTIONS
# =============================================================================

def segment_data(data, labels, segment_size, step_size):
    """
    Splits each EEG trial into overlapping fixed-length segments (sliding window).

    Parameters
    ----------
    data : np.ndarray, shape (nTrials, nChannels, nSamples)
        Raw EEG data. Each trial contains nChannels electrode signals of length
        nSamples. Some samples may be np.nan (padding added in load_and_filter_data).

    labels : np.ndarray, shape (nTrials,)
        Class label for each trial (1-indexed, e.g. 1=thumb, 2=index, 3=pinky).
        Each trial gets the same label for all its segments.

    segment_size : int
        Number of samples per segment.

    step_size : int
        Number of samples to advance the window between segments.
        A step_size < segment_size means consecutive segments overlap.

    Returns
    -------
    segmented_data : np.ndarray, shape (nValidSegments, nChannels, segment_size)
        All valid (non-NaN) segments extracted from all trials.
    
    repeated_labels : np.ndarray, shape (nValidSegments,)
        Label for each segment, copied from the trial it belongs to.

    repeated_indices : np.ndarray, shape (nValidSegments,)
        Original trial index (0-based) for each segment. Used later in eval_model
        to group segments back to their trial for majority voting.
    """
    if segment_size <= 0 or step_size <= 0:
        raise ValueError("segment_size and step_size must be positive.")

    num_trials, num_channels, num_samples = data.shape
    segments = []

    # Slide the window across the time axis of every trial simultaneously.
    # At each position, slice all trials at once → shape (nTrials, nChannels, segment_size).
    for start in range(0, num_samples - segment_size + 1, step_size):
        end = start + segment_size
        segments.append(data[:, :, start:end])

    # Stack along axis 0: shape becomes (nTrials * nWindows, nChannels, segment_size)
    segmented_data = np.concatenate(segments, axis=0)

    # Repeat labels and trial indices to match the new number of rows.
    # np.tile([a,b,c], 3) → [a,b,c,a,b,c,a,b,c]
    repeated_labels = np.tile(labels, len(segments))
    trial_indices = range(num_trials)
    repeated_indices = np.tile(trial_indices, len(segments))

    # Remove any segment that contains at least one NaN value.
    # NaNs come from the zero/NaN padding added during trial extraction
    # when a trial was shorter than maxtriallen.
    valid_mask = ~np.isnan(segmented_data).any(axis=(1, 2))
    repeated_labels  = repeated_labels[valid_mask]
    repeated_indices = repeated_indices[valid_mask]
    segmented_data   = segmented_data[valid_mask, :, :]

    return segmented_data, repeated_labels, repeated_indices


def plot_signals(X_before, X_after, params, segment_idx=0, title='', save_path=None,
                 electrode_names=None):
    """
    Plots EEG signals before and after preprocessing for 3 electrodes side by side.

    The 3 electrodes chosen are: first, middle, and last channel — giving a
    representative spread across the electrode array.

    Parameters
    ----------
    X_before : np.ndarray, shape (nSegments, nChannels, nSamples_raw)
        Segmented EEG before downsampling, filtering, and z-score normalization.
        Sampled at params['srate'].

    X_after : np.ndarray, shape (nSegments, nChannels, nSamples_processed)
        Segmented EEG after all preprocessing (downsample + bandpass + zscore).
        Sampled at params['downsrate'].

    params : dict
        Must contain 'srate', 'downsrate', 'windowlen'.

    segment_idx : int
        Index of the segment to visualise (default: 0).

    title : str
        Figure suptitle, e.g. 'Train set' or 'Eval set'.

    save_path : str or None
        If provided, saves the figure as a PNG to this file path.

    electrode_names : list of str or None
        Full list of channel names (e.g. ['A1', 'A2', ..., 'D32']).
        If provided, names at the 3 selected indices are used in subplot titles.
        If None, falls back to 'Electrode <index>'.
    """
    nChan = X_before.shape[1]

    # Pick 3 electrodes spread across the array
    elec_indices = [0, nChan // 2, nChan - 1]
    if electrode_names is not None:
        elec_labels = [electrode_names[i] for i in elec_indices]
    else:
        elec_labels = [f'Electrode {i}' for i in elec_indices]

    # Time axes (in seconds)
    t_before = np.linspace(0, params['windowlen'], X_before.shape[2])
    t_after  = np.linspace(0, params['windowlen'], X_after.shape[2])

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 9))
    fig.suptitle(f'EEG signal before vs after preprocessing — {title}  (segment #{segment_idx})',
                 fontsize=13, fontweight='bold', y=1.01)

    colors_before = ['#1f77b4', '#2ca02c', '#d62728']   # blue, green, red
    colors_after  = ['#ff7f0e', '#9467bd', '#8c564b']   # orange, purple, brown

    for row, (ch_idx, ch_label, c_b, c_a) in enumerate(
            zip(elec_indices, elec_labels, colors_before, colors_after)):

        ax_before = axes[row, 0]
        ax_after  = axes[row, 1]

        sig_before = X_before[segment_idx, ch_idx, :]
        sig_after  = X_after[segment_idx,  ch_idx, :]

        # --- Before ---
        ax_before.plot(t_before, sig_before, color=c_b, linewidth=0.8)
        ax_before.set_title(f'{ch_label}  |  Raw segment  ({params["srate"]} Hz)',
                            fontsize=10)
        ax_before.set_ylabel('Amplitude (µV)', fontsize=9)
        ax_before.set_xlabel('Time (s)', fontsize=9)
        ax_before.grid(True, linestyle='--', alpha=0.5)
        ax_before.tick_params(labelsize=8)

        # --- After ---
        ax_after.plot(t_after, sig_after, color=c_a, linewidth=0.8)
        ax_after.set_title(
            f'{ch_label}  |  After downsample + bandpass + z-score  ({params["downsrate"]} Hz)',
            fontsize=10)
        ax_after.set_ylabel('Amplitude (z-score)', fontsize=9)
        ax_after.set_xlabel('Time (s)', fontsize=9)
        ax_after.grid(True, linestyle='--', alpha=0.5)
        ax_after.tick_params(labelsize=8)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Plot saved to {save_path}')

    plt.show()


def filter_and_relabel(data, label, keep_labels, new_labels):
    """
    Keeps only the trials belonging to selected classes and remaps their labels
    to a contiguous 1-based integer range expected by the model.

    Why: The raw dataset contains 5 finger classes (1–5). Depending on nclass,
    we only use a subset (e.g. thumb=1 and pinky=4 for 2-class). The model also
    expects labels starting at 1 with no gaps.

    Parameters
    ----------
    data : np.ndarray, shape (nTrials, nChannels, nSamples)
        EEG data for all trials (possibly containing classes we do not want).

    label : np.ndarray, shape (nTrials,)
        Original class labels (raw values from the .mat files, e.g. 1,2,3,4,5).

    keep_labels : list of int
        Which original labels to retain.
        - 2-class: [1, 4]   (thumb, pinky)
        - 3-class: [1, 2, 4] (thumb, index, pinky)

    new_labels : dict {original_label: new_label}
        Mapping from original labels to model-friendly labels.
        - 2-class: {1: 1, 4: 2}
        - 3-class: {1: 1, 2: 2, 4: 3}

    Returns
    -------
    filtered_data : np.ndarray, shape (nKeptTrials, nChannels, nSamples)
        Data with only the trials whose label is in keep_labels.

    filtered_label : np.ndarray, shape (nKeptTrials,)
        Remapped labels (1-based, contiguous) for the kept trials.
    """
    # Boolean mask: True where the trial's label is one of the kept classes
    filtered_label = label[np.isin(label, keep_labels)]
    filtered_data  = data[np.isin(label, keep_labels)]

    # Remap each label using the dictionary
    filtered_label = np.array([new_labels[l] for l in filtered_label])

    return filtered_data, filtered_label


def generate_paths(subj_id, task, nclass, session_num, model_type, data_folder):
    """
    Builds the list of directory paths containing the TRAINING data files for a
    given subject, task, session, and model type.

    The logic differs between the two model types:
    - 'Orig'     : uses ALL available data up to (but not including) the current
                   session (offline baseline + all prior online sessions).
    - 'Finetune' : uses only the same-day Base data (the current session's
                   calibration block) to fine-tune the pre-trained Orig model.

    Parameters
    ----------
    subj_id : int
        Subject identifier (1–21). Used to build the subject folder name (e.g. 'S01').

    task : str
        'MI' for Motor Imagery or 'ME' (anything else) for Motor Execution.
        Determines the filename prefix ('*Imagery' vs '*Movement').

    nclass : int
        Number of classes (2 or 3). Only affects the suffix when model_type='Finetune'.
        Fine-tuning always uses the '{nclass}class_Base' suffix to avoid matching
        the wrong class-count folder (e.g. 2-class must not match 3class_Base).

    session_num : int
        Current online session number (1–5).

    model_type : str
        'Finetune' or anything else (treated as 'Orig').

    data_folder : str
        Root directory containing one sub-folder per subject (e.g. 'S01/', 'S02/', …).

    Returns
    -------
    data_paths : list of str
        Sorted list of directory paths matching the glob patterns. Each path
        points to a folder that load_and_filter_data will iterate over.
    """
    subject_folder = os.path.join(data_folder, f'S{subj_id:02}')

    # Choose prefix based on task
    if task == 'MI':
        prefix = '*Imagery'
    else:
        prefix = '*Movement'

    if model_type == 'Finetune':
        # Fine-tuning: use only the Base block of the current session
        prefix_online = f'{prefix}_Sess{session_num:02}'
        suffix        = f'{nclass}class_Base'
        pattern       = os.path.join(subject_folder, f'{prefix_online}*{suffix}')
        data_paths    = sorted(glob.glob(pattern))
    else:
        # Orig training: start with all offline (no session suffix) data
        offline_pattern = os.path.join(subject_folder, prefix)
        data_paths = sorted(glob.glob(offline_pattern))

        # Then append every prior online session (sessions 1 … session_num-1)
        for session in range(1, session_num):
            prefix_online   = f'{prefix}_Sess{session:02}'
            online_pattern  = os.path.join(subject_folder, f'{prefix_online}*')
            data_paths.extend(sorted(glob.glob(online_pattern)))

    return data_paths


def generate_eval_paths(subj_id, task, nclass, session_num, model_type, data_folder):
    """
    Builds the list of directory paths containing the EVALUATION (online test)
    data files for a given subject, task, session, and model type.

    Unlike generate_paths (which assembles training data), this function always
    targets the online evaluation block of the CURRENT session only.

    Parameters
    ----------
    subj_id : int
        Subject identifier (1–21).

    task : str
        'MI' → prefix 'OnlineImagery', anything else → 'OnlineMovement'.

    nclass : int
        Number of classes (2 or 3). Used to build the suffix of the folder name.

    session_num : int
        Current online session number (1–5).

    model_type : str
        'Finetune' → looks for a folder ending in '{nclass}class_Finetune'.
        Anything else ('Orig') → looks for a folder ending in '{nclass}class_Base'.

    data_folder : str
        Root directory containing one sub-folder per subject.

    Returns
    -------
    data_paths : list of str
        Sorted list of directory paths matching the constructed glob pattern.
        Typically contains exactly one folder (one online evaluation block).
    """
    subject_folder = os.path.join(data_folder, f'S{subj_id:02}')

    # Online data always uses the 'Online' prefix
    if task == 'MI':
        prefix = 'OnlineImagery'
    else:
        prefix = 'OnlineMovement'

    prefix_online = f'{prefix}_Sess{session_num:02}'

    if model_type == 'Finetune':
        suffix  = f'{nclass}class_Finetune'
    else:
        suffix  = f'{nclass}class_Base'

    pattern    = os.path.join(subject_folder, f'{prefix_online}*{suffix}')
    data_paths = sorted(glob.glob(pattern))

    return data_paths


def load_and_filter_data(data_paths, params):
    """
    Loads raw EEG .mat files from disk, extracts trial epochs, applies Common
    Average Reference (CAR), and filters to keep only the relevant finger classes.

    Each .mat file contains one recording block. The function iterates over all
    files in all provided directories, extracts trials based on 'Target' and
    'TrialEnd' events, pads every trial to a fixed duration (maxtriallen seconds)
    using NaN, and concatenates everything into a single dataset.

    Parameters
    ----------
    data_paths : list of str
        List of directory paths returned by generate_paths or generate_eval_paths.
        Each directory may contain one or more .mat files.

    params : dict
        Configuration dictionary. Must contain:
        - 'nclass'      (int)   : 2 or 3 — determines which finger classes to keep.
        - 'maxtriallen' (float) : maximum trial length in seconds. Trials shorter
                                  than this are NaN-padded; longer ones are clipped.
        The key 'srate' is written INTO params during loading (read from the .mat file).

    Returns
    -------
    data : np.ndarray, shape (nTrials, nChannels, maxtriallen * srate)
        Epoched EEG data. Trials are NaN-padded to uniform length. Only trials
        belonging to keep_labels are returned (after filter_and_relabel).

    label : np.ndarray, shape (nTrials,)
        1-based class labels after remapping:
        - 2-class: thumb→1, pinky→2
        - 3-class: thumb→1, index→2, pinky→3

    params : dict
        The input params dict with 'srate' now populated from the last loaded file.

    Notes
    -----
    - CAR (Common Average Reference) is applied per trial: each sample has the
      mean across channels subtracted, which removes common-mode noise.
    - NaN padding (not zero padding) is used so that segment_data can later
      detect and discard incomplete segments that overlap the padding region.
    """
    label = []  # will hold per-trial class labels
    data  = []  # will hold per-trial EEG arrays

    # Define which original finger labels to keep and how to remap them
    if params['nclass'] == 2:
        keep_labels = [1, 4]           # thumb, pinky
        new_labels  = {1: 1, 4: 2}
    elif params['nclass'] == 3:
        keep_labels = [1, 2, 4]        # thumb, index, pinky
        new_labels  = {1: 1, 2: 2, 4: 3}
    else:
        raise ValueError("nclass must be either 2 or 3.")

    for filepath in data_paths:
        for filename in sorted(os.listdir(filepath)):
            cur_data = []

            file_path = os.path.join(filepath, filename)
            print(f"Processing file: {file_path}")

            mat   = scipy.io.loadmat(file_path)
            eeg   = mat['eeg']
            event = mat['event']

            # signals: shape (nChannels, nTotalSamples) — continuous raw EEG
            signals = eeg['data'][0][0]
            # srate: sampling rate in Hz (e.g. 512), stored for downstream use
            params['srate'] = eeg['fsample'][0][0][0][0]

            start_idx, end_idx, target = [], [], []

            # Parse the event structure to find trial boundaries.
            # 'Target' event marks the onset of a trial (cue shown to subject).
            # 'TrialEnd' event marks the end of the response window.
            for i in range(event.shape[1]):
                evt        = event[0, i]
                event_type = evt['type'][0]
                sample     = evt['sample'][0][0]   # sample index (1-based in MATLAB)
                value      = evt['value'][0][0]    # finger class (1–5)

                if event_type == 'Target':
                    start_idx.append(sample - 1)   # convert to 0-based Python index
                    target.append(value)
                elif event_type == 'TrialEnd':
                    end_idx.append(sample - 1)

            cur_label = target  # one label per trial

            for i in range(len(start_idx)):
                # Extract trial epoch from the continuous signal
                tmp = signals[:, int(start_idx[i]):int(end_idx[i])]

                # Clip to maximum allowed trial length
                tmp = tmp[:, :min(np.size(tmp, 1), int(params['maxtriallen'] * params['srate']))]

                # NaN-pad to exactly maxtriallen * srate samples
                # NaN (not 0) so that segment_data can identify and drop padded segments
                tmp = np.pad(
                    tmp,
                    ((0, 0), (0, int(params['maxtriallen'] * params['srate']) - np.size(tmp, 1))),
                    'constant',
                    constant_values=np.nan
                )
                cur_data.append(tmp)

            cur_data = np.array(cur_data)   # shape: (nTrialsInFile, nChannels, nSamples)

            # Common Average Reference (CAR): subtract the mean across channels
            # at each time point. Reduces common noise (e.g. movement artefacts).
            cur_data = cur_data - cur_data.mean(axis=1, keepdims=True)

            data.append(cur_data)
            label.append(cur_label)

    # Concatenate all files/blocks into a single array
    data  = np.concatenate(data, axis=0)    # (nTotalTrials, nChannels, nSamples)
    label = np.concatenate(label, axis=0)
    label = label.flatten()

    print(data.shape)
    print(label.shape)

    # Keep only the relevant finger classes and remap labels
    data, label = filter_and_relabel(data, label, keep_labels, new_labels)

    return data, label, params


def train_models(data, label, save_name, params):
    """
    Trains (or fine-tunes) an EEGNet model on the provided EEG data and saves
    the best checkpoint to disk.

    The full preprocessing pipeline (segmentation → downsampling → bandpass
    filtering → z-score normalization) is applied inside this function before
    training, so the input data should be raw epoched EEG (as returned by
    load_and_filter_data).

    Parameters
    ----------
    data : np.ndarray, shape (nTrials, nChannels, nSamples)
        Raw EEG epochs (NaN-padded, after CAR). Labels are trial-level.

    label : np.ndarray, shape (nTrials,)
        1-based integer class labels (e.g. 1, 2, 3).

    save_name : str
        Full file path (including .h5 extension) where the best model weights
        will be saved by ModelCheckpoint.

    params : dict
        Configuration dictionary. Required keys:
        - 'maxtriallen'    (float) : maximum trial length in seconds.
        - 'windowlen'      (float) : segment length in seconds (e.g. 1.0).
        - 'srate'          (int)   : original sampling rate in Hz (set by load_and_filter_data).
        - 'downsrate'      (int)   : target sampling rate after downsampling (e.g. 100 Hz).
        - 'bandpass_filt'  (list)  : [low_hz, high_hz] for the Butterworth filter (e.g. [4, 40]).
        - 'nclass'         (int)   : number of output classes (2 or 3).
        Optional keys (set automatically if present):
        - 'modelpath'      (str)   : path to a pre-trained .h5 file. If present,
                                     the function fine-tunes that model instead of
                                     training from scratch.
        Keys written by this function:
        - 'dropout_ratio'  (float) : 0.5 for Orig, 0.65 for Finetune.
        - 'epochs'         (int)   : 300 for Orig, 100 for Finetune.
        - 'layers_fine_tune' (int) : number of layers to unfreeze during fine-tuning (12).

    Returns
    -------
    save_name : str
        The same path that was passed in. The best model weights (by val_accuracy)
        have been saved there by ModelCheckpoint.

    Notes on the training pipeline
    --------------------------------
    1. Train/val split (80/20) is done at the TRIAL level before segmentation,
       so no segment from a training trial can appear in validation.
    2. Segmentation uses a 1-second sliding window with step=128 samples,
       generating multiple overlapping samples per trial.
    3. Downsampling reduces the time axis from srate to downsrate Hz.
    4. Bandpass filtering uses a 4th-order Butterworth filter with zero-padding
       (100 samples) on each side to avoid edge artefacts.
    5. Z-score normalization is applied per segment along the time axis.
    6. EEGNet architecture: F1=8, D=2, F2=16, kernLength=32.
    7. EarlyStopping (patience=80) and ReduceLROnPlateau (patience=30) are used.
    8. Fine-tuning freezes all layers except the last 12 and uses lr=1e-4.
    """
    if 'modelpath' in params.keys():
        print(f'Fine-tuning model: {save_name}...')
    else:
        print(f'Training model: {save_name}...')

    
    K.set_image_data_format('channels_last')
    nChan  = np.size(data, axis=1)
    DesiredLen = int(params['windowlen'] * params['downsrate'])  # e.g. 1s * 100Hz = 100 samples
    kernels, chans, samples = 1, nChan, DesiredLen
    batch_size, epochs      = 16, 300
    nTrial = len(data)

    

    X_train, Y_train, X_validate, Y_validate = split_train(data, label, save_name, params, kernels, chans, samples, batch_size, epochs, nTrial, nChan, DesiredLen)

    # -------------------------------------------------------------------------
    # EEGNet model setup
    # -------------------------------------------------------------------------
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    # Higher dropout for fine-tuning to regularize the smaller fine-tune dataset
    if 'modelpath' in params.keys():
        params['dropout_ratio'] = 0.65
    else:
        params['dropout_ratio'] = 0.5

    model = EEGNet(
        nb_classes   = params['nclass'],
        Chans        = chans,
        Samples      = samples,
        dropoutRate  = params['dropout_ratio'],
        kernLength   = 32,    # temporal convolution kernel length (samples)
        F1           = 8,     # number of temporal filters
        D            = 2,     # depth multiplier (spatial filters per temporal filter)
        F2           = 16,    # number of pointwise filters (= F1 * D)
        dropoutType  = 'Dropout'
    )

    model.summary()

    # Callbacks
    # EarlyStopping: stops training if val_loss does not improve for 80 epochs
    callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=80)

    # ReduceLROnPlateau: halves lr if val_loss stagnates for 30 epochs
    callback_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=30)

    # Smaller initial lr for fine-tuning to avoid overwriting pre-trained weights
    if 'modelpath' in params.keys():
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        loss      = 'categorical_crossentropy',
        optimizer = optimizer,
        metrics   = ['accuracy']
    )

    # Save the model weights with the best validation accuracy during training
    checkpointer = ModelCheckpoint(
        filepath       = save_name,
        verbose        = 1,
        monitor        = 'val_accuracy',
        mode           = 'max',
        save_best_only = True
    )

    # Equal weight for all classes (no class imbalance correction here)
    class_weights = {0: 1, 1: 1, 2: 1, 3: 1}

    if 'modelpath' in params.keys():
        # Fine-tuning: load pre-trained weights, freeze early layers
        params['epochs']            = 100
        params['layers_fine_tune']  = 12   # number of layers to keep trainable

        model.load_weights(params['modelpath'])
        model.trainable = True

        num_layers           = len(model.layers)
        num_layers_fine_tune = params['layers_fine_tune']

        # Freeze all layers except the last num_layers_fine_tune
        for model_layer in model.layers[:num_layers - num_layers_fine_tune]:
            print(f"FREEZING LAYER: {model_layer}")
            model_layer.trainable = False
    else:
        params['epochs'] = 300

    model.fit(
        X_train, Y_train,
        batch_size      = batch_size,
        epochs          = params['epochs'],
        verbose         = 2,
        validation_data = (X_validate, Y_validate),
        callbacks       = [checkpointer, callback_es, callback_lr],
        class_weight    = class_weights
    )

    print("Training Finished!")
    print(f"Model saved to {save_name}")
    return save_name


def eval_model(data, label, model_path, params, plot_save_path=None):
    """
    Evaluates a trained EEGNet model on new data using segment-level predictions
    aggregated to trial-level via majority voting.

    The preprocessing pipeline applied here is IDENTICAL to train_models (same
    segment size, step size, downsampling, bandpass filter, z-score), ensuring
    there is no train/eval preprocessing mismatch.

    Parameters
    ----------
    data : np.ndarray, shape (nTrials, nChannels, nSamples)
        Raw EEG epochs (NaN-padded, after CAR), as returned by load_and_filter_data.

    label : np.ndarray, shape (nTrials,)
        Ground-truth 1-based class labels for each trial.

    model_path : str
        Path to the .h5 file containing the saved model weights (output of train_models).

    params : dict
        Configuration dictionary. Required keys (same as train_models):
        - 'windowlen'     (float) : segment length in seconds.
        - 'srate'         (int)   : original sampling rate in Hz.
        - 'downsrate'     (int)   : downsampled rate in Hz.
        - 'bandpass_filt' (list)  : [low_hz, high_hz].
        - 'nclass'        (int)   : number of classes.

    plot_save_path : str or None, optional
        If provided, EEG signal plots (before/after processing) would be saved
        to this directory. Currently disabled (call is commented out).

    Returns
    -------
    accuracy : float
        Trial-level classification accuracy in percent (0–100).
        Computed as: (number of correctly predicted trials / total trials) * 100.
        A trial is "correctly predicted" when the majority-voted class across all
        its segments matches the ground-truth label.

    trial_preds : np.ndarray, shape (nTrials,)
        Predicted class (1-based) for each trial after majority voting.
        Note: if a trial has no valid segments (all NaN), its prediction stays 0
        and will always count as incorrect.

    segment_preds : np.ndarray, shape (nValidSegments,)
        Raw per-segment predicted class (1-based) before majority voting.
        Useful for fine-grained analysis or confusion matrices.

    Notes on the majority voting
    ----------------------------
    For each trial, all segments belonging to that trial (identified via I_eval)
    cast a vote for their predicted class. The class with the most votes wins.
    Ties are broken by np.bincount().argmax(), which returns the smallest class index.
    """
    K.set_image_data_format('channels_last')

    nTrial     = len(data)
    nChan      = np.size(data, axis=1)
    DesiredLen = int(params['windowlen'] * params['downsrate'])
    kernels, chans, samples = 1, nChan, DesiredLen

    X_eval, Y_eval, I_eval = data_eval(data, label, params, kernels, chans, samples, nTrial, nChan, DesiredLen)

    # Rebuild the EEGNet architecture and load the saved weights
    model = EEGNet(
        nb_classes  = params['nclass'],
        Chans       = nChan,
        Samples     = DesiredLen,
        dropoutRate = 0.5,
        kernLength  = 32,
        F1          = 8,
        D           = 2,
        F2          = 16,
        dropoutType = 'Dropout'
    )
    model.load_weights(model_path)
    model.summary()

    probs = model.predict(X_eval, verbose=0) 
 
    # Majority voting par essai
    segment_preds = np.argmax(probs, axis=1) + 1              # 1-indexed
    trial_preds = np.zeros(nTrial, dtype=int)
    for trial_idx in range(nTrial):
        mask = (I_eval == trial_idx)
        if mask.sum() == 0:
            continue
        votes = segment_preds[mask]
        trial_preds[trial_idx] = np.bincount(votes).argmax()  # classe majoritaire
 
    # Online Performance Accuracy
    correct  = np.sum(trial_preds == label)
    accuracy_online = correct / nTrial * 100

    #Validation Online Accuracy
    acc_metric = tf.keras.metrics.CategoricalAccuracy()
    acc_metric.update_state(Y_eval, probs)         
    accuracy = acc_metric.result().numpy() * 100

    print(f'Validation Online Accuracy : {accuracy:.2f}%')
    print(f'Online Performance Accuracy : {accuracy_online:.2f}%')

    return accuracy, I_eval, probs, trial_preds, accuracy_online


def split_train(data, label, save_name, params, kernels, chans, samples, batch_size, epochs, nTrial, nChan, DesiredLen):

    # Random permutation to shuffle trials before splitting
    shuffled_idx = np.random.permutation(nTrial)

    # 80% training / 20% validation split at trial level
    train_percent = 0.8
    train_idx = range(int(train_percent * nTrial))
    train_idx = shuffled_idx[train_idx]
    val_idx   = np.setdiff1d(shuffled_idx, train_idx)

    X_train    = data[train_idx, :, :]
    X_validate = data[val_idx,   :, :]
    Y_train    = label[train_idx]
    Y_validate = label[val_idx]

    # -------------------------------------------------------------------------
    # Preprocessing
    # -------------------------------------------------------------------------
    times      = np.arange(0, params['maxtriallen'], 1 / params['srate'])


    segment_size = int(params['windowlen'] * params['srate'])    # e.g. 1s * 512Hz = 512 samples
    step_size    = 128

    # Segment into overlapping windows (increases sample count, fixed-length input)
    X_train,    Y_train,    I_train    = segment_data(X_train,    Y_train,    segment_size, step_size)
    X_validate, Y_validate, I_validate = segment_data(X_validate, Y_validate, segment_size, step_size)

    # Downsample from srate to downsrate (e.g. 512 Hz → 100 Hz)
    X_train    = resample(X_train,    DesiredLen, t=None, axis=2, window=None, domain='time')
    X_validate = resample(X_validate, DesiredLen, t=None, axis=2, window=None, domain='time')

    # Bandpass filter [4–40 Hz] with zero-padding to reduce edge artefacts
    padding_length  = 100
    padded_train    = np.pad(X_train,    ((0,0),(0,0),(padding_length,padding_length)), 'constant', constant_values=0)
    padded_validate = np.pad(X_validate, ((0,0),(0,0),(padding_length,padding_length)), 'constant', constant_values=0)

    b, a = scipy.signal.butter(4, params['bandpass_filt'], btype='bandpass', fs=params['downsrate'])
    X_train    = scipy.signal.lfilter(b, a, padded_train,    axis=-1)
    X_validate = scipy.signal.lfilter(b, a, padded_validate, axis=-1)

    # Remove the padding added before filtering
    X_train    = X_train[:,    :, padding_length:-padding_length]
    X_validate = X_validate[:, :, padding_length:-padding_length]

    # Z-score normalization per segment along the time axis
    # Makes each segment zero-mean and unit-variance, reducing amplitude differences
    X_train    = scipy.stats.zscore(X_train,    axis=2, nan_policy='omit')
    X_validate = scipy.stats.zscore(X_validate, axis=2, nan_policy='omit')

    # Convert integer labels to one-hot vectors (required by categorical_crossentropy)
    # label - 1 because to_categorical expects 0-based indices
    Y_train    = np_utils.to_categorical(Y_train    - 1)
    Y_validate = np_utils.to_categorical(Y_validate - 1)

    # Reshape to (nSamples, nChannels, nTimepoints, 1) — 'channels_last' format
    X_train    = X_train.reshape(X_train.shape[0],    chans, samples, kernels)
    X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
    
    return X_train, Y_train, X_validate, Y_validate


def data_eval(data, label, params, kernels, chans, samples, nTrial, nChan, DesiredLen):
    
    segment_size = int(params['windowlen'] * params['srate'])
    step_size    = 128
    
    # Segment data; I_eval records which trial each segment came from
    X_eval, Y_eval, I_eval = segment_data(data, label, segment_size, step_size)

    # Downsample
    X_eval = resample(X_eval, DesiredLen, t=None, axis=2, window=None, domain='time')

    # Bandpass filter with edge padding
    padding_length = 100
    padded_eval    = np.pad(X_eval, ((0,0),(0,0),(padding_length,padding_length)), 'constant', constant_values=0)

    b, a   = scipy.signal.butter(4, params['bandpass_filt'], btype='bandpass', fs=params['downsrate'])
    X_eval = scipy.signal.lfilter(b, a, padded_eval, axis=-1)
    X_eval = X_eval[:, :, padding_length:-padding_length]

    # Z-score normalization
    X_eval = scipy.stats.zscore(X_eval, axis=2, nan_policy='omit')
    # One-hot encode labels (needed to match the model output format)
    Y_eval = np_utils.to_categorical(Y_eval - 1)

    # Reshape for EEGNet: (nSegments, nChannels, nTimepoints, 1)
    X_eval = X_eval.reshape(X_eval.shape[0], chans, samples, kernels)

    return X_eval, Y_eval, I_eval


# =============================================================================
# MNE / EEG UTILITIES (shared across ERD, visualization, and ICA scripts)
# =============================================================================

def make_info(srate=1024, n_channels=None):
    """
    Build an MNE Info object with the BioSemi 128-channel montage.

    Parameters
    ----------
    srate : int
        Sampling rate in Hz (default: config.SRATE = 1024).
    n_channels : int or None
        Number of channels to include (default: config.N_CHANNELS = 128).

    Returns
    -------
    info : mne.Info
    ch_names : list of str
    """
    if n_channels is None:
        n_channels = N_CHANNELS
    montage  = mne.channels.make_standard_montage("biosemi128")
    ch_names = montage.ch_names[:n_channels]
    info     = mne.create_info(ch_names=ch_names, sfreq=srate, ch_types="eeg")
    info.set_montage(montage)
    return info, list(ch_names)


def get_folder(subj_id, task, session, nclass, model_type, data_folder=None):
    """
    Resolve the online evaluation data folder path for a given combination.

    Parameters
    ----------
    subj_id    : int   — subject ID (1-21)
    task       : str   — 'MI' or 'ME'
    session    : int   — session number (1 or 2)
    nclass     : int   — number of classes (2 or 3)
    model_type : str   — 'Orig' or 'Finetune'
    data_folder : str or None — root data directory (defaults to config.DATA_FOLDER)

    Returns
    -------
    str — full path to the online evaluation folder
    """
    if data_folder is None:
        data_folder = DATA_FOLDER
    modality    = "Movement" if task == "ME" else "Imagery"
    eval_suffix = "Finetune" if model_type == "Finetune" else "Base"
    return os.path.join(
        data_folder, f"S{subj_id:02}",
        f"Online{modality}_Sess{session:02}_{nclass}class_{eval_suffix}"
    )


def get_offline_folder(subj_id, task, data_folder=None):
    """Resolve the offline source folder for a subject/task pair."""
    if data_folder is None:
        data_folder = DATA_FOLDER
    modality = "Imagery" if task == "MI" else "Movement"
    return os.path.join(data_folder, f"S{subj_id:02}", f"Offline{modality}")


def get_ica_folder(source_folder):
    """Mirror a source folder path to its *_ICA sibling."""
    return source_folder.rstrip("/\\") + "_ICA"


def build_task_vectors(raw, fallback_duration=3.0, smooth_sigma=None):
    """Build task vectors from Target annotations up to the next TrialEnd."""
    if raw.annotations is None or len(raw.annotations) == 0:
        return {}

    sfreq = raw.info["sfreq"]
    n_samples = raw.n_times
    descs = np.asarray(raw.annotations.description, dtype=str)
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    durations = np.asarray(raw.annotations.duration, dtype=float)
    trial_end_onsets = np.sort(onsets[descs == "TrialEnd"])

    present = [name for name in TASK_LABELS if name in set(descs)]
    extras = sorted(
        d for d in set(descs)
        if d not in TASK_LABELS and d != "TrialEnd"
    )
    task_vectors = {}

    for task_name in present + extras:
        vec = np.zeros(n_samples, dtype=float)
        for onset, duration in zip(onsets[descs == task_name],
                                   durations[descs == task_name]):
            if duration <= 0:
                future_ends = trial_end_onsets[trial_end_onsets > onset]
                duration = (
                    future_ends[0] - onset
                    if len(future_ends) else fallback_duration
                )
            start = max(0, int(round(onset * sfreq)))
            end = min(n_samples, int(round((onset + duration) * sfreq)))
            if end <= start:
                end = min(n_samples, start + 1)
            vec[start:end] = 1.0

        if smooth_sigma is not None and np.any(vec):
            from scipy.ndimage import gaussian_filter1d
            vec = gaussian_filter1d(vec, sigma=smooth_sigma)
        task_vectors[task_name] = vec

    return task_vectors


def compute_envelope(signal, sfreq, smooth_hz=None, order=4):
    """
    Smoothed Hilbert amplitude envelope used for ICA-task correlations.

    Matches the ICA correlation visualizer: |hilbert(x)| followed by an optional
    zero-phase Butterworth low-pass filter and clipping of tiny negative
    filtfilt overshoot.
    """
    if smooth_hz is None:
        smooth_hz = ICA_ENVELOPE_SMOOTH_HZ
    env = np.abs(scipy.signal.hilbert(signal))
    nyq = sfreq * 0.5
    if 0 < smooth_hz < nyq and env.size > 3 * order:
        b, a = scipy.signal.butter(order, smooth_hz / nyq, btype="low")
        env = scipy.signal.filtfilt(b, a, env)
        np.maximum(env, 0.0, out=env)
    return env


def build_raw_from_mat_files(mat_files, target_srate=SRATE):
    """
    Concatenate STING .mat files into one MNE Raw object with annotations.

    Signals are converted from microvolts to volts for MNE. Target and TrialEnd
    events are converted to annotations, preserving sample offsets across files.
    """
    all_signals = []
    all_events = []
    sample_offset = 0

    for fpath in mat_files:
        mat = scipy.io.loadmat(fpath)
        eeg = mat["eeg"]
        event = mat["event"]
        signals = eeg["data"][0][0].astype(float)
        srate = int(eeg["fsample"][0][0][0][0])

        if srate != target_srate:
            n_new = int(signals.shape[1] * target_srate / srate)
            signals = scipy.signal.resample(signals, n_new, axis=1)
            sf = target_srate / srate
        else:
            sf = 1.0

        for i in range(event.shape[1]):
            evt = event[0, i]
            etype = str(evt["type"][0]).strip()
            samp = int(evt["sample"][0][0]) - 1

            if etype == "Target":
                try:
                    val = int(evt["value"][0][0])
                except Exception:
                    val = 0
                label = FINGER_LABEL.get(val, f"Target{val}")
                evt_int = EVENT_ID.get(label, 99)
                all_events.append([int(samp * sf) + sample_offset, 0, evt_int])
            elif etype == "TrialEnd":
                all_events.append([
                    int(samp * sf) + sample_offset,
                    0,
                    EVENT_ID["TrialEnd"],
                ])

        all_signals.append(signals)
        sample_offset += signals.shape[1]

    data = np.concatenate(all_signals, axis=1)
    info, _ = make_info(srate=target_srate)
    raw = mne.io.RawArray(data * 1e-6, info, verbose=False)

    if all_events:
        events_arr = np.array(all_events, dtype=int)
        annot = mne.annotations_from_events(
            events_arr,
            sfreq=target_srate,
            event_desc=ID_TO_LABEL,
            verbose=False,
        )
        raw.set_annotations(annot)

    return raw, all_events, ID_TO_LABEL


# =============================================================================
# ICA UTILITIES
# =============================================================================

def find_EMG(ica, inst, threshold=None, l_freq=7, h_freq=45, verbose=None):
    """
    Detect muscle (EMG) ICA components via spectral slope.

    A component whose log-log PSD slope (l_freq to h_freq Hz) exceeds
    `threshold` is flagged as muscle artifact.

    Parameters
    ----------
    ica       : mne.preprocessing.ICA — fitted ICA object
    inst      : mne Raw or Epochs used for source extraction
    threshold : float — slope threshold (default: config.EMG_SLOPE_THRESH = -1)
    l_freq    : float — lower frequency bound (Hz)
    h_freq    : float — upper frequency bound (Hz)

    Returns
    -------
    muscle_indices : list of int
    slopes         : np.ndarray, shape (n_components,)
    """
    if threshold is None:
        threshold = EMG_SLOPE_THRESH
    sources  = ica.get_sources(inst)
    spectrum = sources.compute_psd(fmin=l_freq, fmax=h_freq, picks="misc")
    psds, freqs = spectrum.get_data(return_freqs=True)
    if psds.ndim > 2:
        psds = psds.mean(axis=0)
    slopes = np.polyfit(np.log10(freqs), np.log10(psds).T, 1)[0]
    ica.labels_["muscle"] = [
        idx for idx, slope in enumerate(slopes) if slope > threshold
    ]
    return ica.labels_["muscle"], slopes


def compute_eog_scores(ica, raw_filt, ch_names=None):
    """
    Return one absolute EOG correlation score per ICA component.

    Computes correlations between every fitted ICA component and the frontal
    EOG-proxy channels listed in config.EOG_CH_INDICES (single MNE call,
    threshold left to the caller).
    """
    if ch_names is None:
        ch_names = raw_filt.info["ch_names"]

    eog_ch_list = [ch_names[i] for i in EOG_CH_INDICES if i < len(ch_names)]
    if not eog_ch_list:
        return np.zeros(ica.n_components_)

    try:
        _, scores = ica.find_bads_eog(
            raw_filt,
            ch_name=eog_ch_list,
            threshold=1.0,   # arbitrary; we apply our own threshold downstream
            l_freq=1, h_freq=10,
            reject_by_annotation=True,
            measure="correlation",
            verbose=False,
        )
        scores = np.asarray(scores)
        if scores.ndim == 1:
            return np.abs(scores)
        scores = np.abs(scores)
        if scores.shape[-1] == ica.n_components_:
            return np.max(scores, axis=0)
        if scores.shape[0] == ica.n_components_:
            return np.max(scores, axis=1)
        return np.ravel(scores)[:ica.n_components_]
    except Exception:
        return np.zeros(ica.n_components_)


def apply_ica_threshold(raw, mat_files=None, save_folder=None,
                        eog_threshold=None, emg_slope_thresh=None,
                        n_components=None, method=None,
                        verbose_print=True):
    """
    Canonical ICA artifact removal — same selection method as the interactive
    visualizer. Fits ICA on `raw` (highpass-filtered at 1 Hz for fitting),
    excludes components whose max EOG correlation exceeds `eog_threshold` and
    components whose 7–45 Hz spectral slope exceeds `emg_slope_thresh`, then
    applies the cleaning. Optionally writes per-file <basename>_ICA.mat
    to `save_folder` when `mat_files` is provided.

    Thresholds default to config.EOG_THRESHOLD and config.EMG_SLOPE_THRESH.

    Returns (raw_clean, ica, n_eog, n_muscle).
    """
    if eog_threshold    is None: eog_threshold    = EOG_THRESHOLD
    if emg_slope_thresh is None: emg_slope_thresh = EMG_SLOPE_THRESH
    if n_components     is None: n_components     = ICA_N_COMPONENTS
    if method           is None: method           = ICA_METHOD

    if verbose_print:
        print("\n  Fitting ICA ...")
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        max_iter="auto",
        random_state=42,
        verbose=False,
    )
    raw_filt = raw.copy().filter(1.0, None, verbose=False)
    ica.fit(raw_filt, verbose=False)
    n_comp = ica.n_components_

    eog_scores  = compute_eog_scores(ica, raw_filt)
    eog_indices = [int(i) for i in np.where(eog_scores > eog_threshold)[0]
                   if int(i) < n_comp]

    muscle_raw, _ = find_EMG(
        ica, raw_filt,
        threshold=emg_slope_thresh,
        l_freq=7, h_freq=45,
    )
    muscle_indices = [int(i) for i in muscle_raw if int(i) < n_comp]

    ica.exclude = sorted(set(eog_indices + muscle_indices))
    n_eog    = len(eog_indices)
    n_muscle = len(muscle_indices)

    if verbose_print:
        print(f"  ICA: {len(ica.exclude)} component(s) removed "
              f"(EOG={n_eog}, muscle={n_muscle}): {ica.exclude}")

    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)

    if mat_files and save_folder:
        save_ica_cleaned_mat_files(ica, raw, mat_files, save_folder)

    return raw_clean, ica, n_eog, n_muscle


def save_ica_cleaned_mat_files(ica, raw, mat_files, save_folder):
    """
    Apply the fitted `ica` to each source .mat in `mat_files` and write a
    cleaned copy as <basename>_ICA.mat under `save_folder`.
    """
    os.makedirs(save_folder, exist_ok=True)

    for fpath in mat_files:
        mat     = scipy.io.loadmat(fpath)
        eeg     = mat["eeg"]
        signals = eeg["data"][0][0].astype(float)   # (128, N) µV
        srate   = int(eeg["fsample"][0][0][0][0])

        info_tmp      = raw.info.copy()
        raw_tmp       = mne.io.RawArray(signals * 1e-6, info_tmp, verbose=False)
        raw_tmp_clean = raw_tmp.copy()
        ica.apply(raw_tmp_clean, verbose=False)

        signals_clean = raw_tmp_clean.get_data() * 1e6   # V -> µV

        mat_clean = dict(mat)
        mat_clean["eeg"][0][0]["data"] = signals_clean

        basename  = os.path.splitext(os.path.basename(fpath))[0]
        save_path = os.path.join(save_folder, f"{basename}_ICA.mat")
        scipy.io.savemat(save_path, mat_clean)
        print(f"    Saved: {os.path.basename(save_path)}")

    print(f"  Files saved to: {save_folder}")


def apply_ica_signal(signals_uv, srate,
                     n_components=None,
                     method=None,
                     eog_ch_indices=None,   # accepted for backwards compat; unused
                     eog_threshold=None,
                     emg_slope_thresh=None):
    """
    Apply ICA artifact removal to a raw (n_channels, N) µV signal array.

    Thin wrapper around `apply_ica_threshold` for callers that work on numpy
    arrays. EOG channel indices come from config.EOG_CH_INDICES via
    `compute_eog_scores` — the `eog_ch_indices` parameter is kept for
    backwards compatibility and ignored.

    Returns
    -------
    signals_clean : np.ndarray, shape (n_channels, N) — cleaned signal in µV
    """
    info, _ = make_info(srate=srate)
    raw     = mne.io.RawArray(signals_uv * 1e-6, info, verbose=False)
    raw_clean, _, _, _ = apply_ica_threshold(
        raw,
        mat_files=None, save_folder=None,
        eog_threshold=eog_threshold,
        emg_slope_thresh=emg_slope_thresh,
        n_components=n_components,
        method=method,
        verbose_print=True,
    )
    return raw_clean.get_data() * 1e6


# =============================================================================
# ODS HELPERS (shared by evaluation and reporting scripts)
# =============================================================================

def load_ods_cache(ods_path):
    """Load all sheets of an ODS file into a dict of DataFrames."""
    xl     = pd.ExcelFile(ods_path, engine="odf")
    sheets = {
        name: pd.read_excel(ods_path, engine="odf",
                            sheet_name=name, header=None)
        for name in xl.sheet_names
    }
    print(f"  ODS cache loaded: {list(sheets.keys())}")
    return sheets


def cell_is_missing(cache, sheet_name, row_idx, col_idx):
    """Return True if the cell is absent, NaN, or empty string."""
    if sheet_name not in cache:
        return True
    try:
        val = cache[sheet_name].iloc[row_idx, col_idx]
        return pd.isna(val) or str(val).strip() in ("", "nan")
    except Exception:
        return True


def _update_cache(cache, sheet_name, row_idx, col_idx, value):
    """Update the in-memory ODS cache after a write."""
    if sheet_name not in cache:
        return
    df = cache[sheet_name]
    if row_idx >= len(df):
        n_missing = row_idx - len(df) + 1
        empty = pd.DataFrame(
            [[np.nan] * len(df.columns)] * n_missing,
            columns=df.columns
        )
        cache[sheet_name] = pd.concat([df, empty], ignore_index=True)
        df = cache[sheet_name]
    if col_idx >= len(df.columns):
        for _ in range(col_idx - len(df.columns) + 1):
            df[len(df.columns)] = np.nan
    try:
        df.iloc[:, col_idx] = pd.to_numeric(
            df.iloc[:, col_idx], errors="coerce").astype(float)
    except Exception:
        pass
    df.iloc[row_idx, col_idx] = float(value)


def _get_namespaces(xml_bytes):
    ns = {}
    for prefix, uri in re.findall(
            rb'xmlns(?::([a-zA-Z0-9_-]*))?="([^"]+)"', xml_bytes):
        prefix = prefix.decode() if prefix else ""
        ns[prefix] = uri.decode()
    return ns


def _new_value_cell(TC, VTYPE, VAL, TP, value):
    cell = ET.Element(TC)
    cell.set(VTYPE, "float")
    cell.set(VAL, str(round(value, 4)))
    p = ET.SubElement(cell, TP)
    p.text = str(round(value, 4))
    return cell


def _clone_empty(src, count, REP):
    clone = ET.Element(src.tag)
    if count > 1:
        clone.set(REP, str(count))
    return clone


def _set_cell(row_elem, col_idx, value, ns_map):
    T   = ns_map["table"];  O = ns_map["office"];  TXT = ns_map["text"]
    TC    = f"{{{T}}}table-cell"
    COV   = f"{{{T}}}covered-table-cell"
    REP   = f"{{{T}}}number-columns-repeated"
    VTYPE = f"{{{O}}}value-type"
    VAL   = f"{{{O}}}value"
    TP    = f"{{{TXT}}}p"

    flat = []
    for child in list(row_elem):
        if child.tag in (TC, COV):
            flat.append([child, int(child.get(REP, "1"))])

    cursor = 0; seg_idx = None; offset = 0
    for i, (elem, rep) in enumerate(flat):
        if cursor <= col_idx < cursor + rep:
            seg_idx = i; offset = col_idx - cursor; break
        cursor += rep

    if seg_idx is None:
        existing = sum(r for _, r in flat)
        gap = col_idx - existing
        if gap > 0:
            pad = ET.Element(TC); pad.set(REP, str(gap))
            flat.append([pad, gap])
        flat.append([_new_value_cell(TC, VTYPE, VAL, TP, value), 1])
    else:
        src, rep = flat[seg_idx]
        pieces = []
        if offset > 0:
            pieces.append([_clone_empty(src, offset, REP), offset])
        pieces.append([_new_value_cell(TC, VTYPE, VAL, TP, value), 1])
        after = rep - offset - 1
        if after > 0:
            pieces.append([_clone_empty(src, after, REP), after])
        flat[seg_idx:seg_idx + 1] = pieces

    for child in list(row_elem):
        if child.tag in (TC, COV):
            row_elem.remove(child)
    non_cell = sum(1 for c in row_elem if c.tag not in (TC, COV))
    for i, (elem, _) in enumerate(flat):
        row_elem.insert(non_cell + i, elem)


def _find_table(root, ns_map, table_name):
    T = ns_map["table"]
    for elem in root.iter(f"{{{T}}}table"):
        if elem.get(f"{{{T}}}name") == table_name:
            return elem
    raise RuntimeError(f"Sheet '{table_name}' not found in content.xml")


def _expand_rows(table_elem, ns_map):
    T  = ns_map["table"]
    TR = f"{{{T}}}table-row"
    RR = f"{{{T}}}number-rows-repeated"
    for child in list(table_elem):
        if child.tag == TR:
            rep = int(child.get(RR, "1"))
            if rep > 1:
                child.attrib.pop(RR)
                idx = list(table_elem).index(child)
                for k in range(1, rep):
                    table_elem.insert(idx + k, copy.deepcopy(child))


def write_to_ods(ods_path, row_idx, writes, table_name):
    """
    Write one or more (col_idx, value) pairs to a specific row of an ODS sheet.

    Parameters
    ----------
    ods_path   : str  — path to the .ods file
    row_idx    : int  — 0-based row index
    writes     : list of (col_idx, value) tuples
    table_name : str  — sheet name inside the ODS file
    """
    with zipfile.ZipFile(ods_path, "r") as z:
        content_bytes = z.read("content.xml")

    ns_map = _get_namespaces(content_bytes)
    for prefix, uri in ns_map.items():
        try:
            ET.register_namespace(prefix, uri)
        except Exception:
            pass

    root       = ET.fromstring(content_bytes)
    T          = ns_map["table"]
    TR         = f"{{{T}}}table-row"
    table_elem = _find_table(root, ns_map, table_name)
    _expand_rows(table_elem, ns_map)

    all_rows = [c for c in table_elem if c.tag == TR]
    if row_idx >= len(all_rows):
        raise IndexError(
            f"row_idx={row_idx} out of range in '{table_name}'")

    target_row = all_rows[row_idx]
    for col_idx, value in writes:
        _set_cell(target_row, col_idx, value, ns_map)

    new_content = ET.tostring(root, encoding="unicode")
    new_content = re.sub(r'(=of:)+(?==)',   '=of:',   new_content)
    new_content = re.sub(r'(=oooc:)+(?==)', '=oooc:', new_content)
    if not new_content.startswith("<?xml"):
        new_content = ('<?xml version="1.0" encoding="UTF-8"?>\n'
                       + new_content)

    tmp_path = ods_path + ".tmp"
    with zipfile.ZipFile(ods_path, "r") as zin:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "content.xml":
                    zout.writestr(item, new_content.encode("utf-8"))
                else:
                    zout.writestr(item, zin.read(item.filename))
    shutil.move(tmp_path, ods_path)


# =============================================================================
# CLI HELPERS AND PATH MANIPULATION
# =============================================================================

def prompt_int(prompt, valid=None):
    """Prompt the user for an integer until a valid one is entered."""
    while True:
        try:
            val = int(input(prompt).strip())
            if valid is None or val in valid:
                return val
            print(f"  Please enter one of {valid}")
        except ValueError:
            print("  Please enter a number.")


def prompt_choice(prompt, choices):
    """Prompt the user for a string choice until a valid one is entered."""
    choices_str = " / ".join(choices)
    while True:
        val = input(f"{prompt} [{choices_str}]: ").strip()
        if val in choices:
            return val
        print(f"  Please enter one of: {choices_str}")


def get_standard_args(description="EEG BCI Script"):
    """
    Standardized argument parser for BCI scripts.
    Handles both positional and optional arguments for:
    subj, session, nclass, task, model
    
    Order: <subj> <session> <nclass> <task> <model>
    
    If no arguments are provided, it prompts the user interactively.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("subj",    type=int, nargs="?", default=None)
    p.add_argument("session", type=int, nargs="?", default=None)
    p.add_argument("nclass",  type=int, nargs="?", default=None)
    p.add_argument("task",    type=str, nargs="?", default=None)
    p.add_argument("model",   type=str, nargs="?", default=None)
    
    p.add_argument("--subj",    type=int, dest="opt_subj",    default=None)
    p.add_argument("--session", type=int, dest="opt_session", default=None)
    p.add_argument("--nclass",  type=int, dest="opt_nclass",  default=None)
    p.add_argument("--task",    type=str, dest="opt_task",    default=None)
    p.add_argument("--model",   type=str, dest="opt_model",   default=None)
    
    args = p.parse_args()
    
    subj    = args.subj    or args.opt_subj
    session = args.session or args.opt_session
    nclass  = args.nclass  or args.opt_nclass
    task    = args.task    or args.opt_task
    model   = args.model   or args.opt_model
    
    # If any is missing, prompt interactively
    if subj is None:
        subj = prompt_int("Subject ID [1-21]: ", valid=list(range(1, 22)))
    if session is None:
        session = prompt_int("Session [1-5]: ", valid=list(range(1, 6)))
    if nclass is None:
        nclass = prompt_int("Number of classes [2 / 3]: ", valid=[2, 3])
    if task is None:
        task = prompt_choice("Task", ["MI", "ME"])
    if model is None:
        model = prompt_choice("Model type", ["Orig", "Finetune"])
        
    return subj, session, nclass, task, model


def get_offline_args(description="Offline EEG BCI Script"):
    """
    Standardized argument parser for OFFLINE scripts.
    Handles: <subj> <task>
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument("subj", type=int, nargs="?", default=None)
    p.add_argument("task", type=str, nargs="?", default=None)
    
    p.add_argument("--subj", type=int, dest="opt_subj", default=None)
    p.add_argument("--task", type=str, dest="opt_task", default=None)
    
    args = p.parse_args()
    
    subj = args.subj or args.opt_subj
    task = args.task or args.opt_task
    
    if subj is None:
        subj = prompt_int("Subject ID [1-21]: ", valid=list(range(1, 22)))
    if task is None:
        task = prompt_choice("Task", ["MI", "ME"])
        
    return subj, task


def to_ica_path(orig_path, suffix="_ICA"):
    """
    Mirror an original path to its artifact counterpart (suffix added to folder/basename).
    """
    if os.path.isdir(orig_path):
        return orig_path.rstrip("/\\") + suffix
    base, ext = os.path.splitext(orig_path)
    return base + suffix + ext


def get_ica_fif_path(subj_id, task, session=0, nclass=0, model_type="Orig",
                     ica_folder=None):
    """
    Return the path to the cached ICA .fif file for a given recording.

    Offline recordings (session=0 or model_type='Offline') get a dedicated
    filename so they never collide with online sessions.

    Parameters
    ----------
    subj_id    : int   — subject ID (1-21)
    task       : str   — 'MI' or 'ME'
    session    : int   — online session number (0 = offline)
    nclass     : int   — number of classes (0 = offline)
    model_type : str   — 'Orig', 'Finetune', or 'Offline'
    ica_folder : str or None — override storage directory (defaults to
                               config.ICA_FIF_FOLDER)

    Returns
    -------
    str — full path ending in '-ica.fif' (MNE convention)
    """
    if ica_folder is None:
        ica_folder = ICA_FIF_FOLDER
    os.makedirs(ica_folder, exist_ok=True)

    if model_type == "Offline" or session == 0:
        fname = f"S{subj_id:02}_{task}_Offline-ica.fif"
    else:
        fname = f"S{subj_id:02}_Sess{session:02}_{task}_{nclass}class_{model_type}-ica.fif"

    return os.path.join(ica_folder, fname)


# =============================================================================
# INTERACTIVE ICA VISUALIZATION (Moved from ICA_Interactive_Visualizer.py)
# =============================================================================

def compute_slopes(ica, raw_filt, fmin=7, fmax=45):
    sources  = ica.get_sources(raw_filt)
    spectrum = sources.compute_psd(fmin=fmin, fmax=fmax, picks="misc")
    psds, freqs = spectrum.get_data(return_freqs=True)
    if psds.ndim > 2:
        psds = psds.mean(axis=0)
    return np.polyfit(np.log10(freqs), np.log10(psds).T, 1)[0]


def compute_exclusions(ica, raw_filt, muscle_thresh, eog_thresh, ch_names,
                       manual_extra=None, eog_scores=None, emg_scores=None):
    n_comp = ica.n_components_
    if emg_scores is None:
        _, emg_scores = find_EMG(ica, raw_filt, threshold=muscle_thresh,
                                 l_freq=7, h_freq=45)
    if eog_scores is None:
        eog_scores = compute_eog_scores(ica, raw_filt, ch_names)

    emg_scores = np.asarray(emg_scores)
    eog_scores = np.asarray(eog_scores)
    eog_indices    = [int(i) for i in np.where(eog_scores > eog_thresh)[0]
                      if int(i) < n_comp]
    muscle_indices = np.where(emg_scores > muscle_thresh)[0].tolist()
    extra    = [i for i in (manual_extra or []) if i < n_comp]
    all_excl = list(set(eog_indices + muscle_indices + extra))
    return all_excl, eog_indices, muscle_indices


class TopoHighlighter:
    def __init__(self, info, ch_names):
        self.info     = info
        self.ch_names = ch_names
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.fig.patch.set_facecolor("white")
        self.fig.canvas.manager.set_window_title("Electrode Map")
        self._draw_base()
        plt.show(block=False)
        self.fig.canvas.draw()

    def _draw_base(self, highlighted_idx=None):
        self.ax.clear()
        values = np.zeros(len(self.ch_names))
        if highlighted_idx is not None:
            values[highlighted_idx] = 1.0
        mne.viz.plot_topomap(
            values, self.info, axes=self.ax,
            cmap="Reds", vlim=(0, 1), contours=0,
            extrapolate="head", sphere=(0., 0., 0., 0.095),
            outlines="head", show=False, sensors=True)
        if highlighted_idx is not None:
            ch_name = self.ch_names[highlighted_idx]
            try:
                pos2d = mne.channels.layout._find_topomap_coords(
                    self.info, picks=[highlighted_idx])
                x, y = pos2d[0]
                self.ax.plot(x, y, "o", color="red", markersize=14,
                             markeredgecolor="darkred",
                             markeredgewidth=2, zorder=10)
                self.ax.annotate(
                    ch_name, xy=(x, y),
                    xytext=(x + 0.02, y + 0.02),
                    fontsize=11, fontweight="bold", color="darkred",
                    bbox=dict(boxstyle="round,pad=0.2",
                              fc="white", ec="darkred", alpha=0.85),
                    zorder=11)
            except Exception:
                pass
            self.ax.set_title(f"Selected: {ch_name}",
                              fontsize=12, fontweight="bold",
                              color="darkred")
        else:
            self.ax.set_title("Click a channel in the EEG viewer",
                              fontsize=11, color="gray")

    def highlight(self, ch_name):
        ch_name = ch_name.strip()
        if ch_name not in self.ch_names:
            return
        idx = self.ch_names.index(ch_name)
        print(f"  -> Electrode: {ch_name}  (index {idx})")
        self._draw_base(highlighted_idx=idx)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def find_file_pairs_offline(subj_id, task):
    """Return original and cleaned offline file lists (using _ICA suffix)."""
    raw_folder = get_offline_folder(subj_id, task)
    clean_folder = get_ica_folder(raw_folder)
    raw_files = sorted(glob.glob(os.path.join(raw_folder, "*.mat")))
    if not raw_files: raise FileNotFoundError(f"No offline .mat files in: {raw_folder}")
    if not os.path.isdir(clean_folder): raise FileNotFoundError(f"Offline ICA folder missing: {clean_folder}")
    clean_files = []
    for raw_path in raw_files:
        basename = os.path.splitext(os.path.basename(raw_path))[0]
        clean_path = os.path.join(clean_folder, f"{basename}_ICA.mat")
        if os.path.exists(clean_path): clean_files.append(clean_path)
    return raw_folder, clean_folder, raw_files, clean_files


def load_offline_raw_pair(subj_id, task):
    """Load original and ICA offline files as Raw pair."""
    raw_folder, clean_folder, raw_files, clean_files = find_file_pairs_offline(subj_id, task)
    raw, _, _ = build_raw_from_mat_files(raw_files)
    raw_clean, _, _ = build_raw_from_mat_files(clean_files)
    return raw, raw_clean


def find_file_pairs_online(subj_id, task, session, nclass, model_type):
    """Return original and cleaned online file lists (using _ICA suffix)."""
    raw_folder = get_folder(subj_id, task, session, nclass, model_type)
    clean_folder = get_ica_folder(raw_folder)
    raw_files = sorted(glob.glob(os.path.join(raw_folder, "*.mat")))
    if not raw_files: raise FileNotFoundError(f"No online .mat files in: {raw_folder}")
    if not os.path.isdir(clean_folder): raise FileNotFoundError(f"Online ICA folder missing: {clean_folder}")
    clean_files = []
    for raw_path in raw_files:
        basename = os.path.splitext(os.path.basename(raw_path))[0]
        clean_path = os.path.join(clean_folder, f"{basename}_ICA.mat")
        if os.path.exists(clean_path): clean_files.append(clean_path)
    return raw_folder, clean_folder, raw_files, clean_files


def load_online_raw_pair(subj_id, task, session, nclass, model_type):
    """Load original and ICA online files as Raw pair."""
    raw_folder, clean_folder, raw_files, clean_files = find_file_pairs_online(subj_id, task, session, nclass, model_type)
    raw, _, _ = build_raw_from_mat_files(raw_files)
    raw_clean, _, _ = build_raw_from_mat_files(clean_files)
    return raw, raw_clean


def launch_viewer(raw, raw_clean, subj_id, task, session=0, nclass=0, model_type="Orig", event_id=None, ica=None):
    info, ch_names = make_info()
    if event_id is None: event_id = {"Thumb": 1, "Index": 2, "Middle": 3, "Pinky": 4, "TrialEnd": 9}
    model_label = "Base Model" if model_type == "Orig" else "Fine-tuned Model"
    event_color = {event_id["Thumb"]: "green", event_id["Index"]: "blue", event_id["Middle"]: "orange", event_id["Pinky"]: "red", event_id["TrialEnd"]: "gray"}
    common_kwargs = dict(n_channels=20, duration=10.0, scalings=dict(eeg=50e-6), show_scrollbars=True, show_options=True, block=False, overview_mode="channels", color=dict(eeg="steelblue"), event_color=event_color)
    print("\nOpening viewers (Left: RAW, Right: ICA)...")
    topo = TopoHighlighter(info, ch_names)
    fig_raw = raw.plot(title=f"RAW - S{subj_id:02} | {task} | Sess{session:02} | {model_label}", **common_kwargs)
    clean_title = f"ICA CLEANED"
    if ica: clean_title += f" ({len(ica.exclude)} components removed)"
    clean_title += f" - S{subj_id:02} | {task} | Sess{session:02} | {model_label}"
    fig_clean = raw_clean.plot(title=clean_title, **common_kwargs)
    def connect(fig):
        def on_pick(event):
            artist = event.artist
            if hasattr(artist, "get_text") and artist.get_text().strip() in ch_names: topo.highlight(artist.get_text().strip())
        def on_click(event):
            if event.inaxes:
                for artist in event.inaxes.get_children():
                    if hasattr(artist, "get_text") and hasattr(artist, "get_position"):
                        try:
                            tx, ty = artist.get_position()
                            if abs(event.ydata - ty) < 0.5 and event.xdata < 0 and artist.get_text().strip() in ch_names:
                                topo.highlight(artist.get_text().strip()); break
                        except Exception: pass
        fig.canvas.mpl_connect("pick_event", on_pick); fig.canvas.mpl_connect("button_press_event", on_click)
    connect(fig_raw); connect(fig_clean); plt.show(block=True)


def interactive_ica_setup(raw_filt, ica, slopes, raw_orig, subj_id,
                           task, session, nclass, model_type, manual_extra=None):
    ch_names     = raw_filt.info["ch_names"]
    n_comp       = ica.n_components_
    sfreq        = raw_filt.info["sfreq"]
    manual_extra = []
    src_data = None
    eog_scores = None
    times = np.array([0.0])
    src_norm = np.zeros((n_comp, 1))
    event_colors = {
        "Thumb": "green",
        "Index": "blue",
        "Middle": "orange",
        "Pinky": "red",
        "TrialEnd": "gray",
    }
    N_PER_PAGE_DEFAULT = 32
    state = {
        "eog_thresh":    0.7,
        "muscle_thresh": -1.0,
        "page":          0,
        "n_visible":     N_PER_PAGE_DEFAULT,
        "last_params":   None,
        "excl":          [],
        "eog_i":         [],
        "mus_i":         [],
        "time_start":    0.0,
        "time_window":   10.0,
        "y_scale":       1.0,
        "times_disp":    times,
        "src_norm_disp": src_norm,
    }
    KEEP_COLOR = "#2C7BB6"
    DROP_COLOR = "#D6273F"
    ica.exclude = manual_extra.copy()
    n_pages = int(np.ceil(n_comp / N_PER_PAGE_DEFAULT))
    fig = plt.figure(figsize=(20, 10))
    fig.canvas.manager.set_window_title(
        f"ICA Inspector - S{subj_id:02} | {task} {nclass}-class | "
        f"Session {session}"
    )
    fig.patch.set_facecolor("#F5F5F5")

    def bring_to_front():
        manager = fig.canvas.manager
        window = getattr(manager, "window", None)
        if window is None: return
        try:
            window.deiconify(); window.lift(); window.focus_force(); window.attributes("-topmost", True)
            window.after(250, lambda: window.attributes("-topmost", False))
        except Exception: pass

    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 3], left=0.12, right=0.97, top=0.95, bottom=0.04, wspace=0.08)
    left_gs = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=outer[0], hspace=0.6, height_ratios=[2.5, 0.5, 0.5, 0.5, 1.2, 0.8])
    ax_hist, ax_eog_sl, ax_mus_sl, ax_time_sl, ax_info, ax_btn_area = [fig.add_subplot(left_gs[i]) for i in range(6)]
    right_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1], hspace=0.04, height_ratios=[1, 0.06])
    ax_sources, ax_nav = fig.add_subplot(right_gs[0]), fig.add_subplot(right_gs[1])
    sl_eog = Slider(ax_eog_sl, "EOG thresh", 0.1, 1.0, valinit=state["eog_thresh"], valstep=0.05, color=KEEP_COLOR)
    sl_mus = Slider(ax_mus_sl, "EMG thresh", -3.0, 0.5, valinit=-1.0, valstep=0.1, color=KEEP_COLOR)
    sl_time = Slider(ax_time_sl, "Time window (s)", 0.0, max(0.1, raw_filt.times[-1] - state["time_window"]), valinit=0.0, valstep=1.0, color="#7F8C8D")
    ax_btn_area.axis("off"); btn_pos = ax_btn_area.get_position()
    ax_confirm = fig.add_axes([btn_pos.x0 + btn_pos.width * 0.1, btn_pos.y0, btn_pos.width * 0.8, btn_pos.height * 0.8])
    btn_confirm = Button(ax_confirm, "CONFIRM", color="#27AE60", hovercolor="#2ECC71")
    btn_confirm.label.set_fontsize(12); btn_confirm.label.set_fontweight("bold"); btn_confirm.label.set_color("white")
    ax_nav.axis("off"); nav_pos = ax_nav.get_position()
    ax_pg_sl = fig.add_axes([nav_pos.x0 + nav_pos.width * 0.3, nav_pos.y0, nav_pos.width * 0.4, nav_pos.height])
    sl_page = Slider(ax_pg_sl, "Page", 0, max(1, n_pages - 1), valinit=0, valstep=1, color="#95A5A6")
    threshold_after_id = None
    active_slider = {"name": None}

    def current_n_pages(): return int(np.ceil(n_comp / state["n_visible"]))
    def refresh_page_slider():
        pages = current_n_pages(); state["page"] = min(state["page"], pages - 1)
        sl_page.valmax = max(1, pages - 1); ax_pg_sl.set_xlim(sl_page.valmin, sl_page.valmax)
    def on_page_change(val): state["page"] = min(int(round(val)), current_n_pages() - 1); draw_info(); fig.canvas.draw_idle()
    def set_time_start(val, redraw=True):
        max_start = max(0.0, raw_filt.times[-1] - state["time_window"])
        state["time_start"] = min(max(float(val), 0.0), max_start)
        sl_time.eventson = False; sl_time.set_val(state["time_start"]); sl_time.eventson = True
        if redraw: _reload_src_data(); draw_sources(); fig.canvas.draw_idle()
    def refresh_time_slider():
        max_start = max(0.1, raw_filt.times[-1] - state["time_window"])
        sl_time.valmax = max_start; ax_time_sl.set_xlim(sl_time.valmin, max_start)
    def set_time_window(val): state["time_window"] = min(max(float(val), 1.0), 60.0); refresh_time_slider(); set_time_start(state["time_start"], redraw=True)
    def on_time_change(val): state["time_start"] = float(val); draw_info(); fig.canvas.draw_idle()
    def set_page(page_num):
        state["page"] = max(0, min(int(page_num), current_n_pages() - 1))
        sl_page.eventson = False
        sl_page.set_val(state["page"])
        sl_page.eventson = True
        draw_sources(); draw_info(); fig.canvas.draw_idle()

    def set_visible_sources(n):
        state["n_visible"] = max(4, min(int(n), n_comp))
        refresh_page_slider(); draw_sources(); draw_info(); fig.canvas.draw_idle()

    sl_page.on_changed(on_page_change); sl_time.on_changed(on_time_change)

    ax_info.axis("off"); ax_info.text(0.05, 0.95, "Preparing ICA inspector...\nComputing initial exclusions can take a moment.", transform=ax_info.transAxes, fontsize=10, verticalalignment="top", fontfamily="monospace", bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC", alpha=0.9))
    fig.canvas.draw(); plt.show(block=False); plt.pause(0.1); bring_to_front()

    def _reload_src_data():
        nonlocal src_data
        if src_data is None:
            print("  Extracting ICA sources for preview ...", flush=True)
            src_data = ica.get_sources(raw_orig).get_data()
        t0, width = state.get("time_start", 0.0), state.get("time_window", 10.0)
        start = int(t0 * sfreq); end = min(start + int(width * sfreq), src_data.shape[1]); chunk = src_data[:, start:end]; n_pts = chunk.shape[1]
        state["times_disp"] = np.arange(n_pts) / sfreq + t0; state["src_norm_disp"] = chunk.copy()
        for i in range(n_comp):
            state["src_norm_disp"][i] -= np.median(state["src_norm_disp"][i])
            rng = np.ptp(state["src_norm_disp"][i])
            if rng > 0: state["src_norm_disp"][i] /= rng

    def draw_events(ax, annotate=False):
        if raw_orig.annotations is None: return
        t0, t1 = float(state["times_disp"][0]), float(state["times_disp"][-1]); trans = ax.get_xaxis_transform()
        for onset, desc in zip(raw_orig.annotations.onset, raw_orig.annotations.description):
            onset = float(onset)
            if onset < t0 or onset > t1: continue
            desc = str(desc); color = event_colors.get(desc, "black"); style = "--" if desc == "TrialEnd" else "-"
            ax.axvline(onset, color=color, linestyle=style, linewidth=0.8, alpha=0.35, zorder=0)
            if annotate: ax.text(onset, 0.98, desc, transform=trans, fontsize=6, color=color, rotation=90, va="top", ha="right", alpha=0.85)

    def update_source_layout():
        bbox = outer[1].get_position(fig); nav_h = bbox.height * 0.055; gap = bbox.height * 0.004
        label_w = min(0.16, bbox.width * 0.18); src_x0 = bbox.x0 + label_w
        ax_nav.set_position([bbox.x0, bbox.y0, bbox.width, nav_h])
        ax_sources.set_position([src_x0, bbox.y0 + nav_h + gap, bbox.width - label_w, bbox.height - nav_h - gap])
        nav_pos = ax_nav.get_position(); ax_pg_sl.set_position([nav_pos.x0 + nav_pos.width * 0.3, nav_pos.y0, nav_pos.width * 0.4, nav_pos.height])

    def style_threshold_slider(slider, ax, bad_scores, thresh, higher_is_bad):
        ax.set_facecolor("#F2F2F2")
        for patch in getattr(ax, "_classification_spans", []):
            try: patch.remove()
            except Exception: pass
        ax._classification_spans = []
        lo, hi = slider.valmin, slider.valmax
        keep_span = ax.axvspan(lo, thresh, ymin=0.36, ymax=0.64, color=KEEP_COLOR, alpha=0.18, zorder=0)
        drop_span = ax.axvspan(thresh, hi, ymin=0.36, ymax=0.64, color=DROP_COLOR, alpha=0.18, zorder=0)
        ax._classification_spans = [keep_span, drop_span]
        bad_count = int(np.sum(np.asarray(bad_scores) >= thresh)) if higher_is_bad else int(np.sum(np.asarray(bad_scores) > thresh))
        ratio = min(max(bad_count / n_comp, 0.0), 1.0)
        fill = DROP_COLOR if ratio >= 0.5 else KEEP_COLOR
        try: slider.poly.set_facecolor(fill); slider.poly.set_alpha(0.9); slider.track.set_facecolor("#D8D8D8"); slider.track.set_alpha(0.45)
        except Exception: pass

    def style_sliders():
        if eog_scores is not None: style_threshold_slider(sl_eog, ax_eog_sl, eog_scores, state["eog_thresh"], higher_is_bad=True)
        style_threshold_slider(sl_mus, ax_mus_sl, slopes, state["muscle_thresh"], higher_is_bad=True)

    def draw_histogram():
        ax_hist.clear(); valid_mus = [i for i in state.get("mus_i", []) if i < len(slopes)]
        muscle_mask = np.zeros(len(slopes), dtype=bool); muscle_mask[valid_mus] = True
        kept_slopes, muscle_slopes = slopes[~muscle_mask], slopes[muscle_mask]
        ax_hist.hist(kept_slopes, bins=25, color=KEEP_COLOR, edgecolor="white", linewidth=0.4, alpha=0.85, label="Kept by EMG")
        if len(muscle_slopes): ax_hist.hist(muscle_slopes, bins=25, color=DROP_COLOR, edgecolor="white", linewidth=0.4, alpha=0.7, label="Removed by EMG")
        ax_hist.axvline(state["muscle_thresh"], color=DROP_COLOR, linestyle="--", linewidth=1.4, alpha=0.85, label=f"EMG {state['muscle_thresh']:.1f}")
        ax_hist.set_xlabel("Spectral Slope (7-45 Hz)", fontsize=8); ax_hist.set_ylabel("Count", fontsize=8); ax_hist.set_title("Spectral Slope Distribution", fontsize=9, fontweight="bold")
        ax_hist.legend(fontsize=7, loc="upper left"); ax_hist.tick_params(labelsize=7); ax_hist.grid(axis="y", alpha=0.3); style_sliders()

    def draw_info():
        ax_info.clear(); ax_info.axis("off"); excl, eog_i, mus_i = state.get("excl", []), state.get("eog_i", []), state.get("mus_i", [])
        lines = [f"Total excluded : {len(excl)}  / {n_comp}", f"  EOG          : {len(eog_i)}", f"  Muscle       : {len(mus_i)}", f"  Manual extra : {len(manual_extra)}", "", f"Page {state['page']+1} / {current_n_pages()}", f"Visible sources: {state['n_visible']}", f"Components {state['page']*state['n_visible']}-{min((state['page']+1)*state['n_visible'], n_comp)-1}", f"Time window    : {state['time_window']:.1f} s", f"Vertical zoom  : {state['y_scale']:.2f}x"]
        ax_info.text(0.05, 0.95, "\n".join(lines), transform=ax_info.transAxes, fontsize=9, verticalalignment="top", fontfamily="monospace", bbox=dict(boxstyle="round", fc="white", ec="#CCCCCC", alpha=0.8))

    def draw_sources():
        page, n_vis = state["page"], state["n_visible"]; start_i = page * n_vis; end_i = min(start_i + n_vis, n_comp); shown = end_i - start_i
        update_source_layout(); ax_sources.clear(); ax_sources.set_facecolor("white"); ax_sources.patch.set_zorder(-10)
        if shown <= 0: ax_sources.axis("off"); return
        times_d, src_norm_d = state["times_disp"], state["src_norm_disp"]; offsets = np.arange(shown - 1, -1, -1, dtype=float); tick_labels, tick_colors = [], []
        ax_sources.set_xlim(times_d[0], times_d[-1]); ax_sources.set_ylim(-0.75, shown - 0.25)
        for ax_idx, comp_i in enumerate(range(start_i, end_i)):
            offset = offsets[ax_idx]; is_excl = comp_i in ica.exclude
            if is_excl: ax_sources.axhspan(offset - 0.45, offset + 0.45, color="#FCECEE", zorder=-5)
        draw_events(ax_sources, annotate=True); eog_set, mus_set, manual_set = set(state.get("eog_i", [])), set(state.get("mus_i", [])), set(manual_extra)
        for ax_idx, comp_i in enumerate(range(start_i, end_i)):
            offset = offsets[ax_idx]; is_excl = comp_i in ica.exclude; color, alpha = (DROP_COLOR, 0.55) if is_excl else (KEEP_COLOR, 0.9)
            ax_sources.plot(times_d, offset + src_norm_d[comp_i] * state["y_scale"] * 0.35, color=color, linewidth=0.6 if is_excl else 0.8, alpha=alpha)
            label = f"ICA{comp_i:03d}"
            if is_excl:
                if comp_i in manual_set: label += " [manual]"
                elif comp_i in eog_set and comp_i in mus_set: label += " [EOG+muscle]"
                elif comp_i in eog_set: label += " [EOG]"
                elif comp_i in mus_set: label += " [muscle]"
            tick_labels.append(label); tick_colors.append(DROP_COLOR if is_excl else "#1A1A1A")
        ax_sources.set_yticks(offsets); ax_sources.set_yticklabels(tick_labels, fontsize=7)
        for tick, color in zip(ax_sources.get_yticklabels(), tick_colors): tick.set_color(color); tick.set_horizontalalignment("right"); tick.set_clip_on(False)
        ax_sources.set_xlabel("Time (s)", fontsize=7); ax_sources.tick_params(axis="x", labelsize=6); ax_sources.tick_params(axis="y", length=0, pad=10)
        ax_sources.grid(axis="x", alpha=0.15); [s.set_visible(False) for s in [ax_sources.spines["top"], ax_sources.spines["right"], ax_sources.spines["left"]]]

    def full_redraw():
        nonlocal eog_scores, threshold_after_id
        threshold_after_id = None; params = (state["muscle_thresh"], state["eog_thresh"])
        if state.get("last_params") != params:
            if eog_scores is None: print("  Computing EOG correlation scores once ...", flush=True); eog_scores = compute_eog_scores(ica, raw_filt, ch_names)
            excl, eog_i, mus_i = compute_exclusions(ica, raw_filt, state["muscle_thresh"], state["eog_thresh"], ch_names, manual_extra, eog_scores=eog_scores, emg_scores=slopes)
            state["excl"], state["eog_i"], state["mus_i"], state["last_params"], ica.exclude = excl, eog_i, mus_i, params, excl
        draw_histogram(); draw_info(); draw_sources(); fig.canvas.draw_idle()

    def histogram_only_redraw():
        nonlocal eog_scores; params = (state["muscle_thresh"], state["eog_thresh"])
        if state.get("last_params") != params:
            if eog_scores is None: eog_scores = compute_eog_scores(ica, raw_filt, ch_names)
            excl, eog_i, mus_i = compute_exclusions(ica, raw_filt, state["muscle_thresh"], state["eog_thresh"], ch_names, manual_extra, eog_scores=eog_scores, emg_scores=slopes)
            state["excl"], state["eog_i"], state["mus_i"], ica.exclude = excl, eog_i, mus_i, excl
        draw_histogram(); draw_info(); fig.canvas.draw_idle()

    def on_eog_change(val): state["eog_thresh"] = round(val, 2); schedule_threshold_redraw()
    def on_mus_change(val): state["muscle_thresh"] = round(val, 2); schedule_threshold_redraw()
    def schedule_threshold_redraw():
        nonlocal threshold_after_id; window = getattr(fig.canvas.manager, "window", None); histogram_only_redraw()
        if threshold_after_id is not None:
            try:
                if window is not None: window.after_cancel(threshold_after_id)
            except Exception: pass
            threshold_after_id = None
    def redraw_sources_after_slider_release(): _reload_src_data(); draw_sources(); draw_info(); fig.canvas.draw_idle()
    def on_slider_press(event):
        if event.inaxes == ax_eog_sl: active_slider["name"] = "eog"
        elif event.inaxes == ax_mus_sl: active_slider["name"] = "emg"
        elif event.inaxes == ax_time_sl: active_slider["name"] = "time"
        elif event.inaxes == ax_pg_sl: active_slider["name"] = "page"
    def on_slider_release(event):
        slider_name = active_slider.get("name"); active_slider["name"] = None
        if slider_name in ("emg", "eog"): full_redraw()
        elif slider_name == "time": redraw_sources_after_slider_release()
        elif slider_name == "page": draw_sources(); draw_info(); fig.canvas.draw_idle()

    def on_key(event):
        key = (event.key or "").lower()
        if key in ("right", "shift+right"): step = state["time_window"] * (0.5 if key.startswith("shift") else 0.2); set_time_start(state["time_start"] + step)
        elif key in ("left", "shift+left"): step = state["time_window"] * (0.5 if key.startswith("shift") else 0.2); set_time_start(state["time_start"] - step)
        elif key in ("+", "=", "ctrl+right"): set_time_window(state["time_window"] / 1.25); draw_info()
        elif key in ("-", "_", "ctrl+left"): set_time_window(state["time_window"] * 1.25); draw_info()
        elif key == "pageup": set_visible_sources(state["n_visible"] + 2)
        elif key == "pagedown": set_visible_sources(state["n_visible"] - 2)
        elif key == "ctrl+up": state["y_scale"] = min(state["y_scale"] * 1.25, 8.0); draw_sources(); draw_info(); fig.canvas.draw_idle()
        elif key == "ctrl+down": state["y_scale"] = max(state["y_scale"] / 1.25, 0.125); draw_sources(); draw_info(); fig.canvas.draw_idle()
        elif key == "up": set_page(state["page"] - 1)
        elif key == "down": set_page(state["page"] + 1)
        elif key == "home": set_time_start(0.0)
        elif key == "end": set_time_start(raw_filt.times[-1])
        elif key in ("0", "r"): state["time_window"], state["y_scale"], state["n_visible"] = 10.0, 1.0, N_PER_PAGE_DEFAULT; refresh_page_slider(); refresh_time_slider(); set_time_start(state["time_start"]); draw_info()

    def on_confirm(event):
        print(f"\n  Confirmed - {len(ica.exclude)} component(s) excluded: {sorted(ica.exclude)}"); plt.close(fig)
        raw_clean = raw_orig.copy(); ica.apply(raw_clean, verbose=False)
        event_id = {"Thumb": 1, "Index": 2, "Middle": 3, "Pinky": 4, "TrialEnd": 9}
        launch_viewer(raw_orig, raw_clean, subj_id, task, session, nclass, model_type, event_id, ica)

    sl_eog.on_changed(on_eog_change); sl_mus.on_changed(on_mus_change); btn_confirm.on_clicked(on_confirm)
    fig.canvas.mpl_connect("key_press_event", on_key); fig.canvas.mpl_connect("button_press_event", on_slider_press); fig.canvas.mpl_connect("button_release_event", on_slider_release)
    print("  Preparing inspector window ...", flush=True); _reload_src_data(); full_redraw(); fig.canvas.draw(); fig.canvas.flush_events(); plt.figure(fig.number); plt.pause(0.1); bring_to_front(); plt.show(block=True)
    return ica.exclude


def fit_or_load_ica(raw_filt, ica_cache_path=None):
    """
    Fit FastICA on a highpass-filtered Raw object, or reload from a cached
    .fif file if one already exists at `ica_cache_path`.

    Parameters
    ----------
    raw_filt : mne.io.Raw
        Highpass-filtered (≥1 Hz) copy of the raw data used for fitting.
    ica_cache_path : str or None
        Path to a '-ica.fif' file. If it exists the fit is skipped and the
        saved mixing matrix is loaded. After a new fit the result is saved
        here for future calls.

    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted ICA object with `exclude` reset to [].
    """
    if ica_cache_path and os.path.exists(ica_cache_path):
        print(f"\n  Loading cached ICA: {ica_cache_path}")
        ica = mne.preprocessing.read_ica(ica_cache_path)
        ica.exclude = []
        print(f"  ICA loaded — {ica.n_components_} components (fitting skipped)")
    else:
        print("\n  Fitting ICA ...")
        ica = mne.preprocessing.ICA(
            n_components=ICA_N_COMPONENTS, method=ICA_METHOD,
            max_iter="auto", random_state=42, verbose=False,
        )
        ica.fit(raw_filt, verbose=False)
        print(f"  ICA fitted — {ica.n_components_} components")
        if ica_cache_path:
            os.makedirs(os.path.dirname(ica_cache_path), exist_ok=True)
            ica.save(ica_cache_path, overwrite=True)
            print(f"  ICA saved  : {ica_cache_path}")
    return ica


def apply_ica_to_raw(raw, subj_id=0, task="", session=0, nclass=0,
                     model_type="Orig", ica_cache_path=None):
    """
    Fit (or load cached) ICA, launch the interactive inspector, then apply.

    If `ica_cache_path` points to an existing '-ica.fif' file the fitting step
    is skipped entirely and the saved mixing matrix is reloaded.  The exclude
    list is always reset to [] so the user tunes thresholds fresh each time.
    After a new fit the ICA solution is saved to `ica_cache_path` so subsequent
    calls are instant.
    """
    raw_filt = raw.copy().filter(1.0, None, verbose=False)
    ica = fit_or_load_ica(raw_filt, ica_cache_path)

    print("  Computing spectral slopes ...")
    slopes = compute_slopes(ica, raw_filt, fmin=7, fmax=45)
    print("  Opening interactive ICA inspector ...\n"
          "  -> Adjust EOG and Muscle thresholds with sliders\n"
          "  -> Grey traces = excluded components\n"
          "  -> Click CONFIRM when satisfied\n")
    interactive_ica_setup(raw_filt, ica, slopes, raw,
                          subj_id, task, session, nclass, model_type)
    print(f"  ICA done — {len(ica.exclude)} component(s) excluded: {sorted(ica.exclude)}")
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)
    return raw_clean, ica, 0, 0

    

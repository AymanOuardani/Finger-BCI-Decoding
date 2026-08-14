"""
energy_stats.py

Shared library for the per-source energy <-> task analyses.

Every script that measures how much a given ICA source's trial energy tracks a
finger task goes through this module: it owns the energy extraction, the IQR
outlier cleaning, the association metrics (Pearson, AUC, rank-biserial) and the
Artefact / Non-Artefact partition of the 128 sources.

The offline ICA of a subject is reused for every session (see CLAUDE.md), so an
ICA source index means the same spatial component across sessions — which is
what makes the evolution analyses comparable.

Not an entry point. Used by:
    corr_maps.py                tasks x sources heatmaps
    corr_distributions.py       distribution of the per-source correlation
    fill_separation_metrics.py  Excel tables of the separation metrics
    viz_source_energy.py        per-source diagnostic figures
"""

import os
import sys
import glob
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.stats import rankdata, mannwhitneyu

import corr_heatmap as ch
import fill_energy_table as fet
from Functions import (build_raw_from_mat_files, get_folder, get_offline_folder,
                       compute_exclusions)
from config import TASK_LABELS, EOG_THRESHOLD, EMG_SLOPE_THRESH


# =============================================================================
# CONSTANTS
# =============================================================================

BANDS = {
    "Fullband": (None, None),
    "Delta":    (1.0, 4.0),
    "Theta":    (4.0, 8.0),
    "Alpha":    (8.0, 13.0),
    "Beta":     (13.0, 30.0),
    "Beta+2":   (30.0, 40.0),
    "Gamma":    (30.0, 45.0),
}

FINGERS_BY_NCLASS = {2: ["Thumb", "Pinky"], 3: ["Thumb", "Index", "Pinky"]}

CLASS_COLORS = {"Thumb": "#1f77b4", "Index": "#ff7f0e",
                "Middle": "#2ca02c", "Pinky": "#d62728"}

IQR_K = 0.5          # Tukey fence half-width used everywhere for outlier removal
N_COMPONENTS = 128   # fixed ICA decomposition size for all subjects/recordings

# Short filename-safe tags used to distinguish the two model conditions in
# saved figure/sheet names (Orig = trained on raw data, Finetune = fine-tuned).
MODEL_TAG = {"Orig": "1_Orig", "Finetune": "2_Finetune"}


# =============================================================================
# MANUAL COMPONENT SELECTIONS
# =============================================================================
# Per subject, the components inspected by hand. Stored either as the list of
# components KEPT (artefact = the complement) or the list EXCLUDED (artefact =
# the list itself). A subject absent from this table only supports AUTO.

_KEPT = {
    1: [5, 6, 7, 8, 9, 17, 20, 21, 24, 25, 53, 54, 65, 66, 69, 73, 74, 80, 81,
        85, 87, 90, 91, 96, 98, 99, 100, 105, 108],
    2: [3, 4, 5, 7, 8, 10, 21, 23, 42, 48, 52, 53, 64, 69, 77, 83, 84, 85, 86,
        88, 89, 90, 91],
    3: [2, 4, 6, 7, 8, 10, 11, 13, 14, 15, 18, 25, 29, 32, 33, 40, 42, 44, 46,
        50, 54, 55, 56, 61, 62, 63, 67, 72, 74, 75, 77, 79, 80, 84, 87, 89, 93,
        95, 96, 98, 102, 106, 113],
    6: [3, 4, 6, 8, 14, 15, 18, 23, 24, 26, 35, 38, 41, 43, 46, 49, 52, 53, 58,
        59, 63, 71, 78, 81, 82, 83, 85, 88, 89, 92, 93, 94, 95, 100, 101, 103,
        104, 105, 107, 108],
    9: [7, 17, 20, 23, 26, 30, 33, 34, 44, 45, 49, 50, 55, 59, 63, 66, 67, 69,
        70, 82, 83, 84, 85, 86, 91, 93, 97, 105, 107],
}

_EXCLUDED = {
    4: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 18, 21, 22, 23, 24, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 53, 54, 55, 57, 59, 60, 62, 63, 64, 65, 67, 68, 69, 70, 71,
        74, 75, 77, 79, 80, 81, 85, 86, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98,
        99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
        113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
        127],
}

SUBJECTS_WITH_MANUAL = sorted(set(_KEPT) | set(_EXCLUDED))


def manual_artifacts(subj, n_components=N_COMPONENTS):
    """Manually-defined artefact source indices, or None if the subject has none."""
    if subj in _KEPT:
        return set(range(n_components)) - set(_KEPT[subj])
    if subj in _EXCLUDED:
        return set(_EXCLUDED[subj])
    return None


def manual_label(subj, n_components=N_COMPONENTS):
    """Human-readable description of the manual selection, for figure captions."""
    art = manual_artifacts(subj, n_components)
    if art is None:
        return None
    return f"manual selection  ({n_components - len(art)} kept)"


def auto_artifacts(subj, task="MI"):
    """EOG/EMG components auto-detected on the subject's OFFLINE recording.

    Uses the offline ICA, so the indices stay valid for every session.
    Returns an empty set (with a warning) if detection fails.
    """
    try:
        mats = sorted(glob.glob(os.path.join(get_offline_folder(subj, task), "*.mat")))
        raw, _, _ = build_raw_from_mat_files(mats)
        raw.notch_filter(np.arange(60, 501, 60), verbose=False)
        ica = ch._load_offline_ica(subj, task)
        raw_filt = raw.copy().filter(1.0, None, verbose=False)
        _, eog, emg = compute_exclusions(
            ica, raw_filt,
            muscle_thresh=EMG_SLOPE_THRESH, eog_thresh=EOG_THRESHOLD,
            ch_names=raw_filt.info["ch_names"])
        del raw, raw_filt
        gc.collect()
        return set(eog) | set(emg)
    except Exception as exc:
        print(f"  [warn] offline artifact detection failed: {exc}")
        return set()


def artifact_splits(subj, task="MI", which="all"):
    """Artefact partitions to analyse.

    which: "all" | "auto" | "manuel"
    Returns {tag: (label, artefact_index_set)} with tag in {"AUTO", "MANUEL"}.
    """
    splits = {}
    if which in ("all", "auto"):
        splits["AUTO"] = ("auto offline ICA  (EOG/EMG)", auto_artifacts(subj, task))
    if which in ("all", "manuel"):
        art = manual_artifacts(subj)
        if art is not None:
            splits["MANUEL"] = (manual_label(subj), art)
    return splits


# =============================================================================
# ENERGY EXTRACTION
# =============================================================================

def trial_energy(subj, sess, nclass, model, task="MI", band="Fullband"):
    """Per-trial energy of every ICA source for one recording.

    Energy = sum of squared samples of the (optionally band-filtered) ICA source
    over the trial window, trials in chronological order.

    Returns (E, classes) with E of shape (n_trials, n_components) and classes the
    finger label of each trial.
    """
    if band not in BANDS:
        raise ValueError(f"unknown band {band!r}; expected one of {sorted(BANDS)}")
    lo, hi = BANDS[band]

    folder = get_folder(subj, task, sess, nclass, model)
    mats = sorted(glob.glob(os.path.join(folder, "*.mat")))
    raw, _, _ = build_raw_from_mat_files(mats)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)

    ica = ch._load_offline_ica(subj, task)
    sources = ica.get_sources(raw)
    if lo is not None or hi is not None:
        sources.filter(lo, hi, verbose=False)
    data = sources.get_data()

    trials = fet._chronological_trials(raw)
    classes = np.array([c for c, _s, _e in trials])
    E = np.empty((len(trials), data.shape[0]))
    for ti, (_c, start, end) in enumerate(trials):
        # Energy = sum of squared samples over the trial window (per source).
        E[ti] = np.sum(data[:, start:end] ** 2, axis=1)

    del raw, sources, data
    gc.collect()
    return E, classes


def source_energy(subj, sess, nclass, model, src, task="MI", band="Fullband"):
    """Per-trial energy of a SINGLE ICA source, plus its trial classes.

    Cheaper than trial_energy when only one component is needed.
    """
    E, classes = trial_energy(subj, sess, nclass, model, task, band)
    return E[:, src], classes


def available_recordings(subj, task="MI"):
    """{(session, nclass): [models]} actually present on disk for this subject."""
    return fet._detect_groups(subj, task)


def present_fingers(classes, nclass=None):
    """Finger labels actually present in a recording, in TASK_LABELS order."""
    present = set(classes)
    if nclass in FINGERS_BY_NCLASS:
        return [f for f in FINGERS_BY_NCLASS[nclass] if f in present]
    return [t for t in TASK_LABELS if t in present]


# =============================================================================
# OUTLIER CLEANING
# =============================================================================

def iqr_keep_mask(x, k=IQR_K):
    """Tukey fence mask: keep x within [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return (x >= q1 - k * iqr) & (x <= q3 + k * iqr)


def per_class_keep(energy, classes, k=IQR_K):
    """Tukey fence applied WITHIN each finger class, not globally.

    The per-class variant is the one validated on S04 and used for every
    reported figure; a global fence mixes classes with different energy levels
    and silently drops whole classes.
    """
    keep = np.zeros(len(energy), bool)
    for cl in np.unique(classes):
        idx = np.where(classes == cl)[0]
        keep[idx[iqr_keep_mask(energy[idx], k)]] = True
    return keep


# =============================================================================
# ASSOCIATION METRICS
# =============================================================================

def pearson(a, b):
    """Pearson r, 0.0 when undefined (constant input or fewer than 2 points)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def cohens_d(x, y):
    """Standardised mean difference, pooled standard deviation."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    sp = np.sqrt(((nx - 1) * np.var(x, ddof=1) +
                  (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / sp) if sp > 0 else 0.0


def auc_matrix(E, classes, fingers):
    """AUC = P(energy higher during the task) per (finger, source).

    Wilcoxon identity with average (tie-corrected) ranks, matching scipy's
    Mann-Whitney U. 0.5 = chance.
    """
    n, ncomp = E.shape
    # Rank each source's energy independently across ALL trials (column-wise),
    # using average ranks so ties are handled the same way scipy does.
    ranks = np.apply_along_axis(rankdata, 0, E)
    out = np.full((len(fingers), ncomp), 0.5)
    for fi, finger in enumerate(fingers):
        inside = classes == finger
        n1 = int(inside.sum())
        n0 = n - n1
        if n1 == 0 or n0 == 0:
            continue
        # Mann-Whitney U from the sum of in-class ranks (Wilcoxon rank-sum
        # identity): U = R1 - n1*(n1+1)/2. AUC = U / (n1*n0).
        U = ranks[inside].sum(axis=0) - n1 * (n1 + 1) / 2.0
        out[fi] = U / (n1 * n0)
    return out


def rank_biserial_matrix(E, classes, fingers):
    """Signed rank-biserial r = 2*AUC - 1 in [-1, 1]; 0 = chance."""
    return 2.0 * auc_matrix(E, classes, fingers) - 1.0


def pearson_matrix(E, classes, fingers, k=IQR_K):
    """Signed Pearson r per (finger, source), per-class IQR-cleaned.

    For each source, outlier trials are dropped with a per-class Tukey fence on
    that source's energy, THEN Pearson is computed between the surviving
    energies and the finger's one-vs-rest membership.
    """
    ncomp = E.shape[1]
    out = np.empty((len(fingers), ncomp))
    for c in range(ncomp):
        # Outlier trials are re-evaluated PER SOURCE (each column of E has its
        # own energy scale), then Pearson uses only the surviving trials.
        keep = per_class_keep(E[:, c], classes, k)
        cls_kept, energy_kept = classes[keep], E[keep, c]
        for fi, finger in enumerate(fingers):
            # One-vs-rest binary membership vector for this finger -> point-
            # biserial correlation, equivalent to Pearson with a 0/1 variable.
            out[fi, c] = pearson(energy_kept, (cls_kept == finger).astype(float))
    return out


METRICS = {
    "pearson":      ("Signed Pearson r (0.5xIQR-cleaned)", pearson_matrix, (-1, 1)),
    "auc":          ("AUC   P(E_in > E_out)",              auc_matrix,     (0, 1)),
    "rankbiserial": ("Rank-biserial r = 2*AUC - 1",        rank_biserial_matrix, (-1, 1)),
}


def correlation_matrix(E, classes, fingers, metric="pearson"):
    """Dispatch to the requested metric; returns (fingers x sources) array."""
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(METRICS)}")
    return METRICS[metric][1](E, classes, fingers)


# =============================================================================
# ARTEFACT vs NON-ARTEFACT SEPARATION
# =============================================================================

def separation(values, artifacts, n_components=None):
    """How well |r| separates artefactual from non-artefactual sources.

    values     per-source association values (one finger)
    artifacts  set of artefactual source indices

    Returns a dict with the group sizes, the AUC of the Mann-Whitney test on
    |r| (0.5 = no difference, >0.5 = artefacts more task-correlated), Cohen's d
    and the two-sided p-value. Values are NaN when a group is too small.
    """
    values = np.asarray(values, float)
    ncomp = n_components if n_components is not None else len(values)
    art_idx = [c for c in range(ncomp) if c in artifacts]
    non_idx = [c for c in range(ncomp) if c not in artifacts]

    # Compare groups on |r| (magnitude of association), not signed r, since
    # the question is whether artefacts carry MORE task information, not in
    # which direction.
    a = np.abs(values[art_idx])
    b = np.abs(values[non_idx])
    out = {
        "n_artefact": len(a), "n_non_artefact": len(b),
        "mean_artefact": float(np.mean(values[art_idx])) if len(a) else float("nan"),
        "mean_non_artefact": float(np.mean(values[non_idx])) if len(b) else float("nan"),
        "auc": float("nan"), "cohens_d": float("nan"), "p_value": float("nan"),
        "significant": "",
    }
    if len(a) >= 2 and len(b) >= 2:
        # Mann-Whitney U tests whether |r| tends to be larger in group a
        # (artefacts) than group b (non-artefacts); U/(na*nb) is its AUC form.
        U, p = mannwhitneyu(a, b, alternative="two-sided")
        out["auc"] = float(U / (len(a) * len(b)))
        out["cohens_d"] = cohens_d(a, b)
        out["p_value"] = float(p)
        out["significant"] = "YES" if p < 0.05 else "no"
    return out

"""
generate_subject_report.py

For a given subject:
  1. Checks for missing full-band confusion matrices → generates them + writes to ODS
  2. Checks for missing full-band accuracies in ODS  → generates them + writes to ODS
  3. Checks for missing Alpha/Beta/Beta+ accuracies in ODS → generates them + writes to ODS
  4. Checks for missing accuracy evolution plots     → generates them
  5. Checks for missing saliency maps (2/3-class × Orig/Finetune) → generates them
  6. Reads accuracy values from Tracking.ods
  7. Writes Subject_{XX}_Main_Study.tex into the subject's results folder

ALL checks happen before any generation — nothing is regenerated if it already exists.
"""

import os
import sys
import math
import copy
import re
import shutil
import zipfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import scipy.stats
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xml.etree.ElementTree as ET

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras import backend as K

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = (
    "C:/Users/aymen/Desktop/5 Years Of Engineering/DATASIM/DATASIM - Ouardani"
    "/15 - STING (Début 30-03 et Fin 14-08)/Suivi du Stage"
    "/Itération 1/Week 1/W1D1 - 30-03/Codes/Finger-BCI-Decoding-main/Original_Codes"
)
# Excel/ODS tracking files live in the repo's own Ressources folder
# (this script sits in <repo>/Subjects Reports, so the repo root is one level up).
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCEL_DIR = os.path.join(REPO_DIR, "Ressources")
CM_ROOT = (
    "C:/Users/aymen/Desktop/5 Years Of Engineering/DATASIM/DATASIM - Ouardani"
    "/15 - STING (Début 30-03 et Fin 14-08)/Suivi du Stage"
    "/Itération 2 - Résultats"
)
DATA_FOLDER = (
    "D:/EEG-BCI Dataset for Real-time Robotic Hand Control"
    " at Individual Finger Level"
)
SAVE_FOLDER = (
    "C:/Users/aymen/Desktop/5 Years Of Engineering/DATASIM/DATASIM - Ouardani"
    "/15 - STING (Début 30-03 et Fin 14-08)/Suivi du Stage"
    "/Itération 1/Week 1/W1D1 - 30-03/SavePath"
)
TRACKING_ODS = os.path.join(EXCEL_DIR, "Tracking.ods")

sys.path.insert(0, SCRIPT_DIR)
from Functions import load_and_filter_data, generate_eval_paths, eval_model
from EEGModels_tf import EEGNet

# ── Subject — can be overridden from command line ─────────────────────────────
SUBJ_ID = 4

import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--subj",           type=int, default=SUBJ_ID)
_parser.add_argument("--force-saliency", action="store_true",
                     help="Force regeneration of all saliency maps")
_parser.add_argument("--force-evolution", action="store_true",
                     help="Force regeneration of all evolution plots")
_args, _ = _parser.parse_known_args()
SUBJ_ID         = _args.subj
FORCE_SALIENCY  = _args.force_saliency
FORCE_EVOLUTION = _args.force_evolution

# ── Eval base params ──────────────────────────────────────────────────────────
EVAL_PARAMS_BASE = {
    "maxtriallen":   3,
    "windowlen":     1,
    "block_size":    128,
    "downsrate":     100,
}

# ── Saliency params ───────────────────────────────────────────────────────────
SRATE     = 1024
DOWNSRATE = 100
WINDOWLEN = 1
BANDPASS  = [4, 40]
SESSIONS  = [("ME", 1), ("ME", 2), ("MI", 1), ("MI", 2)]

# ── Saliency combos ───────────────────────────────────────────────────────────
SALIENCY_COMBOS = [
    (2, "Orig"),
    (2, "Finetune"),
    (3, "Orig"),
    (3, "Finetune"),
]

# ── Class names ───────────────────────────────────────────────────────────────
CLASS_NAMES = {2: ["Thumb", "Pinky"], 3: ["Thumb", "Index", "Pinky"]}

# ── ODS row offsets ───────────────────────────────────────────────────────────
# row_idx = subj_id + offset  (0-indexed, S01 → subj_id=1)
# Excel row 6  → 0-indexed 5  → offset = 5  - 1 = 4
# Excel row 34 → 0-indexed 33 → offset = 33 - 1 = 32
# Excel row 59 → 0-indexed 58 → offset = 58 - 1 = 57
# Excel row 83 → 0-indexed 82 → offset = 82 - 1 = 81
MAIN_ROW_OFFSET      = 4
ALPHA_ROW_OFFSET     = 32
BETA_ROW_OFFSET      = 57
BETA_PLUS_ROW_OFFSET = 82   # Beta+ [13–40 Hz]

COL_ONLINE_VAL  = {"Orig": 2, "Finetune": 5}
COL_ONLINE_PERF = {"Orig": 3, "Finetune": 6}

# ── All combos ────────────────────────────────────────────────────────────────
CM_COMBOS = [
    (task, session, nclass, modeltype)
    for task      in ("ME", "MI")
    for session   in (1, 2)
    for nclass    in (2, 3)
    for modeltype in ("Orig", "Finetune")
]

# ── Frequency bands to evaluate + write to ODS ───────────────────────────────
BANDS = {
    "Alpha":  {"bandpass_filt": [8,  13], "row_offset": ALPHA_ROW_OFFSET},
    "Beta":   {"bandpass_filt": [13, 30], "row_offset": BETA_ROW_OFFSET},
    "Beta+":  {"bandpass_filt": [13, 40], "row_offset": BETA_PLUS_ROW_OFFSET},
}

# ── Evolution plot constants ──────────────────────────────────────────────────
EVO_CONDITIONS = [
    ("Sess 1\nBase",        1, "Orig"),
    ("Sess 1\nFine-tuned",  1, "Finetune"),
    ("Sess 2\nBase",        2, "Orig"),
    ("Sess 2\nFine-tuned",  2, "Finetune"),
]
EVO_ROW_OFFSETS = {
    "Full":   MAIN_ROW_OFFSET,
    "Alpha":  ALPHA_ROW_OFFSET,
    "Beta":   BETA_ROW_OFFSET,
    "Beta+":  BETA_PLUS_ROW_OFFSET,
}

# ── Band styling — daltonien-friendly (noir/gris + linestyle + marker) ────────
EVO_BAND_STYLE = {
    "Full":  {"color": "#000000", "marker": "o", "lw": 2.5,
              "linestyle": "-",
              "label": "Full Band [4–40 Hz]"},
    "Alpha": {"color": "#555555", "marker": "s", "lw": 2.0,
              "linestyle": "--",
              "label": "Alpha Band [8–13 Hz]"},
    "Beta":  {"color": "#AAAAAA", "marker": "^", "lw": 2.0,
              "linestyle": "-.",
              "label": "Beta Band [13–30 Hz]"},
    "Beta+": {"color": "#333333", "marker": "D", "lw": 1.8,
              "linestyle": (0, (3, 1, 1, 1, 1, 1)),   # dash-dot-dot
              "label": "Beta+ Band [13–40 Hz]"},
}


# ══════════════════════════════════════════════════════════════════════════════
# ODS CACHE
# ══════════════════════════════════════════════════════════════════════════════

def load_ods_cache(ods_path):
    xl     = pd.ExcelFile(ods_path, engine="odf")
    sheets = {
        name: pd.read_excel(ods_path, engine="odf",
                            sheet_name=name, header=None)
        for name in xl.sheet_names
    }
    print(f"  ODS cache loaded: {list(sheets.keys())}")
    return sheets


def cell_is_missing(cache, sheet_name, row_idx, col_idx):
    if sheet_name not in cache:
        return True
    try:
        val = cache[sheet_name].iloc[row_idx, col_idx]
        return pd.isna(val) or str(val).strip() in ("", "nan")
    except Exception:
        return True


# ══════════════════════════════════════════════════════════════════════════════
# ODS WRITE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

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


def _normalize_formula_prefixes(content):
    """Collapse duplicated formula-namespace prefixes in table:formula values.

    Every ODF formula is stored as ``<prefix>:=<expr>`` where ``<prefix>`` is the
    OpenFormula namespace marker (``of`` in ODF 1.2, ``oooc`` in older files).
    When the file is opened and re-saved (e.g. by LibreOffice) the prefix can get
    duplicated, e.g. ``of:=of:=SUM([.C6:.C26])/21``, which LibreOffice then shows
    to the user as ``=of:=SUM(...)``. This restores a single leading prefix:
    ``of:=of:=of:=SUM(...)`` -> ``of:=SUM(...)``. Clean formulas are left as-is.
    """
    def _collapse(match):
        value = re.sub(
            r'^((?:of|oooc):=)(?:(?:of|oooc):=)+', r'\1', match.group(1))
        return 'table:formula="' + value + '"'

    return re.sub(r'table:formula="([^"]*)"', _collapse, content)


def write_to_ods(ods_path, row_idx, writes, table_name):
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
        raise IndexError(f"row_idx={row_idx} out of range in '{table_name}'")

    target_row = all_rows[row_idx]
    for col_idx, value in writes:
        _set_cell(target_row, col_idx, value, ns_map)

    new_content = ET.tostring(root, encoding="unicode")
    new_content = _normalize_formula_prefixes(new_content)
    if not new_content.startswith("<?xml"):
        new_content = '<?xml version="1.0" encoding="UTF-8"?>\n' + new_content

    tmp_path = ods_path + ".tmp"
    with zipfile.ZipFile(ods_path, "r") as zin:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename == "content.xml":
                    zout.writestr(item, new_content.encode("utf-8"))
                else:
                    zout.writestr(item, zin.read(item.filename))
    shutil.move(tmp_path, ods_path)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED EVAL HELPER
# ══════════════════════════════════════════════════════════════════════════════

def _run_eval(subj_id, task, session, nclass, modeltype, bandpass_filt):
    model_path = os.path.join(
        SAVE_FOLDER,
        f"S{subj_id:02}_Sess{session:02}_{task}_{nclass}class_{modeltype}.h5"
    )
    if not os.path.exists(model_path):
        print(f"         [SKIP] Model not found: "
              f"{os.path.basename(model_path)}")
        return None, None, None, None

    try:
        params = {
            **EVAL_PARAMS_BASE,
            "bandpass_filt": bandpass_filt,
            "nclass":        nclass,
        }
        data_paths = generate_eval_paths(
            subj_id, task, nclass, session,
            model_type=modeltype, data_folder=DATA_FOLDER
        )
        data, label_data, p = load_and_filter_data(data_paths, dict(params))
        p["bandpass_filt"] = bandpass_filt

        accuracy, _, _, trial_preds, accuracy_online = eval_model(
            data, label_data, model_path, p
        )
        print(f"         Val={accuracy:.4f}%  Perf={accuracy_online:.4f}%")
        return accuracy, accuracy_online, trial_preds, label_data

    except Exception as e:
        print(f"         [ERR] Eval failed: {e}")
        return None, None, None, None


def _write_acc_to_ods_and_cache(ods_path, cache, sheet_name, row_idx,
                                 modeltype, accuracy, accuracy_online):
    # ── Write to ODS file ─────────────────────────────────────────────────────
    try:
        write_to_ods(
            ods_path, row_idx,
            [(COL_ONLINE_VAL[modeltype],  accuracy),
             (COL_ONLINE_PERF[modeltype], accuracy_online)],
            table_name=sheet_name,
        )
        print(f"         ODS updated → [{sheet_name}] row {row_idx}")
    except IndexError as e:
        print(f"         [SKIP ODS write] {e}")
        return

    # ── Update cache in-place ─────────────────────────────────────────────────
    if sheet_name not in cache:
        return

    df = cache[sheet_name]

    # ── Pandas drops trailing empty rows on read — expand if needed ───────────
    if row_idx >= len(df):
        n_missing = row_idx - len(df) + 1
        empty = pd.DataFrame(
            [[np.nan] * len(df.columns)] * n_missing,
            columns=df.columns
        )
        cache[sheet_name] = pd.concat(
            [df, empty], ignore_index=True)
        df = cache[sheet_name]

    # ── Write values into the (now large enough) cache ────────────────────────
    for col_idx, val in [
        (COL_ONLINE_VAL[modeltype],  accuracy),
        (COL_ONLINE_PERF[modeltype], accuracy_online),
    ]:
        if col_idx >= len(df.columns):
            continue

        # ── Force column to float64 regardless of original dtype ──────────────
        try:
            df.iloc[:, col_idx] = pd.to_numeric(
                df.iloc[:, col_idx], errors="coerce"
            ).astype(float)
        except Exception:
            df[df.columns[col_idx]] = pd.to_numeric(
                df[df.columns[col_idx]], errors="coerce"
            ).astype(float)

        df.iloc[row_idx, col_idx] = float(val)

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — Full-band confusion matrices + missing main accuracies
# ══════════════════════════════════════════════════════════════════════════════

def generate_missing_cms_and_main_acc(subj_id, subj_folder, cache):
    print("\n── Checking full-band CMs + main accuracies ─────────────────")

    for task, session, nclass, modeltype in CM_COMBOS:
        fname      = (f"S{subj_id:02}_Sess{session:02}_{task}_{nclass}class"
                      f"_{modeltype}_CM.png")
        out_path   = os.path.join(subj_folder, fname)
        sheet_name = f"{task}_Sess{session:02}_{nclass}Class"
        row_idx    = subj_id + MAIN_ROW_OFFSET

        cm_exists  = os.path.exists(out_path)
        val_ok     = not cell_is_missing(cache, sheet_name, row_idx,
                                         COL_ONLINE_VAL[modeltype])
        perf_ok    = not cell_is_missing(cache, sheet_name, row_idx,
                                         COL_ONLINE_PERF[modeltype])
        acc_ok     = val_ok and perf_ok

        if cm_exists and acc_ok:
            print(f"  [OK]   {fname}  +  accuracy in ODS")
            continue

        missing = []
        if not cm_exists: missing.append("CM")
        if not acc_ok:    missing.append("accuracy")
        print(f"  [GEN]  {fname} — need: {', '.join(missing)}")

        acc, acc_online, trial_preds, label_data = _run_eval(
            subj_id, task, session, nclass, modeltype, [4, 40]
        )
        if acc is None:
            continue

        if not acc_ok:
            _write_acc_to_ods_and_cache(
                TRACKING_ODS, cache, sheet_name, row_idx,
                modeltype, acc, acc_online
            )

        if not cm_exists:
            cm   = confusion_matrix(label_data, trial_preds,
                                    labels=list(range(1, nclass + 1)))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=CLASS_NAMES[nclass],
            )
            fig, ax = plt.subplots(figsize=(6, 5))
            disp.plot(ax=ax, colorbar=True, cmap="Blues")
            ax.set_title(
                f"Confusion Matrix — S{subj_id:02} Sess{session:02} "
                f"{task} {nclass}-class {modeltype}\n"
                f"Accuracy: {acc:.2f}%"
            )
            plt.tight_layout()
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"         CM saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — Alpha / Beta / Beta+ accuracies → ODS
# ══════════════════════════════════════════════════════════════════════════════

def generate_missing_band_accuracies(subj_id, cache):
    print("\n── Checking Alpha / Beta / Beta+ accuracies ─────────────────")

    for band_name, band_cfg in BANDS.items():
        band_filter = band_cfg["bandpass_filt"]
        row_offset  = band_cfg["row_offset"]

        for task, session, nclass, modeltype in CM_COMBOS:
            sheet_name = f"{task}_Sess{session:02}_{nclass}Class"
            row_idx    = subj_id + row_offset

            val_ok  = not cell_is_missing(cache, sheet_name, row_idx,
                                          COL_ONLINE_VAL[modeltype])
            perf_ok = not cell_is_missing(cache, sheet_name, row_idx,
                                          COL_ONLINE_PERF[modeltype])

            if val_ok and perf_ok:
                print(f"  [OK]   {band_name} | {sheet_name} | "
                      f"S{subj_id:02} | {modeltype}")
                continue

            print(f"  [GEN]  {band_name} | {sheet_name} | "
                  f"S{subj_id:02} | {modeltype} ...")

            acc, acc_online, _, _ = _run_eval(
                subj_id, task, session, nclass, modeltype, band_filter
            )
            if acc is None:
                continue

            _write_acc_to_ods_and_cache(
                TRACKING_ODS, cache, sheet_name, row_idx,
                modeltype, acc, acc_online
            )


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — Accuracy evolution plots
# ══════════════════════════════════════════════════════════════════════════════

def _evo_read(cache, task, nclass, session, modeltype, row_offset):
    sheet_name = f"{task}_Sess{session:02}_{nclass}Class"
    if sheet_name not in cache:
        return None
    try:
        val  = cache[sheet_name].iloc[
            SUBJ_ID + row_offset, COL_ONLINE_PERF[modeltype]]
        fval = float(val)
        return None if math.isnan(fval) else fval
    except Exception:
        return None


def _evo_has_any_data(cache, task, nclass):
    for band_name, row_offset in EVO_ROW_OFFSETS.items():
        for _, session, modeltype in EVO_CONDITIONS:
            if _evo_read(cache, task, nclass, session,
                         modeltype, row_offset) is not None:
                return True
    return False


def evo_fname(subj_id, task, nclass):
    return f"AccuracyEvolution_S{subj_id:02}_{task}_{nclass}class.png"


def generate_evolution_plot(subj_id, subj_folder, task, nclass, cache):
    x        = np.arange(len(EVO_CONDITIONS))
    x_labels = [c[0] for c in EVO_CONDITIONS]

    fig, ax = plt.subplots(figsize=(9, 5))
    any_data = False

    for band_name, row_offset in EVO_ROW_OFFSETS.items():
        style  = EVO_BAND_STYLE[band_name]
        values = [
            _evo_read(cache, task, nclass, session, modeltype, row_offset)
            for _, session, modeltype in EVO_CONDITIONS
        ]

        # ── split at None gaps ────────────────────────────────────────────────
        xs_segs, ys_segs = [], []
        xs_cur,  ys_cur  = [], []
        for xi, val in zip(x, values):
            if val is not None:
                xs_cur.append(xi); ys_cur.append(val)
                any_data = True
            else:
                if xs_cur:
                    xs_segs.append((xs_cur[:], ys_cur[:]))
                xs_cur, ys_cur = [], []
        if xs_cur:
            xs_segs.append((xs_cur, ys_cur))

        first = True
        for xs, ys in xs_segs:
            ax.plot(xs, ys,
                    color=style["color"],
                    marker=style["marker"],
                    linewidth=style["lw"],
                    linestyle=style["linestyle"],
                    markersize=8,
                    label=style["label"] if first else "_nolegend_",
                    zorder=3)
            first = False

        # ── value annotations ─────────────────────────────────────────────────
        for xi, val in zip(x, values):
            if val is not None:
                ax.annotate(
                    f"{val:.2f}%",
                    xy=(xi, val),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=7.5,
                    color=style["color"],
                    fontweight="bold",
                )

    if not any_data:
        plt.close(fig)
        return False

    # ── chance line — pointillé noir ──────────────────────────────────────────
    chance = 100.0 / nclass
    ax.axhline(chance, color="black", linestyle=":",
               linewidth=1.5, alpha=0.5,
               label=f"Chance level ({chance:.1f}%)")

    # ── session boundary ──────────────────────────────────────────────────────
    ax.axvline(1.5, color="black", linestyle=":", linewidth=1, alpha=0.35)
    ylim_bot = ax.get_ylim()[0]
    ax.text(1.52, ylim_bot + 0.5, "Session boundary",
            fontsize=7.5, color="gray", va="bottom")

    # ── formatting ────────────────────────────────────────────────────────────
    task_full = "Motor Execution" if task == "ME" else "Motor Imagery"
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("Online Performance Accuracy (%)", fontsize=10)
    ax.set_xlabel("Condition", fontsize=10)
    ax.set_title(
        f"Accuracy Evolution — Subject S{subj_id:02}  |  "
        f"{task_full} ({task})  |  {nclass}-class\n"
        f"Full vs Alpha vs Beta vs Beta+",
        fontsize=11, fontweight="bold"
    )
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.grid(axis="x", alpha=0.15)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()

    save_path = os.path.join(subj_folder, evo_fname(subj_id, task, nclass))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")
    return True


def check_and_generate_evolution_plots(subj_id, subj_folder, cache):
    print("\n── Checking accuracy evolution plots ────────────────────────")

    for task in ("ME", "MI"):
        for nclass in (2, 3):
            fname    = evo_fname(subj_id, task, nclass)
            out_path = os.path.join(subj_folder, fname)

            if os.path.exists(out_path) and not FORCE_EVOLUTION:
                print(f"  [OK]   {fname}")
                continue

            if not _evo_has_any_data(cache, task, nclass):
                print(f"  [SKIP] {fname} — no data available")
                continue

            if FORCE_EVOLUTION and os.path.exists(out_path):
                os.remove(out_path)
                print(f"  [FORCE] {fname} — deleted, regenerating ...")
            else:
                print(f"  [GEN]  {fname} ...")

            try:
                saved = generate_evolution_plot(
                    subj_id, subj_folder, task, nclass, cache)
                if not saved:
                    print(f"  [SKIP] {fname} — all values missing")
            except Exception as e:
                print(f"  [ERR]  {fname}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — Saliency maps
# ══════════════════════════════════════════════════════════════════════════════

def get_eval_folder(task, session, nclass, model_type):
    modality    = "Movement" if task == "ME" else "Imagery"
    eval_suffix = "Finetune" if model_type == "Finetune" else "Base"
    return f"Online{modality}_Sess{session:02}_{nclass}class_{eval_suffix}"


def load_and_preprocess_saliency(subj_id, task, session, nclass, model_type):
    folder = os.path.join(
        DATA_FOLDER, f"S{subj_id:02}",
        get_eval_folder(task, session, nclass, model_type)
    )
    if not os.path.exists(folder):
        print(f"    [SKIP] Folder not found: {folder}")
        return None

    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    if not mat_files:
        print(f"    [SKIP] No .mat files in: {folder}")
        return None

    all_segs     = []
    maxtriallen  = 3
    desired_len  = int(WINDOWLEN * DOWNSRATE)
    segment_size = int(WINDOWLEN * SRATE)
    step_size    = 128
    padding_len  = 100
    b, a = scipy.signal.butter(4, BANDPASS, btype="bandpass", fs=DOWNSRATE)

    for fpath in mat_files:
        mat     = scipy.io.loadmat(fpath)
        eeg     = mat["eeg"];  event = mat["event"]
        signals = eeg["data"][0][0].astype(float)
        srate   = int(eeg["fsample"][0][0][0][0])

        start_idx, end_idx = [], []
        for i in range(event.shape[1]):
            evt   = event[0, i]
            etype = evt["type"][0]
            samp  = evt["sample"][0][0]
            if etype == "Target":
                start_idx.append(samp - 1)
            elif etype == "TrialEnd":
                end_idx.append(samp - 1)

        for s, e in zip(start_idx, end_idx):
            tmp = signals[:, s:e].astype(float)
            tmp = tmp[:, :min(tmp.shape[1], int(maxtriallen * srate))]
            tmp = np.pad(tmp,
                         ((0, 0),
                          (0, int(maxtriallen * srate) - tmp.shape[1])),
                         "constant", constant_values=np.nan)
            tmp -= tmp.mean(axis=0, keepdims=True)

            for t0 in range(0, tmp.shape[1] - segment_size + 1, step_size):
                seg = tmp[:, t0:t0 + segment_size]
                if np.isnan(seg).any():
                    continue
                seg      = scipy.signal.resample(seg, desired_len, axis=1)
                padded   = np.pad(seg,
                                  ((0, 0), (padding_len, padding_len)),
                                  "constant", constant_values=0)
                filtered = scipy.signal.lfilter(b, a, padded, axis=-1)
                seg      = filtered[:, padding_len:-padding_len]
                seg      = scipy.stats.zscore(
                    seg, axis=1, nan_policy="omit")
                all_segs.append(seg)

    if not all_segs:
        print(f"    [SKIP] No valid segments in: {folder}")
        return None

    X = np.array(all_segs)
    return X.reshape(X.shape[0], X.shape[1], desired_len, 1)


def compute_saliency(model, X, verbose=True):
    X_tensor = tf.constant(X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        loss = tf.reduce_sum(model(X_tensor, training=False))

    grads    = tape.gradient(loss, X_tensor).numpy()[:, :, :, 0]
    saliency = np.abs(grads.mean(axis=0)).sum(axis=1)
    saliency = ((saliency - saliency.min()) /
                (saliency.max() - saliency.min() + 1e-12))

    mean_sal     = saliency.mean()
    std_sal      = saliency.std()
    outlier_mask = np.abs(saliency - mean_sal) > 2 * std_sal
    outlier_idx  = np.where(outlier_mask)[0].tolist()
    n_outliers   = len(outlier_idx)

    saliency_clean = saliency.astype(float).copy()
    saliency_clean[outlier_mask] = np.nan

    if verbose:
        import mne as _mne
        montage  = _mne.channels.make_standard_montage("biosemi128")
        ch_names = montage.ch_names[:len(saliency)]
        if n_outliers > 0:
            removed = [ch_names[i] for i in outlier_idx]
            print(f"    Outliers removed : {n_outliers} channel(s) → {removed}")
        else:
            print(f"    Outliers removed : 0")

    return saliency_clean, n_outliers, outlier_idx


def saliency_fname(nclass, model_type):
    return f"Saliency_Maps_{nclass}class_{model_type}.png"


def generate_saliency_map(subj_id, subj_folder, nclass, model_type):
    import mne
    K.set_image_data_format("channels_last")

    desired_len     = int(WINDOWLEN * DOWNSRATE)
    nChan           = 128
    saliency_dict   = {}
    outlier_summary = {}

    for task, session in SESSIONS:
        key = (task, session)
        print(f"    [{task} Session {session}] loading data ...")
        X = load_and_preprocess_saliency(
            subj_id, task, session, nclass, model_type)
        if X is None:
            saliency_dict[key]   = None
            outlier_summary[key] = (0, [])
            continue

        model_path = os.path.join(
            SAVE_FOLDER,
            f"S{subj_id:02}_Sess{session:02}_{task}_{nclass}class"
            f"_{model_type}.h5"
        )
        if not os.path.exists(model_path):
            print(f"    [SKIP] Model not found: "
                  f"{os.path.basename(model_path)}")
            saliency_dict[key]   = None
            outlier_summary[key] = (0, [])
            continue

        model = EEGNet(
            nb_classes=nclass, Chans=nChan, Samples=desired_len,
            dropoutRate=0.5, kernLength=32, F1=8, D=2, F2=16,
            dropoutType="Dropout"
        )
        model.load_weights(model_path)
        print(f"    Loaded: {os.path.basename(model_path)} "
              f"— {X.shape[0]} segments")

        saliency, n_out, out_idx = compute_saliency(model, X, verbose=True)
        saliency_dict[key]   = saliency
        outlier_summary[key] = (n_out, out_idx)
        print(f"    Saliency range : [{np.nanmin(saliency):.3f}, "
              f"{np.nanmax(saliency):.3f}]")

    montage  = mne.channels.make_standard_montage("biosemi128")
    ch_names = montage.ch_names[:128]
    info     = mne.create_info(ch_names=ch_names, sfreq=DOWNSRATE,
                               ch_types="eeg")
    info.set_montage(montage)

    me1 = saliency_dict.get(("ME", 1)); me2 = saliency_dict.get(("ME", 2))
    mi1 = saliency_dict.get(("MI", 1)); mi2 = saliency_dict.get(("MI", 2))

    me_valid = [m for m in [me1, me2] if m is not None]
    mi_valid = [m for m in [mi1, mi2] if m is not None]
    avg_me   = np.nanmean(me_valid, axis=0) if me_valid else None
    avg_mi   = np.nanmean(mi_valid, axis=0) if mi_valid else None

    all_valid = [m for m in [me1, me2, mi1, mi2, avg_me, avg_mi]
                 if m is not None]
    if not all_valid:
        print(f"  [SKIP] No valid saliency data for "
              f"{nclass}-class {model_type}.")
        return

    vmax        = max(np.nanmax(m) for m in all_valid)
    model_label = "Base Model" if model_type == "Orig" else "Fine-tuned Model"

    fig = plt.figure(figsize=(13, 8))
    fig.suptitle(
        f"Subject S{subj_id:02}  —  Saliency Maps\n"
        f"{nclass}-class  |  {model_label}",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.12)

    layout = [
        (0, 0, me1,    "ME — Session 1", ("ME", 1)),
        (0, 1, me2,    "ME — Session 2", ("ME", 2)),
        (0, 2, avg_me, "ME — Average",   None),
        (1, 0, mi1,    "MI — Session 1", ("MI", 1)),
        (1, 1, mi2,    "MI — Session 2", ("MI", 2)),
        (1, 2, avg_mi, "MI — Average",   None),
    ]

    for row, col, smap, label, key in layout:
        ax = fig.add_subplot(gs[row, col])
        if col == 0:
            ax.set_ylabel(
                "Motor Execution (ME)" if row == 0 else "Motor Imagery (MI)",
                fontsize=10, fontweight="bold", labelpad=12)

        if key is not None and key in outlier_summary:
            n_out, out_idx = outlier_summary[key]
            removed = [ch_names[i] for i in out_idx]
            if n_out > 0:
                mid     = (n_out + 1) // 2
                out_str = (", ".join(removed[:mid]) + "\n"
                           + ", ".join(removed[mid:])) if n_out > 3 \
                           else ", ".join(removed)
                title_str = f"{label}\n{n_out} outlier(s): {out_str}"
            else:
                title_str = f"{label}\n(no outliers)"
        else:
            title_str = label

        if smap is None:
            ax.axis("off")
            ax.set_title(title_str + "\n[No data]", fontsize=7, color="gray")
            continue

        smap_plot = np.where(np.isnan(smap), 0.0, smap)
        mne.viz.plot_topomap(
            smap_plot, info, axes=ax, cmap="hot_r",
            vlim=(0, vmax), contours=4, extrapolate="head",
            sphere=(0., 0., 0., 0.095), outlines="head",
            show=False, sensors=True,
        )
        ax.set_title(title_str, fontsize=7, fontweight="bold")

    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.68])
    sm = plt.cm.ScalarMappable(
        cmap="hot_r", norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, cax=cbar_ax, label="Normalized saliency")

    save_path = os.path.join(subj_folder, saliency_fname(nclass, model_type))
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def check_and_generate_saliency(subj_id, subj_folder):
    print("\n── Checking saliency maps ───────────────────────────────────")
    for nclass, model_type in SALIENCY_COMBOS:
        fname    = saliency_fname(nclass, model_type)
        sal_path = os.path.join(subj_folder, fname)
        if os.path.exists(sal_path) and not FORCE_SALIENCY:
            print(f"  [OK]   {fname}")
        else:
            if FORCE_SALIENCY and os.path.exists(sal_path):
                os.remove(sal_path)
                print(f"  [FORCE] {fname} — deleted, regenerating ...")
            else:
                print(f"  [GEN]  {fname} ...")
            try:
                generate_saliency_map(
                    subj_id, subj_folder,
                    nclass=nclass, model_type=model_type
                )
            except Exception as e:
                print(f"  [ERR]  {fname}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — LaTeX report
# ══════════════════════════════════════════════════════════════════════════════

def read_acc(cache, task, session, nclass, modeltype, row_offset):
    sheet_name = f"{task}_Sess{session:02}_{nclass}Class"
    if sheet_name not in cache:
        return "XX"
    try:
        val  = cache[sheet_name].iloc[
            SUBJ_ID + row_offset, COL_ONLINE_PERF[modeltype]]
        fval = float(val)
        if math.isnan(fval):
            return "XX"
        return f"{fval:.3f}".rstrip("0").rstrip(".")
    except Exception:
        return "XX"


def img_or_na(subj_folder, filename):
    if os.path.exists(os.path.join(subj_folder, filename)):
        return f"\\includegraphics[width=\\linewidth]{{{filename}}}"
    return (
        "\\fbox{\\parbox[c][2.8cm][c]{0.9\\linewidth}"
        "{\\centering\\scriptsize N/A}}"
    )


def cm_quad_block(subj_folder, session):
    s  = f"{SUBJ_ID:02}"
    se = f"{session:02}"

    def img(task, nclass, modeltype):
        fname = f"S{s}_Sess{se}_{task}_{nclass}class_{modeltype}_CM.png"
        return img_or_na(subj_folder, fname)

    me2b = img("ME", 2, "Orig");    me2f = img("ME", 2, "Finetune")
    me3b = img("ME", 3, "Orig");    me3f = img("ME", 3, "Finetune")
    mi2b = img("MI", 2, "Orig");    mi2f = img("MI", 2, "Finetune")
    mi3b = img("MI", 3, "Orig");    mi3f = img("MI", 3, "Finetune")

    col_spec = (
        "@{}"
        r">{\centering\arraybackslash}p{0.232\linewidth}"
        r">{\centering\arraybackslash}p{0.232\linewidth}"
        r">{\centering\arraybackslash}p{0.232\linewidth}"
        r">{\centering\arraybackslash}p{0.232\linewidth}"
        "@{}"
    )

    return (
        f"\\noindent{{\\footnotesize\\textit{{Session {session}}}}}\n\n"
        f"\\vspace{{4pt}}\n"
        f"{{\\setlength{{\\tabcolsep}}{{2pt}}\n"
        f"\\noindent\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        f"  {{\\scriptsize\\textbf{{ME — 2-class Base}}}} &\n"
        f"  {{\\scriptsize\\textbf{{ME — 2-class Fine-tuned}}}} &\n"
        f"  {{\\scriptsize\\textbf{{ME — 3-class Base}}}} &\n"
        f"  {{\\scriptsize\\textbf{{ME — 3-class Fine-tuned}}}} \\\\\n"
        f"  {me2b} &\n"
        f"  {me2f} &\n"
        f"  {me3b} &\n"
        f"  {me3f} \\\\[6pt]\n"
        f"  {{\\scriptsize\\textbf{{MI — 2-class Base}}}} &\n"
        f"  {{\\scriptsize\\textbf{{MI — 2-class Fine-tuned}}}} &\n"
        f"  {{\\scriptsize\\textbf{{MI — 3-class Base}}}} &\n"
        f"  {{\\scriptsize\\textbf{{MI — 3-class Fine-tuned}}}} \\\\\n"
        f"  {mi2b} &\n"
        f"  {mi2f} &\n"
        f"  {mi3b} &\n"
        f"  {mi3f} \\\\\n"
        f"\\end{{tabular}}}}\n"
    )


def accuracy_table(cache, row_offset, caption_suffix, label_suffix):
    def a(task, session, nclass, modeltype):
        return read_acc(cache, task, session, nclass, modeltype, row_offset)

    return (
        f"\\begin{{table}}[H]\n"
        f"\\centering\n"
        f"\\small\n"
        f"\\renewcommand{{\\arraystretch}}{{1.3}}\n"
        f"\\caption{{Subject S{SUBJ_ID:02} — Majority voting accuracy (\\%) "
        f"across all main study sessions. ({caption_suffix})}}\n"
        f"\\label{{tab:s{SUBJ_ID:02}_{label_suffix}}}\n"
        f"\\begin{{tabular}}{{llcccc}}\n"
        f"\\toprule\n"
        f"\\multirow{{2}}{{*}}{{\\textbf{{Task}}}} &\n"
        f"\\multirow{{2}}{{*}}{{\\textbf{{Classes}}}} &\n"
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{Session 1}}}} &\n"
        f"\\multicolumn{{2}}{{c}}{{\\textbf{{Session 2}}}} \\\\\n"
        f"\\cmidrule(lr){{3-4}} \\cmidrule(lr){{5-6}}\n"
        f"& & \\textbf{{Base}} & \\textbf{{Fine-tuned}} & "
        f"\\textbf{{Base}} & \\textbf{{Fine-tuned}} \\\\\n"
        f"\\midrule\n"
        f"\\multirow{{2}}{{*}}{{ME}}\n"
        f"  & 2-class & {a('ME',1,2,'Orig')} & {a('ME',1,2,'Finetune')} "
        f"& {a('ME',2,2,'Orig')} & {a('ME',2,2,'Finetune')} \\\\\n"
        f"  & 3-class & {a('ME',1,3,'Orig')} & {a('ME',1,3,'Finetune')} "
        f"& {a('ME',2,3,'Orig')} & {a('ME',2,3,'Finetune')} \\\\\n"
        f"\\midrule\n"
        f"\\multirow{{2}}{{*}}{{MI}}\n"
        f"  & 2-class & {a('MI',1,2,'Orig')} & {a('MI',1,2,'Finetune')} "
        f"& {a('MI',2,2,'Orig')} & {a('MI',2,2,'Finetune')} \\\\\n"
        f"  & 3-class & {a('MI',1,3,'Orig')} & {a('MI',1,3,'Finetune')} "
        f"& {a('MI',2,3,'Orig')} & {a('MI',2,3,'Finetune')} \\\\\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}\n"
        f"\\end{{table}}"
    )


def evolution_block(subj_id, subj_folder):
    col_w = "0.47\\linewidth"

    def cell(task, nclass):
        fname    = evo_fname(subj_id, task, nclass)
        task_lbl = "Motor Execution (ME)" if task == "ME" else "Motor Imagery (MI)"
        title    = f"{task_lbl} — {nclass}-class"
        if os.path.exists(os.path.join(subj_folder, fname)):
            content = f"\\includegraphics[width=\\linewidth]{{{fname}}}"
        else:
            content = (
                f"\\fbox{{\\parbox[c][5cm][c]{{0.9\\linewidth}}"
                f"{{\\centering\\small No data available}}}}"
            )
        return (
            f"\\begin{{minipage}}[t]{{{col_w}}}\n"
            f"  \\centering\n"
            f"  {{\\small\\bfseries\\color{{navyblue}} {title}}}\\\\\n"
            f"  \\vspace{{4pt}}\n"
            f"  {content}\n"
            f"\\end{{minipage}}"
        )

    any_exists = any(
        os.path.exists(os.path.join(subj_folder, evo_fname(subj_id, task, nclass)))
        for task in ("ME", "MI") for nclass in (2, 3)
    )
    if not any_exists:
        return ""

    me2 = cell("ME", 2);  me3 = cell("ME", 3)
    mi2 = cell("MI", 2);  mi3 = cell("MI", 3)

    return (
        f"{me2}\\hfill\n{me3}\n\n"
        f"\\vspace{{10pt}}\n\n"
        f"{mi2}\\hfill\n{mi3}"
    )


def saliency_block(subj_folder):
    model_labels = {"Orig": "Base Model", "Finetune": "Fine-tuned Model"}
    lines = []

    for i, (nclass, model_type) in enumerate(SALIENCY_COMBOS):
        fname = saliency_fname(nclass, model_type)
        title = f"{nclass}-class  —  {model_labels[model_type]}"

        lines.append(
            f"{{\\small\\bfseries\\color{{navyblue}} "
            f"Saliency Maps — {title}}}\n\n"
            f"\\vspace{{4pt}}\n"
        )
        if os.path.exists(os.path.join(subj_folder, fname)):
            lines.append(
                f"\\noindent\\hfil\n"
                f"\\includegraphics[width=0.92\\linewidth]{{{fname}}}\n"
                f"\\hfil"
            )
        else:
            lines.append(
                f"\\noindent\\hfil\n"
                f"\\fbox{{\\parbox[c][4cm][c]{{0.6\\linewidth}}"
                f"{{\\centering {title} saliency map "
                f"not yet generated}}}}\n"
                f"\\hfil"
            )

        if i < len(SALIENCY_COMBOS) - 1:
            lines.append(
                "\n\n\\vspace{14pt}\n"
                "\\noindent\\rule{\\linewidth}{0.3pt}\n"
                "\\vspace{6pt}\n"
            )

    return "\n".join(lines)


def generate_tex(subj_id, subj_folder, cache):
    tbl_full      = accuracy_table(cache, MAIN_ROW_OFFSET,
                                   "Full Frequency Band [4--40 Hz]",  "full")
    tbl_alpha     = accuracy_table(cache, ALPHA_ROW_OFFSET,
                                   "Alpha Band Only [8--13 Hz]",      "alpha")
    tbl_beta      = accuracy_table(cache, BETA_ROW_OFFSET,
                                   "Beta Band Only [13--30 Hz]",      "beta")
    tbl_beta_plus = accuracy_table(cache, BETA_PLUS_ROW_OFFSET,
                                   "Beta+ Band Only [13--40 Hz]",     "betaplus")

    evo_content = evolution_block(subj_id, subj_folder)
    evo_page = (
        f"\\newpage\n\n"
        f"%% ── ACCURACY EVOLUTION CURVES ──────────────────────────────────\n"
        f"\\begin{{center}}\n"
        f"  {{\\large\\bfseries\\color{{navyblue}} "
        f"Accuracy Evolution — Full vs Alpha vs Beta vs Beta+}}\\\\\n"
        f"  {{\\small Session 1 Base $\\rightarrow$ Session 1 Fine-tuned "
        f"$\\rightarrow$ Session 2 Base $\\rightarrow$ Session 2 Fine-tuned}}\n"
        f"\\end{{center}}\n\n"
        f"\\vspace{{6pt}}\n"
        f"{evo_content}\n"
    ) if evo_content else ""

    cm_s1 = cm_quad_block(subj_folder, 1)
    cm_s2 = cm_quad_block(subj_folder, 2)
    sal   = saliency_block(subj_folder)

    return f"""\\documentclass[a4paper, 12pt]{{article}}

\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage{{graphicx}}
\\usepackage{{float}}
\\usepackage[margin=1.2cm, top=1.5cm]{{geometry}}
\\usepackage{{caption}}
\\usepackage{{multirow}}
\\usepackage{{booktabs}}
\\usepackage{{xcolor}}
\\usepackage{{array}}
\\usepackage{{tabularx}}

\\definecolor{{navyblue}}{{RGB}}{{0, 32, 96}}
\\captionsetup{{font=small, labelfont=bf}}

\\begin{{document}}

\\begin{{center}}
    {{\\Large\\bfseries\\color{{navyblue}} Subject S{subj_id:02} — Main Study Recap}}\\\\[4pt]
    {{\\small Motor Execution (ME) \\& Motor Imagery (MI) — Sessions 1 \\& 2}}
\\end{{center}}

\\vspace{{4pt}}
\\noindent\\rule{{\\linewidth}}{{0.8pt}}
\\vspace{{4pt}}

%% ── ACCURACY TABLES ─────────────────────────────────────────────────────────
{tbl_full}

\\vspace{{8pt}}
{tbl_alpha}

\\vspace{{8pt}}
{tbl_beta}

\\vspace{{8pt}}
{tbl_beta_plus}

{evo_page}
\\newpage

%% ── CONFUSION MATRICES — FULL BAND [4-40 Hz] ────────────────────────────────
{{\\small\\bfseries\\color{{navyblue}} Confusion Matrices — Full Band [4--40 Hz]}}

\\vspace{{6pt}}
{cm_s1}

\\vspace{{10pt}}
\\noindent\\rule{{\\linewidth}}{{0.3pt}}
\\vspace{{6pt}}

{cm_s2}

\\newpage

%% ── SALIENCY MAPS ────────────────────────────────────────────────────────────
{sal}

\\end{{document}}
"""


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    subj_folder = os.path.join(CM_ROOT, f"Sujet {SUBJ_ID}")
    os.makedirs(subj_folder, exist_ok=True)

    print("\n── Loading ODS cache ────────────────────────────────────────")
    cache = load_ods_cache(TRACKING_ODS)

    # Step 1 ── full-band CMs + main accuracies ────────────────────────────────
    generate_missing_cms_and_main_acc(SUBJ_ID, subj_folder, cache)

    # Step 2 ── Alpha / Beta / Beta+ accuracies ────────────────────────────────
    generate_missing_band_accuracies(SUBJ_ID, cache)

    # Step 3 ── accuracy evolution plots ──────────────────────────────────────
    check_and_generate_evolution_plots(SUBJ_ID, subj_folder, cache)

    # Step 4 ── saliency maps ──────────────────────────────────────────────────
    check_and_generate_saliency(SUBJ_ID, subj_folder)

    # Step 5 ── generate .tex ──────────────────────────────────────────────────
    print(f"\n── Generating report for Subject S{SUBJ_ID:02} ───────────────")
    tex      = generate_tex(SUBJ_ID, subj_folder, cache)
    out_path = os.path.join(
        subj_folder, f"Subject_{SUBJ_ID:02}_Main_Study.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"  Saved: {out_path}")
    print(f"\nCompile with:")
    print(f'  pdflatex -output-directory "{subj_folder}" "{out_path}"')


if __name__ == "__main__":
    main()
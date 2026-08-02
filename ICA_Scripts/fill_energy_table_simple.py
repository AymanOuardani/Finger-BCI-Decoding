"""
fill_energy_table_simple.py

Identical to fill_energy_table.py but skips all image generation.
Produces only the ODS workbook (Ressources/Sujet_XX.ods) with:
  - 10 energy sheets  Sess_XX_Yclass
  - 1 Corr sheet
  - 1 Enrgy_Corr_Evolution sheet

Usage:
    python fill_energy_table_simple.py <subj> [<task>]
"""

import os
import sys
import glob
import zipfile
from xml.sax.saxutils import escape

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Functions import (build_raw_from_mat_files, get_folder, get_offline_folder,
                       compute_exclusions)
from config import (EXCEL_DIR, TASK_LABELS, EOG_THRESHOLD, EMG_SLOPE_THRESH,
                    CORRUPTED_DATA)

import corr_heatmap as ch   # reuse _load_offline_ica

MAX_SESSIONS = 5

HEADER = ["Sujet_id", "Session", "nClass", "Model", "Task", "Class", "Trial",
          "ICA_Source", "Energy_Fullband", "Energy_Alpha",
          "Energy_Beta", "Energy_Beta+2", "Artefact ?"]

CORR_HEADER = ["Session", "nClass", "Model", "Band", "Task", "ICA_Source",
               "Corr_Energy", "Artefact ?"]

# (label, l_freq, h_freq). None -> broadband (no band-pass).
BANDS = [("Fullband", None, None), ("Alpha", 8.0, 13.0),
         ("Beta", 13.0, 30.0), ("Beta+2", 30.0, 40.0)]

EVOL_SHEET = "Enrgy_Corr_Evolution"
N_SRC_DEFAULT = 128


def _energy_task_corr(E, trial_classes, tasks):
    """SIGNED point-biserial PEARSON between each source's per-trial energy and
    each task's one-vs-rest membership, across trials. E: (n_trials, n_comp)."""
    n_trials, n_comp = E.shape
    Ez = E - E.mean(axis=0, keepdims=True)
    es = Ez.std(axis=0, keepdims=True)
    es[es == 0] = 1.0
    Ez = Ez / es
    cls_arr = np.asarray(trial_classes)
    out = np.zeros((len(tasks), n_comp))
    for ti, T in enumerate(tasks):
        m = (cls_arr == T).astype(float)
        m -= m.mean()
        ms = m.std()
        if ms == 0:
            continue
        out[ti] = (m / ms) @ Ez / n_trials
    return out


def _chronological_trials(raw):
    """List of (class_name, start_sample, end_sample) sorted by onset."""
    ann = raw.annotations
    descs = np.asarray(ann.description, dtype=str)
    onsets = np.asarray(ann.onset, dtype=float)
    durations = np.asarray(ann.duration, dtype=float)
    sfreq = float(raw.info["sfreq"])
    n = raw.n_times
    trial_end = np.sort(onsets[descs == "TrialEnd"])

    events = []
    for i in range(len(descs)):
        cls = descs[i]
        if cls not in TASK_LABELS:
            continue
        onset, dur = onsets[i], durations[i]
        if dur <= 0:
            fe = trial_end[trial_end > onset]
            dur = (fe[0] - onset) if len(fe) else 3.0
        s = max(0, int(round(onset * sfreq)))
        e = min(n, int(round((onset + dur) * sfreq)))
        if e <= s:
            e = min(n, s + 1)
        events.append((onset, cls, s, e))

    events.sort(key=lambda t: t[0])
    return [(cls, s, e) for _onset, cls, s, e in events]


def _rows_for_recording(subj_id, task, session, nclass, model):
    folder = get_folder(subj_id, task, session, nclass, model)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    raw, _, _ = build_raw_from_mat_files(mat_files)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)

    ica = ch._load_offline_ica(subj_id, task)
    src_raw = ica.get_sources(raw)
    n_comp = src_raw.ch_names.__len__()

    artifact_comps = set()
    try:
        raw_det = raw.copy().filter(1.0, None, verbose=False)
        _, eog_idx, emg_idx = compute_exclusions(
            ica, raw_det, muscle_thresh=EMG_SLOPE_THRESH,
            eog_thresh=EOG_THRESHOLD, ch_names=raw_det.info["ch_names"],
        )
        artifact_comps = set(eog_idx) | set(emg_idx)
    except Exception as e:
        print(f"      [warn] EOG/EMG detection failed: {e}")
    print(f"      artifact sources: {len(artifact_comps)} / {n_comp}", flush=True)

    trials = _chronological_trials(raw)
    if not trials:
        return {"rows": [], "E_per_band": [np.empty((0, n_comp)) for _ in BANDS],
                "trial_classes": [], "artifacts": artifact_comps, "n_comp": n_comp}

    band_data = []
    for _label, lo, hi in BANDS:
        if lo is None:
            band_data.append(src_raw.get_data())
        else:
            band_data.append(src_raw.copy().filter(lo, hi, verbose=False).get_data())

    n_trials = len(trials)
    trial_classes = [cls for cls, _s, _e in trials]
    E_per_band = [np.empty((n_trials, n_comp)) for _ in BANDS]

    rows = []
    for t_idx, (cls, s, e) in enumerate(trials, start=1):
        for bi, bd in enumerate(band_data):
            E_per_band[bi][t_idx - 1] = np.sum(bd[:, s:e] ** 2, axis=1)
        for c in range(n_comp):
            rows.append([
                subj_id, session, nclass, model, task, cls, t_idx, c,
                float(E_per_band[0][t_idx - 1, c]), float(E_per_band[1][t_idx - 1, c]),
                float(E_per_band[2][t_idx - 1, c]), float(E_per_band[3][t_idx - 1, c]),
                "Yes" if c in artifact_comps else "No",
            ])

    return {"rows": rows, "E_per_band": E_per_band, "trial_classes": trial_classes,
            "artifacts": artifact_comps, "n_comp": n_comp}


# ── ODS construction from scratch ─────────────────────────────────────────────

_MIMETYPE = "application/vnd.oasis.opendocument.spreadsheet"

_MANIFEST = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"'
    ' manifest:version="1.2">'
    '<manifest:file-entry manifest:full-path="/" manifest:version="1.2"'
    f' manifest:media-type="{_MIMETYPE}"/>'
    '<manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>'
    '<manifest:file-entry manifest:full-path="styles.xml" manifest:media-type="text/xml"/>'
    '</manifest:manifest>'
)

_STYLES = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<office:document-styles'
    ' xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"'
    ' office:version="1.2"></office:document-styles>'
)

_CONTENT_HEAD = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<office:document-content'
    ' xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"'
    ' xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"'
    ' xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"'
    ' office:version="1.2"><office:body><office:spreadsheet>'
)
_CONTENT_TAIL = '</office:spreadsheet></office:body></office:document-content>'


def _cell_xml(v):
    if isinstance(v, str):
        return (f'<table:table-cell office:value-type="string">'
                f'<text:p>{escape(v)}</text:p></table:table-cell>')
    if isinstance(v, int):
        return (f'<table:table-cell office:value-type="float" office:value="{v}">'
                f'<text:p>{v}</text:p></table:table-cell>')
    fv = float(v)
    return (f'<table:table-cell office:value-type="float" office:value="{fv!r}">'
            f'<text:p>{round(fv, 6)}</text:p></table:table-cell>')


def _row_xml(values):
    return ("<table:table-row>"
            + "".join(_cell_xml(v) for v in values)
            + "</table:table-row>")


def build_subject_ods(path, sheets):
    """sheets = list of (sheet_name, header_list_or_None, rows_list)."""
    parts = [_CONTENT_HEAD]
    for name, header, rows in sheets:
        ncols = len(header) if header else max((len(r) for r in rows), default=1)
        parts.append(f'<table:table table:name="{escape(name)}">')
        parts.append(f'<table:table-column table:number-columns-repeated="{ncols}"/>')
        if header:
            parts.append(_row_xml(header))
        for r in rows:
            parts.append(_row_xml(r))
        parts.append("</table:table>")
    parts.append(_CONTENT_TAIL)
    content = "".join(parts)

    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", _MIMETYPE, compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/manifest.xml", _MANIFEST)
        zf.writestr("styles.xml", _STYLES)
        zf.writestr("content.xml", content.encode("utf-8"))
    os.replace(tmp, path)


MODELS = ["Orig", "Finetune"]


def _detect_groups(subj_id, task):
    """{(session, nclass): [models present]}"""
    groups = {}
    for sess in range(1, MAX_SESSIONS + 1):
        for ncl in (2, 3):
            models = []
            for model in MODELS:
                if (subj_id, task, sess, ncl, model) in CORRUPTED_DATA:
                    continue
                try:
                    folder = get_folder(subj_id, task, sess, ncl, model)
                except Exception:
                    continue
                if os.path.isdir(folder) and glob.glob(os.path.join(folder, "*.mat")):
                    models.append(model)
            if models:
                groups[(sess, ncl)] = models
    return groups


def fill_subject(subj_id, task="MI"):
    out_path = os.path.join(EXCEL_DIR, f"Sujet_{subj_id:02d}.ods")
    print(f"\n=== Energy workbook for S{subj_id:02} -> {out_path} ===")

    groups = _detect_groups(subj_id, task)
    if not groups:
        print("  Nothing to do — no recording data.")
        return
    print(f"  {len(groups)} session-sheet(s) (one per session x nClass, "
          f"Orig+Finetune merged):")
    for (sess, ncl), models in sorted(groups.items()):
        print(f"     Sess_{sess:02d}_{ncl}class  <- {models}")

    offline_artifacts = set()
    try:
        off_folder  = get_offline_folder(subj_id, task)
        off_mats    = sorted(glob.glob(os.path.join(off_folder, "*.mat")))
        if off_mats:
            raw_off, _, _ = build_raw_from_mat_files(off_mats)
            raw_off.notch_filter(np.arange(60, 501, 60), verbose=False)
            ica_off = ch._load_offline_ica(subj_id, task)
            raw_off_filt = raw_off.copy().filter(1.0, None, verbose=False)
            _, eog_idx, emg_idx = compute_exclusions(
                ica_off, raw_off_filt, muscle_thresh=EMG_SLOPE_THRESH,
                eog_thresh=EOG_THRESHOLD, ch_names=raw_off_filt.info["ch_names"],
            )
            offline_artifacts = set(eog_idx) | set(emg_idx)
            print(f"  Offline artifact sources ({len(offline_artifacts)}): "
                  f"{sorted(offline_artifacts)}")
    except Exception as e:
        print(f"  [warn] Offline artifact detection failed: {e}")

    sheets = []
    corr_rows = []
    evol = {}
    evol_sessions = set()
    evol_tasks = set()
    evol_models = set()
    n_comp_evol = N_SRC_DEFAULT

    for (sess, ncl), models in sorted(groups.items()):
        name = f"Sess_{sess:02d}_{ncl}class"
        print(f"\n  == {name} ({', '.join(models)}) ==", flush=True)

        sheet_rows = []
        per_model = {}
        for model in models:
            print(f"    -- {model} --", flush=True)
            try:
                res = _rows_for_recording(subj_id, task, sess, ncl, model)
            except Exception as ex:
                print(f"      [SKIP] {model} failed: {ex}", flush=True)
                continue
            sheet_rows.extend(res["rows"])
            per_model[model] = (res["E_per_band"], res["trial_classes"],
                                res["artifacts"], res["n_comp"])
        sheets.append((name, HEADER, sheet_rows))

        for model, (E_per_band, classes, artifacts, n_comp) in per_model.items():
            if not classes:
                continue
            tasks = [t for t in TASK_LABELS if t in set(classes)]
            for bi, (band_label, _lo, _hi) in enumerate(BANDS):
                mat = _energy_task_corr(E_per_band[bi], classes, tasks)
                for ti, T in enumerate(tasks):
                    for c in range(n_comp):
                        corr_rows.append([
                            sess, ncl, model, band_label, T, c,
                            float(mat[ti, c]),
                            "Yes" if c in offline_artifacts else "No",
                        ])
                    if ncl == 3:
                        evol[(sess, band_label, T, model)] = mat[ti]
            if ncl == 3:
                evol_sessions.add(sess)
                evol_tasks.update(tasks)
                evol_models.add(model)
                n_comp_evol = n_comp

    sheets.append(("Corr", CORR_HEADER, corr_rows))

    if evol_sessions:
        sessions_sorted = sorted(evol_sessions)
        evol_task_order = [t for t in TASK_LABELS if t in evol_tasks]
        model_order = [m for m in MODELS if m in evol_models]
        src_hdr = [f"Source_{i}" for i in range(n_comp_evol)]
        evol_rows = []
        for band_label, _lo, _hi in BANDS:
            for T in evol_task_order:
                for model in model_order:
                    evol_rows.append(
                        [f"Energy_{band_label} ({T}, 3Class, {model})"] + src_hdr)
                    for s in sessions_sorted:
                        vec = evol.get((s, band_label, T, model))
                        row = [f"Sess{s:02d}"]
                        if vec is None:
                            row += [""] * n_comp_evol
                        else:
                            row += [float(vec[c]) for c in range(n_comp_evol)]
                        evol_rows.append(row)
                    evol_rows.append([])
        sheets.append((EVOL_SHEET, None, evol_rows))

    total = sum(len(r) for _n, _h, r in sheets)
    print(f"\n  Writing {out_path} — {len(sheets)} sheet(s), {total} data row(s) ...",
          flush=True)
    build_subject_ods(out_path, sheets)
    print("  Done.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python fill_energy_table_simple.py <subj> [<task>]")
        sys.exit(1)
    subj_id = int(sys.argv[1])
    task    = sys.argv[2] if len(sys.argv) > 2 else "MI"
    fill_subject(subj_id, task)


if __name__ == "__main__":
    main()

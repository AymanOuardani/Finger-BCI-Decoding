"""
fill_energy_table.py

For one subject, build a dedicated workbook Ressources/Sujet_XX.ods with ONE
SHEET PER (SESSION x nClass), named Sess_XX_<n>class (e.g. Sess_01_2class,
Sess_01_3class). The Orig and Finetune recordings of that session+nClass are
MERGED into the same sheet (distinguished by the Model column). So up to 2
sheets per session (2c, 3c) -> 10 sheets for a 5-session subject. Each sheet
lists, in chronological signal order (trials sorted by onset, NOT grouped by
task), every trial and every ICA source:

  Sujet_id | Session | nClass | Model | Task | Class | Trial | ICA_Source |
  Energy_Fullband | Energy_Alpha | Energy_Beta | Energy_Beta+2 | Artefact ?

  - Energy = sum of squared samples (Sigma x^2) of the band-filtered source over
    the trial window, for each of the four bands.
  - "Artefact ?" = "Yes" if the source is flagged by the automatic EOG/EMG
    detection (same thresholds/logic as clean_ICA.py), else "No".
  - Trial = chronological index within the recording (time order).

It also adds a single long-format sheet "Corr" with, per session / nClass /
MODEL / band / task / source, the SIGNED point-biserial PEARSON correlation in
[-1, 1] between the source's per-trial energy and the task's one-vs-rest
membership. It is computed SEPARATELY per model — Orig and Finetune are NOT
combined into one correlation (different model conditions). Pearson (not
rank-based) is used so the actual energy VALUES matter; the sign tells the
direction (+ = more energy for that task). The same matrices are saved as
heatmaps under <...>/Energy Correlation Maps/S<XX>/<Band>/ named
..._{Model}_{Band}.png — diverging RdBu_r, -1..+1 (+1 red, -1 blue), with that
recording's EOG/EMG artifact sources in red on the x-axis.

Finally it adds an "Enrgy_Corr_Evolution" sheet (3-class): one block per
(band, task, MODEL), with the 128 ICA sources in columns (Source_1..128) and one
row per session (Sess01..). Each cell is that source's signed energy<->task
correlation at that session, for that model.

The subject's OFFLINE ICA is reused for every recording. The workbook is created
if missing and overwritten on re-run (idempotent).

Usage:
    python fill_energy_table.py <subj> [<task>]
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

import corr_heatmap as ch   # reuse _load_offline_ica (sets matplotlib backend)
import matplotlib.pyplot as plt

MAX_SESSIONS = 5

HEADER = ["Sujet_id", "Session", "nClass", "Model", "Task", "Class", "Trial",
          "ICA_Source", "Energy_Fullband", "Energy_Alpha",
          "Energy_Beta", "Energy_Beta+2", "Artefact ?"]

# Long-format sheet summarising, per session / nClass / band / task / source,
# the energy<->task correlation computed over ALL trials of that session+nClass
# (Orig and Finetune trials combined).
CORR_HEADER = ["Session", "nClass", "Model", "Band", "Task", "ICA_Source",
               "Corr_Energy", "Artefact ?"]

# (label, l_freq, h_freq). None -> broadband (no band-pass).
BANDS = [("Fullband", None, None), ("Alpha", 8.0, 13.0),
         ("Beta", 13.0, 30.0), ("Beta+2", 30.0, 40.0)]

EVOL_SHEET = "Enrgy_Corr_Evolution"   # 3-class energy<->task corr per session
N_SRC_DEFAULT = 128

# Energy-correlation heatmaps go next to the existing correlation maps.
ENERGY_CORR_DIR = os.path.join(os.path.dirname(ch.SAVE_DIR), "Energy Correlation Maps")


def _energy_task_corr(E, trial_classes, tasks):
    """SIGNED point-biserial PEARSON between each source's per-trial energy and
    each task's one-vs-rest membership, across trials. E: (n_trials, n_comp).

    Pearson (not Spearman) is used so the actual ENERGY VALUES matter, not just
    their ranks. The result is the SIGNED correlation in [-1, 1]: +1 when the
    source's energy is consistently HIGHER during that task, -1 when it is
    consistently LOWER. In 2-class the two tasks are exact opposites (one
    membership is the complement of the other, so the signs flip); in 3-class
    the one-vs-rest contrasts differ."""
    n_trials, n_comp = E.shape
    # Z-score each source's energy across trials (raw energies, magnitude-aware
    # — unlike a rank-based metric this keeps how much energy differs, not
    # just its rank).
    Ez = E - E.mean(axis=0, keepdims=True)        # raw energies (magnitude-aware)
    es = Ez.std(axis=0, keepdims=True)
    es[es == 0] = 1.0                              # avoid divide-by-zero on constant sources
    Ez = Ez / es
    cls_arr = np.asarray(trial_classes)
    out = np.zeros((len(tasks), n_comp))
    for ti, T in enumerate(tasks):
        # One-vs-rest task membership vector, also standardised, so the dot
        # product with Ez / n_trials is exactly the Pearson correlation
        # coefficient between energy and task membership.
        m = (cls_arr == T).astype(float)
        m -= m.mean()
        ms = m.std()
        if ms == 0:
            continue
        out[ti] = (m / ms) @ Ez / n_trials                      # signed Pearson
    return out


def _plot_signed_heatmap(mat, tasks_list, subj_id, task, mode, band_label,
                         save_path, excluded=None):
    """Save a signed energy<->task correlation heatmap: tasks x sources, diverging
    RdBu_r colormap centered at 0, scale -1..+1 (+1 red, -1 blue). EOG/EMG
    artifact source indices get a red x-axis label."""
    excluded = set(excluded or [])
    n_tasks, n_comp = mat.shape
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_yticks(range(n_tasks)); ax.set_yticklabels(tasks_list)
    ax.set_xticks(range(n_comp))
    xt = ax.set_xticklabels([str(c) for c in range(n_comp)], fontsize=6, rotation=90)
    for c, lbl in zip(range(n_comp), xt):
        if c in excluded:
            lbl.set_color("red")
    ax.set_xlabel("ICA Source")
    ax.set_ylabel("Task")
    ax.set_title(f"S{subj_id:02} | {task} | {mode} | Energy Correlation in the band {band_label}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _chronological_trials(raw):
    """List of (class_name, start_sample, end_sample) sorted by onset (signal
    time order). A task annotation runs to its duration, else to the next
    TrialEnd, else 3 s — same logic as build_task_vectors, per trial."""
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
    """Load one recording, compute per-trial per-source energy in every band,
    and build the flat (trial x source) rows for the energy sheet.

    Returns a dict with:
      rows           list of HEADER-shaped row values, one per (trial, source)
      E_per_band     list (one per BANDS entry) of (n_trials, n_comp) energy arrays
      trial_classes  finger label of each trial, chronological order
      artifacts      set of auto-detected EOG/EMG artefact source indices
      n_comp         number of ICA sources
    """
    folder = get_folder(subj_id, task, session, nclass, model)
    mat_files = sorted(glob.glob(os.path.join(folder, "*.mat")))
    raw, _, _ = build_raw_from_mat_files(mat_files)
    raw.notch_filter(np.arange(60, 501, 60), verbose=False)

    ica = ch._load_offline_ica(subj_id, task)
    src_raw = ica.get_sources(raw)
    n_comp = src_raw.ch_names.__len__()

    # Auto EOG/EMG artifact detection (same as clean_ICA.py).
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
    # E_per_band[b]: (n_trials, n_comp) sum-of-squares energy per trial/source.
    E_per_band = [np.empty((n_trials, n_comp)) for _ in BANDS]

    rows = []
    for t_idx, (cls, s, e) in enumerate(trials, start=1):
        # Energy (sum of squared samples) of every source over this trial's
        # sample window [s:e), computed separately for each band.
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


# ── ODS construction from scratch (one workbook per subject) ──────────────────

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
    """Serialise one Python value (str/int/float) as an ODF table:table-cell
    element, choosing the string vs. float value-type accordingly."""
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
    """sheets = list of (sheet_name, header_list_or_None, rows_list).

    If header is None/empty the rows are written without a separate header row
    (used by the multi-block Evolution sheet)."""
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
        # mimetype must be the first entry and stored uncompressed.
        zf.writestr("mimetype", _MIMETYPE, compress_type=zipfile.ZIP_STORED)
        zf.writestr("META-INF/manifest.xml", _MANIFEST)
        zf.writestr("styles.xml", _STYLES)
        zf.writestr("content.xml", content.encode("utf-8"))
    os.replace(tmp, path)


MODELS = ["Orig", "Finetune"]


def _detect_groups(subj_id, task):
    """{(session, nclass): [models present]} — one sheet per (session, nclass),
    merging the Orig and Finetune recordings. So up to 2 sheets/session
    (2c, 3c) -> 10 for 5 sessions."""
    groups = {}
    for sess in range(1, MAX_SESSIONS + 1):
        for ncl in (2, 3):
            models = []
            for model in MODELS:
                if (subj_id, task, sess, ncl, model) in CORRUPTED_DATA:
                    continue            # skip known-corrupted recordings
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
    """Build the full per-subject energy workbook: one sheet per (session,
    nClass) with per-trial per-source energies, a long-format "Corr" sheet of
    energy<->task correlations, an "Enrgy_Corr_Evolution" sheet (3-class), and
    the accompanying energy-correlation heatmap PNGs. Overwrites the ODS file
    if it already exists (idempotent full rebuild)."""
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

    # Detect artifact sources once from the offline recording so the same set
    # of red-marked sources is used across all sessions and models.
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
    corr_rows = []          # long-format rows for the single "Corr" sheet
    evol = {}               # (sess, band, task, model) -> corr vector, 3-class only
    evol_sessions = set()
    evol_tasks = set()
    evol_models = set()
    n_comp_evol = N_SRC_DEFAULT
    for (sess, ncl), models in sorted(groups.items()):
        name = f"Sess_{sess:02d}_{ncl}class"
        print(f"\n  == {name} ({', '.join(models)}) ==", flush=True)

        sheet_rows = []
        # Keep each model's data SEPARATE: correlations are computed per model
        # (combining Orig + Finetune trials in one correlation is not valid —
        # they are different model conditions). The energy sheet still merges
        # both models' trials (distinguished by the Model column).
        per_model = {}      # model -> (E_per_band, trial_classes, artifacts, n_comp)
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

        # Per-model correlation, heatmaps, and evolution feed.
        for model, (E_per_band, classes, artifacts, n_comp) in per_model.items():
            if not classes:
                continue
            tasks = [t for t in TASK_LABELS if t in set(classes)]
            mode = f"Sess{sess:02d} {ncl}class {model} | EnergyCorr"
            for bi, (band_label, _lo, _hi) in enumerate(BANDS):
                mat = _energy_task_corr(E_per_band[bi], classes, tasks)
                for ti, T in enumerate(tasks):
                    for c in range(n_comp):
                        corr_rows.append([
                            sess, ncl, model, band_label, T, c,
                            float(mat[ti, c]),
                            "Yes" if c in offline_artifacts else "No",
                        ])
                    if ncl == 3:               # feed the Evolution sheet
                        evol[(sess, band_label, T, model)] = mat[ti]
                model_tag = "1_Orig" if model == "Orig" else "2_Finetune"
                fname     = (f"EnergyCorr_S{subj_id:02d}_{task}_Sess{sess:02d}"
                             f"_{ncl}class_{model_tag}_{band_label}.png")
                save_path = os.path.join(ENERGY_CORR_DIR,
                                         f"S{subj_id:02d}", band_label, fname)
                # Path used by _gen_energy_images.py (extra {ncl}class subfolder).
                gen_path  = os.path.join(ENERGY_CORR_DIR,
                                         f"S{subj_id:02d}", band_label,
                                         f"{ncl}class", fname)
                if os.path.exists(save_path) or os.path.exists(gen_path):
                    print(f"      [skip] heatmap already exists: {fname}")
                else:
                    try:
                        _plot_signed_heatmap(mat, tasks, subj_id, task, mode,
                                             band_label, save_path,
                                             excluded=offline_artifacts)
                    except Exception as ex:
                        print(f"      [warn] heatmap failed ({model} {band_label}): {ex}")
            if ncl == 3:
                evol_sessions.add(sess)
                evol_tasks.update(tasks)
                evol_models.add(model)
                n_comp_evol = n_comp

    sheets.append(("Corr", CORR_HEADER, corr_rows))

    # ── Evolution sheet (3-class): one block per (band, task, MODEL); sources
    #    in columns (Source_1..N), one row per session, value = signed corr. ──
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
                    evol_rows.append([])      # blank separator between blocks
        sheets.append((EVOL_SHEET, None, evol_rows))

    total = sum(len(r) for _n, _h, r in sheets)
    print(f"\n  Writing {out_path} — {len(sheets)} sheet(s), {total} data row(s); "
          f"heatmaps -> {ENERGY_CORR_DIR}\\S{subj_id:02d}\\ ...", flush=True)
    build_subject_ods(out_path, sheets)
    print("  Done.")


def main():
    """CLI entry point: parse argv and build the requested subject's workbook."""
    if len(sys.argv) < 2:
        print("Usage: python fill_energy_table.py <subj> [<task>]")
        sys.exit(1)
    subj_id = int(sys.argv[1])
    task    = sys.argv[2] if len(sys.argv) > 2 else "MI"
    fill_subject(subj_id, task)


if __name__ == "__main__":
    main()

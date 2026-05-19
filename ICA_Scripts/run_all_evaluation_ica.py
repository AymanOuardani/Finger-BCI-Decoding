"""
run_all_evaluation_ica.py

Calls eval_ica_model.py for each subject in sequence,
writes Online Val Accuracy + Online Performance Accuracy into the
ICA table (cols 9–14) of the matching per-condition sheet in Tracking.ods.
"""

import subprocess
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    TRACKING_ODS,
    ICA_MODEL_START, ROW_SUBJECT_OFFSET,
)
from Functions import write_to_ods

ICA_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE        = os.path.join(ICA_SCRIPTS_DIR, 'evaluation_ica_log.txt')

# ── Configuration — edit these to run a different combo ──────────────────────
SESSION_NUM = 1
NCLASS      = 3
TASK        = 'MI'
MODELTYPE   = 'Finetune'    # "Orig" = Base Model | "Finetune" = Fine Tune
SUBJECTS    = list(range(1, 22))
# ─────────────────────────────────────────────────────────────────────────────


def run_subject(subj_id):
    cmd = [
        'python', 'eval_ica_model.py',
        str(subj_id), str(SESSION_NUM), str(NCLASS), TASK, MODELTYPE
    ]
    cmd_str = ' '.join(cmd)

    print(f'\n{"="*60}')
    print(f'Subject S{subj_id:02} — {cmd_str}')
    print(f'{"="*60}', flush=True)

    live_keywords = ['accuracy', 'Accuracy', 'Confusion', 'saved', 'ERROR', 'error']
    live_skip     = ['WARNING', 'UserWarning', 'absl', 'TensorFlow', 'keras',
                     'recommend', 'legacy', 'RESULT:']

    all_lines   = []
    result_line = None

    proc = subprocess.Popen(
        cmd, cwd=ICA_SCRIPTS_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )

    for line in proc.stdout:
        line_stripped = line.rstrip()
        all_lines.append(line_stripped)
        if line_stripped.startswith('RESULT:'):
            result_line = line_stripped
        elif (any(k in line_stripped for k in live_keywords)
              and not any(s in line_stripped for s in live_skip)):
            print(f'  {line_stripped}', flush=True)

    proc.wait()

    # Parse accuracy values from the RESULT line printed by eval_ica_model.py
    online_val = online_perf = None
    if result_line:
        try:
            parts = dict(p.split('=') for p in result_line[len('RESULT:'):].split(':'))
            online_val  = float(parts['online_val'])
            online_perf = float(parts['online_perf'])
        except Exception:
            pass

    log_keywords = ['accuracy', 'Accuracy', 'saved', 'ERROR']
    useful_lines = [
        l for l in all_lines
        if any(k in l for k in log_keywords)
        and not any(s in l for s in live_skip)
    ]
    display_lines = useful_lines[-10:] if useful_lines else all_lines[-10:]

    return cmd_str, display_lines, proc.returncode, online_val, online_perf


def main():
    per_sheet = f"{TASK}_Sess{SESSION_NUM:02}_{NCLASS}Class"
    base_col  = ICA_MODEL_START[MODELTYPE]
    col_val   = base_col + 1   # Online Val_Accuracy
    col_perf  = base_col + 2   # Online Perf_Accuracy

    print(f"ICA Evaluation : {TASK}  {NCLASS}-class  {MODELTYPE}  Session {SESSION_NUM}")
    print(f"Sheet          : '{per_sheet}'  |  Cols : {col_val} (Online Val) / {col_perf} (Online Perf)")
    print()

    with open(LOG_FILE, 'w', encoding='utf-8') as log:
        log.write(f'ICA Evaluation Log — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        log.write(f'Config : session={SESSION_NUM}, nclass={NCLASS}, task={TASK}, modeltype={MODELTYPE}\n')
        log.write('='*60 + '\n\n')

        for subj_id in SUBJECTS:
            cmd_str, display_lines, returncode, online_val, online_perf = run_subject(subj_id)

            status = '✅ OK' if returncode == 0 else f'❌ Error (code {returncode})'
            print(f'Status : {status}')

            if online_val is not None and online_perf is not None:
                row_idx = subj_id + ROW_SUBJECT_OFFSET
                write_to_ods(
                    TRACKING_ODS, row_idx,
                    [(col_val, online_val), (col_perf, online_perf)],
                    table_name=per_sheet,
                )
                print(f"  -> ODS [{per_sheet}] S{subj_id:02} "
                      f"col {col_val}={online_val:.4f}% "
                      f"col {col_perf}={online_perf:.4f}%")
            else:
                print(f"  -> ODS skipped (no accuracy found in output)")

            log.write(f'Subject S{subj_id:02}\n')
            log.write(f'Command : {cmd_str}\n')
            log.write(f'Status  : {status}\n')
            log.write('Last 10 lines :\n')
            for line in display_lines:
                log.write(f'  {line}\n')
            log.write('\n' + '-'*60 + '\n\n')
            log.flush()

    print(f'\n\nDone. Log saved to : {LOG_FILE}')


if __name__ == '__main__':
    main()

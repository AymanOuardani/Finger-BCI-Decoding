"""
run_all_training.py

Runs main_model_training.py for all subjects and saves
the command + last 10 lines of output for each run.
"""

import subprocess
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime

from config import SCRIPT_DIR

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE     = os.path.join(SCRIPT_DIR, 'training_log.txt')

SESSION_NUM = 2
NCLASS      = 3
TASK        = 'ME'
MODELTYPE   = 'Finetune'

SUBJECTS    = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]  # 1 to 21 inclusive
# ─────────────────────────────────────────────────────────────────

def run_subject(subj_id):
    cmd = [
        'python', 'main_model_training.py',
        str(subj_id), str(SESSION_NUM), str(NCLASS), TASK, MODELTYPE
    ]
    cmd_str = ' '.join(cmd)

    print(f'\n{"="*60}')
    print(f'Subject S{subj_id:02} — {cmd_str}')
    print(f'{"="*60}', flush=True)

    # Lines to show live in the terminal while training runs
    live_keywords = ['Epoch', 'epoch', 'val_accuracy', 'val_acc', 'loss',
                     'accuracy', 'Accuracy', 'Model saved', 'Training Finished',
                     'train samples']
    live_skip     = ['WARNING', 'UserWarning', 'super()', 'absl', 'TensorFlow',
                     'keras', 'recommend', 'legacy']

    all_lines = []

    proc = subprocess.Popen(
        cmd, cwd=PIPELINE_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )

    for line in proc.stdout:
        line_stripped = line.rstrip()
        all_lines.append(line_stripped)

        # Print live to terminal if it looks useful, skip noise
        if (any(k in line_stripped for k in live_keywords)
                and not any(s in line_stripped for s in live_skip)):
            print(f'  {line_stripped}', flush=True)

    proc.wait()

    # Build the filtered summary for the log (last 10 useful lines)
    log_keywords = ['accuracy', 'Accuracy', 'val_accuracy', 'loss', 'Model saved',
                    'Training Finished', 'Epoch', 'train samples']
    useful_lines = [
        l for l in all_lines
        if any(k in l for k in log_keywords)
        and not any(s in l for s in live_skip)
    ]
    display_lines = useful_lines[-10:] if useful_lines else all_lines[-10:]

    return cmd_str, display_lines, proc.returncode

def main():
    with open(LOG_FILE, 'w', encoding='utf-8') as log:
        log.write(f'Training Log — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        log.write(f'Config : session={SESSION_NUM}, nclass={NCLASS}, task={TASK}, modeltype={MODELTYPE}\n')
        log.write('='*60 + '\n\n')

        for subj_id in SUBJECTS:
            cmd_str, last_10, returncode = run_subject(subj_id)

            # print to terminal
            print('\nLast 10 lines of output :')
            for line in last_10:
                print(f'  {line}')
            status = '✅ OK' if returncode == 0 else f'❌ Error (code {returncode})'
            print(f'Status : {status}')

            # write to log file
            log.write(f'Subject S{subj_id:02}\n')
            log.write(f'Command : {cmd_str}\n')
            log.write(f'Status  : {status}\n')
            log.write('Last 10 lines :\n')
            for line in last_10:
                log.write(f'  {line}\n')
            log.write('\n' + '-'*60 + '\n\n')
            log.flush()  # write immediately in case of crash

    print(f'\n\nDone. Log saved to : {LOG_FILE}')


if __name__ == '__main__':
    main()
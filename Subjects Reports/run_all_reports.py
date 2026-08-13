"""
run_all_reports.py

Batch runner — calls generate_subject_report.py for each subject in SUBJECTS.
Edit the SUBJECTS list below before running.

Each subject adds two sheets (S{XX}_Accuracy and S{XX}_Figures) to the shared
workbook Ressources/Subject_Reports.xlsx; sheets already written for other
subjects are left untouched.

Usage:
    python run_all_reports.py
"""

import os
import subprocess
import sys

SUBJECTS = list(range(3, 22))   # ← edit this list

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "generate_subject_report.py")

failed = []

for subj_id in SUBJECTS:
    print(f"\n{'='*60}")
    print(f"  Generating report for Subject S{subj_id:02}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--subj", str(subj_id)],
        check=False
    )
    if result.returncode != 0:
        print(f"  [WARN] Subject S{subj_id:02} finished with errors — continuing.")
        failed.append(subj_id)

print("\nAll subjects done.")
if failed:
    print("  finished with errors: " +
          ", ".join(f"S{s:02}" for s in failed))

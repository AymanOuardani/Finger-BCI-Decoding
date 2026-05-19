"""
run_all_reports.py

Batch runner — calls generate_subject_report.py for each subject in SUBJECTS.
Edit the SUBJECTS list below before running.

Usage:
    python run_all_reports.py
"""

import subprocess
import sys

SUBJECTS = list(range(3, 22))   # ← edit this list

for subj_id in SUBJECTS:
    print(f"\n{'='*60}")
    print(f"  Generating report for Subject S{subj_id:02}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, "generate_subject_report_V6.py", "--subj", str(subj_id)],
        check=False
    )
    if result.returncode != 0:
        print(f"  [WARN] Subject S{subj_id:02} finished with errors — continuing.")

print("\nAll subjects done.")
"""
viz_compare.py

Side-by-side comparison of original raw signals and the already-generated
ICA-cleaned dataset (*_ICA.mat). Does NOT fit new ICA.
Requires clean_ICA.py to have been run first.

Usage:
    Online:  python viz_compare.py <subj> <sess> <nclass> <task> <model>
    Offline: python viz_compare.py <subj> <task> OFF
"""

import sys
import os
import matplotlib
matplotlib.use("TkAgg")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Functions import (
    load_online_raw_pair, load_offline_raw_pair, launch_viewer,
    get_standard_args, get_offline_args,
)


def compare_online(subj_id, session, nclass, task, model_type):
    print("=" * 60)
    print(f"  Online Viewer: S{subj_id:02} | {task} | Session {session}")
    print("=" * 60)
    raw, raw_clean = load_online_raw_pair(subj_id, task, session, nclass, model_type)
    launch_viewer(raw, raw_clean, subj_id, task, session, nclass, model_type)


def compare_offline(subj_id, task):
    print("=" * 60)
    print(f"  Offline Viewer: S{subj_id:02} | {task}")
    print("=" * 60)
    raw, raw_clean = load_offline_raw_pair(subj_id, task)
    launch_viewer(raw, raw_clean, subj_id, task)


def main():
    try:
        if len(sys.argv) > 1 and sys.argv[-1].upper() == "OFF":
            sys.argv.pop()
            subj_id, task = get_offline_args(description="Raw vs ICA Viewer - Offline")
            compare_offline(subj_id, task)
        else:
            subj_id, session, nclass, task, model_type = get_standard_args(
                description="Raw vs ICA Viewer - Online"
            )
            compare_online(subj_id, session, nclass, task, model_type)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

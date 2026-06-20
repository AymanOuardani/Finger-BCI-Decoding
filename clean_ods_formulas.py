"""
clean_ods_formulas.py

Repairs ODF formula corruption in an .ods file.

LibreOffice duplicates the OpenFormula namespace prefix every time it saves
(``of:=SUM(...)`` -> ``of:=of:=SUM(...)``), which it then displays as
``=of:=SUM(...)``. This rewrites every ``table:formula`` value back to a single
leading prefix, leaving already-clean formulas untouched.

Usage
-----
  One-shot (clean once and exit):
    python clean_ods_formulas.py                 # cleans config.TRACKING_ODS
    python clean_ods_formulas.py path/to/file.ods

  Watch (auto-fix on every save — leave it running while you edit in LibreOffice):
    python clean_ods_formulas.py --watch
    python clean_ods_formulas.py path/to/file.ods --watch --interval 1.5
"""

import os
import re
import sys
import time
import zipfile
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Functions import _normalize_formula_prefixes
from config import TRACKING_ODS

_DOUBLED = re.compile(r'table:formula="(?:of:=|oooc:=){2,}')


def clean_ods(path):
    """Normalize duplicated formula prefixes in `path`.

    Returns the number of corrupted formulas that were fixed (0 = already clean,
    nothing written). Reads/writes content.xml in place, preserving every other
    entry of the .ods zip.
    """
    with zipfile.ZipFile(path) as z:
        data = {n: z.read(n) for n in z.namelist()}

    content = data["content.xml"].decode("utf-8")
    n_bad = len(_DOUBLED.findall(content))
    if n_bad == 0:
        return 0

    data["content.xml"] = _normalize_formula_prefixes(content).encode("utf-8")
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for name, raw in data.items():
            z.writestr(name, raw)
    os.replace(tmp, path)          # atomic swap
    return n_bad


def watch(path, interval):
    """Poll `path` and re-clean it whenever it changes (e.g. a LibreOffice save)."""
    print(f"Watching: {path}\n  interval {interval}s — Ctrl+C to stop.", flush=True)
    last_sig = None
    while True:
        try:
            if os.path.exists(path):
                st  = os.stat(path)
                sig = (st.st_mtime, st.st_size)
                if sig != last_sig:
                    # The file just changed. Wait briefly and confirm it has
                    # stopped changing, so we don't read a half-written save.
                    time.sleep(0.4)
                    st2 = os.stat(path)
                    if (st2.st_mtime, st2.st_size) != sig:
                        continue                 # still being written; retry
                    try:
                        n = clean_ods(path)
                    except (zipfile.BadZipFile, KeyError, FileNotFoundError):
                        continue                 # incomplete/locked; retry next tick
                    if n:
                        print(f"  [{time.strftime('%H:%M:%S')}] fixed {n} "
                              f"corrupted formula(s)", flush=True)
                    # Record the signature AFTER our own write so we don't react
                    # to the change we just made.
                    s = os.stat(path)
                    last_sig = (s.st_mtime, s.st_size)
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


def main():
    ap = argparse.ArgumentParser(description="Fix duplicated ODF formula prefixes in an .ods file.")
    ap.add_argument("ods", nargs="?", default=TRACKING_ODS,
                    help="path to the .ods file (default: config.TRACKING_ODS)")
    ap.add_argument("--watch", action="store_true",
                    help="keep running and auto-fix on every save")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="polling interval in seconds for --watch (default: 2)")
    args = ap.parse_args()

    if args.watch:
        watch(args.ods, args.interval)
    else:
        n = clean_ods(args.ods)
        print(f"Fixed {n} corrupted formula(s) in:\n  {args.ods}" if n
              else f"Already clean:\n  {args.ods}")


if __name__ == "__main__":
    main()

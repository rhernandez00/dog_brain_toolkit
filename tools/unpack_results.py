#!/usr/bin/env python
"""unpack_results.py -- merge Colab result zips back onto the pipeline data disk.

The Colab GPU run (see tools/colab_gpu/) writes one ``result_*.zip`` per finished
part, each holding niftis whose arc-paths are already pipeline-relative to the data
folder (e.g. ``EmoC/results/RSA/basic-block/H-sub-40/r-4_mahalanobis_DogA_DogF.nii.gz``).
Unpacking is therefore a validated merge: extract each member to
``{datafolder}/{arcname}``. Afterwards steps 3-10 of ``searchlight.py`` run exactly
as if the maps had been computed on the workstation.

Usage (from the repo root, full Anaconda interpreter -- see CLAUDE.md):

    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\unpack_results.py DOWNLOADS_DIR
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\unpack_results.py result_step1_H-sub-40.zip --dry-run

Accepts any mix of ``result_*.zip`` files and directories containing them. Existing
files are left untouched unless ``--replace`` is given; ``--dry-run`` reports the
planned copies without writing anything.
"""

import argparse
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from scheduler.paths import get_paths  # noqa: E402


def collect_zips(inputs):
    """Expand files/dirs into a sorted list of result_*.zip paths."""
    zips = []
    for item in inputs:
        if os.path.isdir(item):
            for name in sorted(os.listdir(item)):
                if name.lower().endswith(".zip"):
                    zips.append(os.path.join(item, name))
        elif os.path.isfile(item) and item.lower().endswith(".zip"):
            zips.append(item)
        else:
            print(f"WARNING: skipping {item!r} (not a .zip or directory)")
    return zips


def _safe_member(name):
    """Reject absolute paths and parent-directory escapes; keep only .nii.gz."""
    norm = name.replace("\\", "/")
    if norm.endswith("/"):
        return None
    if os.path.isabs(norm) or ".." in norm.split("/"):
        raise ValueError(f"Unsafe path in zip: {name!r}")
    if not norm.endswith(".nii.gz"):
        return None
    return norm


def unpack_zip(zip_path, datafolder, dataset=None, replace=False, dry_run=False,
               verbose=True):
    """Extract one result zip into ``datafolder``. Returns (written, skipped)."""
    written = skipped = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            member = _safe_member(info.filename)
            if member is None:
                continue
            if dataset and member.split("/")[0] != dataset:
                if verbose:
                    print(f"  (skip {member}: not dataset {dataset})")
                continue
            dst = os.path.join(datafolder, member.replace("/", os.sep))
            if os.path.exists(dst) and not replace:
                skipped += 1
                if verbose:
                    print(f"  exists, skip: {member}")
                continue
            if dry_run:
                written += 1
                if verbose:
                    print(f"  would write: {member}")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with zf.open(info) as src, open(dst, "wb") as out:
                out.write(src.read())
            written += 1
            if verbose:
                print(f"  wrote: {member}")
    return written, skipped


def parse_args():
    ap = argparse.ArgumentParser(description="Merge Colab result zips onto the data disk.")
    ap.add_argument("inputs", nargs="+", help="result_*.zip files and/or directories")
    ap.add_argument("--dataset", default=None, help="Only unpack members of this dataset")
    ap.add_argument("--datafolder", default=None,
                    help="Target data folder (default: machine's pipeline data disk)")
    ap.add_argument("--replace", action="store_true", help="Overwrite existing files")
    ap.add_argument("--dry-run", action="store_true", help="Report without writing")
    return ap.parse_args()


def main():
    a = parse_args()
    datafolder = a.datafolder or get_paths()[0]
    zips = collect_zips(a.inputs)
    if not zips:
        print("No result zips found.")
        return
    print(f"Target datafolder: {datafolder}")
    print(f"{'DRY RUN -- ' if a.dry_run else ''}unpacking {len(zips)} zip(s)\n")
    tot_w = tot_s = 0
    for z in zips:
        print(os.path.basename(z))
        w, s = unpack_zip(z, datafolder, dataset=a.dataset, replace=a.replace,
                          dry_run=a.dry_run, verbose=True)
        tot_w += w
        tot_s += s
    verb = "would write" if a.dry_run else "wrote"
    print(f"\nDone: {verb} {tot_w} file(s), skipped {tot_s} existing.")


if __name__ == "__main__":
    main()

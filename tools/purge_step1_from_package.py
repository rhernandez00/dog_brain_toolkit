#!/usr/bin/env python
"""purge_step1_from_package.py -- strip bundled step-1 maps from existing Colab
packages in place, so Colab recomputes step 1 without re-uploading the .zip.

create_package.py bundles step-1 maps into pkg_*.zip when they pass the
freshness gate (see its docstring) -- Colab then skips step 1 and those maps,
not the verified betas beside them, feed every model. If a package already
sits on Google Drive and you decide that reuse should not have happened,
re-running create_package.py with --no_reuse_step1 means rebuilding the whole
(often multi-GB) zip and re-copying it across the network -- slow. This
rewrites the zip in place instead: drops every bundled step-1 map, flips
manifest.json's step1_done to False, and atomically replaces the file.

Colab's run_colab.py (_step1_present) decides whether to recompute step 1 by
globbing for the maps on disk after unzipping, not by trusting the manifest
flag -- so removing the files is what actually forces the recompute. The
manifest edit just keeps step1_done truthful for anyone reading it later.

Usage (Anaconda Prompt):

    python \\github\\dog_brain_toolkit\\tools\\purge_step1_from_package.py --dir "G:\\My Drive\\rsa_colab\\pkg_corr"
    python \\github\\dog_brain_toolkit\\tools\\purge_step1_from_package.py --dir "G:\\My Drive\\rsa_colab\\pkg_corr" --dry_run
"""

import argparse
import glob
import json
import os
import re
import tempfile
import zipfile

# Matches exactly the arcnames step1_maps_on_disk/step1_corr_maps_on_disk give
# bundled maps in create_package.py, e.g. "r-3_mahalanobis_DogP_HumN.nii.gz" or
# "r-4_correlation_DogP1_HumN2.nii.gz" -- nothing else in a package looks like this.
STEP1_RE = re.compile(r"^r-\d+_(mahalanobis|correlation)_.+\.nii\.gz$", re.IGNORECASE)


def is_step1_arcname(arcname):
    return bool(STEP1_RE.match(os.path.basename(arcname)))


def purge_one(zip_path, dry_run=False, verbose=True):
    """Remove bundled step-1 maps from one package zip. Returns (n_maps, bytes_reclaimed)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        step1_names = {n for n in names if is_step1_arcname(n)}
        if not step1_names:
            if verbose:
                print(f"  {os.path.basename(zip_path)}: no bundled step-1 maps -- skipping.")
            return 0, 0
        manifest_bytes = zf.read("manifest.json") if "manifest.json" in names else None
        infolist = zf.infolist()

    size_before = os.path.getsize(zip_path)
    # ZIP_STORED, so on-disk bytes == file_size -- this comes from the central
    # directory already in `infolist`, no need to read the actual map data.
    step1_bytes = sum(item.file_size for item in infolist if item.filename in step1_names)
    if verbose:
        print(f"  {os.path.basename(zip_path)}: {len(step1_names)} bundled step-1 maps, "
              f"{step1_bytes / 1e6:.1f} MB of {size_before / 1e6:.1f} MB package -- "
              f"{'would purge' if dry_run else 'purging'}.")
    if dry_run:
        return len(step1_names), step1_bytes

    manifest = json.loads(manifest_bytes) if manifest_bytes else None
    if manifest is not None:
        manifest["step1_done"] = False
        manifest["step1_reason"] = "purged post-hoc by purge_step1_from_package.py"

    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(zip_path) or ".", suffix=".zip.tmp")
    os.close(fd)
    try:
        with zipfile.ZipFile(zip_path) as src, \
             zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED, allowZip64=True) as dst:
            for item in infolist:
                if item.filename in step1_names:
                    continue
                data = src.read(item.filename)
                if item.filename == "manifest.json" and manifest is not None:
                    data = json.dumps(manifest, indent=2).encode("utf-8")
                dst.writestr(item, data)
        size_after = os.path.getsize(tmp_path)
        os.replace(tmp_path, zip_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return len(step1_names), max(size_before - size_after, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=r"G:\My Drive\rsa_colab\pkg_corr",
                     help=r"Folder of pkg_*.zip files to purge (default: G:\My Drive\rsa_colab\pkg_corr).")
    ap.add_argument("--dry_run", action="store_true",
                     help="Report what would be purged without modifying anything.")
    a = ap.parse_args()

    zips = sorted(glob.glob(os.path.join(a.dir, "pkg_*.zip")))
    if not zips:
        print(f"No pkg_*.zip files found in {a.dir}")
        return

    total_maps, total_bytes, n_touched = 0, 0, 0
    for zip_path in zips:
        try:
            n, reclaimed = purge_one(zip_path, dry_run=a.dry_run)
        except (zipfile.BadZipFile, OSError) as exc:
            print(f"  {os.path.basename(zip_path)}: FAILED -- {exc}")
            continue
        if n:
            n_touched += 1
        total_maps += n
        total_bytes += reclaimed

    verb = "would touch" if a.dry_run else "touched"
    print(f"\n{verb} {n_touched}/{len(zips)} package(s), {total_maps} step-1 map(s), "
          f"{total_bytes / 1e6:.1f} MB reclaimed.")


if __name__ == "__main__":
    main()

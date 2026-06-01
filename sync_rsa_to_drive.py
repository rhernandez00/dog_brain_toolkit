"""Sync new RSA results to the Google Drive folder.

Scans the RSA results tree for finished model folders and copies their result
artifacts into the mounted Google Drive (Google Drive for Desktop) under
``current-results``, renaming each file from its long technical name to the
short ``{specie}_{rsa_model}`` convention used by ``rsa_utils.create_tables``.

The headline artifact is the **unthresholded** z-map (``*_z.nii.gz``, i.e. the
corrected map's name minus ``_corrected``) so the dashboard viewer's threshold
slider can explore the data freely. The corrected z-map and the Excel table are
also mirrored when present.

Naming convention (must stay in sync with rsa_utils.create_tables)
------------------------------------------------------------------
Source (per model):
    {mean}/{mask_type}-{specie}-r-{radius}_{method}_{rsa_method}_z.nii.gz   (unthresholded)
    {mean}/{mask_type}-{specie}-r-{radius}_{method}_{rsa_method}_z_corrected.nii.gz
    {mean}/{mask_type}-{specie}-r-{radius}_{method}_{rsa_method}.xlsx
Destination (mask_type present):
    {drive_root}/{dataset}/current-results/RSA/{specie}/{mask_type}/{specie}_{rsa_model}_z.nii.gz
    {drive_root}/{dataset}/current-results/RSA/{specie}/{mask_type}/{specie}_{rsa_model}_z_corrected.nii.gz
    {drive_root}/{dataset}/current-results/RSA/{specie}/{mask_type}/{specie}_{rsa_model}.xlsx
Destination (no mask_type):
    {drive_root}/{dataset}/current-results/RSA/{specie}_{rsa_model}_z.nii.gz   (etc.)

"Update" semantics: a file is copied only when it is missing at the destination
or the source is newer (mtime). Use --force to copy regardless, --dry-run to
preview.

Usage
-----
    & "C:\\ProgramData\\anaconda3\\python.exe" sync_rsa_to_drive.py
    & "C:\\ProgramData\\anaconda3\\python.exe" sync_rsa_to_drive.py --dry-run
    & "C:\\ProgramData\\anaconda3\\python.exe" sync_rsa_to_drive.py --models test-model happiness-anticipation-dog
"""

import argparse
import os
import re
import shutil
import sys

try:
    from scheduler.paths import get_paths
except Exception:  # pragma: no cover - allow running from elsewhere
    get_paths = None


# Unthresholded z-map filename, e.g. "b_GreyMatter2mmB-D-r-3_correlation_kendall_z.nii.gz"
# mask_type is optional; specie is D or H. We anchor on the "-{specie}-r-{radius}_"
# segment so a mask_type that itself contains separators is captured correctly.
_Z_RE = re.compile(
    r"^(?:(?P<mask>.+)-)?(?P<specie>[DH])-r-(?P<radius>\d+)_"
    r"(?P<method>[A-Za-z0-9]+)_(?P<rsa_method>[A-Za-z0-9]+)_z\.nii\.gz$"
)


def drive_dest_name(specie, rsa_model, kind):
    """Return the destination *filename* for a copied RSA artifact.

    This mirrors rsa_utils.create_tables: the long technical source name is
    replaced by a short name built from the species and the RSA model (folder)
    name.

    kind: 'z' (unthresholded z-map), 'z_corrected' (cluster-corrected z-map),
          or 'xlsx' (table).
    """
    if kind == 'z':
        return f"{specie}_{rsa_model}_z.nii.gz"
    if kind == 'z_corrected':
        return f"{specie}_{rsa_model}_z_corrected.nii.gz"
    if kind == 'xlsx':
        return f"{specie}_{rsa_model}.xlsx"
    raise ValueError(f"unknown artifact kind: {kind!r}")


def drive_dest_dir(drive_root, dataset, specie, mask_type):
    """Return the destination directory on Google Drive for a result.

    Matches create_tables: with a mask_type the files are nested under
    {specie}/{mask_type}; without one they live directly under .../RSA.
    """
    base = os.path.join(drive_root, dataset, "current-results", "RSA")
    if mask_type:
        return os.path.join(base, specie, mask_type)
    return base


def _needs_copy(src, dst, force):
    if force or not os.path.exists(dst):
        return True
    # Copy when source is newer (allow 1s slack for filesystem mtime resolution).
    return os.path.getmtime(src) > os.path.getmtime(dst) + 1


def iter_model_results(rsa_root):
    """Yield (rsa_model, mean_dir, match) for each model folder with an
    unthresholded z-map. ``match`` is the parsed regex match for that z-map."""
    if not os.path.isdir(rsa_root):
        return
    for entry in sorted(os.listdir(rsa_root)):
        mean_dir = os.path.join(rsa_root, entry, "mean")
        if not os.path.isdir(mean_dir):
            continue  # not a model folder (e.g. per-subject D-sub-XX folders)
        for fname in sorted(os.listdir(mean_dir)):
            m = _Z_RE.match(fname)
            if m:
                yield entry, mean_dir, m
                break  # one unthresholded z-map per model folder


def sync(datafolder, dataset, model, drive_root, models=None,
         include_corrected=True, include_xlsx=True, force=False, dry_run=False):
    rsa_root = os.path.join(datafolder, dataset, "results", "RSA", model)
    print(f"Scanning: {rsa_root}")
    print(f"Drive   : {os.path.join(drive_root, dataset, 'current-results', 'RSA')}\n")

    copied = skipped = missing_src = 0

    for rsa_model, mean_dir, m in iter_model_results(rsa_root):
        if models and rsa_model not in models:
            continue
        mask_type = m.group("mask")
        specie = m.group("specie")
        stem = m.group(0)[: -len("_z.nii.gz")]  # technical name minus _z.nii.gz

        # (source filename, destination filename, kind, enabled)
        artifacts = [
            (f"{stem}_z.nii.gz", drive_dest_name(specie, rsa_model, 'z'), 'z', True),
            (f"{stem}_z_corrected.nii.gz",
             drive_dest_name(specie, rsa_model, 'z_corrected'), 'z_corrected',
             include_corrected),
            (f"{stem}.xlsx", drive_dest_name(specie, rsa_model, 'xlsx'), 'xlsx',
             include_xlsx),
        ]

        dest_dir = drive_dest_dir(drive_root, dataset, specie, mask_type)
        printed_header = False

        for src_name, dst_name, kind, enabled in artifacts:
            if not enabled:
                continue
            src = os.path.join(mean_dir, src_name)
            dst = os.path.join(dest_dir, dst_name)

            if not os.path.exists(src):
                if kind == 'z':  # the required artifact really should exist
                    missing_src += 1
                continue

            if not _needs_copy(src, dst, force):
                skipped += 1
                continue

            if not printed_header:
                print(f"[{rsa_model}]  specie={specie}  mask={mask_type or '-'}")
                printed_header = True

            action = "WOULD COPY" if dry_run else "COPY"
            print(f"  {action:10s} {kind:12s} -> {dst}")
            if not dry_run:
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copyfile(src, dst)
            copied += 1

    print(f"\nDone. {'(dry-run) ' if dry_run else ''}"
          f"copied={copied} up_to_date={skipped} missing_z_src={missing_src}")
    return copied


def main(argv=None):
    default_datafolder = get_paths()[0] if get_paths else None
    default_drive = r"G:\My Drive\Results" if os.name == 'nt' else None

    p = argparse.ArgumentParser(description="Sync new RSA results to Google Drive.")
    p.add_argument("--dataset", default="EmoC")
    p.add_argument("--model", default="basic-block", help="GLM model subfolder")
    p.add_argument("--datafolder", default=default_datafolder,
                   help="Root data folder (defaults to scheduler.paths)")
    p.add_argument("--drive_root", default=default_drive,
                   help=r"Google Drive results root (default G:\My Drive\Results)")
    p.add_argument("--models", nargs="*", default=None,
                   help="Only sync these RSA model folders (default: all)")
    p.add_argument("--no-corrected", action="store_true",
                   help="Skip the cluster-corrected z-map")
    p.add_argument("--no-xlsx", action="store_true", help="Skip the Excel table")
    p.add_argument("--force", action="store_true",
                   help="Copy even if destination is up to date")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be copied without copying")
    args = p.parse_args(argv)

    if not args.datafolder:
        p.error("--datafolder is required (scheduler.paths unavailable)")
    if not args.drive_root:
        p.error("--drive_root is required on this platform")

    sync(args.datafolder, args.dataset, args.model, args.drive_root,
         models=args.models, include_corrected=not args.no_corrected,
         include_xlsx=not args.no_xlsx, force=args.force, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())

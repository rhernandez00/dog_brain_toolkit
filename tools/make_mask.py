#!/usr/bin/env python
"""make_mask.py -- put a dataset's searchlight mask on the beta maps' voxel grid.

Steps 1-3 select voxels with ``data[mask_bool]``, so the mask has to sit on the
*same voxel grid* as the beta maps -- same shape AND same affine. There are two
ways to satisfy that, and only one of them is correct:

  * move the beta maps into template space (step 0.5), then put the mask on the
    template grid -- this script; or
  * move the mask down onto whichever native grid the beta maps happen to be on.

EmoC/H shipped with the second: ``ROI/H/`` still contains ``results_space.nii.gz``,
the MNI grey-matter mask resampled onto one participant's 96x96x52 scanner-native
EPI grid. That makes the pipeline run without complaint while the other 39
participants sit up to 72 mm away.

So: run step 0.5 first, then this, then step 1.

Usage (from the repo root, with the full Anaconda interpreter -- see CLAUDE.md):

    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\make_mask.py --dataset EmoC --specie H

    # explicit source, and a dry run that only reports what it would do
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\make_mask.py --dataset EmoC --specie H \\
        --source Atlas/Hum/MNI/b_greyMatter2mmB.nii.gz --dry_run

The reference grid is taken from a ``beta_manifest.json`` written by step 0.5,
so the mask is matched to what step 1 will actually read. Pass ``--reference``
to use a specific image instead.

Exit code is 0 on success, 1 when nothing could be written.
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import rsa_utils                       # noqa: E402
from scheduler.paths import get_paths  # noqa: E402

#: Default mask source per species, relative to the repo root.
DEFAULT_SOURCE = {
    # NOT Atlas/Hum/MNI/b_greyMatter2mmB.nii.gz -- despite the name, that file
    # holds {-1, 0, 1} (145312 voxels > 0), not a clean binary mask. This one is
    # {0, 1} only (171094 voxels) and sits on the identical grid.
    'H': os.path.join('Atlas', 'Hum', 'greyMatter2mm.nii.gz'),
    'D': os.path.join('Atlas', 'Dog', 'Nitzsche', 'b_GreyMatter2mmB.nii.gz'),
}


def find_reference(datafolder, dataset, model, specie, task):
    """Grid the aligned beta maps sit on, read from any step-0.5 manifest."""
    pattern = os.path.join(
        datafolder, dataset, 'results', 'GLM', model,
        f"{specie}-sub-*", f"ses-*_task-{task}_run-*", 'beta_manifest.json'
    )
    manifests = sorted(glob.glob(pattern))
    if not manifests:
        return None, None

    # Every manifest should agree; if they do not, that is the thing to report.
    grids = {}
    for path in manifests:
        with open(path) as f:
            m = json.load(f)
        key = (tuple(m['shape']), tuple(np.round(np.asarray(m['affine']), 3).ravel()))
        grids.setdefault(key, []).append(path)
    if len(grids) > 1:
        raise SystemExit(
            f"ERROR: step-0.5 manifests disagree on the voxel grid "
            f"({len(grids)} distinct grids across {len(manifests)} runs).\n"
            "Re-run step 0.5 -- a mask cannot match all of these at once."
        )

    with open(manifests[0]) as f:
        m = json.load(f)
    ref = nib.Nifti1Image(
        np.zeros(m['shape'], dtype=np.uint8), np.asarray(m['affine'], dtype=float)
    )
    return ref, f"{len(manifests)} step-0.5 manifests (e.g. {manifests[0]})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='Dataset name, e.g. EmoC')
    ap.add_argument('--specie', required=True, choices=['D', 'H'], help="'D' or 'H'")
    ap.add_argument('--model', default='basic-block', help='GLM model folder (default: basic-block)')
    ap.add_argument('--task', default=None, help='Task name (default: same as dataset)')
    ap.add_argument('--mask_type', default='b_GreyMatter2mmB',
                    help='Name written under ROI/{specie}/ (default: b_GreyMatter2mmB)')
    ap.add_argument('--source', default=None,
                    help='Mask to start from; relative paths resolve against the repo root '
                         f'(default: {DEFAULT_SOURCE})')
    ap.add_argument('--reference', default=None,
                    help='Image defining the target grid (default: read from a step-0.5 manifest)')
    ap.add_argument('--threshold', type=float, default=0.5,
                    help='Binarisation threshold applied after resampling (default: 0.5)')
    ap.add_argument('--dry_run', action='store_true', help='Report what would be written, write nothing')
    args = ap.parse_args()

    datafolder, _, _ = get_paths()
    task = args.task or args.dataset

    source = args.source or DEFAULT_SOURCE[args.specie]
    if not os.path.isabs(source):
        source = os.path.join(REPO, source)
    if not os.path.exists(source):
        print(f"ERROR: source mask not found: {source}")
        return 1

    if args.reference:
        if not os.path.exists(args.reference):
            print(f"ERROR: reference image not found: {args.reference}")
            return 1
        ref_img, ref_desc = nib.load(args.reference), args.reference
    else:
        ref_img, ref_desc = find_reference(datafolder, args.dataset, args.model,
                                           args.specie, task)
        if ref_img is None:
            print("ERROR: no step-0.5 output found under "
                  f"{os.path.join(datafolder, args.dataset, 'results', 'GLM', args.model)} "
                  f"for specie {args.specie}.\n"
                  "Run step 0.5 first, or pass --reference explicitly:\n"
                  f"  python searchlight.py --dataset {args.dataset} --model {args.model} "
                  f"--specie {args.specie} --steps_to_run 0.5")
            return 1

    src_img = nib.load(source)
    dest = os.path.join(datafolder, args.dataset, 'ROI', args.specie, f"{args.mask_type}.nii.gz")

    print(f"Source     : {source}")
    print(f"             shape {src_img.shape[:3]}  origin {np.round(src_img.affine[:3, 3], 1)}")
    print(f"Reference  : {ref_desc}")
    print(f"             shape {ref_img.shape[:3]}  origin {np.round(ref_img.affine[:3, 3], 1)}")
    print(f"Destination: {dest}")

    same_shape = tuple(src_img.shape[:3]) == tuple(ref_img.shape[:3])
    offset = (rsa_utils.grid_offset_mm(src_img.affine, ref_img.affine, ref_img.shape[:3])
              if same_shape else float('inf'))

    if same_shape and offset <= rsa_utils.SPACE_TOLERANCE_MM:
        print(f"\nSource is already on the reference grid ({offset:.2f} mm) -- copying unchanged.")
        out_img = src_img
    else:
        where = 'shape differs' if not same_shape else f"{offset:.1f} mm away"
        print(f"\nSource is not on the reference grid ({where}) -- resampling "
              f"(nearest neighbour, then threshold > {args.threshold}).")
        # order=0 keeps a binary mask binary; anything else invents partial voxels
        resampled = resample_from_to(src_img, (ref_img.shape[:3], ref_img.affine), order=0)
        data = (np.asarray(resampled.dataobj) > args.threshold).astype(np.uint8)
        out_img = nib.Nifti1Image(data, ref_img.affine)
        out_img.header.set_zooms(ref_img.header.get_zooms()[:3])

    n_vox = int((np.asarray(out_img.dataobj) > 0).sum())
    print(f"Mask voxels: {n_vox}")
    if n_vox == 0:
        print("ERROR: resulting mask is empty -- refusing to write it.")
        return 1

    # The mask is about to become the reference every beta map is checked against,
    # so make sure it really landed on the grid we were aiming for.
    rsa_utils.check_same_space(('reference grid', ref_img), [('new mask', out_img)],
                              context='mask construction', strict=True)

    # ROI/D holds 'b_GreyMatter2mmB.nii.gz' while ROI/H holds 'b_greyMatter2mmB.nii.gz',
    # and searchlight's --mask_type defaults to the capital-G spelling. Windows
    # resolves either; Linux does not, so a case-variant sitting next to the file
    # we are about to write is a real trap and worth naming -- including on a dry run.
    dest_dir, want = os.path.dirname(dest), os.path.basename(dest)
    if os.path.isdir(dest_dir):
        for name in os.listdir(dest_dir):
            if name.lower() == want.lower() and name != want:
                print(f"\nNOTE: {dest_dir} contains {name}, which differs from\n"
                      f"      {want} only by case. searchlight.py --mask_type "
                      f"{args.mask_type} expects {want} exactly.\n"
                      "      On a case-insensitive filesystem (Windows) this run will "
                      "correct the on-disk name.\n"
                      "      On a case-sensitive one (Linux) it will not touch this "
                      "file -- delete it manually once step 1 has been re-run.")

    if args.dry_run:
        print("\n--dry_run: nothing written.")
        return 0

    os.makedirs(dest_dir, exist_ok=True)
    if os.path.exists(dest):
        backup = dest.replace('.nii.gz', '_pre-alignment.nii.gz')
        if not os.path.exists(backup):
            shutil.copyfile(dest, backup)
            print(f"\nExisting mask backed up to {backup}")
        else:
            print(f"\nBackup already exists, leaving it: {backup}")

    nib.save(out_img, dest)

    # On a case-insensitive filesystem, writing 'b_GreyMatter2mmB.nii.gz' over an
    # existing 'b_greyMatter2mmB.nii.gz' updates that file's *content* but leaves
    # its *name* however it was on disk -- Windows folds the two into one entry,
    # so nib.save silently keeps the old casing. Detect that (the literal name we
    # asked for is absent from the directory listing) and force it via a
    # throwaway rename, so the shared disk is also correct for the Linux mount.
    # On a genuinely case-sensitive filesystem `want` is already present after
    # nib.save, so this never touches an unrelated stray file left over there.
    entries = os.listdir(dest_dir)
    if want not in entries:
        actual = next(n for n in entries if n.lower() == want.lower())
        tmp = os.path.join(dest_dir, f"_casefix_{want}")
        os.replace(os.path.join(dest_dir, actual), tmp)
        os.replace(tmp, dest)
        print(f"Wrote {dest}  (corrected on-disk name from {actual})")
    else:
        print(f"Wrote {dest}")

    print("\nStep-1 outputs computed against the old mask are not valid -- rerun step 1.")
    return 0


if __name__ == '__main__':
    sys.exit(main())

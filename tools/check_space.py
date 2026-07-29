#!/usr/bin/env python
"""check_space.py -- verify that a dataset's images all share one voxel grid.

The RSA pipeline combines images by array index: the mask selects voxels with
``data[mask_bool]``, crossnobis folds subtract beta maps run by run, and the
group step averages subject maps element-wise. All of that is only meaningful
when every image sits on the *same voxel grid* -- same shape AND same affine.

Matching shapes are not enough. FSL first-level output is the classic trap:
``fmri(regstandard_yn) 1`` only *estimates* the transform into template space
and writes it to ``reg/``; ``stats/pe*.nii.gz`` stay in scanner-native space,
one grid per run. Forty subjects can all be 96x96x52 while sitting tens of
millimetres apart.

This script reports, per participant and run, how far each beta map's grid is
from the mask's, so a dataset can be checked before burning compute on it.

Usage (from the repo root, with the full Anaconda interpreter -- see CLAUDE.md):

    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\check_space.py --dataset EmoC --specie H

    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\check_space.py --dataset EmoC --specie D \\
        --model basic-block --verbose

Exit code is 0 when every grid agrees, 1 when any does not -- so it can gate a
scheduler run.
"""

import argparse
import glob
import json
import os
import sys

import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import rsa_utils                       # noqa: E402
from scheduler.paths import get_paths  # noqa: E402


def find_beta_maps(datafolder, dataset, model, specie, task):
    """Return {(sub, ses, run): (path, layout)} for one beta map per run.

    Prefers the aligned step-0.5 map, falling back to the raw FEAT ``pe1`` for
    runs that have not been converted -- the same precedence
    ``rsa_utils.resolve_beta_map`` applies, so this reports on the files step 1
    would actually open.
    """
    base = os.path.join(datafolder, dataset, 'results', 'GLM', model)
    found = {}

    # legacy first, so aligned maps overwrite them for runs that have both
    for path in sorted(glob.glob(os.path.join(
            base, f"{specie}-sub-*", f"ses-*_task-{task}_run-*.feat", 'stats', 'pe1.nii.gz'))):
        parts = path.split(os.sep)
        sub, stem = parts[-4], parts[-3][:-len('.feat')]
        key = (sub, stem.split('_')[0].replace('ses-', ''), stem.split('run-')[-1])
        found[key] = (path, 'feat')

    # A run counts as converted only when its manifest is present -- the same rule
    # rsa_utils.run_is_aligned applies. Globbing beta_*.nii.gz instead would pick
    # up leftovers from an interrupted or abandoned run.
    for manifest in sorted(glob.glob(os.path.join(
            base, f"{specie}-sub-*", f"ses-*_task-{task}_run-*", 'beta_manifest.json'))):
        run_dir = os.path.dirname(manifest)
        try:
            with open(manifest) as f:
                stims = json.load(f)['stims']
            path = os.path.join(run_dir, stims[0]['file'])
        except (OSError, ValueError, KeyError, IndexError):
            continue                      # unreadable manifest: treat as not converted
        if not os.path.exists(path):
            continue
        parts = run_dir.split(os.sep)
        sub, stem = parts[-2], parts[-1]
        key = (sub, stem.split('_')[0].replace('ses-', ''), stem.split('run-')[-1])
        found[key] = (path, 'aligned')

    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', required=True, help='Dataset name, e.g. EmoC')
    ap.add_argument('--specie', required=True, choices=['D', 'H'], help="'D' or 'H'")
    ap.add_argument('--model', default='basic-block', help='GLM model folder (default: basic-block)')
    ap.add_argument('--task', default=None, help='Task name (default: same as dataset)')
    ap.add_argument('--mask_type', default='b_GreyMatter2mmB', help='Mask selector')
    ap.add_argument('--mask', default=None, help='Explicit mask path, overrides --mask_type')
    ap.add_argument('--tolerance_mm', type=float, default=rsa_utils.SPACE_TOLERANCE_MM,
                    help='Displacement below which two grids count as identical')
    ap.add_argument('--verbose', action='store_true', help='List every run, not just the offenders')
    args = ap.parse_args()

    datafolder, _, _ = get_paths()
    task = args.task or args.dataset

    mask_path = args.mask or os.path.join(
        datafolder, args.dataset, 'ROI', args.specie, f"{args.mask_type}.nii.gz"
    )
    if not os.path.exists(mask_path):
        print(f"ERROR: mask not found: {mask_path}")
        return 2
    mask_img = nib.load(mask_path)
    mask_shape = mask_img.shape[:3]

    print(f"Reference mask : {mask_path}")
    print(f"  shape {mask_shape}  zooms {tuple(round(z, 3) for z in mask_img.header.get_zooms()[:3])}")
    print(f"  axcodes {nib.aff2axcodes(mask_img.affine)}")
    print()

    beta_maps = find_beta_maps(datafolder, args.dataset, args.model, args.specie, task)
    if not beta_maps:
        print(f"ERROR: no beta maps found under "
              f"{os.path.join(datafolder, args.dataset, 'results', 'GLM', args.model)} "
              f"for specie {args.specie}, task {task}.")
        return 2

    n_aligned = sum(1 for _, layout in beta_maps.values() if layout == 'aligned')
    print(f"Beta maps       : {n_aligned}/{len(beta_maps)} runs use the aligned "
          f"step-0.5 layout, {len(beta_maps) - n_aligned} still read raw FEAT pe files")
    print()

    # offset vs mask, and spread of the runs belonging to one participant
    per_run = {}
    per_sub = {}
    unreadable = []
    for key, (path, _layout) in beta_maps.items():
        try:
            img = nib.load(path)
            shape = img.shape[:3]
            affine = img.affine
        except Exception as e:
            # a truncated or empty .nii.gz is itself a finding -- report it and
            # keep going rather than aborting the scan of the whole dataset
            unreadable.append((key, path, e.__class__.__name__))
            per_run[key] = (float('inf'), None, None)
            per_sub.setdefault(key[0], []).append(key)
            continue
        if shape != mask_shape:
            offset = float('inf')
        else:
            offset = rsa_utils.grid_offset_mm(affine, mask_img.affine, mask_shape)
        per_run[key] = (offset, shape, affine)
        per_sub.setdefault(key[0], []).append(key)

    if unreadable:
        print(f"{len(unreadable)} beta map(s) could not be read:")
        for (sub, ses, run), path, err in unreadable[:10]:
            print(f"  {sub} ses-{ses} run-{run}: {err}  {path}")
        if len(unreadable) > 10:
            print(f"  ... and {len(unreadable) - 10} more")
        print()

    print(f"{'participant':>12} {'runs':>5} {'max mm vs mask':>15} {'max mm across runs':>19}")
    print('-' * 56)
    bad_subs = 0
    for sub in sorted(per_sub):
        keys = sorted(per_sub[sub])
        offsets = [per_run[k][0] for k in keys]
        first_affine = per_run[keys[0]][2]
        within = 0.0
        for k in keys[1:]:
            if per_run[k][1] == mask_shape and per_run[keys[0]][1] == mask_shape:
                within = max(within, rsa_utils.grid_offset_mm(
                    per_run[k][2], first_affine, mask_shape))
        worst = max(offsets)
        flag = '' if worst <= args.tolerance_mm and within <= args.tolerance_mm else '  <-- MISMATCH'
        if flag:
            bad_subs += 1
        if flag or args.verbose:
            worst_txt = 'shape differs' if worst == float('inf') else f"{worst:.1f}"
            print(f"{sub:>12} {len(keys):>5} {worst_txt:>15} {within:>19.1f}{flag}")

    print('-' * 56)
    total = len(per_sub)
    if bad_subs == 0:
        print(f"OK: all {total} participants ({len(beta_maps)} runs) are on the mask's grid "
              f"(tolerance {args.tolerance_mm} mm).")
        return 0

    print(f"MISMATCH: {bad_subs}/{total} participants are not on the mask's grid.")
    print()
    print("The pipeline would combine these by array index and produce results that")
    print("look fine but are not anatomically valid. steps 1-3 will now refuse to run.")
    print()
    if n_aligned < len(beta_maps):
        print("Run step 0.5 to write the beta maps on the template grid:")
        print(f"  python searchlight.py --dataset {args.dataset} --model {args.model} \\")
        print(f"      --specie {args.specie} --steps_to_run 0.5")
        print("(needs FSL, so run it on the Linux machine for humans), then re-check.")
    else:
        print("These runs already use the aligned layout, so the mask is the odd one out:")
        print(f"  {mask_path}")
        print("Rebuild it on the same grid the beta maps now sit on -- see")
        print("  tools/make_mask.py --dataset {} --specie {}".format(args.dataset, args.specie))
    return 1


if __name__ == '__main__':
    sys.exit(main())

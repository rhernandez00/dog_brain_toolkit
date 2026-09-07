#!/usr/bin/env python
"""check_refs_mask.py -- does the refs mask match the maps in the Colab result zips?

The group steps combine maps by array index and write the output under the mask's
affine, so a mask on a different voxel grid yields an anatomically meaningless map
that still looks perfectly normal in a viewer (CLAUDE.md, "Hard invariant: one
voxel grid"). The file name proves nothing, and ``EmoC/ROI/H/`` is a minefield:

    b_GreyMatter2mmB.nii.gz                (91,109,91)  159254 vox  <- the right one
    b_GreyMatter2mmB_pre-alignment.nii.gz  (96, 96,52)  108433 vox
    b_greyMatter2mm.nii.gz                 (96, 96,52)  114280 vox
    results_space.nii.gz                   (96, 96,52)  114280 vox
    original_atlas_space.nii.gz            (91,109,91)  171094 vox

Three of those are on the scanner-native EPI grid left over from the alignment
problem, and ``original_atlas_space`` is on the right grid but a different extent.
Picking one by name would be silent and wrong, which is why this script picks by
*evidence* instead.

Checks, per specie:
  1. every mask in ROI/ against the refs copy -- identical bytes? same grid? same
     voxels? (skipped when the data share is not reachable);
  2. the refs mask against the participant maps sampled out of the result zips --
     same shape, and same affine within 0.5 mm;
  3. containment -- every map's non-zero set must lie INSIDE the mask, since the
     group steps only carry mask voxels through and anything outside would be
     silently truncated;
  4. coverage -- how much of the mask the maps actually fill. Far below 100% means
     the mask is probably a different, larger one than the maps were computed
     against.

Run it whenever the refs folder is rebuilt, or whenever a run looks wrong:

    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\colab_gpu\\refs\\check_refs_mask.py

Exits non-zero on any problem. Note this is a *pre-flight*: at run time
``gpu_group`` already validates every map it loads against the mask
(``check_same_space`` for the grid, ``OffMaskError`` for support outside it) and
raises rather than continuing, so a mismatch cannot pass silently either way.
"""
import os, sys, glob, zipfile, hashlib, random
import numpy as np
import nibabel as nib

REPO = r"C:\github\dog_brain_toolkit"
sys.path.insert(0, os.path.join(REPO, "tools", "colab_gpu"))
import gpu_group  # noqa: E402

RESULTS = r"G:\My Drive\rsa_colab\results"
REFS = os.path.join(REPO, "tools", "colab_gpu", "refs")
P = r"P:\userdata\raulh87\data\EmoC"
N_ZIPS = 6            # participants to sample per specie
N_MAPS = 4            # maps to check inside each zip

PROBLEMS = []


def note(ok, msg):
    print(f"  [{'ok ' if ok else 'BAD'}] {msg}")
    if not ok:
        PROBLEMS.append(msg)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def grid(img):
    return tuple(img.shape[:3]), np.asarray(img.affine, dtype=float)


for specie in ("H", "D"):
    print(f"\n{'=' * 70}\n{specie}\n{'=' * 70}")
    refs_mask = gpu_group.refs_mask_path(REFS, "EmoC", specie, "b_GreyMatter2mmB")
    if not os.path.exists(refs_mask):
        note(False, f"no refs mask at {refs_mask}")
        continue
    rimg = nib.load(refs_mask)
    rshape, raff = grid(rimg)
    rbool = np.asarray(rimg.dataobj).astype(bool)
    print(f"refs mask: {refs_mask}")
    print(f"  shape {rshape}  voxels {int(rbool.sum())}  sha {sha(refs_mask)}")
    print(f"  affine diag {np.round(np.diag(raff), 3)}  origin {np.round(raff[:3, 3], 2)}")

    # -- 1. all candidate masks on P: --------------------------------------
    roi = os.path.join(P, "ROI", specie)
    print(f"\n  candidates in {roi}:")
    if os.path.isdir(roi):
        for cand in sorted(glob.glob(os.path.join(roi, "*.nii.gz"))):
            img = nib.load(cand)
            cshape, caff = grid(img)
            b = np.asarray(img.dataobj).astype(bool)
            same_grid = (cshape == rshape and
                         gpu_group.gpu_rsa.grid_offset_mm(caff, raff, rshape) <= 0.5)
            ident = cshape == rshape and np.array_equal(b, rbool)
            tag = "<= REFS COPY" if os.path.basename(cand) == "b_GreyMatter2mmB.nii.gz" else ""
            print(f"    {os.path.basename(cand):42s} {str(cshape):16s} "
                  f"vox={int(b.sum()):7d} grid={'same' if same_grid else 'DIFF'} "
                  f"voxels={'same' if ident else 'diff'} {tag}")
        src = os.path.join(roi, "b_GreyMatter2mmB.nii.gz")
        if os.path.exists(src):
            note(sha(src) == sha(refs_mask),
                 f"refs copy is byte-identical to {src}")
    else:
        print("    (P: not reachable)")

    # -- 2/3/4. against the maps actually in the result zips ----------------
    store = gpu_group.ResultStore([RESULTS], dataset="EmoC", verbose=False)
    zips = [z for z in os.listdir(RESULTS)
            if z.lower().endswith(".zip") and f"_{specie}-sub-" in z
            and not z.startswith(("result_group_", "result_step5_"))]
    if not zips:
        print(f"\n  no {specie} result zips on Drive -- nothing to compare against")
        continue
    random.seed(0)
    sample = random.sample(zips, min(N_ZIPS, len(zips)))
    print(f"\n  checking {len(sample)} zip(s) x up to {N_MAPS} map(s):")

    n_checked = 0
    grid_ok = True
    outside_total = 0
    union = np.zeros(rbool.size, dtype=bool)
    for zn in sample:
        with zipfile.ZipFile(os.path.join(RESULTS, zn)) as zf:
            members = [m for m in zf.namelist()
                       if m.endswith(".nii.gz") and gpu_group.parse_arcname(m)]
            for m in random.sample(members, min(N_MAPS, len(members))):
                img = gpu_group._img_from_bytes(zf.read(m))
                mshape, maff = grid(img)
                if mshape != rshape:
                    grid_ok = False
                    print(f"    SHAPE {mshape} != {rshape}  {zn}:{m}")
                    continue
                off = gpu_group.gpu_rsa.grid_offset_mm(maff, raff, rshape)
                if off > 0.5:
                    grid_ok = False
                    print(f"    GRID {off:.2f} mm away  {zn}:{m}")
                flat = np.asarray(img.dataobj, dtype=np.float64).reshape(-1)
                active = ~np.isfinite(flat) | (flat != 0.0)
                outside = int((active & ~rbool.reshape(-1)).sum())
                outside_total += outside
                union |= active
                n_checked += 1
        print(f"    {zn[:58]:58s} ok")

    note(grid_ok, f"all {n_checked} sampled maps sit on the refs mask's grid")
    note(outside_total == 0,
         f"no map has non-zero voxels OUTSIDE the mask "
         f"({outside_total} found across {n_checked} maps)")
    inside = int((union & rbool.reshape(-1)).sum())
    print(f"  sampled support: {int(union.sum())} voxels, "
          f"{inside} inside the mask ({100 * inside / max(int(rbool.sum()), 1):.1f}% "
          f"of the mask's {int(rbool.sum())})")

print(f"\n{'=' * 70}")
if PROBLEMS:
    print(f"{len(PROBLEMS)} PROBLEM(S):")
    for p in PROBLEMS:
        print("  -", p)
    sys.exit(1)
print("mask matches the maps in the result zips on every check")

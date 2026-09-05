#!/usr/bin/env python
"""gpu_rsa.py -- GPU (PyTorch) reimplementation of RSA pipeline steps 1, 2, 4.

This module is copied into every Colab package by ``tools/create_package.py`` and
is what actually runs on the Colab GPU. It reproduces, faithfully, the Mahalanobis
``stim-wise`` path of the CPU pipeline in ``rsa_utils.py``:

  * Step 1 -- ``calculate_mahalanobis_pairwise_maps`` + ``crossnobis``
              (rsa_utils.py: calculate_mahalanobis_pairwise_maps / crossnobis)
  * Step 2 -- ``_compare_mahalanobis_with_model`` + ``_calculate_model_similarity_map``
              (real model comparison)
  * Step 4 -- the same, over ``reps`` label-permuted models (``shuffle_vector`` scheme)

Design (see the package README / plan):

  * Step 1 is model-independent: it produces the 45 pairwise crossnobis distance
    maps for the 10 stim-wise categories. Run once per participant.
  * Steps 2 & 4 load those 45 maps once into an ``(n_voxels, 45)`` matrix and reduce
    to matmuls: Pearson is a plain matmul; Kendall tau-a factors into a signed
    upper-triangle matmul ``data_sign(n_voxels, 990) @ model_sign(990, n_perms)``.

Everything is computed in float64 to match the CPU (numpy) pipeline to ~1e-12. The
only supported ``dis_method`` is ``mahalanobis`` and the only ``mah_fold`` is
``stim-wise`` (the pipeline's default and the whole point of the GPU port).

This file is intentionally dependency-light (torch, numpy, nibabel only) so it runs
on a stock Colab runtime without importing the rest of the toolkit.
"""

import glob
import itertools
import json
import os
import time
import zlib
import zipfile

import numpy as np
import nibabel as nib
import torch

EPS = float(np.finfo(np.float64).eps)  # 2.22e-16, matches np.finfo(float).eps on CPU
DTYPE = torch.float64


# ===========================================================================
# Voxel-grid validation
#
# Mirror of ``rsa_utils.check_same_space`` -- duplicated rather than imported
# because Colab packages ship only this file (see tools/create_package.py).
# Keep the two in sync.
#
# Every image combined here is combined by array index: the mask selects voxels
# by position, crossnobis folds subtract run means element-wise, and the output
# volume is written with a single affine. All of it is only meaningful if the
# mask and every beta map sit on the same voxel grid -- same shape AND same
# affine. Matching shapes prove nothing: FSL first-level ``stats/pe*.nii.gz``
# stay in scanner-native space, a different grid per run, all the same shape.
# ===========================================================================
SPACE_TOLERANCE_MM = 0.5


class SpaceMismatchError(ValueError):
    """Raised when images that must share a voxel grid do not."""


def grid_offset_mm(affine_a, affine_b, shape):
    """Worst-case displacement in mm between two voxel grids over a volume."""
    shape = tuple(int(s) for s in shape[:3])
    corners = np.array(list(itertools.product(*[(0, s - 1) for s in shape])), dtype=float)
    centre = (np.array(shape, dtype=float) - 1.0) / 2.0
    points = np.vstack([corners, centre])
    hom = np.c_[points, np.ones(len(points))]
    pos_a = (np.asarray(affine_a, dtype=float) @ hom.T).T[:, :3]
    pos_b = (np.asarray(affine_b, dtype=float) @ hom.T).T[:, :3]
    return float(np.linalg.norm(pos_a - pos_b, axis=1).max())


def check_same_space(reference, images, context="", tolerance_mm=SPACE_TOLERANCE_MM,
                     strict=True):
    """Verify every (label, image) in `images` is on the same grid as `reference`.

    `reference` is a (label, nibabel image) pair. Raises SpaceMismatchError with
    a diagnostic listing on any mismatch; returns True otherwise. With
    ``strict=False`` (manifest key ``allow_space_mismatch``, set by
    ``create_package.py --allow_space_mismatch``) it warns and returns False
    instead -- the images are then combined by array index regardless of their
    affines.
    """
    ref_label, ref_img = reference
    ref_shape, ref_affine = tuple(ref_img.shape[:3]), ref_img.affine

    problems = []
    for label, img in images:
        shape = tuple(img.shape[:3])
        if shape != ref_shape:
            problems.append(f"  {label}: shape {shape} != {ref_shape}")
            continue
        offset = grid_offset_mm(img.affine, ref_affine, ref_shape)
        if offset > tolerance_mm:
            problems.append(f"  {label}: same shape but grid is {offset:.1f} mm away")

    if not problems:
        return True

    shown = problems[:12]
    if len(problems) > len(shown):
        shown.append(f"  ... and {len(problems) - len(shown)} more")
    message = (
        f"Voxel-grid mismatch{' in ' + context if context else ''}.\n"
        f"Reference grid: {ref_label}\n" + "\n".join(shown)
        + "\n\nThese images are combined by array index (masking, crossnobis "
          "folds), so they must share one grid. Images that merely share a "
          "shape are NOT in the same space.\n"
          "Most common cause: FSL first-level stats were never resampled into "
          "template space -- FEAT with regstandard_yn=1 only estimates the "
          "transform. Fix the dataset before packaging: run "
          "tools/check_space.py on the pipeline machine.\n"
          "To package and run anyway: create_package.py --allow_space_mismatch."
    )
    if strict:
        raise SpaceMismatchError(message)
    print(f"WARNING: {message}")
    return False


def load_reference_mask(data_root, manifest):
    """Load the searchlight mask -- it defines THE reference voxel grid.

    Returns ``(mask_img, mask_bool)``. Every beta map and every step-1 map is
    validated against ``mask_img``, and every output volume is written with
    ``mask_img.affine``. Taking the affine from whichever input happened to be
    loaded first is what made GPU and CPU outputs disagree by several mm.
    """
    dataset, specie, mask_type = (
        manifest["dataset"], manifest["specie"], manifest["mask_type"])
    mask_path = os.path.join(data_root, dataset, "ROI", specie, f"{mask_type}.nii.gz")
    mask_img = nib.load(mask_path)
    return mask_img, np.asarray(mask_img.dataobj).astype(bool)


# ===========================================================================
# small helpers
# ===========================================================================
def pick_device(prefer_gpu=True):
    """Return a torch device: cuda if available and requested, else cpu."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sphere_offsets(radius, ndim=3):
    """Integer offsets of an isotropic voxel sphere -- identical to the CPU
    searchlight (rsa_utils.calculate_mahalanobis_pairwise_maps)."""
    r = float(radius)
    rad = int(np.floor(r))
    ranges = [np.arange(-rad, rad + 1) for _ in range(ndim)]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, ndim)
    keep = (grid.astype(float) ** 2).sum(axis=1) <= r * r + 1e-12
    return grid[keep]


def stimwise_category(stim, dataset):
    """Collapse a stimulus id to its stim-wise category, exactly like the CPU code."""
    if dataset == "EmoB":
        if "-" not in stim:
            raise ValueError(f"Stimulus {stim!r} has no '-' to derive a category.")
        return stim.split("-")[0]
    if dataset == "EmoC":
        return stim[:-1]
    raise ValueError(f"Dataset {dataset!r} not supported for stim-wise categories.")


def stimwise_categories(stim_types, dataset):
    """Sorted unique stim-wise categories (sorted == numpy's np.unique order,
    which ``crossnobis`` uses internally)."""
    cats = {stimwise_category(s, dataset) for s in stim_types}
    return sorted(cats)


def canonical_pairs(categories):
    """The canonical, order-stable list of category pairs (i < j)."""
    return list(itertools.combinations(list(categories), 2))


# ===========================================================================
# batched Ledoit-Wolf + crossnobis (the compute kernels)
# ===========================================================================
def batched_ledoit_wolf(R):
    """Batched Ledoit-Wolf shrunk covariance, matching sklearn.covariance.LedoitWolf.

    Parameters
    ----------
    R : (B, n_samples, n_features) tensor
        One residual matrix per voxel. Centered internally by its column mean
        (sklearn ``assume_centered=False``).

    Returns
    -------
    cov : (B, n_features, n_features) tensor
        ``(1 - shrinkage) * emp_cov + shrinkage * mu * I`` per voxel.

    Notes
    -----
    Mirrors ``sklearn.covariance._shrunk_covariance.ledoit_wolf_shrinkage`` /
    ``ledoit_wolf`` term for term (validated in tools/colab_gpu/validate_gpu.py):

        emp_cov = Xc.T @ Xc / n ;  mu = trace(emp_cov)/p
        beta_   = sum_k (sum_i Xc2[k,i])^2                     (== sum(X2.T @ X2))
        delta_  = ||emp_cov||_F^2
        beta    = (beta_/n - delta_) / (p*n)
        delta   = (delta_ - p*mu^2) / p
        beta    = min(beta, delta)
        shrink  = 0 if beta == 0 else beta/delta
    """
    B, n, p = R.shape
    Xc = R - R.mean(dim=-2, keepdim=True)               # center columns (samples axis)
    emp_cov = Xc.transpose(-1, -2) @ Xc / n             # (B, p, p)
    diag = torch.diagonal(emp_cov, dim1=-2, dim2=-1)    # (B, p)
    mu = diag.sum(-1) / p                               # (B,)

    Xc2 = Xc * Xc
    s = Xc2.sum(-1)                                     # (B, n) per-sample row sums
    beta_raw = (s * s).sum(-1)                          # (B,) == sum(X2.T @ X2)
    delta_ = (emp_cov * emp_cov).sum(dim=(-1, -2))      # (B,) == ||emp_cov||_F^2

    beta = (beta_raw / n - delta_) / (p * n)            # (B,)
    delta = (delta_ - p * mu * mu) / p                  # (B,)
    beta = torch.minimum(beta, delta)
    shrinkage = torch.where(
        (beta == 0) | (delta <= 0),
        torch.zeros_like(beta),
        beta / delta,
    )

    eye = torch.eye(p, dtype=R.dtype, device=R.device)
    cov = (1.0 - shrinkage)[:, None, None] * emp_cov
    cov = cov + (shrinkage * mu)[:, None, None] * eye
    return cov


def batched_crossnobis(U):
    """Batched cross-validated Mahalanobis (crossnobis) distances.

    Parameters
    ----------
    U : (B, M, C, P) tensor
        Per-voxel, per-run (M partitions), per-condition (C) mean patterns over
        P sphere features. Reproduces ``rsa_utils.crossnobis`` with
        ``shrinkage='ledoitwolf'`` and ``sigma=None``.

    Returns
    -------
    D : (B, C, C) tensor
        Symmetric crossnobis distance matrices (diagonal zero).
    """
    B, M, C, P = U.shape
    if M < 2:
        raise ValueError("crossnobis needs at least 2 partitions (runs).")

    # residuals: run-to-run deviation of condition means -> (B, M*C, P)
    R = (U - U.mean(dim=1, keepdim=True)).reshape(B, M * C, P)
    sigma = batched_ledoit_wolf(R)                     # (B, P, P)

    # Sigma^{-1/2} via eigendecomposition, eigenvalues clipped like the CPU code
    evals, evecs = torch.linalg.eigh(sigma)            # (B,P), (B,P,P)
    evals = torch.clamp(evals, min=EPS)
    inv_sqrt = evecs / torch.sqrt(evals)[:, None, :]   # divide each eigenvector column
    Winvhalf = inv_sqrt @ evecs.transpose(-1, -2)      # (B, P, P), symmetric Sigma^{-1/2}

    # whiten condition means: Z[b,m,c,:] = U[b,m,c,:] @ Winvhalf
    Z = torch.einsum("bmcp,bpk->bmck", U, Winvhalf)    # (B, M, C, P)

    # crossnobis per condition pair, vectorized over the C*(C-1)/2 pairs
    iu, ju = np.triu_indices(C, 1)
    iu = torch.as_tensor(iu, device=U.device)
    ju = torch.as_tensor(ju, device=U.device)
    delta = Z[:, :, iu, :] - Z[:, :, ju, :]            # (B, M, n_pairs, P)
    s = delta.sum(dim=1)                               # (B, n_pairs, P) sum over runs
    ssq = (s * s).sum(-1)                              # (B, n_pairs)
    sumsq = (delta * delta).sum(dim=(1, 3))            # (B, n_pairs)
    dpair = (ssq - sumsq) / (M * (M - 1))              # (B, n_pairs)

    D = torch.zeros(B, C, C, dtype=U.dtype, device=U.device)
    D[:, iu, ju] = dpair
    D[:, ju, iu] = dpair
    return D


# ===========================================================================
# Step 1 -- crossnobis searchlight over the mask
# ===========================================================================
def _beta_path(data_root, dataset, model, specie, sub_N, session, run_N, task, stim):
    """Path to one aligned beta map, matching ``rsa_utils.beta_map_path``.

    Packages carry the step-0.5 layout (``beta_{stim}.nii.gz`` per run folder),
    never the raw FEAT ``stats/pe*`` files -- those are in scanner-native space
    for humans, one grid per run.
    """
    return os.path.join(
        data_root, dataset, "results", "GLM", model,
        f"{specie}-sub-{sub_N:02d}",
        f"ses-{int(session):02d}_task-{task}_run-{int(run_N):02d}",
        f"beta_{stim}.nii.gz",
    )


def _load_category_means(data_root, manifest, mask_bool, ref_img=None):
    """Build the (M, C, V_flat) tensor of per-run, per-category mean beta patterns.

    Averaging over a category's exemplars commutes with the sphere gather, so we
    precompute the C category-mean volumes per partition instead of loading every
    beta per voxel. Returns (means (M, C, V), affine, categories, partitions).

    ``ref_img`` is the mask image; it defines the reference voxel grid. Every beta
    map is validated against it and the returned affine is ``ref_img.affine``, so
    the output header does not depend on file ordering. When it is None the first
    beta map becomes the reference, which still catches the case of one
    participant's runs sitting on different native grids.

    Partitions are the unique ``run_N`` values, NOT the (session, run_N) entries:
    the CPU pipeline appends ``run_N`` as the crossnobis partition
    (calculate_mahalanobis_pairwise_maps), so two sessions that share a run number
    collapse into one partition whose mean pools all their exemplars. Reproducing
    this exactly is what makes the GPU maps match the CPU maps (validated to ~1e-12).
    """
    dataset = manifest["dataset"]
    model = manifest["model"]
    specie = manifest["specie"]
    sub_N = manifest["sub_N"]
    task = manifest["task"]
    stim_types = manifest["stim_types"]
    categories = manifest["categories"]
    entries = manifest["runs"]                    # list of {'session','run_N'}

    cat_index = {c: i for i, c in enumerate(categories)}
    partitions = sorted({int(e["run_N"]) for e in entries})   # unique run_N (== np.unique)
    part_index = {rn: i for i, rn in enumerate(partitions)}

    M, C = len(partitions), len(categories)
    V = int(mask_bool.size)
    sums = np.zeros((M, C, V), dtype=np.float64)
    counts = np.zeros((M, C), dtype=np.float64)
    loaded = []          # (label, img) for the voxel-grid check below
    for entry in entries:
        pi = part_index[int(entry["run_N"])]
        for stim in stim_types:
            path = _beta_path(data_root, dataset, model, specie, sub_N,
                              entry["session"], entry["run_N"], task, stim)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Beta map file not found: {path}")
            img = nib.load(path)
            loaded.append((f"ses-{int(entry['session']):02d} "
                           f"run-{int(entry['run_N']):02d} {stim}", img))
            ci = cat_index[stimwise_category(stim, dataset)]
            sums[pi, ci] += np.asarray(img.dataobj, dtype=np.float64).reshape(-1)
            counts[pi, ci] += 1.0

    # The crossnobis folds below subtract these run means element-wise and the
    # mask selects voxels by position, so every beta must be on one grid.
    if ref_img is not None:
        reference = ("mask", ref_img)
    else:
        reference = (f"first beta ({loaded[0][0]})", loaded[0][1])
    check_same_space(reference, loaded,
                     context=f"GPU step 1 for {specie}-sub-{sub_N:02d}",
                     strict=not manifest.get("allow_space_mismatch", False))
    affine = reference[1].affine

    means = sums / counts[:, :, None]
    return means, affine, categories, [{"run_N": rn} for rn in partitions]


def crossnobis_searchlight(means, mask_bool, categories, runs, radius,
                           device=None, batch=1024, verbose=False):
    """Core step-1 searchlight: crossnobis distance per category pair, per voxel.

    Pure compute (no file I/O) so it can be validated directly against the CPU
    maps. ``means`` is the ``(M, C, V)`` per-run/per-category mean-pattern array
    from ``_load_category_means``. Returns ``{(cat1, cat2): 3d np.float64 volume}``
    with NaN outside the searchlight, in canonical pair order.
    """
    device = device or pick_device()
    shape = mask_bool.shape
    M, C = len(runs), len(categories)
    means_t = torch.as_tensor(means, dtype=DTYPE, device=device).reshape(M * C, means.shape[-1])

    offsets = sphere_offsets(radius, ndim=mask_bool.ndim)
    centers = np.argwhere(mask_bool)                                   # (n_vox, 3)
    dims = np.array(shape)
    strides = np.array([shape[1] * shape[2], shape[2], 1])            # C-order flat idx

    # per-voxel valid neighbour flat-indices (ragged) -> group by neighbour count P
    groups = {}   # P -> list of (center_flat, neighbour_flat array)
    for center in centers:
        neigh = center + offsets
        neigh = neigh[np.all((neigh >= 0) & (neigh < dims), axis=1)]
        neigh = neigh[mask_bool[neigh[:, 0], neigh[:, 1], neigh[:, 2]]]
        P = neigh.shape[0]
        if P == 0:
            continue
        groups.setdefault(P, []).append((int(center @ strides), neigh @ strides))

    pairs = canonical_pairs(categories)              # 45 pairs (sorted-cat order)
    dist_flat = {pair: np.full(int(np.prod(shape)), np.nan, dtype=np.float64)
                 for pair in pairs}
    iu, ju = np.triu_indices(C, 1)
    total = sum(len(v) for v in groups.values())
    done = 0
    for P, items in sorted(groups.items()):
        cflats = np.array([it[0] for it in items])
        nflats_t = torch.as_tensor(np.stack([it[1] for it in items]), device=device)
        for start in range(0, len(items), batch):
            idx = nflats_t[start:start + batch]                       # (b, P)
            b = idx.shape[0]
            U = means_t[:, idx].permute(1, 0, 2).reshape(b, M, C, P)  # (b, M, C, P)
            dpair = batched_crossnobis(U)[:, iu, ju].cpu().numpy()    # (b, n_pairs)
            cf = cflats[start:start + b]
            for k, pair in enumerate(pairs):
                dist_flat[pair][cf] = dpair[:, k]
            done += b
        if verbose:
            print(f"[step1]   P={P:3d}: {len(items):6d} voxels  ({done}/{total})")
    return {pair: dist_flat[pair].reshape(shape) for pair in pairs}


def run_step1(pkg_root, manifest, device=None, batch=1024, verbose=True):
    """Compute pipeline step 1 for the participant, dispatching on ``dis_method``."""
    if manifest.get("dis_method", "mahalanobis") == "correlation":
        return run_step1_correlation(pkg_root, manifest, device=device,
                                     batch=batch, verbose=verbose)
    return run_step1_mahalanobis(pkg_root, manifest, device=device,
                                 batch=batch, verbose=verbose)


def run_step1_mahalanobis(pkg_root, manifest, device=None, batch=1024, verbose=True):
    """Compute the 45 pairwise crossnobis maps for the participant (mahalanobis).

    Writes ``r-{radius}_mahalanobis_{cat1}_{cat2}.nii.gz`` for every category pair
    under ``data/{dataset}/results/RSA/{model}/{specie}-sub-NN/`` inside the package,
    exactly where the CPU pipeline writes them. Returns the list of written paths.
    """
    device = device or pick_device()
    data_root = os.path.join(pkg_root, "data")
    dataset, model, specie, sub_N, radius = (
        manifest["dataset"], manifest["model"], manifest["specie"],
        manifest["sub_N"], manifest["radius"])

    mask_img, mask_bool = load_reference_mask(data_root, manifest)

    t0 = time.time()
    # affine comes back as the mask's -- the reference grid, not "whichever beta
    # map was loaded first" (that made GPU and CPU headers differ by ~5 mm)
    means, affine, categories, runs = _load_category_means(
        data_root, manifest, mask_bool, ref_img=mask_img)
    if verbose:
        print(f"[step1] loaded {len(runs)} runs x {len(categories)} categories in "
              f"{time.time()-t0:.1f}s; mask voxels={int(mask_bool.sum())}, device={device}")

    dist_maps = crossnobis_searchlight(means, mask_bool, categories, runs, radius,
                                       device=device, batch=batch, verbose=verbose)

    out_dir = os.path.join(data_root, dataset, "results", "RSA", model,
                           f"{specie}-sub-{sub_N:02d}")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for (cat1, cat2), vol in dist_maps.items():
        out_path = os.path.join(out_dir, f"r-{radius}_mahalanobis_{cat1}_{cat2}.nii.gz")
        nib.save(nib.Nifti1Image(vol, affine), out_path)
        written.append(out_path)
    if verbose:
        print(f"[step1] wrote {len(written)} maps in {time.time()-t0:.1f}s -> {out_dir}")
    return written


# ===========================================================================
# Correlation path -- per-run Pearson-RDM searchlight (step 1)
# ===========================================================================
def _load_run_betas(data_root, manifest, entry, mask_bool, ref_img=None):
    """Load the participant's stim beta volumes for one run -> (n_stim, V) float64
    plus the affine. Order follows ``manifest['stim_types']``.

    ``ref_img`` is the mask image defining the reference voxel grid; every beta is
    validated against it and the returned affine is the mask's, so the output
    header does not depend on which file was read first."""
    dataset, model, specie, sub_N, task = (
        manifest["dataset"], manifest["model"], manifest["specie"],
        manifest["sub_N"], manifest["task"])
    stim_types = manifest["stim_types"]
    V = int(mask_bool.size)
    betas = np.zeros((len(stim_types), V), dtype=np.float64)
    loaded = []
    for i, stim in enumerate(stim_types):
        path = _beta_path(data_root, dataset, model, specie, sub_N,
                          entry["session"], entry["run_N"], task, stim)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Beta map file not found: {path}")
        img = nib.load(path)
        loaded.append((f"{stim} ({os.path.basename(path)})", img))
        betas[i] = np.asarray(img.dataobj, dtype=np.float64).reshape(-1)

    if ref_img is not None:
        reference = ("mask", ref_img)
    else:
        reference = (f"first beta ({loaded[0][0]})", loaded[0][1])
    check_same_space(
        reference, loaded,
        context=(f"GPU step 1 (correlation) for {specie}-sub-{sub_N:02d} "
                 f"ses-{int(entry['session']):02d} run-{int(entry['run_N']):02d}"),
        strict=not manifest.get("allow_space_mismatch", False))
    return betas, reference[1].affine


def pearson_rdm_searchlight(betas, mask_bool, radius, device=None, batch=1024,
                            verbose=False):
    """Per-voxel Pearson correlation-distance RDM over the stimulus betas of one run.

    Reproduces ``similarity_searchlight`` with dis_method='correlation'
    (``dist = 1 - pearson``) for every stimulus pair. ``betas`` is (n_stim, V).
    Returns ``dist (n_vox, n_pairs) float32`` over the mask voxels (upper triangle,
    canonical stim order) plus the mask flat-indices. All-zero (blank) stimulus
    maps yield distance 0 for their pairs, matching the CPU blank-map handling.
    """
    device = device or pick_device()
    shape = mask_bool.shape
    n_stim = betas.shape[0]
    mask_flat = np.flatnonzero(mask_bool.reshape(-1))
    betas_t = torch.as_tensor(betas, dtype=DTYPE, device=device)          # (n_stim, V)
    blank = (betas_t.abs().sum(dim=1) == 0)                               # (n_stim,)

    offsets = sphere_offsets(radius, ndim=mask_bool.ndim)
    centers = np.argwhere(mask_bool)
    dims = np.array(shape)
    strides = np.array([shape[1] * shape[2], shape[2], 1])
    # group centres by neighbour count so each batch gathers a fixed (n_stim, P)
    groups = {}
    for ci, center in enumerate(centers):
        neigh = center + offsets
        neigh = neigh[np.all((neigh >= 0) & (neigh < dims), axis=1)]
        neigh = neigh[mask_bool[neigh[:, 0], neigh[:, 1], neigh[:, 2]]]
        if neigh.shape[0] == 0:
            continue
        groups.setdefault(neigh.shape[0], []).append((ci, neigh @ strides))

    iu, ju = np.triu_indices(n_stim, 1)
    iu_t = torch.as_tensor(iu, device=device)
    ju_t = torch.as_tensor(ju, device=device)
    blank_pair = (blank[iu_t] | blank[ju_t])                             # (n_pairs,)
    n_pairs = iu.shape[0]
    out = np.full((mask_flat.shape[0], n_pairs), np.nan, dtype=np.float32)
    # np.argwhere and np.flatnonzero enumerate voxels in the same C-order, so a
    # centre's position ``ci`` is exactly its row in ``mask_flat``/``out``.
    done = 0
    total = sum(len(v) for v in groups.values())
    for P, items in sorted(groups.items()):
        cis = np.array([it[0] for it in items])
        nflats = torch.as_tensor(np.stack([it[1] for it in items]), device=device)
        for s in range(0, len(items), batch):
            idx = nflats[s:s + batch]                                    # (b, P)
            b = idx.shape[0]
            X = betas_t[:, idx].permute(1, 0, 2)                         # (b, n_stim, P)
            Xc = X - X.mean(dim=2, keepdim=True)
            Xn = Xc / torch.linalg.norm(Xc, dim=2, keepdim=True)
            corr = Xn @ Xn.transpose(1, 2)                              # (b, n_stim, n_stim)
            dist = 1.0 - corr[:, iu_t, ju_t]                            # (b, n_pairs)
            dist[:, blank_pair] = 0.0                                    # blank-map rule
            out[cis[s:s + b]] = dist.to(torch.float32).cpu().numpy()
            done += b
        if verbose:
            print(f"[step1-corr]   P={P:3d}: {len(items):6d} voxels  ({done}/{total})")
    return out, mask_flat


def run_step1_correlation(pkg_root, manifest, device=None, batch=1024, verbose=True):
    """Per-run Pearson-RDM searchlight (correlation step 1).

    Writes one ``r-{radius}_correlation_{stim1}_{stim2}.nii.gz`` per stimulus pair
    into each run folder ``.../{specie}-sub-NN/ses-XX_task-{task}_run-YY/`` (single
    canonical orientation; the CPU also writes the inverse, but downstream loading
    tries both, and steps 3-10 never re-read step-1 maps). Returns written paths.
    """
    device = device or pick_device()
    data_root = os.path.join(pkg_root, "data")
    dataset, model, specie, sub_N, radius, task = (
        manifest["dataset"], manifest["model"], manifest["specie"], manifest["sub_N"],
        manifest["radius"], manifest["task"])
    stim_types = manifest["stim_types"]
    pairs = canonical_pairs(stim_types)                                 # 780 for 40 stims

    mask_img, mask_bool = load_reference_mask(data_root, manifest)
    shape = mask_bool.shape
    nvox_total = int(np.prod(shape))

    written = []
    t0 = time.time()
    for entry in manifest["runs"]:
        session = f"{int(entry['session']):02d}"
        run_N = int(entry["run_N"])
        betas, affine = _load_run_betas(data_root, manifest, entry, mask_bool,
                                        ref_img=mask_img)
        dist, mask_flat = pearson_rdm_searchlight(betas, mask_bool, radius,
                                                  device=device, batch=batch,
                                                  verbose=verbose)
        run_dir = os.path.join(data_root, dataset, "results", "RSA", model,
                               f"{specie}-sub-{sub_N:02d}",
                               f"ses-{session}_task-{task}_run-{run_N:02d}")
        os.makedirs(run_dir, exist_ok=True)
        for k, (s1, s2) in enumerate(pairs):
            vol = np.full(nvox_total, np.nan, dtype=np.float32)
            vol[mask_flat] = dist[:, k]
            out_path = os.path.join(run_dir, f"r-{radius}_correlation_{s1}_{s2}.nii.gz")
            nib.save(nib.Nifti1Image(vol.reshape(shape), affine), out_path)
            written.append(out_path)
        if verbose:
            print(f"[step1-corr] ses-{session} run-{run_N:02d}: wrote {len(pairs)} maps "
                  f"({time.time()-t0:.1f}s)")
    return written
    return written


# ===========================================================================
# Step 2 / 4 -- model comparison (real + permutations)
# ===========================================================================
def _load_pairwise_map(data_root, manifest, cat1, cat2):
    """Load a step-1 pairwise map, trying both pair orientations (like the CPU
    ``load_pairwise_similarity_map``). Returns a flat float64 array."""
    dataset, model, specie, sub_N, radius = (
        manifest["dataset"], manifest["model"], manifest["specie"],
        manifest["sub_N"], manifest["radius"],
    )
    base = os.path.join(data_root, dataset, "results", "RSA", model,
                        f"{specie}-sub-{sub_N:02d}")
    a = os.path.join(base, f"r-{radius}_mahalanobis_{cat1}_{cat2}.nii.gz")
    b = os.path.join(base, f"r-{radius}_mahalanobis_{cat2}_{cat1}.nii.gz")
    path = a if os.path.exists(a) else (b if os.path.exists(b) else None)
    if path is None:
        raise FileNotFoundError(f"Missing step-1 map for {cat1} vs {cat2} under {base}")
    img = nib.load(path)
    return (np.asarray(img.dataobj, dtype=np.float64).reshape(-1), img,
            os.path.basename(path))


def load_meta_similarity(data_root, manifest, device):
    """Load the 45 step-1 maps once into a per-voxel matrix restricted to the mask.

    Returns ``(data (n_vox, 45) tensor, mask_flat_indices, shape, affine)`` in the
    canonical pair order used by every model.
    """
    mask_img, mask_bool = load_reference_mask(data_root, manifest)
    shape = mask_bool.shape
    mask_flat = np.flatnonzero(mask_bool.reshape(-1))

    pairs = canonical_pairs(manifest["categories"])
    cols, loaded = [], []
    for (c1, c2) in pairs:
        vol, img, name = _load_pairwise_map(data_root, manifest, c1, c2)
        loaded.append((name, img))
        cols.append(vol[mask_flat])
    # these are stacked into one per-voxel matrix and the result is written with
    # the mask's affine, so they must already be on the mask's grid
    check_same_space(("mask", mask_img), loaded,
                     context=f"GPU step 2/4 for {manifest['specie']}-sub-"
                             f"{manifest['sub_N']:02d}",
                     strict=not manifest.get("allow_space_mismatch", False))
    data = np.stack(cols, axis=1)                       # (n_vox, 45)
    data_t = torch.as_tensor(data, dtype=DTYPE, device=device)
    return data_t, mask_flat, shape, mask_img.affine


def load_meta_correlation(data_root, manifest, device, verbose=False):
    """Load every run's 780-pair correlation RDM once, reused across models.

    Returns ``{'runs': [{'session','run_N','data','mask_flat'}, ...], 'shape',
    'affine', 'mask_flat'}`` where each ``data`` is a CPU float32 (n_vox, 780)
    tensor (kept off the GPU; ``run_model_correlation`` moves one run at a time).
    """
    dataset, model, specie, sub_N, radius, task = (
        manifest["dataset"], manifest["model"], manifest["specie"], manifest["sub_N"],
        manifest["radius"], manifest["task"])
    mask_img, mask_bool = load_reference_mask(data_root, manifest)
    shape = mask_bool.shape
    mask_flat = np.flatnonzero(mask_bool.reshape(-1))
    pairs = canonical_pairs(manifest["categories"])                  # 780 stim pairs

    runs_out = []
    for entry in manifest["runs"]:
        session = f"{int(entry['session']):02d}"
        run_N = int(entry["run_N"])
        base = os.path.join(data_root, dataset, "results", "RSA", model,
                            f"{specie}-sub-{sub_N:02d}",
                            f"ses-{session}_task-{task}_run-{run_N:02d}")
        cols, loaded = [], []
        for (s1, s2) in pairs:
            a = os.path.join(base, f"r-{radius}_correlation_{s1}_{s2}.nii.gz")
            b = os.path.join(base, f"r-{radius}_correlation_{s2}_{s1}.nii.gz")
            path = a if os.path.exists(a) else (b if os.path.exists(b) else None)
            if path is None:
                raise FileNotFoundError(f"Missing step-1 correlation map {s1} vs {s2} in {base}")
            img = nib.load(path)
            loaded.append((os.path.basename(path), img))
            cols.append(np.asarray(img.dataobj, dtype=np.float64).reshape(-1)[mask_flat])
        check_same_space(("mask", mask_img), loaded,
                         context=(f"GPU step 2/4 (correlation) for {specie}-sub-"
                                  f"{sub_N:02d} ses-{session} run-{run_N:02d}"),
                         strict=not manifest.get("allow_space_mismatch", False))
        data = np.stack(cols, axis=1).astype(np.float32)             # (n_vox, 780)
        runs_out.append({"session": entry["session"], "run_N": run_N,
                         "data": torch.as_tensor(data), "mask_flat": mask_flat})
        if verbose:
            print(f"[meta-corr] loaded ses-{session} run-{run_N:02d}")
    return {"runs": runs_out, "shape": shape, "affine": mask_img.affine,
            "mask_flat": mask_flat}


def load_meta(data_root, manifest, device):
    """Load the reusable per-voxel RDM data for a package, dispatching on dis_method."""
    if manifest.get("dis_method", "mahalanobis") == "correlation":
        return load_meta_correlation(data_root, manifest, device)
    return load_meta_similarity(data_root, manifest, device)


def read_model_matrix(csv_path, categories):
    """Read an RSA model CSV into a (C, C) symmetric matrix over ``categories``.

    Matches ``read_model_dict`` + ``_get_rsa_model_value``: the diagonal is 0,
    off-diagonal entries come from the (symmetric) CSV, and grouping-masked pairs
    stay NaN so downstream correlation drops them.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    csv_categories = list(df.columns[1:])
    # value lookup by name (CSV is symmetric); row order == csv_categories
    body = df.iloc[:, 1:]
    body.index = csv_categories
    C = len(categories)
    M = np.full((C, C), np.nan, dtype=np.float64)
    for i, a in enumerate(categories):
        for j, b in enumerate(categories):
            if a == b:
                M[i, j] = 0.0
            else:
                M[i, j] = float(body.loc[a, b])
    return M


def _upper_tri_vector(M):
    """Flatten the canonical upper triangle (i < j) of a (C, C) matrix."""
    iu, ju = np.triu_indices(M.shape[0], 1)
    return M[iu, ju]


def build_model_vectors(M, reps, seed):
    """Real model vector + ``reps`` permuted vectors, using ``shuffle_vector``'s
    scheme (permute the C category labels, re-read the pairs).

    Returns an (reps+1, 45) array; row 0 is the real (identity) vector.
    """
    C = M.shape[0]
    iu, ju = np.triu_indices(C, 1)
    rng = np.random.default_rng(seed)
    vecs = [M[iu, ju]]                                   # real
    for _ in range(reps):
        perm = rng.permutation(C)
        Mp = M[np.ix_(perm, perm)]
        vecs.append(Mp[iu, ju])
    return np.stack(vecs, axis=0)                        # (reps+1, 45)


def _kendall_taua(data, model, vox_batch=0):
    """Kendall tau-a between every data row and every model row, NaN-aware.

    data  : (Nv, k)   per-voxel similarity vectors
    model : (Nm, k)   real + permuted model vectors
    returns (Nv, Nm) tau-a (float64), NaN where fewer than 2 jointly-valid items.

    Uses the sign-product factorization: tau numerator = Adata @ Bmodel^T where
    A/B are signed upper-triangle vectors (length k(k-1)/2) zeroed at invalid
    endpoints. The sign matrices are held in float32 and the matmul accumulates
    integers exactly (|entries| = 1, |sum| <= k(k-1)/2 << 2^24 even for k=780),
    so numerator/counts are exact; only the final division is float64.

    ``vox_batch`` chunks the voxel axis to bound the (vox_batch, k(k-1)/2) sign
    matrix -- required for the 780-item correlation RDM. 0 = all voxels at once.
    """
    dev = data.device
    Nv, k = data.shape
    Nm = model.shape[0]
    iu, ju = np.triu_indices(k, 1)
    iu = torch.as_tensor(iu, device=dev)
    ju = torch.as_tensor(ju, device=dev)

    # Signs are computed in the input dtype (so ties/near-ties resolve exactly as in
    # numpy's kendall_tau_a) and only then cast to float32 for an exact integer matmul.
    mvalid = torch.isfinite(model).to(torch.float32)
    m0 = torch.nan_to_num(model, nan=0.0)
    Bm = (torch.sign(m0[:, iu] - m0[:, ju]) * mvalid[:, iu] * mvalid[:, ju]).to(torch.float32)
    Bm_t = Bm.transpose(0, 1).contiguous()
    mvalid_t = mvalid.transpose(0, 1).contiguous()

    step = vox_batch if vox_batch and vox_batch > 0 else Nv
    out = torch.empty(Nv, Nm, dtype=torch.float64, device=dev)
    for s in range(0, Nv, step):
        d = data[s:s + step]
        dvalid = torch.isfinite(d).to(torch.float32)
        d0 = torch.nan_to_num(d, nan=0.0)
        A = (torch.sign(d0[:, iu] - d0[:, ju]) * dvalid[:, iu] * dvalid[:, ju]).to(torch.float32)
        numer = (A @ Bm_t).double()                     # exact integer -> f64
        n = (dvalid @ mvalid_t).double()                # jointly-valid counts (exact)
        denom = n * (n - 1.0) / 2.0
        out[s:s + step] = torch.where(
            denom > 0, numer / denom, torch.full_like(numer, float("nan")))
    return out


def _pearson(data, model, correlation=False):
    """Pearson (or 1-Pearson) between every data row and every model row.

    Matches the CPU ``np.corrcoef`` path: no per-item NaN removal, so a NaN in
    either vector propagates to NaN (grouping models therefore use kendall).
    """
    d = data - data.mean(dim=1, keepdim=True)
    m = model - model.mean(dim=1, keepdim=True)
    dn = d / torch.linalg.norm(d, dim=1, keepdim=True)
    mn = m / torch.linalg.norm(m, dim=1, keepdim=True)
    r = dn @ mn.transpose(0, 1)                          # (Nv, Nm)
    return (1.0 - r) if correlation else r


def compute_similarity(data, model, rsa_method, vox_batch=0):
    """Dispatch to the requested RSA metric -> (Nv, Nm) similarity.

    ``vox_batch`` only affects kendall (chunks the voxel axis for large RDMs).
    """
    if rsa_method == "kendall":
        return _kendall_taua(data, model, vox_batch=vox_batch)
    if rsa_method == "pearson":
        return _pearson(data, model, correlation=False)
    if rsa_method == "correlation":
        return _pearson(data, model, correlation=True)
    raise ValueError(f"Unsupported rsa_method: {rsa_method!r}")


def run_model(pkg_root, manifest, rsa_model, meta=None, device=None, seed=None,
              vox_batch=512, verbose=True):
    """Steps 2 (real) + 4 (permutations) for one RSA model, dispatching on dis_method."""
    if manifest.get("dis_method", "mahalanobis") == "correlation":
        return run_model_correlation(pkg_root, manifest, rsa_model, meta=meta,
                                     device=device, seed=seed, vox_batch=vox_batch,
                                     verbose=verbose)
    return run_model_mahalanobis(pkg_root, manifest, rsa_model, meta=meta,
                                 device=device, seed=seed, verbose=verbose)


def _save_masked(sim_col, mask_flat, shape, affine, path):
    """Write a per-voxel similarity column into a full volume (0 outside/where NaN)."""
    vol = np.zeros(int(np.prod(shape)), dtype=np.float64)
    good = np.isfinite(sim_col)
    vol[mask_flat[good]] = sim_col[good]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(vol.reshape(shape), affine), path)


def run_model_mahalanobis(pkg_root, manifest, rsa_model, meta=None, device=None,
                          seed=None, verbose=True):
    """Steps 2 + 4 for one RSA model, cross-run mahalanobis (one map per subject).

    ``meta`` is the cached ``load_meta_similarity`` result, reused across models.
    Writes the real map under ``results/RSA/{model}/{rsa_model}/{specie}-sub-NN/``
    and ``reps`` permutation maps under ``results/RSA_rnd/...``. Returns (real, rnd_dir).
    """
    device = device or pick_device()
    data_root = os.path.join(pkg_root, "data")
    dataset, model, specie, sub_N = (
        manifest["dataset"], manifest["model"], manifest["specie"], manifest["sub_N"])
    radius, rsa_method, reps, mask_type = (
        manifest["radius"], manifest["rsa_method"], manifest["reps"], manifest["mask_type"])
    categories = manifest["categories"]

    if meta is None:
        meta = load_meta_similarity(data_root, manifest, device)
    data_t, mask_flat, shape, affine = meta

    csv_path = os.path.join(data_root, dataset, "rsa_models", f"{rsa_model}.csv")
    M = read_model_matrix(csv_path, categories)
    if seed is None:
        # stable across processes (unlike hash()) so a recomputed model reproduces
        seed = zlib.crc32(f"{rsa_model}-{specie}-{sub_N}".encode())
    model_vecs = build_model_vectors(M, reps, seed)                # (reps+1, n_pairs)
    model_t = torch.as_tensor(model_vecs, dtype=DTYPE, device=device)

    t0 = time.time()
    sim = compute_similarity(data_t, model_t, rsa_method).cpu().numpy()  # (n_vox, reps+1)

    stem = f"r-{radius}_mahalanobis_{rsa_method}"
    if mask_type:
        stem = f"{mask_type}-{stem}"
    real_dir = os.path.join(data_root, dataset, "results", "RSA", model, rsa_model,
                            f"{specie}-sub-{sub_N:02d}")
    real_path = os.path.join(real_dir, f"{stem}.nii.gz")
    _save_masked(sim[:, 0], mask_flat, shape, affine, real_path)

    rnd_dir = os.path.join(data_root, dataset, "results", "RSA_rnd", model, rsa_model,
                           f"{specie}-sub-{sub_N:02d}")
    for r in range(reps):
        _save_masked(sim[:, r + 1], mask_flat, shape, affine,
                     os.path.join(rnd_dir, f"{stem}_{r:04d}.nii.gz"))

    if verbose:
        print(f"[model] {rsa_model}: real + {reps} perms in {time.time()-t0:.1f}s")
    return real_path, rnd_dir


def run_model_correlation(pkg_root, manifest, rsa_model, meta=None, device=None,
                          seed=None, vox_batch=512, verbose=True):
    """Steps 2 + 4 for one correlation RSA model -- computed and written PER RUN.

    ``meta`` is the cached ``load_meta_correlation`` result (per-run 780-pair RDMs),
    reused across models. Reproduces ``compare_with_model`` (non-mahalanobis): the
    real map is ``{mask_type}-r-{radius}_correlation_{rsa_method}.nii.gz`` per run;
    permutation maps are ``{mask_type}-r-{radius}_correlation_{rsa_method}_{NNNN}``.

    NOTE: the CPU ``compare_with_model`` omits the ``{mask_type}-`` prefix on the
    permutation files, but step 5's reader and the mahalanobis path both expect it,
    so we include it here (otherwise step 5 would not find the rnd maps).
    """
    device = device or pick_device()
    data_root = os.path.join(pkg_root, "data")
    dataset, model, specie, sub_N = (
        manifest["dataset"], manifest["model"], manifest["specie"], manifest["sub_N"])
    radius, rsa_method, reps, mask_type, task = (
        manifest["radius"], manifest["rsa_method"], manifest["reps"],
        manifest["mask_type"], manifest["task"])
    categories = manifest["categories"]

    if meta is None:
        meta = load_meta_correlation(data_root, manifest, device)
    shape, affine = meta["shape"], meta["affine"]

    csv_path = os.path.join(data_root, dataset, "rsa_models", f"{rsa_model}.csv")
    M = read_model_matrix(csv_path, categories)
    if seed is None:
        seed = zlib.crc32(f"{rsa_model}-{specie}-{sub_N}".encode())
    model_vecs = build_model_vectors(M, reps, seed)                    # (reps+1, 780)
    model_t = torch.as_tensor(model_vecs, dtype=DTYPE, device=device)

    prefix = f"{mask_type}-" if mask_type else ""
    stem = f"{prefix}r-{radius}_correlation_{rsa_method}"
    t0 = time.time()
    for run in meta["runs"]:
        session = f"{int(run['session']):02d}"
        run_N = int(run["run_N"])
        data_t = run["data"].to(device)                               # (n_vox, 780)
        sim = compute_similarity(data_t, model_t, rsa_method,
                                 vox_batch=vox_batch).cpu().numpy()    # (n_vox, reps+1)
        run_folder = f"ses-{session}_task-{task}_run-{run_N:02d}"
        real_path = os.path.join(data_root, dataset, "results", "RSA", model, rsa_model,
                                 f"{specie}-sub-{sub_N:02d}", run_folder, f"{stem}.nii.gz")
        _save_masked(sim[:, 0], run["mask_flat"], shape, affine, real_path)
        rnd_folder = os.path.join(data_root, dataset, "results", "RSA_rnd", model,
                                  rsa_model, f"{specie}-sub-{sub_N:02d}", run_folder)
        for r in range(reps):
            _save_masked(sim[:, r + 1], run["mask_flat"], shape, affine,
                         os.path.join(rnd_folder, f"{stem}_{r:04d}.nii.gz"))
        if device.type == "cuda":
            del data_t
            torch.cuda.empty_cache()
    if verbose:
        print(f"[model] {rsa_model}: {len(meta['runs'])} runs x (real + {reps}) "
              f"in {time.time()-t0:.1f}s")
    return rsa_model


# ===========================================================================
# zip helpers (one result zip per finished part)
# ===========================================================================
def _zip_paths(zip_path, data_root, paths):
    """Zip ``paths`` with arcnames relative to ``data_root`` (== pipeline-relative
    to the data folder), so unpack_results can merge them straight onto the disk."""
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for p in paths:
            zf.write(p, arcname=os.path.relpath(p, data_root))
    return zip_path


def _step1_glob(manifest):
    """Glob for step-1 maps -- flat for mahalanobis, recursive (per-run) for correlation."""
    dis = manifest.get("dis_method", "mahalanobis")
    radius = manifest["radius"]
    if dis == "correlation":
        return os.path.join("**", f"r-{radius}_correlation_*.nii.gz")
    return f"r-{radius}_mahalanobis_*.nii.gz"


def _step1_tag(manifest):
    """Short dis_method tag ('mah'/'corr') for step-1 result filenames -- without it,
    a participant's mahalanobis and correlation packages both write
    result_step1_{specie}-sub-NN.zip into the same OUT_DIR and overwrite each other."""
    return "corr" if manifest.get("dis_method", "mahalanobis") == "correlation" else "mah"


def zip_step1_result(pkg_root, manifest, out_dir):
    """Zip the step-1 maps -> result_step1_{mah,corr}_{specie}-sub-NN.zip (per-run
    for correlation)."""
    data_root = os.path.join(pkg_root, "data")
    dataset, model, specie, sub_N = (
        manifest["dataset"], manifest["model"], manifest["specie"], manifest["sub_N"])
    base = os.path.join(data_root, dataset, "results", "RSA", model,
                        f"{specie}-sub-{sub_N:02d}")
    paths = sorted(glob.glob(os.path.join(base, _step1_glob(manifest)), recursive=True))
    name = f"result_step1_{_step1_tag(manifest)}_{specie}-sub-{sub_N:02d}.zip"
    return _zip_paths(os.path.join(out_dir, name), data_root, paths)


def zip_model_result(pkg_root, manifest, rsa_model, out_dir):
    """Zip one model's step-2 + step-4 maps -> result_{rsa_model}_{specie}-sub-NN.zip.

    Recursive glob so it also catches the per-run correlation layout."""
    data_root = os.path.join(pkg_root, "data")
    dataset, model, specie, sub_N = (
        manifest["dataset"], manifest["model"], manifest["specie"], manifest["sub_N"])
    paths = []
    for kind in ("RSA", "RSA_rnd"):
        d = os.path.join(data_root, dataset, "results", kind, model, rsa_model,
                         f"{specie}-sub-{sub_N:02d}")
        paths += sorted(glob.glob(os.path.join(d, "**", "*.nii.gz"), recursive=True))
    name = f"result_{rsa_model}_{specie}-sub-{sub_N:02d}.zip"
    return _zip_paths(os.path.join(out_dir, name), data_root, paths)


def load_manifest(pkg_root):
    with open(os.path.join(pkg_root, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)

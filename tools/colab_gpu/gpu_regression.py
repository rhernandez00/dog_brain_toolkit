"""GPU kernels and pipeline-compatible paths for RSA steps 15--15.5.

Float64 OLS follows rsa_utils.perform_multiple_regression_rsa, including
standardization, finite-row grouping, pseudoinverse, degrees of freedom and
zero/one defaults for unfitted voxels. Only small design inverses and t CDFs
run on the CPU; voxel/permutation matrix products run on the selected device.
"""
import hashlib
import itertools
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

try:
    from . import gpu_rsa
except ImportError:
    import gpu_rsa

VERSION = '1.1.0'


def _response_patterns(data):
    """Group missingness once, with a fast path for ordinary finite data.

    Sorting 100k identical 435-column boolean rows for every permutation batch
    dominated CPU time. Pack exceptional patterns into bytes before sorting.
    """
    finite = np.isfinite(data)
    if finite.all():
        return np.ones((1, data.shape[1]), dtype=bool), np.zeros(len(data), dtype=int)
    packed, membership = np.unique(np.packbits(finite, axis=1), axis=0, return_inverse=True)
    return np.unpackbits(packed, axis=1, count=data.shape[1]).astype(bool), membership


def seed_for(seed, *parts):
    digest = hashlib.sha256(json.dumps([int(seed), *parts]).encode()).digest()
    return int.from_bytes(digest[:8], 'little')


def model_path(root, manifest, name, run):
    folder = Path(root) / 'data' / manifest['dataset'] / 'rsa_models'
    path = folder / f'{name}.csv'
    if not path.is_file():
        path = folder / f'{name}-run-{int(run)}.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing target/control matrix: {path}')
    return path


def read_matrix(path, categories=None):
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    if frame.index.has_duplicates or frame.columns.has_duplicates:
        raise ValueError(f'Duplicate model labels: {path}')
    categories = list(frame.columns) if categories is None else list(categories)
    missing = set(categories) - (set(frame.index) & set(frame.columns))
    if missing:
        raise ValueError(f'{path}: missing model categories {sorted(missing)}')
    values = frame.loc[categories, categories].to_numpy(dtype=np.float64)
    # The CPU reads the lower triangle in column order. Mirror that triangle,
    # including NaNs, rather than assuming the CSV has a filled upper triangle.
    lower = np.tril(values, -1)
    matrix = lower + lower.T
    return categories, matrix


def build_designs(root, manifest, target, controls, run, indices, seed=42):
    categories, matrix = read_matrix(model_path(root, manifest, target, run))
    pairs = list(itertools.combinations(categories, 2))
    triangle = np.triu_indices(len(categories), 1)
    control_vectors = [read_matrix(model_path(root, manifest, c, run), categories)[1][triangle]
                       for c in controls]
    designs = []
    for index in indices:
        permuted = matrix
        if index is not None:
            rng = np.random.default_rng(seed_for(seed, manifest['dataset'], manifest['specie'],
                                                manifest['sub_N'], run, target, index))
            order = rng.permutation(len(categories))
            permuted = matrix[np.ix_(order, order)]
        designs.append(np.column_stack([np.ones(len(pairs)), permuted[triangle], *control_vectors]))
    return pairs, np.asarray(designs)


def fit_designs(data, designs, device='cuda', voxel_batch=2048, permutation_batch=8,
                progress=None):
    """Yield (fit_index, beta/t/p vectors, dof), bounded GPU batches.

    data: (voxels, pairs), designs: (fits, pairs, regressors). NaN patterns can
    vary by voxel and by permutation; both are handled without dropping an
    otherwise usable voxel or using a different standardization from the CPU.
    """
    data = np.asarray(data)
    designs = np.asarray(designs, dtype=np.float64)
    if designs.ndim != 3 or data.ndim != 2 or data.shape[1] != designs.shape[1]:
        raise ValueError('Expected data (voxels,pairs), designs (fits,pairs,regressors).')
    if voxel_batch < 1 or permutation_batch < 1:
        raise ValueError('Batch sizes must be positive.')
    device = torch.device(device)
    n_vox = len(data)
    n_reg = designs.shape[-1]
    started = last_report = time.monotonic()
    if progress:
        progress(f'Preparing finite-row patterns: {n_vox:,} voxels, {data.shape[1]} pairs')
    response_patterns, response_membership = _response_patterns(data)
    if progress:
        progress(f'{len(response_patterns)} response pattern(s); starting {len(designs)} fits')
    group_cache = {}
    for start in range(0, len(designs), permutation_batch):
        batch = designs[start:start + permutation_batch]
        betas = np.zeros((len(batch), n_vox))
        ts = np.zeros_like(betas)
        ps = np.ones_like(betas)
        dofs = np.zeros(len(batch), dtype=int)
        covered = np.zeros(len(batch), dtype=int)
        # Group fits with the same finite rows to share response transforms.
        fit_groups = {}
        for i, design in enumerate(batch):
            finite = np.isfinite(design).all(axis=1)
            fit_groups.setdefault(finite.tobytes(), []).append(i)
        for key, fit_indices in fit_groups.items():
            finite = np.frombuffer(key, dtype=bool)
            if key not in group_cache:
                if len(response_patterns) == 1:
                    groups = [(np.flatnonzero(response_patterns[0] & finite), np.arange(n_vox))]
                else:
                    # Only distinct response patterns enter this sort, not
                    # every voxel again. Excluded design rows can merge them.
                    patterns, remap = np.unique(response_patterns & finite[None, :],
                                                axis=0, return_inverse=True)
                    membership = remap[response_membership]
                    groups = [(np.flatnonzero(pattern), np.flatnonzero(membership == group))
                              for group, pattern in enumerate(patterns)]
                if len(group_cache) >= 8:
                    group_cache.pop(next(iter(group_cache)))
                group_cache[key] = groups
            for rows, voxels in group_cache[key]:
                if len(rows) <= n_reg:
                    continue
                x = batch[fit_indices][:, rows].copy()
                scales = x.std(axis=1, keepdims=True)
                varying = scales > 0
                x = np.where(varying, (x - x.mean(axis=1, keepdims=True)) /
                             np.where(varying, scales, 1), x)
                df = len(rows) - np.linalg.matrix_rank(x)
                inverse = np.linalg.pinv(x.transpose(0, 2, 1) @ x, rcond=1e-15)
                projector = inverse @ x.transpose(0, 2, 1)
                xt = torch.as_tensor(x, dtype=torch.float64, device=device)
                pt = torch.as_tensor(projector, dtype=torch.float64, device=device)
                variance = torch.as_tensor(inverse[:, 1, 1], device=device)[:, None]
                dft = torch.as_tensor(df, device=device)[:, None]
                for v in range(0, len(voxels), voxel_batch):
                    cols = voxels[v:v + voxel_batch]
                    y = torch.as_tensor(np.ascontiguousarray(data[np.ix_(cols, rows)].T),
                                        dtype=torch.float64, device=device)
                    y = y - y.mean(dim=0, keepdim=True)
                    sd = y.std(dim=0, correction=0, keepdim=True)
                    y = y / torch.where(sd > 0, sd, torch.ones_like(sd))
                    coefficients = pt @ y
                    residual = y[None, :, :] - xt @ coefficients
                    sigma2 = (residual * residual).sum(dim=1) / dft
                    se = torch.sqrt(torch.clamp(sigma2 * variance, min=0))
                    b = torch.nan_to_num(coefficients[:, 1, :], nan=0., posinf=0., neginf=0.)
                    t = torch.nan_to_num(torch.where(se > 0, b / se, 0.),
                                         nan=0., posinf=0., neginf=0.)
                    bn, tn = b.cpu().numpy(), t.cpu().numpy()
                    ix = np.ix_(fit_indices, cols)
                    betas[ix], ts[ix] = bn, tn
                    ps[ix] = 2 * student_t.sf(np.abs(tn), df[:, None])
                    now = time.monotonic()
                    if progress and now - last_report >= 30:
                        progress(f'Fits {start + 1}-{start + len(batch)}/{len(designs)}: '
                                 f'{min(v + voxel_batch, len(voxels)):,}/{len(voxels):,} '
                                 f'voxels in pattern; elapsed {now - started:.0f}s')
                        last_report = now
                for i, d in zip(fit_indices, df):
                    if len(voxels) > covered[i]:
                        covered[i], dofs[i] = len(voxels), d
        for i in range(len(batch)):
            yield start + i, betas[i], ts[i], ps[i], int(dofs[i])
        if progress:
            progress(f'Fits {start + len(batch)}/{len(designs)} computed and written '
                     f'({time.monotonic() - started:.0f}s)')


def run_folder(manifest, regression_model, target, entry, rnd=False):
    return (Path(manifest['dataset']) / 'results' /
            ('RSA_regression_rnd' if rnd else 'RSA_regression') /
            manifest['model'] / regression_model / target /
            f"{manifest['specie']}-sub-{manifest['sub_N']:02d}" /
            f"ses-{int(entry['session']):02d}_task-{manifest['task']}_run-{int(entry['run_N']):02d}")


def map_stem(manifest, index=None):
    return f"r-{manifest['radius']}_{manifest['dis_method']}", ('' if index is None else f'_{index:04d}')


def save_vector(path, data, mask, dtype=np.float32, outside=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    volume = np.full(mask.shape, outside, dtype=dtype)
    volume[mask.get_fdata() > 0] = data
    nib.save(nib.Nifti1Image(volume, mask.affine), str(path))


def load_run_data(root, manifest, entry, mask, device, step1_batch=256, progress=None):
    """Reuse step-1 maps when complete; otherwise compute from packaged betas."""
    data_root = Path(root) / 'data'
    base = data_root / manifest['dataset'] / 'results/RSA' / manifest['model'] / (
        f"{manifest['specie']}-sub-{manifest['sub_N']:02d}")
    correlation = manifest['dis_method'] == 'correlation'
    if correlation:
        base /= (f"ses-{int(entry['session']):02d}_task-{manifest['task']}_"
                 f"run-{int(entry['run_N']):02d}")
    pairs = list(itertools.combinations(manifest['categories'], 2))
    paths = []
    for a, b in pairs:
        stem = f"r-{manifest['radius']}_{manifest['dis_method']}"
        path = base / f'{stem}_{a}_{b}.nii.gz'
        if not path.is_file():
            path = base / f'{stem}_{b}_{a}.nii.gz'
        paths.append(path)
    mask_bool = mask.get_fdata() > 0
    if all(p.is_file() for p in paths):
        started = last_report = time.monotonic()
        if progress:
            progress(f'Loading {len(paths)} step-1 NIfTI maps for {int(mask_bool.sum()):,} voxels')
        # Preserve existing float64 step-1 inputs; narrowing them here would
        # change the CPU fit before the float64 solver even starts.
        data = np.empty((int(mask_bool.sum()), len(pairs)), dtype=np.float64)
        for i, path in enumerate(paths):
            img = nib.load(str(path))
            gpu_rsa.check_same_space(('mask', mask), [(str(path), img)])
            data[:, i] = img.get_fdata()[mask_bool]
            now = time.monotonic()
            if progress and (i == 0 or (i + 1) % 50 == 0 or now - last_report >= 30 or i + 1 == len(paths)):
                progress(f'Loaded step-1 maps {i + 1}/{len(paths)} ({now - started:.0f}s)')
                last_report = now
        return data, pairs
    print('Step-1 maps incomplete; computing pairwise data on the selected device.')
    if correlation:
        if list(manifest['categories']) != list(manifest['stim_types']):
            raise ValueError('Correlation package categories must match stim_types order.')
        betas, _ = gpu_rsa._load_run_betas(str(data_root), manifest, entry, mask_bool, ref_img=mask)
        data, _ = gpu_rsa.pearson_rdm_searchlight(betas, mask_bool, manifest['radius'],
                                               device=device, batch=step1_batch, verbose=progress is not None)
        return data, pairs
    gpu_rsa.run_step1(str(root), manifest, device=device, batch=step1_batch)
    return load_run_data(root, manifest, entry, mask, device, step1_batch)


def group_moments(bank, selections, device='cuda', voxel_batch=4096, group_batch=16):
    """Yield bounded chunks of mean/std across units (population std).

    bank is a CPU/disk array (input_maps, voxels); selections is (draws, units).
    Each input bank is loaded once by the caller, avoiding thousands of repeated
    NIfTI reads. Neither the complete bank nor all group draws live on the GPU.
    """
    selections = np.asarray(selections, dtype=int)
    if selections.ndim != 2 or not selections.shape[1]:
        raise ValueError('No averaging units selected.')
    if min(voxel_batch, group_batch) < 1:
        raise ValueError('Batch sizes must be positive.')
    for g in range(0, len(selections), group_batch):
        indices = selections[g:g + group_batch]
        for v in range(0, bank.shape[1], voxel_batch):
            # CPU advanced indexing touches only this chunk of the disk bank.
            values = torch.as_tensor(np.asarray(bank[indices, v:v + voxel_batch]),
                                     dtype=torch.float64, device=device)
            mean = values.mean(dim=1)
            std = values.std(dim=1, correction=0)
            yield g, v, mean.cpu().numpy(), std.cpu().numpy()

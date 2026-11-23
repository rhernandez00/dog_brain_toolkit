"""Float64 GPU kernels for beta-omission steps 1 and 11.2.

Only Mahalanobis stim-wise inputs are used, as in create_package.py. Partitions
are individual session/run entries, matching leave_one_stimulus_out.py.
"""
import numpy as np
import nibabel as nib
import torch

import gpu_rsa as gpu
from stimulus_contribution import _correlate_omissions_torch

VERSION = 1


def category(stimulus, dataset):
    if dataset == 'EmoB' and '-' in stimulus:
        return stimulus.split('-')[0]
    if dataset == 'EmoC':
        return stimulus[:-1]
    raise ValueError(f'Cannot determine category for {dataset}: {stimulus}')


def batched_crossnobis(U, present):
    """Match rsa_utils.crossnobis(..., allow_missing=True), including covariance."""
    B, M, C, P = U.shape
    present = torch.as_tensor(present, dtype=torch.bool, device=U.device)
    D = torch.full((B, C, C), torch.nan, dtype=U.dtype, device=U.device)
    D.diagonal(dim1=-2, dim2=-1).zero_()
    if M < 2:
        return D
    means = (U * present[None, :, :, None]).sum(1) / present.sum(0).clamp_min(1)[None, :, None]
    residuals = (U - means[:, None])[:, present, :]
    covariance = gpu.batched_ledoit_wolf(residuals)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    inverse = (eigenvectors / eigenvalues.clamp_min(gpu.EPS).sqrt()[:, None, :]) @ eigenvectors.transpose(-1, -2)
    Z = torch.einsum('bmcp,bpk->bmck', U, inverse)
    i, j = torch.triu_indices(C, C, offset=1, device=U.device)
    common = present[:, i] & present[:, j]
    count = common.sum(0)
    delta = Z[:, :, i] - Z[:, :, j]
    delta = torch.where(common[None, :, :, None], delta, 0.)
    numerator = delta.sum(1).square().sum(-1) - delta.square().sum((1, 3))
    values = numerator / (count * (count - 1)).clamp_min(1)
    values = torch.where(count[None] >= 2, values, torch.nan)
    D[:, i, j] = D[:, j, i] = values
    return D


def neighborhoods(mask, radius):
    """Build ragged searchlight indices once, reusing them for every omission."""
    centers = np.argwhere(mask)
    lookup = np.full(mask.shape, -1, dtype=np.int32)
    lookup[mask] = np.arange(len(centers), dtype=np.int32)
    offsets = gpu.sphere_offsets(radius, mask.ndim)
    groups = {}
    for index, center in enumerate(centers):
        neighbors = center + offsets
        neighbors = neighbors[np.all((neighbors >= 0) & (neighbors < mask.shape), axis=1)]
        neighbors = neighbors[mask[tuple(neighbors.T)]]
        groups.setdefault(len(neighbors), []).append((index, lookup[tuple(neighbors.T)]))
    return [(np.asarray([row[0] for row in rows]), np.stack([row[1] for row in rows]))
            for _, rows in sorted(groups.items())]


def load_betas(root, manifest, reference, mask, destination):
    """Disk-backed masked betas; only one participant needs to be staged at once."""
    shape = (len(manifest['runs']), len(manifest['stim_types']), int(mask.sum()))
    betas = np.lib.format.open_memmap(destination, mode='w+', dtype=np.float64, shape=shape)
    for unit, entry in enumerate(manifest['runs']):
        for index, stimulus in enumerate(manifest['stim_types']):
            path = gpu._beta_path(str(root / 'data'), manifest['dataset'], manifest['model'],
                manifest['specie'], manifest['sub_N'], entry['session'], entry['run_N'],
                manifest['task'], stimulus)
            image = nib.load(path)
            gpu.check_same_space(('mask', reference), [(path, image)], context='Colab beta omissions',
                                 strict=not manifest.get('allow_space_mismatch', False))
            if image.shape != reference.shape:
                raise ValueError(f'Expected a 3D beta map: {path}')
            betas[unit, index] = image.get_fdata()[mask]
    betas.flush()
    return betas


def category_sums(betas, manifest):
    labels = [category(stimulus, manifest['dataset']) for stimulus in manifest['stim_types']]
    categories = manifest['categories']
    sums = np.stack([betas[:, np.array(labels) == label].sum(1) for label in categories], axis=1)
    counts = np.tile([labels.count(label) for label in categories], (len(betas), 1))
    return sums, counts, [categories.index(label) for label in labels]


def omission_maps(betas, sums, counts, unit, stimulus_index, category_index,
                  groups, pairs, categories, device='cpu', batch=256):
    """Reestimate means and covariance after removing precisely one beta."""
    if batch < 1:
        raise ValueError('STEP1_BATCH must be positive.')
    remaining = counts.copy()
    remaining[unit, category_index] -= 1
    means = sums.copy()
    means[unit, category_index] -= betas[unit, stimulus_index]
    means /= np.maximum(remaining, 1)[:, :, None]
    means[remaining == 0] = 0
    means = torch.as_tensor(means, dtype=torch.float64, device=device)
    output = np.full((betas.shape[-1], len(pairs)), np.nan)
    pair_indices = [(categories.index(a), categories.index(b)) for a, b in pairs]
    for centers, neighbors in groups:
        for start in range(0, len(centers), batch):
            indices = torch.as_tensor(neighbors[start:start + batch], device=device, dtype=torch.long)
            U = means[:, :, indices].permute(2, 0, 1, 3)
            distances = batched_crossnobis(U, remaining > 0)
            values = torch.stack([distances[:, a, b] for a, b in pair_indices], dim=1)
            output[centers[start:start + batch]] = values.cpu().numpy()
    return output


def finite_mean(values, axis=0):
    valid = torch.isfinite(values)
    count = valid.sum(dim=axis)
    total = torch.where(valid, values, 0).sum(dim=axis)
    return torch.where(count > 0, total / count.clamp_min(1), torch.nan)


def group_similarity(bank, participant_rows, method='kendall', device='cpu',
                     voxel_batch=128, comparison_batch=4096):
    """Session-target vs equally weighted other-participant mean; group mean."""
    if voxel_batch < 1 or comparison_batch < 1:
        raise ValueError('VOXEL_BATCH and COMPARISON_BATCH must be positive.')
    result = np.full(bank.shape[1], np.nan)
    keep = np.ones((1, bank.shape[-1]), dtype=bool)
    for start in range(0, bank.shape[1], voxel_batch):
        values = torch.as_tensor(np.array(bank[:, start:start + voxel_batch]),
                                 dtype=torch.float64, device=device)
        means = torch.stack([finite_mean(values[rows]) for rows in participant_rows])
        participants = []
        for index, rows in enumerate(participant_rows):
            peers = finite_mean(torch.cat([means[:index], means[index + 1:]]))
            sessions = []
            for row in rows:
                correlations = _correlate_omissions_torch(
                    values[row], peers, keep,
                    method, device, comparison_batch)[:, 0]
                sessions.append(torch.as_tensor(correlations, device=device))
            participants.append(finite_mean(torch.stack(sessions)))
        result[start:start + values.shape[1]] = finite_mean(torch.stack(participants)).cpu().numpy()
    return result

"""Leave one video out, then compute participant-to-peer RSA (step 11.31).

Dependency-light shared CPU/Colab implementation. Kendall uses tau-b, including
tie correction, as scipy.stats.kendalltau does in step 11.2.
"""
from contextlib import contextmanager
import gzip
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import nibabel as nib
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

VERSION = '1.1.0'


def read_pairs(path):
    """The selected rsa_class CSV alone defines pairs and possible omissions."""
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'cat1', 'cat2'}.issubset(reader.fieldnames or []):
            raise ValueError('Class CSV requires cat1 and cat2 columns.')
        pairs = [((row['cat1'] or '').strip(), (row['cat2'] or '').strip()) for row in reader]
    return validate_pairs(pairs)[0]


def validate_pairs(pairs):
    pairs = [tuple(pair) for pair in pairs]
    if (len(pairs) < 3 or any(len(p) != 2 or not all(p) or p[0] == p[1] for p in pairs)
            or len({tuple(sorted(p)) for p in pairs}) != len(pairs)):
        raise ValueError('Step 11.31 requires unique, non-diagonal cat1/cat2 pairs.')
    stimuli = list(dict.fromkeys(s for pair in pairs for s in pair))
    for stimulus in stimuli:
        if Path(stimulus).name != stimulus or any(c in stimulus for c in '/\\:'):
            raise ValueError(f'Invalid stimulus label: {stimulus!r}')
        if sum(stimulus not in pair for pair in pairs) < 2:
            raise ValueError(f'Fewer than two pairs remain after removing {stimulus}.')
    return pairs, stimuli


def correlate_omissions(target, peers, pairs, method='kendall', device='cpu',
                        comparison_batch=4096):
    """Return (voxels, videos) correlations; remove video from BOTH RDMs.

    On CUDA, each comparison of RDM entries is computed once and contributes
    to every omission retaining both entries. This avoids repeating Kendall's
    quadratic work forty times. Integer-valued float32 sums are exact for these
    RDM sizes; the tau-b division is float64. Call with bounded voxel batches.
    """
    pairs, stimuli = validate_pairs(pairs)
    keep = np.array([[s not in p for p in pairs] for s in stimuli])
    if method not in ('kendall', 'spearman', 'pearson'):
        raise ValueError('rsa_method must be kendall, spearman, or pearson.')
    if not str(device).startswith('cuda'):
        correlate = dict(kendall=kendalltau, spearman=spearmanr, pearson=pearsonr)[method]
        result = np.full((len(target), len(stimuli)), np.nan)
        for j, selected in enumerate(keep):
            for i, (a, b) in enumerate(zip(target[:, selected], peers[:, selected])):
                if np.ptp(a) > 0 and np.ptp(b) > 0:
                    result[i, j] = correlate(a, b)[0]
        return result
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no Colab GPU is available. Select a GPU runtime.')
    return _correlate_omissions_torch(target, peers, keep, method, device, comparison_batch)


def _correlate_omissions_torch(target, peers, keep, method, device, comparison_batch):
    """Torch kernel, also exercised on CPU by the numerical regression tests."""
    import torch
    nstimuli, npairs = keep.shape
    x = torch.as_tensor(target, dtype=torch.float64, device=device)
    y = torch.as_tensor(peers, dtype=torch.float64, device=device)
    selected = torch.as_tensor(keep, device=device)
    if method == 'kendall':
        if npairs * (npairs - 1) // 2 >= 2 ** 24:
            raise ValueError('RDM too large for exact float32 Kendall accumulators.')
        if comparison_batch < 1:
            raise ValueError('comparison_batch must be positive.')
        indices = torch.triu_indices(npairs, npairs, 1, device=device)
        stats = torch.zeros((3 * len(x), nstimuli), dtype=torch.float32, device=device)
        for start in range(0, indices.shape[1], comparison_batch):
            a, b = indices[:, start:start + comparison_batch]
            sx = torch.sign(x[:, a] - x[:, b]).float()
            sy = torch.sign(y[:, a] - y[:, b]).float()
            eligible = (selected[:, a] & selected[:, b]).T.float()
            stats += torch.cat((sx * sy, sx.abs(), sy.abs()), dim=0) @ eligible
        numerator, nx, ny = stats.double().chunk(3, dim=0)
        result = numerator / torch.sqrt(nx * ny)
    else:
        def ranks(values):
            sorted_values, order = values.sort(dim=1)
            n = values.shape[1]
            pos = torch.arange(n, device=device).expand_as(order)
            first = torch.ones_like(order, dtype=torch.bool)
            first[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
            last = torch.ones_like(first)
            last[:, :-1] = sorted_values[:, :-1] != sorted_values[:, 1:]
            lo = torch.where(first, pos, 0).cummax(dim=1).values
            hi = torch.where(last, pos, n).flip((1,)).cummin(dim=1).values.flip((1,))
            return torch.empty_like(values).scatter_(1, order, (lo + hi).double() / 2)
        columns = []
        for mask in selected:
            a, b = x[:, mask], y[:, mask]
            if method == 'spearman':
                a, b = ranks(a), ranks(b)
            a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
            columns.append((a * b).sum(1) / torch.sqrt((a * a).sum(1) * (b * b).sum(1)))
        result = torch.stack(columns, dim=1)
    return result.cpu().numpy()


class MapStore:
    """Read pipeline-relative maps from extracted trees or existing result ZIPs."""
    def __init__(self, roots):
        self.roots = [Path(p) for p in roots]
        self.members = {}
        for root in self.roots:
            archives = [root] if root.suffix.lower() == '.zip' else sorted(root.glob('*.zip'))
            for path in archives:
                with zipfile.ZipFile(path) as archive:
                    for name in archive.namelist():
                        if '/results/RSA/' in name and name.endswith('.nii.gz'):
                            key = name.removeprefix('data/')
                            self.members.setdefault(key, (path, name))

    def resolve(self, relative):
        for root in self.roots:
            if root.is_dir():
                for path in (root / relative, root / 'data' / relative):
                    if path.is_file():
                        return str(path)
        return self.members.get(relative)

    def load(self, source):
        if isinstance(source, str):
            return nib.load(source)
        archive, member = source
        with zipfile.ZipFile(archive) as handle:
            return nib.Nifti1Image.from_bytes(gzip.decompress(handle.read(member)))


def pair_sources(manifest, store, sub, session, run):
    base = (f"{manifest['dataset']}/results/RSA/{manifest['model']}/"
            f"{manifest['specie']}-sub-{sub:02d}")
    if run is not None:
        base += f"/ses-{session:02d}_task-{manifest['task']}_run-{run:02d}"
    sources = []
    for a, b in manifest['pairs']:
        stem = f"r-{manifest['radius']}_{manifest['dis_method']}_"
        source = store.resolve(f'{base}/{stem}{a}_{b}.nii.gz')
        if source is None:
            source = store.resolve(f'{base}/{stem}{b}_{a}.nii.gz')
        if source is None:
            raise FileNotFoundError(f'Missing step-1 map: {base}/{stem}{a}_{b}.nii.gz')
        sources.append(source)
    return sources


def check_grid(image, ref, path):
    # Same corner-displacement criterion as rsa_utils.check_same_space.
    from itertools import product
    if image.shape != ref.shape:
        raise ValueError(f'Voxel grid shape mismatch: {path}: {image.shape} != {ref.shape}')
    corners = np.array(list(product(*[(0, n - 1) for n in ref.shape])) +
                       [tuple((np.array(ref.shape) - 1) / 2)])
    hom = np.c_[corners, np.ones(len(corners))]
    offset = np.linalg.norm((hom @ image.affine.T - hom @ ref.affine.T)[:, :3], axis=1).max()
    if offset > 0.5:
        raise ValueError(f'Voxel grid affine mismatch ({offset:.3f} mm): {path}')


def _atomic_json(path, value):
    partial = Path(str(path) + '.part')
    partial.write_text(json.dumps(value, indent=2))
    os.replace(partial, path)


def run(manifest, roots, output_root, work_root, *, device='cpu', voxel_batch=128,
        comparison_batch=4096, replace_file=False, verbose=True):
    """Write one run-specific group map for every omitted stimulus.

    Each session is correlated against equally weighted peer participant means;
    session correlations are averaged within participant, then participants
    equally. Per-run inputs stay separate; pooled Mahalanobis is processed once.
    Missing pair maps are fatal, absent runs exclude participants with a warning.
    No step-11.2/11.3 files are changed. Disk-backed RDMs bound host RAM usage.
    """
    import warnings
    pairs, stimuli = validate_pairs(manifest['pairs'])
    participants = list(manifest['participants'])
    if len(participants) < 2 or len(set(participants)) != len(participants):
        raise ValueError('At least two unique participants are required.')
    if voxel_batch < 1 or not 0 <= manifest.get('min_percentage_available', 0) <= 1:
        raise ValueError('Invalid voxel_batch or min_percentage_available.')
    fold = manifest.get('mah_fold', 'stim-wise')
    if manifest['dis_method'] == 'mahalanobis' and fold in ('stim-wise', 'run-wise', 'stim-wise-multiple-folds'):
        units = {s: [(None, None)] for s in participants}
    elif (manifest['dis_method'] == 'correlation' or
          (manifest['dis_method'] == 'mahalanobis' and fold == 'stim-wise-all-runs')):
        units = {s: sorted({(int(e['session']), int(e['run_N']))
                            for e in manifest['runs_by_sub'][str(s)]}) for s in participants}
    else:
        raise ValueError('Unsupported dis_method/mah_fold for step 11.31.')
    runs = sorted({r for entries in units.values() for _, r in entries}, key=lambda r: -1 if r is None else r)
    if not runs:
        raise ValueError('No runs found.')
    ref = nib.load(str(manifest['mask']))
    mask_values = np.asarray(ref.dataobj)
    inside = np.isfinite(mask_values) & (mask_values != 0)
    if inside.ndim != 3 or not inside.any():
        raise ValueError('Mask must be a nonempty 3D image.')
    store = MapStore(roots)
    work_root, output_root = Path(work_root), Path(output_root)
    work_root.mkdir(parents=True, exist_ok=True)
    paths = []
    for run_number in runs:
        cohort = [s for s in participants if any(r == run_number for _, r in units[s])]
        absent = sorted(set(participants) - set(cohort))
        if absent:
            warnings.warn(f'Run {run_number}: excluding participants with no session: {absent}')
        if len(cohort) < 2 or len(cohort) / len(participants) < manifest.get('min_percentage_available', 0):
            raise ValueError(f'Run {run_number}: insufficient participants: {cohort}')
        folder = (output_root / manifest['dataset'] / 'results/RSA_same' / manifest['model'] /
                  f"class_{manifest['rsa_class'].removeprefix('class_')}" / 'leave-one-out')
        if manifest['dis_method'] == 'mahalanobis' and fold != 'stim-wise':
            folder /= fold
        folder = folder / 'by_stim' / (f'run-{run_number:02d}' if run_number is not None else 'pooled')
        folder.mkdir(parents=True, exist_ok=True)
        prefix = f"{manifest['mask_type']}-" if manifest.get('mask_type') else ''
        outputs = [folder / f"{prefix}r-{manifest['radius']}_{manifest['rsa_method']}_{s}.nii.gz"
                   for s in stimuli]
        # Input identity invalidates resume when cohort, class or source maps change.
        sources = {(s, session): pair_sources(manifest, store, s, session, r)
                   for s in cohort for session, r in units[s] if r == run_number}
        identity = []
        for key, entries in sources.items():
            for source in entries:
                file = Path(source if isinstance(source, str) else source[0])
                stat = file.stat()
                identity.append([key, str(source), stat.st_size, stat.st_mtime_ns])
        signature = hashlib.sha256(json.dumps(dict(
            manifest=manifest, identity=identity, version=VERSION,
            mask_stat=[Path(manifest['mask']).stat().st_size, Path(manifest['mask']).stat().st_mtime_ns]),
            sort_keys=True).encode()).hexdigest()
        pending = []
        for j, path in enumerate(outputs):
            sidecar = Path(str(path).removesuffix('.nii.gz') + '.json')
            try:
                complete = (path.is_file() and Path(str(path).replace('.nii.gz', '_std.nii.gz')).is_file()
                            and json.loads(sidecar.read_text()).get('signature') == signature)
            except (OSError, ValueError):
                complete = False
            if replace_file or not complete:
                pending.append(j)
        paths.extend(str(p) for p in outputs)
        if not pending:
            if verbose:
                print(f'Run {run_number}: all {len(outputs)} omissions already complete.', flush=True)
            continue
        if verbose:
            print(f'Run {run_number}: {len(cohort)} participants, {len(pairs)} pairs, '
                  f'{len(stimuli)} videos, device={device}. Loading RDMs.', flush=True)
        # Scratch lifetime is one run; remove mmap handles before Windows cleanup.
        with tempfile.TemporaryDirectory(dir=work_root, prefix='rdms-') as scratch:
            with _mapped_rdms(scratch, sources, store, ref, inside, len(pairs)) as rdms:
                nvox = int(inside.sum())
                group = np.empty((nvox, len(stimuli)), dtype=np.float64)
                std = np.empty_like(group)
                for start in range(0, nvox, voxel_batch):
                    stop = min(nvox, start + voxel_batch)
                    means = {s: np.mean([a[start:stop] for (sub, _), a in rdms.items() if sub == s], axis=0)
                             for s in cohort}
                    total = sum(means.values())
                    participant_maps = []
                    for sub in cohort:
                        peers = (total - means[sub]) / (len(cohort) - 1)
                        session_maps = [correlate_omissions(a[start:stop], peers, pairs,
                                        method=manifest['rsa_method'], device=device,
                                        comparison_batch=comparison_batch)
                                        for (s, _), a in rdms.items() if s == sub]
                        participant_maps.append(np.mean(session_maps, axis=0))
                    group[start:stop] = np.mean(participant_maps, axis=0)
                    std[start:stop] = np.std(participant_maps, axis=0)
                    if verbose and (start == 0 or stop == nvox or (start // voxel_batch) % 100 == 0):
                        print(f'Run {run_number}: {stop}/{nvox} voxels.', flush=True)
                for j in pending:
                    path = outputs[j]
                    for values, output in ((group[:, j], path), (std[:, j], Path(str(path).replace('.nii.gz', '_std.nii.gz')))):
                        volume = np.zeros(ref.shape, dtype=np.float64)
                        volume[inside] = values  # Undefined correlations propagate; outside mask = 0.
                        partial = output.with_name(output.name.replace('.nii.gz', '.part.nii.gz'))
                        nib.save(nib.Nifti1Image(volume, ref.affine), str(partial))
                        os.replace(partial, output)
                    _atomic_json(Path(str(path).replace('.nii.gz', '.json')), dict(
                        signature=signature, version=VERSION, stimulus_removed=stimuli[j], run=run_number,
                        participants_requested=participants, participants_included=cohort,
                        participants_missing=absent, pairs_before=len(pairs),
                        pairs_after=sum(stimuli[j] not in p for p in pairs),
                        dis_method=manifest['dis_method'], mah_fold=fold, task=manifest['task'],
                        rsa_method=manifest['rsa_method'],
                        kendall_variant='b' if manifest['rsa_method'] == 'kendall' else None,
                        weighting='sessions within participant, then participants equally; runs separate',
                        statistic='group similarity after omission (not full-minus-omitted difference)',
                        source_inventory=identity, device=str(device)))
                # Do not retain mmap slices via a loop variable at cleanup.
                del means, total, participant_maps, session_maps
    return paths


@contextmanager
def _mapped_rdms(scratch, sources, store, ref, inside, npairs):
    arrays = {}
    try:
        for (sub, session), entries in sources.items():
            path = Path(scratch) / f'{sub}-{session}.npy'
            array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64,
                                              shape=(int(inside.sum()), npairs))
            arrays[sub, session] = array
            for j, source in enumerate(entries):
                image = store.load(source)
                check_grid(image, ref, source)
                values = np.asarray(image.dataobj)[inside]
                if not np.isfinite(values).all():
                    raise ValueError(f'Non-finite RDM values inside mask: {source}')
                array[:, j] = values
            array.flush()
        yield arrays
    finally:
        for array in arrays.values():
            array._mmap.close()

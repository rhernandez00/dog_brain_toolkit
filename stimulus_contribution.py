"""Leave one video out, then compute participant-to-peer RSA (step 11.31).

Dependency-light shared CPU/Colab implementation. Kendall uses tau-b, including
tie correction, as scipy.stats.kendalltau does in step 11.2.
"""
from contextlib import contextmanager, ExitStack
import gzip
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

import nibabel as nib
import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr

VERSION = '1.4.0'
MISSING_DATA_POLICY = 'ignore NaN/Inf in means/std; pairwise finite correlation; undefined correlations remain NaN'


def finite_divide(total, count):
    """Return NaN when no finite observations remain, without divide warnings."""
    total = np.asarray(total, dtype=np.float64)
    return np.divide(total, count, out=np.full_like(total, np.nan), where=np.asarray(count) > 0)


def finite_totals(values):
    total, count = 0, 0
    for value in values:
        value = np.asarray(value, dtype=np.float64)
        valid = np.isfinite(value)
        total = total + np.where(valid, value, 0.0)
        count = count + valid.astype(np.int64)
    return total, count


def finite_mean(values, axis=0, keepdims=False):
    """Mean of available finite values; ignore both NaN and +/-Inf."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    total = np.sum(np.where(valid, values, 0.0), axis=axis, keepdims=keepdims)
    count = valid.sum(axis=axis, keepdims=keepdims)
    return finite_divide(total, count)


def finite_std(values, axis=0):
    """Population standard deviation of available finite values (ddof=0)."""
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    mean = finite_mean(values, axis=axis, keepdims=True)
    centered = np.where(valid, values, 0.0) - np.where(np.isfinite(mean), mean, 0.0)
    variance = finite_divide(np.sum(np.where(valid, centered * centered, 0.0), axis=axis), valid.sum(axis=axis))
    return np.sqrt(variance)


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
                        comparison_batch=4096, include_full=False):
    """Return (voxels, videos) correlations; remove video from BOTH RDMs.

    On CUDA, each comparison of RDM entries is computed once and contributes
    to every omission retaining both entries. This avoids repeating Kendall's
    quadratic work forty times. Integer-valued float32 sums are exact for these
    RDM sizes; the tau-b division is float64. Call with bounded voxel batches.
    """
    pairs, stimuli = validate_pairs(pairs)
    keep = np.array([[s not in p for p in pairs] for s in stimuli])
    if include_full:
        # Compute the full-data 11.2 correlation in the same GPU pass.
        keep = np.vstack((np.ones((1, len(pairs)), dtype=bool), keep))
    if method not in ('kendall', 'spearman', 'pearson'):
        raise ValueError('rsa_method must be kendall, spearman, or pearson.')
    if not str(device).startswith('cuda'):
        correlate = dict(kendall=kendalltau, spearman=spearmanr, pearson=pearsonr)[method]
        result = np.full((len(target), len(keep)), np.nan)
        for j, selected in enumerate(keep):
            for i, (a, b) in enumerate(zip(target[:, selected], peers[:, selected])):
                valid = np.isfinite(a) & np.isfinite(b)
                a, b = a[valid], b[valid]
                if len(a) >= 2 and np.ptp(a) > 0 and np.ptp(b) > 0:
                    result[i, j] = correlate(a, b)[0]
        return result
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but no Colab GPU is available. Select a GPU runtime.')
    return _correlate_omissions_torch(target, peers, keep, method, device, comparison_batch)


def _correlate_omissions_torch(target, peers, keep, method, device, comparison_batch, return_tensor=False):
    """Pairwise-finite Torch kernel; tie-corrected Kendall and average ranks."""
    import torch
    nstimuli, npairs = keep.shape
    x = torch.as_tensor(target, dtype=torch.float64, device=device)
    y = torch.as_tensor(peers, dtype=torch.float64, device=device)
    selected = torch.as_tensor(keep, device=device)
    valid = torch.isfinite(x) & torch.isfinite(y)
    if method == 'kendall':
        if npairs * (npairs - 1) // 2 >= 2 ** 24:
            raise ValueError('RDM too large for exact float32 Kendall accumulators.')
        if comparison_batch < 1:
            raise ValueError('comparison_batch must be positive.')
        indices = torch.triu_indices(npairs, npairs, 1, device=device)
        stats = torch.zeros((3 * len(x), nstimuli), dtype=torch.float32, device=device)
        for start in range(0, indices.shape[1], comparison_batch):
            a, b = indices[:, start:start + comparison_batch]
            common = valid[:, a] & valid[:, b]
            sx = torch.where(common, torch.sign(x[:, a] - x[:, b]), 0.0).float()
            sy = torch.where(common, torch.sign(y[:, a] - y[:, b]), 0.0).float()
            eligible = (selected[:, a] & selected[:, b]).T.float()
            stats += torch.cat((sx * sy, sx.abs(), sy.abs()), dim=0) @ eligible
        numerator, nx, ny = stats.double().chunk(3, dim=0)
        result = numerator / torch.sqrt(nx * ny)
    else:
        def ranks(values, present):
            sorted_values, order = torch.where(present, values, torch.inf).sort(dim=1)
            n = values.shape[1]
            pos = torch.arange(n, device=device).expand_as(order)
            first = torch.ones_like(order, dtype=torch.bool)
            first[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
            last = torch.ones_like(first)
            last[:, :-1] = sorted_values[:, :-1] != sorted_values[:, 1:]
            lo = torch.where(first, pos, 0).cummax(dim=1).values
            hi = torch.where(last, pos, n).flip((1,)).cummin(dim=1).values.flip((1,))
            ranked = torch.empty_like(values).scatter_(1, order, (lo + hi).double() / 2)
            return torch.where(present, ranked, 0.0)
        columns = []
        for mask in selected:
            a, b, present = x[:, mask], y[:, mask], valid[:, mask]
            if method == 'spearman':
                a, b = ranks(a, present), ranks(b, present)
            a, b = torch.where(present, a, 0.0), torch.where(present, b, 0.0)
            count = present.sum(1, keepdim=True)
            a = torch.where(present, a - a.sum(1, keepdim=True) / count.clamp_min(1), 0.0)
            b = torch.where(present, b - b.sum(1, keepdim=True) / count.clamp_min(1), 0.0)
            coefficient = (a * b).sum(1) / torch.sqrt((a * a).sum(1) * (b * b).sum(1))
            columns.append(torch.where(count[:, 0] >= 2, coefficient.clamp(-1, 1), torch.nan))
        result = torch.stack(columns, dim=1)
    return result if return_tensor else result.cpu().numpy()


class MapStore:
    """Read pipeline-relative maps from extracted trees or existing result ZIPs."""
    def __init__(self, roots, index_archives=True):
        self.roots = [Path(p) for p in roots]
        self.members = {}
        for root in self.roots:
            if not index_archives:
                continue
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


def pair_sources(manifest, store, sub, session, run, progress=None):
    base = (f"{manifest['dataset']}/results/RSA/{manifest['model']}/"
            f"{manifest['specie']}-sub-{sub:02d}")
    if run is not None:
        base += f"/ses-{session:02d}_task-{manifest['task']}_run-{run:02d}"
    sources = []
    for index, (a, b) in enumerate(manifest['pairs'], 1):
        if progress is not None:
            progress(index, len(manifest['pairs']), a, b)
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


def _analysis_units(manifest):
    fold = manifest.get('mah_fold', 'stim-wise')
    if manifest['dis_method'] == 'mahalanobis' and fold in ('stim-wise', 'run-wise', 'stim-wise-multiple-folds'):
        return {s: [(None, None)] for s in manifest['participants']}
    if (manifest['dis_method'] == 'correlation' or
            (manifest['dis_method'] == 'mahalanobis' and fold == 'stim-wise-all-runs')):
        return {s: sorted({(int(e['session']), int(e['run_N']))
                           for e in manifest['runs_by_sub'][str(s)]}) for s in manifest['participants']}
    raise ValueError('Unsupported dis_method/mah_fold for step 11.31.')


def _baseline_layout(manifest, output_root):
    """Match rsa_utils._loo_layout's observed step-11.2/11.3 paths."""
    root = (Path(output_root) / manifest['dataset'] / 'results/RSA_same' / manifest['model'] /
            f"class_{manifest['rsa_class'].removeprefix('class_')}" / 'leave-one-out')
    fold = manifest.get('mah_fold', 'stim-wise')
    if manifest['dis_method'] == 'mahalanobis' and fold != 'stim-wise':
        root /= fold
    prefix = f"{manifest['mask_type']}-" if manifest.get('mask_type') else ''
    suffix = f"r-{manifest['radius']}_{manifest['dis_method']}_{manifest['rsa_method']}_task-{manifest['task']}"
    units = _analysis_units(manifest)
    paths = {}
    for sub, entries in units.items():
        for session, run_number in entries:
            subject = f"{manifest['specie']}-sub-{sub:02d}"
            if run_number is not None:
                subject += f"_ses-{session:02d}_task-{manifest['task']}_run-{run_number:02d}"
            paths[sub, session, run_number] = root / subject / f'{prefix}{suffix}.nii.gz'
    group = root / f"{prefix}{manifest['specie']}-{suffix}"
    return units, paths, group


def baseline_output_paths(manifest, output_root):
    """Explicit archive inventory, excluding unrelated pre-existing outputs."""
    _, paths, group = _baseline_layout(manifest, output_root)
    outputs = []
    for path in paths.values():
        outputs.extend((path, Path(str(path).replace('.nii.gz', '.json'))))
    outputs.extend(Path(str(group) + suffix) for suffix in ('_mean.nii.gz', '_std.nii.gz', '_mean.json'))
    return outputs


def participant_output_paths(manifest, output_root):
    """Exact participant omission map/JSON inventory for the requested cohort."""
    _, stimuli = validate_pairs(manifest['pairs'])
    units = _analysis_units(manifest)
    folder = (Path(output_root) / manifest['dataset'] / 'results/RSA_same' / manifest['model'] /
              f"class_{manifest['rsa_class'].removeprefix('class_')}" / 'leave-one-out')
    fold = manifest.get('mah_fold', 'stim-wise')
    if manifest['dis_method'] == 'mahalanobis' and fold != 'stim-wise':
        folder /= fold
    prefix = f"{manifest['mask_type']}-" if manifest.get('mask_type') else ''
    outputs = []
    for sub, entries in units.items():
        for run in sorted({r for _, r in entries}, key=lambda r: -1 if r is None else r):
            run_folder = folder / 'by_stim' / (f'run-{run:02d}' if run is not None else 'pooled')
            for stimulus in stimuli:
                path = run_folder / f"{manifest['specie']}-sub-{sub:02d}" / f"{prefix}r-{manifest['radius']}_{manifest['rsa_method']}_{stimulus}.nii.gz"
                outputs.extend((path, Path(str(path).replace('.nii.gz', '.json'))))
    return outputs


def _write_masked(values, path, ref, inside):
    path.parent.mkdir(parents=True, exist_ok=True)
    volume = np.zeros(ref.shape, dtype=np.float64)
    volume[inside] = values
    partial = path.with_name(path.name.replace('.nii.gz', '.part.nii.gz'))
    nib.save(nib.Nifti1Image(volume, ref.affine), str(partial))
    os.replace(partial, path)


def _average_baseline(manifest, output_root, ref, inside, replace_file, verbose):
    """Step 11.3: average sessions, then runs, then participants equally."""
    import warnings
    units, paths, group = _baseline_layout(manifest, output_root)
    included, missing, inputs, inventory = [], [], [], []
    for sub, entries in units.items():
        if not entries or any(not paths[sub, s, r].is_file() for s, r in entries):
            missing.append(sub)
            continue
        included.append(sub)
        for session, run_number in entries:
            path = paths[sub, session, run_number]
            stat = path.stat()
            inputs.append(str(path))
            inventory.append((str(path), stat.st_size, stat.st_mtime_ns))
    if missing:
        warnings.warn(f'Step 11.3: excluding participants with incomplete step-11.2 maps: {missing}.')
    available = len(included) / len(manifest['participants'])
    if len(included) < 2 or available < manifest.get('min_percentage_available', 0):
        raise ValueError(f'Step 11.3: insufficient complete participants ({len(included)}/{len(units)}).')
    signature = hashlib.sha256(json.dumps(dict(manifest=manifest, inputs=inventory, version=VERSION),
                                         sort_keys=True).encode()).hexdigest()
    mean, std, log = (Path(str(group) + suffix) for suffix in ('_mean.nii.gz', '_std.nii.gz', '_mean.json'))
    if not replace_file and all(p.is_file() for p in (mean, std, log)):
        try:
            if json.loads(log.read_text()).get('signature') == signature:
                if verbose:
                    print('Step 11.3: existing mean/std reused.', flush=True)
                return
        except (OSError, ValueError):
            pass
    if verbose:
        print('Step 11.3: averaging sessions, runs and participants with equal participant weight.', flush=True)
    participant_maps = []
    for sub in included:
        by_run = {}
        for session, run_number in units[sub]:
            path = paths[sub, session, run_number]
            image = nib.load(str(path))
            check_grid(image, ref, path)
            by_run.setdefault(run_number, []).append(np.asarray(image.dataobj)[inside])
        participant_maps.append(finite_mean([finite_mean(maps, axis=0) for maps in by_run.values()], axis=0))
    _write_masked(finite_mean(participant_maps, axis=0), mean, ref, inside)
    _write_masked(finite_std(participant_maps, axis=0), std, ref, inside)
    _atomic_json(log, dict(signature=signature, version=VERSION, missing_data_policy=MISSING_DATA_POLICY, steps=[11.2, 11.3],
                          participants_requested=manifest['participants'], participants_included=included,
                          participants_missing=missing, perc_available=available, input_files=inputs,
                          dis_method=manifest['dis_method'], mah_fold=manifest.get('mah_fold', 'stim-wise'),
                          rsa_method=manifest['rsa_method'], rsa_class=manifest['rsa_class'], rsa_model=None,
                          weighting='sessions within run, runs within participant, then participants equally',
                          output_mean_file=str(mean), output_std_file=str(std)))
    if verbose:
        print(f'Step 11.3 saved: {mean} and {std}', flush=True)


def run(manifest, roots, output_root, work_root, *, device='cpu', voxel_batch=128,
        comparison_batch=4096, replace_file=False, verbose=True, include_baseline=False,
        save_participant_maps=False, load_workers=1):
    """Write one run-specific group map for every omitted stimulus.

    Each session is correlated against equally weighted peer participant means;
    session correlations are averaged within participant, then participants
    equally. Per-run inputs stay separate; pooled Mahalanobis is processed once.
    Missing pair maps are fatal, absent runs exclude participants with a warning.
    include_baseline also writes full-data 11.2 participant maps and the 11.3
    group mean/std using the same loaded RDMs. save_participant_maps additionally
    writes session-averaged participant omission maps under each run/pooled
    folder in <specie>-sub-XX/. Disk-backed arrays bound host RAM.
    """
    import warnings
    pairs, stimuli = validate_pairs(manifest['pairs'])
    participants = list(manifest['participants'])
    if len(participants) < 2 or len(set(participants)) != len(participants):
        raise ValueError('At least two unique participants are required.')
    if load_workers < 1:
        raise ValueError('load_workers must be positive.')
    if voxel_batch < 1 or not 0 <= manifest.get('min_percentage_available', 0) <= 1:
        raise ValueError('Invalid voxel_batch or min_percentage_available.')
    fold = manifest.get('mah_fold', 'stim-wise')
    units = _analysis_units(manifest)
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
    baseline_paths = _baseline_layout(manifest, output_root)[1] if include_baseline else {}
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
            manifest=manifest, identity=identity, version=VERSION, missing_data_policy=MISSING_DATA_POLICY,
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
        participant_pending = {}
        if save_participant_maps:
            for sub in cohort:
                for j, group_path in enumerate(outputs):
                    path = folder / f"{manifest['specie']}-sub-{sub:02d}" / group_path.name
                    try:
                        complete = (path.is_file() and
                                    json.loads(path.with_suffix('').with_suffix('.json').read_text()).get('signature') == signature)
                    except (OSError, ValueError):
                        complete = False
                    if replace_file or not complete:
                        participant_pending[sub, j] = path
        paths.extend(str(p) for p in outputs)
        baseline_pending = []
        for sub, session in (sources if include_baseline else []):
            key = sub, session, run_number
            path = baseline_paths[key]
            try:
                complete = (path.is_file() and
                            json.loads(Path(str(path).replace('.nii.gz', '.json')).read_text()).get('signature') == signature)
            except (OSError, ValueError):
                complete = False
            if replace_file or not complete:
                baseline_pending.append(key)
        if not pending and not baseline_pending and not participant_pending:
            if verbose:
                print(f'Run {run_number}: all {len(outputs)} omissions already complete.', flush=True)
            continue
        if verbose:
            print(f'Run {run_number}: {len(cohort)} participants, {len(pairs)} pairs, '
                  f'{len(stimuli)} videos, device={device}. Loading RDMs.', flush=True)
        # Scratch lifetime is one run; remove mmap handles before Windows cleanup.
        with tempfile.TemporaryDirectory(dir=work_root, prefix='rdms-') as scratch:
            with _mapped_rdms(scratch, sources, store, ref, inside, len(pairs), verbose=verbose, load_workers=load_workers) as rdms, ExitStack() as buffers:
                nvox = int(inside.sum())
                group = np.empty((nvox, len(stimuli)), dtype=np.float64)
                std = np.empty_like(group)
                omission_maps = {}
                for sub in sorted({sub for sub, _ in participant_pending}):
                    array = np.lib.format.open_memmap(Path(scratch) / f'omissions-{sub}.npy',
                                                     mode='w+', dtype=np.float64,
                                                     shape=(nvox, len(stimuli)))
                    buffers.callback(array._mmap.close)
                    omission_maps[sub] = array
                baseline_maps = {key: np.empty(nvox, dtype=np.float64) for key in baseline_pending}
                if include_baseline and verbose:
                    print(f'Step 11.2: full-data participant maps, run={run_number}.', flush=True)
                last_progress = time.monotonic()
                for start in range(0, nvox, voxel_batch):
                    stop = min(nvox, start + voxel_batch)
                    means = {s: finite_mean([a[start:stop] for (sub, _), a in rdms.items() if sub == s], axis=0)
                             for s in cohort}
                    total, counts = finite_totals(means.values())
                    participant_maps = []
                    for sub in cohort:
                        own_valid = np.isfinite(means[sub])
                        peers = finite_divide(total - np.where(own_valid, means[sub], 0.0), counts - own_valid)
                        session_maps = []
                        for (s, session), a in rdms.items():
                            if s != sub:
                                continue
                            values = correlate_omissions(a[start:stop], peers, pairs,
                                                         method=manifest['rsa_method'], device=device,
                                                         comparison_batch=comparison_batch,
                                                         include_full=include_baseline)
                            if include_baseline:
                                key = sub, session, run_number
                                if key in baseline_maps:
                                    baseline_maps[key][start:stop] = values[:, 0]
                                values = values[:, 1:]
                            session_maps.append(values)
                        participant_map = finite_mean(session_maps, axis=0)
                        participant_maps.append(participant_map)
                        if sub in omission_maps:
                            omission_maps[sub][start:stop] = participant_map
                    group[start:stop] = finite_mean(participant_maps, axis=0)
                    std[start:stop] = finite_std(participant_maps, axis=0)
                    if verbose and (start == 0 or stop == nvox or time.monotonic() - last_progress >= 15):
                        print(f'Run {run_number}: {stop}/{nvox} voxels.', flush=True)
                        last_progress = time.monotonic()
                for (sub, j), path in participant_pending.items():
                    _write_masked(omission_maps[sub][:, j], path, ref, inside)
                    _atomic_json(Path(str(path).replace('.nii.gz', '.json')), dict(
                        signature=signature, version=VERSION, missing_data_policy=MISSING_DATA_POLICY,
                        step=11.31, participant=sub, stimulus_removed=stimuli[j], run=run_number,
                        reference_participants=[s for s in cohort if s != sub],
                        sessions=[session for session, r in units[sub] if r == run_number],
                        pairs_before=len(pairs), pairs_after=sum(stimuli[j] not in p for p in pairs),
                        dis_method=manifest['dis_method'], mah_fold=fold, task=manifest['task'],
                        rsa_method=manifest['rsa_method'], source_inventory_signature=signature,
                        weighting='sessions within participant; peer participants equally; runs separate'))
                    if verbose:
                        print(f'Saved participant omission: {sub}, {stimuli[j]} -> {path}', flush=True)
                for key, values in baseline_maps.items():
                    path = baseline_paths[key]
                    _write_masked(values, path, ref, inside)
                    _atomic_json(Path(str(path).replace('.nii.gz', '.json')), dict(
                        signature=signature, version=VERSION, missing_data_policy=MISSING_DATA_POLICY, step=11.2,
                        participant=key[0], session=key[1], run=key[2],
                        reference_participants=[s for s in cohort if s != key[0]],
                        dis_method=manifest['dis_method'], mah_fold=fold,
                        rsa_method=manifest['rsa_method'], source_inventory=identity,
                        weighting='peer sessions within participant, then peer participants equally'))
                    if verbose:
                        print(f'Step 11.2 saved: {path}', flush=True)
                for j in pending:
                    path = outputs[j]
                    for values, output in ((group[:, j], path), (std[:, j], Path(str(path).replace('.nii.gz', '_std.nii.gz')))):
                        volume = np.zeros(ref.shape, dtype=np.float64)
                        volume[inside] = values  # No usable data stays NaN; outside mask = 0.
                        partial = output.with_name(output.name.replace('.nii.gz', '.part.nii.gz'))
                        nib.save(nib.Nifti1Image(volume, ref.affine), str(partial))
                        os.replace(partial, output)
                    _atomic_json(Path(str(path).replace('.nii.gz', '.json')), dict(
                        signature=signature, version=VERSION, missing_data_policy=MISSING_DATA_POLICY, stimulus_removed=stimuli[j], run=run_number,
                        participants_requested=participants, participants_included=cohort,
                        participants_missing=absent, pairs_before=len(pairs),
                        pairs_after=sum(stimuli[j] not in p for p in pairs),
                        dis_method=manifest['dis_method'], mah_fold=fold, task=manifest['task'],
                        rsa_method=manifest['rsa_method'],
                        kendall_variant='b' if manifest['rsa_method'] == 'kendall' else None,
                        weighting='sessions within participant, then participants equally; runs separate',
                        statistic='group similarity after omission (not full-minus-omitted difference)',
                        source_inventory=identity, device=str(device)))
                    if verbose:
                        print(f'Saved omission: {stimuli[j]} -> {path}', flush=True)
                # Do not retain mmap slices via a loop variable at cleanup.
                del means, total, participant_maps, session_maps
    if include_baseline:
        _average_baseline(manifest, output_root, ref, inside, replace_file, verbose)
    return paths


@contextmanager
def _mapped_rdms(scratch, sources, store, ref, inside, npairs, verbose=False, load_workers=1):
    """Load independent sessions with bounded I/O threads, preserving map order."""
    from concurrent.futures import ThreadPoolExecutor
    arrays = {}

    def load(item):
        (sub, session), entries = item
        if verbose:
            print(f'Loading participant {sub}, session {session}: {len(entries)} pair maps.', flush=True)
        path = Path(scratch) / f'{sub}-{session}.npy'
        array = np.lib.format.open_memmap(path, mode='w+', dtype=np.float64,
                                          shape=(int(inside.sum()), npairs))
        arrays[sub, session] = array
        invalid_values = 0
        for j, source in enumerate(entries):
            image = store.load(source)
            check_grid(image, ref, source)
            values = np.asarray(image.dataobj)[inside]
            valid = np.isfinite(values)
            invalid_values += int((~valid).sum())
            array[:, j] = np.where(valid, values, np.nan)
            if verbose and (j + 1) % 100 == 0:
                print(f'  Participant {sub}: {j + 1}/{len(entries)} maps loaded.', flush=True)
        array.flush()
        if verbose and invalid_values:
            print(f'  Participant {sub}: ignoring {invalid_values} non-finite RDM entries as missing data.', flush=True)

    try:
        if load_workers == 1:
            for item in sources.items():
                load(item)
        else:
            with ThreadPoolExecutor(max_workers=load_workers) as executor:
                # Consume all results so load/grid failures propagate before use.
                list(executor.map(load, sources.items()))
        yield arrays
    finally:
        for array in arrays.values():
            array._mmap.close()

"""Joint participant-label randomization for searchlight steps 11.4 and 11.5."""
import csv
import hashlib
import itertools
import json
import os
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.stats import rankdata
from stimulus_contribution import finite_mean, finite_std, finite_divide, finite_totals

VERSION = 'joint-loo-1'


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def publish(path, writer, suffix=''):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=suffix)
    os.close(fd)
    try:
        writer(Path(tmp))
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def write_json(path, value):
    publish(path, lambda p: p.write_text(json.dumps(value, separators=(',', ':')), encoding='utf8'))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def pair_table(path):
    with open(path, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'cat1', 'cat2'} <= set(reader.fieldnames or []):
            raise ValueError('Class CSV requires cat1 and cat2 columns.')
        selected = [(r['cat1'], r['cat2']) for r in reader]
    labels = sorted(set(itertools.chain.from_iterable(selected)))
    pairs = list(itertools.combinations(labels, 2))
    indices = [pairs.index(tuple(sorted(p))) for p in selected]
    if len(indices) < 2 or len(set(indices)) != len(indices):
        raise ValueError(f'Expected unique non-diagonal condition pairs in {path}')
    # Match step 11.2 file preference for selected pairs.
    for index, pair in zip(indices, selected):
        pairs[index] = pair
    return labels, pairs, indices


def pair_path(root, model, specie, sub, session, run, task, radius, dis_method, pair):
    folder = Path(root) / 'results' / 'RSA' / model / f'{specie}-sub-{sub:02d}'
    if run is not None:
        folder /= f'ses-{session:02d}_task-{task}_run-{run:02d}'
    for a, b in (pair, pair[::-1]):
        path = folder / f'r-{radius}_{dis_method}_{a}_{b}.nii.gz'
        if path.is_file():
            return path
    raise FileNotFoundError(path)


def check_grid(img, ref, name):
    if img.shape != ref.shape or not np.allclose(img.affine, ref.affine, rtol=0, atol=1e-4):
        raise ValueError(f'Searchlight grid mismatch: {name}')


class Analysis:
    def __init__(self, datafolder, dataset, session_and_run_all_dict, participants, specie,
                 mask, model, task, radius, rsa_class, dis_method='mahalanobis',
                 rsa_method='kendall', mah_fold='stim-wise', mask_type=None):
        root = Path(datafolder) / dataset
        self.subjects, self.units, self.paths, group = loo_layout(
            datafolder, dataset, session_and_run_all_dict, sorted(participants), specie, model,
            task, radius, rsa_class, dis_method, rsa_method, None, mah_fold, mask_type)
        self.targets = self.subjects
        self.root = root
        self.group = Path(group)
        _, _, _, rnd = loo_layout(datafolder, dataset, session_and_run_all_dict, self.subjects,
                                  specie, model, task, radius, rsa_class, dis_method, rsa_method,
                                  None, mah_fold, mask_type, rnd=True)
        self.rnd = Path(rnd)
        self.plan_path = self.rnd.with_name(self.rnd.name + '_joint.json')
        self.ref = nib.load(str(mask))
        mask_values = self.ref.get_fdata()
        self.inside = np.isfinite(mask_values) & (mask_values != 0)
        if mask_values.ndim != 3 or not self.inside.any():
            raise ValueError('Expected a nonempty 3D mask.')
        self.method = rsa_method
        self.labels, self.pairs, self.selected = pair_table(
            root / 'rsa_models/by_class' / f'class_{rsa_class.removeprefix("class_")}.csv')
        self.files = {}
        hashes = {}
        for sub in self.subjects:
            if not self.units[sub]:
                raise ValueError(f'Missing input units for participant {sub}')
            for session, run in self.units[sub]:
                paths = [pair_path(root, model, specie, sub, session, run, task, radius,
                                   dis_method, pair) for pair in self.pairs]
                self.files[sub, session, run] = paths
                for p in paths:
                    check_grid(nib.load(str(p)), self.ref, p)
                    hashes[p.relative_to(root).as_posix()] = digest(p)
        self.metadata = dict(algorithm=VERSION, participants=self.subjects, labels=self.labels,
                             pairs=self.pairs, selected=self.selected, group=self.group.name,
                             input_fingerprint=signature([hashes, digest(mask)]))
        self.bank = None

    def mappings(self, reps, seed, prepare=False, replace=False):
        if reps < 1 or seed < 0:
            raise ValueError('Expected positive reps_group and nonnegative loo_seed.')
        info = dict(self.metadata, reps_group=reps, seed=seed)
        generation = signature(info)
        if prepare:
            mappings = np.empty((reps, len(self.subjects), len(self.labels)), dtype=int)
            for col, sub in enumerate(self.subjects):
                rng = np.random.default_rng(np.random.SeedSequence([seed, int(sub)]))
                for row in range(reps):
                    mappings[row, col] = rng.permutation(len(self.labels))
            record = dict(info, generation=generation, mappings=mappings.tolist())
            if self.plan_path.exists() and not replace:
                if read_json(self.plan_path) != json.loads(json.dumps(record)):
                    raise ValueError('Different joint plan exists; use --replace_rnd_files.')
            else:
                write_json(self.plan_path, record)
        record = read_json(self.plan_path)
        if record['generation'] != generation:
            raise ValueError('Inputs, cohort, or seed changed; rerun step 11.4.')
        self.generation = generation
        return np.asarray(record['mappings'])

    def load_bank(self):
        if self.bank is None:
            self.bank = {}
            for key, paths in self.files.items():
                self.bank[key] = np.stack([nib.load(str(p)).get_fdata()[self.inside]
                                           for p in paths], axis=-1)

    def indices(self, mapping):
        pairs = list(itertools.combinations(range(len(self.labels)), 2))
        lookup = {pair: i for i, pair in enumerate(pairs)}
        selected = [pairs[i] for i in self.selected]
        return np.asarray([[lookup[tuple(sorted((perm[a], perm[b])))] for a, b in selected]
                           for perm in mapping], dtype=np.int64)

    def pooled_torch(self, indices, device, observed=False, voxel_batch=1024):
        import torch
        from stimulus_contribution import _correlate_omissions_torch
        # Keep the masked source RDMs on-device across successive draws.
        if getattr(self, '_device', None) != device:
            self._tensor = torch.stack([torch.as_tensor(self.bank[sub, None, None],
                                        device=device) for sub in self.subjects])
            self._device = device
        idx = torch.as_tensor(indices, dtype=torch.long, device=device)
        n = self._tensor.shape[1]
        moments = torch.empty((2, n), device=device, dtype=torch.float64)
        individual = torch.empty((len(idx), n), device=device, dtype=torch.float64) if observed else None
        keep = np.ones((1, idx.shape[1]), dtype=bool)
        for start in range(0, n, voxel_batch):
            sl = slice(start, min(start + voxel_batch, n))
            x = self._tensor[:, sl].gather(2, idx[:, None].expand(-1, sl.stop-start, -1))
            valid = torch.isfinite(x)
            filled = torch.where(valid, x, 0.)
            peer = (filled.sum(0) - filled) / (valid.sum(0) - valid.long())
            scores = _correlate_omissions_torch(x.flatten(0, 1), peer.flatten(0, 1), keep,
                                                self.method, device, 128, return_tensor=True)
            scores = scores.reshape(x.shape[:2])
            grouped = scores[[self.subjects.index(s) for s in self.targets]]
            mean = grouped.nanmean(0)
            moments[0, sl] = mean
            moments[1, sl] = ((grouped - mean).square().nanmean(0)).sqrt()
            if observed:
                individual[:, sl] = scores
        rows = {} if not observed else dict(zip(self.bank, individual.cpu().numpy()))
        return moments.cpu().numpy(), rows

    def calculate(self, mapping, device='cpu', voxel_batch=512, observed=False):
        self.load_bank()
        idx = self.indices(mapping)
        n = int(self.inside.sum())
        outputs = np.empty((2, n))
        individual = {k: np.empty(n) for k in self.bank} if observed else {}
        pooled = all(units == [(None, None)] for units in self.units.values())
        if pooled and device.startswith('cuda'):
            return self.pooled_torch(idx, device, observed, voxel_batch)
        for start in range(0, n, voxel_batch):
            sl = slice(start, min(start + voxel_batch, n))
            if pooled:
                x = np.stack([self.bank[sub, None, None][sl][:, cols]
                              for sub, cols in zip(self.subjects, idx)])
                total, count = finite_totals(x)
                valid = np.isfinite(x)
                peer = finite_divide(total - np.where(valid, x, 0), count - valid)
                values = correlation(x, peer, self.method)
                if observed:
                    for sub, row in zip(self.subjects, values):
                        individual[sub, None, None][sl] = row
            else:
                per_sub = {s: [] for s in self.subjects}
                for run in sorted({r for _, _, r in self.bank}):
                    rows = {sub: [(key, self.bank[key][sl][:, cols]) for key in self.bank
                                  if key[0] == sub and key[2] == run]
                            for sub, cols in zip(self.subjects, idx)}
                    means = {s: finite_mean([v for _, v in vs]) for s, vs in rows.items() if vs}
                    if len(means) < 2:
                        raise ValueError(f'Run {run} needs at least two participants.')
                    total, count = finite_totals(means.values())
                    for s, own in means.items():
                        valid = np.isfinite(own)
                        peer = finite_divide(total - np.where(valid, own, 0), count - valid)
                        scores = []
                        for key, target in rows[s]:
                            score = correlation(target, peer, self.method)
                            scores.append(score)
                            if observed:
                                individual[key][sl] = score
                        per_sub[s].append(finite_mean(scores))
                values = [finite_mean(per_sub[s]) for s in self.subjects]
            grouped = [values[self.subjects.index(s)] for s in self.targets]
            outputs[:, sl] = finite_mean(grouped), finite_std(grouped)
        return outputs, individual

    def save(self, path, values):
        data = np.zeros(self.ref.shape, dtype=np.float64)
        data[self.inside] = values
        publish(path, lambda p: nib.save(nib.Nifti1Image(data, self.ref.affine), p), '.nii.gz')

    def draw_paths(self, draw):
        stem = self.rnd.parent / 'mean' / self.rnd.name
        return [Path(f'{stem}_{kind}_{draw:05d}.nii.gz') for kind in ('mean', 'std')] + [
            Path(f'{stem}_mean_{draw:05d}.json')]

    def draw(self, index, mapping, device='cpu', replace=False):
        mean, std, log = self.draw_paths(index)
        receipt = dict(algorithm=VERSION, generation=self.generation, group_permutation=index,
                       participants_included=self.targets, mapping_digest=signature(mapping.tolist()))
        if log.exists() and not replace:
            saved = read_json(log)
            if any(saved.get(k) != v for k, v in receipt.items()):
                raise ValueError(f'Incompatible draw {index}; use --replace_rnd_files.')
            if (mean.exists() and std.exists() and digest(mean) == saved.get('mean_sha256')
                    and digest(std) == saved.get('std_sha256')):
                return
        values, _ = self.calculate(mapping, device)
        self.save(mean, values[0])
        self.save(std, values[1])
        write_json(log, dict(receipt, mean_sha256=digest(mean), std_sha256=digest(std)))

    def observed(self, device):
        identity = np.tile(np.arange(len(self.labels)), (len(self.subjects), 1))
        values, individual = self.calculate(identity, device, observed=True)
        for key, score in individual.items():
            self.save(self.paths[key], score)
        for kind, data in zip(('mean', 'std'), values):
            self.save(f'{self.group}_{kind}.nii.gz', data)
        write_json(f'{self.group}_mean.json', dict(self.metadata, participants_requested=self.subjects,
                   participants_included=self.subjects, participants_missing=[], perc_available=1.0,
                   mean_sha256=digest(f'{self.group}_mean.nii.gz'),
                   input_files=[Path(p).relative_to(self.root).as_posix() for p in self.paths.values()]))


def correlation(x, y, method):
    valid = np.isfinite(x) & np.isfinite(y)
    with np.errstate(invalid='ignore', divide='ignore'):
        if method == 'kendall':
            # Limit temporary storage for classes with many RDM entries.
            num = np.zeros(x.shape[:-1])
            nx, ny = np.zeros_like(num), np.zeros_like(num)
            for i in range(x.shape[-1] - 1):
                j = slice(i + 1, None)
                ok = valid[..., i, None] & valid[..., j]
                a = np.where(ok, np.sign(x[..., i, None] - x[..., j]), 0)
                b = np.where(ok, np.sign(y[..., i, None] - y[..., j]), 0)
                num += (a*b).sum(-1)
                nx += np.abs(a).sum(-1)
                ny += np.abs(b).sum(-1)
            return num / np.sqrt(nx*ny)
        if method == 'spearman':
            x, y = [rankdata(np.where(valid, t, np.nan), axis=-1, nan_policy='omit') for t in (x, y)]
        def center(t):
            return np.where(valid, t - finite_mean(np.where(valid, t, np.nan), -1, True), 0)
        a, b = center(x), center(y)
        return np.clip((a*b).sum(-1) / np.sqrt((a*a).sum(-1)*(b*b).sum(-1)), -1, 1)


def validate_ensemble(stem, reps, observed=None):
    stem = Path(stem)
    plan = read_json(stem.with_name(stem.name + '_joint.json'))
    if (plan['algorithm'] != VERSION or plan['reps_group'] != reps or len(plan['mappings']) != reps
            or signature({k: v for k, v in plan.items() if k not in ('generation', 'mappings')}) != plan['generation']):
        raise ValueError('Step 11.6 requires the complete matching joint permutation ensemble.')
    previous_cohort = None
    for i, mapping in enumerate(plan['mappings']):
        prefix = stem.parent / 'mean' / stem.name
        record = read_json(f'{prefix}_mean_{i:05d}.json')
        cohort = record.get('participants_included')
        if previous_cohort is not None and cohort != previous_cohort:
            raise ValueError(f'Mixed target cohorts at draw {i}.')
        previous_cohort = cohort
        if (record.get('generation') != plan['generation'] or record.get('group_permutation') != i
                or record.get('mapping_digest') != signature(mapping)
                or digest(f'{prefix}_mean_{i:05d}.nii.gz') != record.get('mean_sha256')
                or digest(f'{prefix}_std_{i:05d}.nii.gz') != record.get('std_sha256')):
            raise ValueError(f'Incomplete or mixed joint permutation ensemble at draw {i}.')

    if observed is not None:
        info = read_json(f'{observed}_mean.json')
        if sorted(info['participants_included']) != previous_cohort:
            raise ValueError('Observed and permuted target cohorts differ.')
        if 'input_fingerprint' in info and (info['input_fingerprint'] != plan['input_fingerprint']
                or digest(f'{observed}_mean.nii.gz') != info.get('mean_sha256')):
            raise ValueError('Observed and permuted inputs differ. Import the matching observed archive.')


def loo_layout(datafolder, dataset, session_and_run_all_dict, participants,
                specie, model, task, radius, rsa_class, dis_method, rsa_method,
                rsa_model, mah_fold, mask_type, rnd=False):
    """Shared leave-one-out paths for writers, readers, and resume checks."""
    participants = list(participants)
    if len(participants) < 2 or len(set(participants)) != len(participants):
        raise ValueError('Leave-one-out similarity requires at least two unique participants.')
    if dis_method not in ('mahalanobis', 'correlation'):
        raise ValueError('Steps 11.2/11.3 support mahalanobis and correlation.')
    if dis_method == 'mahalanobis' and mah_fold not in ('stim-wise', 'run-wise'):
        raise ValueError('Steps 11.2/11.3 support Mahalanobis stim-wise and run-wise.')
    if rsa_method not in ('kendall', 'spearman', 'pearson'):
        raise ValueError('rsa_method must be kendall, spearman, or pearson.')
    models = Path(datafolder) / dataset / 'rsa_models'
    if rsa_class is None and (not rsa_model or not ((models / f'{rsa_model}.csv').is_file()
            or any(models.glob(f'{rsa_model}-run-*.csv')))):
        raise ValueError('Steps 11.2/11.3 require rsa_class or an existing rsa_model to select pairs.')
    selection = (f'class_{str(rsa_class).removeprefix("class_")}'
                 if rsa_class is not None else str(rsa_model))
    root = os.path.join(datafolder, dataset, 'results',
                        'RSA_same_rnd' if rnd else 'RSA_same', model,
                        selection, 'leave-one-out')
    if dis_method == 'mahalanobis' and mah_fold != 'stim-wise':
        root = os.path.join(root, mah_fold)
    prefix = f'{mask_type}-' if mask_type else ''
    stem = f'{prefix}r-{radius}_{dis_method}_{rsa_method}'
    if not rnd:
        stem += f'_task-{task}'
    units = {}
    paths = {}
    for sub in participants:
        entries = ([(None, None)] if dis_method == 'mahalanobis' else
                   sorted({(int(e['session']), int(e['run_N']))
                           for e in session_and_run_all_dict[sub]}))
        units[sub] = entries
        for session, run in entries:
            subject = f'{specie}-sub-{sub:02d}'
            if run is not None:
                subject += f'_ses-{session:02d}_task-{task}_run-{run:02d}'
            paths[sub, session, run] = os.path.join(root, subject, stem + '.nii.gz')
    group_stem = f'{prefix}{specie}-r-{radius}_{dis_method}_{rsa_method}_task-{task}'
    group = os.path.join(root, group_stem)
    return participants, units, paths, group



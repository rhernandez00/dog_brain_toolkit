"""Step-1 dissimilarity maps after omitting one beta observation at a time."""
import os
import csv
import json
import warnings
from itertools import combinations

import nibabel as nib
import numpy as np

import rsa_utils as rsa


def _read_pairs(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'cat1', 'cat2'}.issubset(reader.fieldnames or []):
            raise ValueError('Class CSV requires cat1 and cat2 columns.')
        pairs = [(row['cat1'].strip(), row['cat2'].strip()) for row in reader]
    if (not pairs or any(not a or not b or a == b for a, b in pairs)
            or len({frozenset(pair) for pair in pairs}) != len(pairs)):
        raise ValueError('Class CSV requires unique, non-diagonal comparison pairs.')
    return pairs


def _stimwise_category(stimulus, dataset):
    """Use calculate_pairwise_similarity_maps2's Mahalanobis label rule."""
    if dataset == 'EmoC':
        return stimulus[:-1]
    if dataset == 'EmoB' and '-' in stimulus:
        return stimulus.split('-')[0]
    raise ValueError(f'Cannot determine stim-wise category for {dataset}: {stimulus}')


def _resolve_stimwise_pairs(pairs, stim_types, dataset):
    """Project beta-level class selections onto the config's category RDM.

    Already-collapsed category labels are accepted. Drop within-category pairs
    and deduplicate unordered pairs, rather than weighting exemplar counts.
    """
    pairs = list(pairs)
    mapping = {stimulus: _stimwise_category(stimulus, dataset) for stimulus in stim_types}
    categories = set(mapping.values())
    unknown = {label for pair in pairs for label in pair} - categories - set(mapping)
    if unknown:
        raise ValueError(
            f'Class comparisons contain unavailable labels: {sorted(unknown)}. '
            f'Categories derived from the GLM stimulus list: {sorted(categories)}.')
    resolved, seen = [], set()
    for a, b in pairs:
        a = a if a in categories else mapping[a]
        b = b if b in categories else mapping[b]
        key = frozenset((a, b))
        if a != b and key not in seen:
            resolved.append((a, b))
            seen.add(key)
    return resolved


def _observation_records(units, stim_types, dataset, dis_method, mah_fold, model_dict):
    """Return (unit index, beta condition, distance label, CV partition)."""
    stim_indices = {stim: index for index, stim in enumerate(stim_types)}
    records = []
    for unit_index, entry in enumerate(units):
        if dis_method == 'mahalanobis' and mah_fold == 'stim-wise-all-runs':
            if dataset != 'EmoC' or not isinstance(model_dict, dict):
                raise ValueError('stim-wise-all-runs requires EmoC model_dict metadata.')
            _, run_records, _ = rsa._get_emoc_within_run_class_folds(
                model_dict.get(f"run{int(entry['run_N']):02d}"), stim_types)
            records.extend((unit_index, stim, label, fold)
                           for stim, label, fold in run_records)
            continue
        if dis_method == 'mahalanobis' and mah_fold == 'stim-wise-multiple-folds':
            if dataset != 'EmoC':
                raise ValueError('stim-wise-multiple-folds requires EmoC.')
            categories, partitions = rsa._get_emoc_multiple_fold_stim_files(units, model_dict)
            run_metadata = model_dict[f"run{int(entry['run_N']):02d}"]
            for stim, metadata in run_metadata.items():
                if stim not in stim_indices:
                    raise ValueError(f'Metadata condition {stim!r} is absent from stim_types.')
                if metadata['stim_file'] in categories and metadata['partition'] in partitions:
                    records.append((unit_index, stim, metadata['stim_file'], metadata['partition']))
            continue
        for stim in stim_types:
            label = stim
            if dis_method == 'mahalanobis' and mah_fold == 'stim-wise':
                label = _stimwise_category(stim, dataset)
            records.append((unit_index, stim, label, unit_index))
    return records


def _omission_directory(root, units, unit_index, stimulus):
    entry = units[unit_index]
    if sum(int(e['run_N']) == int(entry['run_N']) for e in units) > 1:
        root = os.path.join(root, f"ses-{int(entry['session']):02d}")
    return os.path.join(root, f"run-{int(entry['run_N'])}", stimulus)


def calculate_leave_one_stimulus_out_maps(
        datafolder, dataset, sub_N, session_and_run_dict, specie, model,
        stim_types, mask, task, radius, dis_method, rsa_class,
        mah_fold='stim-wise', model_dict=None, replace_file=False,
        verbose=False, shuffle_runs=False, sigma=None, shrinkage='ledoitwolf'):
    """Recompute a participant's pair maps, excluding exactly one run beta.

    The omitted stimulus is the GLM condition key (for example ``DogA1``),
    not its collapsed class or every occurrence of that key in other runs.
    For Mahalanobis stim-wise, all category pairs come from the GLM stimulus
    list, as in ordinary step 1; rsa_class only names the output directory.
    Other folds retain their class-CSV selection. All remaining observations
    enter covariance estimation. Betas are loaded once using their ORIGINAL
    config indices, including when falling back to FEAT pe files.

    Pooled Mahalanobis folds write directly inside each omission directory.
    Per-run methods write their sets in ``ses-XX_task-..._run-XX`` subfolders.
    A repeated run number adds ``ses-XX`` before ``run-N`` to avoid collisions.
    """
    methods = {'mahalanobis', 'correlation', 'pearson', 'kendall', 'euclidean'}
    folds = {'stim-wise', 'stim-wise-multiple-folds', 'stim-wise-all-runs', 'run-wise'}
    if dis_method not in methods:
        raise ValueError(f'Unsupported leave-one-stimulus-out method: {dis_method}')
    if dis_method == 'mahalanobis' and mah_fold not in folds:
        raise ValueError(f'Unsupported leave-one-stimulus-out Mahalanobis fold: {mah_fold}')
    if not rsa_class:
        raise ValueError('Leave-one-stimulus-out maps require rsa_class.')
    if not session_and_run_dict:
        raise ValueError(f'No runs for {specie}-sub-{sub_N:02d}.')
    units = list(session_and_run_dict)
    identities = [(int(e['session']), int(e['run_N'])) for e in units]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate session/run entries would overwrite omission maps.')
    if not stim_types or len(stim_types) != len(set(stim_types)):
        raise ValueError('stim_types must be nonempty and unique.')
    if not np.isfinite(radius) or radius < 0:
        raise ValueError('Searchlight radius must be finite and nonnegative.')
    if shuffle_runs:
        np.random.shuffle(units)
    class_name = str(rsa_class).removeprefix('class_')
    per_run = dis_method != 'mahalanobis' or mah_fold == 'stim-wise-all-runs'
    stim_indices = {stim: index for index, stim in enumerate(stim_types)}
    records = _observation_records(units, stim_types, dataset, dis_method, mah_fold, model_dict)

    labels_available = {record[2] for record in records}
    if dis_method == 'mahalanobis' and mah_fold == 'stim-wise':
        pairs = list(combinations(sorted(labels_available), 2))
        if not pairs:
            raise ValueError('Mahalanobis stim-wise maps require at least two GLM categories.')
    else:
        pairs = _read_pairs(os.path.join(datafolder, dataset, 'rsa_models', 'by_class',
                                       f'class_{class_name}.csv'))
    unknown = {label for pair in pairs for label in pair} - labels_available
    if unknown:
        raise ValueError(f'Class comparisons contain unavailable labels: {sorted(unknown)}')
    root = os.path.join(datafolder, dataset, 'results', 'RSA_same', model,
                        f'class_{class_name}', 'leave-one-out', 'by_stim',
                        f'{specie}-sub-{sub_N:02d}')
    # Alternate folds must not reuse outputs from the default stim-wise analysis.
    if dis_method == 'mahalanobis' and mah_fold != 'stim-wise':
        root = os.path.join(root, mah_fold)
    jobs = []
    for omitted in records:
        output_dir = _omission_directory(root, units, omitted[0], omitted[1])
        groups = range(len(units)) if per_run else [None]
        for group in groups:
            active = [record for record in records if record != omitted
                      and (group is None or record[0] == group)]
            baseline_labels = {r[2] for r in records if group is None or r[0] == group}
            group_dir = output_dir
            if group is not None:
                analyzed = units[group]
                group_dir = os.path.join(group_dir,
                    f"ses-{int(analyzed['session']):02d}_task-{task}_run-{int(analyzed['run_N']):02d}")
            missing_pairs = {}
            for a, b in pairs:
                if a not in baseline_labels or b not in baseline_labels:
                    continue
                path = os.path.join(group_dir, f'r-{radius}_{dis_method}_{a}_{b}.nii.gz')
                if replace_file or not os.path.exists(path):
                    missing_pairs[(a, b)] = path
            if missing_pairs:
                jobs.append((omitted, active, missing_pairs))
    if not jobs:
        if verbose:
            print(f'All leave-one-stimulus-out maps exist for {specie}-sub-{sub_N:02d}.')
        return True

    mask_img = nib.load(mask)
    mask_data = mask_img.get_fdata().astype(bool)
    maps = {}
    for unit_index, stim, _, _ in records:
        entry = units[unit_index]
        path = rsa.resolve_beta_map(datafolder, dataset, model, specie, sub_N,
            f"{int(entry['session']):02d}", int(entry['run_N']), task,
            stim, stim_indices[stim])
        image = nib.load(path)
        rsa.check_same_space((f'mask {mask}', mask_img), [(path, image)],
                             context='step 1 leave-one-stimulus-out')
        maps[(unit_index, stim)] = image.get_fdata()
    rad = int(np.floor(radius))
    ranges = [np.arange(-rad, rad + 1) for _ in mask_data.shape]
    grid = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1).reshape(-1, mask_data.ndim)
    offsets = grid[np.sum(grid.astype(float) ** 2, axis=1) <= radius ** 2 + 1e-12]
    for job_index, (omitted, active, output_paths) in enumerate(jobs, 1):
        entry = units[omitted[0]]
        print(f'Leave-one-stimulus-out {job_index}/{len(jobs)}: {specie}-sub-{sub_N:02d}, '
              f"ses-{entry['session']}, run-{entry['run_N']}, omit {omitted[1]}", flush=True)
        if dis_method == 'mahalanobis':
            output_maps = {pair: np.full(mask_data.shape, np.nan) for pair in output_paths}
            labels = np.array([r[2] for r in active])
            partitions = np.array([r[3] for r in active])
            condition_indices = {label: i for i, label in enumerate(np.unique(labels))}
            if len(np.unique(partitions)) >= 2 and len(condition_indices) >= 2:
                for center in np.argwhere(mask_data):
                    neighbors = center + offsets
                    neighbors = neighbors[np.all((neighbors >= 0) &
                        (neighbors < np.asarray(mask_data.shape)), axis=1)]
                    neighbors = neighbors[mask_data[tuple(neighbors.T)]]
                    indices = tuple(neighbors.T)
                    Y = np.vstack([maps[(r[0], r[1])][indices] for r in active])
                    distances = rsa.crossnobis(Y, labels, partitions, sigma=sigma,
                                              shrinkage=shrinkage, allow_missing=True)
                    for (a, b), values in output_maps.items():
                        if a in condition_indices and b in condition_indices:
                            values[tuple(center)] = distances[condition_indices[a], condition_indices[b]]
        else:
            active_maps = {r[2]: maps[(r[0], r[1])] for r in active}
            output_maps = {}
            for pair in output_paths:
                a, b = pair
                values = np.full(mask_data.shape, np.nan)
                if a in active_maps and b in active_maps:
                    if np.all(active_maps[a] == 0) or np.all(active_maps[b] == 0):
                        values = np.zeros(mask_data.shape)
                    else:
                        values = rsa.similarity_searchlight(active_maps[a], active_maps[b],
                            mask_data, radius=radius, dis_method=dis_method)
                output_maps[pair] = values
        for pair, path in output_paths.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            nib.save(nib.Nifti1Image(output_maps[pair], mask_img.affine), path)
    return True


def calculate_omission_cross_participant_similarity(
        datafolder, dataset, session_and_run_all_dict, participants, specie,
        mask, model, task, radius, rsa_class, dis_method='mahalanobis',
        rsa_method='kendall', mah_fold='stim-wise', mask_type=None,
        replace_file=False, verbose=False, target_participants=None,
        stim_types=None, model_dict=None, min_percentage_available=1.0):
    """Step 11.2 on beta-omission maps, averaged equally across participants.

    For each omitted run/condition, correlate each target RDM against the mean
    RDM of OTHER participants with that same omission. Average repeated sessions
    within runs, then analyzed runs within participants, then participants.
    All class-CSV comparisons remain present: the omission was applied to the
    beta observations in step 1, rather than to the RDM's category pairs.
    """
    from scipy.stats import kendalltau, spearmanr, pearsonr
    from stimulus_contribution import finite_mean, finite_divide, finite_totals

    participants = sorted(participants)
    if len(participants) < 2 or len(set(participants)) != len(participants):
        raise ValueError('Leave-one-out similarity requires at least two unique participants.')
    targets = participants if target_participants is None else sorted(target_participants)
    if not targets or len(set(targets)) != len(targets) or set(targets) - set(participants):
        raise ValueError('target_participants must be a nonempty unique subset of the reference cohort.')
    if not rsa_class:
        raise ValueError('Leave-one-stimulus-out similarity requires rsa_class.')
    if dis_method not in {'mahalanobis', 'correlation', 'pearson', 'kendall', 'euclidean'}:
        raise ValueError(f'Unsupported leave-one-stimulus-out method: {dis_method}')
    if dis_method == 'mahalanobis' and mah_fold not in {
            'stim-wise', 'stim-wise-multiple-folds', 'stim-wise-all-runs', 'run-wise'}:
        raise ValueError(f'Unsupported leave-one-stimulus-out Mahalanobis fold: {mah_fold}')
    if rsa_method not in ('kendall', 'spearman', 'pearson'):
        raise ValueError('rsa_method must be kendall, spearman, or pearson.')
    if not 0 <= min_percentage_available <= 1:
        raise ValueError('min_percentage_available must be between 0 and 1.')
    if stim_types is None:
        import yaml
        with open(os.path.join(datafolder, dataset, 'config_files', f'{specie}_{model}.yaml')) as stream:
            config = yaml.safe_load(stream)
        stim_types = config['stim_types']
        if model_dict is None:
            model_dict = config.get('model_dict')
    if not stim_types or len(set(stim_types)) != len(stim_types):
        raise ValueError('stim_types must be nonempty and unique.')
    class_name = str(rsa_class).removeprefix('class_')
    pairs = _read_pairs(os.path.join(datafolder, dataset, 'rsa_models', 'by_class',
                                   f'class_{class_name}.csv'))
    if dis_method == 'mahalanobis' and mah_fold == 'stim-wise':
        pairs = _resolve_stimwise_pairs(pairs, stim_types, dataset)
    if len(pairs) < 2:
        raise ValueError('Cross-participant similarity requires at least two unique pairs.')
    units, records = {}, {}
    for sub in participants:
        units[sub] = sorted(session_and_run_all_dict.get(sub, []),
                            key=lambda entry: (int(entry['session']), int(entry['run_N'])))
        identities = [(int(e['session']), int(e['run_N'])) for e in units[sub]]
        if len(identities) != len(set(identities)):
            raise ValueError(f'Duplicate session/run entries for participant {sub}.')
        records[sub] = _observation_records(units[sub], stim_types, dataset,
                                           dis_method, mah_fold, model_dict)
    unknown = {label for pair in pairs for label in pair} - {
        record[2] for observations in records.values() for record in observations}
    if unknown:
        raise ValueError(f'Class comparisons contain unavailable labels: {sorted(unknown)}')
    omissions = sorted({(int(units[sub][r[0]]['run_N']), r[1])
                        for sub in targets for r in records[sub]})
    if not omissions:
        raise ValueError('No runs found for the selected participants.')
    root = os.path.join(datafolder, dataset, 'results', 'RSA_same', model,
                        f'class_{class_name}', 'leave-one-out', 'by_stim')
    output_root = root
    if dis_method == 'mahalanobis' and mah_fold != 'stim-wise':
        output_root = os.path.join(output_root, mah_fold)
    elif dis_method != 'mahalanobis':
        output_root = os.path.join(output_root, dis_method)
    prefix = f'{rsa.strip_nii_extension(mask_type)}-' if mask_type else ''
    per_run = dis_method != 'mahalanobis' or mah_fold == 'stim-wise-all-runs'
    correlate = dict(kendall=kendalltau, spearman=spearmanr, pearson=pearsonr)[rsa_method]
    ref = None
    outputs = []
    for run, stimulus in omissions:
        path = os.path.join(output_root, f'run-{run:02d}',
                            f'{prefix}{specie}-r-{radius}_{rsa_method}_{stimulus}.nii.gz')
        log_path = path[:-7] + '.json'
        outputs.append(path)
        parameters = dict(
            analysis='step-11.2-beta-omission', participants_requested=participants,
            target_participants=targets, omitted_run=run, omitted_stimulus=stimulus,
            dis_method=dis_method, rsa_method=rsa_method, mah_fold=mah_fold,
            task=task, radius=radius, mask=os.path.abspath(mask),
            pairs=[list(pair) for pair in pairs], min_percentage_available=min_percentage_available,
            sessions={str(sub): [[int(e['session']), int(e['run_N'])] for e in units[sub]]
                      for sub in participants})
        if not replace_file and os.path.isfile(path) and os.path.isfile(log_path):
            try:
                with open(log_path) as stream:
                    previous = json.load(stream)
                if previous.get('parameters') == parameters:
                    if verbose:
                        print(f'Skipping existing omission similarity map: {path}', flush=True)
                    continue
            except (OSError, ValueError):
                pass
        if ref is None:
            ref = nib.load(mask)
            mask_data = ref.get_fdata()
            inside = np.isfinite(mask_data) & (mask_data != 0)
            if mask_data.ndim != 3 or not inside.any():
                raise ValueError('The searchlight mask must be a nonempty 3D mask.')
        print(f'Step 11.2: omitted run-{run:02d} {stimulus}', flush=True)
        rdms, inputs = {}, []
        for sub in participants:
            omitted_units = [r[0] for r in records[sub]
                             if int(units[sub][r[0]]['run_N']) == run and r[1] == stimulus]
            if not omitted_units:
                warnings.warn(f'Run {run}, omit {stimulus}: excluding {specie}-sub-{sub:02d} (no matching beta).')
                continue
            subject_root = os.path.join(root, f'{specie}-sub-{sub:02d}')
            if dis_method == 'mahalanobis' and mah_fold != 'stim-wise':
                subject_root = os.path.join(subject_root, mah_fold)
            by_run = {}
            for omitted_unit in omitted_units:
                omission_dir = _omission_directory(subject_root, units[sub], omitted_unit, stimulus)
                for analyzed_unit in (range(len(units[sub])) if per_run else [None]):
                    folder = omission_dir
                    analyzed_run = None
                    if analyzed_unit is not None:
                        entry = units[sub][analyzed_unit]
                        analyzed_run = int(entry['run_N'])
                        folder = os.path.join(folder,
                            f"ses-{int(entry['session']):02d}_task-{task}_run-{analyzed_run:02d}")
                    available_labels = {r[2] for r in records[sub]
                                        if analyzed_unit is None or r[0] == analyzed_unit}
                    values = np.full((int(inside.sum()), len(pairs)), np.nan)
                    for index, (a, b) in enumerate(pairs):
                        if a not in available_labels or b not in available_labels:
                            continue
                        source = os.path.join(folder, f'r-{radius}_{dis_method}_{a}_{b}.nii.gz')
                        if not os.path.isfile(source):
                            source = os.path.join(folder, f'r-{radius}_{dis_method}_{b}_{a}.nii.gz')
                        if not os.path.isfile(source):
                            raise FileNotFoundError(
                                f'Missing omission pair {a}/{b} for {specie}-sub-{sub:02d}, '
                                f'run-{run:02d}, omit {stimulus}: {folder}')
                        image = nib.load(source)
                        rsa.check_same_space(('searchlight mask', ref), [(source, image)],
                                             context='step 11.2 leave-one-stimulus-out')
                        if image.shape != ref.shape:
                            raise ValueError(f'Expected a 3D dissimilarity map: {source}')
                        values[:, index] = image.get_fdata()[inside]
                        inputs.append(source)
                    values[~np.isfinite(values)] = np.nan
                    by_run.setdefault(analyzed_run, []).append(values)
            rdms[sub] = by_run
        included = sorted(rdms)
        if len(included) < 2 or len(included) / len(participants) < min_percentage_available:
            raise ValueError(f'Run {run}, omit {stimulus}: insufficient participants '
                             f'({len(included)}/{len(participants)}).')
        means = {sub: {r: finite_mean(values, axis=0) for r, values in by_run.items()}
                 for sub, by_run in rdms.items()}
        participant_results = {sub: [] for sub in targets if sub in rdms}
        for analyzed_run in {r for by_run in means.values() for r in by_run}:
            reference = {sub: by_run[analyzed_run] for sub, by_run in means.items()
                         if analyzed_run in by_run}
            total, count = finite_totals(reference.values())
            for sub in participant_results:
                if sub not in reference:
                    continue
                own = reference[sub]
                peer = finite_divide(total - np.where(np.isfinite(own), own, 0),
                                     count - np.isfinite(own))
                session_results = []
                for target in rdms[sub][analyzed_run]:
                    result = np.full(int(inside.sum()), np.nan)
                    for voxel, (a, b) in enumerate(zip(target, peer)):
                        valid = np.isfinite(a) & np.isfinite(b)
                        a, b = a[valid], b[valid]
                        if len(a) >= 2 and np.ptp(a) > 0 and np.ptp(b) > 0:
                            result[voxel] = correlate(a, b)[0]
                    session_results.append(result)
                participant_results[sub].append(finite_mean(session_results, axis=0))
        output = np.zeros(ref.shape, dtype=float)
        output[inside] = finite_mean([finite_mean(values, axis=0)
                                      for values in participant_results.values()], axis=0)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        nib.save(nib.Nifti1Image(output, ref.affine), path)
        with open(log_path, 'w') as stream:
            json.dump(dict(parameters=parameters, participants_included=included,
                           targets_included=sorted(participant_results), input_files=inputs,
                           reference='mean RDM of other participants with the same omitted run/stimulus',
                           weighting='sessions within analyzed run, runs within participant, participants equally',
                           missing_data='pairwise finite correlation; undefined correlations remain NaN',
                           output_file=path), stream, indent=2)
        if verbose:
            print(f'Saved omission similarity map: {path}', flush=True)
    return outputs

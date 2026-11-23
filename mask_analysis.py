"""Compute pairwise RSA dissimilarities across the voxels in a mask."""

import argparse
import csv
import hashlib
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
import sys


METHODS = ('mahalanobis', 'correlation', 'pearson', 'kendall', 'euclidean')
FOLDS = ('stim-wise', 'run-wise-multiple-runs', 'run-wise',
         'stim-wise-multiple-folds', 'stim-wise-all-runs')
STEP_DESCRIPTIONS = {
    1: 'Computing mask-level dissimilarities',
    2: 'Comparing participant dissimilarities with an RSA model',
    3: 'Computing the group mean and standard deviation',
    4: 'Computing participant model permutations',
    5: 'Sampling the group permutation distribution',
}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='EmoC', help='Dataset to use')
    parser.add_argument('--datafolder', help='Override the parent directory containing the dataset')
    parser.add_argument('--toolkit_dir', help='Override the directory containing this toolkit and Atlas')
    parser.add_argument('--task', help='Task to use (defaults to the dataset name)')
    parser.add_argument('--steps_to_run', type=int, nargs='+', choices=STEP_DESCRIPTIONS, default=[1],
                        help='Pipeline step to run (default: 1)')
    parser.add_argument('--model', default='basic', help='GLM model to use')
    parser.add_argument('--dis_method', type=str.lower, choices=METHODS, default='mahalanobis',
                        help='Dissimilarity measure (Pearson and Kendall use 1 minus the coefficient)')
    parser.add_argument('--leave_one_stimulus_out', action='store_true',
                        help='Compute dissimilarities omitting one beta map at a time')
    parser.add_argument('--rsa_model', help='RSA model name (required for Steps 2–5)')
    parser.add_argument('--rsa_method', choices=('kendall', 'pearson', 'correlation'), default='kendall',
                        help='Model comparison statistic (Kendall tau-a, Pearson r, or 1 minus Pearson r)')
    parser.add_argument('--model_specie', choices=['D', 'H'],
                        help='Steps 3/5 only: aggregate individual models named <rsa_model>_<specie>-sub-XX')
    parser.add_argument('--reps', type=int, default=100, help='Step 4 permutations per participant or run')
    parser.add_argument('--reps_group', type=int, default=1000, help='Step 5 group permutation draws')
    parser.add_argument('--min_percentage_available', type=float, default=1.0,
                        help='Minimum fraction of usable units for group steps (0 to 1)')
    parser.add_argument('--replace_rnd_files', action='store_true', help='Replace Step 4 and 5 outputs')
    parser.add_argument('--seed', type=int, default=0, help='Seed for model permutations and group draws')
    parser.add_argument('--rsa_class', help='RSA class to use for leave-one-stimulus-out comparisons')
    parser.add_argument('--specie', choices=['D', 'H'], default='H',
                        help="'D' for Dog, 'H' for Human")
    parser.add_argument('--mask_type', default='b_GreyMatter2mmB', help='Name of the ROI or atlas mask')
    parser.add_argument('--mask', type=Path, help='Path to a NIfTI mask')
    parser.add_argument('--radius', type=int, help='Legacy searchlight option, unused for mask analysis')
    parser.add_argument('--replace_file', action='store_true', help='Overwrite existing output files')
    parser.add_argument('--shuffle_participants', action='store_true', help='Shuffle participant order')
    parser.add_argument('--shuffle_runs', action='store_true', help='Shuffle run order')
    parser.add_argument('--allow_space_mismatch', action='store_true',
                        help='Warn about mismatched voxel grids instead of raising an error')
    parser.add_argument('--participants', '--participants_forced', dest='participants_forced',
                        type=int, nargs='+', default=[], help='List of participants to include')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--skip_prefile_check', action='store_true',
                        help='Recompute unit outputs and allow incomplete Step 4 pools in Step 5')
    parser.add_argument('--mah_fold', choices=FOLDS, default='stim-wise',
                        help='Grouping and cross-validation strategy for Mahalanobis distance')
    parser.add_argument('--job_marker_dir', type=Path, help='Directory for the Step 1 completion marker')
    args = parser.parse_args()
    if args.leave_one_stimulus_out and not args.rsa_class:
        parser.error('--leave_one_stimulus_out requires --rsa_class')
    if args.leave_one_stimulus_out and any(step != 1 for step in args.steps_to_run):
        parser.error('--leave_one_stimulus_out applies to Step 1')
    if any(step != 1 for step in args.steps_to_run) and not args.rsa_model:
        parser.error('Steps 2–5 require --rsa_model')
    if args.model_specie and any(step not in (3, 5) for step in args.steps_to_run):
        parser.error('--model_specie applies to Steps 3 and 5')
    if args.reps < 1 or args.reps_group < 1:
        parser.error('--reps and --reps_group must be positive')
    if not 0 <= args.min_percentage_available <= 1:
        parser.error('--min_percentage_available must be between 0 and 1')
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    return args


def _whiten_patterns(patterns, residuals, np):
    """Apply Ledoit–Wolf whitening without building a voxel-by-voxel matrix."""
    residuals = residuals - residuals.mean(axis=0)
    n_samples, n_voxels = residuals.shape
    _, singular_values, basis = np.linalg.svd(residuals, full_matrices=False)
    eigenvalues = singular_values ** 2 / n_samples
    mean_variance = eigenvalues.sum() / n_voxels
    if n_voxels == 1:
        return patterns / np.sqrt(max(mean_variance, np.finfo(float).eps))

    row_sums_squared = np.einsum('ij,ij->i', residuals, residuals)
    covariance_norm_squared = eigenvalues @ eigenvalues
    target_difference = covariance_norm_squared - n_voxels * mean_variance ** 2
    estimation_error = ((row_sums_squared @ row_sums_squared) / n_samples
                        - covariance_norm_squared) / n_samples
    shrinkage = np.clip(estimation_error / target_difference, 0, 1) if target_difference > 0 else 0

    min_variance = np.finfo(float).eps
    baseline = shrinkage * mean_variance
    component_variances = np.maximum(baseline + (1 - shrinkage) * eigenvalues, min_variance)
    coordinates = patterns @ basis.T
    whitened = (coordinates / np.sqrt(component_variances)) @ basis
    if len(basis) < n_voxels:
        whitened += (patterns - coordinates @ basis) / np.sqrt(max(baseline, min_variance))
    return whitened


def _mahalanobis_values(patterns, labels, folds, pairs, cross_validate, np):
    # Crossnobis requires the same set of voxels in every observation.
    patterns = patterns[:, np.isfinite(patterns).all(axis=0)]
    if not patterns.shape[1]:
        raise ValueError('No mask voxels have finite values in all included beta maps.')

    conditions, label_index = np.unique(labels, return_inverse=True)
    condition_indices = {label: index for index, label in enumerate(conditions)}
    if not cross_validate:
        whitened = _whiten_patterns(patterns, patterns, np)
        means = {label: whitened[label_index == index].mean(axis=0)
                 for label, index in condition_indices.items()}
        return [float(np.linalg.norm(means[a] - means[b])) if a in means and b in means
                else float('nan') for a, b in pairs]

    _, fold_index = np.unique(folds, return_inverse=True)
    n_folds = int(fold_index.max()) + 1
    means = np.zeros((n_folds, len(conditions), patterns.shape[1]))
    present = np.zeros(means.shape[:2], dtype=bool)
    if n_folds < 2:
        return [float('nan')] * len(pairs)

    for fold in range(n_folds):
        for condition in range(len(conditions)):
            selected = (fold_index == fold) & (label_index == condition)
            if selected.any():
                means[fold, condition] = patterns[selected].mean(axis=0)
                present[fold, condition] = True
    condition_means = means.sum(axis=0) / present.sum(axis=0)[:, None]
    residuals = (means - condition_means)[present]
    whitened = np.zeros_like(means)
    whitened[present] = _whiten_patterns(means[present], residuals, np)

    values = []
    for a, b in pairs:
        if a not in condition_indices or b not in condition_indices:
            values.append(float('nan'))
            continue
        i, j = condition_indices[a], condition_indices[b]
        shared = present[:, i] & present[:, j]
        n_shared = int(shared.sum())
        if n_shared < 2:
            values.append(float('nan'))
            continue
        differences = whitened[shared, i] - whitened[shared, j]
        total = differences.sum(axis=0)
        cross_product = total @ total - np.einsum('ij,ij->', differences, differences)
        values.append(float(cross_product / (n_shared * (n_shared - 1))))
    return values


def _dissimilarity(x, y, method, np, rsa):
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if method == 'euclidean':
        return float(np.linalg.norm(x - y)) if x.size else float('nan')
    if x.size < 2:
        return float('nan')
    if method == 'kendall':
        return float(1 - rsa.kendall_custom(x, y))
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(1 - np.clip((x @ y) / denominator, -1, 1)) if denominator else float('nan')


def _write_csv_log(path, columns, rows, log):
    """Stage both files before publishing the CSV and its log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = []
    try:
        for destination in (path, path.with_suffix('.log')):
            with NamedTemporaryFile(mode='w', dir=path.parent, encoding='utf-8', newline='',
                                    delete=False) as stream:
                staged.append((Path(stream.name), destination))
                if destination == path:
                    writer = csv.writer(stream)
                    writer.writerow(columns)
                    writer.writerows(rows)
                else:
                    csv_hash = hashlib.sha256(staged[0][0].read_bytes()).hexdigest()
                    stream.write('\n'.join([*log, f'CSV SHA-256: {csv_hash}']) + '\n')
        for temporary, destination in reversed(staged):
            temporary.replace(destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def _per_run(args):
    return args.dis_method != 'mahalanobis' or args.mah_fold in ('run-wise-multiple-runs', 'stim-wise-all-runs')


def _dissimilarity_stem(args, mask_name):
    name = f'{mask_name}_{args.dis_method}'
    return f'{name}_{args.mah_fold}' if args.dis_method == 'mahalanobis' else name


def _run_name(entry, task):
    return f"ses-{int(entry['session']):02d}_task-{task}_run-{int(entry['run_N']):02d}"


def _participant_results(args, datafolder, config, participant, mask_path, mask_img, mask, rsa, np, nib):
    from leave_one_stimulus_out import _observation_records, _omission_directory, _read_pairs

    subject = f'{args.specie}-sub-{participant:02d}'
    units = list(rsa.get_session_and_run_dict(datafolder, args.dataset, args.specie, participant))
    if not units:
        raise ValueError(f'No sessions or runs for {subject}.')
    if len({_run_name(entry, args.task) for entry in units}) != len(units):
        raise ValueError(f'Duplicate session/run entries for {subject}.')
    if args.shuffle_runs:
        np.random.shuffle(units)

    stim_types = config['stim_types']
    stim_indices = {stim: index for index, stim in enumerate(stim_types)}
    if len(stim_indices) < 2 or len(stim_indices) != len(stim_types):
        raise ValueError('The GLM config must list at least two unique stimulus labels.')
    method, fold = args.dis_method, args.mah_fold
    per_run = _per_run(args)
    cross_validate = method == 'mahalanobis' and fold != 'run-wise-multiple-runs'
    root = Path(datafolder, args.dataset, 'results', 'RSA_mask', args.model, subject)
    name = _dissimilarity_stem(args, rsa.strip_nii_extension(mask_path.name))

    records = _observation_records(units, stim_types, args.dataset, method, fold, config.get('model_dict'))
    selected_pairs = None
    if args.leave_one_stimulus_out:
        class_name = args.rsa_class.removeprefix('class_')
        root = root / f'class_{class_name}' / 'leave-one-out' / 'by_stim'
        if method != 'mahalanobis' or fold != 'stim-wise':
            selected_pairs = _read_pairs(Path(datafolder, args.dataset, 'rsa_models', 'by_class',
                                              f'class_{class_name}.csv'))
            unknown = {label for pair in selected_pairs for label in pair} - {row[2] for row in records}
            if unknown:
                raise ValueError(f'Class comparisons contain unavailable labels: {sorted(unknown)}')

    failures = []
    groups = [([row for row in records if row[0] == index], _run_name(unit, args.task))
              for index, unit in enumerate(units)] if per_run else [(records, None)]
    for group, run_name in groups:
        labels = list(dict.fromkeys(row[2] for row in group))
        if len(labels) < 2:
            failures.append(f'{run_name or subject}: fewer than two conditions')
            continue
        if cross_validate and len({row[3] for row in group}) < 2:
            failures.append(f'{run_name or subject}: cross-validation requires at least two folds')
            continue
        pairs = list(combinations(labels, 2)) if selected_pairs is None else [
            (a, b) for a, b in selected_pairs if a in labels and b in labels
        ]
        if not pairs:
            continue

        jobs = []
        for omitted in records if args.leave_one_stimulus_out else (None,):
            directory = root
            if omitted is not None:
                directory = Path(_omission_directory(str(root), units, omitted[0], omitted[1]))
            if run_name is not None:
                directory /= run_name
            result_path = directory / f'{name}.csv'
            if (args.replace_file or args.skip_prefile_check
                    or not result_path.is_file() or not result_path.with_suffix('.log').is_file()):
                jobs.append((omitted, result_path))
        if not jobs:
            continue

        patterns, sources = [], []
        for unit_index, stimulus, label, partition in group:
            entry = units[unit_index]
            path = rsa.resolve_beta_map(datafolder, args.dataset, args.model, args.specie, participant,
                                        entry['session'], entry['run_N'], args.task, stimulus,
                                        stim_indices[stimulus], verbose=args.verbose)
            if not Path(path).is_file():
                failures.append(f'{_run_name(entry, args.task)} {stimulus}: missing beta map {path}')
                continue
            image = nib.load(path)
            rsa.check_same_space((f'mask {mask_path}', mask_img), [(path, image)],
                                 context=f'mask analysis for {subject} {_run_name(entry, args.task)}')
            patterns.append(image.get_fdata()[mask])
            sources.append(f'{_run_name(entry, args.task)} {stimulus} -> {label} '
                           f'(fold {partition}): {path}')
        if len(patterns) != len(group):
            continue

        patterns = np.asarray(patterns)
        record_labels = np.asarray([row[2] for row in group])
        folds = np.asarray([row[3] for row in group])
        estimator = method
        if method == 'mahalanobis':
            estimator = ('cross-validated squared Mahalanobis (Ledoit-Wolf, pooled fold residuals)'
                         if cross_validate else 'within-run Mahalanobis (Ledoit-Wolf, run beta patterns)')

        # All omissions from other runs share the same result for this run.
        baseline_values = None
        for omitted, path in jobs:
            is_baseline = omitted not in group
            keep = np.array([row != omitted for row in group])
            if is_baseline and baseline_values is not None:
                values = baseline_values
            elif method == 'mahalanobis':
                values = _mahalanobis_values(patterns[keep], record_labels[keep], folds[keep],
                                             pairs, cross_validate, np)
            else:
                active = dict(zip(record_labels[keep], patterns[keep], strict=True))
                values = [_dissimilarity(active[a], active[b], method, np, rsa)
                          if a in active and b in active else float('nan') for a, b in pairs]
            if is_baseline:
                baseline_values = values
            log = [
                f'UTC timestamp: {datetime.now(timezone.utc).isoformat()}',
                f'Dataset: {args.dataset}', f'Task: {args.task}', f'GLM model: {args.model}',
                f'RSA model: {args.rsa_model}', f'Participant: {subject}',
                f'Scope: {run_name or "all available sessions/runs"}', f'Mask: {mask_path}',
                f'Mask voxels: {int(mask.sum())}', f'Allow space mismatch: {args.allow_space_mismatch}',
                f'Dissimilarity method: {method}', f'Mahalanobis fold: {fold if method == "mahalanobis" else "n/a"}',
                f'Estimator: {estimator}', f'Pairs: {len(pairs)}',
                f'Undefined dissimilarities (NaN): {sum(not np.isfinite(value) for value in values)}',
                f'CSV: {path}', 'Input beta maps used:',
            ]
            if omitted is not None:
                log.insert(-1, f'Omitted beta: {_run_name(units[omitted[0]], args.task)} {omitted[1]}')
            log.extend(source for source, included in zip(sources, keep, strict=True) if included)
            rows = ((a, b, value) for (a, b), value in zip(pairs, values, strict=True))
            _write_csv_log(path, ['category 1', 'category 2', 'dissimilarity value'], rows, log)
            if args.verbose or not args.leave_one_stimulus_out:
                print(f'Saved {path}')

    if failures:
        raise ValueError('\n'.join(failures))


def _read_model_csv(path, np):
    """Read the lower triangle of a searchlight-style labelled model CSV."""
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        categories = [label.strip() for label in next(reader, [])[1:]]
        rows = [row for row in reader if row]
    if (len(categories) < 2 or '' in categories or len(set(categories)) != len(categories)
            or len(rows) != len(categories) or any(len(row) != len(categories) + 1 for row in rows)):
        raise ValueError(f'Expected a square, labelled RSA model matrix: {path}')
    by_label = {row[0].strip(): row[1:] for row in rows}
    if set(by_label) != set(categories):
        raise ValueError(f'RSA model row and column labels must match: {path}')
    values = np.zeros((len(categories), len(categories)))
    for i, j in combinations(range(len(categories)), 2):
        # Searchlight reads the lower triangle. Accept upper-triangular exports too.
        cell = by_label[categories[j]][i].strip() or by_label[categories[i]][j].strip()
        values[i, j] = values[j, i] = float(cell) if cell else float('nan')
    return categories, values


def _read_dissimilarities(path):
    dissimilarities = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['category 1', 'category 2', 'dissimilarity value']:
            raise ValueError(f'Unexpected Step 1 CSV columns: {path}')
        for row in reader:
            a, b = row['category 1'], row['category 2']
            pair = frozenset((a, b))
            if not a or not b or a == b or pair in dissimilarities:
                raise ValueError(f'Invalid or duplicate pair {a!r}, {b!r} in {path}')
            dissimilarities[pair] = float(row['dissimilarity value'])
    return dissimilarities


def _rsa_score(observed, predicted, method, np):
    finite = np.isfinite(observed) & np.isfinite(predicted)
    observed, predicted = observed[finite], predicted[finite]
    n = len(observed)
    if n < 2:
        return float('nan')
    if method == 'kendall':
        from scipy.stats import kendalltau

        # Convert tau-b to searchlight's tau-a by restoring the untied denominator.
        pair_count = n * (n - 1) / 2
        untied = []
        for vector in (observed, predicted):
            _, counts = np.unique(vector, return_counts=True)
            untied.append(pair_count - (counts @ (counts - 1)) / 2)
        if not all(untied):
            return 0.0
        tau_b = kendalltau(observed, predicted).statistic
        return float(tau_b * np.sqrt(untied[0] * untied[1]) / pair_count)

    observed, predicted = observed - observed.mean(), predicted - predicted.mean()
    denominator = np.linalg.norm(observed) * np.linalg.norm(predicted)
    if not denominator:
        return float('nan')
    pearson = float(np.clip((observed @ predicted) / denominator, -1, 1))
    return 1 - pearson if method == 'correlation' else pearson


def _seeded_rng(seed, key, np):
    digest = hashlib.sha256(key.encode('utf-8')).digest()
    return np.random.default_rng([seed, int.from_bytes(digest[:16], 'little')])


def _file_record(label, path):
    stat = path.stat()
    return f'{label}: {path} (size={stat.st_size}, modified_ns={stat.st_mtime_ns})'


def _cached_result(path, settings):
    log_path = path.with_suffix('.log')
    if not path.is_file() or not log_path.is_file():
        return False
    csv_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    return set([*settings, f'CSV SHA-256: {csv_hash}']).issubset(
        log_path.read_text(encoding='utf-8').splitlines())


def _model_output_paths(root, model_name, subject, entry, args, stem):
    folder = root / model_name / subject
    if entry is not None:
        folder /= _run_name(entry, args.task)
    return folder / f'{stem}_similarity.csv', folder / f'{stem}_permutations.csv'


def _model_comparison(step, args, datafolder, config, units_by_participant,
                      root, dissimilarity_stem, score_stem, settings, rsa, np):
    model_cache = {}
    failures = []
    for participant, units in units_by_participant.items():
        subject = f'{args.specie}-sub-{participant:02d}'
        for entry in units if _per_run(args) else [None]:
            real_path, rnd_path = _model_output_paths(
                root, args.rsa_model, subject, entry, args, score_stem)
            csv_path = real_path if step == 2 else rnd_path
            pair_folder = root / subject
            if entry is not None:
                pair_folder /= _run_name(entry, args.task)
            pair_path = pair_folder / f'{dissimilarity_stem}.csv'
            try:
                run_number = int(entry['run_N']) if entry is not None else 0
                model_path = Path(rsa._resolve_rsa_model_path(
                    datafolder, args.dataset, args.rsa_model, run_number))
                log = settings + [_file_record('Step 1 CSV', pair_path),
                                  _file_record('RSA model CSV', model_path)]
                if step == 4:
                    log += [f'Permutations: {args.reps}', f'Seed: {args.seed}']
                replace = args.replace_file if step == 2 else args.replace_rnd_files
                if not (replace or args.skip_prefile_check) and _cached_result(csv_path, log):
                    continue

                observed_pairs = _read_dissimilarities(pair_path)
                if model_path not in model_cache:
                    model_cache[model_path] = _read_model_csv(model_path, np)
                categories, model_matrix = model_cache[model_path]
                observed_labels = {label for pair in observed_pairs for label in pair}
                if args.dis_method == 'mahalanobis' and args.mah_fold == 'stim-wise-multiple-folds':
                    labels, mapping = rsa._get_emoc_multiple_fold_model_labels(
                        units, config.get('model_dict'), categories)
                    labels = [label for label in labels if label in observed_labels]
                else:
                    labels = [label for label in categories if label in observed_labels]
                    mapping = {label: label for label in labels}
                if len(labels) < 2:
                    raise ValueError(f'RSA model and Step 1 CSV share fewer than two labels: {pair_path}')

                category_index = {label: index for index, label in enumerate(categories)}
                indices = [category_index[mapping[label]] for label in labels]
                expanded_model = model_matrix[np.ix_(indices, indices)]
                row, column = np.triu_indices(len(labels), 1)
                observed = np.asarray([observed_pairs.get(frozenset((labels[i], labels[j])), np.nan)
                                       for i, j in zip(row, column, strict=True)])
                log += [f'Labels compared: {len(labels)}',
                        f'Finite neural pairs: {int(np.isfinite(observed).sum())} of {len(row)}']
                if step == 2:
                    score = _rsa_score(observed, expanded_model[row, column], args.rsa_method, np)
                    header, rows = ['similarity'], [(score,)]
                else:
                    if not np.isfinite(expanded_model[row, column]).all():
                        raise ValueError(f'Model distances must be finite for label permutations: {model_path}')
                    key = f'{args.dataset}/{csv_path.relative_to(root).as_posix()}'
                    rng = _seeded_rng(args.seed, key, np)
                    scores = []
                    for _ in range(args.reps):
                        order = rng.permutation(len(labels))
                        predicted = expanded_model[order[row], order[column]]
                        scores.append(_rsa_score(observed, predicted, args.rsa_method, np))
                    header, rows = ['permutation', 'similarity'], enumerate(scores)
                    log.append('Permutation scheme: jointly shuffle model row and column labels')
                _write_csv_log(csv_path, header, rows, log)
                print(f'Saved {csv_path}')
            except (OSError, ValueError) as error:
                print(f'{csv_path}: {error}', file=sys.stderr)
                failures.append(csv_path)
    if failures:
        raise RuntimeError(f'Step {step} failed for {len(failures)} analysis unit(s).')


def _read_scores(path, randomized):
    columns = ['permutation', 'similarity'] if randomized else ['similarity']
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != columns:
            raise ValueError(f'Expected columns {columns} in {path}')
        if randomized:
            scores = {}
            for row in reader:
                index = int(row['permutation'])
                if index < 0 or index in scores:
                    raise ValueError(f'Invalid or duplicate permutation {index} in {path}')
                scores[index] = float(row['similarity'])
            return scores
        rows = list(reader)
        if len(rows) != 1:
            raise ValueError(f'Expected one observed similarity in {path}')
        return float(rows[0]['similarity'])


def _group_comparison(step, args, datafolder, units_by_participant,
                      root, score_stem, settings, rsa, np):
    randomized = step == 5
    source_models = rsa._source_rsa_models(datafolder, args.dataset, args.rsa_model, args.model_specie)
    paths = []
    for model_name in source_models:
        for participant, units in units_by_participant.items():
            subject = f'{args.specie}-sub-{participant:02d}'
            for entry in units if _per_run(args) else [None]:
                real_path, rnd_path = _model_output_paths(root, model_name, subject, entry, args, score_stem)
                paths.append(rnd_path if randomized else real_path)
    if not paths:
        raise ValueError('No participant or run scores expected for the requested group.')

    available = {}
    unavailable = []
    for path in sorted(paths):
        if not path.is_file():
            unavailable.append(f'Missing: {path}')
            continue
        scores = _read_scores(path, randomized)
        if randomized:
            indices = sorted(index for index in scores if index < args.reps)
            if len(indices) < args.reps and not args.skip_prefile_check:
                unavailable.append(f'Incomplete permutation pool: {path} ({len(indices)}/{args.reps})')
                continue
            scores = [scores[index] for index in indices if np.isfinite(scores[index])]
            if not scores:
                unavailable.append(f'No finite permutation scores: {path}')
                continue
        elif not np.isfinite(scores):
            unavailable.append(f'Undefined similarity: {path}')
            continue
        available[path] = scores

    fraction = len(available) / len(paths)
    if not available or fraction < args.min_percentage_available:
        detail = '\n'.join(unavailable[:10])
        raise ValueError(f'Step {step}: {len(available)}/{len(paths)} usable units ({fraction:.1%}); '
                         f'required {args.min_percentage_available:.1%} and at least one unit.\n{detail}')

    output_model = f'{args.rsa_model}_{args.model_specie}' if args.model_specie else args.rsa_model
    output_dir = root / output_model / 'mean'
    suffix = 'group_permutations' if randomized else 'mean'
    output = output_dir / f'{args.specie}-{score_stem}_{suffix}.csv'
    log = settings + [f'Expected units: {len(paths)}', f'Usable units: {len(available)}',
                      f'Availability: {fraction:.6f}', f'Minimum availability: {args.min_percentage_available}',
                      'Weighting: equal per input unit']
    log += [_file_record('Input CSV', path) for path in available]
    log += unavailable
    if randomized:
        log += [f'Permutations requested per unit: {args.reps}', f'Group permutations: {args.reps_group}',
                f'Seed: {args.seed}', f'Allow incomplete permutation pools: {args.skip_prefile_check}',
                'Sampling: independently draw one finite Step 4 score per unit with replacement']
        if not args.replace_rnd_files and _cached_result(output, log):
            return
        sums = np.zeros(args.reps_group)
        for path, scores in available.items():
            key = f'{args.dataset}/{output_model}/{path.relative_to(root).as_posix()}'
            rng = _seeded_rng(args.seed, key, np)
            pool = np.asarray(scores)
            sums += pool[rng.integers(len(pool), size=args.reps_group)]
            log.append(f'Finite scores in pool: {len(pool)} ({path})')
        header, rows = ['permutation', 'mean'], enumerate(sums / len(available))
    else:
        scores = list(available.values())
        header, rows = ['mean', 'std', 'units'], [(np.mean(scores), np.std(scores), len(scores))]
    _write_csv_log(output, header, rows, log)
    print(f'Saved {output}')


def main():
    args = parse_arguments()
    if args.task is None:
        args.task = args.dataset
    if os.name == 'nt':
        mount = Path(r'P:\userdata\raulh87')
        toolkit = Path(r'C:\github\dog_brain_toolkit')
    else:
        mount = Path('/home/raulh87/mnt/a471/userdata/raulh87')
        toolkit = mount / 'github' / 'dog_brain_toolkit'
    datafolder = os.path.abspath(args.datafolder) if args.datafolder is not None else str(mount / 'data')
    toolkit = Path(args.toolkit_dir).resolve() if args.toolkit_dir is not None else toolkit
    sys.path.append(str(toolkit))

    import nibabel as nib
    import numpy as np
    import yaml
    import rsa_utils as rsa

    config_path = Path(datafolder, args.dataset, 'config_files', f'{args.specie}_{args.model}.yaml')
    with config_path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    print(f'Loaded config from: {config_path}')

    if args.mask is not None:
        mask_path = args.mask.resolve()
    else:
        mask_type = rsa.strip_nii_extension(args.mask_type)
        mask_path = Path(rsa.roi_mask_path(datafolder, args.dataset, args.specie, mask_type))
        if args.specie == 'D' and mask_type != 'cope13' and not mask_path.is_file():
            mask_path = toolkit / 'Atlas' / 'Dog' / 'Nitzsche' / rsa.with_nii_extension(mask_type)
    if 1 in args.steps_to_run:
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        mask = np.isfinite(mask_data) & (mask_data != 0)
        if mask.ndim != 3 or not mask.any():
            raise ValueError(f'Mask must be a 3D image containing finite, nonzero voxels: {mask_path}')
        print(f'Using mask: {mask_path} ({int(mask.sum())} voxels)')
        rsa.set_space_check(strict=not args.allow_space_mismatch)

    participants = list(dict.fromkeys(args.participants_forced or config['participants']))
    if not participants:
        raise ValueError('No participants selected.')
    if args.shuffle_participants:
        np.random.shuffle(participants)

    root = Path(datafolder, args.dataset, 'results', 'RSA_mask', args.model)
    dissimilarity_stem = _dissimilarity_stem(args, rsa.strip_nii_extension(mask_path.name))
    score_stem = f'{dissimilarity_stem}_{args.rsa_method}'
    settings = [
        f'Dataset: {args.dataset}', f'Task: {args.task}', f'GLM model: {args.model}',
        f'RSA model: {args.rsa_model}', f'Species: {args.specie}', f'Mask: {mask_path}',
        f'Dissimilarity: {args.dis_method}', f'RSA method: {args.rsa_method}',
    ]
    if args.dis_method == 'mahalanobis':
        settings.append(f'Mahalanobis fold: {args.mah_fold}')

    units_by_participant = None
    for step in args.steps_to_run:
        print(f'### Step {step}: {STEP_DESCRIPTIONS[step]} ###')
        if step == 1:
            failed = []
            for participant in participants:
                try:
                    _participant_results(args, datafolder, config, participant, mask_path, mask_img,
                                         mask, rsa, np, nib)
                except rsa.SpaceMismatchError:
                    raise
                except (OSError, ValueError) as error:
                    print(f'{args.specie}-sub-{participant:02d}: {error}', file=sys.stderr)
                    failed.append(participant)
            if failed:
                raise RuntimeError(f'Step {step} failed for participants {failed}.')
        else:
            if units_by_participant is None:
                needs_runs = _per_run(args) or (
                    args.dis_method == 'mahalanobis' and args.mah_fold == 'stim-wise-multiple-folds'
                    and any(item in (2, 4) for item in args.steps_to_run)
                )
                units_by_participant = {}
                for participant in participants:
                    units = (rsa.get_session_and_run_dict(datafolder, args.dataset, args.specie, participant)
                             if needs_runs else [])
                    if needs_runs and not units:
                        raise ValueError(f'No sessions or runs for {args.specie}-sub-{participant:02d}.')
                    if len({_run_name(entry, args.task) for entry in units}) != len(units):
                        raise ValueError(f'Duplicate session/run entries for {args.specie}-sub-{participant:02d}.')
                    units_by_participant[participant] = units
            if step in (2, 4):
                _model_comparison(step, args, datafolder, config, units_by_participant,
                                  root, dissimilarity_stem, score_stem, settings, rsa, np)
            else:
                _group_comparison(step, args, datafolder, units_by_participant,
                                  root, score_stem, settings, rsa, np)

        if args.job_marker_dir is not None:
            args.job_marker_dir.mkdir(parents=True, exist_ok=True)
            (args.job_marker_dir / f'{step}.done').write_text(datetime.now(timezone.utc).isoformat(),
                                                             encoding='utf-8')


if __name__ == '__main__':
    main()

"""Resumable Colab orchestration for regression steps 15, 15.3, 15.4, 15.5.

Uses existing participant packages and optional step-1 result ZIPs. Package
code is never imported: deploy this file, gpu_regression.py and gpu_rsa.py.
"""
import hashlib
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import nibabel as nib
import numpy as np
import torch

try:
    from . import gpu_regression as gpu
except ImportError:
    import gpu_regression as gpu


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def copy_with_progress(source, destination):
    size = Path(source).stat().st_size
    started = last_report = time.monotonic()
    log(f'Copying {Path(source).name}: {size / 1e6:.1f} MB')
    copied = 0
    with open(source, 'rb') as src, open(destination, 'wb') as dst:
        while chunk := src.read(8 * 1024 * 1024):
            dst.write(chunk)
            copied += len(chunk)
            now = time.monotonic()
            if now - last_report >= 30:
                log(f'Copied {copied / 1e6:.1f}/{size / 1e6:.1f} MB')
                last_report = now
    log(f'Copy complete ({time.monotonic() - started:.0f}s)')


def extract_safe(archive, destination, prefix=None, beta_only=False):
    destination = Path(destination).resolve()
    last_report = time.monotonic()
    extracted = 0
    log(f'Extracting {Path(archive).name}')
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            name = info.filename
            parts = PurePosixPath(name).parts
            if '\\' in name or ':' in name or name.startswith('/') or '..' in parts:
                raise ValueError(f'Unsafe ZIP member {name!r}')
            if prefix is not None and not name.startswith(prefix):
                continue
            if beta_only and ('_beta_map' not in name or not name.endswith('.nii.gz')):
                continue
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f'Unsafe ZIP member {name!r}')
            if not info.is_dir():
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, target.open('wb') as out:
                    shutil.copyfileobj(source, out)
                extracted += 1
                if time.monotonic() - last_report >= 30:
                    log(f'Extracted {extracted} files from {Path(archive).name}')
                    last_report = time.monotonic()
    log(f'Extracted {extracted} files')


def discover_packages(packages_dir, specie='H', dataset='EmoB', models=None,
                      participants=None, model='basic-block', dis_method='correlation'):
    found = {}
    for path in sorted(Path(packages_dir).glob('pkg_*.zip')):
        with zipfile.ZipFile(path) as zf:
            if 'manifest.json' not in zf.namelist():
                continue
            m = json.loads(zf.read('manifest.json'))
        if (m.get('specie'), m.get('dataset'), m.get('model'), m.get('dis_method')) != (
                specie, dataset, model, dis_method) or 'sub_N' not in m:
            continue
        sub = int(m['sub_N'])
        if participants is not None and sub not in participants:
            continue
        if sub in found:
            raise ValueError(f'Duplicate participant package for {specie}-sub-{sub:02d}')
        if dis_method not in ('correlation', 'mahalanobis') or (
                dis_method == 'mahalanobis' and m.get('mah_fold') != 'stim-wise'):
            raise ValueError('Supported input layouts: correlation or mahalanobis/stim-wise.')
        if not m.get('runs') or len({(r['session'], r['run_N']) for r in m['runs']}) != len(m['runs']):
            raise ValueError(f'Empty/duplicate runs in {path}')
        found[sub] = (path, m)
    if not found:
        raise FileNotFoundError(f'No {dataset}/{model}/{specie}/{dis_method} packages in {packages_dir}')
    reference = next(iter(found.values()))[1]
    for path, m in found.values():
        for key in ('radius', 'mask_type', 'task', 'mah_fold', 'categories'):
            if m[key] != reference[key]:
                raise ValueError(f'{path}: inconsistent {key} across packages')
    if participants is not None and set(participants) != set(found):
        raise ValueError(f'Missing participant packages: {sorted(set(participants) - set(found))}')
    return found


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _receipt_relative(m, regression, target, group=False):
    return (Path(m['dataset']) / 'results/RSA_regression' / m['model'] / regression /
            target / ('mean' if group else f"{m['specie']}-sub-{m['sub_N']:02d}") /
            f"{m['specie']}_colab_regression_receipt.json")


def _cached(path, signature, required_steps, receipt):
    if not Path(path).is_file():
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            r = json.loads(zf.read(receipt.as_posix()))
            return r['signature'] == signature and set(required_steps) <= set(r['steps'])
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return False


def _publish(data_root, folders, zip_path, receipt, payload):
    """Write locally, copy to Drive .part, then rename after the copy completes."""
    data_root, zip_path = Path(data_root), Path(zip_path)
    receipt_path = data_root / receipt
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    local = data_root.parent / 'export.zip'
    log(f'Creating checkpoint {zip_path.name}')
    with zipfile.ZipFile(local, 'w', zipfile.ZIP_STORED, allowZip64=True) as zf:
        paths = {receipt_path}
        for folder in folders:
            paths.update(p for p in (data_root / folder).rglob('*') if p.is_file())
        last_report = time.monotonic()
        for index, path in enumerate(sorted(paths), 1):
            zf.write(path, path.relative_to(data_root).as_posix())
            if time.monotonic() - last_report >= 30:
                log(f'Checkpoint {zip_path.name}: packed {index}/{len(paths)} files')
                last_report = time.monotonic()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    partial = zip_path.with_suffix('.zip.part')
    copy_with_progress(local, partial)
    partial.replace(zip_path)
    local.unlink()
    print(f'Checkpoint saved: {zip_path.name}', flush=True)


def run_regression(packages_dir, out_dir, support_zip, *, specie='H', dataset='EmoB',
                   model='basic-block', dis_method='correlation', models=None,
                   participants=None, regression_model='visual_3', steps=(15, 15.3, 15.4, 15.5),
                   step1_results_dir=None, reps=100, reps_group=1000, seed=42,
                   work_root='/content/regression_work', device='cuda', voxel_batch=2048,
                   permutation_batch=8, group_batch=16, step1_batch=256, force=False):
    """Run requested steps, with one participant/target result ZIP per checkpoint.

    Group means use every configured run with equal weight, as CPU steps
    15.3/15.5 do. Full participant coverage is required; no silent subset means.
    """
    steps = set(steps)
    if not steps or not steps <= {15, 15.3, 15.4, 15.5}:
        raise ValueError('steps must select from 15, 15.3, 15.4, 15.5')
    if min(reps, reps_group, voxel_batch, permutation_batch, group_batch, step1_batch) < 1:
        raise ValueError('Permutation counts and batch sizes must be positive')
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab, or explicitly use device="cpu" for validation.')
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir)
    with zipfile.ZipFile(support_zip) as zf:
        support = json.loads(zf.read('regression_manifest.json'))
    if support['dataset'] != dataset or support['regression_model'] != regression_model:
        raise ValueError('Support package dataset/regression_model differs from settings')
    controls = support['controls']
    if participants is None:
        participants = support['participants_by_species'].get(specie)
    if not participants:
        raise ValueError(f'Support package has no expected participants for {specie}')
    packages = discover_packages(packages_dir, specie, dataset, participants=participants,
                                 model=model, dis_method=dis_method)
    selected = list(models) if models is not None else support['models']
    if not selected or set(selected) & set(controls):
        raise ValueError('Select at least one target; a target cannot also be a control')
    if set(selected) - set(support['models']):
        raise ValueError('Selected target missing from support package; rebuild it with --models')
    reference = next(iter(packages.values()))[1]
    print(f'{dataset} {specie}: {len(packages)} participants, {len(selected)} targets, '
          f'{reps} individual / {reps_group} group permutations; {device}', flush=True)
    written, signatures, archives = [], {}, {}
    # Configuration plus model-file checksums: stale outputs cannot satisfy a
    # changed analysis, even if their output filenames are unchanged.
    support_hash = hashlib.sha256(Path(support_zip).read_bytes()).hexdigest()
    for sub, (package, manifest) in packages.items():
        stat = package.stat()
        step1 = None
        if step1_results_dir:
            tag = 'corr' if dis_method == 'correlation' else 'mah'
            candidate = Path(step1_results_dir) / f'result_step1_{tag}_{specie}-sub-{sub:02d}.zip'
            if candidate.is_file():
                step1 = candidate
        stamp = [stat.st_size, stat.st_mtime_ns]
        step1_stamp = [step1.name, step1.stat().st_size, step1.stat().st_mtime_ns] if step1 else None
        for target in selected:
            signatures[sub, target] = _digest([gpu.VERSION, manifest, target, regression_model,
                                               support_hash, reps, seed, stamp, step1_stamp])
            archives[sub, target] = out_dir / f'result_regression_{target}_{specie}-sub-{sub:02d}.zip'
        requested = steps & {15, 15.4}
        needed = [t for t in selected if requested and (force or not _cached(
            archives[sub, t], signatures[sub, t], requested,
            _receipt_relative(manifest, regression_model, t)))]
        if not needed:
            continue
        with tempfile.TemporaryDirectory(prefix=f'{specie}-{sub:02d}-', dir=work_root) as temp:
            root = Path(temp)
            # Copy the package off Drive first; many small ZIP reads over FUSE
            # are slower than reading the single local copy.
            local = root / 'input.zip'
            copy_with_progress(package, local)
            extract_safe(local, root, prefix='data/')
            local.unlink()
            extract_safe(support_zip, root, prefix='data/')
            if step1:
                copy_with_progress(step1, local)
                prefix = f"{dataset}/results/RSA/{model}/{specie}-sub-{sub:02d}/"
                extract_safe(local, root / 'data', prefix=prefix)
                local.unlink()
            mask, _ = gpu.gpu_rsa.load_reference_mask(str(root / 'data'), manifest)
            participant_steps = {target: set(requested) for target in needed}
            # Restore existing outputs so running step 15.4 later retains 15.
            for target in needed:
                arc = archives[sub, target]
                receipt = _receipt_relative(manifest, regression_model, target)
                if _cached(arc, signatures[sub, target], [], receipt):
                    extract_safe(arc, root / 'data')
                    old = json.loads((root / 'data' / receipt).read_text())
                    participant_steps[target].update(old['steps'])
            for entry in manifest['runs']:
                label = f'{specie}-sub-{sub:02d} session {entry["session"]} run {entry["run_N"]}'
                log(label)
                run_pending, run_receipts, run_archives, run_steps = [], {}, {}, {}
                for target in needed:
                    receipt = gpu.run_folder(manifest, regression_model, target, entry) / 'colab_run_receipt.json'
                    archive = out_dir / 'run_checkpoints' / (
                        f'result_regression_{target}_{specie}-sub-{sub:02d}_'
                        f'ses-{int(entry["session"]):02d}_run-{int(entry["run_N"]):02d}.zip')
                    run_receipts[target], run_archives[target] = receipt, archive
                    run_steps[target] = set(requested)
                    if _cached(archive, signatures[sub, target], [], receipt):
                        extract_safe(archive, root / 'data')
                        old = json.loads((root / 'data' / receipt).read_text())
                        run_steps[target].update(old['steps'])
                    if force or not _cached(archive, signatures[sub, target], requested, receipt):
                        run_pending.append(target)
                if not run_pending:
                    log(f'{label}: all target/run checkpoints restored')
                    continue
                data, all_pairs = gpu.load_run_data(root, manifest, entry, mask, device, step1_batch,
                                                    progress=log)
                pair_columns = {tuple(sorted(p)): i for i, p in enumerate(all_pairs)}
                for target_n, target in enumerate(run_pending, 1):
                    target_started = time.monotonic()
                    log(f'{label}: target {target_n}/{len(run_pending)} {target}')
                    indices = ([None] if 15 in requested else []) + (list(range(reps)) if 15.4 in requested else [])
                    pairs, designs = gpu.build_designs(root, manifest, target, controls, entry['run_N'], indices,
                                                       gpu.seed_for(seed, entry['session']))
                    try:
                        columns = [pair_columns[tuple(sorted(p))] for p in pairs]
                    except KeyError as error:
                        raise ValueError(f'{target}: target categories absent from neural pairwise maps') from error
                    for i, beta, t, p, dof in gpu.fit_designs(
                            data[:, columns], designs, device, voxel_batch, permutation_batch,
                            progress=log):
                        index = indices[i]
                        folder = root / 'data' / gpu.run_folder(manifest, regression_model, target, entry, index is not None)
                        stem, suffix = gpu.map_stem(manifest, index)
                        for stat_name, values in [('beta', beta), ('t', t), ('p', p)]:
                            gpu.save_vector(folder / f'{stem}_{stat_name}_map{suffix}.nii.gz', values, mask,
                                            outside=1 if stat_name == 'p' else 0)
                        record = dict(target_model=target, regression_model=regression_model,
                                      regressors=['intercept', target, *controls], n_pairs=len(pairs), dof=dof,
                                      standardize=True, rnd=index is not None, rnd_index=index,
                                      target_vector=designs[i, :, 1].tolist(), seed=seed,
                                      session=entry['session'], run_N=entry['run_N'], version=gpu.VERSION,
                                      regressor_files={n: str(gpu.model_path(root, manifest, n, entry['run_N']).relative_to(root / 'data'))
                                                       for n in [target, *controls]})
                        (folder / f'{stem}_regression{suffix}.json').write_text(json.dumps(record, indent=2))
                        if i == 0 or (i + 1) % 10 == 0 or i + 1 == len(indices):
                            log(f'{target}: wrote fit {i + 1}/{len(indices)} (beta/t/p + JSON)')
                    _publish(root / 'data',
                             [gpu.run_folder(manifest, regression_model, target, entry, rnd)
                              for rnd in (False, True)], run_archives[target], run_receipts[target],
                             dict(signature=signatures[sub, target], steps=sorted(run_steps[target])))
                    log(f'{target}: target/run complete in {time.monotonic() - target_started:.0f}s')
                del data
            for target in needed:
                folders = [gpu.run_folder(manifest, regression_model, target, manifest['runs'][0], rnd).parent
                           for rnd in (False, True)]
                _publish(root / 'data', folders, archives[sub, target],
                         _receipt_relative(manifest, regression_model, target),
                         dict(signature=signatures[sub, target], steps=sorted(participant_steps[target])))
                written.append(str(archives[sub, target]))
    if steps & {15.3, 15.5}:
        for target in selected:
            required = ({15} if 15.3 in steps else set()) | ({15.4} if 15.5 in steps else set())
            for sub, (_, m) in packages.items():
                if not _cached(archives[sub, target], signatures[sub, target], required,
                               _receipt_relative(m, regression_model, target)):
                    raise FileNotFoundError(f'Missing or stale steps {sorted(required)} for {target}, {specie}-sub-{sub:02d}. Run 15/15.4 first.')
            signature = _digest([gpu.VERSION, [signatures[s, target] for s in packages], reps_group, seed])
            output = out_dir / f'result_regression_group_{target}_{specie}.zip'
            receipt = _receipt_relative(reference, regression_model, target, True)
            group_steps = steps & {15.3, 15.5}
            if not force and _cached(output, signature, group_steps, receipt):
                continue
            with tempfile.TemporaryDirectory(prefix='group-', dir=work_root) as temp:
                root = Path(temp)
                extract_safe(support_zip, root, prefix='data/')
                mask, _ = gpu.gpu_rsa.load_reference_mask(str(root / 'data'), reference)
                for sub in packages:
                    extract_safe(archives[sub, target], root / 'data', beta_only=True)
                if _cached(output, signature, [], receipt):
                    extract_safe(output, root / 'data')
                    group_steps |= set(json.loads((root / 'data' / receipt).read_text())['steps'])
                _group_target(root, packages, target, regression_model, mask, steps, reps, reps_group,
                              seed, device, voxel_batch, group_batch)
                folders = [Path(dataset) / 'results' / family / model / regression_model / target / 'mean'
                           for family in ('RSA_regression', 'RSA_regression_rnd')]
                _publish(root / 'data', folders, output, receipt,
                         dict(signature=signature, steps=sorted(group_steps)))
                written.append(str(output))
    return written


def _group_target(root, packages, target, regression, mask, steps, reps, reps_group,
                  seed, device, voxel_batch, group_batch):
    reference = next(iter(packages.values()))[1]
    mask_bool = mask.get_fdata() > 0
    for rnd, step in [(False, 15.3), (True, 15.5)]:
        if step not in steps:
            continue
        paths, pools = [], []
        for _, m in packages.values():
            for entry in m['runs']:
                pool = []
                for index in (range(reps) if rnd else [None]):
                    folder = gpu.run_folder(m, regression, target, entry, rnd)
                    stem, suffix = gpu.map_stem(m, index)
                    path = root / 'data' / folder / f'{stem}_beta_map{suffix}.nii.gz'
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    pool.append(len(paths))
                    paths.append(path)
                pools.append(pool)
        bank_path = root / 'bank.npy'
        bank = np.lib.format.open_memmap(bank_path, mode='w+', dtype=np.float32,
                                        shape=(len(paths), int(mask_bool.sum())))
        log(f'Step {step}, {target}: loading {len(paths)} beta maps into local masked bank')
        last_report = time.monotonic()
        for i, path in enumerate(paths):
            img = nib.load(str(path))
            gpu.gpu_rsa.check_same_space(('mask', mask), [(str(path), img)])
            bank[i] = img.get_fdata()[mask_bool]
            if (i + 1) % 100 == 0 or time.monotonic() - last_report >= 30:
                log(f'Step {step}, {target}: loaded {i + 1}/{len(paths)} beta maps')
                last_report = time.monotonic()
        rng = np.random.default_rng(gpu.seed_for(seed, target, reference['specie'], 'group'))
        draws = reps_group if rnd else 1
        selections = np.column_stack([rng.choice(pool, size=draws) for pool in pools])
        mean = np.lib.format.open_memmap(root / 'mean.npy', mode='w+', dtype=np.float64,
                                        shape=(draws, bank.shape[1]))
        std = np.lib.format.open_memmap(root / 'std.npy', mode='w+', dtype=np.float64, shape=mean.shape)
        log(f'Step {step}, {target}: reducing {draws} group draws on {device}')
        last_report = time.monotonic()
        for g, v, means, stds in gpu.group_moments(bank, selections, device, voxel_batch, group_batch):
            mean[g:g + len(means), v:v + means.shape[1]] = means
            std[g:g + len(stds), v:v + stds.shape[1]] = stds
            if time.monotonic() - last_report >= 30:
                log(f'Step {step}, {target}: group draws {g + 1}-{g + len(means)}/{draws}, '
                    f'voxel chunk {v + 1}-{v + means.shape[1]}/{bank.shape[1]}')
                last_report = time.monotonic()
        family = 'RSA_regression_rnd' if rnd else 'RSA_regression'
        folder = root / 'data' / reference['dataset'] / 'results' / family / reference['model'] / regression / target / 'mean'
        stem = f"{reference['mask_type']}-{reference['specie']}-r-{reference['radius']}_{reference['dis_method']}_beta"
        log(f'Step {step}, {target}: writing {draws} group mean/std maps')
        last_report = time.monotonic()
        for i in range(draws):
            suffix = f'_{i:05d}' if rnd else ''
            for name, array in [('mean', mean), ('std', std)]:
                gpu.save_vector(folder / f'{stem}_{name}{suffix}.nii.gz', array[i], mask, dtype=np.float64)
            files = [paths[j].relative_to(root / 'data').as_posix() for j in selections[i]]
            (folder / f'{stem}_mean{suffix}.json').write_text(json.dumps(dict(
                statistic='beta', weighting='equal participant/session/run maps',
                file_list=files, perc_available=1., expected_runs=len(pools),
                target_model=target, regression_model=regression, rnd=rnd,
                group_index=i if rnd else None, seed=seed, version=gpu.VERSION), indent=2))
            if i == 0 or (i + 1) % 25 == 0 or time.monotonic() - last_report >= 30:
                log(f'Step {step}, {target}: wrote {i + 1}/{draws} group mean/std maps')
                last_report = time.monotonic()
        del array, bank, mean, std
        for filename in ('bank.npy', 'mean.npy', 'std.npy'):
            (root / filename).unlink()
        print(f'Step {step}: {target}, {draws} group mean/std maps from {len(pools)} runs', flush=True)

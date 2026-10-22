"""Step 16.2: source-participant vs target-species group Kendall tau-b RSA.

No distance estimation, permutations, thresholding, or statistical inference.
The HDF5 matrix indexes searchlight *centres*, not constituent sphere voxels.
"""
from __future__ import annotations

import csv
import copy
import contextlib
import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

VERSION = '2.0.0'
SCHEMA = 'cross-species-participant-to-group-16.2-v2'
LEGACY_SCHEMA = 'dog-human-rsa-16.2-v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_pairs(path):
    with open(path, newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        if not {'cat1', 'cat2'}.issubset(reader.fieldnames or []):
            raise ValueError('Class CSV requires cat1 and cat2 columns.')
        pairs = [(row['cat1'].strip(), row['cat2'].strip()) for row in reader]
    if (len(pairs) < 2 or any(not a or not b or a == b for a, b in pairs)
            or len({tuple(sorted(pair)) for pair in pairs}) != len(pairs)):
        raise ValueError('Class CSV requires at least two unique, non-diagonal pairs; reversed duplicates are invalid.')
    return pairs


def _grid(mask):
    image = nib.load(str(mask))
    values = np.asarray(image.dataobj)
    inside = np.isfinite(values) & (values != 0)
    if values.ndim != 3 or not inside.any():
        raise ValueError(f'Mask must be a nonempty 3D image: {mask}')
    return image, inside


def _check_grid(image, reference, path):
    if image.shape != reference.shape or not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5):
        raise ValueError(f'Voxel grid mismatch against species mask: {path}')


def default_radius(specie):
    """Dataset defaults; the caller independently selects source and target."""
    return {'D': 3, 'H': 4}[specie]


def _mask_name(mask):
    return Path(mask).name.removesuffix('.nii.gz').removesuffix('.nii')


def discover_inputs(datafolder, dataset='EmoC', model='basic-block',
                    rsa_class='all-mahalanobis', radius=None, *,
                    participants=None, target_participants=None,
                    mask=None, target_mask=None, toolkit_dir=None,
                    specie='D', target_specie='H', target_radius=None,
                    mask_type='b_GreyMatter2mmB', target_mask_type='b_GreyMatter2mmB'):
    """Resolve existing pooled step-1 maps, using each species' own config.

    Returned paths are absolute; packaged manifests use equivalent relative paths.
    Dataset ROI masks take precedence; the dog atlas is a legacy fallback.
    """
    if specie not in ('D', 'H') or target_specie not in ('D', 'H') or specie == target_specie:
        raise ValueError('specie and target_specie must be different species (D or H).')
    radius = default_radius(specie) if radius is None else radius
    target_radius = default_radius(target_specie) if target_radius is None else target_radius
    if any(not np.isfinite(r) or r <= 0 for r in (radius, target_radius)):
        raise ValueError('radius and target_radius must be positive.')
    # CLI and package builder parsers can produce int vs float values; identical
    # radii must have identical portable signatures.
    radius, target_radius = float(radius), float(target_radius)
    root = Path(datafolder) / dataset
    class_path = root / 'rsa_models' / 'by_class' / f'class_{rsa_class}.csv'
    pairs = read_pairs(class_path)
    result = dict(schema=SCHEMA, dataset=dataset, model=model, rsa_class=rsa_class,
                  radius=radius, target_radius=target_radius, specie=specie, target_specie=target_specie,
                  dis_method='mahalanobis', mah_fold='stim-wise',
                  rsa_method='kendall', kendall_variant='b', class_csv=str(class_path.resolve()),
                  pairs=pairs, species={})
    toolkit = Path(toolkit_dir or Path(__file__).parent)
    for species, selected, reference_mask in (
            (specie, participants, mask), (target_specie, target_participants, target_mask)):
        species_radius = radius if species == specie else target_radius
        species_mask_type = mask_type if species == specie else target_mask_type
        species_mask_type = species_mask_type.removesuffix('.nii.gz').removesuffix('.nii')
        config_path = root / 'config_files' / f'{species}_{model}.yaml'
        with open(config_path, encoding='utf-8') as stream:
            config = yaml.safe_load(stream)
        selected_ids = list(config['participants'] if selected is None else selected)
        if not selected_ids or len(set(selected_ids)) != len(selected_ids):
            raise ValueError(f'{species}: select a nonempty list of unique participants.')
        unknown = set(selected_ids) - set(config['participants'])
        if unknown:
            raise ValueError(f'{species}: participants absent from configuration: {sorted(unknown)}')
        if reference_mask is None:
            reference_mask = root / 'ROI' / species / f'{species_mask_type}.nii.gz'
            if species == 'D' and not reference_mask.exists():
                reference_mask = toolkit / 'Atlas' / 'Dog' / 'Nitzsche' / f'{species_mask_type}.nii.gz'
        else:
            species_mask_type = _mask_name(reference_mask)
        reference, _ = _grid(reference_mask)
        entries = []
        for participant in selected_ids:
            maps = []
            base = root / 'results' / 'RSA' / model / f'{species}-sub-{participant:02d}'
            for a, b in pairs:
                candidates = [base / f'r-{species_radius:g}_mahalanobis_{x}_{y}.nii.gz' for x, y in [(a, b), (b, a)]]
                path = next((p for p in candidates if p.is_file()), None)
                if path is None:
                    raise FileNotFoundError(f'Missing {species}-sub-{participant:02d} pair {a}/{b}: {candidates}')
                _check_grid(nib.load(str(path)), reference, path)
                maps.append(str(path.resolve()))
            entries.append(dict(participant=int(participant), maps=maps))
        result['species'][species] = dict(config=str(config_path.resolve()), mask=str(Path(reference_mask).resolve()),
                                          mask_type=species_mask_type, radius=species_radius,
                                          participants=entries)
    return result


def _resolve(root, path):
    return Path(root) / path


def _all_files(manifest):
    yield 'class_csv', manifest['class_csv']
    for species in sorted(manifest['species']):
        entry = manifest['species'][species]
        yield f'{species}/config', entry['config']
        yield f'{species}/mask', entry['mask']
        for participant in entry['participants']:
            for index, path in enumerate(participant['maps']):
                yield f"{species}/{participant['participant']}/{index}", path


def input_identity(manifest, root='.'):
    """Content-based identity is portable between workstation and Colab paths."""
    files = {key: sha256_file(_resolve(root, path)) for key, path in _all_files(manifest)}
    settings = {key: manifest[key] for key in ('schema', 'dataset', 'model', 'rsa_class', 'radius',
                                                'target_radius', 'specie', 'target_specie',
                                                'dis_method', 'mah_fold', 'rsa_method', 'kendall_variant', 'pairs')}
    settings['masks'] = {s: manifest['species'][s]['mask_type'] for s in sorted(manifest['species'])}
    settings['participants'] = {s: [p['participant'] for p in manifest['species'][s]['participants']]
                                for s in sorted(manifest['species'])}
    identity = dict(version=VERSION, code_sha256=sha256_file(__file__), settings=settings, files=files)
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(encoded.encode()).hexdigest(), identity


def group_rdms(manifest, species, root='.'):
    entry = manifest['species'][species]
    reference, inside = _grid(_resolve(root, entry['mask']))
    pairs = read_pairs(_resolve(root, manifest['class_csv']))
    if [list(p) for p in pairs] != [list(p) for p in manifest['pairs']]:
        raise ValueError('Manifest pairs do not match the ordered class CSV.')
    participants = entry['participants']
    ids = [p['participant'] for p in participants]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f'{species}: nonempty, unique participants required.')
    total = np.zeros((int(inside.sum()), len(pairs)), dtype=np.float64)
    valid = np.ones(len(total), dtype=bool)
    for participant in participants:
        if len(participant['maps']) != len(pairs):
            raise ValueError('Each participant must supply exactly the selected pairs.')
        print(f"[{species}] Loading participant {participant['participant']}", flush=True)
        for column, path in enumerate(participant['maps']):
            image = nib.load(str(_resolve(root, path)))
            _check_grid(image, reference, path)
            values = np.asarray(image.dataobj, dtype=np.float64)[inside]
            valid &= np.isfinite(values)
            total[:, column] += np.where(np.isfinite(values), values, 0)
    total /= len(participants)
    valid &= np.isfinite(total).all(axis=1) & (np.ptp(total, axis=1) > 0)
    total[~valid] = np.nan
    return reference, inside, total, valid


def pick_device(device='auto'):
    if device not in ('auto', 'cpu', 'cuda'):
        raise ValueError('device must be auto, cpu, or cuda.')
    if device == 'cpu':
        return 'cpu'
    # Avoid loading an optional PyTorch/OpenMP runtime on machines without a
    # CUDA driver (notably CPU-only Anaconda installations on Windows).
    import ctypes
    try:
        if os.name == 'nt':
            ctypes.WinDLL('nvcuda.dll')
        else:
            ctypes.CDLL('libcuda.so.1')
    except OSError:
        if device == 'cuda':
            raise RuntimeError('CUDA requested but the NVIDIA CUDA driver is unavailable.')
        return 'cpu'
    try:
        import torch
        available = torch.cuda.is_available()
    except ImportError:
        available = False
    if device == 'cuda' and not available:
        raise RuntimeError('CUDA requested but a CUDA-enabled PyTorch runtime is unavailable.')
    return 'cuda' if available else 'cpu'


def _signs(rdms):
    i, j = np.triu_indices(rdms.shape[1], 1)
    # Compare instead of subtracting to avoid overflow; invalid rows become zero.
    signs = (rdms[:, i] > rdms[:, j]).astype(np.int8) - (rdms[:, i] < rdms[:, j]).astype(np.int8)
    counts = np.count_nonzero(signs, axis=1)
    return signs, counts


def kendall_block(source, target, device='cpu'):
    """Exact sign-product numerator and tie-adjusted tau-b, stored as float32."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1] or source.shape[1] < 2:
        raise ValueError('RDM arrays must be 2D with the same, at least two, pair columns.')
    a, na = _signs(source)
    b, nb = _signs(target)
    if device == 'cuda':
        import torch
        # Products and partial sums are small integers, exactly represented in
        # float32. Disable TF32/reduced precision, then normalize in float64 on
        # CPU. This is much faster than float64 GEMM on Colab's T4/L4 GPUs.
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.inference_mode():
                x = torch.as_tensor(a, dtype=torch.float32, device='cuda')
                y = torch.as_tensor(b, dtype=torch.float32, device='cuda')
                numerator = (x @ y.T).cpu().numpy().astype(np.float64)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    else:
        numerator = a.astype(np.float64) @ b.astype(np.float64).T
    denominator = np.sqrt(na.astype(np.float64)[:, None] * nb[None, :])
    with np.errstate(divide='ignore', invalid='ignore'):
        result = numerator / denominator
    valid_a = np.isfinite(source).all(axis=1) & (na > 0)
    valid_b = np.isfinite(target).all(axis=1) & (nb > 0)
    result[~valid_a, :] = np.nan
    result[:, ~valid_b] = np.nan
    return result.astype(np.float32)


def _save_species(handle, species, image, inside, rdms, valid, participants):
    group = handle.create_group(species)
    ijk = np.argwhere(inside)
    for name, values in dict(ijk=ijk, xyz_mm=nib.affines.apply_affine(image.affine, ijk),
                             shape=image.shape, affine=image.affine, group_rdms=rdms,
                             valid=valid, participants=participants).items():
        group.create_dataset(name, data=values)
    group.create_dataset('nifti_header', data=np.frombuffer(image.header.binaryblock, dtype=np.uint8))
    group.attrs['header_class'] = type(image.header).__name__


def _run_matrix(manifest, output, *, root='.', device='auto', source_batch=128,
                target_batch=4096, checkpoint_callback=None, target_data=None):
    """Write/resume an HDF5 file. Callback receives a CLOSED partial file.

    Each tile is flushed before its completion bit. Resume uses the original
    tile layout even if new batch sizes are supplied. Completed files are reused
    only when their content identity matches. A single writer is required.
    """
    import h5py
    if source_batch <= 0 or target_batch <= 0:
        raise ValueError('Batch sizes must be positive.')
    if (manifest.get('schema') != SCHEMA or manifest.get('dis_method') != 'mahalanobis'
            or manifest.get('mah_fold') != 'stim-wise' or manifest.get('rsa_method') != 'kendall'
            or manifest.get('kendall_variant') != 'b'):
        raise ValueError('Step 16.2 requires Mahalanobis stim-wise inputs and Kendall tau-b.')
    source, target = manifest['specie'], manifest['target_specie']
    if source == target or {source, target} != {'D', 'H'}:
        raise ValueError('Source and target must be different species (D or H).')
    source_ids = [p['participant'] for p in manifest['species'][source]['participants']]
    if len(source_ids) != 1:
        raise ValueError('run_analysis requires exactly one source participant; use run_comparisons for a cohort.')
    device = pick_device(device)
    print(f'[16.2] device={device}; hashing inputs for safe resume', flush=True)
    signature, identity = input_identity(manifest, root)
    output = Path(output)
    partial = Path(str(output) + '.partial')
    output.parent.mkdir(parents=True, exist_ok=True)
    existing = output if output.exists() else partial
    if existing.exists():
        with h5py.File(existing, 'r') as handle:
            if handle.attrs.get('signature') != signature:
                raise ValueError('Incompatible checkpoint: inputs, settings, or software changed. Choose a new output path.')
            if existing == output:
                if not handle.attrs.get('complete', False):
                    raise ValueError('Final output is not marked complete.')
                print(f'[16.2] Reusing complete {output}', flush=True)
                return output
    if not partial.exists():
        groups = {source: group_rdms(manifest, source, root),
                  target: target_data if target_data is not None else group_rdms(manifest, target, root)}
        n_source, n_target = (len(groups[s][2]) for s in (source, target))
        print(f'[16.2] Matrix {n_source:,} x {n_target:,}; float32 uncompressed size {n_source * n_target * 4 / 2**30:.3f} GiB. '
              'Compression ratio depends on the data.', flush=True)
        # Initialize atomically: an interrupted initial load never masquerades as a checkpoint.
        initializing = Path(str(partial) + '.initializing')
        with h5py.File(initializing, 'w') as handle:
            handle.attrs.update(schema=SCHEMA, version=VERSION, signature=signature,
                                identity=json.dumps(identity, sort_keys=True), complete=False,
                                source_batch=source_batch, target_batch=target_batch, device=device,
                                source_specie=source, target_specie=target, source_participant=source_ids[0],
                                radius=manifest['radius'], target_radius=manifest['target_radius'],
                                matrix_axes=f'{source} participant searchlight centres, {target} group searchlight centres')
            handle.create_dataset('pairs', data=np.asarray(manifest['pairs'], dtype=object),
                                  dtype=h5py.string_dtype('utf-8'))
            for species, (image, inside, rdms, valid) in groups.items():
                _save_species(handle, species, image, inside, rdms, valid,
                              identity['settings']['participants'][species])
                handle[species].attrs.update(reduction='participant' if species == source else 'equal-participant-mean',
                                             radius=manifest['species'][species]['radius'],
                                             mask_type=manifest['species'][species]['mask_type'])
            handle.create_dataset('correlations', shape=(n_source, n_target), dtype='f4', fillvalue=np.nan,
                                  chunks=(min(source_batch, n_source), min(target_batch, n_target)),
                                  compression='gzip', compression_opts=4, shuffle=True)
            handle.create_dataset('completed_blocks', shape=((n_source + source_batch - 1) // source_batch,
                                                             (n_target + target_batch - 1) // target_batch),
                                  dtype='bool', fillvalue=False)
        os.replace(initializing, partial)
        del groups
    with h5py.File(partial, 'r') as handle:
        # Historical v2 files used species names for row/column tile sizes.
        source_batch = int(handle.attrs.get('source_batch', handle.attrs.get('dog_batch', source_batch)))
        target_batch = int(handle.attrs.get('target_batch', handle.attrs.get('human_batch', target_batch)))
        n_source, n_target = handle['correlations'].shape
    for row_index, start in enumerate(range(0, n_source, source_batch)):
        # Close after each source batch so callbacks can safely publish a checkpoint.
        with h5py.File(partial, 'r+') as handle:
            if handle['completed_blocks'][row_index, :].all():
                continue
            source_rdms = handle[f'{source}/group_rdms'][start:start + source_batch]
            for column_index, target_start in enumerate(range(0, n_target, target_batch)):
                if handle['completed_blocks'][row_index, column_index]:
                    continue
                target_rdms = handle[f'{target}/group_rdms'][target_start:target_start + target_batch]
                values = kendall_block(source_rdms, target_rdms, device=device)
                handle['correlations'][start:start + len(source_rdms), target_start:target_start + len(target_rdms)] = values
                handle.flush()
                handle['completed_blocks'][row_index, column_index] = True
                handle.flush()
            print(f'[16.2] {source}-sub-{source_ids[0]:02d} centres {min(start + source_batch, n_source):,}/{n_source:,}', flush=True)
        if checkpoint_callback:
            checkpoint_callback(partial)
    with h5py.File(partial, 'r+') as handle:
        if not handle['completed_blocks'][:].all():
            raise RuntimeError('Incomplete matrix: refusing to publish a final file.')
        handle.attrs['complete'] = True
    os.replace(partial, output)
    print(f'[16.2] Complete: {output}', flush=True)
    return output


def participant_manifests(manifest):
    """Select one source at a time, retaining every selected target participant."""
    source = manifest['specie']
    for participant in manifest['species'][source]['participants']:
        single = copy.deepcopy(manifest)
        single['species'][source]['participants'] = [copy.deepcopy(participant)]
        yield participant['participant'], single


def relative_output_path(manifest, participant):
    """Path relative to the parent data directory, shared by workstation/Colab."""
    source, target = manifest['specie'], manifest['target_specie']
    mask = manifest['species'][source]['mask_type']
    name = (f"{mask}-r-{manifest['radius']:g}_mahalanobis_kendall_"
            f"vs_{target}-r-{manifest['target_radius']:g}.HDF5")
    return (Path(manifest['dataset']) / 'results' / 'RSA_cross-specie' / manifest['model'] /
            f"class_{manifest['rsa_class']}" / f'{source}-sub-{participant:02d}' / name)


class _Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, text):
        self.console.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.console.flush()
        self.log.flush()


def run_analysis(manifest, output, **kwargs):
    """One source participant vs target group, producing an HDF5 and a log only."""
    import sys
    import traceback
    from datetime import datetime, timezone
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.log').open('a', encoding='utf-8') as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(_Tee(sys.stderr, log)):
            print(f'\nStep 16.2 v{VERSION}; UTC {datetime.now(timezone.utc).isoformat()}', flush=True)
            print(json.dumps({key: manifest.get(key) for key in ('dataset', 'model', 'rsa_class', 'specie',
                              'target_specie', 'radius', 'target_radius')}, sort_keys=True), flush=True)
            print('Participants:', {s: [p['participant'] for p in manifest['species'][s]['participants']]
                                     for s in (manifest['specie'], manifest['target_specie'])}, flush=True)
            try:
                return _run_matrix(manifest, output, **kwargs)
            except BaseException:
                traceback.print_exc()
                raise


def run_comparisons(manifest, output_root, *, root='.', **kwargs):
    """Run all selected source participants against a single cached target mean."""
    target_data = group_rdms(manifest, manifest['target_specie'], root)
    outputs = []
    for participant, single in participant_manifests(manifest):
        output = Path(output_root) / relative_output_path(manifest, participant)
        outputs.append(run_analysis(single, output, root=root, target_data=target_data, **kwargs))
    return outputs


def group_cross_similarity(root, dataset, model, rsa_class, specie='D',
                           target_specie='H', radius=None, target_radius=None,
                           mask_type='b_GreyMatter2mmB', participants=None, *,
                           target_participants=None, target_mask_type=None,
                           mask=None, target_mask=None,
                           source_batch=128, target_batch=4096, verbose=False):
    """Stream equal-participant arithmetic means of step-16.2 matrices.

    Axes, comparison pairs, target cohort and target RDMs must agree exactly.
    NaNs propagate; no Fisher transform is applied to Kendall correlations.
    The output retains geometry for CorrelationReader.target_map().
    """
    import h5py
    if {specie, target_specie} != {'D', 'H'} or not rsa_class:
        raise ValueError('Select different D/H species and an rsa_class.')
    if source_batch <= 0 or target_batch <= 0:
        raise ValueError('Batch sizes must be positive.')
    radius = default_radius(specie) if radius is None else radius
    target_radius = default_radius(target_specie) if target_radius is None else target_radius
    mask_type = _mask_name(mask if mask is not None else mask_type)
    if target_mask is not None:
        target_mask_type = _mask_name(target_mask)
    folder = Path(root) / dataset / 'results' / 'RSA_cross-specie' / model / f'class_{rsa_class}'
    name = f'{mask_type}-r-{radius:g}_mahalanobis_kendall_vs_{target_specie}-r-{target_radius:g}.HDF5'
    if participants is None:
        participants = sorted(int(p.parent.name.split('-sub-')[1])
                              for p in folder.glob(f'{specie}-sub-*/{name}'))
    else:
        participants = list(participants)
    if not participants or len(set(participants)) != len(participants):
        raise ValueError('Require a nonempty, unique source participant selection.')
    paths = [folder / f'{specie}-sub-{p:02d}' / name for p in participants]
    output = folder / f'{specie}-group' / name
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(output) + '.partial')
    # Keep all handles open for tiled reads, never load an entire matrix.
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(CorrelationReader(p)).file for p in paths]
        first = files[0]
        shape = first['correlations'].shape
        metadata = ['pairs'] + [f'{s}/{key}' for s in (specie, target_specie)
                               for key in ('ijk', 'shape', 'affine')]
        metadata += [f'{target_specie}/{key}' for key in ('participants', 'group_rdms')]
        for participant, handle in zip(participants, files):
            if (handle.attrs.get('source_specie') != specie
                    or handle.attrs.get('target_specie') != target_specie
                    or handle.attrs.get('source_participant') != participant
                    or handle.attrs.get('radius') != radius
                    or handle.attrs.get('target_radius') != target_radius
                    or handle['correlations'].shape != shape):
                raise ValueError(f'Incompatible step-16.2 settings: participant {participant}')
            # Validate mask selection from saved metadata; original mask files
            # need not be available when aggregating on Colab.
            identity = json.loads(handle.attrs.get('identity', '{}'))
            for species, expected_mask in ((specie, mask_type), (target_specie, target_mask_type)):
                saved_mask = handle[species].attrs.get('mask_type')
                if saved_mask is None:
                    saved_mask = identity.get('settings', {}).get('masks', {}).get(species)
                if expected_mask is not None and saved_mask != _mask_name(expected_mask):
                    raise ValueError(f'Incompatible {species} mask selection: participant {participant}')
            for key in metadata:
                a, b = first[key][:], handle[key][:]
                equal = (np.array_equal(a, b, equal_nan=True) if a.dtype.kind in 'fc'
                         else np.array_equal(a, b))
                if not equal:
                    raise ValueError(f'Incompatible {key}: participant {participant}')
            if target_participants is not None and set(handle[f'{target_specie}/participants'][:]) != set(target_participants):
                raise ValueError('Target participant selection differs from step 16.2.')
        with h5py.File(partial, 'w') as result:
            result.attrs.update(dict(first.attrs))
            for key in ('signature', 'identity', 'source_participant', 'dog_batch', 'human_batch'):
                if key in result.attrs:
                    del result.attrs[key]
            result.attrs.update(complete=False, source_batch=source_batch, target_batch=target_batch,
                                reduction='equal-participant-arithmetic-mean',
                                nan_policy='propagate', input_files=json.dumps([str(p) for p in paths]))
            first.copy('pairs', result)
            for species in (specie, target_specie):
                group = result.create_group(species)
                group.attrs.update(dict(first[species].attrs))
                for key in ('ijk', 'xyz_mm', 'shape', 'affine', 'nifti_header'):
                    first.copy(f'{species}/{key}', group, name=key)
                group.create_dataset('participants', data=participants if species == specie
                                     else first[f'{species}/participants'][:])
                if species == specie:
                    group.attrs['reduction'] = 'equal-participant-mean-of-correlations'
            matrix = result.create_dataset('correlations', shape=shape, dtype='f4',
                        chunks=(min(source_batch, shape[0]), min(target_batch, shape[1])),
                        compression='gzip', shuffle=True)
            for i in range(0, shape[0], source_batch):
                for j in range(0, shape[1], target_batch):
                    index = (slice(i, i + source_batch), slice(j, j + target_batch))
                    total = np.zeros(matrix[index].shape, dtype=np.float64)
                    for handle in files:
                        total += handle['correlations'][index]
                    matrix[index] = total / len(files)
                if verbose:
                    print(f'[16.3] averaged {min(i + source_batch, shape[0])}/{shape[0]} source voxels', flush=True)
            result.attrs['complete'] = True
    os.replace(partial, output)
    return output


def summarize_group_cross_similarity(path, output_dir=None, *, source_batch=128,
                                     target_batch=4096, verbose=False):
    """Step 16.4: source-space threshold counts and mean correlations.

    Twenty linearly spaced thresholds include the finite matrix minimum and
    maximum. Counts use strict >, ignoring undefined correlations. Row means
    use finite entries only, with NaN for rows without finite data. Outside
    the saved source mask is NaN. Read the matrix in bounded tiles.
    """
    if source_batch <= 0 or target_batch <= 0:
        raise ValueError('Batch sizes must be positive.')
    path = Path(path)
    output_dir = Path(output_dir) if output_dir is not None else path.parent.parent / 'mean'
    with CorrelationReader(path) as reader:
        handle = reader.file
        if handle.attrs.get('reduction') != 'equal-participant-arithmetic-mean':
            raise ValueError('Step 16.4 requires a step-16.3 group average.')
        source = handle.attrs['source_specie']
        target = handle.attrs['target_specie']
        if source == target or {source, target} != {'D', 'H'}:
            raise ValueError('Step 16.4 requires different source and target species (D or H).')
        low, high = np.inf, -np.inf
        for _, _, values in reader.iter_blocks(source_batch, target_batch):
            finite = values[np.isfinite(values)]
            if finite.size:
                low = min(low, float(finite.min()))
                high = max(high, float(finite.max()))
        if not np.isfinite(low):
            raise ValueError('Group matrix has no finite correlations.')
        thresholds = np.linspace(low, high, 20)
        n_source, n_target = handle['correlations'].shape
        counts = np.zeros((20, n_source), dtype=np.int64)
        totals = np.zeros(n_source, dtype=np.float64)
        available = np.zeros(n_source, dtype=np.int64)
        for rows, _, values in reader.iter_blocks(source_batch, target_batch):
            values = values.astype(np.float64)
            finite = np.isfinite(values)
            totals[rows] += np.where(finite, values, 0).sum(axis=1, dtype=np.float64)
            available[rows] += finite.sum(axis=1)
            for index, threshold in enumerate(thresholds):
                counts[index, rows] += (finite & (values > threshold)).sum(axis=1)
        means = np.full(n_source, np.nan)
        np.divide(totals, available, out=means, where=available > 0)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = path.stem
        mean_path = output_dir / f'{stem}_mean.nii.gz'
        nib.save(reader.image(source, means), mean_path)
        outputs = []
        for index, threshold in enumerate(thresholds, start=1):
            output = output_dir / f'{stem}_threshold-{index:02d}_count.nii.gz'
            nib.save(reader.image(source, counts[index - 1]), output)
            outputs.append(output)
        provenance = dict(input_file=str(path), output_mean_file=str(mean_path),
                          thresholds=thresholds.tolist(), correlation_min=low,
                          correlation_max=high, comparison='strictly greater than',
                          finite_target_counts=available.tolist(),
                          source_specie=source, target_specie=target,
                          number_target_voxels=n_target, nan_policy='ignore nonfinite correlations; empty row mean is NaN',
                          output_count_files=[str(p) for p in outputs])
        with (output_dir / f'{stem}_summary.json').open('w', encoding='utf-8') as stream:
            json.dump(provenance, stream, indent=2)
        if verbose:
            print(f'[16.4] Saved 20 count maps and mean map to {output_dir}', flush=True)
    return [mean_path, *outputs]


class CorrelationReader:
    """Context-managed, lazy reader for a complete step-16.2 result."""
    def __init__(self, path):
        import h5py
        self.file = h5py.File(path, 'r')
        if self.file.attrs.get('schema') not in (SCHEMA, LEGACY_SCHEMA) or not self.file.attrs.get('complete', False):
            self.file.close()
            raise ValueError('Expected a complete step-16.2 HDF5 file.')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.file.close()

    def iter_blocks(self, source_batch=128, target_batch=4096):
        if source_batch <= 0 or target_batch <= 0:
            raise ValueError('Batch sizes must be positive.')
        matrix = self.file['correlations']
        for row in range(0, matrix.shape[0], source_batch):
            for column in range(0, matrix.shape[1], target_batch):
                rows = slice(row, min(row + source_batch, matrix.shape[0]))
                columns = slice(column, min(column + target_batch, matrix.shape[1]))
                yield rows, columns, matrix[rows, columns]

    def image(self, species, values):
        """Reconstruct values at saved centres; outside-mask locations are NaN."""
        group = self.file[species]
        ijk = group['ijk'][:]
        values = np.asarray(values)
        if values.shape != (len(ijk),):
            raise ValueError('Expected one value per saved searchlight centre.')
        data = np.full(tuple(group['shape'][:]), np.nan, dtype=np.float32)
        data[tuple(ijk.T)] = values
        header_class = getattr(nib, group.attrs['header_class'])
        header = header_class(binaryblock=group['nifti_header'][:].tobytes())
        header.set_data_dtype(np.float32)
        header.set_slope_inter(1, 0)
        image_class = nib.Nifti2Image if isinstance(header, nib.Nifti2Header) else nib.Nifti1Image
        return image_class(data, group['affine'][:], header)

    def target_map(self, source_ijk):
        """Return a target-space NIfTI for an exact source voxel, without saving."""
        source = self.file.attrs.get('source_specie', 'D')
        target = self.file.attrs.get('target_specie', 'H')
        coordinate = np.asarray(source_ijk)
        if coordinate.shape != (3,) or not np.isfinite(coordinate).all() or not np.equal(coordinate, np.rint(coordinate)).all():
            raise ValueError('source_ijk must contain three integer voxel indices.')
        matches = np.flatnonzero((self.file[f'{source}/ijk'][:] == coordinate).all(axis=1))
        if len(matches) != 1:
            raise ValueError('Source coordinate is outside the saved mask.')
        return self.image(target, self.file['correlations'][matches[0], :])

    def human_map(self, dog_ijk):
        """Compatibility alias for D-to-H results, including legacy group files."""
        if self.file.attrs.get('source_specie', 'D') != 'D' or self.file.attrs.get('target_specie', 'H') != 'H':
            raise ValueError('human_map requires D-to-H data; use target_map for other directions.')
        return self.target_map(dog_ijk)


def add_arguments(parser):
    """Step-specific flags shared by the local pipeline and package builder."""
    parser.add_argument('--mask', help='Source reference mask NIfTI path')
    parser.add_argument('--target_mask', help='Target reference mask NIfTI path')
    parser.add_argument('--target_specie', choices=['D', 'H'], default='H', help='Species whose participant dissimilarities are averaged')
    parser.add_argument('--target_radius', type=float, help='Target searchlight radius; defaults to H=4, D=3')
    parser.add_argument('--target_participants', nargs='+', type=int)
    parser.add_argument('--target_mask_type', default='b_GreyMatter2mmB')
    parser.add_argument('--cross_species_output', help='Output ROOT directory; dataset/results/RSA_cross-specie/... is created inside it')
    parser.add_argument('--cross_species_device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--source_batch', type=int, default=128,
                        help='Source mask centres per tile (also for H-to-D)')
    parser.add_argument('--target_batch', type=int, default=4096,
                        help='Target mask centres per tile (also for H-to-D)')


def run_from_pipeline(args):
    """Compatibility adapter; new pipeline dispatch calls rsa_utils directly."""
    import rsa_utils
    from scheduler.paths import get_paths
    datafolder = args.datafolder or get_paths()[0]
    return rsa_utils.calculate_voxelwise_cross_similarity(
        datafolder=datafolder, dataset=args.dataset, model=args.model,
        participants=args.participants_forced or None, specie=args.specie,
        target_specie=args.target_specie, radius=args.radius, target_radius=args.target_radius,
        rsa_class=args.rsa_class, dis_method=args.dis_method, rsa_method=args.rsa_method,
        mah_fold=args.mah_fold, mask_type=args.mask_type, target_mask_type=args.target_mask_type,
        target_participants=args.target_participants, mask=args.mask,
        target_mask=args.target_mask, toolkit_dir=args.toolkit_dir,
        output_root=args.cross_species_output, device=args.cross_species_device,
        source_batch=args.source_batch, target_batch=args.target_batch, verbose=args.verbose)

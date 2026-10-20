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


def discover_inputs(datafolder, dataset='EmoC', model='basic-block',
                    rsa_class='all-mahalanobis', radius=3, *,
                    dog_participants=None, human_participants=None,
                    dog_mask=None, human_mask=None, toolkit_dir=None,
                    specie='D', target_specie='H', target_radius=None,
                    source_participants=None, target_participants=None,
                    mask_type='b_GreyMatter2mmB', target_mask_type='b_GreyMatter2mmB'):
    """Resolve existing pooled step-1 maps, using each species' own config.

    Returned paths are absolute; packaged manifests use equivalent relative paths.
    Dataset ROI masks take precedence; the dog atlas is a legacy fallback.
    """
    if specie not in ('D', 'H') or target_specie not in ('D', 'H') or specie == target_specie:
        raise ValueError('specie and target_specie must be different species (D or H).')
    if target_radius is None:
        target_radius = 4 if target_specie == 'H' else 3
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
    for species, selected, mask in [('D', dog_participants, dog_mask), ('H', human_participants, human_mask)]:
        requested = source_participants if species == specie else target_participants
        if requested is not None:
            if selected is not None and list(selected) != list(requested):
                raise ValueError(f'Conflicting participant selections for {species}.')
            selected = requested
        species_radius = radius if species == specie else target_radius
        species_mask_type = mask_type if species == specie else target_mask_type
        species_mask_type = species_mask_type.removesuffix('.nii.gz').removesuffix('.nii')
        config_path = root / 'config_files' / f'{species}_{model}.yaml'
        with open(config_path, encoding='utf-8') as stream:
            config = yaml.safe_load(stream)
        participants = list(config['participants'] if selected is None else selected)
        if not participants or len(set(participants)) != len(participants):
            raise ValueError(f'{species}: select a nonempty list of unique participants.')
        unknown = set(participants) - set(config['participants'])
        if unknown:
            raise ValueError(f'{species}: participants absent from configuration: {sorted(unknown)}')
        if mask is None:
            mask = root / 'ROI' / species / f'{species_mask_type}.nii.gz'
            if species == 'D' and not mask.exists():
                mask = toolkit / 'Atlas' / 'Dog' / 'Nitzsche' / f'{species_mask_type}.nii.gz'
        else:
            species_mask_type = Path(mask).name.removesuffix('.nii.gz').removesuffix('.nii')
        reference, _ = _grid(mask)
        entries = []
        for participant in participants:
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
        result['species'][species] = dict(config=str(config_path.resolve()), mask=str(Path(mask).resolve()),
                                          mask_type=species_mask_type, radius=species_radius,
                                          participants=entries)
    return result


def _resolve(root, path):
    return Path(root) / path


def _all_files(manifest):
    yield 'class_csv', manifest['class_csv']
    for species in ('D', 'H'):
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
    settings['masks'] = {s: manifest['species'][s]['mask_type'] for s in ('D', 'H')}
    settings['participants'] = {s: [p['participant'] for p in manifest['species'][s]['participants']]
                                for s in ('D', 'H')}
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


def kendall_block(dog, human, device='cpu'):
    """Exact sign-product numerator and tie-adjusted tau-b, stored as float32."""
    dog = np.asarray(dog, dtype=np.float64)
    human = np.asarray(human, dtype=np.float64)
    if dog.ndim != 2 or human.ndim != 2 or dog.shape[1] != human.shape[1] or dog.shape[1] < 2:
        raise ValueError('RDM arrays must be 2D with the same, at least two, pair columns.')
    a, na = _signs(dog)
    b, nb = _signs(human)
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
    valid_a = np.isfinite(dog).all(axis=1) & (na > 0)
    valid_b = np.isfinite(human).all(axis=1) & (nb > 0)
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


def _run_matrix(manifest, output, *, root='.', device='auto', dog_batch=128,
                human_batch=4096, checkpoint_callback=None, target_data=None):
    """Write/resume an HDF5 file. Callback receives a CLOSED partial file.

    Each tile is flushed before its completion bit. Resume uses the original
    tile layout even if new batch sizes are supplied. Completed files are reused
    only when their content identity matches. A single writer is required.
    """
    import h5py
    if dog_batch <= 0 or human_batch <= 0:
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
        nd, nh = (len(groups[s][2]) for s in (source, target))
        print(f'[16.2] Matrix {nd:,} x {nh:,}; float32 uncompressed size {nd * nh * 4 / 2**30:.3f} GiB. '
              'Compression ratio depends on the data.', flush=True)
        # Initialize atomically: an interrupted initial load never masquerades as a checkpoint.
        initializing = Path(str(partial) + '.initializing')
        with h5py.File(initializing, 'w') as handle:
            handle.attrs.update(schema=SCHEMA, version=VERSION, signature=signature,
                                identity=json.dumps(identity, sort_keys=True), complete=False,
                                dog_batch=dog_batch, human_batch=human_batch, device=device,
                                source_specie=source, target_specie=target, source_participant=source_ids[0],
                                radius=manifest['radius'], target_radius=manifest['target_radius'],
                                matrix_axes=f'{source} participant searchlight centres, {target} group searchlight centres')
            handle.create_dataset('pairs', data=np.asarray(manifest['pairs'], dtype=object),
                                  dtype=h5py.string_dtype('utf-8'))
            for species, (image, inside, rdms, valid) in groups.items():
                _save_species(handle, species, image, inside, rdms, valid,
                              identity['settings']['participants'][species])
                handle[species].attrs.update(reduction='participant' if species == source else 'equal-participant-mean',
                                             radius=manifest['species'][species]['radius'])
            handle.create_dataset('correlations', shape=(nd, nh), dtype='f4', fillvalue=np.nan,
                                  chunks=(min(dog_batch, nd), min(human_batch, nh)),
                                  compression='gzip', compression_opts=4, shuffle=True)
            handle.create_dataset('completed_blocks', shape=((nd + dog_batch - 1) // dog_batch,
                                                             (nh + human_batch - 1) // human_batch),
                                  dtype='bool', fillvalue=False)
        os.replace(initializing, partial)
        del groups
    with h5py.File(partial, 'r') as handle:
        dog_batch, human_batch = int(handle.attrs['dog_batch']), int(handle.attrs['human_batch'])
        nd, nh = handle['correlations'].shape
    for di, start in enumerate(range(0, nd, dog_batch)):
        # Close after each dog batch so callbacks can safely publish a checkpoint.
        with h5py.File(partial, 'r+') as handle:
            if handle['completed_blocks'][di, :].all():
                continue
            dog = handle[f'{source}/group_rdms'][start:start + dog_batch]
            for hi, hstart in enumerate(range(0, nh, human_batch)):
                if handle['completed_blocks'][di, hi]:
                    continue
                human = handle[f'{target}/group_rdms'][hstart:hstart + human_batch]
                values = kendall_block(dog, human, device=device)
                handle['correlations'][start:start + len(dog), hstart:hstart + len(human)] = values
                handle.flush()
                handle['completed_blocks'][di, hi] = True
                handle.flush()
            print(f'[16.2] {source}-sub-{source_ids[0]:02d} centres {min(start + dog_batch, nd):,}/{nd:,}', flush=True)
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
                                     for s in ('D', 'H')}, flush=True)
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

    def iter_blocks(self, dog_batch=128, human_batch=4096):
        if dog_batch <= 0 or human_batch <= 0:
            raise ValueError('Batch sizes must be positive.')
        matrix = self.file['correlations']
        for d in range(0, matrix.shape[0], dog_batch):
            for h in range(0, matrix.shape[1], human_batch):
                ds, hs = slice(d, min(d + dog_batch, matrix.shape[0])), slice(h, min(h + human_batch, matrix.shape[1]))
                yield ds, hs, matrix[ds, hs]

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
    parser.add_argument('--dog_participants', nargs='+', type=int)
    parser.add_argument('--human_participants', nargs='+', type=int)
    parser.add_argument('--dog_mask', help='Dog reference mask NIfTI path')
    parser.add_argument('--human_mask', help='Human reference mask NIfTI path')
    parser.add_argument('--target_specie', choices=['D', 'H'], default='H', help='Species whose participant dissimilarities are averaged')
    parser.add_argument('--target_radius', type=float, help='Target searchlight radius; defaults to H=4, D=3')
    parser.add_argument('--target_participants', nargs='+', type=int)
    parser.add_argument('--target_mask_type', default='b_GreyMatter2mmB')
    parser.add_argument('--cross_species_output', help='Output ROOT directory; dataset/results/RSA_cross-specie/... is created inside it')
    parser.add_argument('--cross_species_device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--source_batch', '--dog_batch', dest='dog_batch', type=int, default=128,
                        help='Source mask centres per tile (also for H-to-D)')
    parser.add_argument('--target_batch', '--human_batch', dest='human_batch', type=int, default=4096,
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
        target_participants=args.target_participants, dog_participants=args.dog_participants,
        human_participants=args.human_participants, dog_mask=args.dog_mask,
        human_mask=args.human_mask, toolkit_dir=args.toolkit_dir,
        output_root=args.cross_species_output, device=args.cross_species_device,
        source_batch=args.dog_batch, target_batch=args.human_batch, verbose=args.verbose)

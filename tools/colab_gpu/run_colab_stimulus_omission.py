"""Run beta-omission steps 1 and 11.2 from a cohort of participant packages."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import zipfile

import nibabel as nib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
# Repository execution finds the shared core at the root; packaged/Drive
# execution carries it beside this script instead.
REPO = HERE.parent.parent
if (REPO / 'stimulus_contribution.py').is_file() and str(REPO) not in sys.path:
    sys.path.append(str(REPO))
import gpu_rsa as gpu
import gpu_stimulus_omission as kernels


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def image_from_zip(archive, name):
    return nib.Nifti1Image.from_bytes(gzip.decompress(archive.read(name)))


def _safe_name(name):
    if ('\\' in name or ':' in name or name.startswith('/')
            or '..' in PurePosixPath(name).parts):
        raise ValueError(f'Unsafe archive member: {name}')


def discover_packages(packages_dir, specie, dataset, model, participants=None,
                      minimum_participants=2):
    """Preflight the full cohort, masks, class pairs, and package identities."""
    found = {}
    common = None
    reference = None
    mask = None
    fields = ('dataset', 'model', 'specie', 'task', 'radius', 'mask_type', 'dis_method',
              'mah_fold', 'rsa_method', 'rsa_class', 'stim_types', 'categories', 'pairs',
              'participants', 'allow_space_mismatch')
    for path in sorted(Path(packages_dir).glob('pkg_*.zip')):
        with zipfile.ZipFile(path) as archive:
            manifest = json.loads(archive.read('manifest.json'))
            if (manifest.get('specie'), manifest.get('dataset'), manifest.get('model')) != (specie, dataset, model):
                continue
            if not manifest.get('leave_one_stimulus_out'):
                continue
            if manifest.get('omission_version') != kernels.VERSION:
                raise ValueError(f'{path.name}: rebuild this beta-omission package with the current code.')
            if manifest['dis_method'] != 'mahalanobis' or manifest['mah_fold'] != 'stim-wise':
                raise ValueError('Colab beta omissions require Mahalanobis stim-wise inputs.')
            sub = int(manifest['sub_N'])
            if participants is not None and sub not in participants:
                continue
            if sub in found:
                raise ValueError(f'Duplicate participant packages: {specie}-sub-{sub:02d}')
            signature = {key: manifest[key] for key in fields}
            signature['comparison_pairs'] = manifest.get('comparison_pairs', manifest['pairs'])
            signature['code'] = {name: [archive.getinfo('code/' + name).CRC,
                                        archive.getinfo('code/' + name).file_size]
                                 for name in ('gpu_rsa.py', 'gpu_stimulus_omission.py',
                                              'run_colab_stimulus_omission.py', 'stimulus_contribution.py')}
            if common is not None and signature != common:
                raise ValueError(f'Package parameters or cohorts differ: {path.name}')
            common = signature
            names = [info.filename for info in archive.infolist()]
            if len(names) != len(set(names)):
                raise ValueError(f'Duplicate ZIP members: {path.name}')
            for name in names:
                _safe_name(name)
            pairs = [(row['cat1'].strip(), row['cat2'].strip()) for row in
                     csv.DictReader(io.StringIO(archive.read(manifest['class_csv']).decode('utf-8-sig')))]
            comparison_pairs = manifest.get('comparison_pairs', manifest['pairs'])
            if pairs != [tuple(pair) for pair in comparison_pairs] or len(pairs) < 2:
                raise ValueError(f'Class CSV and manifest pairs differ: {path.name}')
            for stimulus in manifest['stim_types']:
                if not stimulus or any(c in stimulus for c in '/\\:') or stimulus in ('.', '..'):
                    raise ValueError(f'Invalid stimulus label: {stimulus}')
            identities = [(int(e['session']), int(e['run_N'])) for e in manifest['runs']]
            if not identities or len(set(identities)) != len(identities):
                raise ValueError(f'Empty or duplicate session/run list: {path.name}')
            if len(manifest['stim_types']) != len(set(manifest['stim_types'])):
                raise ValueError(f'Duplicate beta labels: {path.name}')
            mask_arc = f'data/{dataset}/ROI/{specie}/{manifest["mask_type"]}.nii.gz'
            image = image_from_zip(archive, mask_arc)
            current_mask = np.isfinite(image.get_fdata()) & (image.get_fdata() != 0)
            if image.ndim != 3 or not current_mask.any():
                raise ValueError('The searchlight mask must be a nonempty 3D mask.')
            if reference is not None:
                gpu.check_same_space(('cohort mask', reference), [(path.name, image)], strict=True)
                if not np.array_equal(mask, current_mask):
                    raise ValueError(f'Package mask contents differ: {path.name}')
            else:
                reference, mask = image, current_mask
            # CRC/size fingerprints change when a package's inputs or code change.
            runtime_files = [Path(__file__), Path(kernels.__file__), Path(gpu.__file__),
                             Path(sys.modules['stimulus_contribution'].__file__)]
            identity = digest(dict(package=[(i.filename, i.CRC, i.file_size) for i in archive.infolist()],
                                   runtime=[hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_files]))
            found[sub] = dict(path=path, manifest=manifest, identity=identity)
    if not found:
        raise ValueError('No matching beta-omission packages. Build with create_package.py --leave_one_stimulus_out.')
    expected = sorted(participants if participants is not None else common['participants'])
    if len(expected) < minimum_participants or len(expected) != len(set(expected)):
        raise ValueError(f'Provide at least {minimum_participants} unique participants.')
    if sorted(found) != expected:
        raise ValueError(f'Participant packages do not cover the cohort: missing={sorted(set(expected) - set(found))}, '
                         f'extra={sorted(set(found) - set(expected))}. Set PARTICIPANTS only for an intentional subset.')
    return found, reference, mask


def omission_folder(manifest, unit, stimulus):
    entry = manifest['runs'][unit]
    root = (f"{manifest['dataset']}/results/RSA_same/{manifest['model']}/class_{manifest['rsa_class']}/"
            f"leave-one-out/by_stim/{manifest['specie']}-sub-{manifest['sub_N']:02d}")
    if sum(int(e['run_N']) == int(entry['run_N']) for e in manifest['runs']) > 1:
        root += f"/ses-{int(entry['session']):02d}"
    return f"{root}/run-{int(entry['run_N'])}/{stimulus}"


def records_for(package, out):
    manifest = package['manifest']
    records = []
    for unit, entry in enumerate(manifest['runs']):
        for index, stimulus in enumerate(manifest['stim_types']):
            receipt = dict(version=kernels.VERSION, package=package['identity'], unit=unit, stimulus=stimulus)
            folder = omission_folder(manifest, unit, stimulus)
            names = [f"{folder}/r-{manifest['radius']}_mahalanobis_{a}_{b}.nii.gz" for a, b in manifest['pairs']]
            name = (f"result_step1_omission_{manifest['dataset']}_{manifest['model']}_class-{manifest['rsa_class']}_"
                    f"{manifest['specie']}-sub-{manifest['sub_N']:02d}_ses-{int(entry['session']):02d}_"
                    f"run-{int(entry['run_N']):02d}_{stimulus}_{digest(receipt)[:12]}.zip")
            records.append(dict(unit=unit, stimulus_index=index, stimulus=stimulus,
                                run=int(entry['run_N']), folder=folder, members=names,
                                path=out / name, receipt=receipt))
    return records


def complete(path, receipt, members):
    try:
        with zipfile.ZipFile(path) as archive:
            return (json.loads(archive.comment) == receipt
                    and set(members).issubset(archive.namelist()))
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def publish(path, receipt, writer, work):
    """Finish locally, copy to Drive as .part, then publish the completed ZIP."""
    local = work / 'publish.zip'
    with zipfile.ZipFile(local, 'w', zipfile.ZIP_STORED, allowZip64=True) as archive:
        writer(archive)
        archive.comment = json.dumps(receipt, sort_keys=True).encode()
    temporary = Path(str(path) + '.part')
    shutil.copyfile(local, temporary)
    os.replace(temporary, path)
    local.unlink()


def write_image(archive, name, values, mask, reference, outside=np.nan):
    volume = np.full(mask.shape, outside, dtype=np.float64)
    volume[mask] = values
    content = nib.Nifti1Image(volume, reference.affine).to_bytes()
    archive.writestr(name, gzip.compress(content, compresslevel=1, mtime=0))


def _remove_scratch(path, work):
    path, work = Path(path).resolve(), Path(work).resolve()
    if path == work or not path.is_relative_to(work):
        raise ValueError(f'Scratch cleanup escapes the work directory: {path}')
    if path.is_dir():
        shutil.rmtree(path)
    elif path.is_file():
        path.unlink()


def _extract_betas(package, destination):
    manifest = package['manifest']
    with zipfile.ZipFile(package['path']) as archive:
        for entry in manifest['runs']:
            print(f"  extracting session {entry['session']}, run {entry['run_N']}", flush=True)
            for stimulus in manifest['stim_types']:
                name = (f"data/{manifest['dataset']}/results/GLM/{manifest['model']}/"
                        f"{manifest['specie']}-sub-{manifest['sub_N']:02d}/"
                        f"ses-{int(entry['session']):02d}_task-{manifest['task']}_run-{int(entry['run_N']):02d}/"
                        f'beta_{stimulus}.nii.gz')
                _safe_name(name)
                target = destination / name
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError(f'Unsafe beta path: {name}')
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)


def run(packages_dir, out_dir, *, specie='D', dataset='EmoC', model='basic-block',
        participants=None, steps=(1, 11.2), work='/content/stimulus_omission_work',
        device=None, step1_batch=256, voxel_batch=128, comparison_batch=4096,
        force=False, min_percentage_available=1.0, allow_single_participant=False):
    if not steps or len(set(steps)) != len(steps) or any(step not in (1, 11.2) for step in steps):
        raise ValueError('STEPS must contain 1 and/or 11.2.')
    if not 0 <= min_percentage_available <= 1:
        raise ValueError('min_percentage_available must be between 0 and 1.')
    if min(step1_batch, voxel_batch, comparison_batch) < 1:
        raise ValueError('Batch sizes must be positive.')
    if allow_single_participant and 11.2 in steps:
        raise ValueError('A single-participant worker can only compute step 1.')
    packages, reference, mask = discover_packages(packages_dir, specie, dataset, model, participants,
                                                 minimum_participants=1 if allow_single_participant else 2)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    out, work = Path(out_dir), Path(work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    print(f'Cohort: {sorted(packages)}; device={device}; steps={steps}', flush=True)
    records = {sub: records_for(package, out) for sub, package in packages.items()}
    groups = None
    if 1 in steps:
        for sub, package in packages.items():
            pending = [r for r in records[sub] if force or not complete(r['path'], r['receipt'], r['members'])]
            if not pending:
                print(f'Step 1 {specie}-sub-{sub:02d}: all omission checkpoints exist.', flush=True)
                continue
            manifest = package['manifest']
            staging = work / 'participant-data'
            _remove_scratch(staging, work)
            beta_bytes = len(manifest['runs']) * len(manifest['stim_types']) * int(mask.sum()) * 8
            with zipfile.ZipFile(package['path']) as archive:
                extract_bytes = sum(i.file_size for i in archive.infolist() if '/results/GLM/' in i.filename)
            if 1.1 * (beta_bytes + extract_bytes) > shutil.disk_usage(work).free:
                raise RuntimeError('Insufficient local disk for this participant and masked beta cache.')
            print(f'Step 1 {specie}-sub-{sub:02d}: {len(pending)} omissions remaining.', flush=True)
            _extract_betas(package, staging)
            beta_path = work / 'betas.npy'
            betas = kernels.load_betas(staging, manifest, reference, mask, beta_path)
            sums, counts, category_indices = kernels.category_sums(betas, manifest)
            if groups is None:
                groups = kernels.neighborhoods(mask, manifest['radius'])
            for index, record in enumerate(pending, 1):
                print(f"  omission {index}/{len(pending)}: run-{record['run']:02d} {record['stimulus']}", flush=True)
                values = kernels.omission_maps(betas, sums, counts, record['unit'], record['stimulus_index'],
                    category_indices[record['stimulus_index']], groups, manifest['pairs'], manifest['categories'],
                    device, step1_batch)
                def writer(archive):
                    for pair_index, name in enumerate(record['members']):
                        write_image(archive, name, values[:, pair_index], mask, reference)
                    archive.writestr(record['folder'] + '/omission.json', json.dumps(dict(
                        receipt=record['receipt'], removed_stimulus=record['stimulus'],
                        removed_from=manifest['runs'][record['unit']], pairs=manifest['pairs']), indent=2))
                publish(record['path'], record['receipt'], writer, work)
            del sums, counts
            betas._mmap.close()
            del betas
            _remove_scratch(beta_path, work)
            _remove_scratch(staging, work)
    outputs = [str(r['path']) for rows in records.values() for r in rows] if 1 in steps else []
    if 11.2 not in steps:
        return outputs
    manifest = next(iter(packages.values()))['manifest']
    comparison_pairs = manifest.get('comparison_pairs', manifest['pairs'])
    pair_indices = {frozenset(pair): index for index, pair in enumerate(manifest['pairs'])}
    selected_pair_indices = [pair_indices[frozenset(pair)] for pair in comparison_pairs]
    omissions = sorted({(r['run'], r['stimulus']) for rows in records.values() for r in rows})
    for run_number, stimulus in omissions:
        selected = {sub: [r for r in rows if (r['run'], r['stimulus']) == (run_number, stimulus)]
                    for sub, rows in records.items()}
        selected = {sub: rows for sub, rows in selected.items() if rows}
        if len(selected) < 2 or len(selected) / len(packages) < min_percentage_available:
            raise ValueError(f'Insufficient participants for run-{run_number:02d}, {stimulus}: {sorted(selected)}')
        if len(selected) != len(packages):
            print(f'  excluding participants without this beta/run: {sorted(set(packages) - set(selected))}', flush=True)
        receipt = dict(version=kernels.VERSION, method=manifest['rsa_method'],
                       participants=sorted(selected), min_percentage_available=min_percentage_available,
                       inputs=[r['receipt'] for rows in selected.values() for r in rows])
        token = digest(receipt)[:12]
        destination = out / (f"result_step11_2_omission_{dataset}_{model}_class-{manifest['rsa_class']}_"
                             f'{specie}_run-{run_number:02d}_{stimulus}_{token}.zip')
        prefix = f"{manifest['mask_type']}-" if manifest['mask_type'] else ''
        relative = (f"{dataset}/results/RSA_same/{model}/class_{manifest['rsa_class']}/leave-one-out/"
                    f"by_stim/run-{run_number:02d}/{prefix}{specie}-r-{manifest['radius']}_{manifest['rsa_method']}_{stimulus}.nii.gz")
        outputs.append(str(destination))
        if not force and complete(destination, receipt, [relative, relative[:-7] + '.json']):
            print(f'Step 11.2 run-{run_number:02d} {stimulus}: checkpoint exists.', flush=True)
            continue
        print(f'Step 11.2 run-{run_number:02d} {stimulus}: participants {sorted(selected)}', flush=True)
        observations = [(sub, r) for sub, rows in selected.items() for r in rows]
        bank_shape = (len(observations), int(mask.sum()), len(comparison_pairs))
        if np.prod(bank_shape) * 8 * 1.1 > shutil.disk_usage(work).free:
            raise RuntimeError('Insufficient local disk for the masked RDM bank.')
        bank_path = work / 'rdms.npy'
        bank = np.lib.format.open_memmap(bank_path, mode='w+', dtype=np.float64, shape=bank_shape)
        for row, (sub, record) in enumerate(observations):
            if not complete(record['path'], record['receipt'], record['members']):
                raise FileNotFoundError(f"Missing/incomplete step-1 omission checkpoint: {record['path']}. Run step 1 first.")
            with zipfile.ZipFile(record['path']) as archive:
                for column, pair_index in enumerate(selected_pair_indices):
                    name = record['members'][pair_index]
                    image = image_from_zip(archive, name)
                    gpu.check_same_space(('mask', reference), [(name, image)], context='Colab omission step 11.2',
                                         strict=not manifest.get('allow_space_mismatch', False))
                    if image.shape != reference.shape:
                        raise ValueError(f'Expected a 3D pair map: {name}')
                    bank[row, :, column] = image.get_fdata()[mask]
        participant_rows = [[i for i, (sub, _) in enumerate(observations) if sub == participant]
                            for participant in selected]
        values = kernels.group_similarity(bank, participant_rows, manifest['rsa_method'], device,
                                           voxel_batch, comparison_batch)
        def writer(archive):
            write_image(archive, relative, values, mask, reference, outside=0.)
            archive.writestr(relative[:-7] + '.json', json.dumps(dict(
                receipt=receipt, omitted_run=run_number, omitted_stimulus=stimulus,
                participants_requested=sorted(packages), participants_included=sorted(selected),
                comparison_pairs=comparison_pairs,
                input_archives=[str(r['path']) for _, r in observations],
                reference='other participants with the same omitted beta/run',
                weighting='sessions within participant, then participants equally'), indent=2))
        publish(destination, receipt, writer, work)
        bank._mmap.close()
        del bank
        _remove_scratch(bank_path, work)
    print(f'Completed. Result ZIPs: {out}', flush=True)
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--specie', choices=['D', 'H'], default='D')
    parser.add_argument('--dataset', default='EmoC')
    parser.add_argument('--model', default='basic-block')
    parser.add_argument('--participants', type=int, nargs='+')
    parser.add_argument('--steps', type=float, nargs='+', default=[1, 11.2])
    parser.add_argument('--work', default='/content/stimulus_omission_work')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--step1-batch', type=int, default=256)
    parser.add_argument('--voxel-batch', type=int, default=128)
    parser.add_argument('--comparison-batch', type=int, default=4096)
    parser.add_argument('--min-percentage-available', type=float, default=1.0)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--step1-worker', action='store_true', help='Allow one participant for a supervised step-1 worker')
    args = parser.parse_args()
    run(args.packages, args.out, specie=args.specie, dataset=args.dataset, model=args.model,
        participants=args.participants, steps=args.steps, work=args.work, device=args.device,
        step1_batch=args.step1_batch, voxel_batch=args.voxel_batch,
        comparison_batch=args.comparison_batch, force=args.force,
        min_percentage_available=args.min_percentage_available,
        allow_single_participant=args.step1_worker)

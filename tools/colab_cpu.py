"""Pack the workstation RSA code and run its supported steps from a flat Drive folder.

The pipeline expects ``datafolder/dataset``. On Colab a temporary local symlink
maps that layout to the flat Drive folder, so every result is written through to
``rsa_colab/results`` using the original pipeline's filenames.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


DRIVE_ROOT = Path('/content/drive/MyDrive/rsa_colab')
SUPPORTED_STEPS = (2, 3, 4, 5, 6, 7, 8, 9, 10)
NOTEBOOK_STEPS = (2, 3, 6, 7, 8, 9, 10)
DEFAULT_OPTIONS = dict(dataset='EmoC', model='basic-block',
                       rsa_model='action_tendency__all', specie='D',
                       participants=None, dis_method='mahalanobis',
                       rsa_method='kendall', mah_fold='stim-wise',
                       mask_type='b_GreyMatter2mmB', radius=3, reps=100,
                       reps_group=1000, z_threshold=3.1,
                       cluster_threshold=0.05, min_percentage_available=1.0,
                       min_dist_mm=8.0, replace_file=False,
                       replace_rnd_files=False, report_title=None,
                       model_specie=None)
PACKAGE_FILES = (
    'searchlight.py', 'rsa_utils.py', 'utils.py', 'preprocess_functions.py',
    'publication_report.py', 'tools/colab_cpu.py',
    'Atlas/Dog/Czeibert_dictionary.csv',
    'Atlas/Dog/Nitzsche/brain2mm.nii.gz',
    'Atlas/Dog/Nitzsche/b_GreyMatter2mmB.nii.gz',
    'Atlas/Dog/Nitzsche/b_GreyMatter2mm.nii.gz',
    'Atlas/Dog/Nitzsche/Czeibert_labels2mm.nii.gz',
    'Atlas/Hum/AAL_dictionary.csv',
    'Atlas/Hum/AAL3.nii.gz',
    'Atlas/Hum/MNI152_T1_2mm_brain.nii.gz',
)
REQUIREMENTS = 'numpy\npandas\nscipy\nnibabel\nnilearn\nPyYAML\nmatplotlib\nscikit-learn\nipywidgets\npython-docx>=1.1,<2\n'
DATA_ARCHIVE_NAME = 'input_data.zip'


def participants_from_bids(database_csv: str | Path) -> list[int]:
    """Read the unique participant IDs from the dataset's BIDS run manifest."""
    database_csv = Path(database_csv)
    with database_csv.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or 'sub_N' not in reader.fieldnames:
            raise ValueError(f'BIDS manifest lacks sub_N column: {database_csv}')
        ids = set()
        for row_number, row in enumerate(reader, start=2):
            value = (row.get('sub_N') or '').strip()
            if not value:
                raise ValueError(f'Missing sub_N in {database_csv}:{row_number}')
            try:
                participant = int(value)
            except ValueError as exc:
                raise ValueError(f'Invalid sub_N {value!r} in {database_csv}:{row_number}') from exc
            if participant < 1:
                raise ValueError(f'Invalid sub_N {value!r} in {database_csv}:{row_number}')
            ids.add(participant)
    if not ids:
        raise ValueError(f'BIDS manifest contains no participants: {database_csv}')
    return sorted(ids)


def selected_participants(data_root: Path, dataset: str, model: str,
                          specie: str, requested: list[int] | None) -> list[int] | None:
    """Use an explicit list, otherwise the packaged BIDS-backed selection."""
    if requested is not None:
        return requested
    bids = data_root / 'BIDS' / f'{specie}_database-details.csv'
    if not bids.is_file():
        return None
    available = set(participants_from_bids(bids))
    receipt = data_root / 'rsa_colab_inputs.json'
    if receipt.is_file():
        manifest = json.loads(receipt.read_text(encoding='utf-8'))
        if (manifest.get('dataset'), manifest.get('model'), manifest.get('specie')) == (dataset, model, specie):
            chosen = [int(p) for p in manifest.get('ready_participants',
                                                   manifest['participants'])]
            if not chosen:
                raise ValueError(f'No participants with step-1 maps in {receipt}')
            unknown = sorted(set(chosen) - available)
            if unknown:
                raise ValueError(f'Packaged participants missing from {bids}: {unknown}')
            return chosen
    return sorted(available)


def pack_for_CPU_colab(
    datafolder: str | Path,
    output_zip: str | Path | None = None,
    *,
    dataset: str = 'EmoC',
    model: str = 'basic-block',
    specie: str = 'D',
    participants: list[int] | None = None,
    mask_type: str = 'b_GreyMatter2mmB',
    radius: int = 3,
    allow_missing: bool = False,
) -> Path:
    """Pack step-2/4 input data from ``datafolder/dataset`` for the CPU notebook.

    The archive has the flat ``rsa_colab`` layout: config, BIDS run table,
    step-1 pairwise maps for ``radius``, and a dataset-local ROI mask when required. It
    intentionally excludes ``rsa_models``; add the model CSV manually.
    With no participant list, all unique ``sub_N`` values in the BIDS run
    manifest are selected. Set ``allow_missing`` to pack available maps while
    recording participants whose step-1 maps are missing.
    """
    datafolder = Path(datafolder).expanduser().resolve()
    if not dataset or Path(dataset).name != dataset or dataset in ('.', '..'):
        raise ValueError('dataset must be a single directory name')
    if specie not in ('D', 'H'):
        raise ValueError('specie must be D or H')
    if radius < 1:
        raise ValueError('radius must be a positive integer')
    dataset_root = datafolder / dataset
    if not dataset_root.is_dir():
        raise FileNotFoundError(f'Dataset folder missing: {dataset_root}')
    if output_zip is None:
        output_zip = Path(__file__).resolve().parents[1] / 'rsa_colab' / DATA_ARCHIVE_NAME
    output_zip = Path(output_zip).expanduser().resolve()

    config = dataset_root / 'config_files' / f'{specie}_{model}.yaml'
    bids = dataset_root / 'BIDS' / f'{specie}_database-details.csv'
    for required in (config, bids):
        if not required.is_file():
            raise FileNotFoundError(required)
    manifest_participants = participants_from_bids(bids)
    chosen = manifest_participants if participants is None else participants
    if not chosen or any(int(p) < 1 for p in chosen):
        raise ValueError('participants must contain positive subject numbers')
    chosen = sorted({int(p) for p in chosen})
    unknown = sorted(set(chosen) - set(manifest_participants))
    if unknown:
        raise ValueError(f'Participants not listed in {bids}: {unknown}')

    files = [config, bids]
    mask_stem = mask_type.removesuffix('.nii.gz').removesuffix('.nii')
    if specie == 'H' or mask_stem == 'cope13':
        mask = dataset_root / 'ROI' / specie / f'{mask_stem}.nii.gz'
        if not mask.is_file():
            raise FileNotFoundError(f'Dataset-local mask missing: {mask}')
        files.append(mask)
    per_participant = {}
    missing = []
    missing_participants = []
    for participant in chosen:
        folder = (dataset_root / 'results' / 'RSA' / model /
                  f'{specie}-sub-{participant:02d}')
        maps = []
        if folder.is_dir():
            # os.walk uses scandir entries; avoid a separate network stat call
            # for every NIfTI on mounted data shares.
            for current, _, names in os.walk(folder):
                maps.extend(Path(current) / name for name in names
                            if name.endswith('.nii.gz') and f'r-{radius}_' in name)
            maps.sort()
        if not maps:
            missing.append(str(folder))
            missing_participants.append(participant)
        else:
            files.extend(maps)
            per_participant[f'{specie}-sub-{participant:02d}'] = len(maps)
    if missing and not allow_missing:
        raise FileNotFoundError(f'Step-1 r-{radius} pairwise maps missing for: ' + ', '.join(missing))

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(dataset=dataset, model=model, specie=specie, radius=radius,
                    participants=chosen, mask_type=mask_stem,
                    participant_source=f'BIDS/{specie}_database-details.csv:sub_N',
                    ready_participants=sorted(set(chosen) - set(missing_participants)),
                    missing_step1_participants=missing_participants,
                    step1_maps=per_participant,
                    note='Add rsa_models CSV files manually before running steps 2 or 4.')
    temporary = output_zip.with_name(output_zip.name + '.tmp')
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for source in files:
                archive.write(source, arcname=source.relative_to(dataset_root).as_posix(),
                              compress_type=(zipfile.ZIP_STORED if source.name.endswith('.nii.gz')
                                             else zipfile.ZIP_DEFLATED))
            archive.writestr('rsa_colab_inputs.json', json.dumps(manifest, indent=2) + '\n')
        temporary.replace(output_zip)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f'Packed {len(files)} input files ({sum(per_participant.values())} step-1 maps) '
          f'for {len(chosen)} BIDS participants: {output_zip} '
          f'({output_zip.stat().st_size:,} bytes)')
    if missing_participants:
        print(f'Missing step-1 maps for participants {missing_participants}; '
              'recorded in rsa_colab_inputs.json and excluded from default runs.')
    return output_zip


def unpack_data(archive_path: str | Path, drive_root: str | Path = DRIVE_ROOT) -> dict:
    """Unpack a CPU input archive to the flat rsa_colab folder."""
    drive_root, archive_path = Path(drive_root), Path(archive_path)
    allowed = {'config_files', 'BIDS', 'ROI', 'results'}
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or 'rsa_colab_inputs.json' not in names:
            raise ValueError('Input archive has duplicate paths or no manifest')
        manifest = json.loads(archive.read('rsa_colab_inputs.json'))
        for name in names:
            parts = name.split('/')
            if (name.startswith('/') or '\\' in name or not parts or
                    any(part in ('', '.', '..') for part in parts) or
                    (name != 'rsa_colab_inputs.json' and parts[0] not in allowed)):
                raise ValueError(f'Unsafe input archive member: {name}')
        drive_root.mkdir(parents=True, exist_ok=True)
        for name in names:
            target = drive_root.joinpath(*name.split('/'))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    return manifest


def pack_functions(repo_root: str | Path, output_zip: str | Path) -> Path:
    """Build a small, deterministic toolkit archive from this Git checkout."""
    repo_root, output_zip = Path(repo_root).resolve(), Path(output_zip).resolve()
    missing = [name for name in PACKAGE_FILES if not (repo_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing toolkit files: {missing}')
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in PACKAGE_FILES:
            archive.write(repo_root / name, arcname=f'toolkit/{name}')
        archive.writestr('toolkit/requirements-colab.txt', REQUIREMENTS)
    print(f'Packed {len(PACKAGE_FILES)} toolkit files: {output_zip} ({output_zip.stat().st_size:,} bytes)')
    return output_zip


def unpack_functions(archive_path: str | Path, drive_root: str | Path = DRIVE_ROOT) -> Path:
    """Extract the package into rsa_colab/toolkit, rejecting unexpected members."""
    drive_root, archive_path = Path(drive_root), Path(archive_path)
    drive_root.mkdir(parents=True, exist_ok=True)
    expected = {f'toolkit/{name}' for name in PACKAGE_FILES}
    expected.add('toolkit/requirements-colab.txt')
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        if members != expected:
            raise ValueError(f'Unexpected package contents: {sorted(members ^ expected)}')
        for member in sorted(members):
            target = drive_root / member
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    return drive_root / 'toolkit'


def prepare_workspace(drive_root: str | Path = DRIVE_ROOT, dataset: str = 'EmoC') -> Path:
    """Return a datafolder whose dataset directory points at the flat Drive root.

    A Windows test fixture may instead live at ``rsa_colab/_local_data/EmoC``.
    Colab's link lives on its local filesystem; no symlink is written to Drive.
    """
    root = Path(drive_root).resolve()
    if not dataset or Path(dataset).name != dataset or dataset in ('.', '..'):
        raise ValueError('dataset must be a single directory name')
    if not root.is_dir():
        raise FileNotFoundError(root)
    local_fixture = root / '_local_data' / dataset
    if local_fixture.is_dir():
        return local_fixture.parent
    if os.name == 'nt':
        raise RuntimeError('Windows testing requires rsa_colab/_local_data/<dataset>; Colab creates a local symlink.')
    parent = Path(tempfile.gettempdir()) / 'rsa_colab_cpu_data'
    parent.mkdir(parents=True, exist_ok=True)
    link = parent / dataset
    if link.is_symlink():
        if link.resolve() != root:
            link.unlink()
    elif link.exists():
        raise FileExistsError(f'Cannot replace existing dataset path: {link}')
    if not link.exists():
        link.symlink_to(root, target_is_directory=True)
    return parent


def _validate_inputs(root: Path, dataset: str, model: str, specie: str,
                     rsa_model: str, step: int, participants: list[int] | None,
                     model_specie: str | None = None) -> None:
    data_root = root / '_local_data' / dataset
    if not data_root.is_dir():
        data_root = root
    if step not in SUPPORTED_STEPS:
        raise ValueError(f'Unsupported step {step}; choose from {SUPPORTED_STEPS}')
    if specie not in ('D', 'H'):
        raise ValueError('specie must be D or H')
    if not (data_root / 'config_files' / f'{specie}_{model}.yaml').is_file():
        raise FileNotFoundError(data_root / 'config_files' / f'{specie}_{model}.yaml')
    model_dir = data_root / 'rsa_models'
    if model_specie and (step not in (3, 5) or model_specie not in ('D', 'H')):
        raise ValueError('model_specie may be D or H for steps 3 and 5 only')
    if not model_specie and not (model_dir / f'{rsa_model}.csv').is_file() and not (model_dir / f'{rsa_model}-run-1.csv').is_file():
        raise FileNotFoundError(f'RSA model CSV missing: {rsa_model} in {model_dir}')
    if step in (2, 3, 4, 5) and not (data_root / 'BIDS' / f'{specie}_database-details.csv').is_file():
        raise FileNotFoundError(data_root / 'BIDS' / f'{specie}_database-details.csv')
    if participants is not None and (not participants or any(int(p) < 1 for p in participants)):
        raise ValueError('participants must contain positive subject numbers')
    bids = data_root / 'BIDS' / f'{specie}_database-details.csv'
    if participants is not None and bids.is_file():
        unknown = sorted(set(map(int, participants)) - set(participants_from_bids(bids)))
        if unknown:
            raise ValueError(f'Participants not listed in {bids}: {unknown}')


def command_for_step(root: str | Path, *, step: int, dataset: str = 'EmoC',
                     model: str = 'basic-block', rsa_model: str = 'action_tendency__all',
                     specie: str = 'D', dis_method: str = 'mahalanobis',
                     rsa_method: str = 'kendall', mah_fold: str = 'stim-wise',
                     mask_type: str = 'b_GreyMatter2mmB', radius: int = 3,
                     participants: list[int] | None = None, reps: int = 100,
                     reps_group: int = 1000, z_threshold: float = 3.1,
                     cluster_threshold: float = 0.05,
                     min_percentage_available: float = 1.0,
                     min_dist_mm: float = 8.0, replace_file: bool = False,
                     replace_rnd_files: bool = False,
                     report_title: str | None = None,
                     model_specie: str | None = None) -> list[str]:
    root = Path(root).resolve()
    _validate_inputs(root, dataset, model, specie, rsa_model, step, participants,
                     model_specie)
    toolkit = root / 'toolkit'
    if not (toolkit / 'searchlight.py').is_file():
        raise FileNotFoundError(f'Unpack the functions first: {toolkit / "searchlight.py"}')
    datafolder = prepare_workspace(root, dataset)
    cmd = [sys.executable, '-u', str(toolkit / 'searchlight.py'),
           '--datafolder', str(datafolder), '--toolkit_dir', str(toolkit),
           '--dataset', dataset, '--model', model, '--rsa_model', rsa_model,
           '--specie', specie, '--steps_to_run', str(step),
           '--dis_method', dis_method, '--rsa_method', rsa_method,
           '--mah_fold', mah_fold, '--mask_type', mask_type, '--radius', str(radius),
           '--reps', str(reps), '--reps_group', str(reps_group),
           '--z_threshold', str(z_threshold), '--cluster_threshold', str(cluster_threshold),
           '--min_percentage_available', str(min_percentage_available),
           '--min_dist_mm', str(min_dist_mm), '--wait_time', '0']
    selected = selected_participants(datafolder / dataset, dataset, model,
                                     specie, participants)
    if selected:
        cmd.extend(['--participants_forced', *map(str, selected)])
    if replace_file:
        cmd.append('--replace_file')
    if replace_rnd_files:
        cmd.append('--replace_rnd_files')
    if report_title:
        cmd.extend(['--report_title', report_title])
    if model_specie:
        cmd.extend(['--model_specie', model_specie])
    return cmd


def run_step(root: str | Path = DRIVE_ROOT, **options) -> None:
    step = int(options['step'])
    if step not in NOTEBOOK_STEPS:
        raise ValueError(f'Notebook runs {NOTEBOOK_STEPS}; steps 4 and 5 are job files')
    cmd = command_for_step(root, **options)
    _run_checked(cmd, step)


def _run_checked(cmd: list[str], step: int) -> None:
    """Require the pipeline's completion marker as well as a zero exit code."""
    with tempfile.TemporaryDirectory(prefix='rsa_colab_marker_') as marker_dir:
        print('Running:', ' '.join(cmd), flush=True)
        subprocess.run([*cmd, '--job_marker_dir', marker_dir], check=True)
        if not (Path(marker_dir) / f'{step}.done').is_file():
            raise RuntimeError(f'Step {step} exited without its completion marker')


def create_job(root: str | Path = DRIVE_ROOT, **options) -> Path:
    """Write a pending step-4 or step-5 job, without starting computation."""
    root = Path(root).resolve()
    step = int(options['step'])
    if step not in (4, 5):
        raise ValueError('Only steps 4 and 5 can be queued here')
    options = {**DEFAULT_OPTIONS, **options}
    command_for_step(root, **options)  # preflight all paths and parameters
    canonical = json.dumps(options, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    model = options.get('rsa_model', 'action_tendency__all')
    job_dir = root / 'jobs' / 'pending'
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / f'step{step}_{model}_{digest}.json'
    job = {'step': step, 'options': options,
           'requires': 'step 1 pairwise maps' if step == 4 else 'step 4 permutation maps',
           'result_folder': 'results/RSA_rnd',
           'run': 'python toolkit/tools/colab_cpu.py run-job <path-to-this-json>'}
    if not path.exists():
        path.write_text(json.dumps(job, indent=2) + '\n', encoding='utf-8')
    print(f'Job saved: {path}')
    return path


def run_job(path: str | Path, root: str | Path | None = None) -> None:
    """Execute one saved job in a CPU Python environment with Drive mounted."""
    path = Path(path).resolve()
    root = Path(root).resolve() if root is not None else path.parents[2]
    job = json.loads(path.read_text(encoding='utf-8'))
    options = job['options']
    if job['step'] not in (4, 5) or options.get('step') != job['step']:
        raise ValueError('Invalid step 4/5 job')
    cmd = command_for_step(root, **options)
    try:
        _run_checked(cmd, job['step'])
    except Exception:
        destination = root / 'jobs' / 'failed'
        destination.mkdir(parents=True, exist_ok=True)
        path.replace(destination / path.name)
        raise
    destination = root / 'jobs' / 'completed'
    destination.mkdir(parents=True, exist_ok=True)
    path.replace(destination / path.name)
    print(f'Completed job: {destination / path.name}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    pack = sub.add_parser('pack', help='Build toolkit zip from this checkout')
    pack.add_argument('--repo', default=str(Path(__file__).resolve().parents[1]))
    pack.add_argument('--output', required=True)
    pack_data = sub.add_parser('pack-data', help='Build input_data.zip from datafolder/dataset')
    pack_data.add_argument('--datafolder', required=True)
    pack_data.add_argument('--output', default=None)
    pack_data.add_argument('--dataset', default='EmoC')
    pack_data.add_argument('--model', default='basic-block')
    pack_data.add_argument('--specie', choices=('D', 'H'), default='D')
    pack_data.add_argument('--participants', type=int, nargs='+', default=None)
    pack_data.add_argument('--mask_type', default='b_GreyMatter2mmB')
    pack_data.add_argument('--radius', type=int, default=3)
    pack_data.add_argument('--allow-missing', action='store_true',
                           help='Pack available maps and record participants missing step-1 maps')
    run = sub.add_parser('run-job', help='Run one saved step-4/5 job')
    run.add_argument('job_file')
    run.add_argument('--root', default=None)
    args = parser.parse_args()
    if args.action == 'pack':
        pack_functions(args.repo, args.output)
    elif args.action == 'pack-data':
        pack_for_CPU_colab(args.datafolder, args.output, dataset=args.dataset,
                           model=args.model, specie=args.specie,
                           participants=args.participants, mask_type=args.mask_type,
                           radius=args.radius,
                           allow_missing=args.allow_missing)
    else:
        run_job(args.job_file, args.root)


if __name__ == '__main__':
    main()

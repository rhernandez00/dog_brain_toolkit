"""Launch a self-contained step-11.31 ZIP through `colab exec -s NAME -f FILE`.

Mount Drive first with `colab drivemount -s NAME`. In a Colab kernel, optional
CONTRIBUTION_JOB = dict(package_zip=..., out_dir=...) settings persist between
exec commands. Ordinary `python launch_stimulus_contribution.py --help` works too.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout, ExitStack
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

DEFAULT_PACKAGE = '/content/drive/MyDrive/rsa_colab/contributions_H.zip'
DEFAULT_OUTPUT = '/content/drive/MyDrive/rsa_colab/results_stimulus_contribution_H'
DEFAULT_WORK = '/content/stimulus_work'
REQUIRED_PACKAGE_VERSION = '1.4.0'


class _Tee:
    def __init__(self, screen, log):
        self.screen, self.log = screen, log

    def write(self, text):
        self.screen.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.screen.flush()
        self.log.flush()


def extract_package(package_zip, work_root, allow_legacy=False, stage_archive=False, extract_workers=1, stage_memory=False):
    """Extract once per archive identity; preserve mtimes for resume checks."""
    if extract_workers < 1:
        raise ValueError('extract_workers must be positive.')
    staged = None
    with zipfile.ZipFile(package_zip) as archive, ExitStack() as files:
        manifest = json.loads(archive.read('manifest.json'))
        if manifest.get('version') != REQUIRED_PACKAGE_VERSION and not (
                allow_legacy and manifest.get('version') == '1.3.0'):
            raise ValueError('Rebuild the package with code version 1.4.0, or supply '
                             'code_override for an existing 1.3.0 data package.')
        if (manifest.get('kind') != 'stimulus-contribution-11.31'
                or not manifest.get('includes_maps') or not manifest.get('class_csv')
                or not manifest.get('input_map_count')):
            raise ValueError('Rebuild with the current packager defaults: all maps and the class CSV must be bundled.')
        infos = archive.infolist()
        map_count = sum(i.filename.startswith('data/') and '/results/RSA/' in i.filename
                        and i.filename.endswith('.nii.gz') for i in infos)
        if map_count != manifest['input_map_count']:
            raise ValueError(f'Incomplete package: expected {manifest["input_map_count"]} maps; found {map_count}.')
        identity = hashlib.sha256(json.dumps([(i.filename, i.CRC, i.file_size) for i in infos]).encode()).hexdigest()
        root = Path(work_root).resolve() / ('package_' + identity[:16])
        for info in infos:
            if not (root / info.filename).resolve().is_relative_to(root):
                raise ValueError(f'Unsafe archive member: {info.filename}')
        complete = root / '.extracted'
        if complete.is_file() and complete.read_text() == identity:
            print(f'Reusing extracted package: {root}', flush=True)
            return root, manifest
        Path(work_root).mkdir(parents=True, exist_ok=True)
        required = sum(i.file_size for i in infos)
        staging_bytes = Path(package_zip).stat().st_size if stage_archive or stage_memory else 0
        if required * 1.05 + (0 if stage_memory else staging_bytes) > shutil.disk_usage(work_root).free:
            raise RuntimeError(f'Insufficient disk space: extraction needs {required / 1e9:.2f} GB.')
        if stage_memory:
            import psutil
            if not hasattr(os, 'memfd_create'):
                raise RuntimeError('In-memory archive staging requires Linux memfd support.')
            if psutil.virtual_memory().available < staging_bytes * 1.1 + 4 * 2**30:
                raise RuntimeError('Insufficient RAM for in-memory ZIP staging.')
            buffer = files.enter_context(os.fdopen(os.memfd_create('contribution-input'), 'w+b'))
            prefix = Path(work_root) / ('input_' + identity[:16] + '.zip.part')
            copied = 0
            if prefix.is_file():
                copied = prefix.stat().st_size
                if copied > staging_bytes:
                    raise RuntimeError('Staged prefix is larger than source archive.')
                print(f'Moving existing {copied / 1e9:.2f} GB staged prefix into RAM.', flush=True)
                with prefix.open('rb') as src:
                    shutil.copyfileobj(src, buffer, length=64 * 2**20)
                prefix.unlink()
            print(f'Staging {staging_bytes / 1e9:.2f} GB archive in RAM; resuming at {copied / 1e9:.2f} GB.', flush=True)
            last_progress = time.monotonic()
            with Path(package_zip).open('rb') as src:
                src.seek(copied)
                while True:
                    block = src.read(64 * 2**20)
                    if not block:
                        break
                    buffer.write(block)
                    copied += len(block)
                    if time.monotonic() - last_progress >= 15:
                        print(f'Staging in RAM: {copied / 1e9:.2f}/{staging_bytes / 1e9:.2f} GB.', flush=True)
                        last_progress = time.monotonic()
            if copied != staging_bytes:
                raise RuntimeError('Incomplete in-memory archive.')
            buffer.seek(0)
            archive = files.enter_context(zipfile.ZipFile(buffer))
        elif stage_archive:
            staged = Path(work_root) / ('input_' + identity[:16] + '.zip')
            partial = staged.with_suffix('.zip.part')
            print(f'Staging archive sequentially on local disk: {staged}', flush=True)
            copied, last_progress = 0, time.monotonic()
            with Path(package_zip).open('rb') as src, partial.open('wb') as dst:
                while True:
                    block = src.read(64 * 2**20)
                    if not block:
                        break
                    dst.write(block)
                    copied += len(block)
                    if time.monotonic() - last_progress >= 15:
                        print(f'Staging: {copied / 1e9:.2f}/{staging_bytes / 1e9:.2f} GB.', flush=True)
                        last_progress = time.monotonic()
            if copied != staging_bytes:
                raise RuntimeError('Incomplete staged archive.')
            partial.replace(staged)
            archive = files.enter_context(zipfile.ZipFile(staged))
        print(f'Extracting {map_count} maps ({required / 1e9:.2f} GB) to {root}; workers={extract_workers}', flush=True)
        def extract(info):
            target = root / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_name(target.name + '.extract.part')
                with archive.open(info.filename) as src, partial.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=2**20)
                partial.replace(target)
        from concurrent.futures import ThreadPoolExecutor
        last_progress = time.monotonic()
        with ThreadPoolExecutor(max_workers=extract_workers) as executor:
            for index, _ in enumerate(executor.map(extract, infos), 1):
                if index == len(infos) or index % 1000 == 0 or time.monotonic() - last_progress >= 15:
                    print(f'Extraction: {index}/{len(infos)} files.', flush=True)
                    last_progress = time.monotonic()
        complete.write_text(identity)
    if staged is not None:
        staged.unlink()  # Free space before allocating the larger RDM scratch.
    return root, manifest


def run_job(package_zip=DEFAULT_PACKAGE, out_dir=DEFAULT_OUTPUT, work_root=DEFAULT_WORK,
            voxel_batch=128, comparison_batch=4096, force=False, device='cuda',
            install_dependencies=True, code_override=None, save_participant_maps=False, load_workers=1,
            stage_archive=False, extract_workers=1, stage_memory=False, rdm_work_root=None, release_extracted_inputs=False):
    """Stream progress to the CLI and persist the same output in a Drive log."""
    package_zip, out_dir = Path(package_zip), Path(out_dir)
    if not package_zip.is_file():
        raise FileNotFoundError(f'Package not found: {package_zip}. Mount Drive and check the uploaded filename.')
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'step11_31.log').open('a', buffering=1, encoding='utf-8') as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            print(f'\nSteps 11.2, 11.3 and 11.31 started: {time.strftime("%Y-%m-%d %H:%M:%S %Z")}', flush=True)
            print(f'Package: {package_zip}\nResults: {out_dir}', flush=True)
            try:
                if install_dependencies:
                    missing = [name for name in ('numpy', 'scipy', 'nibabel') if importlib.util.find_spec(name) is None]
                    if missing:
                        print(f'Installing dependencies: {missing}', flush=True)
                        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
                import numpy as np
                import nibabel as nib
                if str(device).startswith('cuda'):
                    import torch
                    if not torch.cuda.is_available():
                        raise RuntimeError('No CUDA GPU found. Create the session with colab new --gpu T4.')
                    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
                root, manifest = extract_package(package_zip, work_root, allow_legacy=bool(code_override),
                                                 stage_archive=stage_archive, extract_workers=extract_workers, stage_memory=stage_memory)
                if code_override:
                    override = Path(code_override)
                    files = ('stimulus_contribution.py', 'run_colab_stimulus_contribution.py')
                    if any(not (override / name).is_file() for name in files):
                        raise FileNotFoundError('code_override must contain the contribution core and runner.')
                    source_version = manifest['version']
                    hashes = {name: hashlib.sha256((override / name).read_bytes()).hexdigest() for name in files}
                    for name in files:
                        shutil.copyfile(override / name, root / name)
                    manifest = dict(manifest, version=REQUIRED_PACKAGE_VERSION,
                                    source_package_version=source_version, code_override_sha256=hashes)
                    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
                    print(f'Using updated code {REQUIRED_PACKAGE_VERSION}; original data ZIP and mask unchanged.', flush=True)
                sys.path.insert(0, str(root))
                importlib.invalidate_caches()
                for name in ('run_colab_stimulus_contribution', 'stimulus_contribution'):
                    sys.modules.pop(name, None)
                contribution = importlib.import_module('stimulus_contribution')
                runner = importlib.import_module('run_colab_stimulus_contribution')
                if contribution.VERSION != manifest['version']:
                    raise ValueError('Code/package version mismatch; rebuild the package.')
                pairs = contribution.read_pairs(root / manifest['class_csv'])
                labels = contribution.validate_pairs(pairs)[1]
                print(f'{manifest["dataset"]} / {manifest["specie"]}, '
                      f'{manifest["dis_method"]} / {manifest["mah_fold"]}, '
                      f'{len(manifest["participants"])} participants, {len(pairs)} pairs.', flush=True)
                print(f'Omissions ({len(labels)}): {labels}', flush=True)
                if str(device).startswith('cuda'):
                    import itertools
                    test_pairs = list(itertools.combinations(['A', 'B', 'C', 'D', 'E'], 2))
                    rng = np.random.default_rng(1131)
                    x, y = rng.integers(0, 5, (2, 9, len(test_pairs))).astype(float)
                    x[0] = 1
                    x[1, :3] = np.nan
                    y[2, 2:5] = np.inf
                    x[3] = np.nan
                    y[4, :-1] = np.nan
                    for method in ('kendall', 'spearman', 'pearson'):
                        expected = contribution.correlate_omissions(x, y, test_pairs, method=method, include_full=True)
                        actual = contribution.correlate_omissions(x, y, test_pairs, method=method, device=device, include_full=True)
                        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
                    print('CPU/GPU parity check passed.', flush=True)
                mask = np.asarray(nib.load(str(root / manifest['mask'])).dataobj)
                nvox = int((np.isfinite(mask) & (mask != 0)).sum())
                if manifest['dis_method'] == 'mahalanobis' and manifest['mah_fold'] != 'stim-wise-all-runs':
                    sessions = len(manifest['participants'])
                else:
                    entries = [e for rows in manifest['runs_by_sub'].values() for e in rows]
                    sessions = max(sum(int(e['run_N']) == r for e in entries)
                                   for r in {int(e['run_N']) for e in entries})
                required = sessions * nvox * len(pairs) * 8
                if save_participant_maps:
                    required += len(manifest['participants']) * nvox * len(labels) * 8
                scratch = Path(rdm_work_root) if rdm_work_root else Path(work_root) / 'rdms'
                scratch.mkdir(parents=True, exist_ok=True)
                free = shutil.disk_usage(scratch).free
                print(f'RDM scratch: {required / 1e9:.2f} GB required, {free / 1e9:.2f} GB available.', flush=True)
                if required * 1.05 > free:
                    raise RuntimeError('Insufficient local disk space for RDM scratch arrays.')
                def release_inputs():
                    owned = root.resolve()
                    maps = owned / 'data'
                    if (not owned.is_relative_to(Path(work_root).resolve())
                            or not owned.name.startswith('package_')
                            or not (owned / '.extracted').is_file()
                            or out_dir.resolve().is_relative_to(maps)):
                        raise ValueError('Refusing to release inputs outside the owned extraction directory.')
                    if maps.is_dir():
                        print(f'Releasing extracted input maps after computation: {maps}', flush=True)
                        shutil.rmtree(maps)
                        (owned / '.extracted').unlink()
                result = runner.run_package(root, [], out_dir, work=str(scratch),
                                            device=device, voxel_batch=voxel_batch,
                                            comparison_batch=comparison_batch, force=force,
                                            save_participant_maps=save_participant_maps, load_workers=load_workers,
                                            before_archive=release_inputs if release_extracted_inputs else None)
                print(f'RESULT_ZIP={result}', flush=True)
                return result
            except BaseException:
                import traceback
                traceback.print_exc()
                raise


def main():
    # colab exec sends this file into a persistent IPython kernel. Do not parse
    # that kernel's connection-file flags as script arguments.
    try:
        in_kernel = get_ipython() is not None
    except NameError:
        in_kernel = False
    if in_kernel:
        return run_job(**globals().get('CONTRIBUTION_JOB', {}))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-zip', default=DEFAULT_PACKAGE)
    parser.add_argument('--out-dir', default=DEFAULT_OUTPUT)
    parser.add_argument('--work-root', default=DEFAULT_WORK)
    parser.add_argument('--voxel-batch', type=int, default=128)
    parser.add_argument('--comparison-batch', type=int, default=4096)
    parser.add_argument('--load-workers', type=int, default=1, help='Parallel session RDM loaders')
    parser.add_argument('--stage-archive', action='store_true', help='Copy input ZIP locally before extraction; needs temporary extra disk space')
    parser.add_argument('--extract-workers', type=int, default=1)
    parser.add_argument('--stage-memory', action='store_true', help='Stage ZIP in Linux RAM instead of duplicating Drive cache on disk')
    parser.add_argument('--rdm-work-root', help='Separate scratch directory, e.g. /dev/shm on high-RAM Colab')
    parser.add_argument('--release-extracted-inputs', action='store_true', help='Release owned extracted pairwise maps after computation, before ZIP creation')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--save-participant-maps', '--save_participant_maps', action='store_true',
                        help='Save and archive participant omission maps')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--code-override', help='Directory with updated core/runner for an existing data ZIP')
    return run_job(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()

"""Colab entry point: stage a step-16.2 package and publish closed checkpoints."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import cross_species_rsa as rsa


def publish(source, target):
    """Copy a closed file through .part before replacing the Drive snapshot."""
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError('Local staging and Drive output must be different paths.')
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(target) + '.part')
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def extract_package(package, staging):
    staging = Path(staging).resolve()
    staging.mkdir(parents=True, exist_ok=True)
    # A fresh extraction prevents stale input files after a package change.
    with zipfile.ZipFile(package) as archive:
        for name in archive.namelist():
            candidate = (staging / name).resolve()
            if not candidate.is_relative_to(staging):
                raise ValueError(f'Unsafe package member: {name}')
        archive.extractall(staging)
    return staging


def run_colab(package, output, staging='/content/cross_species_16_2', *,
              device='auto', source_batch=128, target_batch=4096, checkpoint_minutes=15):
    """output is a root directory; publish one HDF5 + log per source participant.

    Closed progress snapshots use the final HDF5 name with complete=False,
    avoiding permanent sidecar checkpoint files in the result directory.
    """
    if checkpoint_minutes < 0:
        raise ValueError('checkpoint_minutes must be nonnegative.')
    staging = Path(staging)
    output = Path(output)
    if output.suffix.lower() in ('.h5', '.hdf5'):
        raise ValueError('output must be a ROOT directory, not a single HDF5 filename.')
    if staging.resolve() == output.resolve() or staging.resolve().is_relative_to(output.resolve()) or output.resolve().is_relative_to(staging.resolve()):
        raise ValueError('Output and local staging directories must be separate, not nested.')
    print(f'[Colab] Extracting {package} into {staging}', flush=True)
    extract_package(package, staging / 'package')
    root = staging / 'package'
    with open(root / 'manifest.json', encoding='utf-8') as stream:
        manifest = json.load(stream)
    signature, _ = rsa.input_identity(manifest, root)
    if manifest.get('source_signature') != signature:
        raise ValueError('Packaged inputs or runner version differ from the package signature.')
    import h5py
    target_data = rsa.group_rdms(manifest, manifest['target_specie'], root)
    outputs = []
    for participant, single in rsa.participant_manifests(manifest):
        relative = rsa.relative_output_path(single, participant)
        remote = output / relative
        local = staging / 'results' / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        local_checkpoint = Path(str(local) + '.partial')
        single_signature, _ = rsa.input_identity(single, root)
        if not local.exists() and not local_checkpoint.exists() and remote.exists():
            with h5py.File(remote, 'r') as handle:
                if handle.attrs.get('signature') != single_signature:
                    raise ValueError(f'Incompatible remote output: {remote}. Use a new output root.')
                complete = bool(handle.attrs.get('complete', False))
            shutil.copy2(remote, local if complete else local_checkpoint)
            if remote.with_suffix('.log').exists():
                shutil.copy2(remote.with_suffix('.log'), local.with_suffix('.log'))
        last_copy = time.monotonic()

        def checkpoint(path, force=False):
            nonlocal last_copy
            if force or time.monotonic() - last_copy >= checkpoint_minutes * 60:
                print('[Colab] Publishing closed progress snapshot to Drive', flush=True)
                publish(path, remote)
                if local.with_suffix('.log').exists():
                    publish(local.with_suffix('.log'), remote.with_suffix('.log'))
                last_copy = time.monotonic()

        try:
            result = rsa.run_analysis(single, local, root=root, device=device,
                                      source_batch=source_batch, target_batch=target_batch,
                                      checkpoint_callback=checkpoint, target_data=target_data)
        except (Exception, KeyboardInterrupt):
            if local_checkpoint.exists():
                with h5py.File(local_checkpoint, 'r') as handle:
                    compatible = handle.attrs.get('signature') == single_signature
                if compatible:
                    checkpoint(local_checkpoint, force=True)
            if local.with_suffix('.log').exists():
                publish(local.with_suffix('.log'), remote.with_suffix('.log'))
            raise
        publish(result, remote)
        publish(local.with_suffix('.log'), remote.with_suffix('.log'))
        print(f'[Colab] Saved {remote}', flush=True)
        outputs.append(remote)
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', required=True)
    parser.add_argument('--output', required=True, help='Output root for dataset/results/RSA_cross-specie/...')
    parser.add_argument('--staging', default='/content/cross_species_16_2')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--source_batch', type=int, default=128)
    parser.add_argument('--target_batch', type=int, default=4096)
    parser.add_argument('--checkpoint_minutes', type=float, default=15)
    args = parser.parse_args()
    run_colab(**vars(args))


if __name__ == '__main__':
    main()

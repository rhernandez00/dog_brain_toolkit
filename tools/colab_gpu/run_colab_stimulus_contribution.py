"""Colab runner for steps 11.2, 11.3 and 11.31; pipeline-compatible outputs."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
for path in (HERE, HERE.parents[1]):
    sys.path.insert(0, str(path))
import stimulus_contribution as contribution


def run_package(package_root, sources, output, work='/content/stimulus_work',
                device='cuda', voxel_batch=128, comparison_batch=4096, force=False,
                class_csv=None, dis_method=None, mah_fold=None, save_participant_maps=False, load_workers=1, before_archive=None):
    root = Path(package_root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('version') != contribution.VERSION:
        raise ValueError('Code/package version mismatch. Extract the latest package and restart the runtime.')
    manifest['mask'] = str(root / manifest['mask'])
    class_path = Path(class_csv) if class_csv else (root / manifest['class_csv'] if manifest.get('class_csv') else None)
    if class_path is None:
        raise ValueError('Set CLASS_CSV to your rsa_class CSV (cat1/cat2 columns).')
    manifest['pairs'] = contribution.read_pairs(class_path)
    if class_csv:
        manifest['rsa_class'] = class_path.stem.removeprefix('class_')
    if dis_method:
        manifest['dis_method'] = dis_method
    if mah_fold:
        manifest['mah_fold'] = mah_fold
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = contribution.run(manifest, [root] + list(sources), output, work,
                             device=device, voxel_batch=voxel_batch,
                             comparison_batch=comparison_batch, replace_file=force,
                             include_baseline=True, save_participant_maps=save_participant_maps, load_workers=load_workers)
    if before_archive is not None:
        before_archive()
    # Pipeline-relative archive members can be merged with tools/unpack_results.py.
    destination = output / (f"result_stimulus_contribution_{manifest['dataset']}_{manifest['rsa_class']}_"
                            f"{manifest['specie']}_r-{manifest['radius']}_{manifest['dis_method']}_"
                            f"{manifest['mah_fold']}_{manifest['rsa_method']}.zip")
    partial = Path(str(destination) + '.part')
    print(f'Writing results ZIP to Drive/output folder: {destination}', flush=True)
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for mean in paths:
            for suffix in ('.nii.gz', '_std.nii.gz', '.json'):
                path = Path(mean.removesuffix('.nii.gz') + suffix)
                archive.write(path, path.relative_to(output).as_posix())
        if save_participant_maps:
            for path in contribution.participant_output_paths(manifest, output):
                archive.write(path, path.relative_to(output).as_posix())
        for path in contribution.baseline_output_paths(manifest, output):
            archive.write(path, path.relative_to(output).as_posix())
    partial.replace(destination)
    print(f'Completed steps 11.2, 11.3 and {len(paths)} step-11.31 omission maps. Results: {destination}', flush=True)
    return str(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pkg', required=True)
    parser.add_argument('--sources', nargs='*', default=[])
    parser.add_argument('--out', required=True)
    parser.add_argument('--work', default='/content/stimulus_work')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--voxel-batch', type=int, default=128)
    parser.add_argument('--comparison-batch', type=int, default=4096)
    parser.add_argument('--load-workers', type=int, default=1, help='Parallel session RDM loaders')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--save-participant-maps', '--save_participant_maps', action='store_true',
                        help='Save and archive participant omission maps (sessions averaged, runs separate)')
    parser.add_argument('--class-csv')
    parser.add_argument('--dis_method', choices=['mahalanobis', 'correlation'])
    parser.add_argument('--mah_fold')
    a = parser.parse_args()
    run_package(a.pkg, a.sources, a.out, a.work, a.device, a.voxel_batch, a.comparison_batch, a.force,
                a.class_csv, a.dis_method, a.mah_fold, a.save_participant_maps, a.load_workers)

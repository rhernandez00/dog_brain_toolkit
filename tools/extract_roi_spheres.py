r"""Run step 12 for every registered ROI, in shuffled order within each species.

Example (from any directory):
    python C:\github\dog_brain_toolkit\tools\extract_roi_spheres.py

The ROI CSV stays unchanged. Existing sphere outputs are skipped by step 12.
Rows may supply voxel_coords or, when absent, world-mm columns x, y, z.
Millimetres use each species' selected mask affine (human MNI / dog Nitzsche).
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys

import pandas as pd


def default_paths(platform=None):
    """Use the same Windows/Linux mount layout as searchlight.py."""
    if (os.name if platform is None else platform) == 'nt':
        return r'P:\userdata\raulh87\data', r'C:\github\dog_brain_toolkit'
    base = '/home/raulh87/mnt/a471/userdata/raulh87'
    return f'{base}/data', f'{base}/github/dog_brain_toolkit'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_datafolder, default_toolkit = default_paths()
    parser.add_argument('--datafolder', type=Path, default=default_datafolder)
    parser.add_argument('--toolkit-dir', type=Path, default=default_toolkit)
    parser.add_argument('--dataset', default='EmoB')
    parser.add_argument('--model', default='basic-block')
    parser.add_argument('--rsa-model', default='only-body')
    parser.add_argument('--species', nargs='+', choices=['D', 'H'], default=['D', 'H'])
    parser.add_argument('--dog-radius', type=int, default=3)
    parser.add_argument('--human-radius', type=int, default=4)
    parser.add_argument('--mask-type', default='b_GreyMatter2mmB')
    parser.add_argument('--seed', type=int, help='Repeatable ROI shuffle; omitted means a new order each run.')
    args = parser.parse_args(argv)
    datafolder = args.datafolder.resolve()
    toolkit = args.toolkit_dir.resolve()
    script = toolkit / 'searchlight.py'
    database_path = datafolder / args.dataset / 'ROI' / 'roi_database.csv'
    database = pd.read_csv(database_path)
    required = {'specie', 'peak_id'}
    if required - set(database.columns):
        raise ValueError(f'ROI database is missing columns: {sorted(required - set(database.columns))}')
    if 'voxel_coords' not in database and not {'x', 'y', 'z'}.issubset(database.columns):
        raise ValueError('ROI database needs voxel_coords or all three world-mm columns x, y, z.')
    if not script.is_file():
        raise FileNotFoundError(script)
    radii = {'D': args.dog_radius, 'H': args.human_radius}
    if min(radii.values()) <= 0:
        raise ValueError('Radii must be positive.')
    species = list(dict.fromkeys(args.species))
    for specie in species:
        if not (database['specie'] == specie).any():
            raise ValueError(f'No ROI entries found for species {specie}')
        for path in (
            datafolder / args.dataset / 'config_files' / f'{specie}_{args.model}.yaml',
            datafolder / args.dataset / 'BIDS' / f'{specie}_database-details.csv',
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

    print(f'ROI database: {database_path} ({len(database)} rows)', flush=True)
    print('ROI rows shuffled within each species; CSV row numbers are one-based data rows.', flush=True)
    for specie in species:
        count = int((database['specie'] == specie).sum())
        print(f'\nStep 12: {specie}, {count} rows, correlation, radius {radii[specie]}', flush=True)
        command = [
            sys.executable, '-u', str(script),
            '--datafolder', str(datafolder), '--toolkit_dir', str(toolkit),
            '--dataset', args.dataset, '--task', args.dataset,
            '--specie', specie, '--model', args.model,
            '--rsa_model', args.rsa_model, '--dis_method', 'correlation',
            '--radius', str(radii[specie]), '--mask_type', args.mask_type,
            '--steps_to_run', '12', '--shuffle_rois', '--verbose',
        ]
        if args.seed is not None:
            command.extend(['--roi_seed', str(args.seed)])
        # Inherit terminal output directly so the current ROI and run stay visible.
        subprocess.run(command, cwd=toolkit, check=True)
    print('\nStep 12 finished. Check the output for per-run extraction errors.', flush=True)


if __name__ == '__main__':
    main()

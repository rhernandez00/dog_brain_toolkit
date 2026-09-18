#!/usr/bin/env python
"""Package Colab regression steps 15/15.3/15.4/15.5, or supplement old packages.

Existing Drive packages (small support ZIP; no imaging data duplicated):
  python tools/create_regression_package.py --dataset EmoB --specie H D \
    --from_packages "G:/My Drive/rsa_colab/pkg_EmoB" --regression_model visual_3 \
    --out tools/colab_gpu/packages/regression_EmoB

New dataset (aligned beta packages + models/controls/masks/code/notebook):
  python tools/create_regression_package.py --dataset EmoC --specie H \
    --regression_model visual_3 --models TARGET --out UPLOAD_FOLDER
"""
import argparse
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for location in (HERE, REPO):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

COLAB = HERE / 'colab_gpu'


def build_support(packages_dir, out_dir, regression_model, controls, *, dataset='EmoB',
                  species=('H',), models=None, participants=None, model='basic-block',
                  dis_method='correlation'):
    """Read only small ZIP members. Existing large input ZIPs stay in place."""
    from colab_gpu.run_colab_regression import discover_packages
    controls = list(dict.fromkeys(controls))
    if not controls:
        raise ValueError('The regression model lists no controls')
    payload, by_species, targets = {}, {}, None
    csv_cache = {}
    def add(name, value):
        if name in payload and payload[name] != value:
            raise ValueError(f'Packages contain different versions of {name}; rebuild consistent inputs')
        payload[name] = value
    for specie in species:
        packages = discover_packages(packages_dir, specie, dataset, participants=participants,
                                     model=model, dis_method=dis_method)
        by_species[specie] = list(packages)
        for path, manifest in packages.values():
            names = list(models) if models is not None else [n for n in manifest['models'] if n not in controls]
            if targets is None:
                targets = names
            if names != targets:
                raise ValueError('Package target lists differ; specify a common --models list')
            if not names or set(names) & set(controls):
                raise ValueError('Need target models distinct from the controls')
            with zipfile.ZipFile(path) as zf:
                members = set(zf.namelist())
                if participants is None:
                    config_path = f'data/{dataset}/config_files/{specie}_{model}.yaml'
                    config = yaml.safe_load(zf.read(config_path))
                    expected = {int(p) for p in config['participants']}
                    if expected != set(packages):
                        raise ValueError(f'{specie}: configured participants differ from available packages; '
                                         f'missing={sorted(expected - set(packages))}, '
                                         f'extra={sorted(set(packages) - expected)}. '
                                         'Use --participants only for an intentional subset.')
                mask = f"data/{dataset}/ROI/{specie}/{manifest['mask_type']}.nii.gz"
                add(mask, zf.read(mask))
                for run in sorted({e['run_N'] for e in manifest['runs']}):
                    frames = {}
                    for name in [*names, *controls]:
                        csv = f'data/{dataset}/rsa_models/{name}.csv'
                        if csv not in members:
                            csv = f'data/{dataset}/rsa_models/{name}-run-{run}.csv'
                        if csv not in members:
                            raise FileNotFoundError(f'{path.name}: missing {name} for run {run}; rebuild the input package including this model')
                        info = zf.getinfo(csv)
                        cache_key = (csv, info.CRC, info.file_size)
                        if cache_key not in csv_cache:
                            content = zf.read(csv)
                            add(csv, content)
                            csv_cache[cache_key] = pd.read_csv(io.BytesIO(content), index_col=0)
                        frames[name] = csv_cache[cache_key]
                    for name in names:
                        labels = set(frames[name].columns)
                        if labels - set(manifest['categories']):
                            raise ValueError(f'{name}: model categories absent from {path.name} neural data')
                        for control in controls:
                            if labels - (set(frames[control].columns) & set(frames[control].index)):
                                raise ValueError(f'{control} does not cover the categories of target {name}')
        print(f'{specie}: {len(packages)} participant packages validated')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = dict(kind='regression_support', version=1, dataset=dataset, model=model,
                    dis_method=dis_method, regression_model=regression_model, controls=controls,
                    models=targets, participants_by_species=by_species)
    output = out_dir / f'regression_support_{dataset}_{regression_model}.zip'
    partial = output.with_suffix('.zip.part')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, content in sorted(payload.items()):
            zf.writestr(name, content)
        zf.writestr(f'data/{dataset}/rsa_models/regression_models/{regression_model}.csv',
                    '\n'.join(controls) + '\n')
        zf.writestr('regression_manifest.json', json.dumps(manifest, indent=2))
        for name in ('gpu_regression.py', 'run_colab_regression.py', 'gpu_rsa.py'):
            zf.write(COLAB / name, f'code/{name}')
        zf.write(COLAB / 'colab_rsa_regression.ipynb', 'colab_rsa_regression.ipynb')
    partial.replace(output)
    # Loose notebook for easy opening in Colab; code is bootstrapped from ZIP.
    shutil.copyfile(COLAB / 'colab_rsa_regression.ipynb', out_dir / 'colab_rsa_regression.ipynb')
    print(f'Support package: {output} ({output.stat().st_size / 1e6:.2f} MB)')
    print(f'Targets: {targets}\nControls: {controls}')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default='EmoB')
    parser.add_argument('--specie', nargs='+', choices=['H', 'D'], default=['H'])
    parser.add_argument('--model', default='basic-block')
    parser.add_argument('--regression_model', default='visual_3')
    parser.add_argument('--controls', nargs='+', help='Explicit controls; otherwise read the regression_model CSV')
    parser.add_argument('--models', nargs='+', help='Targets; defaults to packaged models minus controls, or the dataset catalogue for new packages')
    parser.add_argument('--participants', nargs='+', type=int)
    parser.add_argument('--from_packages', type=Path, help='Reuse existing participant ZIPs without copying their imaging data')
    parser.add_argument('--out', type=Path, default=COLAB / 'packages/regression')
    parser.add_argument('--dis_method', choices=['correlation', 'mahalanobis'], default='correlation')
    parser.add_argument('--radius', type=int, default=None)
    parser.add_argument('--mask_type', default='b_GreyMatter2mmB')
    parser.add_argument('--reps', type=int, default=100)
    args = parser.parse_args()
    from scheduler.paths import get_paths
    from create_package import build_package, resolve_models
    from rsa_utils import _read_regression_model_names
    datafolder, _, _ = get_paths()
    controls = args.controls or _read_regression_model_names(str(
        Path(datafolder) / args.dataset / 'rsa_models/regression_models' / f'{args.regression_model}.csv'))
    packages_dir = args.from_packages
    targets = args.models
    if packages_dir is None:
        packages_dir = args.out / 'packages'
        for specie in args.specie:
            config = Path(datafolder) / args.dataset / 'config_files' / f'{specie}_{args.model}.yaml'
            cfg = yaml.safe_load(config.read_text())
            participants = args.participants or cfg['participants']
            if targets is None:
                from create_package import get_runs
                runs = [e['run_N'] for sub in participants for e in get_runs(datafolder, args.dataset, specie, sub)]
                all_names, _, _ = resolve_models(datafolder, args.dataset, [], True, False, args.dis_method, runs)
                targets = [n for n in all_names if n not in controls]
            for sub in participants:
                build_package(specie, int(sub), targets + controls, False, False, args.dataset,
                              args.model, args.radius, args.dis_method, 'stim-wise', 'kendall',
                              args.reps, args.mask_type, str(packages_dir))
    build_support(packages_dir, args.out, args.regression_model, controls,
                  dataset=args.dataset, species=args.specie, models=targets,
                  participants=args.participants, model=args.model, dis_method=args.dis_method)


if __name__ == '__main__':
    main()

"""Build a portable step-11.31 Colab GPU package (support-only or with maps)."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import stimulus_contribution as contribution


def build_package(output, *, specie='H', dataset='EmoC', model='basic-block',
                  radius=None, rsa_class='all', rsa_method='kendall',
                  mask_type='b_GreyMatter2mmB', datafolder=None,
                  refs=None, participants=None, include_maps=False, replace=False,
                  class_csv=None, dis_method='mahalanobis', mah_fold='stim-wise'):
    """Offline defaults use the committed EmoC config/database/mask snapshot.

    Pairs and possible omissions always come from the authoritative class CSV.
    Without a class CSV the support package requests one in the notebook.
    A support package consumes existing step-1 ZIPs
    or an unpacked data tree on Colab; it never estimates missing distances.
    """
    refs = Path(refs or REPO / 'tools/colab_gpu/refs' / dataset)
    root = Path(datafolder) / dataset if datafolder else None
    config = (root / 'config_files' / f'{specie}_{model}.yaml' if root else
              refs / f'{specie}_{model}.yaml')
    cfg = yaml.safe_load(config.read_text())
    radius = radius if radius is not None else (3 if specie == 'D' else 4)
    selected = list(participants or cfg['participants'])
    if set(selected) - set(cfg['participants']):
        raise ValueError('Participants must belong to the config cohort.')
    database = pd.read_csv(root / 'BIDS' / f'{specie}_database-details.csv' if root else
                           refs / f'{specie}_database-details.csv')
    runs = {str(s): [dict(session=int(row.session), run_N=int(row.run_N))
                     for row in database[database.sub_N == s].itertuples()] for s in selected}
    if any(not entries for entries in runs.values()):
        raise ValueError('Database has no runs for one or more selected participants.')
    class_name = rsa_class.removeprefix('class_')
    class_path = Path(class_csv) if class_csv else (root / 'rsa_models/by_class' / f'class_{class_name}.csv' if root else None)
    pairs = contribution.read_pairs(class_path) if class_path else []
    stimuli = contribution.validate_pairs(pairs)[1] if pairs else []
    pair_origin = str(class_path) if class_path else 'class CSV supplied in the notebook'
    mask = root / 'ROI' / specie / f'{mask_type}.nii.gz' if root else refs / f'{specie}_{mask_type}.nii.gz'
    manifest = dict(version=contribution.VERSION, kind='stimulus-contribution-11.31',
                    dataset=dataset, model=model, specie=specie, task=cfg.get('task', dataset),
                    radius=radius, mask='inputs/mask.nii.gz', mask_type=mask_type,
                    dis_method=dis_method, mah_fold=mah_fold, rsa_class=class_name,
                    rsa_method=rsa_method, participants=selected, runs_by_sub=runs,
                    pairs=pairs, pair_origin=pair_origin, min_percentage_available=0,
                    config_source=str(config), includes_maps=include_maps,
                    class_csv='inputs/class.csv' if class_path else None)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not replace:
        raise FileExistsError(f'Package already exists: {output}')
    pending = {}
    if include_maps:
        if not datafolder or not pairs:
            raise ValueError('--include-maps requires --datafolder and a class CSV.')
        store = contribution.MapStore([datafolder])
        ref = contribution.nib.load(str(mask))
        for s in selected:
            entries = runs[str(s)] if dis_method == 'correlation' or mah_fold == 'stim-wise-all-runs' else [dict(session=None, run_N=None)]
            for e in entries:
                for source in contribution.pair_sources(manifest, store, s, e['session'], e['run_N']):
                    if not isinstance(source, str):
                        raise ValueError('For bundling maps, provide an unpacked datafolder.')
                    contribution.check_grid(store.load(source), ref, source)
                    pending[source] = Path(source).relative_to(datafolder).as_posix()
    partial = Path(str(output) + '.part')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
        archive.write(mask, 'inputs/mask.nii.gz')
        archive.write(config, 'inputs/config.yaml')
        if class_path:
            archive.write(class_path, 'inputs/class.csv')
        archive.write(REPO / 'stimulus_contribution.py', 'stimulus_contribution.py')
        archive.write(REPO / 'tools/colab_gpu/run_colab_stimulus_contribution.py', 'run_colab_stimulus_contribution.py')
        archive.write(REPO / 'tools/colab_gpu/colab_stimulus_contribution.ipynb', 'colab_stimulus_contribution.ipynb')
        archive.write(REPO / 'docs/stimulus_contribution_colab.md', 'README.md')
        for source, relative in pending.items():
            archive.write(source, 'data/' + relative)
    partial.replace(output)
    notebook = output.parent / 'colab_stimulus_contribution.ipynb'
    if notebook.resolve() != (REPO / 'tools/colab_gpu/colab_stimulus_contribution.ipynb').resolve():
        shutil.copy2(REPO / 'tools/colab_gpu/colab_stimulus_contribution.ipynb', notebook)
    print(f'Created {output}: {len(selected)} participants, {len(stimuli) if pairs else "CSV-defined"} omissions, '
          f'{len(pairs)} pairs, {len(pending)} bundled maps.', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', dest='output', required=True)
    parser.add_argument('--specie', choices=['H', 'D'], default='H')
    parser.add_argument('--dataset', default='EmoC')
    parser.add_argument('--model', default='basic-block')
    parser.add_argument('--radius', type=int)
    parser.add_argument('--rsa_class', default='all')
    parser.add_argument('--rsa_method', choices=['kendall', 'pearson', 'spearman'], default='kendall')
    parser.add_argument('--mask_type', default='b_GreyMatter2mmB')
    parser.add_argument('--datafolder')
    parser.add_argument('--class-csv', help='Authoritative rsa_class CSV with cat1/cat2 columns')
    parser.add_argument('--dis_method', choices=['mahalanobis', 'correlation'], default='mahalanobis')
    parser.add_argument('--mah_fold', default='stim-wise', choices=['stim-wise', 'run-wise', 'stim-wise-all-runs', 'stim-wise-multiple-folds'])
    parser.add_argument('--refs')
    parser.add_argument('--participants', nargs='+', type=int)
    parser.add_argument('--include-maps', action='store_true')
    parser.add_argument('--replace', action='store_true', help='Atomically replace an existing package')
    build_package(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()

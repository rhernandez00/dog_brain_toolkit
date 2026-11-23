"""Build a self-contained step-11.31 Colab GPU ZIP, including all input maps."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import threading
import time
import zipfile

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import stimulus_contribution as contribution


class _Progress:
    """Flushed phase progress, including a heartbeat during blocking file I/O."""
    def __init__(self, phase, interval=10):
        self.phase, self.interval = phase, interval
        self.started = self.last_print = time.monotonic()
        self.status = phase
        self.stop = threading.Event()
        self.lock = threading.Lock()

    def _print(self, message):
        elapsed = time.monotonic() - self.started
        print(f'[{self.phase} | {elapsed:.0f}s] {message}', flush=True)
        self.last_print = time.monotonic()

    def update(self, message, announce=False):
        with self.lock:
            self.status = message
            if announce or time.monotonic() - self.last_print >= self.interval:
                self._print(message)

    def _heartbeat(self):
        while not self.stop.wait(self.interval):
            with self.lock:
                if time.monotonic() - self.last_print >= self.interval:
                    self._print('Still working: ' + self.status)

    def __enter__(self):
        self.update('Starting.', announce=True)
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join()
        self.update(('Failed: ' if exc_type else 'Finished: ') + self.status, announce=True)



def read_database_runs(database_path, participants):
    """Use only session/run pairs explicitly recorded for each participant.

    When present, the bold flag is authoritative: FALSE or missing availability
    excludes a row. Legacy CSVs without that column use their recorded rows.
    """
    database = pd.read_csv(database_path)
    columns = ['sub_N', 'session', 'run_N']
    if not set(columns).issubset(database.columns):
        raise ValueError('Database details CSV requires sub_N, session and run_N columns.')
    if 'bold' in database.columns:
        flags = database['bold'].astype('string').str.strip().str.casefold().fillna('false')
        true_values = {'true', 't', 'yes', 'y', '1', '1.0'}
        false_values = {'false', 'f', 'no', 'n', '0', '0.0', ''}
        invalid = set(flags) - true_values - false_values
        if invalid:
            raise ValueError(f'Unrecognized BOLD availability values in database: {sorted(invalid)}')
        database = database.loc[flags.isin(true_values)]
    records = database[columns].copy()
    for column in columns:
        values = pd.to_numeric(records[column], errors='raise')
        if values.isna().any() or (values < 1).any() or (values % 1 != 0).any():
            raise ValueError(f'Database {column} values must be positive integers.')
        records[column] = values.astype(int)
    records = records.drop_duplicates().sort_values(columns)
    return {str(sub): [dict(session=int(row.session), run_N=int(row.run_N))
                       for row in records[records.sub_N == sub].itertuples()]
            for sub in participants}

def build_package(output, *, specie='H', dataset='EmoC', model='basic-block',
                  radius=None, rsa_class='all', rsa_method='kendall',
                  mask_type='b_GreyMatter2mmB', datafolder=None,
                  refs=None, participants=None, include_maps=True, replace=False,
                  class_csv=None, dis_method='mahalanobis', mah_fold='stim-wise',
                  database_details=None):
    """Bundle live config, database, class CSV, mask, pair maps and Colab scripts.

    Pairs and possible omissions always come from the authoritative class CSV.
    The normal package needs no additional inputs on Drive. Explicit
    include_maps=False enables the optional small support-only package.
    Missing required data is fatal; never silently fall back to a snapshot.
    """
    if include_maps and datafolder is None:
        from scheduler.paths import get_paths
        datafolder = get_paths()[0]
    if datafolder is not None:
        datafolder = Path(datafolder).expanduser().resolve()
    refs = Path(refs or REPO / 'tools/colab_gpu/refs' / dataset)
    root = Path(datafolder) / dataset if datafolder else None
    config = (root / 'config_files' / f'{specie}_{model}.yaml' if root else
              refs / f'{specie}_{model}.yaml')
    print(f'Reading config: {config}', flush=True)
    cfg = yaml.safe_load(config.read_text())
    radius = radius if radius is not None else (3 if specie == 'D' else 4)
    selected = list(participants or cfg['participants'])
    if set(selected) - set(cfg['participants']):
        raise ValueError('Participants must belong to the config cohort.')
    database_path = (Path(database_details) if database_details is not None else
                     root / 'BIDS' / f'{specie}_database-details.csv' if root else
                     refs / f'{specie}_database-details.csv')
    database_path = database_path.expanduser().resolve()
    print(f'Reading participant/run database: {database_path}', flush=True)
    runs = read_database_runs(database_path, selected)
    for sub in selected:
        print(f'Database units for {specie}-sub-{sub:02d}: '
              f'{[(e["session"], e["run_N"]) for e in runs[str(sub)]]}', flush=True)
    unavailable = [int(sub) for sub in selected if not runs[str(sub)]]
    if unavailable:
        raise ValueError(f'Database {database_path} has no BOLD-available runs for participants: {unavailable}')
    class_name = rsa_class.removeprefix('class_')
    class_path = Path(class_csv) if class_csv else (root / 'rsa_models/by_class' / f'class_{class_name}.csv' if root else None)
    pairs = contribution.read_pairs(class_path) if class_path else []
    stimuli = contribution.validate_pairs(pairs)[1] if pairs else []
    pair_origin = str(class_path) if class_path else 'class CSV supplied to the direct runner'
    mask = root / 'ROI' / specie / f'{mask_type}.nii.gz' if root else refs / f'{specie}_{mask_type}.nii.gz'
    if not mask.is_file() and specie == 'D':
        mask = REPO / 'Atlas/Dog/Nitzsche' / f'{mask_type}.nii.gz'
    manifest = dict(version=contribution.VERSION, kind='stimulus-contribution-11.31',
                    steps=[11.2, 11.3, 11.31],
                    dataset=dataset, model=model, specie=specie, task=cfg.get('task', dataset),
                    radius=radius, mask='inputs/mask.nii.gz', mask_type=mask_type,
                    dis_method=dis_method, mah_fold=mah_fold, rsa_class=class_name,
                    rsa_method=rsa_method, participants=selected, runs_by_sub=runs,
                    pairs=pairs, pair_origin=pair_origin, min_percentage_available=0,
                    config_source=str(config), database_source=str(database_path),
                    includes_maps=include_maps,
                    class_csv='inputs/class.csv' if class_path else None)
    output = Path(output)
    if output.suffix.lower() != '.zip':
        output = Path(str(output) + '.zip')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not replace:
        raise FileExistsError(f'Package already exists: {output}')
    if output.is_dir():
        raise IsADirectoryError(output)
    pending = {}
    if include_maps:
        if not datafolder or not pairs:
            raise ValueError('--include-maps requires --datafolder and a class CSV.')
        store = contribution.MapStore([datafolder], index_archives=False)
        print(f'Input data: {datafolder}\nClass CSV: {class_path}\n'
              f'Locating maps for {len(selected)} participants...', flush=True)
        by_sub = {s: (runs[str(s)] if dis_method == 'correlation' or mah_fold == 'stim-wise-all-runs'
                       else [dict(session=None, run_N=None)]) for s in selected}
        total_maps = sum(len(entries) for entries in by_sub.values()) * len(pairs)
        completed = 0
        with _Progress('Locate') as progress:
            progress.update(f'{total_maps} maps across {len(selected)} participants; '
                            f'{len(pairs)} pairs per session/run.', announce=True)
            for sub_index, s in enumerate(selected, 1):
                for unit_index, e in enumerate(by_sub[s], 1):
                    unit = (f'{specie}-sub-{s:02d} ({sub_index}/{len(selected)} participants), '
                            f'session {e["session"]}, run {e["run_N"]} '
                            f'({unit_index}/{len(by_sub[s])} units)')
                    progress.update(f'{unit}: locating {len(pairs)} pair maps; '
                                    f'{completed}/{total_maps} located.', announce=True)

                    def locating(index, count, a, b):
                        progress.update(f'{unit}: locating pair {index}/{count} ({a}, {b}); '
                                        f'{completed}/{total_maps} located.')

                    sources = contribution.pair_sources(manifest, store, s, e['session'], e['run_N'],
                                                        progress=locating)
                    for pair_index, source in enumerate(sources, 1):
                        if not isinstance(source, str):
                            raise ValueError('For bundling maps, provide an unpacked datafolder.')
                        pending[source] = Path(source).relative_to(datafolder).as_posix()
                        completed += 1
                        progress.update(f'{unit}: located {pair_index}/{len(sources)}; '
                                        f'overall {completed}/{total_maps} ({100 * completed / total_maps:.1f}%).',
                                        announce=pair_index == len(sources) or completed % 100 == 0)
    manifest['input_map_count'] = len(pending)
    manifest['input_map_bytes'] = 0
    with _Progress('Inventory') as progress:
        for index, source in enumerate(pending, 1):
            progress.update(f'Reading file size {index}/{len(pending)}: {source}')
            manifest['input_map_bytes'] += Path(source).stat().st_size
        progress.update(f'{len(pending)} unique maps, {manifest["input_map_bytes"] / 1e9:.2f} GB.', announce=True)
    print(f'Writing {output}: {len(pending)} pairwise maps '
          f'({manifest["input_map_bytes"] / 1e9:.2f} GB before ZIP compression).', flush=True)
    partial = Path(str(output) + '.part')
    with _Progress('ZIP') as progress, zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
        progress.update('Writing bundled metadata, mask and scripts.', announce=True)
        archive.writestr('manifest.json', json.dumps(manifest, indent=2))
        archive.write(mask, 'inputs/mask.nii.gz')
        archive.write(config, 'inputs/config.yaml')
        archive.write(database_path, 'inputs/database-details.csv')
        if class_path:
            archive.write(class_path, 'inputs/class.csv')
        archive.write(REPO / 'stimulus_contribution.py', 'stimulus_contribution.py')
        archive.write(REPO / 'tools/colab_gpu/run_colab_stimulus_contribution.py', 'run_colab_stimulus_contribution.py')
        archive.write(REPO / 'tools/colab_gpu/launch_stimulus_contribution.py', 'launch_stimulus_contribution.py')
        archive.write(REPO / 'docs/stimulus_contribution_colab.md', 'README.md')
        for index, (source, relative) in enumerate(pending.items(), 1):
            progress.update(f'Writing map {index}/{len(pending)}: {source}')
            archive.write(source, 'data/' + relative)
            progress.update(f'Bundled {index}/{len(pending)} maps '
                            f'({100 * index / len(pending):.1f}%); '
                            f'ZIP written {archive.fp.tell() / 1e9:.2f} GB.',
                            announce=index == len(pending) or index % 100 == 0)
        progress.update('Finalizing ZIP directory.', announce=True)
    print(f'Finalizing package: {partial} -> {output}', flush=True)
    partial.replace(output)
    launcher = output.parent / 'launch_stimulus_contribution.py'
    if launcher.resolve() != (REPO / 'tools/colab_gpu/launch_stimulus_contribution.py').resolve():
        shutil.copy2(REPO / 'tools/colab_gpu/launch_stimulus_contribution.py', launcher)
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
    parser.add_argument('--datafolder', help='Pipeline data root (default: scheduler.paths.get_paths for this machine)')
    parser.add_argument('--class-csv', help='Authoritative rsa_class CSV with cat1/cat2 columns')
    parser.add_argument('--database-details', help='Authoritative database CSV; default: <datafolder>/<dataset>/BIDS/<specie>_database-details.csv')
    parser.add_argument('--dis_method', choices=['mahalanobis', 'correlation'], default='mahalanobis')
    parser.add_argument('--mah_fold', default='stim-wise', choices=['stim-wise', 'run-wise', 'stim-wise-all-runs', 'stim-wise-multiple-folds'])
    parser.add_argument('--refs', help='Reference snapshot for --support-only without --datafolder')
    parser.add_argument('--participants', nargs='+', type=int)
    bundle = parser.add_mutually_exclusive_group()
    bundle.add_argument('--include-maps', dest='include_maps', action='store_true',
                        help='Bundle all required input maps (default)')
    bundle.add_argument('--support-only', dest='include_maps', action='store_false',
                        help='Explicitly build a small support ZIP; external input maps are then required')
    parser.set_defaults(include_maps=True)
    parser.add_argument('--replace', action='store_true', help='Atomically replace an existing package')
    build_package(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()

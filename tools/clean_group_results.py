#!/usr/bin/env python
"""Invalidate searchlight.py steps 3 and 5-10 after a cohort change.

Preview by default; --apply deletes the listed files. Never descends into
participant folders. Run with the workers stopped, then recompute steps 3,5-10.
"""
import argparse
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from scheduler.paths import get_paths


def component(value):
    """CLI values used as directory names must be single literal components."""
    if not value or value in ('.', '..') or any(c in value for c in '/\\:*?[]'):
        raise argparse.ArgumentTypeError('Expected a literal directory name')
    return value


def contained_file(path, root):
    """Reject symlinks/junction escapes; deletion is always file-by-file."""
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        raise ValueError(f'Refusing path outside cleanup root: {path}') from None
    if path.is_symlink():
        raise ValueError(f'Refusing symbolic link: {path}')
    return path.is_file()


def matching_files(folder, pattern, root):
    if not folder.exists():
        return []
    return [p for p in folder.iterdir()
            if pattern.fullmatch(p.name) and contained_file(p, root)]


def plan_cleanup(dataset_root, model, rsa_models, species, dis_method,
                 rsa_method, mirror_root=None, marker_dir=None):
    """Return existing files only; no writes and no scientific dependencies.

    All masks, radii and thresholds for the selected method pair are invalidated:
    step 6's cache is shared across masks/radii, and step 8 across masks/thresholds.
    Keeping any of those consumers would leave stale results after a cohort change.
    """
    results = dataset_root / 'results'
    files = set()
    for rsa_model in rsa_models:
        for specie in species:
            stem = (rf'(?:.+-)?{re.escape(specie)}-r-\d+_'
                    rf'{re.escape(dis_method)}_{re.escape(rsa_method)}_')
            # Include maps, logs, JSON provenance, CSVs and publication reports.
            pattern = re.compile(stem + r'.+')
            for tree, leaf in (('RSA', 'mean'), ('RSA', 'dist'), ('RSA_rnd', 'mean')):
                folder = results / tree / model / rsa_model / leaf
                files.update(matching_files(folder, pattern, dataset_root))
            # Step 6 is outside the per-RSA-model folder and omits method/radius.
            pattern = re.compile(re.escape(f'{specie}-{rsa_model}')
                                 + r'_(?:mean|std)(?:\.nii\.gz|_log\.txt)')
            files.update(matching_files(results / 'RSA_rnd' / model,
                                        pattern, dataset_root))
            if mirror_root is not None:
                # Step 10 mirrors omit GLM model, method and radius. Opt-in only.
                folder = mirror_root / dataset_root.name / 'current-results' / 'RSA'
                pattern = re.compile(re.escape(f'{specie}_{rsa_model}_zt') + r'.+')
                files.update(matching_files(folder, pattern, mirror_root))
                specie_folder = folder / specie
                if specie_folder.exists():
                    for mask_folder in specie_folder.iterdir():
                        if mask_folder.is_dir():
                            files.update(matching_files(mask_folder, pattern, mirror_root))
    if marker_dir is not None:
        for step in (3, 5, 6, 7, 8, 9, 10):
            marker = marker_dir / f'{step}.done'
            if contained_file(marker, marker_dir):
                files.add(marker)
    return sorted(files, key=str)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datafolder', type=Path, default=Path(get_paths()[0]))
    parser.add_argument('--dataset', type=component, required=True)
    parser.add_argument('--model', type=component, required=True, help='GLM model')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--rsa_model', type=component, nargs='+', help='One or more RSA model names')
    selection.add_argument('--all', action='store_true', help='All RSA model folders under this GLM model')
    parser.add_argument('--specie', nargs='+', choices=('D', 'H'), required=True)
    parser.add_argument('--dis_method', type=component, default='mahalanobis')
    parser.add_argument('--rsa_method', type=component, default='kendall')
    parser.add_argument('--mirror_root', type=Path,
                        help='Also clean step-10 mirrors, e.g. "G:\\My Drive\\Results"; mirrors share names across GLM models/methods')
    parser.add_argument('--job_marker_dir', type=Path,
                        help='Optional dedicated marker folder: clear only 3,5-10.done; queue records are untouched')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true', help='Delete the previewed files')
    mode.add_argument('--dry_run', '--dry-run', action='store_true', help='Preview only (default)')
    args = parser.parse_args(argv)
    dataset_root = (args.datafolder / args.dataset).absolute()
    if not dataset_root.is_dir():
        parser.error(f'Dataset directory does not exist: {dataset_root}')
    models = args.rsa_model
    if args.all:
        models = set()
        for tree in ('RSA', 'RSA_rnd'):
            folder = dataset_root / 'results' / tree / args.model
            if folder.exists():
                models.update(p.name for p in folder.iterdir() if p.is_dir())
        models = sorted(models)
    paths = plan_cleanup(dataset_root, args.model, models, args.specie,
                         args.dis_method, args.rsa_method, args.mirror_root,
                         args.job_marker_dir)
    print('Scope: steps 3,5-10; ALL masks, radii and thresholds for selected methods.')
    print('Participant results, permutations and queue records are preserved.')
    print('Stop workers before applying cleanup. Queue fresh group jobs afterwards.')
    print(f'{"DELETE" if args.apply else "PREVIEW"}: {len(paths)} files')
    for path in paths:
        print(path)
    if args.apply:
        # The complete plan is validated before the first mutation. Recheck each
        # path immediately before unlinking in case a directory was replaced.
        for path in paths:
            root = (args.job_marker_dir if args.job_marker_dir is not None
                    and path.parent == args.job_marker_dir else
                    args.mirror_root if args.mirror_root is not None
                    and path.is_relative_to(args.mirror_root) else dataset_root)
            if contained_file(path, root):
                path.unlink()
        print(f'Cleanup complete: {len(paths)} planned files removed.')
    else:
        print('No files changed. Repeat with --apply to delete.')
    print('Before rerunning 3 5 6 7 8 9 10, finish steps 2 and 4 for the new participant.')
    print('Step 5 needs every participant\'s step-4 maps; regenerate any purged by step 7.6.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

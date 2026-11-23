"""Unpack step-11.2/11.3/11.31 result ZIPs, adding species to by_stim names.

The archive is never modified. Species defaults to the result ZIP name and/or
its participant-map paths. --specie explicitly overrides omission labels only.
"""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import zipfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for folder in (HERE, REPO):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def archive_species(path, names):
    evidence = set()
    match = re.search(r'^result_stimulus_contribution_.+_([HD])_r-\d+(?:\.\d+)?_', Path(path).name)
    if match:
        evidence.add(match.group(1))
    for name in names:
        evidence.update(re.findall(r'(?:^|/)([HD])-sub-\d+', name.replace('\\', '/')))
    if len(evidence) > 1:
        raise ValueError(f'Conflicting archive species in {path}: {sorted(evidence)}. Select --specie explicitly.')
    return next(iter(evidence), None)


def safe_relative(name):
    path = PurePosixPath(name.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or any(':' in part for part in path.parts):
        raise ValueError(f'Unsafe ZIP member: {name!r}')
    return path


def corrected_name(name, specie, override=False):
    """Rename means, _std maps and JSON; leave ordinary baseline paths intact."""
    path = safe_relative(name)
    if 'by_stim' not in path.parts:
        return path
    if not path.name.endswith(('.nii.gz', '.json')):
        return path
    match = re.search(r'(?:^|(?<=-))r-\d+(?:\.\d+)?_', path.name)
    if match is None:
        raise ValueError(f'Cannot locate radius token in omission filename: {name}')
    prefix, tail = path.name[:match.start()], path.name[match.start():]
    tagged = re.search(r'(?:^|-)([HD])-$', prefix)
    if tagged:
        previous = tagged.group(1)
        if previous != specie and not override:
            raise ValueError(f'Omission filename species {previous} conflicts with archive species {specie}: {name}')
        prefix = prefix[:tagged.start(1)] + specie + '-'
    else:
        prefix += specie + '-'
    return path.with_name(prefix + tail)


def unpack_archive(source, datafolder, specie=None, replace=False, dry_run=False):
    """Preflight every target before copying; write each file atomically."""
    source, datafolder = Path(source), Path(datafolder).resolve()
    with zipfile.ZipFile(source) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        # Reject traversal even in an otherwise ignored member.
        relative_paths = [safe_relative(info.filename) for info in infos]
        inferred = None
        try:
            inferred = archive_species(source, [info.filename for info in infos])
        except ValueError:
            if specie is None:
                raise
        selected = specie or inferred
        if selected not in ('H', 'D'):
            raise ValueError(f'Cannot infer participant species from {source.name}; use --specie H or --specie D.')
        print(f'{source.name}: omission species={selected}' +
              (f' (explicit override; archive={inferred or "unknown"})' if specie else ' (from archive)'), flush=True)
        planned, seen = [], set()
        for info, original in zip(infos, relative_paths):
            if (len(original.parts) < 4 or original.parts[1:3] != ('results', 'RSA_same')
                    or not original.name.endswith(('.nii.gz', '.json'))):
                continue
            renamed = corrected_name(info.filename, selected, override=specie is not None)
            destination = datafolder.joinpath(*renamed.parts)
            if not destination.resolve().is_relative_to(datafolder):
                raise ValueError(f'Target escapes datafolder: {destination}')
            key = os.path.normcase(str(destination))
            if key in seen:
                raise ValueError(f'Multiple ZIP members map to the same target: {destination}')
            seen.add(key)
            planned.append((info, original, renamed, destination))
        if not planned:
            raise ValueError(f'No pipeline results found in {source}')
        renamed_count = sum(old != new for _, old, new, _ in planned)
        skipped = sum(path.is_file() and not replace for _, _, _, path in planned)
        print(f'  {len(planned)} result files; {renamed_count} species renames; {skipped} existing files skipped.', flush=True)
        if dry_run:
            for _, old, new, _ in [entry for entry in planned if entry[1] != entry[2]][:6]:
                print(f'  {old.name} -> {new.name}', flush=True)
            return dict(files=len(planned), renamed=renamed_count, copied=0, skipped=skipped, specie=selected)
        copied = 0
        for info, original, renamed, destination in planned:
            if destination.is_file() and not replace:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix='.' + destination.name + '.', suffix='.part', dir=destination.parent)
            try:
                with os.fdopen(fd, 'wb') as output:
                    if 'by_stim' in original.parts and original.name.endswith('.json'):
                        metadata = json.loads(archive.read(info))
                        metadata['specie'] = selected
                        metadata['source_archive'] = source.name
                        if specie is not None:
                            metadata['source_archive_specie'] = inferred
                            metadata['filename_species_override'] = True
                        output.write(json.dumps(metadata, indent=2).encode('utf-8'))
                    else:
                        with archive.open(info) as input_stream:
                            shutil.copyfileobj(input_stream, output, length=1024 * 1024)
                os.replace(temporary, destination)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
            copied += 1
            if copied % 100 == 0:
                print(f'  Copied {copied}/{len(planned) - skipped} files.', flush=True)
        print(f'  Finished: {copied} copied, {skipped} skipped -> {datafolder}', flush=True)
        return dict(files=len(planned), renamed=renamed_count, copied=copied, skipped=skipped, specie=selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', help='Result ZIP(s), or folders containing result_stimulus_contribution_*.zip')
    parser.add_argument('--datafolder', help='Destination pipeline data root; defaults to scheduler.paths.get_paths()')
    parser.add_argument('--specie', choices=['H', 'D'], help='Override the species label on by_stim outputs; default: infer from archive')
    parser.add_argument('--replace', action='store_true', help='Replace existing destination files')
    parser.add_argument('--dry-run', action='store_true', help='Preview without creating files or folders')
    args = parser.parse_args()
    if args.datafolder is None:
        from scheduler.paths import get_paths
        args.datafolder = get_paths()[0]
    archives = []
    for value in args.sources:
        path = Path(value)
        found = sorted(path.glob('result_stimulus_contribution_*.zip')) if path.is_dir() else [path]
        if not found:
            raise FileNotFoundError(f'No contribution result ZIPs found in {path}')
        archives.extend(found)
    for archive in dict.fromkeys(archives):
        unpack_archive(archive, args.datafolder, specie=args.specie, replace=args.replace, dry_run=args.dry_run)


if __name__ == '__main__':
    main()

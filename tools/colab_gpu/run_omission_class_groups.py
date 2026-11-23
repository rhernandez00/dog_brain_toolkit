"""Run additional step-11.2 classes using existing class_all omission checkpoints.

The original runner stays byte-identical so running jobs and checkpoint identities
remain compatible. Class selection affects only group comparisons and outputs.
"""
import argparse
import csv
from pathlib import Path
import run_colab_stimulus_omission as runner


def select_pairs(manifest, csv_path):
    categories = set(manifest['categories'])
    mapping = {s: s[:-1] for s in manifest['stim_types']}
    available = {frozenset(p) for p in manifest['pairs']}
    selected, seen = [], set()
    with open(csv_path, encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            pair = []
            for key in ('cat1', 'cat2'):
                label = row[key].strip()
                label = label if label in categories else mapping.get(label, label)
                if label not in categories:
                    raise ValueError(f'Unavailable class label: {label}')
                pair.append(label)
            identity = frozenset(pair)
            if len(identity) == 1 or identity in seen:
                continue
            if identity not in available:
                raise ValueError(f'Class pair is absent from source maps: {pair}')
            selected.append(pair)
            seen.add(identity)
    if len(selected) < 2:
        raise ValueError('Class needs at least two distinct category comparisons.')
    return selected


def run_class(packages, source, output, specie, rsa_class, csv_path, work, device='cuda'):
    original_records, original_digest = runner.records_for, runner.digest
    def source_records(package, unused_output):
        records = original_records(package, Path(source))
        manifest = package['manifest']
        manifest['comparison_pairs'] = select_pairs(manifest, csv_path)
        manifest['rsa_class'] = rsa_class
        return records
    def class_digest(value):
        if 'inputs' in value and 'method' in value:
            value['rsa_class'] = rsa_class
            value['class_csv_sha256'] = runner.hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()
        return original_digest(value)
    # Validate exact inclusion using the established full-cohort policy guard.
    from supervise_stimulus_omission import cohort_coverage
    cohort, _, _ = runner.discover_packages(packages, specie, 'EmoC', 'basic-block')
    _, fraction = cohort_coverage(cohort, specie, ['H:20:1'] if specie == 'H' else [])
    runner.records_for, runner.digest = source_records, class_digest
    try:
        return runner.run(packages, output, specie=specie, steps=(11.2,), work=work,
                          device=device, min_percentage_available=fraction)
    finally:
        runner.records_for, runner.digest = original_records, original_digest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('packages', 'source', 'output', 'class-csv', 'work'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--specie', choices=('D', 'H'), required=True)
    parser.add_argument('--rsa-class', choices=('hum', 'dog'), required=True)
    parser.add_argument('--device', default='cuda', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    run_class(args.packages, args.source, args.output, args.specie, args.rsa_class,
              args.class_csv, args.work, args.device)

"""Queue one low-priority human step-2 job per dog-derived ROI model."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scheduler.dag import build_single_job
from scheduler.jobs import ALL_STATES
from scheduler.paths import get_paths, get_queue_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create', action='store_true', help='Submit jobs; otherwise preview only')
    args = parser.parse_args()
    datafolder, _, _ = get_paths()
    dataset_dir = Path(datafolder) / 'EmoB'
    pattern = re.compile(r'^(LcSSG_mah|RmSSG_mah|RcSG_mah)_D-sub-\d+\.csv$')
    models = sorted(path.stem for path in (dataset_dir / 'rsa_models').iterdir()
                    if path.is_file() and pattern.fullmatch(path.name))
    counts = Counter(model.split('_D-sub-')[0] for model in models)
    if counts != {'LcSSG_mah': 24, 'RmSSG_mah': 24, 'RcSG_mah': 24}:
        raise ValueError(f'Expected 24 individual models per region; found {dict(counts)}')

    print(f'{len(models)} individual models = {len(models)} step-2 jobs, priority 3')
    if not args.create:
        return

    queue = get_queue_dir(datafolder)
    names = set()
    for state in ALL_STATES:
        names.update(path.name for path in (queue / state).iterdir()
                     if path.suffix == '.json')
    batch = 'individual-roi-step2-' + uuid.uuid4().hex
    receipt = Path(__file__).resolve().parents[1] / f'{batch}.json'
    details = {'batch_id': batch, 'queue': str(queue), 'models': models,
               'expected_jobs': len(models),
               'created_jobs': 0, 'priority': 3}
    receipt.write_text(json.dumps(details, indent=2), encoding='utf-8')
    submitted = []
    for model in models:
        job = build_single_job(dataset='EmoB', model='basic-block',
                               rsa_model=model, specie='H', step=2,
                               dis_method='mahalanobis', mah_fold='stim-wise',
                               priority=3, verbose=False, replace_file=False)
        stem = job['job_id']
        filename = f'{stem}.json'
        if filename in names:
            suffix = 2
            while f'{stem}__dup{suffix}.json' in names:
                suffix += 1
            filename = f'{stem}__dup{suffix}.json'
        job.update(submission_batch=batch,
                   created_at=datetime.utcnow().isoformat())
        destination = queue / 'pending' / filename
        try:
            with destination.open('x', encoding='utf-8') as handle:
                json.dump(job, handle, indent=2)
        except FileExistsError as exc:
            raise RuntimeError(f'Queue filename was claimed concurrently: {destination}') from exc
        names.add(filename)
        submitted.append(destination)
        details['created_jobs'] += 1
        receipt.write_text(json.dumps(details, indent=2), encoding='utf-8')
        print(f'Created job for {model}', flush=True)

    verified = 0
    for destination in submitted:
        for state in ALL_STATES:
            try:
                record = json.loads((queue / state / destination.name).read_text(encoding='utf-8'))
            except FileNotFoundError:
                continue
            if record.get('submission_batch') == batch:
                verified += 1
                break
    details['verified_jobs'] = verified
    receipt.write_text(json.dumps(details, indent=2), encoding='utf-8')
    print(f'Created {details["created_jobs"]}; verified {verified}. Receipt: {receipt}')


if __name__ == '__main__':
    main()

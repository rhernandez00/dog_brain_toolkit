"""Queue ten human step-4 jobs per dog-derived ROI model; dry-run by default."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scheduler.paths import get_paths, get_queue_dir
from scheduler.dag import build_single_job
from scheduler.jobs import create_job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create', action='store_true', help='Write jobs to the shared queue')
    args = parser.parse_args()
    datafolder, _, _ = get_paths()
    model_dir = Path(datafolder) / 'EmoB' / 'rsa_models'
    pattern = re.compile(r'^(LcSSG_mah|RmSSG_mah|RcSG_mah)_D-sub-\d+\.csv$')
    models = sorted(p.stem for p in model_dir.iterdir() if p.is_file() and pattern.fullmatch(p.name))
    counts = Counter(name.split('_D-sub-')[0] for name in models)
    if set(counts) != {'LcSSG_mah', 'RmSSG_mah', 'RcSG_mah'}:
        raise ValueError(f'Expected individual models for all three regions; found {dict(counts)}')
    repetitions = 10
    batch_id = 'individual-roi-step4-' + uuid.uuid4().hex
    jobs = []
    for model in models:
        for repetition in range(1, repetitions + 1):
            job = build_single_job(
                dataset='EmoB', model='basic-block', rsa_model=model,
                specie='H', step=4, dis_method='mahalanobis', mah_fold='stim-wise',
                priority=2, min_percentage_available=1.0, verbose=False,
                replace_file=False)
            job.update(shuffle_participants=True, submission_batch=batch_id,
                       submission_repetition=repetition)
            jobs.append(job)
    print(f'Models by region: {dict(counts)}')
    print(f'{len(models)} models x {repetitions} repetitions = {len(jobs)} jobs; target H, step 4')
    if not args.create:
        print('Dry run only. Pass --create to enqueue.')
        return
    queue = get_queue_dir(datafolder)
    receipt = Path(__file__).resolve().parents[1] / f'{batch_id}.json'
    details = dict(batch_id=batch_id, queue=str(queue), models=models,
                   expected_jobs=len(jobs), created_jobs=0, verified_jobs=0)
    receipt.write_text(json.dumps(details, indent=2), encoding='utf-8')
    for index, job in enumerate(jobs, 1):
        with redirect_stdout(io.StringIO()):
            if not create_job(queue, job):
                raise RuntimeError(f'Failed to create {job["job_id"]}')
        details['created_jobs'] = index
        receipt.write_text(json.dumps(details, indent=2), encoding='utf-8')
        if index % repetitions == 0:
            print(f'Created {repetitions} jobs for {job["rsa_model"]}', flush=True)
    # Workers may already have claimed jobs; inspect every queue state.
    found = set()
    for state in ('pending', 'waiting', 'running', 'completed', 'failed'):
        for path in (queue / state).glob('EmoB__basic-block__*.json'):
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
            except FileNotFoundError:
                continue  # a worker moved it while this state was scanned
            if record.get('submission_batch') == batch_id:
                found.add((record['rsa_model'], record['submission_repetition']))
    details['verified_jobs'] = len(found)
    receipt.write_text(json.dumps(details, indent=2), encoding='utf-8')
    print(f'Created {len(jobs)} jobs; verified {len(found)} queue records. Receipt: {receipt}')


if __name__ == '__main__':
    main()

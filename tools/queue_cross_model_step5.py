"""Queue 15 priority-1 step-5 jobs for each D-derived ROI model family."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scheduler.dag import build_single_job
from scheduler.jobs import ALL_STATES
from scheduler.paths import get_paths, get_queue_dir
from run_jobs import build_command
import rsa_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--create', action='store_true', help='Submit jobs; otherwise preview only')
    args = parser.parse_args()
    datafolder, git_folder, python_exe = get_paths()
    models = ['LcSSG_mah', 'RmSSG_mah', 'RcSG_mah']
    for model in models:
        sources = rsa_utils._source_rsa_models(datafolder, 'EmoB', model, 'D')
        if len(sources) != 24:
            raise ValueError(f'{model}: expected 24 D-derived individual models, found {len(sources)}')
    jobs = []
    for model in models:
        job = build_single_job(dataset='EmoB', model='basic-block', rsa_model=model,
                               specie='H', step=5, model_specie='D',
                               dis_method='mahalanobis', mah_fold='stim-wise',
                               priority=1, min_percentage_available=1.0,
                               verbose=False, replace_file=False)
        command = build_command(job, git_folder, python_exe, 'marker-check')
        if command[command.index('--model_specie') + 1] != 'D':
            raise RuntimeError(f'Scheduler did not forward --model_specie for {model}')
        for repetition in range(1, 16):
            jobs.append((dict(job), repetition))
    print(f'{len(models)} model families x 15 copies = {len(jobs)} H step-5 jobs, priority 1')
    if not args.create:
        return

    queue = get_queue_dir(datafolder)
    names = set()
    for state in ALL_STATES:
        names.update(path.name for path in (queue / state).iterdir() if path.suffix == '.json')
    batch = 'cross-model-step5-' + uuid.uuid4().hex
    receipt = Path(__file__).resolve().parents[1] / f'{batch}.json'
    details = {'batch_id': batch, 'queue': str(queue), 'models': models,
               'model_specie': 'D', 'specie': 'H', 'priority': 1,
               'expected_jobs': len(jobs), 'created_jobs': 0}
    receipt.write_text(json.dumps(details, indent=2), encoding='utf-8')
    submitted = []
    for job, repetition in jobs:
        stem = job['job_id']
        filename = f'{stem}.json'
        if filename in names:
            suffix = 2
            while f'{stem}__dup{suffix}.json' in names:
                suffix += 1
            filename = f'{stem}__dup{suffix}.json'
        job.update(shuffle_participants=True, submission_batch=batch,
                   submission_repetition=repetition,
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
    verified = 0
    for destination in submitted:
        for state in ALL_STATES:
            try:
                record = json.loads((queue / state / destination.name).read_text(encoding='utf-8'))
            except FileNotFoundError:
                continue
            if (record.get('submission_batch') == batch
                    and record.get('priority') == 1
                    and record.get('model_specie') == 'D'):
                verified += 1
                break
    details['verified_jobs'] = verified
    receipt.write_text(json.dumps(details, indent=2), encoding='utf-8')
    print(f'Created {details["created_jobs"]}; verified {verified}. Receipt: {receipt}')


if __name__ == '__main__':
    main()

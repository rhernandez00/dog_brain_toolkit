#!/usr/bin/env python
"""Schedule an ARBITRARY set of pipeline steps for many RSA models at once.

Why this exists: ``schedule_rsa.py`` can only queue a *contiguous* range
(``--start_step`` .. ``--target_step``), because it walks the dependency graph
backwards from the target. A request like "steps 3 and 5-10" (step 4 already
ran, or is running) is not expressible there.

This script does not compute or wire dependencies at all. Each requested
(model, species, step) is created as an independent job, ``status=pending``,
``deps=[]``, via ``build_single_job`` -- the caller is assumed to already know
the dependency graph and is asking for exactly these steps. If a step's real
inputs are missing on disk, ``searchlight.py`` raises and the job lands in
``failed/`` with the error -- that failure is the signal that a dependency was
missed, not a queue-side check.

Models come from the ``_models.csv`` registry: one row per hypothesis, with a
``groupings_possible`` column holding a stringified list, so the RSA model files
on disk are named ``{model}__{grouping}.csv``. Rows are filtered by
``--dis_method``.

Jobs are always created with ``verbose=True`` so each step runs under
``searchlight.py --verbose``.

Usage
-----
  python schedule_steps.py --steps 3,5-10 --dry_run     # preview + counts
  python schedule_steps.py --steps 3,5-10               # actually queue them
  python schedule_steps.py --steps 4 --rsa_model action_tendency__all
"""
import argparse
import ast
import csv
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root):
    if os.path.exists(os.path.join(_root, 'scheduler', 'paths.py')):
        break
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

from scheduler.paths import get_paths, get_queue_dir  # noqa: E402
from scheduler.dag import build_single_job  # noqa: E402
from scheduler.jobs import create_job, DEFAULT_PRIORITY, PRIORITIES  # noqa: E402


def read_registry(models_csv, dis_method=None):
    out = []
    with open(models_csv, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if dis_method and row['dis_method'] != dis_method:
                continue
            for g in ast.literal_eval(row['groupings_possible']):
                out.append({
                    'rsa_model': '%s__%s' % (row['model'], g),
                    'dis_method': row['dis_method'],
                    'mah_fold': row['mah_fold'],
                })
    return out


def parse_steps(spec):
    steps = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            steps.extend(range(int(a), int(b) + 1))
        elif part:
            steps.append(int(part))
    return sorted(set(steps))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='EmoC')
    p.add_argument('--model', default='basic-block')
    p.add_argument('--rsa_model', help='Schedule only this model (skips the registry)')
    p.add_argument('--models_csv')
    p.add_argument('--dis_method', default='mahalanobis')
    p.add_argument('--rsa_method', default='kendall')
    p.add_argument('--mah_fold', default='stim-wise')
    p.add_argument('--specie', default='D,H')
    p.add_argument('--steps', required=True, help="e.g. '3,5-10' or '4'")
    p.add_argument('--z_threshold', type=float, default=3.1)
    p.add_argument('--reps', type=int, default=100)
    p.add_argument('--reps_group', type=int, default=1000)
    p.add_argument('--min_percentage_available', type=float, default=1.0,
                   help='Minimum fraction of the dataset required to process the analysis')
    p.add_argument('--priority', type=int, default=DEFAULT_PRIORITY,
                   choices=list(PRIORITIES),
                   help='Queue priority: 1 runs first, 3 last (default: %d)'
                        % DEFAULT_PRIORITY)
    p.add_argument('--replace_rnd_files', action='store_true')
    p.add_argument('--dry_run', action='store_true',
                   help='Show what would be queued without writing anything')
    args = p.parse_args()

    datafolder, _, _ = get_paths()
    queue_dir = get_queue_dir(datafolder)
    models_csv = args.models_csv or os.path.join(
        datafolder, args.dataset, 'rsa_models', '_models.csv')

    if args.rsa_model:
        entries = [{'rsa_model': args.rsa_model, 'dis_method': args.dis_method,
                    'mah_fold': args.mah_fold}]
    else:
        entries = read_registry(models_csv, args.dis_method)

    steps = parse_steps(args.steps)
    species = [s.strip() for s in args.specie.split(',') if s.strip()]

    print('queue   : %s' % queue_dir)
    print('models  : %d   species: %s' % (len(entries), species))
    print('steps   : %s  (no dependency check -- each job is independent, status=pending)' % steps)
    print('params  : dis=%s rsa=%s mah=%s zt=%s r=%s rg=%s min_pct=%s verbose=True priority=%d'
          % (args.dis_method, args.rsa_method, args.mah_fold,
             args.z_threshold, args.reps, args.reps_group,
             args.min_percentage_available, args.priority))
    print('total   : %d jobs\n' % (len(entries) * len(species) * len(steps)))

    made = 0
    for e in entries:
        for sp in species:
            for step in steps:
                j = build_single_job(
                    dataset=args.dataset, model=args.model,
                    rsa_model=e['rsa_model'], specie=sp, step=step,
                    z_threshold=args.z_threshold, reps=args.reps,
                    reps_group=args.reps_group, rsa_method=args.rsa_method,
                    dis_method=e['dis_method'], mah_fold=e['mah_fold'],
                    min_percentage_available=args.min_percentage_available,
                    replace_rnd_files=args.replace_rnd_files,
                    verbose=True,
                    priority=args.priority,
                )
                if not args.dry_run:
                    create_job(queue_dir, j)
                made += 1
                if args.dry_run and made <= len(steps) * 2:
                    print('  [dry] %-46s %s step%02d -> %s'
                          % (j['rsa_model'], j['specie'], j['step'], j['status']))

    verb = 'would create' if args.dry_run else 'created'
    print('\n%s %d job(s), all pending' % (verb, made))
    if not args.dry_run:
        print("Run 'python run_jobs.py --max_jobs 0 --loop' to execute them.")


if __name__ == '__main__':
    main()

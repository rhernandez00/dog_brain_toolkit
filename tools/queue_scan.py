#!/usr/bin/env python
"""Fast state scan of the RSA job queue, per (rsa_model, specie, step).

Why this exists: the obvious approach -- glob the queue for each job id you care
about -- issues (models x species x steps x states) globs against a network
mount and takes minutes. 50 models x 2 species x 8 steps is 800 lookups; that
reliably blows past a two-minute timeout. This script instead does exactly ONE
listdir per state folder (5 total, ~13s for ~5600 files) and resolves every job
id from that in-memory index.

Usage
-----
  # every mahalanobis model in the registry, steps 3 and 5-10, both species
  python queue_scan.py --steps 3,5-10

  # a single model
  python queue_scan.py --rsa_model action_tendency__all --steps 4

  # summary counts only
  python queue_scan.py --steps 3,5-10 --summary
"""
import argparse
import ast
import csv
import os
import re
import sys

# Resolve the repo root by walking up until scheduler/paths.py appears, so this
# works no matter where the skill directory is relative to the checkout.
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root):
    if os.path.exists(os.path.join(_root, 'scheduler', 'paths.py')):
        break
    _root = os.path.dirname(_root)
sys.path.insert(0, _root)

from scheduler.paths import get_paths, get_queue_dir  # noqa: E402
from scheduler.dag import make_job_id, STEP_LABELS  # noqa: E402

# Precedence: a fresh in-flight instance outranks a stale completed/failed
# record for the same id (mirrors job_status.find_state).
STATES = ('running', 'pending', 'waiting', 'completed', 'failed')

GLYPH = {
    'running': 'RUN', 'pending': 'RDY', 'waiting': 'wait',
    'completed': 'OK', 'failed': 'FAIL', None: '-',
}

# {job_id}.json or {job_id}__dup{N}.json -> job_id
_FNAME = re.compile(r'^(?P<id>.+?)(?:__dup\d+)?\.json$')


def scan_queue(queue_dir):
    """Map job_id -> most-active state, with one listdir per state folder."""
    best = {}
    for state in STATES:
        d = os.path.join(queue_dir, state)
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            m = _FNAME.match(fn)
            if m:
                best.setdefault(m.group('id'), state)
    return best


def read_registry(models_csv, dis_method=None):
    """Expand _models.csv into one entry per (model, grouping) pair.

    groupings_possible is a stringified Python list, e.g. "['all', 'cross']",
    and the RSA model file on disk is named {model}__{grouping}.csv.
    """
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
    """'3,5-10' -> [3, 5, 6, 7, 8, 9, 10]"""
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
    p.add_argument('--rsa_model', help='Scan only this model (skips the registry)')
    p.add_argument('--models_csv', help='Default: {datafolder}/{dataset}/rsa_models/_models.csv')
    p.add_argument('--dis_method', default='mahalanobis')
    p.add_argument('--rsa_method', default='kendall')
    p.add_argument('--mah_fold', default='stim-wise')
    p.add_argument('--specie', default='D,H', help='Comma list (default: D,H)')
    p.add_argument('--steps', default='3,5-10')
    p.add_argument('--z_threshold', type=float, default=3.1)
    p.add_argument('--reps', type=int, default=100)
    p.add_argument('--reps_group', type=int, default=1000)
    p.add_argument('--summary', action='store_true', help='Counts only, no matrix')
    p.add_argument('--missing_only', action='store_true',
                   help='List only (model, specie, step) with no job in the queue')
    args = p.parse_args()

    datafolder, _, _ = get_paths()
    queue_dir = str(get_queue_dir(datafolder))
    models_csv = args.models_csv or os.path.join(
        datafolder, args.dataset, 'rsa_models', '_models.csv')

    if args.rsa_model:
        entries = [{'rsa_model': args.rsa_model, 'dis_method': args.dis_method,
                    'mah_fold': args.mah_fold}]
    else:
        entries = read_registry(models_csv, args.dis_method)

    steps = parse_steps(args.steps)
    species = [s.strip() for s in args.specie.split(',') if s.strip()]

    print('queue : %s' % queue_dir)
    print('models: %d   steps: %s   species: %s'
          % (len(entries), steps, species))
    print('params: dis=%s rsa=%s mah=%s zt=%s r=%s rg=%s\n'
          % (args.dis_method, args.rsa_method, args.mah_fold,
             args.z_threshold, args.reps, args.reps_group))

    index = scan_queue(queue_dir)

    counts = {}
    missing = []
    rows = []
    for e in entries:
        cells = []
        for step in steps:
            for sp in species:
                jid = make_job_id(
                    args.dataset, args.model, e['rsa_model'], sp, step,
                    args.z_threshold, args.reps, args.reps_group,
                    args.rsa_method, e['dis_method'], e['mah_fold'])
                state = index.get(jid)
                counts[state] = counts.get(state, 0) + 1
                cells.append((step, sp, state))
                if state is None:
                    missing.append((e['rsa_model'], sp, step))
        rows.append((e['rsa_model'], cells))

    if args.missing_only:
        print('Not in queue at all (%d):' % len(missing))
        for m, sp, st in missing:
            print('  %-46s %s step%02d' % (m, sp, st))
    elif not args.summary:
        head = '%-46s' % 'rsa_model'
        for step in steps:
            head += ' %-11s' % ('s%d(%s)' % (step, '/'.join(species)))
        print(head)
        print('-' * len(head))
        for name, cells in rows:
            line = '%-46s' % name
            by_step = {}
            for step, sp, state in cells:
                by_step.setdefault(step, []).append(GLYPH[state])
            for step in steps:
                line += ' %-11s' % '/'.join(by_step[step])
            print(line)

    print('\nTotals across %d cells:' % sum(counts.values()))
    for state in STATES + (None,):
        if counts.get(state):
            print('  %-10s %d' % (GLYPH[state], counts[state]))
    print('\nStep labels: ' + ', '.join(
        '%d=%s' % (s, STEP_LABELS.get(s, '?')) for s in steps))


if __name__ == '__main__':
    main()

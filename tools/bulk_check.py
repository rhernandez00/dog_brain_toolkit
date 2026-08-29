#!/usr/bin/env python
"""
bulk_check.py — fill the pipeline dashboard's cache for many RSA models at once.

What it is
----------
``pipeline_dashboard.py`` checks one step of one model per button press, which is
fine for a look at a single model and hopeless for a whole battery. This script
runs the *same* probes (``pipeline_console.PROBES``) and writes the results into
the *same* per-user cache file the Check button writes, so the dashboard shows
them exactly as if you had clicked every button yourself.

Switch the dashboard's **⟳ Live** toggle on and an open page fills in while this
runs — it re-reads the cache file every few seconds (see ``CACHE_POLL_MS`` in
``pipeline_dashboard.py``) and redraws when it changed, so you do not have to
reload it. With the toggle off (the default) the page updates only when you press
a button, change a parameter, or reload — switching it on afterwards picks up
everything written in the meantime.

Usage
-----
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\bulk_check.py ^
        --dataset EmoC --specie D --dis_method correlation --steps 5,7

    # every step of every mahalanobis model, both species
    ... --dis_method mahalanobis --steps 0-10 --specie D
    ... --dis_method mahalanobis --steps 0-10 --specie H

Models come from the central ``rsa_models/_models.csv`` manifest (via
``models_manifest.py``), filtered by ``--dis_method`` — and by ``--mah_fold`` too
if you pass it, which only narrows anything for mahalanobis — then expanded over
each row's ``groupings_possible``. That is the same selection the dashboard's
rsa_model dropdown makes, so the two always agree on what exists.
``--rsa_model`` restricts the run to one (repeatable, or comma-separated).

Matching the dashboard's parameter panel
----------------------------------------
A cached result is keyed by the full parameter signature, so **every parameter
here must match what the panel shows** or the page will read a different entry
and still say NOT CHECKED. The defaults are the panel's defaults
(``pipeline_dashboard.DEFAULTS``); change one here only if you changed it there.

Two wrinkles this handles for you:

* **mah_fold when the distance method is not mahalanobis.** The fold is greyed
  out in the panel but still part of the signature, and the probe paths ignore
  it. One probe result is therefore written under *every* fold value, so
  whichever one the disabled dropdown happens to be showing finds it.
* **Steps 0 and 1 are model-independent** (beta maps and pairwise maps do not
  live in the rsa_model's folder). ``pipeline_dashboard.store_step`` files those
  under a reduced, shared key, so they are probed **once** per distinct shared
  signature instead of once per model — for a 50-model battery that is one
  step-1 scan instead of fifty — and every matching model shows the result.
"""

import argparse
import os
import sys
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)  # tools/ lives one level below the repo root
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline_console as pc          # noqa: E402  the probe logic
import pipeline_dashboard as dash      # noqa: E402  importing does NOT start a server
import models_manifest as mm           # noqa: E402  central _models.csv reader


def parse_steps(spec):
    """'3,5-10' -> [3, 5, 6, 7, 8, 9, 10] (same syntax as schedule_steps.py)."""
    steps = []
    for part in str(spec).split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            steps.extend(range(int(a), int(b) + 1))
        elif part:
            steps.append(int(part))
    return [s for s in sorted(set(steps)) if s in pc.STEPS]


def parse_models(values):
    """Flatten repeated / comma-separated --rsa_model values."""
    out = []
    for v in values or []:
        out.extend(m.strip() for m in v.split(',') if m.strip())
    return out


def registry_models(datafolder, dataset, dis_method, mah_fold=None):
    """``[(rsa_model, mah_fold)]`` for one distance method, in manifest order.

    Same selection the dashboard's rsa_model menu makes (see
    ``models_manifest.concrete_models_for``): filtered by ``dis_method``, and by
    ``mah_fold`` too where the fold is a real choice — i.e. mahalanobis. With no
    ``mah_fold`` given, every fold of the method is included, each model carrying
    its own. Going through ``models_manifest`` rather than reading _models.csv by
    hand also makes suffix-less models (a family whose only file has no
    ``__grouping``) resolve the way the dashboard resolves them."""
    dirs = mm.rsa_models_dirs(datafolder, dataset)
    want = mm.normalize_dis_method(dis_method)
    by_fold = mm.uses_fold(want) and mah_fold
    out, seen = [], set()
    for row in mm.load_rows(dirs):
        if row['dis_method'] != want:
            continue
        if by_fold and row['mah_fold'] != mah_fold:
            continue
        for grouping in row['groupings']:
            name = mm.concrete_model_name(dirs, row['model'], grouping)
            if name in seen:
                continue
            seen.add(name)
            out.append((name, row['mah_fold'] or 'stim-wise'))
    return out


_MARK = {
    pc.DONE: ('OK', pc.green),
    pc.PARTIAL: ('~', pc.yellow),
    pc.MISSING: ('X', pc.red),
    pc.NA: ('-', pc.grey),
    pc.UNKNOWN: ('?', pc.grey),
}


def print_summary_table(models, steps, verdicts):
    """Model x step matrix of verdicts, printed at the end of a run.

    ``verdicts`` is ``{rsa_model: {step: verdict}}``, built up alongside the
    per-step prints above so this costs no extra probing — it just lays out
    results already computed."""
    name_w = min(max((len(m) for m, _ in models), default=5), 42)
    col_w = 4
    print(pc.bold("Summary"))
    header = f"{'model':<{name_w}}" + ''.join(f"{s:>{col_w}}" for s in steps)
    print(header)
    print('-' * len(header))
    counts = {v: 0 for v in _MARK}
    for rsa_model, _ in models:
        row = verdicts.get(rsa_model, {})
        name = rsa_model if len(rsa_model) <= name_w else rsa_model[:name_w - 1] + '…'
        cells = []
        for step in steps:
            v = row.get(step)
            mark, style = _MARK.get(v, ('.', pc.grey))
            if v in counts:
                counts[v] += 1
            cells.append(style(f"{mark:>{col_w}}"))
        print(f"{name:<{name_w}}" + ''.join(cells))
    print('-' * len(header))
    legend = '  '.join(style(f"{mark}={verdict}") for verdict, (mark, style) in _MARK.items())
    print(legend)
    totals = '  '.join(style(f"{verdict}:{counts[verdict]}")
                        for verdict, (mark, style) in _MARK.items() if counts[verdict])
    if totals:
        print(totals)
    print()


def folds_to_write(dis_method, csv_fold):
    """Which mah_fold signatures a result should be written under.

    For mahalanobis the fold decides where the files live, so only its own. For
    any other distance method the probe ignores it, but it is still in the cache
    signature — so write the same result under every fold the greyed-out dropdown
    could be showing."""
    if (dis_method or '').strip().lower() == 'mahalanobis':
        return [csv_fold]
    return sorted({csv_fold, *dash.MAH_FOLD_OPTIONS})


def main():
    D = dash.DEFAULTS
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default=D['dataset'])
    p.add_argument('--model', default=D['model'], help='GLM model (default: %(default)s)')
    p.add_argument('--specie', default=D['specie'], choices=['D', 'H'])
    p.add_argument('--dis_method', default=D['dis_method'])
    p.add_argument('--mah_fold',
                   help='Restrict to one Mahalanobis fold (mahalanobis only; '
                        'default: every fold of the method). Also the fold written '
                        'into the cache signature for the models it selects.')
    p.add_argument('--rsa_method', default=D['rsa_method'])
    p.add_argument('--steps', default='0-10',
                   help="Steps to probe, e.g. '5,7' or '3,5-10' (default: %(default)s)")
    p.add_argument('--rsa_model', action='append',
                   help='Probe only this model (repeatable, or comma-separated)')
    p.add_argument('--radius', type=int, default=D['radius'],
                   help='Searchlight radius (default: auto — 3 dog / 4 human)')
    p.add_argument('--z_threshold', type=float, default=D['z_threshold'])
    p.add_argument('--mask_type', default=D['mask_type'])
    p.add_argument('--reps', type=int, default=D['reps'])
    p.add_argument('--reps_group', type=int, default=D['reps_group'])
    p.add_argument('--skip_checked', action='store_true',
                   help='Leave steps that already have a cached result alone')
    p.add_argument('--verbose', action='store_true',
                   help='Print every filename probed (slow and chatty)')
    p.add_argument('--dry_run', action='store_true',
                   help='List what would be probed, touch nothing')
    args = p.parse_args()

    steps = parse_steps(args.steps)
    if not steps:
        p.error(f'--steps {args.steps!r} selected no step of {sorted(pc.STEPS)}')

    def make_params(rsa_model, mah_fold):
        return dash.params_from_inputs(
            dataset=args.dataset, model=args.model, rsa_model=rsa_model,
            specie=args.specie, dis_method=args.dis_method, mah_fold=mah_fold,
            rsa_method=args.rsa_method, radius=args.radius,
            z_threshold=args.z_threshold, mask_type=args.mask_type,
            reps=args.reps, reps_group=args.reps_group,
        )

    only = parse_models(args.rsa_model)
    registry = registry_models(dash.DATAFOLDER, args.dataset, args.dis_method,
                               args.mah_fold)
    if only:
        # keep the manifest's fold for a named model when it knows it
        folds = dict(registry)
        models = [(m, args.mah_fold or folds.get(m, D['mah_fold'])) for m in only]
    else:
        models = registry
    if not models:
        p.error(f'no model with dis_method={args.dis_method!r}'
                + (f' / mah_fold={args.mah_fold!r}' if args.mah_fold else '')
                + f' in the _models.csv of {args.dataset} — nothing to check')

    scope = args.dis_method + (f' / {args.mah_fold}' if args.mah_fold else '')
    print(f"{len(models)} model(s) x {len(steps)} step(s) — {args.dataset} / "
          f"{args.specie} / {scope}")
    print(f"cache: {dash.CACHE_PATH}")
    shared = [s for s in steps if dash.SHARED_STEPS and s in dash.SHARED_STEPS]
    if shared:
        print(f"steps {shared} are model-independent — probed once per distinct "
              f"shared signature, then reused for every matching model")
    if args.dry_run:
        for i, (m, fold) in enumerate(models, 1):
            print(f"  [{i}/{len(models)}] {m}  (mah_fold={fold} -> "
                  f"{', '.join(folds_to_write(args.dis_method, fold))})")
        return
    print()

    t0 = datetime.now()
    n_probes = n_reused = n_skipped = 0
    seen_shared = {}   # shared signature -> result already probed in this run
    verdicts = {}      # rsa_model -> {step: verdict}, for the summary table

    for i, (rsa_model, csv_fold) in enumerate(models, 1):
        probe_params = make_params(rsa_model, csv_fold)
        cache = dash.load_cache()
        print(f"[{i}/{len(models)}] {rsa_model}")
        row = verdicts.setdefault(rsa_model, {})

        for step in steps:
            if args.skip_checked:
                hit = dash.cached_step(cache, probe_params, step)
                if hit:
                    n_skipped += 1
                    print(f"  step {step}: already checked ({hit['verdict']}) — skipped")
                    row[step] = hit['verdict']
                    continue

            ssig = dash.shared_signature(probe_params, step)
            if ssig and ssig in seen_shared:
                # another model in this run already probed exactly these files
                n_reused += 1
                r = seen_shared[ssig]
                print(f"  step {step}: {r['verdict']} — {r['summary']}  [shared, reused]")
                row[step] = r['verdict']
                continue

            r = dash.run_probe(probe_params, step, verbose=args.verbose)
            n_probes += 1
            print(f"  step {step}: {r['verdict']} — {r['summary']}")
            row[step] = r['verdict']
            if ssig:
                # one write, under the reduced key — every model matching it now
                # shows this result, so no need to repeat it per fold below
                seen_shared[ssig] = dash.store_step(cache, probe_params, step, r)
                continue
            for fold in folds_to_write(args.dis_method, csv_fold):
                dash.store_step(cache, make_params(rsa_model, fold), step, r)

        # save per model, so an interrupt keeps everything done so far
        dash.save_cache(cache)

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\ndone in {elapsed:.1f}s — {len(models)} model(s), steps {steps}: "
          f"{n_probes} probe(s) run, {n_reused} reused from a shared step, "
          f"{n_skipped} already cached")
    print()
    print_summary_table(models, steps, verdicts)


if __name__ == '__main__':
    main()

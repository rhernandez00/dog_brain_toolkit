#!/usr/bin/env python
"""
pipeline_dashboard.py — Dash web dashboard for RSA pipeline progress.

A browser dashboard that shows how far each RSA model has progressed through the
pipeline (steps 0-10) by checking the actual output files on disk. It reuses the
file-probing logic in ``pipeline_console.py``.

Design goals (per request)
---------------------------
* **Editable parameters in the page** — dataset, model, rsa_model, specie,
  method, rsa_method, radius, z_threshold, mask_type, reps, reps_group. Change
  any of them and the view reflects that parameter set.
* **Check only on demand** — scanning the disk is slow, so nothing is checked
  automatically. You press "Check" on a step (or "Check all") to run the probe.
* **Memory** — once a step is checked, its result is remembered in a cache file
  and shown on the next launch, without re-scanning. Editing parameters shows
  the remembered results for *that* parameter set (or "not checked" if new).
* **Clear memory** — every step has a "Clear" button; there is also a
  "Clear all (this parameter set)" button, for when you redo a step and want to
  re-verify it.

Running it
----------
    python tools/pipeline_dashboard.py                # http://127.0.0.1:8060
    python tools/pipeline_dashboard.py --port 8062    # pick another port
    RSA_DASHBOARD_PORT=8062 python tools/pipeline_dashboard.py

On Windows use the full interpreter path, e.g.:
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\pipeline_dashboard.py

**It will not interfere with your other consoles.** It runs as its own process
on its own port (default 8060 — distinct from the 8050 unified dashboard and the
8051 model builder), binds to 127.0.0.1 only, and runs single-process with the
reloader disabled. Pressing Ctrl+C in *this* terminal stops *only* this app; your
other Dash apps keep running in their own terminals. If port 8060 is already
taken, pass a different ``--port``.

The memory cache is stored per-user at:
    ~/.rsa_pipeline_dashboard_cache.json
(kept off the shared data disk so two machines don't clobber each other's view).
"""

import argparse
import json
import os
import sys
import types
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)  # tools/ lives one level below the repo root
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pipeline_console as pc  # noqa: E402  (reuse the probe logic; sibling in tools/)
from scheduler.paths import get_paths, get_queue_dir  # noqa: E402
from scheduler.dag import build_single_job, build_job_graph  # noqa: E402
from scheduler.jobs import create_job  # noqa: E402

# ---------------------------------------------------------------------------
# Version — bump VERSION and update LAST_CHANGE on every edit to this file.
# See the "Versioning pipeline_dashboard.py" rule in CLAUDE.md.
# ---------------------------------------------------------------------------
VERSION = "1.4.0"
LAST_CHANGE = ("Status checks now follow Mahalanobis fold-specific step-2 and "
               "step-4 paths: direct results for stim-wise and multiple-folds, "
               "and one result/permutation set per run for stim-wise-all-runs.")

# Final step of the pipeline; "Schedule from here" queues start_step .. FINAL_STEP.
FINAL_STEP = 10

from dash import Dash, dcc, html, Input, Output, State, ALL, callback_context, no_update  # noqa: E402

# Steps whose output is one map *per participant* (probes report a per_sub
# breakdown for these). "Schedule missing" queues one job per missing subject
# for these; every other step produces a single group map -> one job.
PER_PARTICIPANT_STEPS = {0, 1, 2, 4}

# ---------------------------------------------------------------------------
# Parameters that define a "run" (and therefore the cache signature)
# ---------------------------------------------------------------------------
PARAM_KEYS = [
    'dataset', 'model', 'rsa_model', 'specie', 'method', 'mah_fold', 'rsa_method',
    'radius', 'z_threshold', 'mask_type', 'reps', 'reps_group',
]

DEFAULTS = {
    'dataset': 'EmoC',
    'model': 'basic-block',
    'rsa_model': None,
    'specie': 'D',
    'method': 'mahalanobis',
    'mah_fold': 'stim-wise',
    'rsa_method': 'kendall',
    'radius': None,          # None -> auto (3 dog / 4 human)
    'z_threshold': 3.1,
    'mask_type': 'b_GreyMatter2mmB',
    'reps': 100,
    'reps_group': 1000,
}

# Mahalanobis folding strategies (searchlight.py --mah_fold). Only meaningful when
# method == 'mahalanobis'; it decides where/which pairwise maps land on disk, so it
# is part of the run signature — two folds of the same model are tracked apart.
MAH_FOLD_OPTIONS = [
    'stim-wise', 'stim-wise-multiple-folds', 'stim-wise-all-runs', 'run-wise',
]

VERDICT_COLOR = {
    pc.DONE: '#1a7f37',
    pc.PARTIAL: '#bf8700',
    pc.MISSING: '#cf222e',
    pc.NA: '#8a8a8a',
    pc.UNKNOWN: '#8a8a8a',
}
NOT_CHECKED = 'NOT CHECKED'
NOT_CHECKED_COLOR = '#9aa0a6'

DATAFOLDER, _, _ = get_paths()
CACHE_PATH = os.path.join(os.path.expanduser('~'), '.rsa_pipeline_dashboard_cache.json')


def _log(msg):
    """Print to the console the app was launched from, so button presses that
    trigger a (possibly slow) disk scan are visible even before the page updates."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Cache (persisted JSON, keyed by parameter signature)
# ---------------------------------------------------------------------------
def load_cache():
    try:
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache):
    try:
        tmp = CACHE_PATH + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp, CACHE_PATH)
    except Exception as e:
        print(f"[pipeline_dashboard] warning: could not save cache: {e}")


def signature(params):
    return " | ".join(f"{k}={params.get(k)}" for k in PARAM_KEYS)


def params_from_inputs(dataset, model, rsa_model, specie, method, mah_fold, rsa_method,
                       radius, z_threshold, mask_type, reps, reps_group):
    def _int_or_none(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    return {
        'dataset': (dataset or '').strip(),
        'model': (model or '').strip(),
        'rsa_model': rsa_model,
        'specie': specie or 'D',
        'method': (method or '').strip(),
        'mah_fold': (mah_fold or 'stim-wise').strip(),
        'rsa_method': (rsa_method or '').strip(),
        'radius': _int_or_none(radius),
        'z_threshold': float(z_threshold) if z_threshold not in (None, '') else 3.1,
        'mask_type': (mask_type or '').strip(),
        'reps': _int_or_none(reps) or 100,
        'reps_group': _int_or_none(reps_group) or 1000,
    }


def make_ctx(params):
    ns = types.SimpleNamespace(**{k: params.get(k) for k in PARAM_KEYS})
    return pc.build_ctx(ns, DATAFOLDER)


def _log_verbose_detail(step, label, r):
    """Print every filename the probe checked for this step, marked found/missing.

    Called only when the "Full verbose mode" toggle is on. ``detail`` is built
    by the probe functions in pipeline_console.py (populated only when they're
    called with verbose=True) as a list of
    ``{'sub': int_or_None, 'path': str, 'status': DONE|MISSING|PATTERN}``.
    PATTERN entries are glob patterns searched for steps whose exact filenames
    can't be predicted in advance (e.g. GLM pe*.nii.gz, permutation-indexed
    files) — the FOUND lines right after them are what that pattern matched.
    """
    detail = r.get('detail') or []
    if not detail:
        _log(f"  [verbose] step {step} ({label}): no per-file detail for this step "
             f"(N/A or inputs unavailable)")
        return
    _log(f"  [verbose] step {step} ({label}): {len(detail)} file(s)/pattern(s) checked")
    for d in detail:
        sub = f"sub-{d['sub']:02d}  " if d.get('sub') is not None else ""
        if d['status'] == pc.DONE:
            mark = 'FOUND  '
        elif d['status'] == pc.MISSING:
            mark = 'MISSING'
        else:
            mark = 'PATTERN'
        _log(f"    [{mark}] {sub}{d['path']}")
    n_found = sum(1 for d in detail if d['status'] == pc.DONE)
    n_missing = sum(1 for d in detail if d['status'] == pc.MISSING)
    _log(f"  [verbose] step {step} totals: {n_found} found, {n_missing} missing")


def run_probe(params, step, verbose=False):
    """Run one step's disk probe and return a JSON-serialisable result dict."""
    label = pc.STEP_LABELS.get(step, f'Step {step}')
    t0 = datetime.now()
    _log(f"  scanning step {step} ({label}) ...")
    ctx = make_ctx(params)
    r = pc.PROBES[step](ctx, verbose=verbose)
    failures = pc.find_failure_info(ctx, step)
    elapsed = (datetime.now() - t0).total_seconds()
    if verbose:
        _log_verbose_detail(step, label, r)
    _log(f"  step {step} ({label}) -> {r['verdict']} — {r['summary']}  [{elapsed:.1f}s]")
    return {
        'verdict': r['verdict'],
        'summary': r['summary'],
        'expected': r.get('expected') or [],
        'found': r.get('found') or [],
        'per_sub': [list(x) for x in (r.get('per_sub') or [])],
        'failures': [list(x) for x in failures],
        'checked_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'resolve_note': ctx._resolve_error or '',
    }


# ---------------------------------------------------------------------------
# Scheduling — create independent (no-dependency) jobs in the shared queue
# ---------------------------------------------------------------------------
def _schedule_jobs(params, step, participants, overwrite):
    """Create one independent pending job per entry in ``participants``.

    ``participants`` is a list of subject ints, or ``[None]`` for the single
    group-map jobs. Returns the list of scheduled entries. ``create_job``
    always succeeds — if a job with the same id already exists anywhere in
    the queue (still in flight, or a previous completed/failed run), it's
    scheduled again under a disambiguated filename with
    ``shuffle_participants`` set, rather than being skipped.
    """
    queue_dir = get_queue_dir(DATAFOLDER)
    # The searchlight overwrite flag differs by step family: rnd steps (4/5)
    # write into RSA_rnd and honour --replace_rnd_files; the rest use --replace_file.
    replace_rnd = bool(overwrite and step in (4, 5))
    created = []
    for sub in participants:
        job = build_single_job(
            dataset=params['dataset'], model=params['model'],
            rsa_model=params['rsa_model'], specie=params['specie'], step=step,
            z_threshold=params['z_threshold'], reps=params['reps'],
            reps_group=params['reps_group'],
            rsa_method=params['rsa_method'], dis_method=params['method'],
            mah_fold=params['mah_fold'],
            participant=sub, radius=params['radius'], mask_type=params['mask_type'],
            replace_file=bool(overwrite), replace_rnd_files=replace_rnd,
        )
        create_job(queue_dir, job)
        created.append(sub)
    return created


def _fmt_maps(maps):
    names = ['group' if s is None else f'sub-{int(s):02d}' for s in maps]
    return ', '.join(names) if names else '—'


def _msg_span(text, ok=True):
    color = '#1a7f37' if ok else '#bf8700'
    return html.Span(text, style={'color': color, 'fontWeight': '600'})


def _schedule_result_msg(params, step, created, overwrite):
    label = pc.STEP_LABELS.get(step, f'Step {step}')
    tag = ' [overwrite]' if overwrite else ''
    txt = (f"Step {step} ({label}), specie {params['specie']}{tag}: "
           f"scheduled {len(created)} job(s)")
    if created:
        txt += f" → {_fmt_maps(created)}"
    return _msg_span(txt, ok=bool(created))


def _schedule_missing(params, step, probe_result, overwrite):
    """Schedule jobs for the not-DONE maps of ``step`` from a fresh probe."""
    if step in PER_PARTICIPANT_STEPS:
        per_sub = probe_result.get('per_sub') or []
        subs = [int(s) for s, v, _ in per_sub if v in (pc.MISSING, pc.PARTIAL)]
        if not subs:
            return _msg_span(
                f"Step {step}: no missing/partial participants to schedule "
                f"(verdict {probe_result.get('verdict')}).", ok=True)
        created = _schedule_jobs(params, step, subs, overwrite)
        return _schedule_result_msg(params, step, created, overwrite)
    # group step -> a single map
    if probe_result.get('verdict') == pc.DONE:
        return _msg_span(
            f"Step {step} already DONE — nothing to schedule "
            f"(use Details → Schedule to force a re-run).", ok=True)
    created = _schedule_jobs(params, step, [None], overwrite)
    return _schedule_result_msg(params, step, created, overwrite)


def _schedule_from_here(params, start_step, overwrite):
    """Schedule the full *dependent* DAG from ``start_step`` through FINAL_STEP.

    Unlike ``_schedule_jobs`` (which queues independent, ready-to-run jobs for a
    single step), this uses ``build_job_graph`` — the scheduler's nested-job
    builder. It queues one whole-step job per needed step; each job ``waits`` on
    its in-graph dependencies and is promoted to ``pending`` automatically by
    ``run_jobs.py`` as those deps complete, so the steps run in order without
    any manual sequencing. Steps below ``start_step`` are assumed to exist on
    disk already.

    ``radius`` / ``mask_type`` and the overwrite flags are not threaded through
    ``build_job_graph``, so they are injected per job here to match the rest of
    the dashboard's scheduled jobs. Returns the list of scheduled step numbers.
    ``create_job`` always succeeds — a job id that already exists elsewhere in
    the queue is scheduled again under a disambiguated filename instead of
    being skipped (see ``create_job`` in ``scheduler/jobs.py``).
    """
    queue_dir = get_queue_dir(DATAFOLDER)
    jobs = build_job_graph(
        dataset=params['dataset'], model=params['model'],
        rsa_model=params['rsa_model'], specie=params['specie'],
        target_step=FINAL_STEP, start_step=start_step,
        z_threshold=params['z_threshold'], reps=params['reps'],
        reps_group=params['reps_group'],
        rsa_method=params['rsa_method'], dis_method=params['method'],
        mah_fold=params['mah_fold'],
    )
    created = []
    for job in jobs:
        step = job['step']
        # Inject the dashboard-specific fields build_job_graph does not set, so
        # these jobs run with the same radius / mask / overwrite as every other
        # map scheduled from this UI. rnd steps (4/5) honour --replace_rnd_files.
        job['radius'] = params['radius']
        job['mask_type'] = params['mask_type']
        job['replace_file'] = bool(overwrite)
        job['replace_rnd_files'] = bool(overwrite and step in (4, 5))
        create_job(queue_dir, job)
        created.append(step)
    return created


def _steps_list(steps):
    return ', '.join(str(s) for s in sorted(steps)) if steps else '—'


def _schedule_from_here_msg(params, start_step, created, overwrite):
    tag = ' [overwrite]' if overwrite else ''
    txt = (f"Schedule steps {start_step}→{FINAL_STEP} (dependent DAG), "
           f"specie {params['specie']}{tag}: created {len(created)} job(s)")
    if created:
        txt += f" → steps {_steps_list(created)}"
    return _msg_span(txt, ok=bool(created))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
URL_BASE = os.environ.get('PIPELINE_DASHBOARD_URL_BASE', '/')
app = Dash(__name__, url_base_pathname=URL_BASE, title='RSA Pipeline Status')
server = app.server


def _input(id_, value, width='150px', type_='text'):
    return dcc.Input(id=id_, value=value, type=type_, debounce=True,
                     style={'width': width, 'marginRight': '10px'})


def param_panel():
    return html.Div([
        html.Div([
            html.Label('dataset'), _input('p-dataset', DEFAULTS['dataset'], '120px'),
            html.Label('model'), _input('p-model', DEFAULTS['model'], '140px'),
            html.Label('rsa_model'),
            dcc.Dropdown(id='p-rsa_model', options=[], value=None,
                         style={'width': '220px', 'display': 'inline-block',
                                'verticalAlign': 'middle', 'marginRight': '10px'}),
            html.Button('⟳ reload models', id='reload-models', n_clicks=0),
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Label('specie'),
            dcc.Dropdown(id='p-specie', options=[{'label': 'Dog (D)', 'value': 'D'},
                                                 {'label': 'Human (H)', 'value': 'H'}],
                         value=DEFAULTS['specie'],
                         style={'width': '120px', 'display': 'inline-block',
                                'verticalAlign': 'middle', 'marginRight': '10px'}),
            html.Label('method'), _input('p-method', DEFAULTS['method'], '120px'),
            html.Label('mah_fold'),
            dcc.Dropdown(id='p-mah_fold',
                         options=[{'label': m, 'value': m} for m in MAH_FOLD_OPTIONS],
                         value=DEFAULTS['mah_fold'], clearable=False,
                         style={'width': '210px', 'display': 'inline-block',
                                'verticalAlign': 'middle', 'marginRight': '10px'}),
            html.Label('rsa_method'), _input('p-rsa_method', DEFAULTS['rsa_method'], '100px'),
            html.Label('radius'), _input('p-radius', '', '60px', 'number'),
            html.Label('z_threshold'), _input('p-z_threshold', DEFAULTS['z_threshold'], '70px', 'number'),
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Label('mask_type'), _input('p-mask_type', DEFAULTS['mask_type'], '160px'),
            html.Label('reps'), _input('p-reps', DEFAULTS['reps'], '70px', 'number'),
            html.Label('reps_group'), _input('p-reps_group', DEFAULTS['reps_group'], '80px', 'number'),
        ]),
    ], style={'padding': '12px', 'background': '#f6f8fa', 'borderRadius': '8px',
              'border': '1px solid #d0d7de'})


app.layout = html.Div([
    html.H2('RSA Pipeline Status'),
    html.Div([
        html.Span(f'v{VERSION}', style={'fontWeight': '600'}),
        html.Span(f'  —  last change: {LAST_CHANGE}', style={'color': '#57606a'}),
    ], style={'fontSize': '12px', 'margin': '-6px 0 10px'}),
    html.P('Checks the actual output files on disk. Nothing is scanned until you '
           'press a "Check" button. Results are remembered per parameter set.',
           style={'color': '#57606a'}),
    param_panel(),
    html.Div([
        html.Button('▶ Check all steps', id='check-all', n_clicks=0,
                    style={'marginRight': '10px', 'fontWeight': 'bold'}),
        html.Button('🗑 Clear all (this parameter set)', id='clear-all', n_clicks=0,
                    style={'marginRight': '10px'}),
        dcc.Checklist(
            id='verbose-mode',
            options=[{'label': ' 🔍 Full verbose mode (print every filename checked, '
                               'found or missing, to the console)',
                      'value': 'v'}],
            value=[],
            style={'display': 'inline-block', 'color': '#57606a', 'marginRight': '10px'},
        ),
        html.Span(id='sig-label', style={'color': '#57606a', 'marginLeft': '10px'}),
    ], style={'margin': '14px 0'}),
    html.Div(id='schedule-msg', style={'margin': '8px 0', 'minHeight': '20px'}),
    html.Div(id='status-table'),
    html.Hr(),
    html.H4('Step detail'),
    html.Div([
        dcc.Checklist(
            id='detail-overwrite',
            options=[{'label': ' overwrite existing files when scheduling '
                               '(adds --replace_file / --replace_rnd_files)',
                      'value': 'ow'}],
            value=[],
            style={'display': 'inline-block', 'color': '#57606a'},
        ),
    ], style={'margin': '4px 0'}),
    html.Div(id='detail-panel', style={'minHeight': '80px'}),
    html.Hr(),
    html.Details([
        html.Summary('How to run / help'),
        dcc.Markdown(
            "* Change any parameter above, then press **Check all steps** or a "
            "per-step **Check**.\n"
            "* Green = all files present, orange = partial (e.g. some participants "
            "or permutations missing), red = missing, grey = N/A or unknown.\n"
            "* **Details** shows the per-participant breakdown and, if a scheduler "
            "job failed, the recorded error (the 'why').\n"
            "* **Sched missing** (blue, per step) probes that step now and queues one "
            "*independent* job per missing map — one job per missing participant for "
            "steps 0/1/2/4, or one job for the single group map otherwise. Jobs land "
            "in the shared `job_queue/pending/` and are picked up by `run_jobs.py`.\n"
            "* **Sched from here** (purple, per step) queues every step from that row "
            "through step 10 as a *dependent DAG*: each later step `waits` on the "
            "steps it needs and is promoted to `pending` automatically as they "
            "finish, so the whole chain runs in order from one click. Steps below the "
            "chosen one are assumed to already exist on disk; steps that are already "
            "done are skipped by `searchlight.py` unless you tick **overwrite**. This "
            "is the same nested-job graph that `schedule_rsa.py` builds.\n"
            "* Inside **Details**, each participant has its own **Schedule** button "
            "(and group steps get a **Schedule this step** button) so you can queue "
            "any single map — including one that is already DONE, to force a re-run.\n"
            "* Scheduling a job whose id already exists in the queue (pending, "
            "waiting, running, completed, or failed) never gets skipped — it's queued "
            "again under a `__dup{N}` filename with `--shuffle_participants` set, so a "
            "duplicate run walks participants/permutations in a different order than "
            "the other instance instead of racing it file-by-file.\n"
            "* Tick **overwrite** (above the detail panel) before scheduling to add "
            "`--replace_file` (and `--replace_rnd_files` for steps 4/5) so existing "
            "files are recomputed instead of skipped.\n"
            "* Scheduled jobs are created with **no dependencies** — they run as soon "
            "as a worker is free, so make sure a map's inputs (earlier steps) already "
            "exist before scheduling it.\n"
            "* Results are remembered in `~/.rsa_pipeline_dashboard_cache.json`. "
            "Use a step's **Clear** or **Clear all** after you redo a step so it "
            "gets re-checked.\n"
            "* Parameters must match how the model was run — `method`, `rsa_method`, "
            "`radius`, `mask_type`, `z_threshold` are encoded in the filenames.\n"
            "* This app runs on its own port; closing it does not affect your other "
            "consoles."
        ),
    ]),
    dcc.Store(id='cache-version', data=0),
    dcc.Store(id='selected-step', data=None),
], style={'maxWidth': '1100px', 'margin': '0 auto', 'fontFamily': 'system-ui, sans-serif'})


# ---------------------------------------------------------------------------
# Callback: populate rsa_model dropdown from the dataset folder
# ---------------------------------------------------------------------------
@app.callback(
    Output('p-rsa_model', 'options'),
    Output('p-rsa_model', 'value'),
    Input('p-dataset', 'value'),
    Input('p-model', 'value'),
    Input('reload-models', 'n_clicks'),
    State('p-rsa_model', 'value'),
)
def populate_models(dataset, model, _n, current):
    _log(f"reloading rsa_model list for dataset={dataset!r} model={model!r} ...")
    models, _folder = pc.list_rsa_models(DATAFOLDER, (dataset or '').strip())
    _log(f"  found {len(models)} rsa_model(s)")
    options = [{'label': m, 'value': m} for m in models]
    value = current if current in models else (models[0] if models else None)
    return options, value


# ---------------------------------------------------------------------------
# Callback: mutating actions (check all / clear all / per-step check/clear/details)
# ---------------------------------------------------------------------------
@app.callback(
    Output('cache-version', 'data'),
    Output('selected-step', 'data'),
    Output('schedule-msg', 'children'),
    Input('check-all', 'n_clicks'),
    Input('clear-all', 'n_clicks'),
    Input({'type': 'step-check', 'index': ALL}, 'n_clicks'),
    Input({'type': 'step-clear', 'index': ALL}, 'n_clicks'),
    Input({'type': 'step-details', 'index': ALL}, 'n_clicks'),
    Input({'type': 'step-schedule-missing', 'index': ALL}, 'n_clicks'),
    Input({'type': 'step-schedule-from-here', 'index': ALL}, 'n_clicks'),
    Input({'type': 'detail-schedule-sub', 'step': ALL, 'sub': ALL}, 'n_clicks'),
    Input({'type': 'detail-schedule-group', 'index': ALL}, 'n_clicks'),
    State('cache-version', 'data'),
    State('detail-overwrite', 'value'),
    State('verbose-mode', 'value'),
    State('p-dataset', 'value'), State('p-model', 'value'), State('p-rsa_model', 'value'),
    State('p-specie', 'value'), State('p-method', 'value'), State('p-mah_fold', 'value'),
    State('p-rsa_method', 'value'),
    State('p-radius', 'value'), State('p-z_threshold', 'value'), State('p-mask_type', 'value'),
    State('p-reps', 'value'), State('p-reps_group', 'value'),
    prevent_initial_call=True,
)
def do_action(_ca, _cl, _sc, _sx, _sd, _sm, _sfh, _dss, _dsg, version, overwrite_val,
              verbose_val,
              dataset, model, rsa_model, specie, method, mah_fold, rsa_method,
              radius, z_threshold, mask_type, reps, reps_group):
    trig = callback_context.triggered
    if not trig or trig[0]['value'] in (None, 0):
        return no_update, no_update, no_update
    prop = trig[0]['prop_id']

    def _id_from_prop(p):
        # p looks like '{"index":7,"type":"step-check"}.n_clicks'
        try:
            return json.loads(p.split('.n_clicks')[0])
        except Exception:
            return None

    idd = _id_from_prop(prop) or {}
    ttype = idd.get('type')
    overwrite = 'ow' in (overwrite_val or [])
    verbose = 'v' in (verbose_val or [])

    params = params_from_inputs(dataset, model, rsa_model, specie, method, mah_fold,
                                rsa_method, radius, z_threshold, mask_type, reps, reps_group)
    if not params['rsa_model']:
        _log(f"button pressed ({prop}) but no rsa_model selected — ignoring")
        return no_update, no_update, _msg_span('⚠ pick an rsa_model first', ok=False)
    sig = signature(params)
    cache = load_cache()
    entry = cache.setdefault(sig, {'params': params, 'steps': {}})
    entry['params'] = params
    selected = no_update
    msg = no_update

    if prop.startswith('check-all'):
        _log(f"'Check all steps' pressed (verbose={verbose}) — {sig}")
        t0 = datetime.now()
        for step in pc.STEPS:
            entry['steps'][str(step)] = run_probe(params, step, verbose=verbose)
        elapsed = (datetime.now() - t0).total_seconds()
        _log(f"'Check all steps' done in {elapsed:.1f}s")
        # focus the first incomplete step
        selected = None
        for step in pc.STEPS:
            v = entry['steps'][str(step)]['verdict']
            if v not in (pc.DONE, pc.NA):
                selected = step
                break
    elif prop.startswith('clear-all'):
        _log(f"'Clear all' pressed — {sig}")
        cache.pop(sig, None)
        save_cache(cache)
        return (version or 0) + 1, None, no_update
    elif ttype == 'step-check':
        step = idd.get('index')
        if step is not None:
            _log(f"'Check' pressed for step {step} (verbose={verbose}) — {sig}")
            entry['steps'][str(step)] = run_probe(params, step, verbose=verbose)
            selected = step
    elif ttype == 'step-clear':
        step = idd.get('index')
        if step is not None:
            _log(f"'Clear' pressed for step {step} — {sig}")
            entry['steps'].pop(str(step), None)
            selected = no_update
    elif ttype == 'step-details':
        step = idd.get('index')
        _log(f"'Details' pressed for step {step}")
        selected = step if step is not None else no_update
    elif ttype == 'step-schedule-missing':
        step = idd.get('index')
        if step is not None:
            _log(f"'Sched missing' pressed for step {step} "
                 f"(overwrite={overwrite}, verbose={verbose}) — {sig}")
            # probe fresh so we schedule exactly what is missing right now, and
            # remember the result so the table/detail reflect it.
            result = run_probe(params, step, verbose=verbose)
            entry['steps'][str(step)] = result
            selected = step
            msg = _schedule_missing(params, step, result, overwrite)
    elif ttype == 'step-schedule-from-here':
        step = idd.get('index')
        if step is not None:
            _log(f"'Sched from here' pressed at step {step} (overwrite={overwrite}) — {sig}")
            created = _schedule_from_here(params, step, overwrite)
            msg = _schedule_from_here_msg(params, step, created, overwrite)
            selected = step
    elif ttype == 'detail-schedule-sub':
        step, sub = idd.get('step'), idd.get('sub')
        if step is not None and sub is not None:
            _log(f"'Schedule' pressed for step {step} sub-{sub} (overwrite={overwrite}) — {sig}")
            created = _schedule_jobs(params, step, [int(sub)], overwrite)
            msg = _schedule_result_msg(params, step, created, overwrite)
            selected = step
    elif ttype == 'detail-schedule-group':
        step = idd.get('index')
        if step is not None:
            _log(f"'Schedule step' (group) pressed for step {step} (overwrite={overwrite}) — {sig}")
            created = _schedule_jobs(params, step, [None], overwrite)
            msg = _schedule_result_msg(params, step, created, overwrite)
            selected = step

    save_cache(cache)
    return (version or 0) + 1, selected, msg


# ---------------------------------------------------------------------------
# Callback: render the status table (cheap — reads cache only, never scans)
# ---------------------------------------------------------------------------
@app.callback(
    Output('status-table', 'children'),
    Output('sig-label', 'children'),
    Input('cache-version', 'data'),
    Input('p-dataset', 'value'), Input('p-model', 'value'), Input('p-rsa_model', 'value'),
    Input('p-specie', 'value'), Input('p-method', 'value'), Input('p-mah_fold', 'value'),
    Input('p-rsa_method', 'value'),
    Input('p-radius', 'value'), Input('p-z_threshold', 'value'), Input('p-mask_type', 'value'),
    Input('p-reps', 'value'), Input('p-reps_group', 'value'),
)
def render_table(_v, dataset, model, rsa_model, specie, method, mah_fold, rsa_method,
                 radius, z_threshold, mask_type, reps, reps_group):
    params = params_from_inputs(dataset, model, rsa_model, specie, method, mah_fold,
                                rsa_method, radius, z_threshold, mask_type, reps, reps_group)
    sig = signature(params)
    cache = load_cache()
    steps_cache = cache.get(sig, {}).get('steps', {})

    header = html.Tr([html.Th(h, style={'textAlign': 'left', 'padding': '6px 10px'})
                      for h in ['#', 'Step', 'Status', 'Last checked', 'Actions']])
    rows = [header]
    for step in pc.STEPS:
        label = pc.STEP_LABELS.get(step, f'Step {step}')
        c = steps_cache.get(str(step))
        if c:
            verdict = c['verdict']
            color = VERDICT_COLOR.get(verdict, NOT_CHECKED_COLOR)
            badge = html.Span(f'{verdict} — {c["summary"]}',
                              style={'color': 'white', 'background': color,
                                     'padding': '2px 8px', 'borderRadius': '10px',
                                     'fontSize': '13px'})
            when = c.get('checked_at', '')
        else:
            badge = html.Span(NOT_CHECKED, style={'color': 'white',
                              'background': NOT_CHECKED_COLOR, 'padding': '2px 8px',
                              'borderRadius': '10px', 'fontSize': '13px'})
            when = '—'
        actions = html.Div([
            html.Button('Check', id={'type': 'step-check', 'index': step}, n_clicks=0,
                        style={'marginRight': '6px'}),
            html.Button('Clear', id={'type': 'step-clear', 'index': step}, n_clicks=0,
                        style={'marginRight': '6px'}),
            html.Button('Details', id={'type': 'step-details', 'index': step}, n_clicks=0,
                        style={'marginRight': '6px'}),
            html.Button('Sched missing', id={'type': 'step-schedule-missing', 'index': step},
                        n_clicks=0, title='Probe this step now, then queue a job for each '
                        'missing map',
                        style={'background': '#0969da', 'color': 'white',
                               'border': 'none', 'borderRadius': '4px',
                               'padding': '2px 8px', 'marginRight': '6px'}),
            html.Button('Sched from here',
                        id={'type': 'step-schedule-from-here', 'index': step}, n_clicks=0,
                        title=f'Queue every step from {step} through {FINAL_STEP} as a '
                              'dependent DAG (later steps wait for earlier ones)',
                        style={'background': '#8250df', 'color': 'white',
                               'border': 'none', 'borderRadius': '4px',
                               'padding': '2px 8px'}),
        ])
        rows.append(html.Tr([
            html.Td(step, style={'padding': '6px 10px'}),
            html.Td(label, style={'padding': '6px 10px'}),
            html.Td(badge, style={'padding': '6px 10px'}),
            html.Td(when, style={'padding': '6px 10px', 'color': '#57606a',
                                 'fontSize': '12px'}),
            html.Td(actions, style={'padding': '6px 10px'}),
        ], style={'borderTop': '1px solid #eaeef2'}))

    table = html.Table(rows, style={'borderCollapse': 'collapse', 'width': '100%'})
    label = f"parameter set: {sig}"
    if not params['rsa_model']:
        label = "⚠ pick an rsa_model first"
    return table, label


# ---------------------------------------------------------------------------
# Callback: render the detail panel for the selected step (reads cache only)
# ---------------------------------------------------------------------------
@app.callback(
    Output('detail-panel', 'children'),
    Input('selected-step', 'data'),
    Input('cache-version', 'data'),
    State('p-dataset', 'value'), State('p-model', 'value'), State('p-rsa_model', 'value'),
    State('p-specie', 'value'), State('p-method', 'value'), State('p-mah_fold', 'value'),
    State('p-rsa_method', 'value'),
    State('p-radius', 'value'), State('p-z_threshold', 'value'), State('p-mask_type', 'value'),
    State('p-reps', 'value'), State('p-reps_group', 'value'),
)
def render_detail(step, _v, dataset, model, rsa_model, specie, method, mah_fold, rsa_method,
                  radius, z_threshold, mask_type, reps, reps_group):
    if step is None:
        return html.Span('Select a step\'s "Details" (or "Check" a step) to see the '
                         'per-participant breakdown here.', style={'color': '#57606a'})
    params = params_from_inputs(dataset, model, rsa_model, specie, method, mah_fold,
                                rsa_method, radius, z_threshold, mask_type, reps, reps_group)
    sig = signature(params)
    c = load_cache().get(sig, {}).get('steps', {}).get(str(step))
    label = pc.STEP_LABELS.get(step, f'Step {step}')
    if not c:
        return html.Div([html.B(f'Step {step} — {label}'), html.Br(),
                         html.Span('Not checked yet for this parameter set. Press '
                                   '"Check" on that step to get per-map Schedule '
                                   'buttons, or "Sched missing" to probe and queue '
                                   'the missing maps in one go.',
                                   style={'color': '#57606a'})])

    color = VERDICT_COLOR.get(c['verdict'], NOT_CHECKED_COLOR)
    blocks = [
        html.Div([html.B(f'Step {step} — {label}  '),
                  html.Span(c['verdict'], style={'color': 'white', 'background': color,
                            'padding': '2px 8px', 'borderRadius': '10px'}),
                  html.Span(f'  {c["summary"]}')]),
        html.Div(f'checked at {c.get("checked_at", "")}',
                 style={'color': '#57606a', 'fontSize': '12px', 'margin': '4px 0'}),
    ]
    if c.get('resolve_note'):
        blocks.append(html.Div('note: ' + c['resolve_note'],
                               style={'color': '#bf8700', 'fontSize': '12px'}))

    per_sub = c.get('per_sub') or []
    if per_sub:
        items = []
        for sub, verdict, note in per_sub:
            col = VERDICT_COLOR.get(verdict, NOT_CHECKED_COLOR)
            mark = {'DONE': '✓', 'PARTIAL': '~', 'MISSING': '✗'}.get(verdict, '?')
            items.append(html.Li([
                html.Span(f'{mark} ', style={'color': col, 'fontWeight': 'bold'}),
                html.Span(f'sub-{int(sub):02d}: ', style={'fontFamily': 'monospace'}),
                html.Span(f'{verdict} — {note}', style={'color': col}),
                html.Button('Schedule',
                            id={'type': 'detail-schedule-sub', 'step': step, 'sub': int(sub)},
                            n_clicks=0, title='Queue a job for just this participant',
                            style={'marginLeft': '8px', 'fontSize': '11px',
                                   'padding': '0 6px', 'cursor': 'pointer'}),
            ], style={'breakInside': 'avoid'}))
        blocks.append(html.Details([
            html.Summary(f'per-participant ({len(per_sub)}) — Schedule queues that '
                         'one map (honours the overwrite checkbox above)'),
            html.Ul(items, style={'columns': '2', 'fontSize': '13px'}),
        ], open=True))

    for lbl, key in [('expected', 'expected'), ('found', 'found')]:
        vals = c.get(key) or []
        if vals:
            blocks.append(html.Details([
                html.Summary(f'{lbl} ({len(vals)})'),
                html.Ul([html.Li(html.Code(str(p))) for p in vals[:12]],
                        style={'fontSize': '12px'}),
            ]))

    failures = c.get('failures') or []
    if failures:
        fitems = []
        for job_id, err, log_path in failures:
            fitems.append(html.Li([
                html.Code(job_id), html.Br(),
                html.Span(f'error: {err}' if err else 'error: (none recorded)',
                          style={'color': '#cf222e'}), html.Br(),
                html.Span(f'log: {log_path}' if log_path else '',
                          style={'color': '#57606a', 'fontSize': '12px'}),
            ]))
        blocks.append(html.Div([
            html.B('recorded scheduler failures (why):', style={'color': '#cf222e'}),
            html.Ul(fitems),
        ]))

    # --- scheduling controls -------------------------------------------------
    if step in PER_PARTICIPANT_STEPS:
        blocks.append(html.Div(
            'Per-participant step: use the Schedule button next to a participant '
            'above to queue that one map, or "Sched missing" in the table to queue '
            'every missing/partial participant at once.',
            style={'color': '#57606a', 'fontSize': '12px', 'margin': '8px 0'}))
    else:
        blocks.append(html.Div([
            html.Button('▶ Schedule this step',
                        id={'type': 'detail-schedule-group', 'index': step}, n_clicks=0,
                        style={'background': '#0969da', 'color': 'white', 'border': 'none',
                               'borderRadius': '4px', 'padding': '4px 10px',
                               'cursor': 'pointer', 'marginRight': '8px'}),
            html.Span('single group map — queues one job '
                      '(honours the overwrite checkbox above).',
                      style={'color': '#57606a', 'fontSize': '12px'}),
        ], style={'margin': '8px 0'}))
    return html.Div(blocks)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='RSA pipeline status dashboard (Dash)')
    parser.add_argument('--port', type=int,
                        default=int(os.environ.get('RSA_DASHBOARD_PORT', '8060')),
                        help='Port to serve on (default 8060; distinct from 8050/8051)')
    parser.add_argument('--host', default='127.0.0.1',
                        help='Host to bind (default 127.0.0.1 — local only)')
    args = parser.parse_args()
    print(f"[pipeline_dashboard] version    : v{VERSION} - {LAST_CHANGE}")
    print(f"[pipeline_dashboard] datafolder : {DATAFOLDER}")
    print(f"[pipeline_dashboard] cache file : {CACHE_PATH}")
    print(f"[pipeline_dashboard] open       : http://{args.host}:{args.port}")
    print("[pipeline_dashboard] Ctrl+C stops ONLY this app.")
    # debug=False + use_reloader=False => single process, no child procs, and
    # Ctrl+C here never touches your other consoles.
    app.run(debug=False, use_reloader=False, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

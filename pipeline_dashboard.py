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
    python pipeline_dashboard.py                # http://127.0.0.1:8060
    python pipeline_dashboard.py --port 8062    # pick another port
    RSA_DASHBOARD_PORT=8062 python pipeline_dashboard.py

On Windows use the full interpreter path, e.g.:
    & "C:\\ProgramData\\anaconda3\\python.exe" pipeline_dashboard.py

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pipeline_console as pc  # noqa: E402  (reuse the probe logic)
from scheduler.paths import get_paths  # noqa: E402

from dash import Dash, dcc, html, Input, Output, State, ALL, callback_context, no_update  # noqa: E402

# ---------------------------------------------------------------------------
# Parameters that define a "run" (and therefore the cache signature)
# ---------------------------------------------------------------------------
PARAM_KEYS = [
    'dataset', 'model', 'rsa_model', 'specie', 'method', 'rsa_method',
    'radius', 'z_threshold', 'mask_type', 'reps', 'reps_group',
]

DEFAULTS = {
    'dataset': 'EmoC',
    'model': 'basic-block',
    'rsa_model': None,
    'specie': 'D',
    'method': 'mahalanobis',
    'rsa_method': 'kendall',
    'radius': None,          # None -> auto (3 dog / 4 human)
    'z_threshold': 3.1,
    'mask_type': 'b_GreyMatter2mmB',
    'reps': 100,
    'reps_group': 1000,
}

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


def params_from_inputs(dataset, model, rsa_model, specie, method, rsa_method,
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


def run_probe(params, step):
    """Run one step's disk probe and return a JSON-serialisable result dict."""
    ctx = make_ctx(params)
    r = pc.PROBES[step](ctx)
    failures = pc.find_failure_info(ctx, step)
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
    html.P('Checks the actual output files on disk. Nothing is scanned until you '
           'press a "Check" button. Results are remembered per parameter set.',
           style={'color': '#57606a'}),
    param_panel(),
    html.Div([
        html.Button('▶ Check all steps', id='check-all', n_clicks=0,
                    style={'marginRight': '10px', 'fontWeight': 'bold'}),
        html.Button('🗑 Clear all (this parameter set)', id='clear-all', n_clicks=0,
                    style={'marginRight': '10px'}),
        html.Span(id='sig-label', style={'color': '#57606a', 'marginLeft': '10px'}),
    ], style={'margin': '14px 0'}),
    html.Div(id='status-table'),
    html.Hr(),
    html.H4('Step detail'),
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
    models, _folder = pc.list_rsa_models(DATAFOLDER, (dataset or '').strip())
    options = [{'label': m, 'value': m} for m in models]
    value = current if current in models else (models[0] if models else None)
    return options, value


# ---------------------------------------------------------------------------
# Callback: mutating actions (check all / clear all / per-step check/clear/details)
# ---------------------------------------------------------------------------
@app.callback(
    Output('cache-version', 'data'),
    Output('selected-step', 'data'),
    Input('check-all', 'n_clicks'),
    Input('clear-all', 'n_clicks'),
    Input({'type': 'step-check', 'index': ALL}, 'n_clicks'),
    Input({'type': 'step-clear', 'index': ALL}, 'n_clicks'),
    Input({'type': 'step-details', 'index': ALL}, 'n_clicks'),
    State('cache-version', 'data'),
    State('p-dataset', 'value'), State('p-model', 'value'), State('p-rsa_model', 'value'),
    State('p-specie', 'value'), State('p-method', 'value'), State('p-rsa_method', 'value'),
    State('p-radius', 'value'), State('p-z_threshold', 'value'), State('p-mask_type', 'value'),
    State('p-reps', 'value'), State('p-reps_group', 'value'),
    prevent_initial_call=True,
)
def do_action(_ca, _cl, _sc, _sx, _sd, version,
              dataset, model, rsa_model, specie, method, rsa_method,
              radius, z_threshold, mask_type, reps, reps_group):
    trig = callback_context.triggered
    if not trig or trig[0]['value'] in (None, 0):
        return no_update, no_update
    prop = trig[0]['prop_id']

    params = params_from_inputs(dataset, model, rsa_model, specie, method, rsa_method,
                                radius, z_threshold, mask_type, reps, reps_group)
    if not params['rsa_model']:
        return no_update, no_update
    sig = signature(params)
    cache = load_cache()
    entry = cache.setdefault(sig, {'params': params, 'steps': {}})
    entry['params'] = params
    selected = no_update

    def _step_from_prop(p):
        # p looks like '{"index":7,"type":"step-check"}.n_clicks'
        try:
            return json.loads(p.split('.n_clicks')[0])['index']
        except Exception:
            return None

    if prop.startswith('check-all'):
        for step in pc.STEPS:
            entry['steps'][str(step)] = run_probe(params, step)
        # focus the first incomplete step
        selected = None
        for step in pc.STEPS:
            v = entry['steps'][str(step)]['verdict']
            if v not in (pc.DONE, pc.NA):
                selected = step
                break
    elif prop.startswith('clear-all'):
        cache.pop(sig, None)
        save_cache(cache)
        return (version or 0) + 1, None
    elif '"type":"step-check"' in prop or "'type': 'step-check'" in prop:
        step = _step_from_prop(prop)
        if step is not None:
            entry['steps'][str(step)] = run_probe(params, step)
            selected = step
    elif '"type":"step-clear"' in prop or "'type': 'step-clear'" in prop:
        step = _step_from_prop(prop)
        if step is not None:
            entry['steps'].pop(str(step), None)
            selected = no_update
    elif '"type":"step-details"' in prop or "'type': 'step-details'" in prop:
        step = _step_from_prop(prop)
        selected = step if step is not None else no_update

    save_cache(cache)
    return (version or 0) + 1, selected


# ---------------------------------------------------------------------------
# Callback: render the status table (cheap — reads cache only, never scans)
# ---------------------------------------------------------------------------
@app.callback(
    Output('status-table', 'children'),
    Output('sig-label', 'children'),
    Input('cache-version', 'data'),
    Input('p-dataset', 'value'), Input('p-model', 'value'), Input('p-rsa_model', 'value'),
    Input('p-specie', 'value'), Input('p-method', 'value'), Input('p-rsa_method', 'value'),
    Input('p-radius', 'value'), Input('p-z_threshold', 'value'), Input('p-mask_type', 'value'),
    Input('p-reps', 'value'), Input('p-reps_group', 'value'),
)
def render_table(_v, dataset, model, rsa_model, specie, method, rsa_method,
                 radius, z_threshold, mask_type, reps, reps_group):
    params = params_from_inputs(dataset, model, rsa_model, specie, method, rsa_method,
                                radius, z_threshold, mask_type, reps, reps_group)
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
            html.Button('Details', id={'type': 'step-details', 'index': step}, n_clicks=0),
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
    State('p-specie', 'value'), State('p-method', 'value'), State('p-rsa_method', 'value'),
    State('p-radius', 'value'), State('p-z_threshold', 'value'), State('p-mask_type', 'value'),
    State('p-reps', 'value'), State('p-reps_group', 'value'),
)
def render_detail(step, _v, dataset, model, rsa_model, specie, method, rsa_method,
                  radius, z_threshold, mask_type, reps, reps_group):
    if step is None:
        return html.Span('Select a step\'s "Details" (or "Check" a step) to see the '
                         'per-participant breakdown here.', style={'color': '#57606a'})
    params = params_from_inputs(dataset, model, rsa_model, specie, method, rsa_method,
                                radius, z_threshold, mask_type, reps, reps_group)
    sig = signature(params)
    c = load_cache().get(sig, {}).get('steps', {}).get(str(step))
    label = pc.STEP_LABELS.get(step, f'Step {step}')
    if not c:
        return html.Div([html.B(f'Step {step} — {label}'), html.Br(),
                         html.Span('Not checked yet for this parameter set. Press '
                                   '"Check" on that step.', style={'color': '#57606a'})])

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
            ]))
        blocks.append(html.Details([
            html.Summary(f'per-participant ({len(per_sub)})'),
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
    print(f"[pipeline_dashboard] datafolder : {DATAFOLDER}")
    print(f"[pipeline_dashboard] cache file : {CACHE_PATH}")
    print(f"[pipeline_dashboard] open       : http://{args.host}:{args.port}")
    print("[pipeline_dashboard] Ctrl+C stops ONLY this app.")
    # debug=False + use_reloader=False => single process, no child procs, and
    # Ctrl+C here never touches your other consoles.
    app.run(debug=False, use_reloader=False, host=args.host, port=args.port)


if __name__ == '__main__':
    main()

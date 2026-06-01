"""Scheduler tab — view, schedule, retry and inspect RSA pipeline jobs.

A thin Dash UI over the existing file-based job queue (scheduler/ package).
All job-state logic lives in scheduler.jobs / scheduler.dag; this module only
reads the queue, renders it, and triggers the same operations the CLI does.

Standalone:  python -m viz.scheduler_app   ->  http://127.0.0.1:8052
Mounted:     dashboard.py sets $SCHEDULER_URL_BASE and serves app.server.
"""

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

from dash import Dash, html, dcc, dash_table, no_update
from dash.dependencies import Input, Output, State

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scheduler.paths import get_paths, get_queue_dir
from scheduler.dag import build_job_graph, STEP_LABELS
from scheduler.jobs import create_job, load_job, save_job, list_jobs_in_state
from viz import dash_kwargs

STATES = ("running", "pending", "waiting", "completed", "failed")
STATE_COLOR = {
    "running":   "#e0a020",
    "pending":   "#4a90d9",
    "waiting":   "#7a7a7a",
    "completed": "#2faf5a",
    "failed":    "#e23c3c",
}

BG, PANEL_BG, ACCENT = "#ffffff", "#f3f5f9", "#4472C4"
INK, MUTED, LINE = "#222222", "#667085", "#d5dbe5"
DARK_BG = BG  # legacy name, now the light page background
INPUT_STYLE = {"backgroundColor": "#ffffff", "color": INK,
               "border": f"1px solid {LINE}", "borderRadius": "6px", "padding": "5px 8px"}


# --- Queue access (defensive: the share may not be mounted) ---------------

def _queue_dir():
    """Return the queue Path, or None if the data share is unavailable."""
    try:
        datafolder, _, _ = get_paths()
        return get_queue_dir(datafolder)
    except Exception:
        return None


def gather_jobs(queue_dir):
    """All jobs across every state, newest activity first."""
    rows = []
    if queue_dir is None:
        return rows
    for state in STATES:
        for path in list_jobs_in_state(queue_dir, state):
            try:
                job = load_job(path)
            except Exception:
                continue
            rows.append({
                "state": state,
                "job_id": job.get("job_id", path.stem),
                "dataset": job.get("dataset", ""),
                "model": job.get("model", ""),
                "rsa_model": job.get("rsa_model", ""),
                "specie": job.get("specie", ""),
                "step": job.get("step", ""),
                "label": job.get("label", ""),
                "machine": job.get("machine") or "",
                "error": (job.get("error") or "")[:80],
            })
    order = {s: i for i, s in enumerate(STATES)}
    rows.sort(key=lambda r: (order.get(r["state"], 9), r["dataset"], r["rsa_model"],
                             r["specie"], r["step"] if isinstance(r["step"], int) else 99))
    return rows


def find_job_path(queue_dir, job_id):
    for state in STATES:
        p = queue_dir / state / f"{job_id}.json"
        if p.exists():
            return state, p
    return None, None


def latest_log(queue_dir, job_id):
    """Return (path, text) of the most recent log file for a job."""
    log_dir = queue_dir / "logs"
    if not log_dir.is_dir():
        return None, ""
    logs = sorted(log_dir.glob(f"{job_id}_*.log"))
    if not logs:
        return None, ""
    chosen = logs[-1]
    try:
        text = chosen.read_text(errors="replace")
    except Exception as e:
        text = f"<could not read log: {e}>"
    # Tail to keep the UI light.
    if len(text) > 20000:
        text = "... (truncated) ...\n" + text[-20000:]
    return chosen, text


def retry_job(queue_dir, job_id):
    """Move a failed job back into the queue (pending if deps done, else waiting)."""
    state, path = find_job_path(queue_dir, job_id)
    if state != "failed":
        return f"Can only retry failed jobs ({job_id} is {state})."
    job = load_job(path)
    completed = {p.stem for p in (queue_dir / "completed").glob("*.json")}
    deps_done = all(d in completed for d in job.get("deps", []))
    new_state = "pending" if deps_done else "waiting"
    job["status"] = new_state
    job["error"] = None
    job["started_at"] = None
    job["completed_at"] = None
    job["machine"] = None
    dest = queue_dir / new_state / path.name
    path.rename(dest)
    save_job(dest, job)
    return f"Retried {job_id} -> {new_state}."


def cancel_job(queue_dir, job_id):
    """Remove a pending/waiting/failed job from the queue (not running)."""
    state, path = find_job_path(queue_dir, job_id)
    if state is None:
        return f"{job_id} not found."
    if state in ("running", "completed"):
        return f"Refusing to cancel a {state} job."
    path.unlink()
    return f"Cancelled {job_id} (was {state})."


# --- App ------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True, **dash_kwargs("SCHEDULER_URL_BASE"))
app.title = "RSA Scheduler"

TABLE_COLS = [
    {"name": "State", "id": "state"},
    {"name": "Dataset", "id": "dataset"},
    {"name": "Model", "id": "model"},
    {"name": "RSA model", "id": "rsa_model"},
    {"name": "Sp", "id": "specie"},
    {"name": "Step", "id": "step"},
    {"name": "Label", "id": "label"},
    {"name": "Machine", "id": "machine"},
    {"name": "Error", "id": "error"},
]


def _field(label, control):
    return html.Div([html.Label(label, style={"fontSize": "11px", "color": MUTED, "display": "block"}),
                     control])


app.layout = html.Div(style={"backgroundColor": BG, "color": INK, "minHeight": "100vh",
                             "padding": "12px 16px", "fontFamily": "'Segoe UI', Arial, sans-serif"}, children=[
    html.H2("RSA Scheduler", style={"textAlign": "center", "margin": "4px 0 10px", "color": INK}),

    # ---- Schedule new analysis ----
    html.Div(style={"backgroundColor": PANEL_BG, "borderRadius": "8px", "padding": "10px 14px",
                    "border": f"1px solid {LINE}", "marginBottom": "10px"}, children=[
        html.H4("Schedule new analysis", style={"margin": "0 0 8px", "color": INK}),
        html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "alignItems": "flex-end"}, children=[
            _field("Dataset", dcc.Input(id="sch-dataset", value="EmoC", type="text",
                                        style={**INPUT_STYLE, "width": "90px"})),
            _field("GLM model", dcc.Input(id="sch-model", value="basic-block", type="text",
                                          style={**INPUT_STYLE, "width": "130px"})),
            _field("RSA model", dcc.Input(id="sch-rsa", value="", type="text", placeholder="rsa model name",
                                          style={**INPUT_STYLE, "width": "160px"})),
            _field("Species", dcc.Dropdown(id="sch-specie",
                    options=[{"label": "Both", "value": "both"}, {"label": "Dog", "value": "D"},
                             {"label": "Human", "value": "H"}], value="both", clearable=False,
                    style={"width": "100px", "color": "#000"})),
            _field("Target step", dcc.Dropdown(id="sch-target",
                    options=[{"label": f"{s} — {STEP_LABELS[s]}", "value": s} for s in sorted(STEP_LABELS)],
                    value=10, clearable=False, style={"width": "210px", "color": "#000"})),
            _field("z-threshold", dcc.Input(id="sch-zt", value=3.1, type="number", step=0.1,
                                            style={**INPUT_STYLE, "width": "90px"})),
            _field("reps", dcc.Input(id="sch-reps", value=100, type="number",
                                     style={**INPUT_STYLE, "width": "80px"})),
            _field("reps_group", dcc.Input(id="sch-repsg", value=1000, type="number",
                                           style={**INPUT_STYLE, "width": "100px"})),
            html.Button("Schedule", id="sch-submit", n_clicks=0,
                        style={"height": "34px", "padding": "0 20px", "backgroundColor": ACCENT,
                               "color": "white", "border": "none", "borderRadius": "6px",
                               "cursor": "pointer", "fontWeight": "bold"}),
        ]),
        html.Div(id="sch-msg", style={"fontSize": "12px", "color": "#2f7d4f", "marginTop": "8px",
                                      "fontFamily": "Consolas, monospace", "whiteSpace": "pre-wrap"}),
    ]),

    # ---- Job table ----
    html.Div(style={"backgroundColor": PANEL_BG, "borderRadius": "8px", "padding": "10px 14px",
                    "border": f"1px solid {LINE}", "marginBottom": "10px"}, children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "6px"}, children=[
            html.H4("Jobs", style={"margin": 0, "color": INK}),
            html.Button("Refresh", id="sch-refresh", n_clicks=0,
                        style={"padding": "4px 14px", "backgroundColor": "#eef1f6", "color": INK,
                               "border": f"1px solid {LINE}", "borderRadius": "6px", "cursor": "pointer"}),
            html.Span(id="sch-source", style={"fontSize": "11px", "color": MUTED}),
            html.Div(style={"marginLeft": "auto", "display": "flex", "gap": "8px"}, children=[
                html.Button("Retry selected", id="sch-retry", n_clicks=0,
                            style={"padding": "4px 14px", "backgroundColor": "#2faf5a", "color": "white",
                                   "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
                html.Button("Cancel selected", id="sch-cancel", n_clicks=0,
                            style={"padding": "4px 14px", "backgroundColor": "#e23c3c", "color": "white",
                                   "border": "none", "borderRadius": "4px", "cursor": "pointer"}),
            ]),
        ]),
        dash_table.DataTable(
            id="sch-table", columns=TABLE_COLS, data=[],
            row_selectable="single", page_size=20,
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#eef1f6", "color": INK, "fontWeight": "bold"},
            style_cell={"backgroundColor": "#ffffff", "color": INK, "border": f"1px solid {LINE}",
                        "fontSize": "12px", "padding": "4px 8px", "textAlign": "left",
                        "maxWidth": "260px", "overflow": "hidden", "textOverflow": "ellipsis"},
            style_data_conditional=[
                {"if": {"filter_query": f"{{state}} = '{s}'", "column_id": "state"},
                 "color": c, "fontWeight": "bold"} for s, c in STATE_COLOR.items()
            ],
        ),
    ]),

    # ---- Log / detail viewer ----
    html.Div(style={"backgroundColor": PANEL_BG, "borderRadius": "8px", "padding": "10px 14px",
                    "border": f"1px solid {LINE}"}, children=[
        html.H4("Job detail & log", style={"margin": "0 0 6px", "color": INK}),
        html.Div(id="sch-detail", style={"fontSize": "12px", "color": INK,
                                         "fontFamily": "Consolas, monospace", "marginBottom": "6px"}),
        html.Pre(id="sch-log", style={"backgroundColor": "#f3f5f9", "color": INK, "padding": "10px",
                                      "border": f"1px solid {LINE}",
                                      "borderRadius": "6px", "maxHeight": "360px", "overflowY": "auto",
                                      "fontSize": "11px", "whiteSpace": "pre-wrap"}),
    ]),

    dcc.Interval(id="sch-tick", interval=15000, n_intervals=0),
    dcc.Store(id="sch-action-store"),
])


@app.callback(
    Output("sch-table", "data"), Output("sch-source", "children"),
    Input("sch-refresh", "n_clicks"), Input("sch-tick", "n_intervals"),
    Input("sch-action-store", "data"),
)
def cb_refresh(_n, _t, _action):
    qd = _queue_dir()
    if qd is None:
        return [], "Queue unavailable — data share not mounted."
    rows = gather_jobs(qd)
    counts = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1
    summary = "  ".join(f"{s}:{counts.get(s, 0)}" for s in STATES)
    return rows, f"{qd}    |    {summary}"


@app.callback(
    Output("sch-msg", "children"),
    Input("sch-submit", "n_clicks"),
    State("sch-dataset", "value"), State("sch-model", "value"), State("sch-rsa", "value"),
    State("sch-specie", "value"), State("sch-target", "value"), State("sch-zt", "value"),
    State("sch-reps", "value"), State("sch-repsg", "value"),
    prevent_initial_call=True,
)
def cb_schedule(_n, dataset, model, rsa_model, specie, target, zt, reps, repsg):
    if not (dataset and rsa_model):
        return "Dataset and RSA model are required."
    qd = _queue_dir()
    if qd is None:
        return "Queue unavailable — data share not mounted."
    species = ["H", "D"] if specie == "both" else [specie]
    created, lines = 0, []
    for sp in species:
        jobs = build_job_graph(dataset=dataset, model=model, rsa_model=rsa_model, specie=sp,
                               target_step=int(target), z_threshold=float(zt),
                               reps=int(reps), reps_group=int(repsg))
        for job in jobs:
            ok = create_job(qd, job)
            created += int(bool(ok))
            mark = "+" if ok else "="
            lines.append(f"  [{mark}] {sp} step {job['step']:02d} {job['label']:<26} -> {job['status']}")
    return f"Created {created} new job(s):\n" + "\n".join(lines)


@app.callback(
    Output("sch-action-store", "data"),
    Input("sch-retry", "n_clicks"), Input("sch-cancel", "n_clicks"),
    State("sch-table", "data"), State("sch-table", "selected_rows"),
    prevent_initial_call=True,
)
def cb_action(retry_clicks, cancel_clicks, data, selected):
    from dash import ctx
    if not selected or not data:
        return {"msg": "no selection", "ts": datetime.now().isoformat()}
    qd = _queue_dir()
    if qd is None:
        return {"msg": "queue unavailable", "ts": datetime.now().isoformat()}
    job_id = data[selected[0]]["job_id"]
    trigger = ctx.triggered_id
    if trigger == "sch-retry":
        msg = retry_job(qd, job_id)
    else:
        msg = cancel_job(qd, job_id)
    return {"msg": msg, "ts": datetime.now().isoformat()}


@app.callback(
    Output("sch-detail", "children"), Output("sch-log", "children"),
    Input("sch-table", "selected_rows"),
    State("sch-table", "data"),
)
def cb_detail(selected, data):
    if not selected or not data:
        return "Select a job to see its details and log.", ""
    qd = _queue_dir()
    if qd is None:
        return "Queue unavailable.", ""
    job_id = data[selected[0]]["job_id"]
    state, path = find_job_path(qd, job_id)
    if path is None:
        return f"{job_id} no longer in queue.", ""
    job = load_job(path)
    detail = (f"{job_id}\nstate={state}  machine={job.get('machine') or '-'}  "
              f"started={job.get('started_at') or '-'}  completed={job.get('completed_at') or '-'}")
    if job.get("error"):
        detail += f"\nerror: {job['error']}"
    log_path, text = latest_log(qd, job_id)
    if log_path:
        detail += f"\nlog: {log_path}"
    return detail, text or "<no log file found>"


if __name__ == "__main__":
    app.run(debug=True, port=8052)

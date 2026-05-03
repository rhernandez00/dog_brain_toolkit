"""
RSA Model Builder — interactive editor for RSA dissimilarity matrices.

Reads stimulus definitions from an experiment YAML (model_dict block) and lets
the user build a dissimilarity matrix interactively, with bulk edit rules by
stimulus attribute (label / specie_shown / run), full and grouped views, and
CSV export in the format consumed by rsa_utils.read_model_dict.

Launch:  python rsa_model_builder.py
Then open http://127.0.0.1:8051 in your browser.

Requires: dash, plotly, pandas, numpy, pyyaml
"""

import os
import io
import math
import json
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, no_update, ctx
from dash.dependencies import Input, Output, State, ALL


DEFAULT_YAML = r"G:\My Drive\Results\EmoC\config_files\D_basic-block.yaml"
DEFAULT_EXPORT_DIR = r"G:\My Drive\Results\EmoC\rsa_models"

# Attributes that are never shown as grouping/bulk-rule options. Everything
# else found in the YAML stim dict (plus the synthetic 'run' field when
# combined-mode is in use) is exposed dynamically.
HIDDEN_ATTRS = {"color", "name"}

ALL_RUNS_KEY = "__all__"
NAN_SENTINEL = "NaN"  # how NaN is rendered in the table / CSV


# ---------------------------------------------------------------------------
# YAML / stim loading
# ---------------------------------------------------------------------------

def load_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_runs(cfg: dict) -> list:
    md = cfg.get("model_dict") or {}
    return list(md.keys())


def load_stims(cfg: dict, run_key: str) -> list:
    """Build the ordered stim list for a single run or all runs combined."""
    md = cfg.get("model_dict") or {}
    if run_key == ALL_RUNS_KEY:
        run_keys = list(md.keys())
    else:
        if run_key not in md:
            raise ValueError(f"Run '{run_key}' not in model_dict")
        run_keys = [run_key]

    stims = []
    for rk in run_keys:
        run_block = md[rk] or {}
        for stim_name, attrs in run_block.items():
            entry = {"name": stim_name, "run": rk}
            # Carry over every attribute from the YAML so the UI can expose
            # any category present in model_dict (except hidden ones).
            for k, v in (attrs or {}).items():
                entry[k] = v
            entry.setdefault("color", "#cccccc")
            stims.append(entry)
    return stims


def discover_attrs(stims: list) -> list:
    """Ordered list of grouping attribute keys found in the YAML, minus
    HIDDEN_ATTRS. 'run' is appended only if more than one run is present."""
    seen = []
    for s in stims:
        for k in s.keys():
            if k in HIDDEN_ATTRS or k == "run":
                continue
            if k not in seen:
                seen.append(k)
    runs = {s.get("run") for s in stims}
    if len(runs) > 1:
        seen.append("run")
    return seen


def display_name(stim: dict, combined: bool) -> str:
    return f"{stim['run']}_{stim['name']}" if combined else stim["name"]


# ---------------------------------------------------------------------------
# Grouping / view helpers
# ---------------------------------------------------------------------------

def _group_key(stim: dict, group_by: Sequence[str]) -> str:
    if not group_by:
        return stim["name"]  # full view
    parts = []
    for k in group_by:
        v = stim.get(k, "")
        # condense common values for nicer codes (Dog/Hum stay; label is single letter already)
        parts.append(str(v))
    return "".join(parts)


def axis_codes(stims: list, group_by: Sequence[str], combined: bool) -> tuple:
    """Return (labels, stim_to_group_index).

    Full view: labels are the stim display names; mapping is identity.
    Grouped view: labels are unique group keys preserving first-seen order.
    """
    if not group_by:
        labels = [display_name(s, combined) for s in stims]
        return labels, list(range(len(stims)))

    seen = {}
    mapping = []
    for s in stims:
        key = _group_key(s, group_by)
        if key not in seen:
            seen[key] = len(seen)
        mapping.append(seen[key])
    labels = list(seen.keys())
    return labels, mapping


# ---------------------------------------------------------------------------
# Matrix invariants
# ---------------------------------------------------------------------------

def fresh_matrix(n: int) -> np.ndarray:
    m = np.full((n, n), np.nan, dtype=np.float64)
    np.fill_diagonal(m, 0.0)
    return m


def enforce_invariants(m: np.ndarray) -> np.ndarray:
    """Force symmetry (mirror upper to lower) and diag=0."""
    m = np.array(m, dtype=np.float64, copy=True)
    iu = np.triu_indices_from(m, k=1)
    m[(iu[1], iu[0])] = m[iu]
    np.fill_diagonal(m, 0.0)
    return m


def set_pair(m: np.ndarray, i: int, j: int, value):
    """Write (i,j) and (j,i); diagonal stays 0."""
    if i == j:
        return
    v = float(value) if value is not None and not (isinstance(value, float) and math.isnan(value)) else np.nan
    m[i, j] = v
    m[j, i] = v


def broadcast_grouped_edit(matrix_full: np.ndarray, mapping: list, gi: int, gj: int, value):
    """Set every (i,j) whose grouped indices are (gi,gj) to value (and mirror)."""
    rows = [i for i, g in enumerate(mapping) if g == gi]
    cols = [j for j, g in enumerate(mapping) if g == gj]
    for i in rows:
        for j in cols:
            if i == j:
                continue
            set_pair(matrix_full, i, j, value)


def grouped_view(matrix_full: np.ndarray, mapping: list, n_groups: int) -> tuple:
    """Collapse full matrix to grouped matrix.

    For each grouped cell, take the unique value across the underlying block.
    If the block has mixed values, return NaN and mark mixed=True.
    """
    g = np.full((n_groups, n_groups), np.nan, dtype=np.float64)
    mixed = np.zeros((n_groups, n_groups), dtype=bool)
    mapping = np.asarray(mapping)
    for gi in range(n_groups):
        rows = np.where(mapping == gi)[0]
        for gj in range(n_groups):
            cols = np.where(mapping == gj)[0]
            if len(rows) == 0 or len(cols) == 0:
                continue
            block = matrix_full[np.ix_(rows, cols)]
            # Collapse considering NaN as a distinct value
            flat = block.flatten()
            uniq = []
            for v in flat:
                if math.isnan(v):
                    if not any(isinstance(u, float) and math.isnan(u) for u in uniq):
                        uniq.append(float("nan"))
                else:
                    if v not in [u for u in uniq if not (isinstance(u, float) and math.isnan(u))]:
                        uniq.append(v)
            if len(uniq) == 1:
                g[gi, gj] = uniq[0]
            else:
                g[gi, gj] = np.nan
                mixed[gi, gj] = True
    np.fill_diagonal(g, 0.0)
    return g, mixed


# ---------------------------------------------------------------------------
# Bulk rule
# ---------------------------------------------------------------------------

def apply_bulk_rule(matrix_full: np.ndarray, stims: list,
                    lhs_attr: str, lhs_val: str,
                    rhs_attr: str, rhs_val: str,
                    value, only_nan: bool = False) -> np.ndarray:
    rows = [i for i, s in enumerate(stims)
            if (lhs_val == "*" or str(s.get(lhs_attr, "")) == lhs_val)]
    cols = [j for j, s in enumerate(stims)
            if (rhs_val == "*" or str(s.get(rhs_attr, "")) == rhs_val)]
    for i in rows:
        for j in cols:
            if i == j:
                continue
            if only_nan and not math.isnan(matrix_full[i, j]):
                continue
            set_pair(matrix_full, i, j, value)
    return matrix_full


# ---------------------------------------------------------------------------
# Serialization for dcc.Store (JSON-safe)
# ---------------------------------------------------------------------------

def matrix_to_json(m: np.ndarray):
    return [[None if math.isnan(v) else float(v) for v in row] for row in m]


def matrix_from_json(data) -> np.ndarray:
    arr = np.array(
        [[np.nan if v is None else float(v) for v in row] for row in data],
        dtype=np.float64,
    )
    return arr


def parse_value(text):
    """Convert user input to a float or NaN. Empty/'nan' -> NaN."""
    if text is None:
        return np.nan
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        if isinstance(text, float) and math.isnan(text):
            return np.nan
        return float(text)
    s = str(text).strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def to_export_dataframe(matrix: np.ndarray, labels: list) -> pd.DataFrame:
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    return df


def dataframe_to_csv_string(df: pd.DataFrame) -> str:
    """Match the example file: leading comma row, NaN written as 'NaN', integers without trailing .0."""
    def fmt(v):
        if isinstance(v, float):
            if math.isnan(v):
                return "NaN"
            if v.is_integer():
                return str(int(v))
            return repr(v)
        return str(v)

    lines = []
    lines.append("," + ",".join(str(c) for c in df.columns))
    for idx, row in df.iterrows():
        lines.append(str(idx) + "," + ",".join(fmt(v) for v in row.tolist()))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Heatmap rendering
# ---------------------------------------------------------------------------

def build_heatmap(matrix: np.ndarray, labels: list, axis_colors: list,
                  mixed_mask: np.ndarray = None) -> go.Figure:
    z = matrix.copy()
    text = np.empty(z.shape, dtype=object)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z[i, j]
            if math.isnan(v):
                text[i, j] = "NaN" if not (mixed_mask is not None and mixed_mask[i, j]) else "mix"
            else:
                text[i, j] = (str(int(v)) if float(v).is_integer() else f"{v:.3g}")

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Viridis",
            zmin=0,
            zmax=1,
            hovertemplate="row=%{y}<br>col=%{x}<br>value=%{text}<extra></extra>",
            showscale=True,
        )
    )
    n = len(labels)
    # Tick font color bands (using the stim color attribute if available)
    fig.update_layout(
        margin=dict(l=80, r=20, t=30, b=80),
        height=max(450, 22 * n + 120),
        xaxis=dict(side="bottom", tickangle=-60, automargin=True),
        yaxis=dict(autorange="reversed", automargin=True),
        plot_bgcolor="#fafafa",
    )
    # Colored axis label "ticks": draw small colored squares as shapes if axis_colors provided
    if axis_colors and len(axis_colors) == n:
        shapes = []
        for i, c in enumerate(axis_colors):
            shapes.append(dict(
                type="rect", xref="x", yref="paper",
                x0=i - 0.5, x1=i + 0.5, y0=-0.06, y1=-0.03,
                fillcolor=c, line=dict(width=0),
            ))
            shapes.append(dict(
                type="rect", xref="paper", yref="y",
                x0=-0.06, x1=-0.03, y0=i - 0.5, y1=i + 0.5,
                fillcolor=c, line=dict(width=0),
            ))
        fig.update_layout(shapes=shapes)
    return fig


def representative_color(stims: list, mapping: list, n_groups: int) -> list:
    out = ["#cccccc"] * n_groups
    for i, g in enumerate(mapping):
        if out[g] == "#cccccc":
            out[g] = stims[i].get("color", "#cccccc")
    return out


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "RSA Model Builder"


def attr_value_options(stims: list, attr: str):
    if not stims or not attr:
        return [{"label": "(any)", "value": "*"}]
    vals = []
    for s in stims:
        v = str(s.get(attr, ""))
        if v not in vals:
            vals.append(v)
    return [{"label": "(any)", "value": "*"}] + [{"label": v, "value": v} for v in vals]


CONTROL_BOX = {"border": "1px solid #ddd", "borderRadius": "6px",
               "padding": "10px", "marginBottom": "10px",
               "background": "#fcfcfc"}

app.layout = html.Div([
    dcc.Store(id="store-cfg"),       # parsed YAML
    dcc.Store(id="store-stims"),     # ordered stim list
    dcc.Store(id="store-matrix"),    # full matrix (json)
    dcc.Store(id="store-meta"),      # {"combined": bool, "yaml_path": str}
    dcc.Download(id="download-csv"),

    html.H2("RSA Model Builder", style={"marginBottom": "4px"}),
    html.Div("Build dissimilarity matrices from experiment YAML stim definitions.",
             style={"color": "#666", "marginBottom": "10px"}),

    # ---- Top bar: load YAML + run picker + view toggle ----
    html.Div([
        html.Div([
            html.Label("YAML config path"),
            dcc.Input(id="input-yaml", type="text", value=DEFAULT_YAML,
                      style={"width": "100%"}),
        ], style={"flex": "3", "marginRight": "10px"}),
        html.Div([
            html.Label(" "),
            html.Button("Load YAML", id="btn-load", n_clicks=0,
                        style={"width": "100%", "height": "32px"}),
        ], style={"flex": "1", "marginRight": "10px"}),
        html.Div([
            html.Label("Run"),
            dcc.Dropdown(id="dd-run", options=[], value=None, clearable=False),
        ], style={"flex": "1", "marginRight": "10px"}),
        html.Div([
            html.Label("View"),
            dcc.RadioItems(id="radio-view",
                           options=[{"label": "Full", "value": "full"},
                                    {"label": "Grouped", "value": "grouped"}],
                           value="grouped",
                           inline=True),
        ], style={"flex": "1", "marginRight": "10px"}),
        html.Div([
            html.Label("Group by"),
            dcc.Checklist(id="chk-groupby",
                          options=[],
                          value=[],
                          inline=True),
        ], style={"flex": "2"}),
    ], style={"display": "flex", "alignItems": "flex-end", **CONTROL_BOX}),

    html.Div(id="status", style={"color": "#a33", "marginBottom": "8px"}),

    # ---- Main: heatmap + side panel ----
    html.Div([
        html.Div([
            dcc.Graph(id="heatmap", config={"displayModeBar": True}),
            html.Div([
                html.Label("Edit selected cell — value (blank or 'NaN' clears):"),
                html.Div([
                    dcc.Input(id="cell-row", type="text", placeholder="row", disabled=True,
                              style={"width": "120px", "marginRight": "6px"}),
                    dcc.Input(id="cell-col", type="text", placeholder="col", disabled=True,
                              style={"width": "120px", "marginRight": "6px"}),
                    dcc.Input(id="cell-value", type="text", placeholder="value",
                              style={"width": "80px", "marginRight": "6px"}),
                    html.Button("Set", id="btn-set-cell", n_clicks=0),
                ], style={"display": "flex"}),
            ], style={"marginTop": "6px"}),
        ], style={"flex": "3", "marginRight": "12px"}),

        html.Div([
            html.H4("Bulk rules", style={"marginTop": 0}),
            html.Div([
                html.Div([
                    html.Label("Row attr"),
                    dcc.Dropdown(id="bulk-lhs-attr",
                                 options=[], value=None, clearable=False),
                ], style={"flex": "1", "marginRight": "6px"}),
                html.Div([
                    html.Label("Row value"),
                    dcc.Dropdown(id="bulk-lhs-val", options=[], value="*", clearable=False),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "marginBottom": "6px"}),
            html.Div([
                html.Div([
                    html.Label("Col attr"),
                    dcc.Dropdown(id="bulk-rhs-attr",
                                 options=[], value=None, clearable=False),
                ], style={"flex": "1", "marginRight": "6px"}),
                html.Div([
                    html.Label("Col value"),
                    dcc.Dropdown(id="bulk-rhs-val", options=[], value="*", clearable=False),
                ], style={"flex": "1"}),
            ], style={"display": "flex", "marginBottom": "6px"}),
            html.Div([
                html.Label("Value"),
                html.Div([
                    dcc.Input(id="bulk-value", type="text", value="0",
                              style={"flex": "1", "marginRight": "4px"}),
                    html.Button("0", id="btn-quick-0", n_clicks=0,
                                style={"width": "30px", "marginRight": "2px"}),
                    html.Button("1", id="btn-quick-1", n_clicks=0,
                                style={"width": "30px", "marginRight": "2px"}),
                    html.Button("NaN", id="btn-quick-nan", n_clicks=0,
                                style={"width": "44px"}),
                ], style={"display": "flex"}),
                html.Div("Tip: leave blank or type 'NaN' to clear pairs.",
                         style={"fontSize": "11px", "color": "#888"}),
            ], style={"marginBottom": "6px"}),
            dcc.Checklist(id="bulk-only-nan",
                          options=[{"label": " only fill NaN cells", "value": "only_nan"}],
                          value=[]),
            html.Button("Apply rule", id="btn-bulk-apply", n_clicks=0,
                        style={"width": "100%", "marginTop": "6px"}),
            html.Button("Fill all empty (NaN) cells with above value",
                        id="btn-fill-nan", n_clicks=0,
                        style={"width": "100%", "marginTop": "4px"}),
            html.Hr(),
            html.Button("Reset matrix (NaN, diag=0)", id="btn-reset", n_clicks=0,
                        style={"width": "100%", "marginBottom": "6px"}),
            html.Button("Mirror upper -> lower", id="btn-mirror", n_clicks=0,
                        style={"width": "100%", "marginBottom": "6px"}),
            html.Hr(),
            html.H4("Export"),
            html.Label("Filename"),
            dcc.Input(id="export-filename", type="text", value="my-model.csv",
                      style={"width": "100%", "marginBottom": "6px"}),
            html.Label("Export folder (saved on server)"),
            dcc.Input(id="export-folder", type="text", value=DEFAULT_EXPORT_DIR,
                      style={"width": "100%", "marginBottom": "6px"}),
            html.Button("Export CSV (download + save)", id="btn-export", n_clicks=0,
                        style={"width": "100%"}),
            html.Div(id="export-status", style={"color": "#393", "marginTop": "6px"}),
        ], style={"flex": "1", **CONTROL_BOX}),
    ], style={"display": "flex"}),

    html.Div([
        html.H4("Matrix table (editable)"),
        dash_table.DataTable(
            id="table",
            editable=True,
            columns=[],
            data=[],
            style_table={"overflowX": "auto", "maxHeight": "420px",
                         "overflowY": "auto"},
            style_cell={"textAlign": "center", "minWidth": "60px",
                        "maxWidth": "120px", "padding": "4px",
                        "fontFamily": "monospace"},
            style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold"},
            fixed_rows={"headers": True},
            fixed_columns={"headers": True, "data": 1},
        ),
    ], style={"marginTop": "12px"}),
], style={"fontFamily": "Segoe UI, Arial, sans-serif", "margin": "12px"})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("store-cfg", "data"),
    Output("dd-run", "options"),
    Output("dd-run", "value"),
    Output("status", "children"),
    Input("btn-load", "n_clicks"),
    State("input-yaml", "value"),
    prevent_initial_call=False,
)
def load_yaml_cb(n, path):
    path = (path or "").strip()
    if not path or not os.path.exists(path):
        return None, [], None, f"YAML not found: {path}"
    try:
        cfg = load_yaml(path)
    except Exception as e:
        return None, [], None, f"Failed to parse YAML: {e}"
    runs = list_runs(cfg)
    if not runs:
        return None, [], None, "YAML has no model_dict / runs."
    options = [{"label": r, "value": r} for r in runs]
    options.append({"label": "All runs (combined)", "value": ALL_RUNS_KEY})
    return cfg, options, runs[0], f"Loaded {path} ({len(runs)} runs)."


@app.callback(
    Output("store-stims", "data"),
    Output("store-matrix", "data"),
    Output("store-meta", "data"),
    Input("store-cfg", "data"),
    Input("dd-run", "value"),
    State("input-yaml", "value"),
    prevent_initial_call=True,
)
def build_stims(cfg, run_key, yaml_path):
    if not cfg or not run_key:
        return no_update, no_update, no_update
    stims = load_stims(cfg, run_key)
    n = len(stims)
    if n == 0:
        return [], [[]], {"combined": False, "yaml_path": yaml_path}
    m = fresh_matrix(n)
    return stims, matrix_to_json(m), {"combined": run_key == ALL_RUNS_KEY,
                                      "yaml_path": yaml_path}


@app.callback(
    Output("chk-groupby", "options"),
    Output("chk-groupby", "value"),
    Output("bulk-lhs-attr", "options"),
    Output("bulk-lhs-attr", "value"),
    Output("bulk-rhs-attr", "options"),
    Output("bulk-rhs-attr", "value"),
    Input("store-stims", "data"),
    State("chk-groupby", "value"),
    State("bulk-lhs-attr", "value"),
    State("bulk-rhs-attr", "value"),
)
def populate_attrs(stims, current_group, current_lhs, current_rhs):
    if not stims:
        return [], [], [], None, [], None
    attrs = discover_attrs(stims)
    opts = [{"label": k, "value": k} for k in attrs]
    # Sensible defaults: prefer specie_shown + label if present, else first attr.
    default_group = [k for k in ("specie_shown", "label") if k in attrs] or attrs[:1]
    group_val = [k for k in (current_group or []) if k in attrs] or default_group
    default_attr = "label" if "label" in attrs else attrs[0]
    lhs = current_lhs if current_lhs in attrs else default_attr
    rhs = current_rhs if current_rhs in attrs else default_attr
    return opts, group_val, opts, lhs, opts, rhs


@app.callback(
    Output("bulk-lhs-val", "options"),
    Output("bulk-lhs-val", "value"),
    Input("bulk-lhs-attr", "value"),
    Input("store-stims", "data"),
)
def update_lhs_vals(attr, stims):
    opts = attr_value_options(stims or [], attr)
    return opts, "*"


@app.callback(
    Output("bulk-rhs-val", "options"),
    Output("bulk-rhs-val", "value"),
    Input("bulk-rhs-attr", "value"),
    Input("store-stims", "data"),
)
def update_rhs_vals(attr, stims):
    opts = attr_value_options(stims or [], attr)
    return opts, "*"


def _current_view(stims, matrix_full, view_mode, group_by, combined):
    if view_mode == "full" or not group_by:
        labels = [display_name(s, combined) for s in stims]
        mapping = list(range(len(stims)))
        m = enforce_invariants(matrix_full)
        mixed = np.zeros_like(m, dtype=bool)
        return labels, mapping, m, mixed
    labels, mapping = axis_codes(stims, group_by, combined)
    m, mixed = grouped_view(matrix_full, mapping, len(labels))
    return labels, mapping, m, mixed


@app.callback(
    Output("heatmap", "figure"),
    Output("table", "columns"),
    Output("table", "data"),
    Input("store-stims", "data"),
    Input("store-matrix", "data"),
    Input("radio-view", "value"),
    Input("chk-groupby", "value"),
    State("store-meta", "data"),
)
def render(stims, matrix_data, view_mode, group_by, meta):
    if not stims or not matrix_data:
        return go.Figure(), [], []
    matrix_full = matrix_from_json(matrix_data)
    combined = bool(meta and meta.get("combined"))
    labels, mapping, m, mixed = _current_view(stims, matrix_full, view_mode,
                                              group_by or [], combined)
    axis_colors = representative_color(stims, mapping, len(labels))
    fig = build_heatmap(m, labels, axis_colors, mixed_mask=mixed)

    # Table
    columns = [{"name": "", "id": "__row__", "editable": False}]
    for c in labels:
        columns.append({"name": c, "id": c, "editable": True})
    rows = []
    for i, lab in enumerate(labels):
        row = {"__row__": lab}
        for j, c in enumerate(labels):
            v = m[i, j]
            row[c] = "" if math.isnan(v) else (str(int(v)) if float(v).is_integer() else f"{v:.3g}")
        rows.append(row)
    return fig, columns, rows


@app.callback(
    Output("cell-row", "value"),
    Output("cell-col", "value"),
    Input("heatmap", "clickData"),
)
def heatmap_click(click):
    if not click or not click.get("points"):
        return no_update, no_update
    p = click["points"][0]
    return p.get("y", ""), p.get("x", "")


@app.callback(
    Output("bulk-value", "value"),
    Input("btn-quick-0", "n_clicks"),
    Input("btn-quick-1", "n_clicks"),
    Input("btn-quick-nan", "n_clicks"),
    prevent_initial_call=True,
)
def quick_fill_value(n0, n1, nn):
    t = ctx.triggered_id
    if t == "btn-quick-0":
        return "0"
    if t == "btn-quick-1":
        return "1"
    if t == "btn-quick-nan":
        return "NaN"
    return no_update


@app.callback(
    Output("store-matrix", "data", allow_duplicate=True),
    Input("btn-set-cell", "n_clicks"),
    Input("btn-bulk-apply", "n_clicks"),
    Input("btn-fill-nan", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    Input("btn-mirror", "n_clicks"),
    Input("table", "data"),
    State("store-matrix", "data"),
    State("store-stims", "data"),
    State("store-meta", "data"),
    State("radio-view", "value"),
    State("chk-groupby", "value"),
    State("cell-row", "value"),
    State("cell-col", "value"),
    State("cell-value", "value"),
    State("bulk-lhs-attr", "value"),
    State("bulk-lhs-val", "value"),
    State("bulk-rhs-attr", "value"),
    State("bulk-rhs-val", "value"),
    State("bulk-value", "value"),
    State("bulk-only-nan", "value"),
    State("table", "columns"),
    prevent_initial_call=True,
)
def edit_matrix(n_set, n_bulk, n_fill_nan, n_reset, n_mirror, table_data,
                matrix_data, stims, meta, view_mode, group_by,
                cell_row, cell_col, cell_value,
                lhs_attr, lhs_val, rhs_attr, rhs_val, bulk_value, only_nan_chk,
                columns):
    if not stims or not matrix_data:
        return no_update
    trigger = ctx.triggered_id
    matrix_full = matrix_from_json(matrix_data)
    combined = bool(meta and meta.get("combined"))
    n = len(stims)

    if trigger == "btn-reset":
        matrix_full = fresh_matrix(n)
        return matrix_to_json(matrix_full)

    if trigger == "btn-mirror":
        iu = np.triu_indices_from(matrix_full, k=1)
        matrix_full[(iu[1], iu[0])] = matrix_full[iu]
        np.fill_diagonal(matrix_full, 0.0)
        return matrix_to_json(matrix_full)

    if trigger == "btn-set-cell":
        if not cell_row or not cell_col:
            return no_update
        labels, mapping, _, _ = _current_view(stims, matrix_full, view_mode,
                                              group_by or [], combined)
        if cell_row not in labels or cell_col not in labels:
            return no_update
        gi = labels.index(cell_row)
        gj = labels.index(cell_col)
        val = parse_value(cell_value)
        if view_mode == "full" or not group_by:
            set_pair(matrix_full, gi, gj, val)
        else:
            broadcast_grouped_edit(matrix_full, mapping, gi, gj, val)
        return matrix_to_json(matrix_full)

    if trigger == "btn-bulk-apply":
        val = parse_value(bulk_value)
        only_nan = "only_nan" in (only_nan_chk or [])
        apply_bulk_rule(matrix_full, stims, lhs_attr, lhs_val,
                        rhs_attr, rhs_val, val, only_nan=only_nan)
        return matrix_to_json(matrix_full)

    if trigger == "btn-fill-nan":
        val = parse_value(bulk_value)
        apply_bulk_rule(matrix_full, stims, "label", "*", "label", "*",
                        val, only_nan=True)
        return matrix_to_json(matrix_full)

    if trigger == "table":
        # Sync from edited table back into matrix_full.
        labels, mapping, _, _ = _current_view(stims, matrix_full, view_mode,
                                              group_by or [], combined)
        if not table_data or not columns:
            return no_update
        col_ids = [c["id"] for c in columns if c["id"] != "__row__"]
        if col_ids != labels:
            return no_update
        for i, row in enumerate(table_data):
            for j, c in enumerate(col_ids):
                v = parse_value(row.get(c, ""))
                if i == j:
                    continue
                if view_mode == "full" or not group_by:
                    set_pair(matrix_full, i, j, v)
                else:
                    broadcast_grouped_edit(matrix_full, mapping, i, j, v)
        return matrix_to_json(matrix_full)

    return no_update


@app.callback(
    Output("download-csv", "data"),
    Output("export-status", "children"),
    Input("btn-export", "n_clicks"),
    State("store-stims", "data"),
    State("store-matrix", "data"),
    State("store-meta", "data"),
    State("radio-view", "value"),
    State("chk-groupby", "value"),
    State("export-filename", "value"),
    State("export-folder", "value"),
    prevent_initial_call=True,
)
def do_export(n, stims, matrix_data, meta, view_mode, group_by, fname, folder):
    if not stims or not matrix_data:
        return no_update, "Nothing to export."
    matrix_full = matrix_from_json(matrix_data)
    combined = bool(meta and meta.get("combined"))
    labels, mapping, m, _ = _current_view(stims, matrix_full, view_mode,
                                          group_by or [], combined)
    df = to_export_dataframe(m, labels)
    csv_text = dataframe_to_csv_string(df)
    fname = (fname or "my-model.csv").strip()
    if not fname.lower().endswith(".csv"):
        fname += ".csv"
    saved_msg = ""
    if folder:
        try:
            os.makedirs(folder, exist_ok=True)
            full_path = os.path.join(folder, fname)
            with open(full_path, "w", encoding="utf-8", newline="") as f:
                f.write(csv_text)
            saved_msg = f"Saved: {full_path}"
        except Exception as e:
            saved_msg = f"Save failed: {e}"
    return dict(content=csv_text, filename=fname), saved_msg


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8051)

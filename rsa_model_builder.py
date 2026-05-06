"""
RSA Model Builder — interactive editor for RSA dissimilarity matrices.

Launch:  python rsa_model_builder.py   →   http://127.0.0.1:8051
Requires: dash, plotly>=5.0, pandas, numpy, pyyaml
"""

import os
import math
import json
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
import plotly.graph_objects as go
import plotly.colors as pc
from dash import Dash, html, dcc, dash_table, no_update, ctx
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate


DEFAULT_YAML       = r"G:\My Drive\Results\EmoC\config_files\D_basic-block.yaml"
DEFAULT_EXPORT_DIR = r"G:\My Drive\Results\EmoC\rsa_models"
MAX_UNDO = 50

HIDDEN_ATTRS   = {"color"}
ALL_RUNS_KEY   = "__all__"
NAN_SENTINEL   = "NaN"

SEPARATOR_OPTIONS = [
    {"label": "none  ('')",        "value": ""},
    {"label": "hyphen  ('-')",     "value": "-"},
    {"label": "underscore  ('_')", "value": "_"},
    {"label": "space  (' ')",      "value": " "},
]

COLORSCALE_OPTIONS = [
    "Viridis","Plasma","Inferno","Magma","Cividis",
    "Blues","Greens","Reds","Greys","Purples","Oranges",
    "RdBu","RdYlGn","RdYlBu","BrBG","PiYG","PRGn","PuOr","Spectral",
    "Hot","Jet","Rainbow","YlOrRd","YlGnBu","PuBuGn",
]

DEFAULT_STYLE = {
    "cell_size":        40,
    "cell_gap":         4,
    "cell_radius":      6,
    "colorscale":       "Viridis",
    "cbar_min":         0.0,
    "cbar_max":         1.0,
    "show_colorbar":    True,
    "use_legend":       False,
    "legend_show_nan":  True,
    "nan_color":        "#cccccc",
    "diag_color":       "#f0f0f0",
    "mixed_color":      "#ff9999",
    "show_values":      True,
    "val_font_size":    9,
    "val_font_color":   "#ffffff",
    "label_font_size":  11,
    "label_font_color": "#222222",
    "show_x_labels":    True,
    "show_y_labels":    True,
    "x_label_angle":    -45,
    "y_label_angle":    0,
    "bg_color":         "#ffffff",
    "paper_bg":         "#ffffff",
}

# ---------------------------------------------------------------------------
# YAML / stim loading
# ---------------------------------------------------------------------------

def load_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def list_runs(cfg: dict) -> list:
    return list((cfg.get("model_dict") or {}).keys())

def load_stims(cfg: dict, run_key: str) -> list:
    md = cfg.get("model_dict") or {}
    run_keys = list(md.keys()) if run_key == ALL_RUNS_KEY else [run_key]
    stims = []
    for rk in run_keys:
        for stim_name, attrs in (md.get(rk) or {}).items():
            entry = {"name": stim_name, "run": rk}
            for k, v in (attrs or {}).items():
                entry[k] = v
            entry.setdefault("color", "#cccccc")
            stims.append(entry)
    return stims

def discover_attrs(stims: list) -> list:
    seen = ["stim"]
    for s in stims:
        for k in s.keys():
            if k in HIDDEN_ATTRS or k in ("run", "name"):
                continue
            if k not in seen:
                seen.append(k)
    if len({s.get("run") for s in stims}) > 1:
        seen.append("run")
    return seen

def display_name(stim: dict, combined: bool) -> str:
    return f"{stim['run']}_{stim['name']}" if combined else stim["name"]

# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def _stim_val(s: dict, attr: str) -> str:
    return str(s.get("name", "")) if attr == "stim" else str(s.get(attr, ""))

def _group_key(stim: dict, group_by: Sequence[str], sep: str = "_") -> str:
    if not group_by:
        return stim["name"]
    return sep.join(_stim_val(stim, k) for k in group_by)

def axis_codes(stims, group_by, combined, sep="_"):
    if not group_by:
        return [display_name(s, combined) for s in stims], list(range(len(stims)))
    seen, mapping = {}, []
    for s in stims:
        k = _group_key(s, group_by, sep)
        if k not in seen:
            seen[k] = len(seen)
        mapping.append(seen[k])
    return list(seen.keys()), mapping

# ---------------------------------------------------------------------------
# Matrix math
# ---------------------------------------------------------------------------

def fresh_matrix(n):
    m = np.full((n, n), np.nan, dtype=np.float64)
    np.fill_diagonal(m, 0.0)
    return m

def enforce_invariants(m):
    m = np.array(m, dtype=np.float64, copy=True)
    iu = np.triu_indices_from(m, k=1)
    m[(iu[1], iu[0])] = m[iu]
    np.fill_diagonal(m, 0.0)
    return m

def set_pair(m, i, j, value):
    if i == j: return
    v = np.nan if (value is None or (isinstance(value, float) and math.isnan(value))) else float(value)
    m[i, j] = m[j, i] = v

def broadcast_grouped_edit(mf, mapping, gi, gj, value):
    rows = [i for i, g in enumerate(mapping) if g == gi]
    cols = [j for j, g in enumerate(mapping) if g == gj]
    for i in rows:
        for j in cols:
            set_pair(mf, i, j, value)

def grouped_view(mf, mapping, n_groups):
    g = np.full((n_groups, n_groups), np.nan)
    mixed = np.zeros((n_groups, n_groups), dtype=bool)
    mapping = np.asarray(mapping)
    for gi in range(n_groups):
        rows = np.where(mapping == gi)[0]
        for gj in range(n_groups):
            cols = np.where(mapping == gj)[0]
            if not len(rows) or not len(cols): continue
            flat = mf[np.ix_(rows, cols)].flatten()
            uniq = []
            for v in flat:
                if math.isnan(v):
                    if not any(isinstance(u, float) and math.isnan(u) for u in uniq):
                        uniq.append(float("nan"))
                elif v not in [u for u in uniq if not (isinstance(u, float) and math.isnan(u))]:
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

def apply_bulk_rule(mf, stims, lhs_attr, lhs_val, rhs_attr, rhs_val, value, only_nan=False):
    rows = [i for i, s in enumerate(stims) if lhs_val == "*" or _stim_val(s, lhs_attr) == lhs_val]
    cols = [j for j, s in enumerate(stims) if rhs_val == "*" or _stim_val(s, rhs_attr) == rhs_val]
    for i in rows:
        for j in cols:
            if i == j: continue
            if only_nan and not math.isnan(mf[i, j]): continue
            set_pair(mf, i, j, value)
    return mf

# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def matrix_to_json(m):
    return [[None if math.isnan(v) else float(v) for v in row] for row in m]

def matrix_from_json(data):
    return np.array([[np.nan if v is None else float(v) for v in row] for row in data])

def parse_value(text):
    if text is None: return np.nan
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return np.nan if (isinstance(text, float) and math.isnan(text)) else float(text)
    s = str(text).strip()
    if not s or s.lower() == "nan": return np.nan
    try: return float(s)
    except ValueError: return np.nan

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def scan_model_files(folder):
    try:
        return sorted(f for f in os.listdir(folder)
                      if f.lower().endswith(".csv") and not f.endswith("_style.json"))
    except Exception:
        return []

def load_model_into_matrix(csv_path, stim_labels, current_matrix_json):
    df = pd.read_csv(csv_path, index_col=0)
    mf = matrix_from_json(current_matrix_json)
    n_matched = 0
    for i, ri in enumerate(stim_labels):
        if ri not in df.index:
            continue
        for j, ci in enumerate(stim_labels):
            if ci not in df.columns:
                continue
            v = df.loc[ri, ci]
            mf[i, j] = np.nan if pd.isna(v) else float(v)
            n_matched += 1
    return mf, n_matched

def to_export_dataframe(matrix, labels):
    return pd.DataFrame(matrix, index=labels, columns=labels)

def dataframe_to_csv_string(df):
    def fmt(v):
        if isinstance(v, float):
            if math.isnan(v): return "NaN"
            if v.is_integer(): return str(int(v))
            return repr(v)
        return str(v)
    lines = ["," + ",".join(str(c) for c in df.columns)]
    for idx, row in df.iterrows():
        lines.append(str(idx) + "," + ",".join(fmt(v) for v in row.tolist()))
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
# Cell-based heatmap renderer (soft-edge rectangles)
# ---------------------------------------------------------------------------

def _rrect(x0, y0, x1, y1, r):
    """SVG path for a rounded rectangle."""
    r = max(0.0, min(float(r), (x1-x0)/2 - 0.1, (y1-y0)/2 - 0.1))
    if r == 0:
        return f"M {x0},{y0} L {x1},{y0} L {x1},{y1} L {x0},{y1} Z"
    return (f"M {x0+r},{y0} L {x1-r},{y0} Q {x1},{y0} {x1},{y0+r} "
            f"L {x1},{y1-r} Q {x1},{y1} {x1-r},{y1} "
            f"L {x0+r},{y1} Q {x0},{y1} {x0},{y1-r} "
            f"L {x0},{y0+r} Q {x0},{y0} {x0+r},{y0} Z")

def _v_to_color(v, colorscale, vmin, vmax, nan_color):
    if math.isnan(v): return nan_color
    t = float(np.clip((v - vmin) / max(vmax - vmin, 1e-9), 0, 1))
    return pc.sample_colorscale(colorscale, [t])[0]

def build_cell_heatmap(matrix, labels, style, mixed_mask=None, axis_colors=None):
    n = len(labels)
    if n == 0:
        return go.Figure()

    S  = style or DEFAULT_STYLE
    cs = max(int(S.get("cell_size",  40)), 6)
    gp = max(int(S.get("cell_gap",    4)), 0)
    r  = max(int(S.get("cell_radius", 6)), 0)
    colorscale     = S.get("colorscale",       "Viridis")
    cmin           = float(S.get("cbar_min",    0.0))
    cmax           = float(S.get("cbar_max",    1.0))
    show_cbar      = bool(S.get("show_colorbar", True))
    use_legend     = bool(S.get("use_legend",    False))
    legend_show_nan= bool(S.get("legend_show_nan", True))
    nan_color      = S.get("nan_color",  "#cccccc")
    diag_color     = S.get("diag_color", "#f0f0f0")
    mix_color      = S.get("mixed_color","#ff9999")
    show_vals      = bool(S.get("show_values",    True))
    vfs            = int(S.get("val_font_size",   9))
    vfc            = S.get("val_font_color","#ffffff")
    lfs            = int(S.get("label_font_size", 11))
    lfc            = S.get("label_font_color","#222222")
    show_x_labels  = bool(S.get("show_x_labels", True))
    show_y_labels  = bool(S.get("show_y_labels", True))
    x_ang          = float(S.get("x_label_angle", -45))
    y_ang          = float(S.get("y_label_angle",  0))
    bg             = S.get("bg_color",   "#ffffff")
    paper_bg       = S.get("paper_bg",   "#ffffff")

    if cmax <= cmin: cmax = cmin + 1.0

    # Batch color mapping for non-special cells
    flat_norm = []
    flat_idx  = []
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            is_special = (i == j or math.isnan(v) or
                          (mixed_mask is not None and mixed_mask[i, j]))
            if not is_special:
                flat_idx.append(i * n + j)
                flat_norm.append(float(np.clip((v - cmin) / (cmax - cmin), 0, 1)))
    if flat_norm:
        mapped = pc.sample_colorscale(colorscale, flat_norm)
    else:
        mapped = []
    color_map = {idx: c for idx, c in zip(flat_idx, mapped)}

    shapes, annotations = [], []
    hov_x, hov_y, hov_text, hov_cd = [], [], [], []

    for i in range(n):
        for j in range(n):
            x0 = j * cs + gp / 2;  x1 = (j+1)*cs - gp/2
            y0 = i * cs + gp / 2;  y1 = (i+1)*cs - gp/2
            v  = matrix[i, j]

            if i == j:
                fc, cell_txt = diag_color, "0"
            elif mixed_mask is not None and mixed_mask[i, j]:
                fc, cell_txt = mix_color, "mix"
            elif math.isnan(v):
                fc, cell_txt = nan_color, ""
            else:
                fc = color_map.get(i*n+j, nan_color)
                cell_txt = str(int(v)) if float(v).is_integer() else f"{v:.2g}"

            shapes.append(dict(type="path", path=_rrect(x0, y0, x1, y1, r),
                               fillcolor=fc, line_width=0))

            cx, cy = (x0+x1)/2, (y0+y1)/2
            hov_x.append(cx); hov_y.append(cy)
            hov_text.append(f"<b>{labels[i]}</b> × <b>{labels[j]}</b><br>"
                            f"{'NaN' if not cell_txt else cell_txt}")
            hov_cd.append([labels[i], labels[j]])

            if show_vals and cell_txt:
                annotations.append(dict(x=cx, y=cy, text=cell_txt, showarrow=False,
                                        font=dict(size=vfs, color=vfc),
                                        xanchor="center", yanchor="middle",
                                        xref="x", yref="y"))

    # Axis label annotations
    max_chars = max((len(l) for l in labels), default=1)
    x_rad = abs(math.radians(x_ang))
    y_rad = abs(math.radians(y_ang))
    x_label_depth = (max(cs, max_chars * lfs * 0.65 * math.sin(x_rad) + lfs) + gp*2) if show_x_labels else gp*2
    y_label_width = (max(cs, max_chars * lfs * 0.65 * math.cos(y_rad) + lfs) + gp*2) if show_y_labels else gp*2

    if show_x_labels:
        for j, lab in enumerate(labels):
            annotations.append(dict(
                x=j*cs + cs/2, y=n*cs + gp*2,
                text=lab, showarrow=False,
                font=dict(size=lfs, color=lfc),
                textangle=x_ang,
                xanchor="right" if x_ang != 0 else "center",
                yanchor="top", xref="x", yref="y"))
    if show_y_labels:
        for i, lab in enumerate(labels):
            annotations.append(dict(
                x=-gp*2, y=i*cs + cs/2,
                text=lab, showarrow=False,
                font=dict(size=lfs, color=lfc),
                textangle=y_ang,
                xanchor="right", yanchor="middle",
                xref="x", yref="y"))

    fig = go.Figure()

    # Invisible scatter for click/hover
    fig.add_trace(go.Scatter(
        x=hov_x, y=hov_y,
        mode="markers",
        marker=dict(size=max(cs-gp, 4), opacity=0, symbol="square"),
        text=hov_text, hoverinfo="text",
        customdata=hov_cd,
        showlegend=False,
    ))

    # Colorbar / legend
    if use_legend:
        # collect unique non-diagonal values
        seen_nan = False
        unique_vals = []
        for i in range(n):
            for j in range(n):
                if i == j: continue
                v = matrix[i, j]
                if math.isnan(v):
                    if not seen_nan:
                        seen_nan = True
                elif v not in unique_vals:
                    unique_vals.append(v)
        unique_vals.sort()
        if seen_nan and legend_show_nan:
            unique_vals.append(float("nan"))
        for v in unique_vals:
            if math.isnan(v):
                color, label = nan_color, "NaN"
            else:
                color = _v_to_color(v, colorscale, cmin, cmax, nan_color)
                label = str(int(v)) if float(v).is_integer() else f"{v:.3g}"
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color=color, size=14, symbol="square"),
                name=label, showlegend=True, hoverinfo="none"))
    elif show_cbar:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(colorscale=colorscale, cmin=cmin, cmax=cmax,
                        color=[cmin], showscale=True,
                        colorbar=dict(thickness=14, len=0.75, x=1.02,
                                      tickfont=dict(size=10))),
            hoverinfo="none", showlegend=False))

    total = n * cs
    fig.update_layout(
        shapes=shapes, annotations=annotations,
        xaxis=dict(range=[-y_label_width, total + cs*0.3],
                   showgrid=False, showticklabels=False, zeroline=False,
                   fixedrange=True),
        yaxis=dict(range=[total + x_label_depth, -cs*0.3],
                   showgrid=False, showticklabels=False, zeroline=False,
                   fixedrange=True, scaleanchor="x", scaleratio=1),
        plot_bgcolor=bg, paper_bgcolor=paper_bg,
        margin=dict(l=10, r=60, t=20, b=10),
        height=max(350, total + int(x_label_depth) + 60),
        hovermode="closest",
        dragmode=False,
        showlegend=use_legend,
        legend=dict(x=1.04, y=0.5, xanchor="left", yanchor="middle",
                    bgcolor="rgba(255,255,255,0.8)", bordercolor="#ccc", borderwidth=1,
                    font=dict(size=11)) if use_legend else {},
    )
    return fig

def style_to_summary(style, group_by, sep):
    """Return a human-readable JSON-serialisable dict of the options used."""
    return {
        "figure_style": {k: style.get(k, DEFAULT_STYLE.get(k)) for k in DEFAULT_STYLE},
        "group_by": group_by or [],
        "separator": "_" if sep is None else sep,
    }

def representative_color(stims, mapping, n_groups):
    out = ["#cccccc"] * n_groups
    for i, g in enumerate(mapping):
        if out[g] == "#cccccc":
            out[g] = stims[i].get("color", "#cccccc")
    return out

# ---------------------------------------------------------------------------
# Current-view helper
# ---------------------------------------------------------------------------

def _current_view(stims, mf, view_mode, group_by, combined, sep="_"):
    if view_mode == "full" or not group_by:
        labels  = [display_name(s, combined) for s in stims]
        mapping = list(range(len(stims)))
        m       = enforce_invariants(mf)
        return labels, mapping, m, np.zeros_like(m, dtype=bool)
    labels, mapping = axis_codes(stims, group_by, combined, sep)
    m, mixed = grouped_view(mf, mapping, len(labels))
    return labels, mapping, m, mixed

# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "RSA Model Builder"

def attr_value_options(stims, attr):
    if not stims or not attr: return [{"label": "(any)", "value": "*"}]
    vals = []
    for s in stims:
        v = _stim_val(s, attr)
        if v not in vals: vals.append(v)
    return [{"label": "(any)", "value": "*"}] + [{"label": v, "value": v} for v in vals]

# ── shared style constants ──────────────────────────────────────────────────
CBOX  = {"border": "1px solid #ddd", "borderRadius": "6px",
         "padding": "10px", "marginBottom": "10px", "background": "#fcfcfc"}
_B    = {"padding": "2px 8px", "cursor": "pointer", "fontSize": "12px"}
BTN_M = {**_B, "width": "26px"}
BTN_R = {**_B, "width": "26px", "color": "#a33"}

# ── control builder helpers ─────────────────────────────────────────────────
def _num(id_, val, mn=None, mx=None, step=1, w="70px"):
    kw = {"type": "number", "value": val, "step": step,
          "style": {"width": w, "fontSize": "12px"}}
    if mn is not None: kw["min"] = mn
    if mx is not None: kw["max"] = mx
    return dcc.Input(id=id_, **kw)

def _color(id_, val):
    return dcc.Input(id=id_, type="color", value=val,
                     style={"width": "44px", "height": "28px", "padding": "1px",
                            "border": "1px solid #ccc", "borderRadius": "3px"})

def _lbl(text): return html.Span(text, style={"fontSize": "11px", "color": "#555",
                                               "marginBottom": "2px", "display": "block"})

def _ctrl(label, control):
    return html.Div([_lbl(label), control],
                    style={"marginRight": "14px", "marginBottom": "6px"})

# ── style panel ─────────────────────────────────────────────────────────────
def _style_panel():
    DS = DEFAULT_STYLE
    row = lambda children: html.Div(children,
        style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-end",
               "marginBottom": "4px"})
    return html.Div([
        html.Div("⚙ Figure Style",
                 style={"fontWeight": "bold", "fontSize": "14px",
                        "marginBottom": "8px"}),
        html.Div([
            # ── Cells ──────────────────────────────────────────────────────
            html.Div("Cells", style={"fontWeight":"bold","fontSize":"12px",
                                     "color":"#444","marginBottom":"3px"}),
            row([
                _ctrl("Cell size (px)", _num("ctrl-cell-size", DS["cell_size"], 6, 160)),
                _ctrl("Gap (px)",       _num("ctrl-cell-gap",  DS["cell_gap"],  0, 40)),
                _ctrl("Radius (px)",    _num("ctrl-cell-radius",DS["cell_radius"],0, 60)),
            ]),
            # ── Colorscale ─────────────────────────────────────────────────
            html.Div("Colorscale", style={"fontWeight":"bold","fontSize":"12px",
                                          "color":"#444","marginBottom":"3px",
                                          "marginTop":"6px"}),
            row([
                _ctrl("Scale", html.Div(dcc.Dropdown(
                    id="ctrl-colorscale",
                    options=[{"label": c, "value": c} for c in COLORSCALE_OPTIONS],
                    value=DS["colorscale"], clearable=False,
                    style={"width": "160px", "fontSize": "12px"}))),
                _ctrl("Min", _num("ctrl-cbar-min", DS["cbar_min"], step=0.05, w="70px")),
                _ctrl("Max", _num("ctrl-cbar-max", DS["cbar_max"], step=0.05, w="70px")),
                html.Div([
                    _lbl(" "),
                    dcc.Checklist(id="ctrl-show-cbar",
                                  options=[{"label": " Colorbar", "value": "y"}],
                                  value=["y"] if DS["show_colorbar"] else [],
                                  style={"fontSize": "12px"})
                ], style={"marginRight": "14px", "marginBottom": "6px"}),
                html.Div([
                    _lbl(" "),
                    dcc.Checklist(id="ctrl-use-legend",
                                  options=[{"label": " Use legend", "value": "y"}],
                                  value=["y"] if DS["use_legend"] else [],
                                  style={"fontSize": "12px"})
                ], style={"marginRight": "14px", "marginBottom": "6px"}),
                html.Div([
                    _lbl(" "),
                    dcc.Checklist(id="ctrl-legend-show-nan",
                                  options=[{"label": " Legend: show NaN", "value": "y"}],
                                  value=["y"] if DS["legend_show_nan"] else [],
                                  style={"fontSize": "12px"})
                ], style={"marginRight": "14px", "marginBottom": "6px"}),
                _ctrl("NaN color",   _color("ctrl-nan-color",  DS["nan_color"])),
                _ctrl("Diag color",  _color("ctrl-diag-color", DS["diag_color"])),
                _ctrl("Mixed color", _color("ctrl-mixed-color",DS["mixed_color"])),
            ]),
            # ── Cell values ────────────────────────────────────────────────
            html.Div("Cell values", style={"fontWeight":"bold","fontSize":"12px",
                                           "color":"#444","marginBottom":"3px",
                                           "marginTop":"6px"}),
            row([
                html.Div([
                    _lbl(" "),
                    dcc.Checklist(id="ctrl-show-values",
                                  options=[{"label": " Show values", "value": "y"}],
                                  value=["y"] if DS["show_values"] else [],
                                  style={"fontSize": "12px"})
                ], style={"marginRight": "14px", "marginBottom": "6px"}),
                _ctrl("Value size (pt)", _num("ctrl-val-font-size",  DS["val_font_size"],  4, 40)),
                _ctrl("Value color",     _color("ctrl-val-font-color", DS["val_font_color"])),
            ]),
            # ── Axis labels ────────────────────────────────────────────────
            html.Div("Axis labels", style={"fontWeight":"bold","fontSize":"12px",
                                           "color":"#444","marginBottom":"3px",
                                           "marginTop":"6px"}),
            row([
                _ctrl("Label size (pt)", _num("ctrl-label-font-size",  DS["label_font_size"], 4, 40)),
                _ctrl("Label color",     _color("ctrl-label-font-color",DS["label_font_color"])),
                html.Div([
                    _lbl(" "),
                    dcc.Checklist(id="ctrl-show-x-labels",
                                  options=[{"label": " X labels", "value": "y"}],
                                  value=["y"] if DS["show_x_labels"] else [],
                                  style={"fontSize": "12px"})
                ], style={"marginRight": "14px", "marginBottom": "6px"}),
                html.Div([
                    _lbl(" "),
                    dcc.Checklist(id="ctrl-show-y-labels",
                                  options=[{"label": " Y labels", "value": "y"}],
                                  value=["y"] if DS["show_y_labels"] else [],
                                  style={"fontSize": "12px"})
                ], style={"marginRight": "14px", "marginBottom": "6px"}),
                _ctrl("X angle (°)",     _num("ctrl-x-angle", DS["x_label_angle"], -90, 90, 5)),
                _ctrl("Y angle (°)",     _num("ctrl-y-angle", DS["y_label_angle"], -90, 90, 5)),
            ]),
            # ── Background ─────────────────────────────────────────────────
            html.Div("Background", style={"fontWeight":"bold","fontSize":"12px",
                                          "color":"#444","marginBottom":"3px",
                                          "marginTop":"6px"}),
            row([
                _ctrl("Plot bg",  _color("ctrl-bg-color",  DS["bg_color"])),
                _ctrl("Paper bg", _color("ctrl-paper-bg",  DS["paper_bg"])),
            ]),
            # ── Presets ────────────────────────────────────────────────────
            html.Hr(style={"margin": "10px 0"}),
            html.Div("Presets", style={"fontWeight":"bold","fontSize":"12px",
                                       "color":"#444","marginBottom":"6px"}),
            html.Div([
                dcc.Input(id="preset-name", type="text", placeholder="Preset name…",
                          style={"width": "140px", "fontSize": "12px",
                                 "marginRight": "6px"}),
                html.Button("Save", id="btn-save-preset", n_clicks=0,
                            style={**_B, "marginRight": "12px"}),
                dcc.Dropdown(id="dd-load-preset", options=[], placeholder="Load preset…",
                             clearable=True,
                             style={"width": "180px", "fontSize": "12px",
                                    "display": "inline-block", "marginRight": "6px",
                                    "verticalAlign": "middle"}),
                html.Button("Delete", id="btn-delete-preset", n_clicks=0,
                            style={**_B, "marginRight": "8px"}),
                html.Button("Restore last session", id="btn-restore-session", n_clicks=0,
                            style={**_B}),
                html.Span(id="preset-status",
                          style={"marginLeft": "10px", "fontSize": "11px", "color": "#393"}),
            ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "center",
                      "gap": "4px"}),
        ], style={"padding": "4px 2px 4px 2px"}),
    ], style={**CBOX})


app.layout = html.Div([
    dcc.Store(id="store-cfg"),
    dcc.Store(id="store-stims"),
    dcc.Store(id="store-matrix"),
    dcc.Store(id="store-meta"),
    dcc.Store(id="store-groupby",  data=[]),
    dcc.Store(id="store-sep",      data="_"),
    dcc.Store(id="store-style",    storage_type="local", data=None),
    dcc.Store(id="store-presets",  storage_type="local", data={}),
    dcc.Store(id="store-undo-stack", data=[]),
    dcc.Store(id="store-redo-stack", data=[]),
    dcc.Store(id="store-kbd",        data=None),
    dcc.Store(id="store-last-model", storage_type="local", data=None),
    dcc.Download(id="download-csv"),

    html.H2("RSA Model Builder", style={"marginBottom": "4px"}),
    html.Div("Build dissimilarity matrices from experiment YAML stim definitions.",
             style={"color": "#666", "marginBottom": "10px"}),

    # ── Top bar ──────────────────────────────────────────────────────────────
    html.Div([
        html.Div([html.Label("YAML config path"),
                  dcc.Input(id="input-yaml", type="text", value=DEFAULT_YAML,
                            style={"width": "100%"})],
                 style={"flex": "3", "marginRight": "10px"}),
        html.Div([html.Label(" "),
                  html.Button("Load YAML", id="btn-load", n_clicks=0,
                              style={"width": "100%", "height": "32px"})],
                 style={"flex": "1", "marginRight": "10px"}),
        html.Div([html.Label("Run"),
                  dcc.Dropdown(id="dd-run", options=[], value=None, clearable=False)],
                 style={"flex": "1", "marginRight": "10px"}),
        html.Div([html.Label("View"),
                  dcc.RadioItems(id="radio-view",
                                 options=[{"label": "Full",    "value": "full"},
                                          {"label": "Grouped", "value": "grouped"}],
                                 value="grouped", inline=True)],
                 style={"flex": "1"}),
    ], style={"display": "flex", "alignItems": "flex-end", **CBOX}),

    # ── Model loader panel ───────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("Saved models", style={"fontWeight": "bold", "fontSize": "15px",
                                             "marginRight": "16px"}),
            dcc.Dropdown(id="dd-model-file", options=[], placeholder="Select a .csv model…",
                         clearable=True,
                         style={"width": "380px", "display": "inline-block",
                                "verticalAlign": "middle", "fontSize": "13px"}),
            html.Button("⟳", id="btn-scan-models", n_clicks=0,
                        style={**_B, "marginLeft": "6px"},
                        title="Scan folder for CSV models"),
            html.Button("Load", id="btn-load-model", n_clicks=0,
                        style={**_B, "marginLeft": "6px"}),
            html.Button("Reset matrix", id="btn-reset-model", n_clicks=0,
                        style={**_B, "marginLeft": "20px", "color": "#a33"},
                        title="Clear matrix to NaN (diagonal = 0)"),
            html.Span(id="model-load-status",
                      style={"marginLeft": "12px", "fontSize": "11px", "color": "#555"}),
            html.Span(" │ ", style={"color": "#ccc", "marginLeft": "12px"}),
            html.Span("Ctrl+Z undo · Ctrl+Shift+Z redo",
                      style={"fontSize": "11px", "color": "#aaa", "marginLeft": "6px",
                             "fontStyle": "italic"}),
        ], style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "4px"}),
    ], style={**CBOX}),

    # ── Group by panel ───────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span("Group by", style={"fontWeight": "bold", "fontSize": "15px",
                                          "marginRight": "20px"}),
            html.Span("Separator:", style={"marginRight": "6px", "fontSize": "13px"}),
            dcc.Dropdown(id="dd-sep", options=SEPARATOR_OPTIONS, value="_",
                         clearable=False,
                         style={"width": "210px", "display": "inline-block",
                                "verticalAlign": "middle"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
        html.Div(id="div-groupby-list",
                 style={"display": "flex", "flexWrap": "wrap", "gap": "4px",
                        "minHeight": "34px", "alignItems": "center"}),
        html.Div([
            dcc.Dropdown(id="dd-gb-add", options=[], placeholder="Add field…",
                         clearable=True, style={"width": "200px"}),
            html.Button("Add", id="btn-gb-add", n_clicks=0,
                        style={**_B, "marginLeft": "6px", "height": "32px",
                               "padding": "2px 12px"}),
        ], style={"display": "flex", "alignItems": "center", "marginTop": "8px"}),
    ], style={**CBOX}),

    # ── Figure style panel ───────────────────────────────────────────────────
    _style_panel(),

    html.Div(id="status", style={"color": "#a33", "marginBottom": "8px"}),

    # ── Main ─────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            dcc.Graph(id="heatmap", config={"displayModeBar": True,
                                             "toImageButtonOptions": {"format": "png",
                                                                       "scale": 2}}),
            html.Div([
                html.Label("Edit cell — value (blank/'NaN' clears):"),
                html.Div([
                    dcc.Input(id="cell-row", type="text", placeholder="row",
                              disabled=True, style={"width": "120px", "marginRight": "6px"}),
                    dcc.Input(id="cell-col", type="text", placeholder="col",
                              disabled=True, style={"width": "120px", "marginRight": "6px"}),
                    dcc.Input(id="cell-value", type="text", placeholder="value",
                              style={"width": "80px", "marginRight": "6px"}),
                    html.Button("Set", id="btn-set-cell", n_clicks=0),
                ], style={"display": "flex"}),
                html.Div("0 = identical · 0.5 = somewhat different · 1 = completely different · NaN = excluded",
                         style={"fontSize": "11px", "color": "#888", "marginTop": "4px",
                                "fontStyle": "italic"}),
            ], style={"marginTop": "6px"}),
        ], style={"flex": "3", "marginRight": "12px"}),

        html.Div([
            html.H4("Bulk rules", style={"marginTop": 0}),
            html.Div([
                html.Div([html.Label("Row attr"),
                          dcc.Dropdown(id="bulk-lhs-attr", options=[], value=None, clearable=False)],
                         style={"flex": "1", "marginRight": "6px"}),
                html.Div([html.Label("Row value"),
                          dcc.Dropdown(id="bulk-lhs-val", options=[], value="*", clearable=False)],
                         style={"flex": "1"}),
            ], style={"display": "flex", "marginBottom": "6px"}),
            html.Div([
                html.Div([html.Label("Col attr"),
                          dcc.Dropdown(id="bulk-rhs-attr", options=[], value=None, clearable=False)],
                         style={"flex": "1", "marginRight": "6px"}),
                html.Div([html.Label("Col value"),
                          dcc.Dropdown(id="bulk-rhs-val", options=[], value="*", clearable=False)],
                         style={"flex": "1"}),
            ], style={"display": "flex", "marginBottom": "6px"}),
            html.Div([
                html.Label("Value"),
                html.Div([
                    dcc.Input(id="bulk-value", type="text", value="0",
                              style={"flex": "1", "marginRight": "4px"}),
                    html.Button("0",   id="btn-quick-0",   n_clicks=0,
                                style={"width": "30px", "marginRight": "2px"}),
                    html.Button("1",   id="btn-quick-1",   n_clicks=0,
                                style={"width": "30px", "marginRight": "2px"}),
                    html.Button("NaN", id="btn-quick-nan", n_clicks=0,
                                style={"width": "44px"}),
                ], style={"display": "flex"}),
                html.Div("Tip: blank or 'NaN' clears.",
                         style={"fontSize": "11px", "color": "#888"}),
            ], style={"marginBottom": "6px"}),
            dcc.Checklist(id="bulk-only-nan",
                          options=[{"label": " only fill NaN cells", "value": "only_nan"}],
                          value=[]),
            html.Button("Apply rule", id="btn-bulk-apply", n_clicks=0,
                        style={"width": "100%", "marginTop": "6px"}),
            html.Button("Fill all NaN with above value", id="btn-fill-nan", n_clicks=0,
                        style={"width": "100%", "marginTop": "4px"}),
            html.Button("Set same-group pairs → 0", id="btn-same-to-0", n_clicks=0,
                        style={"width": "100%", "marginTop": "4px"},
                        title="Sets all off-diagonal pairs where both stimuli share the same group-by key to 0"),
            html.Hr(),
            html.Button("Reset matrix (NaN, diag=0)", id="btn-reset", n_clicks=0,
                        style={"width": "100%", "marginBottom": "6px"}),
            html.Button("Mirror upper → lower", id="btn-mirror", n_clicks=0,
                        style={"width": "100%", "marginBottom": "6px"}),
            html.Hr(),
            html.H4("Export"),
            html.Label("Filename"),
            dcc.Input(id="export-filename", type="text", value="my-model.csv",
                      style={"width": "100%", "marginBottom": "6px"}),
            html.Label("Export folder (saved on server)"),
            dcc.Input(id="export-folder", type="text", value=DEFAULT_EXPORT_DIR,
                      style={"width": "100%", "marginBottom": "6px"}),
            html.Div("CSV + companion _style.json will be saved.",
                     style={"fontSize": "11px", "color": "#888", "marginBottom": "6px"}),
            html.Button("Export CSV + style", id="btn-export", n_clicks=0,
                        style={"width": "100%"}),
            html.Div(id="export-status", style={"color": "#393", "marginTop": "6px"}),
        ], style={"flex": "1", **CBOX}),
    ], style={"display": "flex"}),

    html.Div([
        html.H4("Matrix table (editable)"),
        dash_table.DataTable(
            id="table", editable=True, columns=[], data=[],
            style_table={"overflowX": "auto", "maxHeight": "380px", "overflowY": "auto"},
            style_cell={"textAlign": "center", "minWidth": "60px", "maxWidth": "120px",
                        "padding": "4px", "fontFamily": "monospace"},
            style_header={"backgroundColor": "#f0f0f0", "fontWeight": "bold"},
            fixed_rows={"headers": True},
            fixed_columns={"headers": True, "data": 1},
        ),
    ], style={"marginTop": "12px"}),
], style={"fontFamily": "Segoe UI, Arial, sans-serif", "margin": "12px"})


# ===========================================================================
# Clientside — keyboard shortcuts
# ===========================================================================

app.clientside_callback(
    """
    function(_ignore) {
        if (!window._rsa_kbd_bound) {
            window._rsa_kbd_bound = true;
            document.addEventListener('keydown', function(e) {
                var isZ = e.key === 'z' || e.key === 'Z';
                if (!isZ || !(e.ctrlKey || e.metaKey)) return;
                e.preventDefault();
                var action = e.shiftKey ? 'redo' : 'undo';
                window.dash_clientside.set_props('store-kbd', {data: action + ':' + Date.now()});
            });
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("store-kbd", "data"),
    Input("heatmap", "id"),
)

# ===========================================================================
# Callbacks
# ===========================================================================

# ── YAML / run ────────────────────────────────────────────────────────────
@app.callback(
    Output("store-cfg",    "data"),
    Output("dd-run",       "options"),
    Output("dd-run",       "value"),
    Output("status",       "children"),
    Input("btn-load",      "n_clicks"),
    State("input-yaml",    "value"),
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
    opts = [{"label": r, "value": r} for r in runs]
    opts.append({"label": "All runs (combined)", "value": ALL_RUNS_KEY})
    return cfg, opts, runs[0], f"Loaded {path} ({len(runs)} runs)."


@app.callback(
    Output("store-stims",  "data"),
    Output("store-matrix", "data"),
    Output("store-meta",   "data"),
    Input("store-cfg",     "data"),
    Input("dd-run",        "value"),
    State("input-yaml",    "value"),
    prevent_initial_call=True,
)
def build_stims(cfg, run_key, yaml_path):
    if not cfg or not run_key:
        return no_update, no_update, no_update
    stims = load_stims(cfg, run_key)
    if not stims:
        return [], [[]], {"combined": False, "yaml_path": yaml_path}
    return (stims, matrix_to_json(fresh_matrix(len(stims))),
            {"combined": run_key == ALL_RUNS_KEY, "yaml_path": yaml_path})


# ── Undo / Redo ───────────────────────────────────────────────────────────
@app.callback(
    Output("store-matrix",     "data", allow_duplicate=True),
    Output("store-undo-stack", "data", allow_duplicate=True),
    Output("store-redo-stack", "data", allow_duplicate=True),
    Input("store-kbd",         "data"),
    State("store-matrix",      "data"),
    State("store-undo-stack",  "data"),
    State("store-redo-stack",  "data"),
    prevent_initial_call=True,
)
def undo_redo_cb(kbd, matrix_data, undo_stack, redo_stack):
    if not kbd:
        raise PreventUpdate
    action     = kbd.split(":")[0]
    undo_stack = list(undo_stack or [])
    redo_stack = list(redo_stack or [])
    if action == "undo":
        if not undo_stack: raise PreventUpdate
        redo_stack.append(matrix_data)
        return undo_stack.pop(), undo_stack, redo_stack
    if action == "redo":
        if not redo_stack: raise PreventUpdate
        undo_stack.append(matrix_data)
        return redo_stack.pop(), undo_stack, redo_stack
    raise PreventUpdate


# ── Reset undo stacks when a new run is loaded ────────────────────────────
@app.callback(
    Output("store-undo-stack", "data"),
    Output("store-redo-stack", "data"),
    Input("store-stims",       "data"),
    prevent_initial_call=True,
)
def reset_undo_on_new_stims(_stims):
    return [], []


# ── Model scan ────────────────────────────────────────────────────────────
@app.callback(
    Output("dd-model-file", "options"),
    Output("dd-model-file", "value"),
    Input("btn-scan-models",   "n_clicks"),
    State("store-last-model",  "data"),
    prevent_initial_call=False,
)
def scan_models_cb(n, last_model):
    files = scan_model_files(DEFAULT_EXPORT_DIR)
    opts  = [{"label": f, "value": os.path.join(DEFAULT_EXPORT_DIR, f)} for f in files]
    vals  = [o["value"] for o in opts]
    default = last_model if last_model in vals else (vals[0] if vals else None)
    return opts, default


# ── Model load ────────────────────────────────────────────────────────────
@app.callback(
    Output("store-matrix",     "data", allow_duplicate=True),
    Output("store-undo-stack", "data", allow_duplicate=True),
    Output("store-redo-stack", "data", allow_duplicate=True),
    Output("store-last-model", "data"),
    Output("model-load-status","children"),
    Input("btn-load-model",    "n_clicks"),
    State("dd-model-file",     "value"),
    State("store-stims",       "data"),
    State("store-meta",        "data"),
    State("store-matrix",      "data"),
    State("store-undo-stack",  "data"),
    prevent_initial_call=True,
)
def load_model_cb(n, fpath, stims, meta, matrix_data, undo_stack):
    if not fpath or not stims or not matrix_data:
        return no_update, no_update, no_update, no_update, "No model selected or no stims loaded."
    combined    = bool(meta and meta.get("combined"))
    stim_labels = [display_name(s, combined) for s in stims]
    try:
        mf, n_matched = load_model_into_matrix(fpath, stim_labels, matrix_data)
    except Exception as e:
        return no_update, no_update, no_update, no_update, f"Error: {e}"
    stack = list(undo_stack or [])
    stack.append(matrix_data)
    if len(stack) > MAX_UNDO: stack = stack[-MAX_UNDO:]
    return (matrix_to_json(mf), stack, [], fpath,
            f"Loaded {os.path.basename(fpath)} ({n_matched} cells matched).")


# ── Group-by ──────────────────────────────────────────────────────────────
@app.callback(
    Output("store-groupby", "data"),
    Input("store-stims",    "data"),
    State("store-groupby",  "data"),
    prevent_initial_call=True,
)
def init_groupby(stims, cur):
    if not stims: return []
    attrs = discover_attrs(stims)
    valid = [k for k in (cur or []) if k in attrs]
    if not valid:
        valid = [k for k in ("specie_shown", "label") if k in attrs] or attrs[:2]
    return valid


@app.callback(
    Output("div-groupby-list", "children"),
    Output("dd-gb-add",        "options"),
    Input("store-groupby",     "data"),
    Input("store-stims",       "data"),
)
def render_groupby_list(gb, stims):
    gb = gb or []
    if not stims:
        return [], []
    attrs    = discover_attrs(stims)
    add_opts = [{"label": a, "value": a} for a in attrs if a not in gb]
    if not gb:
        return [html.Span("(no grouping — full matrix view)",
                          style={"color": "#999", "fontStyle": "italic"})], add_opts
    chips = []
    for i, attr in enumerate(gb):
        chips.append(html.Div([
            html.Button("↑", id={"type": "gb-up",   "index": i}, n_clicks=0,
                        disabled=(i == 0),           style=BTN_M),
            html.Button("↓", id={"type": "gb-down", "index": i}, n_clicks=0,
                        disabled=(i == len(gb)-1),   style=BTN_M),
            html.Span(attr, style={"padding": "0 6px", "fontFamily": "monospace",
                                   "fontSize": "13px"}),
            html.Button("×", id={"type": "gb-rm",   "index": i}, n_clicks=0,
                        style=BTN_R),
        ], style={"display": "inline-flex", "alignItems": "center",
                  "border": "1px solid #9ab", "borderRadius": "16px",
                  "padding": "2px 6px", "background": "#eef4ff",
                  "marginRight": "4px"}))
    return chips, add_opts


@app.callback(
    Output("store-groupby", "data",  allow_duplicate=True),
    Output("dd-gb-add",     "value"),
    Input({"type": "gb-up",   "index": ALL}, "n_clicks"),
    Input({"type": "gb-down", "index": ALL}, "n_clicks"),
    Input({"type": "gb-rm",   "index": ALL}, "n_clicks"),
    Input("btn-gb-add",     "n_clicks"),
    State("store-groupby",  "data"),
    State("dd-gb-add",      "value"),
    prevent_initial_call=True,
)
def mutate_groupby(up, down, rm, _add, gb, add_val):
    trigger = ctx.triggered_id
    if trigger is None: return no_update, no_update
    gb = list(gb or [])
    if isinstance(trigger, dict):
        action, idx = trigger["type"], trigger["index"]
        if action == "gb-up"   and idx > 0:            gb[idx-1], gb[idx] = gb[idx], gb[idx-1]
        elif action == "gb-down" and idx < len(gb)-1:  gb[idx], gb[idx+1] = gb[idx+1], gb[idx]
        elif action == "gb-rm":                        gb.pop(idx)
        return gb, no_update
    if trigger == "btn-gb-add":
        if add_val and add_val not in gb: gb.append(add_val)
        return gb, None
    return no_update, no_update


@app.callback(Output("store-sep", "data"), Input("dd-sep", "value"))
def update_sep(val): return val if val is not None else "_"


# ── Bulk attr value dropdowns ─────────────────────────────────────────────
@app.callback(
    Output("bulk-lhs-attr", "options"), Output("bulk-lhs-attr", "value"),
    Output("bulk-rhs-attr", "options"), Output("bulk-rhs-attr", "value"),
    Input("store-stims", "data"),
    State("bulk-lhs-attr", "value"), State("bulk-rhs-attr", "value"),
)
def populate_bulk_attrs(stims, lhs, rhs):
    if not stims: return [], None, [], None
    attrs = discover_attrs(stims)
    opts  = [{"label": k, "value": k} for k in attrs]
    d     = "label" if "label" in attrs else attrs[0]
    return opts, (lhs if lhs in attrs else d), opts, (rhs if rhs in attrs else d)


@app.callback(
    Output("bulk-lhs-val", "options"), Output("bulk-lhs-val", "value"),
    Input("bulk-lhs-attr", "value"), Input("store-stims", "data"),
)
def upd_lhs_vals(attr, stims): return attr_value_options(stims or [], attr), "*"


@app.callback(
    Output("bulk-rhs-val", "options"), Output("bulk-rhs-val", "value"),
    Input("bulk-rhs-attr", "value"), Input("store-stims", "data"),
)
def upd_rhs_vals(attr, stims): return attr_value_options(stims or [], attr), "*"


# ── Style store: update from controls ────────────────────────────────────
@app.callback(
    Output("store-style", "data", allow_duplicate=True),
    Input("ctrl-cell-size",        "value"),
    Input("ctrl-cell-gap",         "value"),
    Input("ctrl-cell-radius",      "value"),
    Input("ctrl-colorscale",       "value"),
    Input("ctrl-cbar-min",         "value"),
    Input("ctrl-cbar-max",         "value"),
    Input("ctrl-show-cbar",        "value"),
    Input("ctrl-use-legend",       "value"),
    Input("ctrl-legend-show-nan",  "value"),
    Input("ctrl-nan-color",        "value"),
    Input("ctrl-diag-color",       "value"),
    Input("ctrl-mixed-color",      "value"),
    Input("ctrl-show-values",      "value"),
    Input("ctrl-val-font-size",    "value"),
    Input("ctrl-val-font-color",   "value"),
    Input("ctrl-label-font-size",  "value"),
    Input("ctrl-label-font-color", "value"),
    Input("ctrl-show-x-labels",    "value"),
    Input("ctrl-show-y-labels",    "value"),
    Input("ctrl-x-angle",          "value"),
    Input("ctrl-y-angle",          "value"),
    Input("ctrl-bg-color",         "value"),
    Input("ctrl-paper-bg",         "value"),
    prevent_initial_call=True,
)
def update_style(cs, cg, cr, colorscale, cmin, cmax, show_cb, use_leg, leg_nan,
                 nc, dc, mc, show_v, vfs, vfc, lfs, lfc, show_xl, show_yl, xa, ya, bg, pbg):
    DS = DEFAULT_STYLE
    return {
        "cell_size":        cs        if cs        is not None else DS["cell_size"],
        "cell_gap":         cg        if cg        is not None else DS["cell_gap"],
        "cell_radius":      cr        if cr        is not None else DS["cell_radius"],
        "colorscale":       colorscale or DS["colorscale"],
        "cbar_min":         cmin      if cmin      is not None else DS["cbar_min"],
        "cbar_max":         cmax      if cmax      is not None else DS["cbar_max"],
        "show_colorbar":    "y" in (show_cb  or []),
        "use_legend":       "y" in (use_leg  or []),
        "legend_show_nan":  "y" in (leg_nan  or []),
        "nan_color":        nc   or DS["nan_color"],
        "diag_color":       dc   or DS["diag_color"],
        "mixed_color":      mc   or DS["mixed_color"],
        "show_values":      "y" in (show_v or []),
        "val_font_size":    vfs  if vfs  is not None else DS["val_font_size"],
        "val_font_color":   vfc  or DS["val_font_color"],
        "label_font_size":  lfs  if lfs  is not None else DS["label_font_size"],
        "label_font_color": lfc  or DS["label_font_color"],
        "show_x_labels":    "y" in (show_xl or []),
        "show_y_labels":    "y" in (show_yl or []),
        "x_label_angle":    xa   if xa   is not None else DS["x_label_angle"],
        "y_label_angle":    ya   if ya   is not None else DS["y_label_angle"],
        "bg_color":         bg   or DS["bg_color"],
        "paper_bg":         pbg  or DS["paper_bg"],
    }


# ── Style controls: restore from store (session restore / preset load) ────
@app.callback(
    Output("ctrl-cell-size",        "value"),
    Output("ctrl-cell-gap",         "value"),
    Output("ctrl-cell-radius",      "value"),
    Output("ctrl-colorscale",       "value"),
    Output("ctrl-cbar-min",         "value"),
    Output("ctrl-cbar-max",         "value"),
    Output("ctrl-show-cbar",        "value"),
    Output("ctrl-use-legend",       "value"),
    Output("ctrl-legend-show-nan",  "value"),
    Output("ctrl-nan-color",        "value"),
    Output("ctrl-diag-color",       "value"),
    Output("ctrl-mixed-color",      "value"),
    Output("ctrl-show-values",      "value"),
    Output("ctrl-val-font-size",    "value"),
    Output("ctrl-val-font-color",   "value"),
    Output("ctrl-label-font-size",  "value"),
    Output("ctrl-label-font-color", "value"),
    Output("ctrl-show-x-labels",    "value"),
    Output("ctrl-show-y-labels",    "value"),
    Output("ctrl-x-angle",          "value"),
    Output("ctrl-y-angle",          "value"),
    Output("ctrl-bg-color",         "value"),
    Output("ctrl-paper-bg",         "value"),
    Input("btn-restore-session",    "n_clicks"),
    Input("dd-load-preset",         "value"),
    State("store-style",            "data"),
    State("store-presets",          "data"),
    prevent_initial_call=False,
)
def restore_controls(n_restore, preset_name, saved_style, presets):
    trigger = ctx.triggered_id
    DS = DEFAULT_STYLE
    # On first-ever visit there's nothing stored — skip the initial fire
    if trigger is None and saved_style is None:
        raise PreventUpdate
    if trigger == "dd-load-preset" and preset_name:
        s = {**DS, **((presets or {}).get(preset_name, {}))}
    else:
        s = {**DS, **(saved_style or {})}
    return (
        s["cell_size"], s["cell_gap"], s["cell_radius"],
        s["colorscale"], s["cbar_min"], s["cbar_max"],
        ["y"] if s["show_colorbar"] else [],
        ["y"] if s.get("use_legend", False) else [],
        ["y"] if s.get("legend_show_nan", True) else [],
        s["nan_color"], s["diag_color"], s["mixed_color"],
        ["y"] if s["show_values"] else [],
        s["val_font_size"], s["val_font_color"],
        s["label_font_size"], s["label_font_color"],
        ["y"] if s.get("show_x_labels", True) else [],
        ["y"] if s.get("show_y_labels", True) else [],
        s["x_label_angle"], s["y_label_angle"],
        s["bg_color"], s["paper_bg"],
    )


# ── Preset management ─────────────────────────────────────────────────────
@app.callback(
    Output("store-presets", "data", allow_duplicate=True),
    Output("preset-status", "children"),
    Input("btn-save-preset",   "n_clicks"),
    Input("btn-delete-preset", "n_clicks"),
    State("preset-name",       "value"),
    State("dd-load-preset",    "value"),
    State("store-style",       "data"),
    State("store-presets",     "data"),
    prevent_initial_call=True,
)
def manage_presets(n_save, n_del, name, selected, style, presets):
    presets = dict(presets or {})
    trigger = ctx.triggered_id
    msg = ""
    if trigger == "btn-save-preset":
        if not (name or "").strip():
            raise PreventUpdate
        presets[name.strip()] = style or DEFAULT_STYLE
        msg = f'Saved "{name.strip()}"'
    elif trigger == "btn-delete-preset":
        if selected and selected in presets:
            del presets[selected]
            msg = f'Deleted "{selected}"'
    return presets, msg


# Sole owner of dd-load-preset options — fires on load and after any preset change
@app.callback(
    Output("dd-load-preset", "options"),
    Input("store-presets", "data"),
)
def sync_preset_opts(presets):
    return [{"label": k, "value": k} for k in sorted(presets or {})]


# ── Render heatmap + table ────────────────────────────────────────────────
@app.callback(
    Output("heatmap",      "figure"),
    Output("table",        "columns"),
    Output("table",        "data"),
    Input("store-stims",   "data"),
    Input("store-matrix",  "data"),
    Input("radio-view",    "value"),
    Input("store-groupby", "data"),
    Input("store-sep",     "data"),
    Input("store-style",   "data"),
    State("store-meta",    "data"),
)
def render(stims, matrix_data, view_mode, group_by, sep, style, meta):
    if not stims or not matrix_data:
        return go.Figure(), [], []
    mf       = matrix_from_json(matrix_data)
    combined = bool(meta and meta.get("combined"))
    S        = {**DEFAULT_STYLE, **(style or {})}
    labels, mapping, m, mixed = _current_view(stims, mf, view_mode,
                                              group_by or [], combined, "_" if sep is None else sep)
    axis_col = representative_color(stims, mapping, len(labels))
    fig      = build_cell_heatmap(m, labels, S, mixed_mask=mixed, axis_colors=axis_col)

    # Table
    cols = [{"name": "", "id": "__row__", "editable": False}]
    for c in labels:
        cols.append({"name": c, "id": c, "editable": True})
    rows = []
    for i, lab in enumerate(labels):
        row = {"__row__": lab}
        for j, c in enumerate(labels):
            v = m[i, j]
            row[c] = "" if math.isnan(v) else (str(int(v)) if float(v).is_integer() else f"{v:.3g}")
        rows.append(row)
    return fig, cols, rows


# ── Click cell ────────────────────────────────────────────────────────────
@app.callback(
    Output("cell-row", "value"),
    Output("cell-col", "value"),
    Input("heatmap",   "clickData"),
)
def heatmap_click(click):
    if not click or not click.get("points"):
        return no_update, no_update
    p  = click["points"][0]
    cd = p.get("customdata")
    if cd and len(cd) >= 2:
        return cd[0], cd[1]
    return p.get("y", ""), p.get("x", "")


# ── Quick value buttons ───────────────────────────────────────────────────
@app.callback(
    Output("bulk-value",   "value"),
    Input("btn-quick-0",   "n_clicks"),
    Input("btn-quick-1",   "n_clicks"),
    Input("btn-quick-nan", "n_clicks"),
    prevent_initial_call=True,
)
def quick_val(n0, n1, nn):
    t = ctx.triggered_id
    return "0" if t == "btn-quick-0" else "1" if t == "btn-quick-1" else "NaN" if t == "btn-quick-nan" else no_update


# ── Matrix edits ──────────────────────────────────────────────────────────
@app.callback(
    Output("store-matrix",     "data", allow_duplicate=True),
    Output("store-undo-stack", "data", allow_duplicate=True),
    Output("store-redo-stack", "data", allow_duplicate=True),
    Input("btn-set-cell",      "n_clicks"),
    Input("btn-bulk-apply",    "n_clicks"),
    Input("btn-fill-nan",      "n_clicks"),
    Input("btn-same-to-0",     "n_clicks"),
    Input("btn-reset",         "n_clicks"),
    Input("btn-reset-model",   "n_clicks"),
    Input("btn-mirror",        "n_clicks"),
    Input("table",             "data"),
    State("store-matrix",      "data"),
    State("store-stims",       "data"),
    State("store-meta",        "data"),
    State("radio-view",        "value"),
    State("store-groupby",     "data"),
    State("store-sep",         "data"),
    State("cell-row",          "value"),
    State("cell-col",          "value"),
    State("cell-value",        "value"),
    State("bulk-lhs-attr",     "value"),
    State("bulk-lhs-val",      "value"),
    State("bulk-rhs-attr",     "value"),
    State("bulk-rhs-val",      "value"),
    State("bulk-value",        "value"),
    State("bulk-only-nan",     "value"),
    State("table",             "columns"),
    State("store-undo-stack",  "data"),
    prevent_initial_call=True,
)
def edit_matrix(n_set, n_bulk, n_fill, n_same0, n_reset, n_reset_model,
                n_mirror, tbl_data,
                matrix_data, stims, meta, view_mode, group_by, sep,
                cell_row, cell_col, cell_val,
                lhs_attr, lhs_val, rhs_attr, rhs_val, bulk_val, only_nan_chk,
                columns, undo_stack):
    if not stims or not matrix_data:
        return no_update, no_update, no_update
    trigger  = ctx.triggered_id
    mf       = matrix_from_json(matrix_data)
    combined = bool(meta and meta.get("combined"))
    sep      = "_" if sep is None else sep

    def _commit(new_mf):
        stack = list(undo_stack or [])
        stack.append(matrix_data)
        if len(stack) > MAX_UNDO: stack = stack[-MAX_UNDO:]
        return matrix_to_json(new_mf), stack, []

    if trigger in ("btn-reset", "btn-reset-model"):
        return _commit(fresh_matrix(len(stims)))

    if trigger == "btn-mirror":
        iu = np.triu_indices_from(mf, k=1)
        mf[(iu[1], iu[0])] = mf[iu]
        np.fill_diagonal(mf, 0.0)
        return _commit(mf)

    if trigger == "btn-set-cell":
        if not cell_row or not cell_col:
            return no_update, no_update, no_update
        labels, mapping, _, _ = _current_view(stims, mf, view_mode, group_by or [], combined, sep)
        if cell_row not in labels or cell_col not in labels:
            return no_update, no_update, no_update
        gi, gj = labels.index(cell_row), labels.index(cell_col)
        val = parse_value(cell_val)
        if view_mode == "full" or not group_by: set_pair(mf, gi, gj, val)
        else: broadcast_grouped_edit(mf, mapping, gi, gj, val)
        return _commit(mf)

    if trigger == "btn-bulk-apply":
        apply_bulk_rule(mf, stims, lhs_attr, lhs_val, rhs_attr, rhs_val,
                        parse_value(bulk_val), only_nan="only_nan" in (only_nan_chk or []))
        return _commit(mf)

    if trigger == "btn-fill-nan":
        apply_bulk_rule(mf, stims, "stim", "*", "stim", "*",
                        parse_value(bulk_val), only_nan=True)
        return _commit(mf)

    if trigger == "btn-same-to-0":
        if group_by:
            for i, si in enumerate(stims):
                ki = _group_key(si, group_by, sep)
                for j, sj in enumerate(stims):
                    if i == j: continue
                    if _group_key(sj, group_by, sep) == ki:
                        set_pair(mf, i, j, 0.0)
        return _commit(mf)

    if trigger == "table":
        labels, mapping, _, _ = _current_view(stims, mf, view_mode, group_by or [], combined, sep)
        if not tbl_data or not columns:
            return no_update, no_update, no_update
        col_ids = [c["id"] for c in columns if c["id"] != "__row__"]
        if col_ids != labels:
            return no_update, no_update, no_update
        for i, row in enumerate(tbl_data):
            for j, c in enumerate(col_ids):
                v = parse_value(row.get(c, ""))
                if i == j: continue
                if view_mode == "full" or not group_by: set_pair(mf, i, j, v)
                else: broadcast_grouped_edit(mf, mapping, i, j, v)
        return _commit(mf)

    return no_update, no_update, no_update


# ── Export ────────────────────────────────────────────────────────────────
@app.callback(
    Output("download-csv",   "data"),
    Output("export-status",  "children"),
    Input("btn-export",      "n_clicks"),
    State("store-stims",     "data"),
    State("store-matrix",    "data"),
    State("store-meta",      "data"),
    State("radio-view",      "value"),
    State("store-groupby",   "data"),
    State("store-sep",       "data"),
    State("store-style",     "data"),
    State("export-filename", "value"),
    State("export-folder",   "value"),
    prevent_initial_call=True,
)
def do_export(n, stims, matrix_data, meta, view_mode, group_by, sep, style, fname, folder):
    if not stims or not matrix_data:
        return no_update, "Nothing to export."
    mf       = matrix_from_json(matrix_data)
    combined = bool(meta and meta.get("combined"))
    S        = {**DEFAULT_STYLE, **(style or {})}
    labels, _, m, _ = _current_view(stims, mf, view_mode, group_by or [], combined, "_" if sep is None else sep)
    csv_text = dataframe_to_csv_string(to_export_dataframe(m, labels))
    fname = ((fname or "my-model.csv").strip())
    if not fname.lower().endswith(".csv"): fname += ".csv"
    saved_msg = ""
    if folder:
        try:
            os.makedirs(folder, exist_ok=True)
            csv_path  = os.path.join(folder, fname)
            json_path = csv_path.replace(".csv", "_style.json")
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write(csv_text)
            opts = style_to_summary(S, group_by, sep)
            opts["exported_at"] = str(pd.Timestamp.now())
            opts["labels"]      = labels
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(opts, f, indent=2)
            saved_msg = f"Saved: {csv_path}  +  {os.path.basename(json_path)}"
        except Exception as e:
            saved_msg = f"Save failed: {e}"
    return dict(content=csv_text, filename=fname), saved_msg


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8051)

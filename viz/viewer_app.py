"""Brain Viewer (live dashboard) — dual-species 3D result viewer.

Dogs and humans live in different anatomical spaces, so they are shown as two
independent 3D panels (Dog | Human), each toggleable, driven by one shared RSA
model and one shared z-threshold. The threshold re-applies in memory.

This live viewer is 3D-only (the failsafe site keeps 2D slices). Each panel is a
Plotly 3D rendering: supra-threshold z-map voxels over a faint brain surface.
An RSA model matrix (rounded-square cells + category legend) sits on top.

Light "PowerPoint" palette: white background, Office accent colors.

Standalone:  python -m viz.viewer_app   ->  http://127.0.0.1:8054
Mounted:     dashboard.py sets $VIEWER_URL_BASE and serves app.server.
"""

import os
import sys

import numpy as np
import pandas as pd
from dash import Dash, html, dcc, dash_table, no_update
from dash.dependencies import Input, Output, State

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import datasource, stimuli, niftiutil, dash_kwargs

# --- light PowerPoint-style palette ---------------------------------------
BG, PANEL, INK, MUTED, LINE, ACCENT = "#ffffff", "#f3f5f9", "#222222", "#667085", "#d5dbe5", "#4472C4"
INPUT_STYLE = {"backgroundColor": "#ffffff", "color": INK,
               "border": f"1px solid {LINE}", "borderRadius": "6px", "padding": "5px 8px"}
SPECIES = [("D", "Dog"), ("H", "Human")]
DEFAULT_DATASET = "EmoC"
THRESHOLD_PRESETS = [2.3, 3.1, 3.9]

_cache = {s: {} for s, _ in SPECIES}
_shared = {"datafolder": datasource.resolve_datafolder(DEFAULT_DATASET),
           "dataset": DEFAULT_DATASET, "modality": "RSA", "roi": None, "model": ""}


# --- data loading ---------------------------------------------------------

def _load_species(datafolder, dataset, modality, roi, model, specie):
    c = _cache[specie]
    c.clear()
    try:
        atlas, hi_aff, lo_aff, lo_shape = niftiutil.load_atlas(specie)
    except Exception as e:
        return f"{specie}: atlas load failed ({e})"
    c.update(atlas=atlas, hi_aff=hi_aff, lo_aff=lo_aff, lo_shape=lo_shape,
             overlay_lo=None, atlas_vol=None, vmin=0.0, vmax=1.0)
    if not (model and roi):
        return f"{specie}: atlas only"
    path, kind = datasource.overlay_path(datafolder, dataset, modality, specie, roi, model)
    if path is None:
        return f"{specie}: no map for '{model}'"
    ov_lo, ov_aff, _ = niftiutil.load_nifti(path)
    atlas_vol = niftiutil.resample_lowres_to_highres(atlas, hi_aff, ov_lo.shape, ov_aff)
    nz = ov_lo[np.abs(ov_lo) > 1e-6]
    c.update(overlay_lo=ov_lo, atlas_vol=atlas_vol,
             vmin=float(np.min(nz)) if nz.size else 0.0,
             vmax=float(np.max(nz)) if nz.size else 1.0)
    note = "" if kind == "unthresholded" else "  [corrected map — slider limited]"
    return f"{specie}: {kind} {model}  (z {c['vmin']:.1f}..{c['vmax']:.1f}){note}"


def _scan_models_union(datafolder, dataset, modality, roi):
    models = set()
    for s, _ in SPECIES:
        models.update(datasource.scan_models(datafolder, dataset, modality, s, roi))
    return sorted(models)


def _scan_rois_union(datafolder, dataset, modality):
    rois = set()
    for s, _ in SPECIES:
        rois.update(datasource.scan_roi_types(datafolder, dataset, modality, s))
    return sorted(rois)


def _load_table(specie, z_threshold):
    s = _shared
    path = datasource.table_path(s["datafolder"], s["dataset"], s["modality"], specie,
                                 s["roi"], s["model"], z_threshold=z_threshold)
    if path is None:
        return [], [], f"{specie}: no cluster table at z={z_threshold} — schedule a job for this threshold."
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    cols = [{"name": str(col), "id": str(col)} for col in df.columns]
    return cols, df.to_dict("records"), f"{specie}: {len(df)} clusters — {os.path.basename(path)}"


# --- RSA matrix as rounded-square cells + category legend -----------------

def _cell_color(v, vmax):
    """White -> Office blue by |value|/vmax (NaN -> white)."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or not vmax:
        return "rgb(255,255,255)"
    t = min(abs(float(v)) / vmax, 1.0)
    r = int(255 + t * (68 - 255)); g = int(255 + t * (114 - 255)); b = int(255 + t * (196 - 255))
    return f"rgb({r},{g},{b})"


def _matrix_grid(datafolder, dataset, model):
    path = os.path.join(datafolder, dataset, "rsa_models", f"{model}.csv")
    if not os.path.exists(path):
        return html.Div("No RSA model matrix for this model.", style={"color": MUTED, "fontSize": "12px"})
    df = pd.read_csv(path, index_col=0)
    conds = [str(c) for c in df.index]
    vals = df.values.astype(float)
    vmax = float(np.nanmax(np.abs(vals))) or 1.0
    n = len(conds)

    # category-name legend on top
    legend = html.Div([
        html.Span([
            html.Span(style={"display": "inline-block", "width": "12px", "height": "12px",
                             "borderRadius": "3px", "background": stimuli.label_color(lab),
                             "marginRight": "5px", "verticalAlign": "middle"}),
            html.Span(d["name"], style={"fontSize": "12px", "color": INK}),
        ], style={"marginRight": "16px", "whiteSpace": "nowrap"})
        for lab, d in stimuli.LABEL_DEF.items()
    ], style={"display": "flex", "flexWrap": "wrap", "gap": "4px 0", "marginBottom": "10px"})

    cells = []
    for i, row in enumerate(vals):
        cells.append(html.Div(conds[i], style={"fontSize": "10px", "color": INK, "textAlign": "right",
                                                "alignSelf": "center", "paddingRight": "4px",
                                                "fontFamily": "Consolas, monospace"}))
        for j, v in enumerate(row):
            cells.append(html.Div(
                title=f"{conds[i]} × {conds[j]} = {v:.2f}",
                style={"width": "28px", "height": "28px", "borderRadius": "7px",
                       "background": _cell_color(v, vmax), "border": f"1px solid {LINE}"}))
    grid = html.Div(cells, style={"display": "grid",
                                  "gridTemplateColumns": f"52px repeat({n}, 28px)",
                                  "gap": "4px", "justifyContent": "start"})
    return html.Div([legend, grid])


# --- app + layout ---------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True, **dash_kwargs("VIEWER_URL_BASE"))
app.title = "Brain Viewer"


def _species_block(code, name):
    return html.Div(id=f"vw-{code}-block", style={"backgroundColor": PANEL, "borderRadius": "8px",
                    "padding": "8px 12px", "marginBottom": "10px", "border": f"1px solid {LINE}"}, children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px"}, children=[
            html.H4(name, style={"margin": "2px 0", "color": INK}),
            html.Span(id=f"vw-{code}-status", style={"fontSize": "11px", "color": ACCENT}),
            html.Span(id=f"vw-{code}-tablenote", style={"fontSize": "11px", "color": "#a36", "marginLeft": "auto"}),
        ]),
        dcc.Graph(id=f"vw-{code}-vol", style={"height": "460px"}),
        dash_table.DataTable(
            id=f"vw-{code}-table", columns=[], data=[], page_size=8,
            style_table={"overflowX": "auto", "maxHeight": "240px", "overflowY": "auto", "marginTop": "6px"},
            style_header={"backgroundColor": "#eef1f6", "color": INK, "fontWeight": "bold"},
            style_cell={"backgroundColor": "#fff", "color": INK, "border": f"1px solid {LINE}",
                        "fontSize": "11px", "padding": "3px 6px", "textAlign": "center"}),
    ])


app.layout = html.Div(style={"backgroundColor": BG, "color": INK, "minHeight": "100vh",
                             "padding": "10px 16px", "fontFamily": "'Segoe UI', Arial, sans-serif"}, children=[
    html.H2("Brain Viewer", style={"textAlign": "center", "margin": "4px 0 8px", "color": INK}),

    html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "alignItems": "flex-end",
                    "padding": "10px 12px", "backgroundColor": PANEL, "borderRadius": "8px",
                    "border": f"1px solid {LINE}", "marginBottom": "8px"}, children=[
        html.Div([html.Label("Data folder", style={"fontSize": "11px", "color": MUTED}),
                  dcc.Input(id="vw-datafolder", value=_shared["datafolder"], type="text",
                            style={**INPUT_STYLE, "width": "240px"})]),
        html.Div([html.Label("Dataset", style={"fontSize": "11px", "color": MUTED}),
                  dcc.Input(id="vw-dataset", value=DEFAULT_DATASET, type="text",
                            style={**INPUT_STYLE, "width": "80px"})]),
        html.Div([html.Label("Modality", style={"fontSize": "11px", "color": MUTED}),
                  dcc.Dropdown(id="vw-modality", options=[{"label": "RSA", "value": "RSA"},
                               {"label": "GLM", "value": "GLM"}], value="RSA", clearable=False,
                               style={"width": "90px"})]),
        html.Div([html.Label("ROI type", style={"fontSize": "11px", "color": MUTED}),
                  dcc.Dropdown(id="vw-roi", options=[], value=None, style={"width": "190px"})]),
        html.Div([html.Label("Model / Contrast", style={"fontSize": "11px", "color": MUTED}),
                  dcc.Dropdown(id="vw-model", options=[], value=None, style={"width": "280px"})]),
        html.Button("Load", id="vw-load", n_clicks=0,
                    style={"height": "36px", "padding": "0 24px", "backgroundColor": ACCENT, "color": "white",
                           "border": "none", "borderRadius": "6px", "cursor": "pointer", "fontWeight": "bold"}),
        html.Span(id="vw-source", style={"fontSize": "11px", "color": MUTED, "marginLeft": "auto"}),
    ]),

    html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "20px", "alignItems": "center",
                    "padding": "8px 12px", "backgroundColor": PANEL, "borderRadius": "8px",
                    "border": f"1px solid {LINE}", "marginBottom": "8px"}, children=[
        html.Div([html.Label("z-threshold (explore unthresholded map)",
                             style={"fontSize": "11px", "color": MUTED}),
                  dcc.Slider(id="vw-zt", min=0, max=8, step=0.1, value=3.1,
                             marks={p: str(p) for p in THRESHOLD_PRESETS},
                             tooltip={"placement": "bottom"})], style={"width": "320px"}),
        html.Div([html.Label("Cluster table @ z", style={"fontSize": "11px", "color": MUTED}),
                  dcc.Dropdown(id="vw-table-zt", options=[{"label": str(p), "value": p} for p in THRESHOLD_PRESETS],
                               value=3.1, clearable=False, style={"width": "100px"})]),
        dcc.Checklist(id="vw-show", options=[{"label": " Dog", "value": "D"},
                      {"label": " Human", "value": "H"}], value=["D", "H"],
                      inline=True, style={"fontSize": "13px"}),
    ]),

    html.Div(style={"backgroundColor": PANEL, "borderRadius": "8px", "padding": "10px 14px",
             "border": f"1px solid {LINE}", "marginBottom": "10px"}, children=[
        html.H4("RSA model matrix", style={"margin": "0 0 8px", "color": INK}),
        html.Div(id="vw-matrix"),
    ]),

    _species_block("D", "Dog"),
    _species_block("H", "Human"),

    dcc.Store(id="vw-loaded", data=0),
])


# --- callbacks ------------------------------------------------------------

@app.callback(Output("vw-roi", "options"), Output("vw-roi", "value"), Output("vw-source", "children"),
              Input("vw-modality", "value"), Input("vw-datafolder", "value"), Input("vw-dataset", "value"))
def cb_rois(modality, datafolder, dataset):
    rois = _scan_rois_union(datafolder, dataset, modality)
    return [{"label": r, "value": r} for r in rois], (rois[0] if rois else None), datasource.describe_source(dataset)


@app.callback(Output("vw-model", "options"), Output("vw-model", "value"),
              Input("vw-roi", "value"),
              State("vw-datafolder", "value"), State("vw-dataset", "value"), State("vw-modality", "value"))
def cb_models(roi, datafolder, dataset, modality):
    if not roi:
        return [], None
    models = _scan_models_union(datafolder, dataset, modality, roi)
    return [{"label": m, "value": m} for m in models], (models[0] if models else None)


@app.callback(
    Output("vw-loaded", "data"),
    Output("vw-D-status", "children"), Output("vw-H-status", "children"),
    Output("vw-matrix", "children"),
    Input("vw-load", "n_clicks"),
    State("vw-datafolder", "value"), State("vw-dataset", "value"), State("vw-modality", "value"),
    State("vw-roi", "value"), State("vw-model", "value"), State("vw-loaded", "data"),
    prevent_initial_call=True,
)
def cb_load(_n, datafolder, dataset, modality, roi, model, loaded):
    _shared.update(datafolder=datafolder, dataset=dataset, modality=modality, roi=roi, model=model or "")
    statuses = {code: _load_species(datafolder, dataset, modality, roi, model, code) for code, _ in SPECIES}
    matrix = _matrix_grid(datafolder, dataset, model) if (modality == "RSA" and model) else \
        html.Div("Matrix shown for RSA models only.", style={"color": MUTED, "fontSize": "12px"})
    return (loaded or 0) + 1, statuses["D"], statuses["H"], matrix


@app.callback(
    Output("vw-D-table", "columns"), Output("vw-D-table", "data"), Output("vw-D-tablenote", "children"),
    Output("vw-H-table", "columns"), Output("vw-H-table", "data"), Output("vw-H-tablenote", "children"),
    Input("vw-loaded", "data"), Input("vw-table-zt", "value"),
)
def cb_tables(_loaded, table_zt):
    if not _shared.get("model"):
        return [], [], "", [], [], ""
    dc, dd, dn = _load_table("D", table_zt)
    hc, hd, hn = _load_table("H", table_zt)
    return dc, dd, dn, hc, hd, hn


def _render_species(code, zt, show):
    c = _cache[code]
    if c.get("atlas") is None or code not in show:
        return niftiutil.empty_fig(height=460)
    return niftiutil.make_volume_fig(c.get("overlay_lo"), zt, c.get("vmin", 0), c.get("vmax", 1),
                                     title=f"{code}", height=460, atlas_lowres=c.get("atlas_vol"))


@app.callback(
    Output("vw-D-vol", "figure"), Output("vw-H-vol", "figure"),
    Output("vw-D-block", "style"), Output("vw-H-block", "style"),
    Input("vw-loaded", "data"), Input("vw-zt", "value"), Input("vw-show", "value"),
)
def cb_render(_loaded, zt, show):
    show = show or []
    base = {"backgroundColor": PANEL, "borderRadius": "8px", "padding": "8px 12px",
            "marginBottom": "10px", "border": f"1px solid {LINE}"}
    d_style = base if "D" in show else {**base, "display": "none"}
    h_style = base if "H" in show else {**base, "display": "none"}
    return _render_species("D", zt, show), _render_species("H", zt, show), d_style, h_style


if __name__ == "__main__":
    app.run(debug=True, port=8054)

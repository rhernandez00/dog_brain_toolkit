"""Brain Viewer (Phase 2) — dual-species result viewer.

Dogs and humans live in different anatomical spaces, so they are shown as two
independent brain panels (Dog | Human), each toggleable, driven by one shared
RSA model selection.

Thresholding model (per project decision):
  * The interactive z-threshold slider operates on the **unthresholded** z-map
    (step 7 output, mirrored to current-results as ``{specie}_{model}_z.nii.gz``)
    so the user can freely explore the data. Re-thresholding happens in memory.
  * Clusters / tables come from **threshold-specific** cluster-corrected jobs
    (default thresholds z = 2.3, 3.1, 3.9). The "Table @ z" selector picks which
    corrected output to read; if absent, the panel says a job must be scheduled.

Each panel offers 2D orthogonal slices and a 3D rendering (supra-threshold blob
over a faint brain surface). An RSA model matrix with stimulus chips sits on top.

Standalone:  python -m viz.viewer_app   ->  http://127.0.0.1:8054
Mounted:     dashboard.py sets $VIEWER_URL_BASE and serves app.server.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, no_update
from dash.dependencies import Input, Output, State

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viz import datasource, stimuli, niftiutil, dash_kwargs

DARK_BG, PANEL_BG, ACCENT = "#0f0f23", "#1a1a2e", "#4a90d9"
INPUT_STYLE = {"backgroundColor": "#16213e", "color": "white",
               "border": "1px solid #333", "borderRadius": "4px", "padding": "4px 8px"}
SPECIES = [("D", "Dog"), ("H", "Human")]
DEFAULT_DATASET = "EmoC"
THRESHOLD_PRESETS = [2.3, 3.1, 3.9]

# Single-user server-side cache: raw arrays kept so the threshold slider
# re-renders without touching disk.
_cache = {s: {} for s, _ in SPECIES}
_shared = {"datafolder": datasource.resolve_datafolder(DEFAULT_DATASET),
           "dataset": DEFAULT_DATASET, "modality": "RSA", "roi": None, "model": ""}


# --- data loading ---------------------------------------------------------

def _load_species(datafolder, dataset, modality, roi, model, specie):
    """Populate _cache[specie] with atlas + unthresholded overlay; return status."""
    c = _cache[specie]
    c.clear()
    try:
        atlas, hi_aff, lo_aff, lo_shape = niftiutil.load_atlas(specie)
    except Exception as e:
        return f"{specie}: atlas load failed ({e})"
    c.update(atlas=atlas, hi_aff=hi_aff, lo_aff=lo_aff, lo_shape=lo_shape,
             overlay_hi=None, overlay_lo=None, atlas_vol=None, vmin=0.0, vmax=1.0)
    if not (model and roi):
        return f"{specie}: atlas only"
    path, kind = datasource.overlay_path(datafolder, dataset, modality, specie, roi, model)
    if path is None:
        return f"{specie}: no map for '{model}'"
    ov_lo, ov_aff, _ = niftiutil.load_nifti(path)
    ov_hi = niftiutil.resample_lowres_to_highres(ov_lo, ov_aff, atlas.shape, hi_aff)
    # Atlas sampled onto the overlay's own grid -> 3D surface aligns with blob.
    atlas_vol = niftiutil.resample_lowres_to_highres(atlas, hi_aff, ov_lo.shape, ov_aff)
    nz = ov_hi[np.abs(ov_hi) > 1e-6]
    c.update(overlay_hi=ov_hi, overlay_lo=ov_lo, atlas_vol=atlas_vol,
             vmin=float(np.min(nz)) if nz.size else 0.0,
             vmax=float(np.max(nz)) if nz.size else 1.0)
    note = "" if kind == "unthresholded" else "  [corrected map — slider limited; schedule unthresholded]"
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


def _rsa_matrix_fig(datafolder, dataset, model):
    path = os.path.join(datafolder, dataset, "rsa_models", f"{model}.csv")
    if not os.path.exists(path):
        return niftiutil.empty_fig("RSA model matrix", height=340), []
    df = pd.read_csv(path, index_col=0)
    fig = go.Figure(go.Heatmap(z=df.values, x=list(df.columns), y=list(df.index),
                               colorscale="Viridis", colorbar=dict(title="dissim", thickness=10)))
    fig.update_layout(title=dict(text=f"RSA model: {model}", font=dict(size=13)),
                      margin=dict(l=60, r=10, t=34, b=60), height=340,
                      paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG, font_color="white",
                      yaxis=dict(autorange="reversed"))
    chips = []
    for cond in df.index:
        color = stimuli.label_color(str(cond)[-1])
        chips.append(html.Span(str(cond), style={
            "backgroundColor": color, "color": "#000", "borderRadius": "6px",
            "padding": "4px 8px", "margin": "2px", "fontFamily": "Consolas, monospace",
            "fontSize": "11px", "fontWeight": "bold", "display": "inline-block"}))
    return fig, chips


def _load_table(specie, z_threshold):
    """(columns, data, note) for a species cluster table at a threshold."""
    s = _shared
    path = datasource.table_path(s["datafolder"], s["dataset"], s["modality"], specie,
                                 s["roi"], s["model"], z_threshold=z_threshold)
    if path is None:
        return [], [], f"{specie}: no cluster table at z={z_threshold} — schedule a job for this threshold."
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    cols = [{"name": str(col), "id": str(col)} for col in df.columns]
    return cols, df.to_dict("records"), f"{specie}: {len(df)} clusters — {os.path.basename(path)}"


# --- app + layout ---------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True, **dash_kwargs("VIEWER_URL_BASE"))
app.title = "Brain Viewer"


def _species_block(code, name):
    return html.Div(id=f"vw-{code}-block", style={"backgroundColor": PANEL_BG, "borderRadius": "8px",
                    "padding": "8px 12px", "marginBottom": "10px"}, children=[
        html.Div(style={"display": "flex", "alignItems": "center", "gap": "10px"}, children=[
            html.H4(name, style={"margin": "2px 0", "color": "#e0e0ff"}),
            html.Span(id=f"vw-{code}-status", style={"fontSize": "11px", "color": "#9ad"}),
            html.Span(id=f"vw-{code}-coord", style={"fontSize": "11px", "color": "#b0b0ff",
                      "fontFamily": "Consolas, monospace", "marginLeft": "16px"}),
            html.Span(id=f"vw-{code}-tablenote", style={"fontSize": "11px", "color": "#caa", "marginLeft": "auto"}),
        ]),
        html.Div(style={"display": "flex", "gap": "6px", "flexWrap": "wrap"}, children=[
            html.Div(dcc.Graph(id=f"vw-{code}-sag", style={"height": "300px"}), style={"flex": "1 1 220px"}),
            html.Div(dcc.Graph(id=f"vw-{code}-cor", style={"height": "300px"}), style={"flex": "1 1 220px"}),
            html.Div(dcc.Graph(id=f"vw-{code}-axi", style={"height": "300px"}), style={"flex": "1 1 220px"}),
            html.Div(dcc.Graph(id=f"vw-{code}-vol", style={"height": "300px"}), style={"flex": "1 1 280px"}),
        ]),
        html.Div(style={"display": "flex", "gap": "10px", "marginTop": "4px"}, children=[
            html.Div([html.Label("Sagittal", style={"fontSize": "10px", "color": "#888"}),
                      dcc.Slider(id=f"vw-{code}-slsag", min=0, max=10, step=1, value=5,
                                 tooltip={"placement": "bottom"})], style={"flex": "1"}),
            html.Div([html.Label("Coronal", style={"fontSize": "10px", "color": "#888"}),
                      dcc.Slider(id=f"vw-{code}-slcor", min=0, max=10, step=1, value=5,
                                 tooltip={"placement": "bottom"})], style={"flex": "1"}),
            html.Div([html.Label("Axial", style={"fontSize": "10px", "color": "#888"}),
                      dcc.Slider(id=f"vw-{code}-slaxi", min=0, max=10, step=1, value=5,
                                 tooltip={"placement": "bottom"})], style={"flex": "1"}),
        ]),
        dash_table.DataTable(
            id=f"vw-{code}-table", columns=[], data=[], row_selectable="single", page_size=8,
            style_table={"overflowX": "auto", "maxHeight": "220px", "overflowY": "auto", "marginTop": "6px"},
            style_header={"backgroundColor": "#16213e", "color": "white", "fontWeight": "bold"},
            style_cell={"backgroundColor": DARK_BG, "color": "white", "border": "1px solid #222",
                        "fontSize": "11px", "padding": "3px 6px", "textAlign": "center"}),
    ])


app.layout = html.Div(style={"backgroundColor": DARK_BG, "color": "white", "minHeight": "100vh",
                             "padding": "10px 16px", "fontFamily": "'Segoe UI', Arial, sans-serif"}, children=[
    html.H2("Brain Viewer", style={"textAlign": "center", "margin": "4px 0 8px", "color": "#e0e0ff"}),

    # config bar
    html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "alignItems": "flex-end",
                    "padding": "10px 12px", "backgroundColor": PANEL_BG, "borderRadius": "8px",
                    "marginBottom": "8px"}, children=[
        html.Div([html.Label("Data folder", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Input(id="vw-datafolder", value=_shared["datafolder"], type="text",
                            style={**INPUT_STYLE, "width": "240px"})]),
        html.Div([html.Label("Dataset", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Input(id="vw-dataset", value=DEFAULT_DATASET, type="text",
                            style={**INPUT_STYLE, "width": "80px"})]),
        html.Div([html.Label("Modality", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Dropdown(id="vw-modality", options=[{"label": "RSA", "value": "RSA"},
                               {"label": "GLM", "value": "GLM"}], value="RSA", clearable=False,
                               style={"width": "90px", "color": "#000"})]),
        html.Div([html.Label("ROI type", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Dropdown(id="vw-roi", options=[], value=None, style={"width": "190px", "color": "#000"})]),
        html.Div([html.Label("Model / Contrast", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Dropdown(id="vw-model", options=[], value=None, style={"width": "280px", "color": "#000"})]),
        html.Button("Load", id="vw-load", n_clicks=0,
                    style={"height": "36px", "padding": "0 24px", "backgroundColor": ACCENT, "color": "white",
                           "border": "none", "borderRadius": "6px", "cursor": "pointer", "fontWeight": "bold"}),
        html.Span(id="vw-source", style={"fontSize": "11px", "color": "#888", "marginLeft": "auto"}),
    ]),

    # controls bar
    html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "20px", "alignItems": "center",
                    "padding": "8px 12px", "backgroundColor": PANEL_BG, "borderRadius": "8px",
                    "marginBottom": "8px"}, children=[
        html.Div([html.Label("Exploration z-threshold (unthresholded map)",
                             style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Slider(id="vw-zt", min=0, max=8, step=0.1, value=3.1,
                             marks={p: str(p) for p in THRESHOLD_PRESETS},
                             tooltip={"placement": "bottom"})], style={"width": "300px"}),
        html.Div([html.Label("Overlay opacity", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Slider(id="vw-opacity", min=0.1, max=1, step=0.05, value=0.8,
                             marks={0.1: ".1", 1: "1"}, tooltip={"placement": "bottom"})],
                 style={"width": "150px"}),
        html.Div([html.Label("Cluster table @ z", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Dropdown(id="vw-table-zt", options=[{"label": str(p), "value": p} for p in THRESHOLD_PRESETS],
                               value=3.1, clearable=False, style={"width": "100px", "color": "#000"})]),
        dcc.Checklist(id="vw-view", options=[{"label": " 2D slices", "value": "slices"},
                      {"label": " 3D", "value": "volume"}], value=["slices", "volume"],
                      inline=True, style={"fontSize": "13px"}),
        dcc.Checklist(id="vw-show", options=[{"label": " Dog", "value": "D"},
                      {"label": " Human", "value": "H"}], value=["D", "H"],
                      inline=True, style={"fontSize": "13px"}),
    ]),

    # RSA model matrix + chips
    html.Div(id="vw-matrix-wrap", style={"backgroundColor": PANEL_BG, "borderRadius": "8px",
             "padding": "8px 12px", "marginBottom": "10px"}, children=[
        html.Div(id="vw-chips", style={"marginBottom": "4px"}),
        dcc.Graph(id="vw-matrix", style={"height": "340px"}),
    ]),

    _species_block("D", "Dog"),
    _species_block("H", "Human"),

    dcc.Store(id="vw-loaded", data=0),
])


# --- callbacks: option population -----------------------------------------

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


# --- callbacks: load (atlas + overlay + matrix + slider ranges) -----------

_SLIDER_OUTPUTS = []
for _c, _ in SPECIES:
    for _ax in ("slsag", "slcor", "slaxi"):
        _SLIDER_OUTPUTS += [Output(f"vw-{_c}-{_ax}", "min"), Output(f"vw-{_c}-{_ax}", "max"),
                            Output(f"vw-{_c}-{_ax}", "value")]


@app.callback(
    [Output("vw-loaded", "data"),
     Output("vw-D-status", "children"), Output("vw-H-status", "children"),
     Output("vw-matrix", "figure"), Output("vw-chips", "children")]
    + _SLIDER_OUTPUTS,
    Input("vw-load", "n_clicks"),
    State("vw-datafolder", "value"), State("vw-dataset", "value"), State("vw-modality", "value"),
    State("vw-roi", "value"), State("vw-model", "value"), State("vw-loaded", "data"),
    prevent_initial_call=True,
)
def cb_load(_n, datafolder, dataset, modality, roi, model, loaded):
    _shared.update(datafolder=datafolder, dataset=dataset, modality=modality, roi=roi, model=model or "")
    statuses = {code: _load_species(datafolder, dataset, modality, roi, model, code) for code, _ in SPECIES}

    slider_vals = []
    for code, _ in SPECIES:
        atlas = _cache[code].get("atlas")
        shape = atlas.shape if atlas is not None else (10, 10, 10)
        for ax in range(3):
            slider_vals += [0, shape[ax] - 1, shape[ax] // 2]

    if modality == "RSA" and model:
        matrix_fig, chips = _rsa_matrix_fig(datafolder, dataset, model)
    else:
        matrix_fig, chips = niftiutil.empty_fig("RSA model matrix", height=340), []

    return ([(loaded or 0) + 1, statuses["D"], statuses["H"], matrix_fig, chips] + slider_vals)


# --- callbacks: cluster tables (per-threshold, from disk) -----------------

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


# --- callbacks: render (in-memory threshold) ------------------------------

def _coord_text(code, sag, cor, axi):
    """Low-res voxel + world-mm coordinate string for the current hi-res slice."""
    c = _cache[code]
    if c.get("hi_aff") is None or c.get("lo_aff") is None:
        return ""
    world = niftiutil.voxel_to_world((sag, cor, axi), c["hi_aff"])
    lo = niftiutil.world_to_voxel(world, c["lo_aff"])
    return (f"low-res vox ({lo[0]}, {lo[1]}, {lo[2]})  |  "
            f"mm ({world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f})")


def _render_species(code, zt, opacity, view, show, sag, cor, axi):
    c = _cache[code]
    atlas = c.get("atlas")
    if atlas is None or code not in show:
        e = niftiutil.empty_fig()
        return e, e, e, e
    ov_hi, ov_lo = c.get("overlay_hi"), c.get("overlay_lo")
    vmin, vmax = c.get("vmin", 0), c.get("vmax", 1)
    if "slices" in view:
        fs = niftiutil.make_slice_fig(atlas, ov_hi, 0, sag, opacity, zt, vmin, vmax, title="Sagittal", height=300)
        fc = niftiutil.make_slice_fig(atlas, ov_hi, 1, cor, opacity, zt, vmin, vmax, title="Coronal", height=300)
        fa = niftiutil.make_slice_fig(atlas, ov_hi, 2, axi, opacity, zt, vmin, vmax, title="Axial", height=300)
    else:
        fs = fc = fa = niftiutil.empty_fig("(slices hidden)", height=300)
    fv = (niftiutil.make_volume_fig(ov_lo, zt, vmin, vmax, title=f"{code} 3D", height=300,
                                    atlas_lowres=c.get("atlas_vol"))
          if "volume" in view else niftiutil.empty_fig("(3D hidden)", height=300))
    return fs, fc, fa, fv


@app.callback(
    [Output("vw-D-sag", "figure"), Output("vw-D-cor", "figure"), Output("vw-D-axi", "figure"), Output("vw-D-vol", "figure"),
     Output("vw-H-sag", "figure"), Output("vw-H-cor", "figure"), Output("vw-H-axi", "figure"), Output("vw-H-vol", "figure"),
     Output("vw-D-block", "style"), Output("vw-H-block", "style"),
     Output("vw-D-coord", "children"), Output("vw-H-coord", "children")],
    [Input("vw-loaded", "data"), Input("vw-zt", "value"), Input("vw-opacity", "value"),
     Input("vw-view", "value"), Input("vw-show", "value"),
     Input("vw-D-slsag", "value"), Input("vw-D-slcor", "value"), Input("vw-D-slaxi", "value"),
     Input("vw-H-slsag", "value"), Input("vw-H-slcor", "value"), Input("vw-H-slaxi", "value")],
)
def cb_render(_loaded, zt, opacity, view, show, dsag, dcor, daxi, hsag, hcor, haxi):
    view = view or []
    show = show or []
    d = _render_species("D", zt, opacity, view, show, dsag, dcor, daxi)
    h = _render_species("H", zt, opacity, view, show, hsag, hcor, haxi)
    base = {"backgroundColor": PANEL_BG, "borderRadius": "8px", "padding": "8px 12px", "marginBottom": "10px"}
    d_style = base if "D" in show else {**base, "display": "none"}
    h_style = base if "H" in show else {**base, "display": "none"}
    return (*d, *h, d_style, h_style, _coord_text("D", dsag, dcor, daxi), _coord_text("H", hsag, hcor, haxi))


# --- callbacks: cluster navigation ----------------------------------------

def _make_nav(code):
    @app.callback(
        Output(f"vw-{code}-slsag", "value", allow_duplicate=True),
        Output(f"vw-{code}-slcor", "value", allow_duplicate=True),
        Output(f"vw-{code}-slaxi", "value", allow_duplicate=True),
        Input(f"vw-{code}-table", "selected_rows"),
        State(f"vw-{code}-table", "data"),
        prevent_initial_call=True,
    )
    def _nav(selected, data):
        c = _cache[code]
        if not selected or not data or c.get("hi_aff") is None:
            return no_update, no_update, no_update
        row = data[selected[0]]

        def _find(axis):
            exact = {axis, f"vox_{axis}", f"peak_{axis}", {"x": "i", "y": "j", "z": "k"}[axis]}
            # exact names first, then any voxel-coord column for this axis
            for k in row:
                if str(k).lower() in exact:
                    return k
            for k in row:
                kl = str(k).lower()
                if kl.endswith(f"{axis}_vox") or kl.endswith(f"_{axis}_vox"):
                    return k
            return None

        xc, yc, zc = _find("x"), _find("y"), _find("z")
        if not (xc and yc and zc):
            return no_update, no_update, no_update
        lo_vox = (int(row[xc]), int(row[yc]), int(row[zc]))
        world = niftiutil.voxel_to_world(lo_vox, c["lo_aff"])
        hi = niftiutil.world_to_voxel(world, c["hi_aff"])
        s = c["atlas"].shape
        return (int(np.clip(hi[0], 0, s[0] - 1)), int(np.clip(hi[1], 0, s[1] - 1)),
                int(np.clip(hi[2], 0, s[2] - 1)))
    return _nav


_nav_D = _make_nav("D")
_nav_H = _make_nav("H")


if __name__ == "__main__":
    app.run(debug=True, port=8054)

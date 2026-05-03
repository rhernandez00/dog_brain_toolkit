"""
Brain Viewer — Interactive visual analytics for dog & human neuroimaging results.

Launch:  python brain_viewer.py
Then open http://127.0.0.1:8050 in your browser.

Requires: dash, plotly, nibabel, numpy, pandas, openpyxl
Install missing deps:  pip install dash
"""

import os, glob
import numpy as np
import nibabel as nib
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, html, dcc, dash_table, no_update
from dash.dependencies import Input, Output, State
import base64
from io import BytesIO

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    datafolder=r"G:\My Drive\Results",
    dataset="EmoC",
    modality="RSA",
    specie="D",
    roi_type="b_GreyMatter2mmB",
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ATLAS_PATHS = {
    "D": {
        "low":  os.path.join(SCRIPT_DIR, "Atlas", "Dog", "Czeibert", "brain2mm.nii.gz"),
        "high": os.path.join(SCRIPT_DIR, "Atlas", "Dog", "Czeibert", "brain.nii.gz"),
    },
    "H": {
        "low":  os.path.join(SCRIPT_DIR, "Atlas", "Hum", "MNI152_T1_2mm_brain.nii.gz"),
        "high": os.path.join(SCRIPT_DIR, "Atlas", "Hum", "MNI152_T1_2mm_brain.nii.gz"),
    },
}

OVERLAY_COLORSCALE = "Hot"
ATLAS_COLORSCALE = "Gray"

# ---------------------------------------------------------------------------
# Server-side cache (single user app)
# ---------------------------------------------------------------------------
_cache = {
    "atlas": None,       # hi-res atlas 3D array (normalized 0-1)
    "overlay": None,     # hi-res overlay 3D array (resampled)
    "hi_affine": None,
    "lo_affine": None,
    "lo_shape": None,
    "hi_shape": None,
    "vmin": 0,
    "vmax": 1,
    "specie": None,
    "model": "",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_nifti(path):
    img = nib.load(path)
    return np.asanyarray(img.dataobj, dtype=np.float32), img.affine, img.header


def voxel_to_world(vox_coords, affine):
    vox = np.array([*vox_coords, 1.0])
    return tuple((affine @ vox)[:3])


def world_to_voxel(world_coords, affine):
    inv = np.linalg.inv(affine)
    w = np.array([*world_coords, 1.0])
    return tuple(np.round((inv @ w)[:3]).astype(int))


def resample_lowres_to_highres(low_data, low_affine, high_shape, high_affine):
    inv_low = np.linalg.inv(low_affine)
    ii, jj, kk = np.mgrid[0:high_shape[0], 0:high_shape[1], 0:high_shape[2]]
    flat = np.vstack([ii.ravel(), jj.ravel(), kk.ravel(), np.ones(ii.size)])
    world = high_affine @ flat
    vox_low = np.round((inv_low @ world)[:3]).astype(int)
    lo_s = np.array(low_data.shape).reshape(3, 1)
    mask = np.all((vox_low >= 0) & (vox_low < lo_s), axis=0)
    out = np.zeros(ii.size, dtype=np.float32)
    out[mask] = low_data[vox_low[0, mask], vox_low[1, mask], vox_low[2, mask]]
    return out.reshape(high_shape)


def scan_roi_types(datafolder, dataset, modality, specie):
    base = os.path.join(datafolder, dataset, "current-results", modality, specie)
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))


def scan_available_models(datafolder, dataset, modality, specie, roi_type):
    d = os.path.join(datafolder, dataset, "current-results", modality, specie, roi_type)
    files = glob.glob(os.path.join(d, f"{specie}_*_z_corrected.nii.gz"))
    prefix = f"{specie}_"
    suffix = "_z_corrected.nii.gz"
    return sorted(os.path.basename(f)[len(prefix):-len(suffix)] for f in files)


def get_result_paths(datafolder, dataset, modality, specie, roi_type, model_name):
    d = os.path.join(datafolder, dataset, "current-results", modality, specie, roi_type)
    nifti = os.path.join(d, f"{specie}_{model_name}_z_corrected.nii.gz")
    table = os.path.join(d, f"{specie}_{model_name}.xlsx")
    if not os.path.exists(table):
        table = os.path.join(d, f"{specie}_{model_name}.csv")
    return nifti, table


def make_slice_fig(axis, slice_idx, opacity, show_crosshair, cross_positions, title=""):
    atlas = _cache["atlas"]
    overlay = _cache["overlay"]
    if atlas is None:
        return empty_fig(title)

    if axis == 0:
        bg = np.rot90(atlas[slice_idx, :, :])
        ov = np.rot90(overlay[slice_idx, :, :]) if overlay is not None else None
    elif axis == 1:
        bg = np.rot90(atlas[:, slice_idx, :])
        ov = np.rot90(overlay[:, slice_idx, :]) if overlay is not None else None
    else:
        bg = np.rot90(atlas[:, :, slice_idx])
        ov = np.rot90(overlay[:, :, slice_idx]) if overlay is not None else None

    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=bg, colorscale=ATLAS_COLORSCALE, showscale=False, hoverinfo="skip"))

    if ov is not None:
        ov_masked = np.where(np.abs(ov) > 1e-6, ov, np.nan)
        if not np.all(np.isnan(ov_masked)):
            fig.add_trace(go.Heatmap(
                z=ov_masked, colorscale=OVERLAY_COLORSCALE,
                opacity=opacity, showscale=True,
                zmin=_cache["vmin"], zmax=_cache["vmax"],
                colorbar=dict(title="z", len=0.6, thickness=12),
                hoverinfo="skip",
            ))

    if show_crosshair and cross_positions:
        cx, cy = cross_positions
        fig.add_shape(type="line", x0=cx, x1=cx, y0=0, y1=bg.shape[0]-1,
                      line=dict(color="cyan", width=1, dash="dot"))
        fig.add_shape(type="line", x0=0, x1=bg.shape[1]-1, y0=cy, y1=cy,
                      line=dict(color="cyan", width=1, dash="dot"))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=5, r=5, t=30, b=5),
        xaxis=dict(visible=False, scaleanchor="y", constrain="domain"),
        yaxis=dict(visible=False, constrain="domain"),
        plot_bgcolor="black", paper_bgcolor="#1a1a2e",
        font_color="white", height=420,
    )
    return fig


def empty_fig(title=""):
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        margin=dict(l=5, r=5, t=30, b=5),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="black", paper_bgcolor="#1a1a2e",
        font_color="white", height=420,
        annotations=[dict(text="No data loaded", showarrow=False,
                          font=dict(size=16, color="#555"), xref="paper", yref="paper", x=0.5, y=0.5)],
    )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Dash(__name__)
app.title = "Brain Viewer"

DARK_BG = "#0f0f23"
PANEL_BG = "#1a1a2e"
ACCENT = "#4a90d9"
INPUT_STYLE = {"backgroundColor": "#16213e", "color": "white", "border": "1px solid #333", "borderRadius": "4px", "padding": "4px 8px"}

app.layout = html.Div(style={"backgroundColor": DARK_BG, "color": "white", "minHeight": "100vh",
                              "padding": "10px 16px", "fontFamily": "'Segoe UI', Arial, sans-serif"}, children=[

    html.H2("Brain Viewer", style={"textAlign": "center", "margin": "4px 0 8px", "letterSpacing": "1px", "color": "#e0e0ff"}),

    # ---- Config bar ----
    html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "alignItems": "flex-end",
                     "padding": "10px 12px", "backgroundColor": PANEL_BG, "borderRadius": "8px", "marginBottom": "8px"}, children=[
        html.Div([html.Label("Data folder", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Input(id="input-datafolder", value=DEFAULTS["datafolder"], type="text",
                            style={**INPUT_STYLE, "width": "260px"})]),
        html.Div([html.Label("Dataset", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Input(id="input-dataset", value=DEFAULTS["dataset"], type="text",
                            style={**INPUT_STYLE, "width": "80px"})]),
        html.Div([html.Label("Species", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Dropdown(id="dd-specie", options=[{"label": "Dog", "value": "D"}, {"label": "Human", "value": "H"}],
                               value=DEFAULTS["specie"], clearable=False,
                               style={"width": "100px"})]),
        html.Div([html.Label("Modality", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Dropdown(id="dd-modality", options=[{"label": "RSA", "value": "RSA"}, {"label": "GLM", "value": "GLM"}],
                               value=DEFAULTS["modality"], clearable=False,
                               style={"width": "90px"})]),
        html.Div([html.Label("ROI type", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Dropdown(id="dd-roi", options=[], value=None, style={"width": "190px"})]),
        html.Div([html.Label("Model / Contrast", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Dropdown(id="dd-model", options=[], value=None, style={"width": "300px"})]),
        html.Button("Load", id="btn-load", n_clicks=0,
                     style={"height": "36px", "padding": "0 24px", "backgroundColor": ACCENT, "color": "white",
                            "border": "none", "borderRadius": "6px", "cursor": "pointer", "fontWeight": "bold",
                            "fontSize": "14px"}),
    ]),

    # ---- Viewer toolbar ----
    html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "14px", "alignItems": "center",
                     "padding": "6px 12px", "backgroundColor": PANEL_BG, "borderRadius": "8px", "marginBottom": "6px"}, children=[
        html.Div([html.Label("Overlay opacity", style={"fontSize": "11px", "color": "#aaa", "marginRight": "4px"}),
                  dcc.Slider(id="slider-opacity", min=0, max=1, step=0.05, value=0.75,
                             marks={0: "0", 0.5: ".5", 1: "1"}, tooltip={"placement": "bottom"})],
                 style={"width": "180px"}),
        html.Div([dcc.Checklist(id="chk-crosshair", options=[{"label": " Crosshair", "value": "on"}], value=["on"],
                                style={"fontSize": "13px"})]),
        html.Div(id="coord-display", style={"fontSize": "12px", "fontFamily": "Consolas, monospace",
                                             "marginLeft": "auto", "color": "#b0b0ff"}),
        html.Button("Export slices (PNG)", id="btn-export", n_clicks=0,
                     style={"height": "28px", "padding": "0 14px", "backgroundColor": "#e07020", "color": "white",
                            "border": "none", "borderRadius": "4px", "cursor": "pointer", "fontSize": "12px"}),
    ]),

    # ---- Slice sliders ----
    html.Div(style={"display": "flex", "gap": "10px", "marginBottom": "4px"}, children=[
        html.Div([html.Label("Sagittal (X)", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Slider(id="sl-sag", min=0, max=10, step=1, value=5, tooltip={"placement": "bottom"})],
                 style={"flex": "1"}),
        html.Div([html.Label("Coronal (Y)", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Slider(id="sl-cor", min=0, max=10, step=1, value=5, tooltip={"placement": "bottom"})],
                 style={"flex": "1"}),
        html.Div([html.Label("Axial (Z)", style={"fontSize": "11px", "color": "#aaa"}),
                  dcc.Slider(id="sl-axi", min=0, max=10, step=1, value=5, tooltip={"placement": "bottom"})],
                 style={"flex": "1"}),
    ]),

    # ---- Three views ----
    html.Div(style={"display": "flex", "gap": "4px"}, children=[
        html.Div(dcc.Graph(id="fig-sag", config={"displayModeBar": True,
                 "toImageButtonOptions": {"format": "png", "scale": 4, "filename": "sagittal"}},
                 style={"height": "420px"}), style={"flex": "1"}),
        html.Div(dcc.Graph(id="fig-cor", config={"displayModeBar": True,
                 "toImageButtonOptions": {"format": "png", "scale": 4, "filename": "coronal"}},
                 style={"height": "420px"}), style={"flex": "1"}),
        html.Div(dcc.Graph(id="fig-axi", config={"displayModeBar": True,
                 "toImageButtonOptions": {"format": "png", "scale": 4, "filename": "axial"}},
                 style={"height": "420px"}), style={"flex": "1"}),
    ]),

    # ---- Status ----
    html.Div(id="status-bar", style={"fontSize": "12px", "color": "#888", "padding": "4px 0"}),

    # ---- Cluster table ----
    html.Div(style={"marginTop": "10px"}, children=[
        html.H4("Cluster Table", style={"margin": "4px 0", "color": "#e0e0ff"}),
        html.Div("Click a row to navigate to the cluster peak.", style={"fontSize": "11px", "color": "#777", "marginBottom": "4px"}),
        dash_table.DataTable(
            id="cluster-table", columns=[], data=[],
            style_table={"overflowX": "auto", "maxHeight": "300px", "overflowY": "auto"},
            style_header={"backgroundColor": "#16213e", "color": "white", "fontWeight": "bold", "textAlign": "center"},
            style_cell={"backgroundColor": DARK_BG, "color": "white", "border": "1px solid #222",
                        "textAlign": "center", "fontSize": "12px", "padding": "4px 8px"},
            style_data_conditional=[{"if": {"state": "selected"}, "backgroundColor": "#1a3a5f", "border": "1px solid #4a90d9"}],
            row_selectable="single", page_size=15,
        ),
    ]),

    dcc.Download(id="download-png"),
    dcc.Store(id="store-loaded", data=False),
])

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(Output("dd-roi", "options"), Output("dd-roi", "value"),
              Input("dd-specie", "value"), Input("dd-modality", "value"),
              Input("input-datafolder", "value"), Input("input-dataset", "value"))
def cb_roi_options(specie, modality, datafolder, dataset):
    rois = scan_roi_types(datafolder, dataset, modality, specie)
    opts = [{"label": r, "value": r} for r in rois]
    default = DEFAULTS["roi_type"] if DEFAULTS["roi_type"] in rois else (rois[0] if rois else None)
    return opts, default


@app.callback(Output("dd-model", "options"), Output("dd-model", "value"),
              Input("dd-roi", "value"),
              State("input-datafolder", "value"), State("input-dataset", "value"),
              State("dd-modality", "value"), State("dd-specie", "value"))
def cb_model_options(roi, datafolder, dataset, modality, specie):
    if not roi:
        return [], None
    models = scan_available_models(datafolder, dataset, modality, specie, roi)
    opts = [{"label": m, "value": m} for m in models]
    return opts, models[0] if models else None


@app.callback(
    Output("store-loaded", "data"),
    Output("sl-sag", "min"), Output("sl-sag", "max"), Output("sl-sag", "value"),
    Output("sl-cor", "min"), Output("sl-cor", "max"), Output("sl-cor", "value"),
    Output("sl-axi", "min"), Output("sl-axi", "max"), Output("sl-axi", "value"),
    Output("cluster-table", "columns"), Output("cluster-table", "data"),
    Output("status-bar", "children"),
    Input("btn-load", "n_clicks"),
    State("input-datafolder", "value"), State("input-dataset", "value"),
    State("dd-specie", "value"), State("dd-modality", "value"),
    State("dd-roi", "value"), State("dd-model", "value"),
    prevent_initial_call=True,
)
def cb_load(n, datafolder, dataset, specie, modality, roi, model):
    paths = ATLAS_PATHS.get(specie)
    if not paths:
        return [no_update] * 12 + ["Unknown species"]

    # Load atlases
    hi_data, hi_aff, _ = load_nifti(paths["high"])
    lo_data, lo_aff, _ = load_nifti(paths["low"])
    hi_max = np.percentile(hi_data[hi_data > 0], 99.5) if np.any(hi_data > 0) else 1
    hi_data = np.clip(hi_data / hi_max, 0, 1)

    _cache["atlas"] = hi_data
    _cache["hi_affine"] = hi_aff
    _cache["lo_affine"] = lo_aff
    _cache["lo_shape"] = lo_data.shape
    _cache["hi_shape"] = hi_data.shape
    _cache["specie"] = specie
    _cache["overlay"] = None
    _cache["vmin"] = 0
    _cache["vmax"] = 1
    _cache["model"] = model or ""

    table_cols, table_data = [], []
    status_parts = [f"Atlas loaded: {specie} ({hi_data.shape})"]

    if model and roi:
        nifti_path, table_path = get_result_paths(datafolder, dataset, modality, specie, roi, model)
        if os.path.exists(nifti_path):
            ov_data, ov_aff, _ = load_nifti(nifti_path)
            ov_hi = resample_lowres_to_highres(ov_data, ov_aff, hi_data.shape, hi_aff)
            _cache["overlay"] = ov_hi
            nz = ov_hi[np.abs(ov_hi) > 1e-6]
            if len(nz) > 0:
                _cache["vmin"] = float(np.min(nz))
                _cache["vmax"] = float(np.max(nz))
            status_parts.append(f"Overlay: {model}")
        else:
            status_parts.append(f"Overlay not found: {nifti_path}")

        if os.path.exists(table_path):
            df = pd.read_excel(table_path) if table_path.endswith(".xlsx") else pd.read_csv(table_path)
            table_cols = [{"name": str(c), "id": str(c)} for c in df.columns]
            table_data = df.to_dict("records")
            status_parts.append(f"Table: {len(df)} rows")

    sx, sy, sz = hi_data.shape
    return (
        True,
        0, sx-1, sx//2,
        0, sy-1, sy//2,
        0, sz-1, sz//2,
        table_cols, table_data,
        " | ".join(status_parts),
    )


@app.callback(
    Output("fig-sag", "figure"), Output("fig-cor", "figure"), Output("fig-axi", "figure"),
    Output("coord-display", "children"),
    Input("sl-sag", "value"), Input("sl-cor", "value"), Input("sl-axi", "value"),
    Input("slider-opacity", "value"), Input("chk-crosshair", "value"),
    Input("store-loaded", "data"),
)
def cb_update_views(sag, cor, axi, opacity, crosshair, loaded):
    if _cache["atlas"] is None:
        return empty_fig("Sagittal"), empty_fig("Coronal"), empty_fig("Axial"), "Load data to begin"

    show_xh = "on" in (crosshair or [])
    shape = _cache["atlas"].shape

    sag_cross = (cor, shape[2] - 1 - axi) if show_xh else None
    cor_cross = (sag, shape[2] - 1 - axi) if show_xh else None
    axi_cross = (sag, shape[1] - 1 - cor) if show_xh else None

    fig_s = make_slice_fig(0, sag, opacity, show_xh, sag_cross, "Sagittal")
    fig_c = make_slice_fig(1, cor, opacity, show_xh, cor_cross, "Coronal")
    fig_a = make_slice_fig(2, axi, opacity, show_xh, axi_cross, "Axial")

    # Coordinates
    hi_aff = _cache["hi_affine"]
    lo_aff = _cache["lo_affine"]
    world = voxel_to_world((sag, cor, axi), hi_aff)
    lo_vox = world_to_voxel(world, lo_aff)

    txt = (f"Low-res voxel: ({lo_vox[0]}, {lo_vox[1]}, {lo_vox[2]})  |  "
           f"World mm: ({world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f})  |  "
           f"Hi-res voxel: ({sag}, {cor}, {axi})")
    return fig_s, fig_c, fig_a, txt


@app.callback(
    Output("sl-sag", "value", allow_duplicate=True),
    Output("sl-cor", "value", allow_duplicate=True),
    Output("sl-axi", "value", allow_duplicate=True),
    Input("cluster-table", "selected_rows"),
    State("cluster-table", "data"),
    prevent_initial_call=True,
)
def cb_nav_cluster(selected, data):
    if not selected or not data or _cache["hi_affine"] is None:
        return no_update, no_update, no_update
    row = data[selected[0]]
    lo_aff = _cache["lo_affine"]
    hi_aff = _cache["hi_affine"]

    x_col = next((c for c in row if c.lower() in ("x", "vox_x", "peak_x", "i")), None)
    y_col = next((c for c in row if c.lower() in ("y", "vox_y", "peak_y", "j")), None)
    z_col = next((c for c in row if c.lower() in ("z", "vox_z", "peak_z", "k")), None)

    if x_col and y_col and z_col:
        lo_vox = (int(row[x_col]), int(row[y_col]), int(row[z_col]))
        world = voxel_to_world(lo_vox, lo_aff)
        hi_vox = world_to_voxel(world, hi_aff)
        s = _cache["hi_shape"]
        return (
            int(np.clip(hi_vox[0], 0, s[0]-1)),
            int(np.clip(hi_vox[1], 0, s[1]-1)),
            int(np.clip(hi_vox[2], 0, s[2]-1)),
        )
    return no_update, no_update, no_update


@app.callback(
    Output("download-png", "data"),
    Input("btn-export", "n_clicks"),
    State("sl-sag", "value"), State("sl-cor", "value"), State("sl-axi", "value"),
    State("slider-opacity", "value"),
    prevent_initial_call=True,
)
def cb_export(n, sag, cor, axi, opacity):
    if _cache["atlas"] is None:
        return no_update
    try:
        import plotly.io as pio
        from plotly.subplots import make_subplots

        combined = make_subplots(rows=1, cols=3, subplot_titles=["Sagittal", "Coronal", "Axial"],
                                 horizontal_spacing=0.03)
        for i, (axis, idx) in enumerate([(0, sag), (1, cor), (2, axi)]):
            fig = make_slice_fig(axis, idx, opacity, False, None, "")
            for trace in fig.data:
                combined.add_trace(trace, row=1, col=i+1)

        model = _cache.get("model", "result")
        combined.update_layout(
            height=500, width=1500,
            plot_bgcolor="black", paper_bgcolor="white", font_color="black",
            title=dict(text=f"{model}", font=dict(size=14)),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        for ax in ["xaxis", "xaxis2", "xaxis3", "yaxis", "yaxis2", "yaxis3"]:
            combined.update_layout(**{ax: dict(visible=False)})

        img_bytes = pio.to_image(combined, format="png", scale=4, engine="kaleido")
        encoded = base64.b64encode(img_bytes).decode()
        return dict(content=encoded, filename=f"brain_{model}.png", base64=True)
    except Exception:
        return no_update


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Brain Viewer")
    print("  Open http://127.0.0.1:8050 in your browser")
    print("=" * 60)
    app.run(debug=True, port=8050)

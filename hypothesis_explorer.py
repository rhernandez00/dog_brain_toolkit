#!/usr/bin/env python
"""
hypothesis_explorer.py — EmoC RSA hypothesis-tree explorer (standalone Dash app).

Two linked halves on one page:

  1. A **hypothesis tree** you author yourself. Each node is one you name, give
     categories + notes, and link to an RSA model; the connector (link) into each
     node can be named too. The tree is drawn top-down and each node is coloured by
     whether its linked model has results (dog / human / both / none / unlinked).
     Click a node to select it. Add / delete / reorder nodes and Save the tree to
     disk. Keep several named trees (e.g. one that splits by species first, another
     by valence first).

  2. A row of **comparison panels**. Selecting a node loads its model's maps into
     every un-pinned panel; pin a panel to freeze it while you browse other nodes,
     so you can lay maps side by side. Each panel independently shows the Dog brain,
     the Human brain, or Both, and picks the map type (group average / z-map /
     cluster-corrected) — all drawn as 2D atlas slices.

Standalone only (own port, default 8055):
    & "C:\\ProgramData\\anaconda3\\python.exe" hypothesis_explorer.py
    & "C:\\ProgramData\\anaconda3\\python.exe" hypothesis_explorer.py --port 8056

Reuses viz/datasource.py (result resolution), viz/niftiutil.py (atlas + 2D slices),
viz/stimuli.py (stimulus categories) and viz/hypothesis_tree.py (tree model).
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, no_update, callback_context
from dash.dependencies import Input, Output, State

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz import datasource, niftiutil, stimuli, hypothesis_tree as ht
# Reuse the RSA Model Builder's polished rounded-cell matrix renderer so a linked
# model is shown here exactly as it looks in the builder (with its saved style).
import rsa_model_builder as rmb

# --- palette (matches the other viz apps) ---------------------------------
BG, PANEL, INK, MUTED, LINE, ACCENT = "#ffffff", "#f3f5f9", "#222222", "#667085", "#d5dbe5", "#4472C4"
INPUT_STYLE = {"backgroundColor": "#ffffff", "color": INK,
               "border": f"1px solid {LINE}", "borderRadius": "6px", "padding": "5px 8px"}
BTN = {"height": "32px", "padding": "0 14px", "backgroundColor": ACCENT, "color": "white",
       "border": "none", "borderRadius": "6px", "cursor": "pointer", "fontWeight": "bold"}
BTN2 = {**BTN, "backgroundColor": "#eef1f6", "color": INK, "border": f"1px solid {LINE}",
        "fontWeight": "normal"}

DEFAULT_DATASET = "EmoC"
MODALITIES = ["RSA", "GLM"]
MAPTYPES = [("mean", "Group average"), ("z", "Z-map"), ("corrected", "Cluster-corrected")]
AXES = [("0", "Slice X"), ("1", "Slice Y"), ("2", "Slice Z")]
N_PANELS = 3
CORRECTED_ZT_TRIES = [3.1, 2.3, 3.9]

# status -> (colour, human label)
STATUS_STYLE = {
    "both":     ("#1a7f37", "Dog + Human"),
    "D":        ("#3b7dd8", "Dog only"),
    "H":        ("#e08a1e", "Human only"),
    "none":     ("#cf4b4b", "Linked · no results"),
    "unlinked": ("#c9ced6", "No model linked"),
}

# --- caches (module-level; keyed so re-selecting is instant) --------------
_ATLAS = {}          # specie -> (hi, hi_aff, lo_aff, lo_shape)
_ATLAS_ON_GRID = {}  # (specie, shape, aff_hash) -> atlas resampled onto overlay grid
_MAP_CACHE = {}      # (datafolder,dataset,modality,roi,specie,model,maptype,zt) -> (data,aff) or None


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _atlas(specie):
    if specie not in _ATLAS:
        _ATLAS[specie] = niftiutil.load_atlas(specie)
    return _ATLAS[specie]


def _atlas_on_grid(specie, shape, aff):
    key = (specie, tuple(shape), hash(aff.tobytes()))
    if key not in _ATLAS_ON_GRID:
        hi, hi_aff, _lo_aff, _lo_shape = _atlas(specie)
        _ATLAS_ON_GRID[key] = niftiutil.resample_lowres_to_highres(hi, hi_aff, shape, aff)
    return _ATLAS_ON_GRID[key]


def _map_path(datafolder, dataset, modality, roi, specie, model, maptype, zt):
    if maptype == "mean":
        return datasource.mean_path(datafolder, dataset, modality, specie, roi, model)
    if maptype == "z":
        p, _kind = datasource.overlay_path(datafolder, dataset, modality, specie, roi, model)
        return p
    if maptype == "corrected":
        for z in [zt] + CORRECTED_ZT_TRIES + [None]:
            p = datasource.corrected_path(datafolder, dataset, modality, specie, roi, model, z_threshold=z)
            if p:
                return p
    return None


def _load_map(datafolder, dataset, modality, roi, specie, model, maptype, zt):
    key = (datafolder, dataset, modality, roi, specie, model, maptype, round(float(zt), 2))
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    result = None
    if model and roi:
        path = _map_path(datafolder, dataset, modality, roi, specie, model, maptype, zt)
        if path:
            try:
                data, aff, _hdr = niftiutil.load_nifti(path)
                result = (data, aff)
            except Exception:
                result = None
    _MAP_CACHE[key] = result
    return result


def available_models(datafolder, dataset, modality, roi):
    """Union of runnable model CSVs and any model that already has results."""
    models = set()
    folder = os.path.join(datafolder or "", dataset or "", "rsa_models")
    for p in glob.glob(os.path.join(folder, "*.csv")):
        stem = os.path.splitext(os.path.basename(p))[0]
        if not stem.startswith("_"):          # skip _MODEL_BATTERY_MANIFEST etc.
            models.add(stem)
    if roi:
        for sp in ("D", "H"):
            try:
                models |= set(datasource.scan_models(datafolder, dataset, modality, sp, roi))
            except Exception:
                pass
    return sorted(models)


# --- RSA model matrix, drawn with the RSA Model Builder's renderer ---------

def _model_heatmap(datafolder, dataset, model):
    """Return a Plotly figure of the linked model's dissimilarity matrix, rendered
    with rsa_model_builder.build_cell_heatmap and the model's saved _style.json
    (falls back to a compact default so a 40x40 matrix stays readable)."""
    if not model:
        return niftiutil.empty_fig("Link a model to load its matrix.", height=300)
    path = os.path.join(datafolder or "", dataset or "", "rsa_models", f"{model}.csv")
    if not os.path.exists(path):
        return niftiutil.empty_fig(f"No matrix CSV for '{model}'.", height=300)
    try:
        df = pd.read_csv(path, index_col=0)
        labels = [str(x) for x in df.index]
        matrix = rmb.enforce_invariants(df.values.astype(float))
    except Exception as e:
        return niftiutil.empty_fig(f"Matrix unreadable: {e}", height=300)
    sidecar = rmb.load_style_sidecar(path)
    fig_style = sidecar.get("figure_style") if isinstance(sidecar, dict) else None
    if fig_style:
        style = {**rmb.DEFAULT_STYLE, **fig_style}
    else:  # no saved style — compact so the full matrix fits the side panel
        style = {**rmb.DEFAULT_STYLE, "cell_size": 20, "val_font_size": 8, "label_font_size": 9}
    return rmb.build_cell_heatmap(matrix, labels, style)


# ---------------------------------------------------------------------------
# Tree diagram
# ---------------------------------------------------------------------------

def _short(label, n=20):
    label = label or ""
    return label if len(label) <= n else label[: n - 1] + "…"


def tree_figure(tree, selected_id, result_sets):
    root = tree["root"]
    pos = ht.compute_layout(root)
    max_depth = max(d for _n, d, _p in ht.iter_nodes(root))

    edge_x, edge_y = [], []
    link_x, link_y, link_t = [], [], []          # named connectors (links)
    for node, _d, parent in ht.iter_nodes(root):
        if parent is not None:
            x0, y0 = pos[parent["id"]]; x1, y1 = pos[node["id"]]
            edge_x += [x0, x1, None]; edge_y += [-y0, -y1, None]
            if node.get("link"):
                link_x.append((x0 + x1) / 2); link_y.append((-y0 - y1) / 2)
                link_t.append(node["link"])

    node_x, node_y, colors, texts, hovers, line_w, line_c, custom = [], [], [], [], [], [], [], []
    for node, _d, _p in ht.iter_nodes(root):
        x, y = pos[node["id"]]
        st = ht.node_status(node.get("model"), result_sets)
        color, st_label = STATUS_STYLE[st]
        node_x.append(x); node_y.append(-y); colors.append(color)
        texts.append(_short(node.get("label")))
        hovers.append(f"<b>{node.get('label','')}</b><br>model: {node.get('model') or '—'}"
                      f"<br>status: {st_label}"
                      f"<br>categories: {', '.join(node.get('categories') or []) or '—'}")
        sel = node["id"] == selected_id
        line_w.append(3 if sel else 1)
        line_c.append("#111" if sel else "#ffffff")
        custom.append(node["id"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                  line=dict(color="#b7c0d0", width=1.5), hoverinfo="skip", showlegend=False))
    if link_t:
        fig.add_trace(go.Scatter(
            x=link_x, y=link_y, mode="text", text=link_t,
            textfont=dict(size=10, color="#5a6474"), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=texts, textposition="middle right",
        textfont=dict(size=11, color=INK), customdata=custom,
        marker=dict(size=20, color=colors, line=dict(width=line_w, color=line_c)),
        hovertext=hovers, hoverinfo="text", showlegend=False))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        height=max(320, 150 + max_depth * 120),
        xaxis=dict(visible=False, range=[-0.6, max(node_x) + 2.2]),
        yaxis=dict(visible=False))
    return fig


def status_legend():
    items = []
    for st, (color, label) in STATUS_STYLE.items():
        items.append(html.Span([
            html.Span(style={"display": "inline-block", "width": "12px", "height": "12px",
                             "background": color, "borderRadius": "50%", "border": f"1px solid {LINE}",
                             "marginRight": "5px", "verticalAlign": "middle"}),
            html.Span(label, style={"fontSize": "11px", "color": MUTED, "marginRight": "14px"}),
        ]))
    return html.Div(items, style={"marginTop": "4px"})


# ---------------------------------------------------------------------------
# App + layout
# ---------------------------------------------------------------------------

URL_BASE = os.environ.get("EXPLORER_URL_BASE", "/")
app = Dash(__name__, url_base_pathname=URL_BASE, suppress_callback_exceptions=True,
           title="EmoC Hypothesis Explorer")
server = app.server


def _labeled(label, comp):
    return html.Div([html.Label(label, style={"fontSize": "11px", "color": MUTED}), comp])


def top_bar():
    default_folder = datasource.resolve_datafolder(DEFAULT_DATASET)
    return html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "alignItems": "flex-end",
                    "padding": "10px 12px", "backgroundColor": PANEL, "borderRadius": "8px",
                    "border": f"1px solid {LINE}", "marginBottom": "8px"}, children=[
        _labeled("Data folder", dcc.Input(id="ex-datafolder", value=default_folder, type="text",
                 style={**INPUT_STYLE, "width": "230px"})),
        _labeled("Dataset", dcc.Input(id="ex-dataset", value=DEFAULT_DATASET, type="text",
                 style={**INPUT_STYLE, "width": "80px"})),
        _labeled("Modality", dcc.Dropdown(id="ex-modality", options=[{"label": m, "value": m} for m in MODALITIES],
                 value="RSA", clearable=False, style={"width": "90px"})),
        _labeled("ROI / mask", dcc.Dropdown(id="ex-roi", options=[], value=None, style={"width": "190px"})),
        html.Button("Load data", id="ex-load", n_clicks=0, style=BTN),
        html.Div(style={"width": "1px", "height": "34px", "background": LINE, "margin": "0 4px"}),
        _labeled("Tree", dcc.Dropdown(id="ex-tree-select", options=[], value=None, style={"width": "200px"})),
        _labeled("New / Save as name", dcc.Input(id="ex-tree-name", value="my-tree", type="text",
                 style={**INPUT_STYLE, "width": "150px"})),
        html.Button("New", id="ex-tree-new", n_clicks=0, style=BTN2),
        html.Button("Save", id="ex-tree-save", n_clicks=0, style=BTN),
        html.Button("Delete", id="ex-tree-delete", n_clicks=0, style=BTN2),
        html.Span(id="ex-source", style={"fontSize": "11px", "color": MUTED, "marginLeft": "auto"}),
        html.Span(id="ex-tree-msg", style={"fontSize": "11px", "color": "#1a7f37", "width": "100%"}),
    ])


def node_editor():
    return html.Div(style={"backgroundColor": PANEL, "borderRadius": "8px", "padding": "10px 12px",
                    "border": f"1px solid {LINE}"}, children=[
        html.H4("Selected node", style={"margin": "0 0 8px", "color": INK}),
        _labeled("Label", dcc.Input(id="ed-label", type="text", debounce=True,
                 style={**INPUT_STYLE, "width": "100%"})),
        _labeled("Link label (line from parent)", dcc.Input(id="ed-link", type="text", debounce=True,
                 placeholder="e.g. split by species", style={**INPUT_STYLE, "width": "100%"})),
        _labeled("Categories (comma-separated)", dcc.Input(id="ed-categories", type="text", debounce=True,
                 style={**INPUT_STYLE, "width": "100%"})),
        html.Div(f"stimulus codes: {', '.join(stimuli.condition_code(s, l) for s, l in stimuli.stimulus_conditions())}",
                 style={"fontSize": "10px", "color": MUTED, "margin": "2px 0 6px"}),
        _labeled("Notes", dcc.Textarea(id="ed-notes", style={"width": "100%", "height": "48px", **INPUT_STYLE})),
        _labeled("Filter models", dcc.Input(id="ed-model-filter", type="text", debounce=True,
                 placeholder="e.g. val · dog · cross", style={**INPUT_STYLE, "width": "100%"})),
        _labeled("Linked model", dcc.Dropdown(id="ed-model", options=[], value=None, style={"width": "100%"})),
        html.Div([
            html.Button("Apply", id="ed-apply", n_clicks=0, style={**BTN, "marginRight": "6px"}),
            html.Button("+ Child", id="ed-add-child", n_clicks=0, style={**BTN2, "marginRight": "6px"}),
            html.Button("+ Sibling", id="ed-add-sibling", n_clicks=0, style={**BTN2, "marginRight": "6px"}),
            html.Button("Delete", id="ed-delete", n_clicks=0, style={**BTN2, "marginRight": "6px"}),
            html.Button("↑", id="ed-up", n_clicks=0, style={**BTN2, "padding": "0 10px", "marginRight": "6px"}),
            html.Button("↓", id="ed-down", n_clicks=0, style={**BTN2, "padding": "0 10px"}),
        ], style={"margin": "8px 0"}),
        html.Div(id="ed-status", style={"fontSize": "11px", "color": MUTED, "marginBottom": "6px"}),
        html.Details([
            html.Summary("Model matrix (builder view)", style={"cursor": "pointer", "fontSize": "12px"}),
            html.Div([
                html.Button("Load model", id="ed-load-model", n_clicks=0, style={**BTN, "marginRight": "8px"}),
                html.Span(id="ed-matrix-note", style={"fontSize": "11px", "color": MUTED}),
            ], style={"margin": "6px 0"}),
            html.Div(dcc.Graph(id="ed-matrix-graph", figure=niftiutil.empty_fig(height=260),
                     config={"displayModeBar": False}),
                     style={"maxHeight": "540px", "overflowY": "auto"}),
        ], open=False),
    ])


def panel(i):
    return html.Div(id=f"pl-{i}-block", style={"flex": "1 1 340px", "minWidth": "320px",
                    "backgroundColor": PANEL, "borderRadius": "8px", "padding": "8px 10px",
                    "border": f"1px solid {LINE}"}, children=[
        html.Div(style={"display": "flex", "gap": "6px", "alignItems": "center", "flexWrap": "wrap"}, children=[
            html.B(f"Panel {i + 1}", style={"color": INK}),
            dcc.Checklist(id=f"pl-{i}-enable", options=[{"label": " on", "value": "on"}],
                          value=(["on"] if i == 0 else []), style={"fontSize": "12px"}),
            dcc.Checklist(id=f"pl-{i}-pin", options=[{"label": " 📌pin", "value": "pin"}],
                          value=[], style={"fontSize": "12px"}),
        ]),
        html.Div(style={"display": "flex", "gap": "6px", "flexWrap": "wrap", "margin": "6px 0"}, children=[
            dcc.Dropdown(id=f"pl-{i}-species", options=[{"label": "Dog", "value": "D"},
                         {"label": "Human", "value": "H"}, {"label": "Both", "value": "B"}],
                         value="D", clearable=False, style={"width": "90px"}),
            dcc.Dropdown(id=f"pl-{i}-maptype", options=[{"label": l, "value": v} for v, l in MAPTYPES],
                         value="z", clearable=False, style={"width": "150px"}),
            dcc.Dropdown(id=f"pl-{i}-axis", options=[{"label": l, "value": v} for v, l in AXES],
                         value="2", clearable=False, style={"width": "95px"}),
        ]),
        dcc.Dropdown(id=f"pl-{i}-model", options=[], value=None, placeholder="model…",
                     style={"width": "100%", "marginBottom": "6px"}),
        html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center"}, children=[
            html.Span("slice", style={"fontSize": "11px", "color": MUTED}),
            html.Div(dcc.Slider(id=f"pl-{i}-frac", min=0, max=1, step=0.02, value=0.5,
                     marks=None, tooltip={"placement": "bottom"}), style={"flex": "1"}),
        ]),
        html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center"}, children=[
            html.Span("z ≥", style={"fontSize": "11px", "color": MUTED}),
            html.Div(dcc.Slider(id=f"pl-{i}-zt", min=0, max=8, step=0.1, value=3.1,
                     marks={0: "0", 3.1: "3.1", 8: "8"}, tooltip={"placement": "bottom"}),
                     style={"flex": "1"}),
        ]),
        html.Div(id=f"pl-{i}-note", style={"fontSize": "11px", "color": ACCENT, "minHeight": "16px"}),
        dcc.Graph(id=f"pl-{i}-dog", style={"height": "230px"}),
        dcc.Graph(id=f"pl-{i}-hum", style={"height": "230px"}),
    ])


app.layout = html.Div(style={"backgroundColor": BG, "color": INK, "minHeight": "100vh",
                      "padding": "10px 14px", "fontFamily": "'Segoe UI', Arial, sans-serif"}, children=[
    html.H2("EmoC Hypothesis Explorer", style={"textAlign": "center", "margin": "4px 0 8px"}),
    top_bar(),
    html.Div(style={"display": "flex", "gap": "10px", "alignItems": "flex-start", "marginBottom": "10px"}, children=[
        html.Div(style={"flex": "2 1 640px", "backgroundColor": PANEL, "borderRadius": "8px",
                 "padding": "8px 10px", "border": f"1px solid {LINE}"}, children=[
            html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "10px"}, children=[
                html.H4("Hypothesis tree", style={"margin": "0 0 4px", "color": INK}),
                html.Span(id="ex-tree-title", style={"fontSize": "12px", "color": MUTED}),
            ]),
            status_legend(),
            dcc.Graph(id="ex-tree-graph", config={"displayModeBar": False}),
        ]),
        html.Div(style={"flex": "1 1 300px", "maxWidth": "360px"}, children=[node_editor()]),
    ]),
    html.Div(style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
             children=[panel(i) for i in range(N_PANELS)]),

    dcc.Store(id="ex-tree-store"),
    dcc.Store(id="ex-selected"),
    dcc.Store(id="ex-dataver", data=0),
    dcc.Store(id="ex-models", data=[]),
])


# ---------------------------------------------------------------------------
# Callbacks — data source
# ---------------------------------------------------------------------------

@app.callback(Output("ex-roi", "options"), Output("ex-roi", "value"), Output("ex-source", "children"),
              Input("ex-modality", "value"), Input("ex-datafolder", "value"), Input("ex-dataset", "value"))
def cb_rois(modality, datafolder, dataset):
    rois = set()
    for sp in ("D", "H"):
        try:
            rois.update(datasource.scan_roi_types(datafolder, dataset, modality, sp))
        except Exception:
            pass
    rois = sorted(rois)
    return ([{"label": r, "value": r} for r in rois], (rois[0] if rois else None),
            datasource.describe_source(dataset))


@app.callback(Output("ex-dataver", "data"), Output("ex-models", "data"),
              Input("ex-load", "n_clicks"),
              State("ex-datafolder", "value"), State("ex-dataset", "value"),
              State("ex-modality", "value"), State("ex-roi", "value"), State("ex-dataver", "data"))
def cb_load(_n, datafolder, dataset, modality, roi, ver):
    _MAP_CACHE.clear()
    models = available_models(datafolder, dataset, modality, roi)
    return (ver or 0) + 1, models


@app.callback(Output("ed-model", "options"), Output("pl-0-model", "options"),
              Output("pl-1-model", "options"), Output("pl-2-model", "options"),
              Input("ex-models", "data"), Input("ed-model-filter", "value"))
def cb_model_options(models, filt):
    models = models or []
    opts_all = [{"label": m, "value": m} for m in models]
    if filt:
        f = filt.lower()
        ed_opts = [o for o in opts_all if f in o["value"].lower()]
    else:
        ed_opts = opts_all
    return ed_opts, opts_all, opts_all, opts_all


# ---------------------------------------------------------------------------
# Callbacks — tree management (dropdown + persistence)
# ---------------------------------------------------------------------------

@app.callback(Output("ex-tree-select", "options"),
              Input("ex-datafolder", "value"), Input("ex-dataset", "value"), Input("ex-tree-msg", "children"))
def cb_tree_list(datafolder, dataset, _msg):
    names = ht.list_trees(ht.trees_dir(datafolder, dataset))
    return [{"label": n, "value": n} for n in names]


@app.callback(Output("ex-tree-store", "data", allow_duplicate=True),
              Output("ex-selected", "data", allow_duplicate=True),
              Output("ex-tree-msg", "children", allow_duplicate=True),
              Input("ex-tree-select", "value"),
              State("ex-datafolder", "value"), State("ex-dataset", "value"),
              prevent_initial_call=True)
def cb_tree_open(name, datafolder, dataset):
    if not name:
        return no_update, no_update, no_update
    try:
        tree = ht.load_tree(ht.trees_dir(datafolder, dataset), name)
    except Exception as e:
        return no_update, no_update, f"⚠ could not open '{name}': {e}"
    return tree, tree["root"]["id"], f"opened '{name}'"


@app.callback(Output("ex-tree-store", "data", allow_duplicate=True),
              Output("ex-selected", "data", allow_duplicate=True),
              Output("ex-tree-msg", "children", allow_duplicate=True),
              Input("ex-tree-new", "n_clicks"),
              State("ex-tree-name", "value"), State("ex-dataset", "value"),
              prevent_initial_call=True)
def cb_tree_new(_n, name, dataset):
    root_cats = [stimuli.condition_code(s, l) for s, l in stimuli.stimulus_conditions()]
    tree = ht.new_tree(name or "my-tree", dataset or DEFAULT_DATASET, root_categories=root_cats)
    return tree, tree["root"]["id"], f"new tree '{tree['name']}' (unsaved — press Save)"


@app.callback(Output("ex-tree-msg", "children", allow_duplicate=True),
              Input("ex-tree-save", "n_clicks"),
              State("ex-tree-store", "data"), State("ex-tree-name", "value"),
              State("ex-datafolder", "value"), State("ex-dataset", "value"),
              prevent_initial_call=True)
def cb_tree_save(_n, tree, name, datafolder, dataset):
    if not tree:
        return "⚠ nothing to save — create or open a tree first"
    if name:
        tree = dict(tree, name=name)
    try:
        path = ht.save_tree(ht.trees_dir(datafolder, dataset), tree)
    except Exception as e:
        return f"⚠ save failed: {e}"
    return f"saved → {path}"


@app.callback(Output("ex-tree-msg", "children", allow_duplicate=True),
              Input("ex-tree-delete", "n_clicks"),
              State("ex-tree-select", "value"),
              State("ex-datafolder", "value"), State("ex-dataset", "value"),
              prevent_initial_call=True)
def cb_tree_delete(_n, name, datafolder, dataset):
    if not name:
        return "⚠ pick a saved tree to delete"
    ht.delete_tree(ht.trees_dir(datafolder, dataset), name)
    return f"deleted '{name}'"


# ---------------------------------------------------------------------------
# Callbacks — node edits + structure (single writer of tree-store)
# ---------------------------------------------------------------------------

@app.callback(Output("ex-tree-store", "data", allow_duplicate=True),
              Output("ex-selected", "data", allow_duplicate=True),
              Input("ed-apply", "n_clicks"), Input("ed-add-child", "n_clicks"),
              Input("ed-add-sibling", "n_clicks"), Input("ed-delete", "n_clicks"),
              Input("ed-up", "n_clicks"), Input("ed-down", "n_clicks"),
              State("ex-tree-store", "data"), State("ex-selected", "data"),
              State("ed-label", "value"), State("ed-link", "value"),
              State("ed-categories", "value"),
              State("ed-notes", "value"), State("ed-model", "value"),
              prevent_initial_call=True)
def cb_node_action(_a, _c, _s, _d, _u, _dn, tree, sel_id, label, link, categories, notes, model):
    if not tree or not sel_id:
        return no_update, no_update
    trig = callback_context.triggered[0]["prop_id"].split(".")[0]
    root = tree["root"]
    node = ht.find_node(root, sel_id)
    if node is None:
        return no_update, no_update
    new_sel = no_update
    if trig == "ed-apply":
        node["label"] = (label or "").strip() or node["label"]
        node["link"] = (link or "").strip()
        node["categories"] = [c.strip() for c in (categories or "").split(",") if c.strip()]
        node["notes"] = notes or ""
        node["model"] = model or None
    elif trig == "ed-add-child":
        child = ht.add_child(root, sel_id, ht.new_node("New node"))
        new_sel = child["id"] if child else no_update
    elif trig == "ed-add-sibling":
        sib = ht.add_sibling(root, sel_id, ht.new_node("New node"))
        new_sel = sib["id"] if sib else no_update
    elif trig == "ed-delete":
        parent = ht.find_parent(root, sel_id)
        if parent is not None and ht.delete_node(root, sel_id):
            new_sel = parent["id"]
    elif trig in ("ed-up", "ed-down"):
        ht.move_node(root, sel_id, -1 if trig == "ed-up" else 1)
    return tree, new_sel


@app.callback(Output("ex-selected", "data", allow_duplicate=True),
              Input("ex-tree-graph", "clickData"), prevent_initial_call=True)
def cb_node_click(click):
    if click and click.get("points"):
        cd = click["points"][0].get("customdata")
        if cd:
            return cd
    return no_update


# ---------------------------------------------------------------------------
# Callbacks — render tree + editor form
# ---------------------------------------------------------------------------

@app.callback(Output("ex-tree-graph", "figure"), Output("ex-tree-title", "children"),
              Input("ex-tree-store", "data"), Input("ex-selected", "data"), Input("ex-dataver", "data"),
              State("ex-datafolder", "value"), State("ex-dataset", "value"),
              State("ex-modality", "value"), State("ex-roi", "value"))
def cb_render_tree(tree, sel_id, _ver, datafolder, dataset, modality, roi):
    if not tree:
        fig = niftiutil.empty_fig("Create or open a tree", height=320)
        fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL)
        return fig, ""
    result_sets = ht.models_with_results(datafolder, dataset, modality, roi) if roi else {"D": set(), "H": set()}
    title = f"{tree.get('name', '')} · {sum(1 for _ in ht.iter_nodes(tree['root']))} nodes"
    if tree.get("notes"):
        title += f" · {tree['notes']}"
    return tree_figure(tree, sel_id, result_sets), title


@app.callback(Output("ed-label", "value"), Output("ed-link", "value"),
              Output("ed-categories", "value"), Output("ed-notes", "value"),
              Output("ed-model", "value"), Output("ed-status", "children"),
              Input("ex-selected", "data"), Input("ex-tree-store", "data"), Input("ex-dataver", "data"),
              State("ex-datafolder", "value"), State("ex-dataset", "value"),
              State("ex-modality", "value"), State("ex-roi", "value"))
def cb_editor_fill(sel_id, tree, _ver, datafolder, dataset, modality, roi):
    if not tree or not sel_id:
        return "", "", "", "", None, "Select a node in the tree."
    node = ht.find_node(tree["root"], sel_id)
    if node is None:
        return "", "", "", "", None, "Node not found."
    result_sets = ht.models_with_results(datafolder, dataset, modality, roi) if roi else {"D": set(), "H": set()}
    st = ht.node_status(node.get("model"), result_sets)
    _color, st_label = STATUS_STYLE[st]
    status = f"status: {st_label}" + (f"  ·  model: {node['model']}" if node.get("model") else "")
    return (node.get("label", ""), node.get("link", ""), ", ".join(node.get("categories") or []),
            node.get("notes", ""), node.get("model"), status)


@app.callback(Output("ed-matrix-graph", "figure"), Output("ed-matrix-note", "children"),
              Input("ed-load-model", "n_clicks"),
              State("ex-selected", "data"), State("ex-tree-store", "data"),
              State("ex-datafolder", "value"), State("ex-dataset", "value"),
              prevent_initial_call=True)
def cb_load_model_matrix(_n, sel_id, tree, datafolder, dataset):
    if not tree or not sel_id:
        return no_update, "select a node first"
    node = ht.find_node(tree["root"], sel_id)
    model = node.get("model") if node else None
    if not model:
        return niftiutil.empty_fig("no model linked", height=260), "link a model to this node first"
    return _model_heatmap(datafolder, dataset, model), f"loaded {model}"


# ---------------------------------------------------------------------------
# Callbacks — sync un-pinned panel models to the selected node
# ---------------------------------------------------------------------------

@app.callback(Output("pl-0-model", "value"), Output("pl-1-model", "value"), Output("pl-2-model", "value"),
              Input("ex-selected", "data"), Input("ex-tree-store", "data"),
              State("pl-0-pin", "value"), State("pl-1-pin", "value"), State("pl-2-pin", "value"))
def cb_sync_panel_models(sel_id, tree, pin0, pin1, pin2):
    model = None
    if tree and sel_id:
        node = ht.find_node(tree["root"], sel_id)
        model = node.get("model") if node else None
    pins = [pin0, pin1, pin2]
    return tuple((no_update if ("pin" in (p or [])) else model) for p in pins)


# ---------------------------------------------------------------------------
# Callbacks — panel rendering (one per panel)
# ---------------------------------------------------------------------------

def _panel_species_fig(datafolder, dataset, modality, roi, specie, model, maptype, axis, frac, zt):
    loaded = _load_map(datafolder, dataset, modality, roi, specie, model, maptype, zt)
    label = {"D": "Dog", "H": "Human"}[specie]
    if loaded is None:
        return niftiutil.empty_fig(f"{label}: no {maptype} map", height=230), 0
    data, aff = loaded
    atlas = _atlas_on_grid(specie, data.shape, aff)
    ax = int(axis)
    idx = int(round(float(frac) * (data.shape[ax] - 1)))
    nz = data[np.abs(data) > 1e-6]
    if maptype == "z":
        thr = float(zt)
        vmin, vmax = thr, float(np.max(np.abs(nz))) if nz.size else thr + 1
    else:
        thr = 1e-6
        vmin, vmax = 0.0, float(np.max(np.abs(nz))) if nz.size else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    supra = int(np.sum(np.abs(data) >= thr))
    fig = niftiutil.make_slice_fig(atlas, data, ax, idx, opacity=0.8, z_threshold=thr,
                                   vmin=vmin, vmax=vmax, title=f"{label} · {model}", height=230)
    return fig, supra


def _register_panel(i):
    @app.callback(
        Output(f"pl-{i}-dog", "figure"), Output(f"pl-{i}-hum", "figure"),
        Output(f"pl-{i}-dog", "style"), Output(f"pl-{i}-hum", "style"),
        Output(f"pl-{i}-block", "style"), Output(f"pl-{i}-note", "children"),
        Input(f"pl-{i}-enable", "value"), Input(f"pl-{i}-species", "value"),
        Input(f"pl-{i}-model", "value"), Input(f"pl-{i}-maptype", "value"),
        Input(f"pl-{i}-axis", "value"), Input(f"pl-{i}-frac", "value"), Input(f"pl-{i}-zt", "value"),
        Input("ex-dataver", "data"),
        State("ex-datafolder", "value"), State("ex-dataset", "value"),
        State("ex-modality", "value"), State("ex-roi", "value"))
    def _cb(enable, species, model, maptype, axis, frac, zt, _ver,
            datafolder, dataset, modality, roi):
        base = {"flex": "1 1 340px", "minWidth": "320px", "backgroundColor": PANEL,
                "borderRadius": "8px", "padding": "8px 10px", "border": f"1px solid {LINE}"}
        gshow = {"height": "230px"}
        ghide = {"display": "none"}
        if "on" not in (enable or []):
            return (no_update, no_update, ghide, ghide, {**base, "display": "none"}, "")
        if not model:
            empty = niftiutil.empty_fig("select a node or model", height=230)
            return (empty, empty, gshow, ghide, base, "no model")
        show_d = species in ("D", "B")
        show_h = species in ("H", "B")
        dog_fig = hum_fig = niftiutil.empty_fig(height=230)
        note = []
        if show_d:
            dog_fig, nd = _panel_species_fig(datafolder, dataset, modality, roi, "D",
                                             model, maptype, axis, frac, zt)
            note.append(f"D:{nd}vx")
        if show_h:
            hum_fig, nh = _panel_species_fig(datafolder, dataset, modality, roi, "H",
                                             model, maptype, axis, frac, zt)
            note.append(f"H:{nh}vx")
        return (dog_fig, hum_fig, gshow if show_d else ghide, gshow if show_h else ghide,
                base, "  ".join(note))
    return _cb


for _i in range(N_PANELS):
    _register_panel(_i)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="EmoC RSA hypothesis-tree explorer")
    ap.add_argument("--port", type=int, default=int(os.environ.get("EXPLORER_PORT", "8055")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(f"[hypothesis_explorer] open http://{args.host}:{args.port}")
    app.run(debug=args.debug, use_reloader=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

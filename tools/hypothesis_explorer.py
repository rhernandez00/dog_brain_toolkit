#!/usr/bin/env python
"""
hypothesis_explorer.py — EmoC RSA hypothesis-tree explorer (standalone Dash app).

The tree is **auto-generated** from the dataset's ``rsa_models`` folder — you no
longer author it. On start-up the app scans that folder for model CSVs and builds
a read-only tree with one node per model *stem* (the part before ``__{grouping}``,
e.g. emo-id, val3, all-categories_bipolar) under a single root. The
``_MODEL_BATTERY_MANIFEST.csv`` (when present) only supplies curated ordering and
per-model descriptions; models absent from it still appear, driven by the files on
disk. Two linked halves on one page:

  1. The **hypothesis tree** (read-only). Each node is one hypothesis; the
     **Grouping** toggle in the top bar is the main branching axis
     (collapse / within / cross / dog / hum): it decides which concrete
     "{hypothesis}__{grouping}" model each node resolves to, so flipping it
     re-colours the whole tree and re-loads every un-pinned panel at once. Each
     node is coloured by whether its resolved model has results (dog / human /
     both / none). Click a node to select it — the **Selected node** panel then
     shows its label, grouping, resolved model, the manifest **description** as
     notes, and the model's dissimilarity matrix rendered with the RSA Model
     Builder's viewer.

  2. A row of **comparison panels**. Selecting a node loads its model's maps into
     every un-pinned panel; pin a panel to freeze it while you browse other
     nodes, so you can lay maps side by side. Each panel independently shows the
     Dog brain, the Human brain, or Both, and picks the map type (group average /
     z-map / cluster-corrected) — all drawn as 2D atlas slices.

Standalone only (own port, default 8055):
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\hypothesis_explorer.py
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\hypothesis_explorer.py --port 8056

Reuses viz/datasource.py (result resolution), viz/niftiutil.py (atlas + 2D slices)
and viz/hypothesis_tree.py (tree model / traversal / status).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, no_update
from dash.dependencies import Input, Output, State

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)  # tools/ lives one level below the repo root
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from viz import datasource, niftiutil, hypothesis_tree as ht
from scheduler.paths import get_paths   # canonical model home (pipeline data disk)
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


# ---------------------------------------------------------------------------
# The model battery — grouping is the main branching axis
# ---------------------------------------------------------------------------
# Battery models are named "{hypothesis}__{grouping}.csv"; the trailing token
# after "__" is one of GROUPINGS. A tree node links to a *stem* (e.g. "emo-id" or
# "all-categories_bipolar"); the global Grouping toggle decides which concrete
# "{stem}__{grouping}" model is loaded into the tree colouring and panels.
# Suffix-less models (e.g. "agent-species-id") are grouping-agnostic and resolve
# to themselves under every grouping. The set of models is discovered by scanning
# the dataset's rsa_models folder; the manifest CSV, when present, only enriches
# them with curated ordering and descriptions.

GROUPINGS = ["collapse", "within", "cross", "dog", "hum"]
GROUPING_DESC = {
    "collapse": "collapsed across agent species (Dog/Hum pooled)",
    "within":   "within agent species only (Dog-Dog & Hum-Hum)",
    "cross":    "cross agent species only (Dog-Hum) — agent-invariant test",
    "dog":      "Dog-shown block only",
    "hum":      "Hum-shown block only",
}
DEFAULT_GROUPING = "collapse"

_MANIFEST_CACHE = {}


def _manifest(datafolder, dataset):
    """{model_name: {'hypothesis','grouping','description'}} from the battery
    manifest CSV, if present. Cached per (datafolder, dataset); empty if absent.
    Insertion order follows the CSV rows, so hypotheses stay in manifest order."""
    key = (datafolder, dataset)
    if key not in _MANIFEST_CACHE:
        out = {}
        for folder in _model_dirs(datafolder, dataset):    # first manifest found wins
            path = os.path.join(folder, "_MODEL_BATTERY_MANIFEST.csv")
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            for _, row in df.iterrows():
                out[str(row["model"])] = {
                    "hypothesis": str(row.get("hypothesis", "") or ""),
                    "grouping": str(row.get("grouping", "") or ""),
                    "description": str(row.get("description", "") or ""),
                }
            break
        _MANIFEST_CACHE[key] = out
    return _MANIFEST_CACHE[key]


# Non-model CSVs that live in the rsa_models folder but aren't RSA matrices.
_NON_MODEL_CSVS = {"_MODEL_BATTERY_MANIFEST.csv"}


def _model_dirs(datafolder, dataset):
    """``rsa_models`` folders to search for model CSVs / the manifest: the active
    results root first, then the canonical pipeline data disk where
    ``build_rsa_models.py`` writes (``P:\\userdata\\raulh87\\data`` on Windows /
    the network mount on Linux), de-duplicated. The pipeline-disk entry is what
    lets models authored there (e.g. the ``all-categories_*`` battery) show up in
    the explorer even when results are being viewed from the Google Drive mirror,
    which may not have those CSVs synced yet."""
    dirs = []
    if datafolder:
        dirs.append(os.path.join(datafolder, dataset or "", "rsa_models"))
    try:
        dirs.append(os.path.join(get_paths()[0], dataset or "", "rsa_models"))
    except Exception:
        pass
    seen, out = set(), []
    for d in dirs:
        key = os.path.normcase(os.path.abspath(d))
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _find_model_csv(datafolder, dataset, model):
    """Absolute path to ``{model}.csv`` in the first model dir that has it, else
    None. Lets the matrix renderer read a CSV that only exists on the pipeline
    disk while results are viewed from the Drive mirror."""
    if not model:
        return None
    for folder in _model_dirs(datafolder, dataset):
        path = os.path.join(folder, f"{model}.csv")
        if os.path.isfile(path):
            return path
    return None


def scan_model_csvs(datafolder, dataset):
    """Every RSA model name (``.csv`` stem) physically present at the top level of
    any of the dataset's ``rsa_models`` folders (results root + pipeline disk),
    sorted and de-duplicated. Skips the manifest, any file whose name starts with
    ``_``, and subfolders (e.g. ``by_class/``)."""
    names = set()
    for folder in _model_dirs(datafolder, dataset):
        try:
            entries = os.listdir(folder)
        except Exception:
            continue
        for fn in entries:
            if not fn.lower().endswith(".csv") or fn.startswith("_") or fn in _NON_MODEL_CSVS:
                continue
            if os.path.isfile(os.path.join(folder, fn)):
                names.add(fn[:-4])   # strip ".csv"
    return sorted(names)


def battery_models(datafolder, dataset):
    """Every RSA model available for the dataset: the union of the model CSVs found
    in the ``rsa_models`` folder and any listed in the battery manifest, sorted.
    Folder-driven, so hand-built models (e.g. ``all-categories_*``) auto-populate
    even when they are absent from the manifest."""
    names = set(scan_model_csvs(datafolder, dataset))
    names.update(_manifest(datafolder, dataset).keys())
    return sorted(names)


def model_description(datafolder, dataset, model):
    """Description for a concrete model. Prefers the manifest 'description' (incl.
    its grouping clause); for folder-only models with a ``__{grouping}`` suffix,
    synthesises ``"{stem} | {grouping description}"`` so the notes stay useful."""
    if not model:
        return ""
    desc = _manifest(datafolder, dataset).get(model, {}).get("description", "")
    if desc:
        return desc
    stem, grp = split_model(model)
    if grp:
        return f"{stem} | {GROUPING_DESC.get(grp, grp)}"
    return ""


def split_model(name):
    """(hypothesis, grouping) for a model filename stem. The grouping is the
    trailing '__{grouping}' token when it is one of GROUPINGS, else None."""
    if not name:
        return name, None
    base, sep, suffix = name.rpartition("__")
    if sep and suffix in GROUPINGS:
        return base, suffix
    return name, None


def hypothesis_index(models):
    """{hypothesis: {grouping_or_None: model_name}} over the given model names."""
    idx = {}
    for name in (models or []):
        hyp, grp = split_model(name)
        idx.setdefault(hyp, {})[grp] = name
    return idx


def resolve_model(stored, grouping, idx):
    """Concrete model name for a node's stored value under the active grouping.
    `stored` may be a bare hypothesis or a legacy full "{hyp}__{grouping}" name."""
    if not stored:
        return None
    hyp, _grp = split_model(stored)               # normalise any legacy full name
    variants = idx.get(hyp)
    if not variants:
        return stored                              # unknown model — use as-is
    if grouping in variants:
        return variants[grouping]
    if None in variants:                           # grouping-agnostic (agent-species-id)
        return variants[None]
    return f"{hyp}__{grouping}"                     # no such variant — canonical name


def ordered_hypotheses(datafolder, dataset):
    """Unique model stems to show as tree nodes: manifest hypotheses first (in CSV
    row order, keeping the battery's curated order), then any other stems
    discovered by scanning the folder, sorted. A stem is ``split_model(name)[0]``."""
    hyps, seen = [], set()
    for meta in _manifest(datafolder, dataset).values():        # 1. curated battery order
        h = meta.get("hypothesis") or ""
        if h and h not in seen:
            seen.add(h)
            hyps.append(h)
    extra = set()                                               # 2. folder-only stems
    for name in scan_model_csvs(datafolder, dataset):
        stem = split_model(name)[0]
        if stem and stem not in seen:
            extra.add(stem)
    hyps.extend(sorted(extra))
    return hyps


def build_auto_tree(datafolder, dataset):
    """Read-only tree auto-generated by scanning the dataset's rsa_models folder: a
    single root with one child per model stem (manifest hypotheses first, in CSV
    order, then any other stems found on disk). Each child stores the *bare* stem in
    ``model``; the global Grouping toggle resolves the concrete "{stem}__{grouping}"
    model. Empty (root only) if the folder holds no model CSVs."""
    tree = ht.new_tree(f"{dataset} RSA battery", dataset,
                       notes="auto-generated by scanning the rsa_models folder")
    tree["root"]["label"] = f"{dataset} RSA battery"
    tree["root"]["model"] = None
    hyps = ordered_hypotheses(datafolder, dataset)
    short = {h: (model_description(datafolder, dataset,
                 next((m for m in battery_models(datafolder, dataset)
                       if split_model(m)[0] == h), h)).split(" | ")[0].strip())
             for h in hyps}
    for h in hyps:
        tree["root"]["children"].append(
            ht.new_node(label=h, model=h, notes=short.get(h, "")))
    return tree


# --- RSA model matrix, drawn with the RSA Model Builder's renderer ---------

def _model_heatmap(datafolder, dataset, model):
    """Return a Plotly figure of the linked model's dissimilarity matrix, rendered
    with rsa_model_builder.build_cell_heatmap and the model's saved _style.json
    (falls back to a compact default so a 40x40 matrix stays readable)."""
    if not model:
        return niftiutil.empty_fig("Select a hypothesis node to load its matrix.", height=300)
    path = _find_model_csv(datafolder, dataset, model)
    if not path:
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
# Tree diagram (drawn left-to-right: root at the left, hypotheses stacked down)
# ---------------------------------------------------------------------------

def _short(label, n=28):
    label = label or ""
    return label if len(label) <= n else label[: n - 1] + "…"


def tree_figure(tree, selected_id, result_sets, grouping, idx):
    root = tree["root"]
    pos = ht.compute_layout(root)                 # {id: (leaf_x, depth)}
    leaves = [lx for (lx, _d) in pos.values()]
    leaf_span = (max(leaves) - min(leaves)) if leaves else 0
    max_depth = max(d for _n, d, _p in ht.iter_nodes(root))

    def XY(node_id):                              # x = depth (L->R), y = -leaf position
        lx, d = pos[node_id]
        return float(d), -float(lx)

    edge_x, edge_y = [], []
    link_x, link_y, link_t = [], [], []           # named connectors (links)
    for node, _d, parent in ht.iter_nodes(root):
        if parent is not None:
            x0, y0 = XY(parent["id"]); x1, y1 = XY(node["id"])
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
            if node.get("link"):
                link_x.append((x0 + x1) / 2); link_y.append((y0 + y1) / 2)
                link_t.append(node["link"])

    node_x, node_y, colors, texts, hovers, line_w, line_c, custom = [], [], [], [], [], [], [], []
    for node, _d, _p in ht.iter_nodes(root):
        x, y = XY(node["id"])
        hyp, _g = split_model(node.get("model") or "")
        resolved = resolve_model(node.get("model"), grouping, idx)
        st = ht.node_status(resolved, result_sets)
        color, st_label = STATUS_STYLE[st]
        node_x.append(x); node_y.append(y); colors.append(color)
        texts.append(_short(node.get("label")))
        hovers.append(f"<b>{node.get('label','')}</b><br>hypothesis: {hyp or '—'}"
                      f"<br>grouping: {grouping}<br>model: {resolved or '—'}"
                      f"<br>status: {st_label}")
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
        textfont=dict(size=12, color=INK), customdata=custom,
        marker=dict(size=20, color=colors, line=dict(width=line_w, color=line_c)),
        hovertext=hovers, hoverinfo="text", showlegend=False))
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor=PANEL, plot_bgcolor=PANEL,
        height=max(360, 90 + int(round(leaf_span)) * 46),
        xaxis=dict(visible=False, range=[-0.4, max_depth + 2.4]),
        yaxis=dict(visible=False, range=[-leaf_span - 0.6, 0.6]))
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


def status_dot(color):
    return html.Span(style={"display": "inline-block", "width": "10px", "height": "10px",
                            "background": color, "borderRadius": "50%",
                            "border": f"1px solid {LINE}", "marginRight": "5px",
                            "verticalAlign": "middle"})


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
        html.Button("Reload results", id="ex-reload", n_clicks=0, style=BTN2),
        html.Div(style={"width": "1px", "height": "34px", "background": LINE, "margin": "0 4px"}),
        _labeled("Grouping (main branch)", dcc.Dropdown(id="ex-grouping",
                 options=[{"label": g, "value": g, "title": GROUPING_DESC[g]} for g in GROUPINGS],
                 value=DEFAULT_GROUPING, clearable=False, style={"width": "150px"})),
        html.Span(id="ex-source", style={"fontSize": "11px", "color": MUTED, "marginLeft": "auto"}),
    ])


def node_readout():
    """Read-only panel: everything is derived from the selected node's model."""
    return html.Div(style={"backgroundColor": PANEL, "borderRadius": "8px", "padding": "10px 12px",
                    "border": f"1px solid {LINE}"}, children=[
        html.H4("Selected node", style={"margin": "0 0 8px", "color": INK}),
        html.Div(id="ed-title", style={"fontSize": "15px", "fontWeight": "bold", "color": INK,
                 "marginBottom": "3px"}),
        html.Div(id="ed-status", style={"fontSize": "12px", "color": MUTED, "marginBottom": "8px"}),
        html.Div("Notes", style={"fontSize": "11px", "color": MUTED}),
        html.Div(id="ed-notes-text", style={"fontSize": "12px", "color": INK, "whiteSpace": "pre-wrap",
                 "background": "#ffffff", "border": f"1px solid {LINE}", "borderRadius": "6px",
                 "padding": "6px 8px", "margin": "2px 0 10px", "minHeight": "42px"}),
        html.Div([
            html.Span("Model matrix (builder view)", style={"fontSize": "12px", "color": MUTED}),
            html.Span(id="ed-matrix-note", style={"fontSize": "11px", "color": ACCENT, "marginLeft": "8px"}),
        ], style={"margin": "2px 0 6px"}),
        html.Div(dcc.Graph(id="ed-matrix-graph", figure=niftiutil.empty_fig(height=260),
                 config={"displayModeBar": False}),
                 style={"maxHeight": "540px", "overflowY": "auto"}),
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
        html.Div(style={"flex": "1 1 300px", "maxWidth": "360px"}, children=[node_readout()]),
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


@app.callback(Output("ex-tree-store", "data"), Output("ex-selected", "data"), Output("ex-models", "data"),
              Input("ex-datafolder", "value"), Input("ex-dataset", "value"))
def cb_build_tree(datafolder, dataset):
    """Auto-build the read-only tree from the battery manifest (fires on load)."""
    _MANIFEST_CACHE.pop((datafolder, dataset), None)   # re-read fresh from disk
    tree = build_auto_tree(datafolder, dataset)
    return tree, tree["root"]["id"], battery_models(datafolder, dataset)


@app.callback(Output("ex-dataver", "data"), Input("ex-reload", "n_clicks"),
              State("ex-dataver", "data"), prevent_initial_call=True)
def cb_reload(_n, ver):
    """Drop cached maps so freshly-synced results are re-read; re-renders panels/tree."""
    _MAP_CACHE.clear()
    return (ver or 0) + 1


@app.callback(Output("pl-0-model", "options"), Output("pl-1-model", "options"), Output("pl-2-model", "options"),
              Input("ex-models", "data"))
def cb_panel_options(models):
    # Panels keep the full battery model list (manual per-panel override), grouping-labelled.
    opts = []
    for m in (models or []):
        hyp, grp = split_model(m)
        opts.append({"label": (f"{hyp} · {grp}" if grp else hyp), "value": m})
    return opts, opts, opts


# ---------------------------------------------------------------------------
# Callbacks — node selection (tree click)
# ---------------------------------------------------------------------------

@app.callback(Output("ex-selected", "data", allow_duplicate=True),
              Input("ex-tree-graph", "clickData"), prevent_initial_call=True)
def cb_node_click(click):
    if click and click.get("points"):
        cd = click["points"][0].get("customdata")
        if cd:
            return cd
    return no_update


# ---------------------------------------------------------------------------
# Callbacks — render tree + read-only node panel
# ---------------------------------------------------------------------------

@app.callback(Output("ex-tree-graph", "figure"), Output("ex-tree-title", "children"),
              Input("ex-tree-store", "data"), Input("ex-selected", "data"), Input("ex-grouping", "value"),
              Input("ex-roi", "value"), Input("ex-dataver", "data"),
              State("ex-datafolder", "value"), State("ex-dataset", "value"),
              State("ex-modality", "value"), State("ex-models", "data"))
def cb_render_tree(tree, sel_id, grouping, roi, _ver, datafolder, dataset, modality, models):
    if not tree or not tree.get("root", {}).get("children"):
        fig = niftiutil.empty_fig("No RSA model CSVs found in this dataset's rsa_models folder.", height=320)
        fig.update_layout(paper_bgcolor=PANEL, plot_bgcolor=PANEL)
        return fig, ""
    result_sets = ht.models_with_results(datafolder, dataset, modality, roi) if roi else {"D": set(), "H": set()}
    idx = hypothesis_index(models)
    n_hyp = sum(1 for _ in ht.iter_nodes(tree["root"])) - 1
    title = f"{tree.get('name', '')} · grouping = {grouping} · {n_hyp} hypotheses"
    return tree_figure(tree, sel_id, result_sets, grouping, idx), title


@app.callback(Output("ed-title", "children"), Output("ed-status", "children"),
              Output("ed-notes-text", "children"),
              Input("ex-selected", "data"), Input("ex-tree-store", "data"), Input("ex-grouping", "value"),
              Input("ex-roi", "value"), Input("ex-dataver", "data"),
              State("ex-datafolder", "value"), State("ex-dataset", "value"),
              State("ex-modality", "value"), State("ex-models", "data"))
def cb_node_readout(sel_id, tree, grouping, roi, _ver, datafolder, dataset, modality, models):
    if not tree or not sel_id:
        return "—", "Select a node in the tree.", ""
    node = ht.find_node(tree["root"], sel_id)
    if node is None:
        return "—", "Node not found.", ""
    stored = node.get("model")
    if not stored:                                    # the root
        n_hyp = sum(1 for _ in ht.iter_nodes(tree["root"])) - 1
        return (node.get("label", ""),
                f"{n_hyp} hypotheses · grouping = {grouping}",
                node.get("notes", "") or "Pick a hypothesis node to load its model.")
    idx = hypothesis_index(models)
    resolved = resolve_model(stored, grouping, idx)
    result_sets = ht.models_with_results(datafolder, dataset, modality, roi) if roi else {"D": set(), "H": set()}
    st = ht.node_status(resolved, result_sets)
    color, st_label = STATUS_STYLE[st]
    title = f"{node.get('label', '')}  ·  {grouping}"
    status = [f"model: {resolved or '—'}   ", status_dot(color), html.Span(st_label)]
    desc = model_description(datafolder, dataset, resolved) or "(no description in manifest)"
    parts = [p.strip() for p in desc.split(" | ")]
    notes = [html.Div(p, style={"marginBottom": "3px"}) for p in parts]
    return title, status, notes


@app.callback(Output("ed-matrix-graph", "figure"), Output("ed-matrix-note", "children"),
              Input("ex-selected", "data"), Input("ex-tree-store", "data"), Input("ex-grouping", "value"),
              State("ex-datafolder", "value"), State("ex-dataset", "value"), State("ex-models", "data"))
def cb_node_matrix(sel_id, tree, grouping, datafolder, dataset, models):
    if not tree or not sel_id:
        return niftiutil.empty_fig("Select a node", height=260), ""
    node = ht.find_node(tree["root"], sel_id)
    stored = node.get("model") if node else None
    if not stored:
        return niftiutil.empty_fig("Pick a hypothesis node to see its matrix.", height=260), ""
    model = resolve_model(stored, grouping, hypothesis_index(models))
    if not model:
        return niftiutil.empty_fig("no model for this grouping", height=260), ""
    return _model_heatmap(datafolder, dataset, model), model


# ---------------------------------------------------------------------------
# Callbacks — sync un-pinned panel models to the selected node
# ---------------------------------------------------------------------------

@app.callback(Output("pl-0-model", "value"), Output("pl-1-model", "value"), Output("pl-2-model", "value"),
              Input("ex-selected", "data"), Input("ex-tree-store", "data"), Input("ex-grouping", "value"),
              State("pl-0-pin", "value"), State("pl-1-pin", "value"), State("pl-2-pin", "value"),
              State("ex-models", "data"))
def cb_sync_panel_models(sel_id, tree, grouping, pin0, pin1, pin2, models):
    model = None
    if tree and sel_id:
        node = ht.find_node(tree["root"], sel_id)
        stored = node.get("model") if node else None
        model = resolve_model(stored, grouping, hypothesis_index(models))
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
        Input("ex-roi", "value"), Input("ex-dataver", "data"),
        State("ex-datafolder", "value"), State("ex-dataset", "value"), State("ex-modality", "value"))
    def _cb(enable, species, model, maptype, axis, frac, zt, roi, _ver,
            datafolder, dataset, modality):
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
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("EXPLORER_PORT", os.environ.get("PORT", "8055"))))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(f"[hypothesis_explorer] open http://{args.host}:{args.port}")
    app.run(debug=args.debug, use_reloader=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

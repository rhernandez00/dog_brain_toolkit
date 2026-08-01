#!/usr/bin/env python
"""
glm_designer.py — a simplified, experiment-specific clone of FSL FEAT's model
setup (Dash app, port 8061).

It does **one** job: take a working ``.fsf`` design you supply, replace its
EV/contrast model with one built from this experiment's conditions, and write a
copy. Everything else in the design — input paths, TR, volumes, smoothing,
registration, thresholds, output directory — is carried over from your template
untouched, because it is already correct there and will be rewritten per
subject/run later by the pipeline (``rsa_utils.calculate_beta_maps``).

So the app edits exactly the two things FEAT's "Stats" tab edits and nothing else:

1. **EVs** — auto-populated from the config's ``stim_types``, in order. You do
   not type them; pick species (D/H) and model (``basic`` / ``basic-block`` / …)
   and the conditions come from
   ``{datafolder}/{dataset}/config_files/{specie}_{model}.yaml``.
2. **Contrasts** — a FEAT-style grid: one row per contrast, one numeric cell per
   condition. Type the weights exactly as in the FEAT contrast editor.

Per-EV settings (waveform shape, convolution, temporal filtering, temporal
derivative) are lifted from the template's EV 1 and reused for every condition,
so a design that already works keeps working. The output stays in
``con_mode orig``: weights are per condition and the "real" contrast vectors
(with the temporal-derivative columns zeroed) are generated for you, which is
what makes the result loadable by ``feat`` and by the FEAT GUI.

Typical use
-----------
1. Pick **Species** and **Model** → conditions appear.
2. Point **Template .fsf** at the design to copy, e.g.
   ``P:\\userdata\\raulh87\\data\\EmoC\\FSL_designs\\H_basic.fsf`` (or upload one).
3. Build contrasts — **Identity** gives one per condition; **Mean by …** builds
   one per group (emotion label, species shown, run label, partition …) reading
   the config's ``model_dict``; or add blank rows and type the numbers.
4. **Preview**, then **Save** to a new ``.fsf``.

The generated file is a *template*: ``set fmri(custom{i})`` points at
``{timing folder}/{condition}.txt``, which is the naming
``rsa_utils.calculate_beta_maps`` fills in per subject/session/run.

Not covered (on purpose): F-tests, per-EV overrides, orthogonalisation, and
anything outside the Stats tab. If the supplied template contains F-tests the
app says so — they are dropped from the copy.

Run:
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\glm_designer.py
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys

import yaml
from dash import Dash, ctx, dash_table, dcc, html, no_update
from dash.dependencies import Input, Output, State

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)  # tools/ lives one level below the repo root
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import fsf_design as fd
from scheduler.paths import get_paths

VERSION = "1.0.0"
LAST_CHANGE = ("First release: FEAT-style EV/contrast designer that copies a supplied "
               ".fsf and swaps in conditions from a config's stim_types plus a "
               "contrast grid.")

# --- palette (matches the other viz apps) ---------------------------------
BG, PANEL, INK, MUTED, LINE, ACCENT = "#ffffff", "#f3f5f9", "#222222", "#667085", "#d5dbe5", "#4472C4"
INPUT_STYLE = {"backgroundColor": "#ffffff", "color": INK,
               "border": f"1px solid {LINE}", "borderRadius": "6px", "padding": "5px 8px"}
BTN = {"height": "32px", "padding": "0 14px", "backgroundColor": ACCENT, "color": "white",
       "border": "none", "borderRadius": "6px", "cursor": "pointer", "fontWeight": "bold"}
BTN2 = {**BTN, "backgroundColor": "#eef1f6", "color": INK, "border": f"1px solid {LINE}",
        "fontWeight": "normal"}
CARD = {"backgroundColor": PANEL, "border": f"1px solid {LINE}", "borderRadius": "10px",
        "padding": "12px 14px", "marginBottom": "12px"}
LBL = {"color": MUTED, "fontSize": "12px", "marginRight": "6px"}

SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".rsa_glm_designer_settings.json")
URL_BASE = os.environ.get("GLM_DESIGNER_URL_BASE", "/")

# Template texts are kept server-side and referenced by key, so a 350 kB design
# never travels through a dcc.Store on every callback.
_TEMPLATE_CACHE = {}

# Config fields that describe *how a condition looks*, not *what group it is in*.
_NON_GROUP_FIELDS = {"color", "stim_file"}


# --- settings ---------------------------------------------------------------

def load_settings():
    defaults = {
        "datafolder": get_paths()[0],
        "dataset": "EmoC",
        "specie": "H",
        "model": "basic",
        "template": "",
        "output": "",
        "timing_dir": "",
        "contrasts": {},   # "{specie}_{model}" -> [{name, weights}]
    }
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
            defaults.update(json.load(fh))
    except (OSError, ValueError):
        pass
    return defaults


def save_settings(settings):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
    except OSError as exc:
        print(f"[glm_designer] could not save settings: {exc}")


SETTINGS = load_settings()


# --- config reading ---------------------------------------------------------

def config_path(datafolder, dataset, specie, model):
    return os.path.join(datafolder, dataset, "config_files", f"{specie}_{model}.yaml")


def list_models(datafolder, dataset, specie):
    """Model names with a config for this species, e.g. ['basic', 'basic-block']."""
    pattern = os.path.join(datafolder, dataset, "config_files", f"{specie}_*.yaml")
    out = []
    for path in sorted(glob.glob(pattern)):
        stem = os.path.splitext(os.path.basename(path))[0]
        out.append(stem[len(specie) + 1:])
    return out


def read_config(datafolder, dataset, specie, model):
    """Return ``(stim_types, per-condition attribute dict, error)``.

    The attribute dict is ``{field: [value per condition]}``, read from the first
    run of ``model_dict`` — that is what drives the colour chips and the
    "Mean by …" contrast builder.
    """
    path = config_path(datafolder, dataset, specie, model)
    if not os.path.exists(path):
        return [], {}, f"config not found: {path}"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [], {}, f"could not read {os.path.basename(path)}: {exc}"

    stim_types = list(cfg.get("stim_types") or [])
    if not stim_types:
        return [], {}, f"{os.path.basename(path)} defines no stim_types"

    model_dict = cfg.get("model_dict") or {}
    first_run = model_dict.get(sorted(model_dict)[0]) if model_dict else {}
    fields = set()
    for stim in stim_types:
        fields.update((first_run.get(stim) or {}).keys())

    attrs = {}
    for field in sorted(fields):
        attrs[field] = [str((first_run.get(s) or {}).get(field, "")) for s in stim_types]
    return stim_types, attrs, ""


def group_fields(stim_types, attrs):
    """Attribute fields worth grouping by: more than one value, fewer than all."""
    out = []
    for field, values in attrs.items():
        if field in _NON_GROUP_FIELDS:
            continue
        n_distinct = len({v for v in values if v})
        if 1 <= n_distinct < len(stim_types):
            out.append(field)
    return out


# --- path defaults ----------------------------------------------------------

def default_template(datafolder, dataset, specie, model):
    """Best guess at the design to copy, preferring one that exists on disk."""
    candidates = [
        os.path.join(datafolder, dataset, "FSL_designs", f"{specie}_{model}.fsf"),
        os.path.join(datafolder, dataset, "FSL_designs", f"{specie}_basic.fsf"),
    ]
    if specie == "D":
        candidates.append(os.path.join(_REPO_ROOT, "FSL_designs", "basic_DHRF.fsf"))
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def default_output(datafolder, dataset, specie, model):
    return os.path.join(datafolder, dataset, "FSL_designs", f"{specie}_{model}_designed.fsf")


def default_timing_dir(template_dir, model):
    """Timing folder for the new design: the template's, retargeted at ``model``.

    FEAT designs are written on Linux, so the ``/models/{model}/`` segment is
    swapped in place and POSIX separators are kept.
    """
    if not template_dir:
        return ""
    parts = template_dir.split("/")
    for i, part in enumerate(parts[:-1]):
        if part == "models":
            parts[i + 1] = model
            break
    return "/".join(parts)


# --- contrast table helpers -------------------------------------------------

def col_id(i):
    return f"c{i}"


def table_columns(stim_types):
    cols = [{"name": "Contrast", "id": "__name", "type": "text", "editable": True}]
    for i, stim in enumerate(stim_types):
        cols.append({"name": stim, "id": col_id(i), "type": "numeric", "editable": True})
    return cols


def table_styles(stim_types):
    """Zebra rows, and negative weights in red like FEAT's contrast editor."""
    styles = [{"if": {"row_index": "odd"}, "backgroundColor": "#f7f9fc"}]
    styles += [{"if": {"filter_query": "{%s} < 0" % col_id(i), "column_id": col_id(i)},
                "color": "#b3261e", "fontWeight": "bold"}
               for i in range(len(stim_types))]
    return styles


def contrasts_to_rows(contrasts):
    rows = []
    for con in contrasts:
        row = {"__name": con.name}
        for i, w in enumerate(con.weights):
            row[col_id(i)] = w
        rows.append(row)
    return rows


def rows_to_contrasts(rows, n_ev):
    out = []
    for r, row in enumerate(rows or [], start=1):
        weights = []
        for i in range(n_ev):
            val = (row or {}).get(col_id(i), 0)
            try:
                weights.append(float(val) if val not in (None, "") else 0.0)
            except (TypeError, ValueError):
                weights.append(0.0)
        name = str((row or {}).get("__name") or f"con{r}").strip() or f"con{r}"
        out.append(fd.Contrast(name=name, weights=weights))
    return out


def contrast_report(contrasts):
    """One line per contrast: sum and how many conditions it touches."""
    if not contrasts:
        return "no contrasts — FEAT needs at least one."
    bits = []
    for i, con in enumerate(contrasts, start=1):
        total = sum(con.weights)
        n_nz = sum(1 for w in con.weights if w)
        kind = "differential" if abs(total) < 1e-9 and n_nz else "sum"
        if not n_nz:
            kind = "EMPTY"
        bits.append(f"{i}. {con.name}: Σ={total:g}, {n_nz} cond, {kind}")
    return " | ".join(bits)


# --- template loading -------------------------------------------------------

def cache_template(key, text):
    _TEMPLATE_CACHE[key] = text
    return key


def template_text(key):
    return _TEMPLATE_CACHE.get(key or "")


def describe_template(text):
    """Human-readable one-liner plus the parsed summary dict."""
    info = fd.template_summary(text)
    opts = info["options"]
    bits = [f"{info['n_evs']} EV(s)", f"{info['n_contrasts']} contrast(s)"]
    if opts:
        bits.append(f"shape={opts.shape}, convolve={opts.convolve}, "
                    f"tempfilt={opts.tempfilt_yn}, deriv={opts.deriv_yn}")
    if info["n_ftests"]:
        bits.append(f"⚠ {info['n_ftests']} F-test(s) — NOT carried over")
    if info["level"] != 1:
        bits.append(f"⚠ level={info['level']} (not a first-level design)")
    if opts and opts.shape not in (2, 3):
        # Square/sinusoid EVs need skip/off/on/phase/stop/period, which this
        # designer does not write — every EV it generates reads a timing file.
        bits.append(f"⚠ EV waveform shape={opts.shape} is not a custom timing file")
    if opts and opts.convolve not in (0, 3):
        # Gamma/basis-function convolutions need extra per-EV parameters
        # (gammasigma, gammadelay, basisfnum, …) that are not regenerated.
        bits.append(f"⚠ convolution={opts.convolve} needs extra per-EV parameters")
    return " · ".join(bits), info


# --- layout -----------------------------------------------------------------

def _row(*children, **kw):
    style = {"display": "flex", "alignItems": "center", "gap": "8px",
             "flexWrap": "wrap", **kw.pop("style", {})}
    return html.Div(list(children), style=style, **kw)


_init_models = list_models(SETTINGS["datafolder"], SETTINGS["dataset"], SETTINGS["specie"])
_init_model = SETTINGS["model"] if SETTINGS["model"] in _init_models else (
    _init_models[0] if _init_models else SETTINGS["model"])

app = Dash(__name__, url_base_pathname=URL_BASE, suppress_callback_exceptions=True,
           title="GLM Designer")

app.layout = html.Div(style={"backgroundColor": BG, "color": INK, "padding": "16px 20px",
                             "fontFamily": "Segoe UI, Roboto, sans-serif",
                             "maxWidth": "1600px", "margin": "0 auto"}, children=[

    html.Div([
        html.H2("GLM Designer", style={"margin": "0 0 2px 0"}),
        html.Div(f"FSL FEAT design builder · v{VERSION} — conditions come from the "
                 f"config's stim_types; everything outside the EV/contrast model is "
                 f"copied from your template.",
                 style={"color": MUTED, "fontSize": "13px"}),
    ], style={"marginBottom": "14px"}),

    # --- 1. experiment -----------------------------------------------------
    html.Div(style=CARD, children=[
        html.Div("1 · Experiment", style={"fontWeight": "bold", "marginBottom": "8px"}),
        _row(
            html.Span("Data folder", style=LBL),
            dcc.Input(id="gd-datafolder", value=SETTINGS["datafolder"], type="text",
                      debounce=True, style={**INPUT_STYLE, "width": "260px"}),
            html.Span("Dataset", style=LBL),
            dcc.Input(id="gd-dataset", value=SETTINGS["dataset"], type="text",
                      debounce=True, style={**INPUT_STYLE, "width": "110px"}),
            html.Span("Species", style=LBL),
            dcc.RadioItems(id="gd-specie", value=SETTINGS["specie"],
                           options=[{"label": " Dog (D)", "value": "D"},
                                    {"label": " Human (H)", "value": "H"}],
                           inline=True, style={"fontSize": "13px"},
                           inputStyle={"marginRight": "4px", "marginLeft": "10px"}),
            html.Span("Model", style=LBL),
            dcc.Dropdown(id="gd-model", value=_init_model,
                         options=[{"label": m, "value": m} for m in _init_models],
                         clearable=False, style={"width": "220px"}),
            html.Button("🔄 Rescan configs", id="gd-rescan", n_clicks=0, style=BTN2),
        ),
        html.Div(id="gd-config-status", style={"color": MUTED, "fontSize": "12px",
                                               "marginTop": "8px"}),
        html.Div(id="gd-conditions", style={"marginTop": "8px"}),
    ]),

    # --- 2. template -------------------------------------------------------
    html.Div(style=CARD, children=[
        html.Div("2 · Template design (copied, then its EVs/contrasts replaced)",
                 style={"fontWeight": "bold", "marginBottom": "8px"}),
        _row(
            html.Span("Template .fsf", style=LBL),
            dcc.Input(id="gd-template", value=SETTINGS["template"], type="text",
                      debounce=True, style={**INPUT_STYLE, "width": "620px"}),
            html.Button("📂 Load", id="gd-load-template", n_clicks=0, style=BTN),
            dcc.Upload(id="gd-upload", children=html.Button("⬆ Upload…", style=BTN2),
                       multiple=False),
        ),
        html.Div(id="gd-template-status", style={"color": MUTED, "fontSize": "12px",
                                                 "marginTop": "8px"}),
        _row(
            html.Span("EV timing folder", style=LBL),
            dcc.Input(id="gd-timing-dir", value=SETTINGS["timing_dir"], type="text",
                      debounce=True, style={**INPUT_STYLE, "width": "620px"}),
            html.Span("each EV becomes {folder}/{condition}.txt — the pipeline "
                      "rewrites these per subject/session/run",
                      style={**LBL, "fontStyle": "italic"}),
            style={"marginTop": "8px"},
        ),
    ]),

    # --- 3. contrasts ------------------------------------------------------
    html.Div(style=CARD, children=[
        html.Div("3 · Contrasts", style={"fontWeight": "bold", "marginBottom": "8px"}),
        _row(
            html.Button("＋ Add contrast", id="gd-add", n_clicks=0, style=BTN2),
            html.Button("Identity (one per condition)", id="gd-identity", n_clicks=0,
                        style=BTN2),
            html.Span("Mean by", style=LBL),
            dcc.Dropdown(id="gd-groupfield", options=[], clearable=False,
                         style={"width": "180px"}),
            html.Button("＋ Add group contrasts", id="gd-addgroup", n_clicks=0, style=BTN2),
            html.Button("Load from template", id="gd-fromtemplate", n_clicks=0, style=BTN2),
            html.Button("Clear", id="gd-clear", n_clicks=0, style=BTN2),
        ),
        html.Div(id="gd-contrast-status", style={"color": MUTED, "fontSize": "12px",
                                                 "margin": "8px 0"}),
        dash_table.DataTable(
            id="gd-table",
            columns=[], data=[],
            editable=True, row_deletable=True,
            fixed_columns={"headers": True, "data": 1},
            style_table={"overflowX": "auto", "minWidth": "100%"},
            style_cell={"fontFamily": "Segoe UI, Roboto, sans-serif", "fontSize": "13px",
                        "padding": "4px 8px", "textAlign": "center", "minWidth": "62px",
                        "width": "62px", "maxWidth": "62px"},
            style_cell_conditional=[{"if": {"column_id": "__name"},
                                     "textAlign": "left", "minWidth": "180px",
                                     "width": "180px", "maxWidth": "180px",
                                     "fontWeight": "bold"}],
            style_header={"backgroundColor": "#e9edf4", "fontWeight": "bold"},
            style_data_conditional=[],
        ),
    ]),

    # --- 4. output ---------------------------------------------------------
    html.Div(style=CARD, children=[
        html.Div("4 · Output", style={"fontWeight": "bold", "marginBottom": "8px"}),
        _row(
            html.Span("Save as", style=LBL),
            dcc.Input(id="gd-output", value=SETTINGS["output"], type="text",
                      debounce=True, style={**INPUT_STYLE, "width": "620px"}),
            html.Button("👁 Preview", id="gd-preview", n_clicks=0, style=BTN2),
            html.Button("💾 Save .fsf", id="gd-save", n_clicks=0, style=BTN),
            dcc.Checklist(id="gd-overwrite", options=[{"label": " overwrite if it exists",
                                                       "value": "ow"}],
                          value=[], inline=True, style={"fontSize": "12px"}),
        ),
        html.Div(id="gd-save-status", style={"fontSize": "13px", "marginTop": "8px"}),
        html.Pre(id="gd-preview-box",
                 style={"backgroundColor": "#0f1115", "color": "#d7dce3",
                        "padding": "10px", "borderRadius": "8px", "fontSize": "12px",
                        "maxHeight": "420px", "overflow": "auto", "marginTop": "8px",
                        "whiteSpace": "pre", "display": "none"}),
    ]),

    dcc.Store(id="gd-config"),        # {stim_types, attrs, fields}
    dcc.Store(id="gd-template-key"),  # key into _TEMPLATE_CACHE
])


# --- callbacks --------------------------------------------------------------

@app.callback(Output("gd-model", "options"), Output("gd-model", "value"),
              Input("gd-datafolder", "value"), Input("gd-dataset", "value"),
              Input("gd-specie", "value"), Input("gd-rescan", "n_clicks"),
              State("gd-model", "value"))
def cb_models(datafolder, dataset, specie, _n, current):
    models = list_models(datafolder or "", dataset or "", specie or "H")
    options = [{"label": m, "value": m} for m in models]
    if current in models:
        return options, current
    return options, (models[0] if models else None)


@app.callback(Output("gd-config", "data"), Output("gd-config-status", "children"),
              Output("gd-conditions", "children"), Output("gd-groupfield", "options"),
              Output("gd-groupfield", "value"),
              Input("gd-datafolder", "value"), Input("gd-dataset", "value"),
              Input("gd-specie", "value"), Input("gd-model", "value"))
def cb_config(datafolder, dataset, specie, model):
    if not model:
        return None, "no config files found for this species.", "", [], None

    stim_types, attrs, err = read_config(datafolder or "", dataset or "", specie, model)
    if err:
        return None, f"⚠ {err}", "", [], None

    fields = group_fields(stim_types, attrs)
    colors = attrs.get("color") or [""] * len(stim_types)
    names = attrs.get("label_name") or [""] * len(stim_types)

    chips = []
    for i, stim in enumerate(stim_types):
        color = colors[i] if colors[i].startswith("#") and len(colors[i]) == 7 else LINE
        chips.append(html.Div([
            html.Span(str(i + 1), style={"color": MUTED, "fontSize": "11px",
                                         "marginRight": "5px"}),
            html.Span(style={"display": "inline-block", "width": "10px", "height": "10px",
                             "borderRadius": "3px", "backgroundColor": color,
                             "marginRight": "5px"}),
            html.Span(stim, style={"fontWeight": "bold", "fontSize": "12px"}),
            html.Span(f" {names[i]}" if names[i] else "",
                      style={"color": MUTED, "fontSize": "11px", "marginLeft": "4px"}),
        ], style={"border": f"1px solid {LINE}", "borderRadius": "6px",
                  "padding": "3px 7px", "backgroundColor": "#ffffff"}))

    status = (f"{config_path(datafolder, dataset, specie, model)} — "
              f"{len(stim_types)} condition(s)")
    data = {"stim_types": stim_types, "attrs": attrs, "fields": fields}
    opts = [{"label": f, "value": f} for f in fields]
    return (data, status,
            html.Div(chips, style={"display": "flex", "gap": "6px", "flexWrap": "wrap"}),
            opts, (fields[0] if fields else None))


@app.callback(Output("gd-template", "value"), Output("gd-output", "value"),
              Input("gd-specie", "value"), Input("gd-model", "value"),
              State("gd-datafolder", "value"), State("gd-dataset", "value"),
              prevent_initial_call=True)
def cb_default_paths(specie, model, datafolder, dataset):
    """Retarget the template/output paths when the design being built changes."""
    if not model:
        return no_update, no_update
    return (default_template(datafolder or "", dataset or "", specie, model),
            default_output(datafolder or "", dataset or "", specie, model))


@app.callback(Output("gd-template-key", "data"), Output("gd-template-status", "children"),
              Output("gd-timing-dir", "value"),
              Input("gd-load-template", "n_clicks"), Input("gd-upload", "contents"),
              Input("gd-template", "value"),
              State("gd-upload", "filename"), State("gd-model", "value"))
def cb_load_template(_n, upload, path, upload_name, model):
    trigger = ctx.triggered_id

    if trigger == "gd-upload" and upload:
        try:
            _header, b64 = upload.split(",", 1)
            text = base64.b64decode(b64).decode("utf-8", errors="replace")
        except (ValueError, UnicodeDecodeError) as exc:
            return None, f"⚠ could not decode upload: {exc}", no_update
        key = f"upload:{upload_name}"
    else:
        if not path:
            return None, "no template selected.", no_update
        if not os.path.exists(path):
            return None, f"⚠ not found: {path}", no_update
        try:
            text = fd.read_template(path)
        except OSError as exc:
            return None, f"⚠ could not read: {exc}", no_update
        key = path

    try:
        desc, info = describe_template(text)
    except fd.FsfError as exc:
        return None, f"⚠ {exc}", no_update

    cache_template(key, text)
    label = upload_name if key.startswith("upload:") else os.path.basename(key)
    # The timing folder always follows the template just loaded — the path field
    # is retargeted automatically when species/model change, so keeping an
    # earlier value here would silently point the new design at the old model's
    # timing files. Type your own folder after loading and it stays until the
    # next load.
    return key, f"✔ {label} — {desc}", default_timing_dir(info["custom_dir"], model or "")


@app.callback(Output("gd-table", "columns"), Output("gd-table", "data"),
              Output("gd-contrast-status", "children"),
              Output("gd-table", "style_data_conditional"),
              Input("gd-config", "data"),
              Input("gd-add", "n_clicks"), Input("gd-identity", "n_clicks"),
              Input("gd-addgroup", "n_clicks"), Input("gd-fromtemplate", "n_clicks"),
              Input("gd-clear", "n_clicks"),
              State("gd-table", "data"), State("gd-groupfield", "value"),
              State("gd-template-key", "data"),
              State("gd-specie", "value"), State("gd-model", "value"))
def cb_contrasts(config, _a, _i, _g, _t, _c, rows, field, tkey, specie, model):
    if not config:
        return [], [], "load a config first.", []

    stim_types = config["stim_types"]
    n_ev = len(stim_types)
    columns = table_columns(stim_types)
    styles = table_styles(stim_types)
    trigger = ctx.triggered_id

    def result(contrasts, note):
        return (columns, contrasts_to_rows(contrasts),
                f"{note} {contrast_report(contrasts)}".strip(), styles)

    if trigger == "gd-config" or trigger is None:
        # New condition set: restore this design's saved contrasts if they still
        # fit, otherwise start from the FEAT default of one contrast per EV.
        saved = SETTINGS.get("contrasts", {}).get(f"{specie}_{model}") or []
        restored = [fd.Contrast(name=c["name"], weights=c["weights"])
                    for c in saved if len(c.get("weights", [])) == n_ev]
        contrasts = restored or fd.identity_contrasts(stim_types)
        return result(contrasts, "restored saved contrasts." if restored
                      else "started from identity contrasts (one per condition).")

    contrasts = rows_to_contrasts(rows, n_ev)

    if trigger == "gd-add":
        contrasts.append(fd.Contrast(name=f"con{len(contrasts) + 1}",
                                     weights=[0.0] * n_ev))
        note = "added an empty contrast."
    elif trigger == "gd-identity":
        contrasts = fd.identity_contrasts(stim_types)
        note = "replaced with identity contrasts."
    elif trigger == "gd-clear":
        contrasts = []
        note = "cleared."
    elif trigger == "gd-addgroup":
        if not field:
            return result(contrasts, "⚠ pick a field to group by.")
        groups = config["attrs"].get(field) or []
        added = fd.mean_contrasts_by_group(stim_types, groups)
        contrasts.extend(added)
        note = f"added {len(added)} mean contrast(s) by {field}."
    elif trigger == "gd-fromtemplate":
        text = template_text(tkey)
        if not text:
            return result(contrasts, "⚠ load a template first.")
        from_tpl = fd.template_contrasts(text)
        if not from_tpl:
            return result(contrasts, "⚠ template has no contrasts.")
        if len(from_tpl[0].weights) != n_ev:
            return result(contrasts,
                          f"⚠ template contrasts span {len(from_tpl[0].weights)} EV(s) "
                          f"but this config has {n_ev} condition(s) — cannot map them.")
        contrasts = from_tpl
        note = f"loaded {len(contrasts)} contrast(s) from the template."
    else:
        note = ""

    return result(contrasts, note)


def assemble(config, tkey, timing_dir, rows):
    """Build the output .fsf text. Returns ``(text, error_message)``."""
    if not config:
        return None, "no config loaded."
    text = template_text(tkey)
    if not text:
        return None, "no template loaded — set a template .fsf and click Load."

    stim_types = config["stim_types"]
    contrasts = rows_to_contrasts(rows, len(stim_types))
    if not contrasts:
        return None, "add at least one contrast — FEAT will not run without one."

    empty = [c.name for c in contrasts if not any(c.weights)]
    if empty:
        return None, f"contrast(s) with all-zero weights: {', '.join(empty)}"

    custom = fd.custom_files_for(timing_dir, stim_types)
    try:
        return fd.build_fsf(text, stim_types, custom, contrasts), ""
    except fd.FsfError as exc:
        return None, str(exc)


@app.callback(Output("gd-preview-box", "children"), Output("gd-preview-box", "style"),
              Output("gd-save-status", "children", allow_duplicate=True),
              Input("gd-preview", "n_clicks"),
              State("gd-config", "data"), State("gd-template-key", "data"),
              State("gd-timing-dir", "value"), State("gd-table", "data"),
              State("gd-preview-box", "style"),
              prevent_initial_call=True)
def cb_preview(_n, config, tkey, timing_dir, rows, style):
    text, err = assemble(config, tkey, timing_dir, rows)
    if err:
        return "", {**style, "display": "none"}, html.Span(f"⚠ {err}", style={"color": "#b3261e"})

    lines = text.splitlines()
    n_ev = len(config["stim_types"])
    n_con = len(rows or [])
    head = (f"# --- preview: {n_ev} EV(s), {n_con} contrast(s), "
            f"{len(lines)} lines ------------------------\n\n")
    return head + text, {**style, "display": "block"}, html.Span(
        f"Preview only — nothing written yet. {len(lines)} lines.", style={"color": MUTED})


@app.callback(Output("gd-save-status", "children"),
              Input("gd-save", "n_clicks"),
              State("gd-config", "data"), State("gd-template-key", "data"),
              State("gd-timing-dir", "value"), State("gd-table", "data"),
              State("gd-output", "value"), State("gd-overwrite", "value"),
              State("gd-specie", "value"), State("gd-model", "value"),
              State("gd-datafolder", "value"), State("gd-dataset", "value"),
              State("gd-template", "value"),
              prevent_initial_call=True)
def cb_save(_n, config, tkey, timing_dir, rows, output, overwrite,
            specie, model, datafolder, dataset, template):
    text, err = assemble(config, tkey, timing_dir, rows)
    if err:
        return html.Span(f"⚠ {err}", style={"color": "#b3261e"})
    if not output:
        return html.Span("⚠ set an output path.", style={"color": "#b3261e"})
    if os.path.exists(output) and "ow" not in (overwrite or []):
        return html.Span(f"⚠ {output} exists — tick 'overwrite' to replace it.",
                         style={"color": "#b3261e"})

    try:
        fd.write_fsf(output, text)
    except OSError as exc:
        return html.Span(f"⚠ could not write: {exc}", style={"color": "#b3261e"})

    contrasts = rows_to_contrasts(rows, len(config["stim_types"]))
    SETTINGS.update({"datafolder": datafolder, "dataset": dataset, "specie": specie,
                     "model": model, "template": template, "output": output,
                     "timing_dir": timing_dir})
    SETTINGS.setdefault("contrasts", {})[f"{specie}_{model}"] = [
        {"name": c.name, "weights": c.weights} for c in contrasts]
    save_settings(SETTINGS)

    n_ev = len(config["stim_types"])
    return html.Span(
        f"✔ wrote {output} — {n_ev} EV(s), {len(contrasts)} contrast(s). "
        f"Run it with:  feat {output}",
        style={"color": "#1a7f37"})


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="GLM Designer — FSL FEAT design builder")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("GLM_DESIGNER_PORT",
                                               os.environ.get("PORT", "8061"))))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(f"[glm_designer] v{VERSION}")
    print(f"[glm_designer] settings : {SETTINGS_PATH}")
    print(f"[glm_designer] data     : {SETTINGS['datafolder']} / {SETTINGS['dataset']}")
    print(f"[glm_designer] open http://{args.host}:{args.port}")
    app.run(debug=args.debug, use_reloader=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

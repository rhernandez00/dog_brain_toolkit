"""
RSA Model Builder — interactive editor for RSA dissimilarity matrices.

Building a new model is a four-step pick:
    1. config file  — <dataset>/config_files/<D|H>_<glm model>.yaml, the same file
                      the pipeline reloads in rsa_utils.py; `stim_types` is the
                      pool of stimuli everything else is derived from
    2. dis_method   — 'mahalanobis' (default) or 'correlation'
    3. mah_fold     — 'stim-wise' | 'stim-wise-multiple-folds' | 'stim-wise-all-runs'
                      (Mahalanobis only), the folding used in step 2 of the
                      searchlight by calculate_pairwise_similarity_maps2
    4. rows/columns — derived from 1–3 so the matrix holds exactly the stimulus
                      pairs that combination can produce, and no others

Launch:  python rsa_model_builder.py   →   http://127.0.0.1:8051
Requires: dash, plotly>=5.0, pandas, numpy, pyyaml
"""

import os
import sys
import math
import json
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
import plotly.graph_objects as go
import plotly.colors as pc
from dash import Dash, html, dcc, no_update, ctx
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viz import datasource, dash_kwargs

# Resolve the dataset root (Google Drive -> network -> $DBT_RESULTS_ROOT).
DEFAULT_DATASET    = "EmoC"
_DATAFOLDER        = datasource.resolve_datafolder(DEFAULT_DATASET, must_have_results=False)
DEFAULT_CONFIG_DIR = os.path.join(_DATAFOLDER, DEFAULT_DATASET, "config_files")
DEFAULT_CONFIG     = "D_basic-block.yaml"
DEFAULT_EXPORT_DIR = os.path.join(_DATAFOLDER, DEFAULT_DATASET, "rsa_models")
MAX_UNDO = 50

HIDDEN_ATTRS   = {"color"}
ALL_RUNS_KEY   = "__all__"
NAN_SENTINEL   = "NaN"

# Config files are named <specie>_<glm model>.yaml — see rsa_utils.py, where the
# pipeline rebuilds the same path from `specie` and `model`.
SPECIE_NAMES = {"D": "Dog", "H": "Human"}

# Step 2 of searchlight (calculate_pairwise_similarity_maps2 in rsa_utils.py).
DIS_METHOD_OPTIONS = [
    {"label": " Mahalanobis (crossnobis)", "value": "mahalanobis"},
    {"label": " Correlation distance",     "value": "correlation"},
]

EMOC_ONLY_FOLDS = {"stim-wise-multiple-folds", "stim-wise-all-runs"}

MAH_FOLD_HELP = {
    "stim-wise": ("Collapses every exemplar of the same stimulus type across runs and uses "
                  "the runs themselves as cross-validation folds. One subject-level map per "
                  "category pair: <sub>/r-<radius>_mahalanobis_<A>_<B>.nii.gz"),
    "stim-wise-multiple-folds": ("EmoC only. Folds are the repeated `partition` values of each exact "
                  "stimulus (config `stim_file`); only stimuli seen in ≥2 partitions qualify. "
                  "One subject-level map per exact-stimulus pair."),
    "stim-wise-all-runs": ("EmoC only. A separate class-level crossnobis inside each run, using the "
                  "exemplar numbers (DogA1…DogA4) as the folds. One map per class pair per run: "
                  "<sub>/ses-XX_task-<task>_run-XX/r-<radius>_mahalanobis_<A>_<B>.nii.gz"),
}

MAH_FOLD_OPTIONS = [
    {"label": " stim-wise",                "value": "stim-wise"},
    {"label": " stim-wise-multiple-folds", "value": "stim-wise-multiple-folds"},
    {"label": " stim-wise-all-runs",       "value": "stim-wise-all-runs"},
]

# Right-click menu on the matrix. `value` None means NaN (pair left out of the
# model); every other value must stay inside [0, 1] — that is the range the
# colorbar and step 2 of the searchlight assume.
DISSIM_MIN, DISSIM_MAX = 0.0, 1.0

DISSIM_PRESETS = [
    {"label": "0",   "value": 0.0,  "desc": "identical stimuli — no difference at all"},
    {"label": "0.5", "value": 0.5,  "desc": "partially different — alike on some attributes, not others"},
    {"label": "1",   "value": 1.0,  "desc": "completely different — maximally dissimilar"},
    {"label": "NaN", "value": None, "desc": "excluded — this pair is dropped from the model fit"},
]

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

def scan_config_files(folder: str) -> list:
    """YAML configs in `folder`, species-prefixed ones (D_*/H_*) first."""
    try:
        names = [f for f in os.listdir(folder)
                 if f.lower().endswith((".yaml", ".yml"))]
    except Exception:
        return []
    return sorted(names, key=lambda n: (specie_from_filename(n) is None, n.lower()))

def specie_from_filename(name: str):
    """'D_basic-block.yaml' -> 'D'.  Returns None when the name is not prefixed."""
    base = os.path.basename(name or "")
    if len(base) >= 2 and base[0].upper() in SPECIE_NAMES and not base[1].isalnum():
        return base[0].upper()
    return None

def config_file_label(name: str) -> str:
    specie = specie_from_filename(name)
    return f"{name}   ·   {SPECIE_NAMES[specie]}" if specie else name

def export_dir_for_config(config_path: str) -> str:
    """<root>/<dataset>/config_files/x.yaml  ->  <root>/<dataset>/rsa_models."""
    dataset_dir = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))
    return os.path.join(dataset_dir, "rsa_models")

# ---------------------------------------------------------------------------
# Axis derivation — which rows/columns a (dis_method, mah_fold) pair allows
#
# These mirror rsa_utils.calculate_pairwise_similarity_maps2 (step 2 of the
# searchlight) and check_existing_similarity_maps, so the labels here are
# exactly the labels the pipeline will use in its output filenames and will
# look up in the exported model CSV.
# ---------------------------------------------------------------------------

def _uniq(seq):
    out, seen = [], set()
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out

def _model_dict_is_run_keyed(model_dict: dict) -> bool:
    return any(isinstance(v, dict) and any(isinstance(x, dict) for x in v.values())
               for v in (model_dict or {}).values())

def stim_meta_index(cfg: dict) -> dict:
    """{stim name: [metadata dicts]} collected from every run of model_dict."""
    model_dict = cfg.get("model_dict") or {}
    index = {}
    if _model_dict_is_run_keyed(model_dict):
        for run_key, run_dict in model_dict.items():
            if not isinstance(run_dict, dict):
                continue
            for stim, meta in run_dict.items():
                if isinstance(meta, dict):
                    index.setdefault(stim, []).append({**meta, "run": run_key})
    else:
        for stim, meta in model_dict.items():
            if isinstance(meta, dict):
                index.setdefault(stim, []).append(dict(meta))
    return index

def _merge_attrs(metas: list) -> dict:
    """Keep only the attributes every member agrees on — the rest vary."""
    metas = [m for m in metas if isinstance(m, dict)]
    if not metas:
        return {}
    shared = set(metas[0])
    for meta in metas[1:]:
        shared &= set(meta)
    out = {}
    for key in shared:
        values = [meta[key] for meta in metas]
        try:
            if len(set(values)) == 1:
                out[key] = values[0]
        except TypeError:  # unhashable value — skip it
            continue
    return out

def stim_attrs(index: dict, stim: str) -> dict:
    """Attributes for one raw stimulus name, exact match first."""
    if stim in index:
        return _merge_attrs(index[stim])
    # Flat model_dicts key on the bare label (P, H, …) while stim_types carry a
    # prefix (Dog-P) — fall back to the longest matching suffix.
    for key in sorted(index, key=len, reverse=True):
        if key and stim.endswith(key):
            return _merge_attrs(index[key])
    return {}

def _entity(name: str, attrs: dict, run: str = "", extra: dict = None) -> dict:
    entity = {"name": name, "run": run}
    for key, value in (attrs or {}).items():
        if key in ("name", "run"):
            continue
        entity[key] = value
    entity.update(extra or {})
    entity.setdefault("color", "#cccccc")
    return entity

def config_run_dicts(cfg: dict) -> list:
    """Ordered (run key, run dict) pairs — mirrors the pipeline's run loop."""
    model_dict = cfg.get("model_dict") or {}
    runs = cfg.get("runs") or []
    if not runs:
        raise ValueError("Config has no 'runs'.")
    pairs = []
    for run in runs:
        run_key = f"run{int(run):02d}"
        run_dict = model_dict.get(run_key)
        if not isinstance(run_dict, dict):
            raise ValueError(f"model_dict is missing metadata for {run_key}.")
        pairs.append((run_key, run_dict))
    return pairs

def _stim_wise_key(dataset: str, stim: str):
    """Category a stimulus collapses into under mah_fold='stim-wise'."""
    if dataset == "EmoB":
        return stim.split("-")[0] if "-" in stim else None
    if dataset == "EmoC":
        return stim[:-1]
    raise ValueError(
        f"mah_fold 'stim-wise' derives categories for EmoB and EmoC only, not {dataset!r}."
    )

def _strip_exemplar(stim: str):
    """'DogA4' -> 'DogA'.  None when the name has no trailing exemplar number."""
    cut = len(stim)
    while cut > 0 and stim[cut - 1].isdigit():
        cut -= 1
    return stim[:cut] if 0 < cut < len(stim) else None

def multiple_fold_stim_files(cfg: dict):
    """Exact EmoC stimuli repeated across ≥2 partitions, plus their shared folds."""
    stim_files, partitions = [], {}
    for run_key, run_dict in config_run_dicts(cfg):
        for stim, meta in run_dict.items():
            if not isinstance(meta, dict):
                raise ValueError(f"model_dict[{run_key!r}][{stim!r}] must be a dictionary.")
            stim_file = meta.get("stim_file")
            if not stim_file or "partition" not in meta:
                raise ValueError(
                    f"model_dict[{run_key!r}][{stim!r}] requires 'stim_file' and 'partition'."
                )
            if stim_file not in partitions:
                partitions[stim_file] = set()
                stim_files.append(stim_file)
            partitions[stim_file].add(meta["partition"])
    repeated = [s for s in stim_files if len(partitions[s]) >= 2]
    if len(repeated) < 2:
        raise ValueError("EmoC stim-wise-multiple-folds requires at least two repeated stimuli.")
    folds = set.intersection(*(partitions[s] for s in repeated))
    if len(folds) < 2:
        raise ValueError(
            "Repeated EmoC stimuli do not share at least two partitions for cross-validation."
        )
    return repeated, folds

def within_run_classes(run_key: str, run_dict: dict, stim_types: list):
    """Class labels and their exemplar folds for one run (mah_fold='stim-wise-all-runs')."""
    known = set(stim_types)
    class_members = {}
    for stim in run_dict:
        if stim not in known:
            raise ValueError(
                f"{run_key}: condition {stim!r} is not present in the GLM stimulus list."
            )
        class_label = _strip_exemplar(stim)
        if class_label is None:
            raise ValueError(
                f"EmoC condition {stim!r} must end with its exemplar number "
                "for within-run class folding."
            )
        partition = int(stim[len(class_label):])
        members = class_members.setdefault(class_label, {})
        if partition in members:
            raise ValueError(
                f"Class {class_label!r} has multiple conditions for exemplar fold {partition}."
            )
        members[partition] = stim
    if len(class_members) < 2:
        raise ValueError("Within-run class folding requires at least two stimulus classes.")
    common = set.intersection(*(set(m) for m in class_members.values()))
    if len(common) < 2:
        raise ValueError(
            "Within-run class folding requires at least two shared exemplar folds "
            "for every stimulus class."
        )
    return class_members, common

def fold_run_options(cfg: dict) -> list:
    """Run keys that yield valid within-run class folds, for the run-scope picker."""
    options = []
    try:
        run_dicts = config_run_dicts(cfg)
    except ValueError:
        return options
    stim_types = cfg.get("stim_types") or []
    for run_key, run_dict in run_dicts:
        try:
            within_run_classes(run_key, run_dict, stim_types)
        except ValueError:
            continue
        options.append(run_key)
    return options

def derive_axis(cfg: dict, dis_method: str, mah_fold: str, run_scope: str = ALL_RUNS_KEY):
    """Rows/columns allowed by this (dis_method, mah_fold) pair.

    Returns (entities, note).  Raises ValueError with the pipeline's own message
    when the config cannot support the requested combination.
    """
    dataset    = cfg.get("dataset")
    stim_types = list(cfg.get("stim_types") or [])
    if not stim_types:
        raise ValueError("Config has no 'stim_types' — nothing to build a model from.")
    index = stim_meta_index(cfg)

    if dis_method != "mahalanobis":
        entities = [_entity(s, stim_attrs(index, s)) for s in stim_types]
        note = (f"Every pair of the {len(stim_types)} config stim_types, one map per run: "
                f"<sub>/ses-XX_task-<task>_run-XX/r-<radius>_{dis_method}_<A>_<B>.nii.gz")
        return entities, note

    if mah_fold == "stim-wise":
        members = {}
        for stim in stim_types:
            key = _stim_wise_key(dataset, stim)
            if key is not None:
                members.setdefault(key, []).append(stim)
        if len(members) < 2:
            raise ValueError(
                f"'stim-wise' collapses the {len(stim_types)} stim_types of this config into "
                f"{len(members)} categor{'y' if len(members) == 1 else 'ies'} — at least two are "
                "needed. This config is probably already at category level."
            )
        entities = [
            _entity(category, _merge_attrs([a for a in (stim_attrs(index, s) for s in stims) if a]))
            for category, stims in members.items()
        ]
        note = (f"{len(entities)} categories collapsed from {len(stim_types)} stim_types "
                f"({'strip the trailing exemplar digit' if dataset == 'EmoC' else 'text before the first hyphen'}); "
                "runs are the folds. One subject-level map per pair.")
        return entities, note

    if mah_fold in EMOC_ONLY_FOLDS and dataset != "EmoC":
        raise ValueError(
            f"mah_fold option {mah_fold!r} is only implemented for dataset 'EmoC'. "
            "For dataset 'EmoB', use 'stim-wise'."
        )

    if mah_fold == "stim-wise-multiple-folds":
        stim_files, folds = multiple_fold_stim_files(cfg)
        repeated = set(stim_files)
        by_stim_file = {}
        for run_key, run_dict in config_run_dicts(cfg):
            for stim, meta in run_dict.items():
                stim_file = meta.get("stim_file")
                if stim_file in repeated:
                    by_stim_file.setdefault(stim_file, []).append({**meta, "run": run_key,
                                                                   "__stim__": stim})
        entities = []
        for stim_file in stim_files:
            metas = by_stim_file.get(stim_file, [])
            attrs = _merge_attrs([{k: v for k, v in m.items() if k != "__stim__"} for m in metas])
            classes = _uniq([c for c in (_strip_exemplar(m["__stim__"]) for m in metas) if c])
            extra = {"class": classes[0]} if len(classes) == 1 else {}
            entities.append(_entity(stim_file, attrs, extra=extra))
        note = (f"{len(entities)} exact stimuli repeated across partitions "
                f"{sorted(folds)}; those partitions are the folds. One subject-level map per pair. "
                "The pipeline also accepts a model written at class level.")
        return entities, note

    if mah_fold == "stim-wise-all-runs":
        run_dicts = config_run_dicts(cfg)
        if run_scope and run_scope != ALL_RUNS_KEY:
            run_dicts = [(k, d) for k, d in run_dicts if k == run_scope]
            if not run_dicts:
                raise ValueError(f"Config has no metadata for {run_scope}.")
        per_run, metas_by_class = [], {}
        for run_key, run_dict in run_dicts:
            class_members, common = within_run_classes(run_key, run_dict, stim_types)
            per_run.append((run_key, list(class_members)))
            for class_label, members in class_members.items():
                for partition, stim in members.items():
                    if partition in common:
                        metas_by_class.setdefault(class_label, []).extend(index.get(stim, []))
        labels = _uniq([c for _, classes in per_run for c in classes])
        entities = [_entity(c, _merge_attrs(metas_by_class.get(c, []))) for c in labels]
        scope = "every run" if len(run_dicts) > 1 else run_dicts[0][0]
        note = (f"{len(entities)} classes ({scope}); the exemplar numbers are the folds. "
                "One map per class pair per run.")
        return entities, note

    raise ValueError(
        f"Invalid mah_fold option: {mah_fold!r}. Supported: 'stim-wise', "
        "'stim-wise-multiple-folds', 'stim-wise-all-runs'."
    )

def carry_over_matrix(old_matrix_json, old_labels, new_labels):
    """Re-use values from the previous axis for labels that survived the change."""
    fresh = fresh_matrix(len(new_labels))
    if not old_matrix_json or not old_labels:
        return fresh
    old = matrix_from_json(old_matrix_json)
    positions = {label: i for i, label in enumerate(old_labels)}
    if old.shape[0] != len(old_labels):
        return fresh
    for i, row_label in enumerate(new_labels):
        for j, col_label in enumerate(new_labels):
            if i == j or row_label not in positions or col_label not in positions:
                continue
            fresh[i, j] = old[positions[row_label], positions[col_label]]
    return fresh

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

def style_sidecar_path(csv_path):
    root, ext = os.path.splitext(csv_path or "")
    return f"{root}_style.json" if ext.lower() == ".csv" else f"{csv_path}_style.json"

def load_style_sidecar(csv_path):
    json_path = style_sidecar_path(csv_path)
    if not json_path or not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}

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

app = Dash(__name__, suppress_callback_exceptions=True, **dash_kwargs("BUILDER_URL_BASE"))
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

STEP_TITLE = {"fontWeight": "bold", "fontSize": "15px"}

def _step_no(n):
    return html.Span(n, style={
        "display": "inline-flex", "alignItems": "center", "justifyContent": "center",
        "width": "20px", "height": "20px", "borderRadius": "50%",
        "background": "#6b8dbd", "color": "#fff", "fontSize": "12px",
        "fontWeight": "bold", "marginRight": "8px", "flexShrink": "0"})

def _chips(labels, color="#eef4ff", border="#9ab", limit=None):
    shown = labels if limit is None else labels[:limit]
    out = [html.Span(str(l), style={
        "display": "inline-block", "border": f"1px solid {border}", "borderRadius": "10px",
        "padding": "1px 7px", "margin": "2px", "background": color,
        "fontFamily": "monospace", "fontSize": "11px"}) for l in shown]
    if limit is not None and len(labels) > limit:
        out.append(html.Span(f"+{len(labels) - limit} more",
                             style={"fontSize": "11px", "color": "#888", "margin": "2px"}))
    return out

def _err(msg):
    return html.Div(msg, style={"color": "#a33", "fontSize": "12px", "background": "#fff4f4",
                                "border": "1px solid #ecc", "borderRadius": "4px",
                                "padding": "6px 8px"})

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


# ── Right-click dissimilarity menu ──────────────────────────────────────────
CTX_MENU_BASE = {
    "position": "fixed", "zIndex": 3000, "minWidth": "290px",
    "background": "#ffffff", "border": "1px solid #bbb", "borderRadius": "7px",
    "boxShadow": "0 6px 20px rgba(0,0,0,0.18)", "padding": "6px 0",
    "fontFamily": "Segoe UI, Arial, sans-serif",
}
CTX_MENU_HIDDEN = {**CTX_MENU_BASE, "display": "none"}

def _ctx_menu():
    items = []
    for i, p in enumerate(DISSIM_PRESETS):
        items.append(html.Button(
            [html.Span(p["label"], style={"fontWeight": "700", "fontSize": "13px",
                                          "minWidth": "34px", "display": "inline-block"}),
             html.Span("— " + p["desc"], style={"fontSize": "11px", "color": "#666"})],
            id={"type": "ctx-preset", "idx": i}, n_clicks=0,
            className="ctx-menu-item",
            style={"display": "flex", "alignItems": "baseline", "gap": "6px",
                   "width": "100%", "textAlign": "left", "border": "none",
                   "background": "transparent", "padding": "5px 12px",
                   "cursor": "pointer"}))
    return html.Div([
        html.Div(id="ctx-menu-title",
                 style={"fontSize": "11px", "fontWeight": "700", "color": "#333",
                        "padding": "4px 12px 6px 12px", "borderBottom": "1px solid #eee",
                        "marginBottom": "4px"}),
        html.Div(items),
        html.Div([
            html.Div(f"Custom value — clamped to {DISSIM_MIN:g}–{DISSIM_MAX:g}:",
                     style={"fontSize": "11px", "color": "#555", "marginBottom": "3px"}),
            html.Div([
                # No min/max attributes on purpose: dcc.Input silently refuses to
                # propagate an out-of-range number, so the user would type 2.5,
                # press Set, and see nothing happen. The clamp callback below
                # snaps the box to [0, 1] instead, visibly.
                dcc.Input(id="ctx-manual", type="number", value=None, step=0.01,
                          placeholder="0.00 – 1.00", debounce=False,
                          style={"width": "110px", "marginRight": "6px", "fontSize": "12px"}),
                html.Button("Set", id="btn-ctx-manual", n_clicks=0,
                            style={**_B, "padding": "3px 12px"}),
            ], style={"display": "flex", "alignItems": "center"}),
        ], style={"borderTop": "1px solid #eee", "marginTop": "4px",
                  "padding": "7px 12px 3px 12px"}),
    ], id="ctx-menu", style=CTX_MENU_HIDDEN)


# Inline styles cannot express :hover, and a repo-level assets/ folder would be
# picked up by every Dash app in this directory — so scope the rule to this app.
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}<title>{%title%}</title>{%favicon%}{%css%}
        <style>
            .ctx-menu-item:hover { background: #eef4ff; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>"""


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
    dcc.Store(id="store-app-mode",   data="edit"),
    dcc.Store(id="store-ctxmenu",    data=None),
    dcc.Store(id="store-ctxhover",   data=None),
    dcc.Download(id="download-csv"),

    html.H2("RSA Model Builder", style={"marginBottom": "4px"}),
    html.Div("Build a dissimilarity model whose rows and columns match exactly what "
             "step 2 of the searchlight will produce.",
             style={"color": "#666", "marginBottom": "10px"}),

    # ── ① Config file ────────────────────────────────────────────────────────
    html.Div([
        html.Div([_step_no("1"), html.Span("Config file", style=STEP_TITLE)],
                 style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
        html.Div([
            html.Div([_lbl("Config folder"),
                      dcc.Input(id="input-config-dir", type="text", value=DEFAULT_CONFIG_DIR,
                                debounce=True,
                                style={"width": "100%", "fontSize": "12px"})],
                     style={"flex": "3", "marginRight": "8px"}),
            html.Div([_lbl(" "),
                      html.Button("⟳ Scan", id="btn-scan-configs", n_clicks=0,
                                  style={**_B, "height": "30px"},
                                  title="Re-scan the folder for .yaml configs")],
                     style={"marginRight": "12px"}),
            html.Div([_lbl("Config (D… = dog, H… = human)"),
                      dcc.Dropdown(id="dd-config", options=[], value=None,
                                   placeholder="Select a config…", clearable=False,
                                   style={"fontSize": "12px"})],
                     style={"flex": "4"}),
        ], style={"display": "flex", "alignItems": "flex-end"}),
        html.Div(id="config-summary", style={"marginTop": "8px", "fontSize": "12px"}),
    ], style={**CBOX}),

    # ── ② Dissimilarity method ───────────────────────────────────────────────
    html.Div([
        html.Div([_step_no("2"), html.Span("Dissimilarity method", style=STEP_TITLE)],
                 style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
        dcc.RadioItems(id="radio-dis-method", options=DIS_METHOD_OPTIONS,
                       value="mahalanobis", inline=True,
                       labelStyle={"marginRight": "24px", "fontSize": "13px"}),
    ], style={**CBOX}),

    # ── ③ Mahalanobis folding ────────────────────────────────────────────────
    html.Div([
        html.Div([_step_no("3"), html.Span("Mahalanobis folding (mah_fold)", style=STEP_TITLE)],
                 style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
        dcc.RadioItems(id="radio-mah-fold", options=MAH_FOLD_OPTIONS, value="stim-wise",
                       inline=True, labelStyle={"marginRight": "24px", "fontSize": "13px"}),
        html.Div(id="mah-fold-help",
                 style={"fontSize": "11px", "color": "#666", "marginTop": "6px",
                        "fontStyle": "italic", "maxWidth": "900px"}),
        html.Div([
            _lbl("Run scope"),
            dcc.Dropdown(id="dd-run", options=[], value=ALL_RUNS_KEY, clearable=False,
                         style={"width": "260px", "fontSize": "12px"}),
        ], id="section-run-scope", style={"marginTop": "8px", "display": "none"}),
    ], id="section-mah-fold", style={**CBOX}),

    # ── ④ Rows / columns ─────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Div([_step_no("4"), html.Span("Rows and columns", style=STEP_TITLE)],
                     style={"display": "flex", "alignItems": "center"}),
            html.Div([
                html.Span("View:", style={"fontSize": "12px", "color": "#555",
                                          "marginRight": "8px"}),
                dcc.RadioItems(id="radio-view",
                               options=[{"label": " Full",    "value": "full"},
                                        {"label": " Grouped", "value": "grouped"}],
                               value="full", inline=True,
                               labelStyle={"marginRight": "12px", "fontSize": "12px"}),
            ], style={"display": "flex", "alignItems": "center", "marginLeft": "auto"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
        html.Div(id="axis-summary", style={"fontSize": "12px"}),
    ], style={**CBOX}),

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
        html.Span("Mode", style={"fontWeight": "bold", "fontSize": "15px",
                                 "marginRight": "16px"}),
        dcc.RadioItems(id="radio-app-mode",
                       options=[{"label": "View", "value": "view"},
                                {"label": "Edit", "value": "edit"}],
                       value="edit", inline=True,
                       labelStyle={"marginRight": "18px", "fontWeight": "600"}),
    ], style={**CBOX, "display": "flex", "alignItems": "center",
              "border": "2px solid #6b8dbd", "background": "#f4f8ff"}),

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
    ], id="section-groupby-panel", style={**CBOX}),

    # ── Figure style panel ───────────────────────────────────────────────────
    _style_panel(),

    html.Div(id="status", style={"color": "#a33", "marginBottom": "8px"}),

    # ── Main ─────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            # clear_on_unhover keeps the right-click menu honest: without it the
            # last hovered cell would linger and a right-click on empty canvas
            # would edit whatever the mouse passed over last.
            dcc.Graph(id="heatmap", clear_on_unhover=True,
                      config={"displayModeBar": True,
                              "toImageButtonOptions": {"format": "png",
                                                       "scale": 2}}),
            html.Div("Right-click a cell to set its dissimilarity.",
                     id="ctx-hint",
                     style={"fontSize": "11px", "color": "#888", "fontStyle": "italic",
                            "marginTop": "2px"}),
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
            ], id="section-cell-edit", style={"marginTop": "6px"}),
        ], style={"flex": "3", "marginRight": "12px"}),

        html.Div([
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
            ], id="section-bulk-rules"),
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

    # ── Right-click dissimilarity menu ───────────────────────────────────────
    # Top level (not inside the graph column) so no ancestor's overflow can clip
    # it; position:fixed is set from the click coordinates by the callback below.
    _ctx_menu(),
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
# Clientside — right-click cell menu
# ===========================================================================

# Plotly has no right-click event, so the cell under the cursor is taken from
# the hover of the invisible marker trace build_cell_heatmap() draws on top of
# the shapes. Kept on `window` because the contextmenu listener below is a plain
# DOM handler and cannot read Dash state.
app.clientside_callback(
    """
    function(hov, mode) {
        window._rsa_edit_mode = (mode !== 'view');
        var h = null;
        if (hov && hov.points && hov.points.length) {
            var cd = hov.points[0].customdata;
            if (cd && cd.length >= 2) h = {row: cd[0], col: cd[1]};
        }
        window._rsa_hover = h;
        return window.dash_clientside.no_update;
    }
    """,
    Output("store-ctxhover", "data"),
    Input("heatmap", "hoverData"),
    Input("radio-app-mode", "value"),
)

app.clientside_callback(
    """
    function(_ignore) {
        if (!window._rsa_ctx_bound) {
            window._rsa_ctx_bound = true;
            var open = function(d) {
                window.dash_clientside.set_props('store-ctxmenu', {data: d});
            };
            document.addEventListener('contextmenu', function(e) {
                var plot = document.getElementById('heatmap');
                if (!plot || !plot.contains(e.target)) return;
                if (!window._rsa_edit_mode) return;       // View mode: read-only
                var h = window._rsa_hover;
                if (!h) return;                           // not over a cell
                e.preventDefault();
                open({row: h.row, col: h.col, x: e.clientX, y: e.clientY, ts: Date.now()});
            });
            // Any click outside the menu, Escape, or a scroll dismisses it.
            document.addEventListener('mousedown', function(e) {
                if (e.button === 2) return;   // right-click: let contextmenu re-open it
                var m = document.getElementById('ctx-menu');
                if (m && m.style.display !== 'none' && !m.contains(e.target)) open(null);
            });
            document.addEventListener('keydown', function(e) {
                if (e.key === 'Escape') open(null);
            });
            window.addEventListener('scroll', function() {
                var m = document.getElementById('ctx-menu');
                if (m && m.style.display !== 'none') open(null);
            }, true);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("store-ctxmenu", "data"),
    Input("heatmap", "id"),
)

# Hard-clamp the custom value to [0, 1] as it is typed — `min`/`max` on a number
# input only mark it invalid in the browser, they do not stop the value arriving.
app.clientside_callback(
    """
    function(v) {
        if (v === null || v === undefined || v === '') return window.dash_clientside.no_update;
        var n = parseFloat(v);
        if (isNaN(n)) return null;
        var c = Math.min(%s, Math.max(%s, n));
        return (c === n) ? window.dash_clientside.no_update : c;
    }
    """ % (DISSIM_MAX, DISSIM_MIN),
    Output("ctx-manual", "value"),
    Input("ctx-manual", "value"),
    prevent_initial_call=True,
)

# ===========================================================================
# Callbacks
# ===========================================================================

# ── YAML / run ────────────────────────────────────────────────────────────
@app.callback(
    Output("section-bulk-rules",    "style"),
    Output("section-cell-edit",     "style"),
    Output("section-groupby-panel", "style"),
    Output("ctx-hint",              "style"),
    Input("radio-app-mode",         "value"),
)
def toggle_app_mode(mode):
    hint = {"fontSize": "11px", "color": "#888", "fontStyle": "italic", "marginTop": "2px"}
    if mode == "view":
        hidden = {"display": "none"}
        return hidden, hidden, hidden, hidden
    return ({"display": "block"},
            {"marginTop": "6px", "display": "block"},
            {**CBOX, "display": "block"},
            {**hint, "display": "block"})


# ── ① Config file ─────────────────────────────────────────────────────────
@app.callback(
    Output("dd-config",       "options"),
    Output("dd-config",       "value"),
    Input("btn-scan-configs", "n_clicks"),
    Input("input-config-dir", "value"),
    State("dd-config",        "value"),
    prevent_initial_call=False,
)
def scan_configs_cb(n, folder, current):
    names = scan_config_files((folder or "").strip())
    opts  = [{"label": config_file_label(f), "value": f} for f in names]
    if current in names:
        return opts, current
    return opts, (DEFAULT_CONFIG if DEFAULT_CONFIG in names else (names[0] if names else None))


@app.callback(
    Output("store-cfg",       "data"),
    Output("config-summary",  "children"),
    Output("export-folder",   "value"),
    Output("status",          "children"),
    Input("dd-config",        "value"),
    Input("input-config-dir", "value"),
)
def load_config_cb(name, folder):
    folder = (folder or "").strip()
    if not name:
        return None, _err(f"No .yaml config found in: {folder or '(no folder)'}"), \
               no_update, "Select a config file to start."
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        # Transient while a folder change re-scans; the new options arrive next.
        return None, _err(f"Config not found: {path}"), no_update, f"Config not found: {path}"
    try:
        cfg = load_yaml(path) or {}
    except Exception as e:
        return None, _err(f"Failed to parse YAML: {e}"), no_update, ""
    cfg["__path__"] = path

    specie  = specie_from_filename(name) or cfg.get("specie") or "?"
    runs    = cfg.get("runs") or []
    stims   = cfg.get("stim_types") or []
    fact = lambda k, v: html.Span([
        html.Span(f"{k} ", style={"color": "#777"}),
        html.B(str(v))], style={"marginRight": "18px"})
    summary = html.Div([
        html.Div([
            fact("dataset", cfg.get("dataset", "?")),
            fact("specie", f"{specie} ({SPECIE_NAMES.get(specie, 'unknown')})"),
            fact("GLM model", cfg.get("model", "?")),
            fact("task", cfg.get("task", "?")),
            fact("runs", len(runs)),
            fact("participants", len(cfg.get("participants") or [])),
        ], style={"marginBottom": "4px"}),
        html.Div([
            html.Span(f"stim_types ({len(stims)}): ",
                      style={"color": "#777", "marginRight": "4px"}),
            *_chips(stims, limit=24),
        ]),
    ])
    return (cfg, summary, export_dir_for_config(path),
            f"Loaded {path}")


# ── ③ Mahalanobis folding ─────────────────────────────────────────────────
@app.callback(
    Output("section-mah-fold",  "style"),
    Output("radio-mah-fold",    "options"),
    Output("radio-mah-fold",    "value"),
    Input("radio-dis-method",   "value"),
    Input("store-cfg",          "data"),
    State("radio-mah-fold",     "value"),
)
def sync_mah_fold(dis_method, cfg, current):
    if dis_method != "mahalanobis":
        return {"display": "none"}, MAH_FOLD_OPTIONS, current or "stim-wise"
    dataset = (cfg or {}).get("dataset")
    opts = []
    for opt in MAH_FOLD_OPTIONS:
        blocked = opt["value"] in EMOC_ONLY_FOLDS and dataset not in (None, "EmoC")
        label = opt["label"] + (f"  (EmoC only — this config is {dataset})" if blocked else "")
        opts.append({"label": label, "value": opt["value"], "disabled": blocked})
    allowed = [o["value"] for o in opts if not o["disabled"]]
    return {**CBOX, "display": "block"}, opts, (current if current in allowed else "stim-wise")


@app.callback(
    Output("mah-fold-help",   "children"),
    Input("radio-mah-fold",   "value"),
    Input("radio-dis-method", "value"),
)
def show_mah_fold_help(mah_fold, dis_method):
    return MAH_FOLD_HELP.get(mah_fold, "") if dis_method == "mahalanobis" else ""


@app.callback(
    Output("section-run-scope", "style"),
    Output("dd-run",            "options"),
    Output("dd-run",            "value"),
    Input("radio-dis-method",   "value"),
    Input("radio-mah-fold",     "value"),
    Input("store-cfg",          "data"),
    State("dd-run",             "value"),
)
def sync_run_scope(dis_method, mah_fold, cfg, current):
    # Only stim-wise-all-runs produces a different label set per run; every other
    # combination aggregates over all runs.
    if dis_method != "mahalanobis" or mah_fold != "stim-wise-all-runs" or not cfg:
        return {"display": "none"}, [], ALL_RUNS_KEY
    runs = fold_run_options(cfg)
    opts = [{"label": "All runs (union of classes)", "value": ALL_RUNS_KEY}]
    opts += [{"label": r, "value": r} for r in runs]
    values = [o["value"] for o in opts]
    return ({"marginTop": "8px", "display": "block"}, opts,
            current if current in values else ALL_RUNS_KEY)


# ── ④ Rows / columns ──────────────────────────────────────────────────────
@app.callback(
    Output("store-stims",     "data"),
    Output("store-matrix",    "data"),
    Output("store-meta",      "data"),
    Output("axis-summary",    "children"),
    Input("store-cfg",        "data"),
    Input("radio-dis-method", "value"),
    Input("radio-mah-fold",   "value"),
    Input("dd-run",           "value"),
    State("store-matrix",     "data"),
    State("store-meta",       "data"),
)
def build_axis(cfg, dis_method, mah_fold, run_scope, matrix_data, meta):
    if not cfg:
        return [], [[]], {"combined": False}, _err("Load a config file first.")
    fold = mah_fold if dis_method == "mahalanobis" else None
    try:
        entities, note = derive_axis(cfg, dis_method, fold, run_scope or ALL_RUNS_KEY)
    except Exception as e:
        combo = f"dis_method={dis_method!r}" + (f", mah_fold={fold!r}" if fold else "")
        return ([], [[]], {"combined": False},
                _err([html.B("This config cannot produce that combination "), f"({combo}). ",
                      html.Br(), str(e)]))

    labels = [e["name"] for e in entities]
    matrix = carry_over_matrix(matrix_data, (meta or {}).get("labels"), labels)
    n_pairs = len(labels) * (len(labels) - 1) // 2
    summary = html.Div([
        html.Div([html.B(f"{len(labels)} labels"),
                  html.Span(f" · {n_pairs} pairs to fill", style={"color": "#777"})],
                 style={"marginBottom": "4px"}),
        html.Div(_chips(labels, color="#eaf6ea", border="#9c9")),
        html.Div(note, style={"fontSize": "11px", "color": "#666", "marginTop": "6px",
                              "fontStyle": "italic"}),
    ])
    new_meta = {
        "combined":   False,
        "labels":     labels,
        "config":     cfg.get("__path__"),
        "dataset":    cfg.get("dataset"),
        "specie":     cfg.get("specie"),
        "glm_model":  cfg.get("model"),
        "dis_method": dis_method,
        "mah_fold":   fold,
        "run_scope":  (run_scope or ALL_RUNS_KEY) if fold == "stim-wise-all-runs" else None,
    }
    return entities, matrix_to_json(matrix), new_meta, summary


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
    Input("export-folder",     "value"),
    State("store-last-model",  "data"),
    prevent_initial_call=False,
)
def scan_models_cb(n, folder, last_model):
    folder = (folder or DEFAULT_EXPORT_DIR).strip()
    files = scan_model_files(folder)
    opts  = [{"label": f, "value": os.path.join(folder, f)} for f in files]
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
    Output("store-app-mode",   "data"),
    Output("radio-app-mode",   "value"),
    Output("store-groupby",    "data", allow_duplicate=True),
    Output("store-sep",        "data", allow_duplicate=True),
    Output("radio-view",       "value"),
    Output("store-style",      "data", allow_duplicate=True),
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
        return (no_update, no_update, no_update, no_update,
                "No model selected or no stims loaded.",
                no_update, no_update, no_update, no_update, no_update, no_update)
    combined    = bool(meta and meta.get("combined"))
    stim_labels = [display_name(s, combined) for s in stims]
    try:
        mf, n_matched = load_model_into_matrix(fpath, stim_labels, matrix_data)
    except Exception as e:
        return (no_update, no_update, no_update, no_update, f"Error: {e}",
                no_update, no_update, no_update, no_update, no_update, no_update)
    stack = list(undo_stack or [])
    stack.append(matrix_data)
    if len(stack) > MAX_UNDO: stack = stack[-MAX_UNDO:]
    saved_style = load_style_sidecar(fpath)
    return (matrix_to_json(mf), stack, [], fpath,
            f"Loaded {os.path.basename(fpath)} ({n_matched} cells matched).",
            "view",
            "view",
            saved_style.get("group_by", no_update),
            saved_style.get("separator", no_update),
            saved_style.get("view_mode", no_update),
            saved_style.get("figure_style", no_update))


# ── Group-by ──────────────────────────────────────────────────────────────
@app.callback(
    Output("store-groupby", "data"),
    Input("store-stims",    "data"),
    State("store-groupby",  "data"),
    prevent_initial_call=True,
)
def init_groupby(stims, cur):
    # Default to no grouping: the exported CSV must carry the axis labels the
    # pipeline looks up, and a grouped view exports the group labels instead.
    if not stims: return []
    attrs = discover_attrs(stims)
    return [k for k in (cur or []) if k in attrs]


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


# ── Render heatmap ────────────────────────────────────────────────────────
@app.callback(
    Output("heatmap",      "figure"),
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
        return go.Figure()
    mf       = matrix_from_json(matrix_data)
    combined = bool(meta and meta.get("combined"))
    S        = {**DEFAULT_STYLE, **(style or {})}
    labels, mapping, m, mixed = _current_view(stims, mf, view_mode,
                                              group_by or [], combined, "_" if sep is None else sep)
    axis_col = representative_color(stims, mapping, len(labels))
    return build_cell_heatmap(m, labels, S, mixed_mask=mixed, axis_colors=axis_col)


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


# ── Right-click menu: show / place / dismiss ──────────────────────────────
@app.callback(
    Output("ctx-menu",       "style"),
    Output("ctx-menu-title", "children"),
    Output("ctx-manual",     "value", allow_duplicate=True),
    Input("store-ctxmenu",   "data"),
    Input({"type": "ctx-preset", "idx": ALL}, "n_clicks"),
    Input("btn-ctx-manual",  "n_clicks"),
    prevent_initial_call=True,
)
def ctx_menu_toggle(ctxdata, _presets, _manual):
    # Any of the menu's own buttons closes it; only store-ctxmenu opens it.
    if ctx.triggered_id != "store-ctxmenu" or not ctxdata:
        return CTX_MENU_HIDDEN, no_update, None
    # Measured at ~350 × 215 px; clamp so a click near the right/bottom edge
    # still lands the whole menu inside the window.
    x = max(4, int(ctxdata.get("x", 0)))
    y = max(4, int(ctxdata.get("y", 0)))
    style = {**CTX_MENU_BASE, "display": "block",
             "left": f"max(4px, min({x}px, 100vw - 360px))",
             "top":  f"max(4px, min({y}px, 100vh - 230px))"}
    title = f'{ctxdata.get("row", "?")}  ×  {ctxdata.get("col", "?")}'
    return style, title, None


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
    Input({"type": "ctx-preset", "idx": ALL}, "n_clicks"),
    Input("btn-ctx-manual",    "n_clicks"),
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
    State("store-ctxmenu",     "data"),
    State("ctx-manual",        "value"),
    State("store-undo-stack",  "data"),
    prevent_initial_call=True,
)
def edit_matrix(n_set, n_bulk, n_fill, n_same0, n_reset, n_reset_model,
                n_mirror, n_ctx_presets, n_ctx_manual,
                matrix_data, stims, meta, view_mode, group_by, sep,
                cell_row, cell_col, cell_val,
                lhs_attr, lhs_val, rhs_attr, rhs_val, bulk_val, only_nan_chk,
                ctxdata, ctx_manual, undo_stack):
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

    # Right-click menu — a preset button, or the custom 0–1 box next to it.
    is_preset = isinstance(trigger, dict) and trigger.get("type") == "ctx-preset"
    if is_preset or trigger == "btn-ctx-manual":
        if not ctxdata:
            return no_update, no_update, no_update
        if is_preset:
            idx = trigger.get("idx")
            if not (0 <= idx < len(DISSIM_PRESETS)):
                return no_update, no_update, no_update
            val = DISSIM_PRESETS[idx]["value"]
            val = np.nan if val is None else float(val)
        else:
            val = parse_value(ctx_manual)
            if math.isnan(val):
                return no_update, no_update, no_update   # empty box: nothing to set
            val = float(np.clip(val, DISSIM_MIN, DISSIM_MAX))
        labels, mapping, _, _ = _current_view(stims, mf, view_mode, group_by or [], combined, sep)
        row, col = ctxdata.get("row"), ctxdata.get("col")
        if row not in labels or col not in labels:
            return no_update, no_update, no_update
        gi, gj = labels.index(row), labels.index(col)
        if view_mode == "full" or not group_by: set_pair(mf, gi, gj, val)
        else: broadcast_grouped_edit(mf, mapping, gi, gj, val)
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
    meta     = meta or {}
    mf       = matrix_from_json(matrix_data)
    combined = bool(meta.get("combined"))
    S        = {**DEFAULT_STYLE, **(style or {})}
    labels, _, m, _ = _current_view(stims, mf, view_mode, group_by or [], combined, "_" if sep is None else sep)
    csv_text = dataframe_to_csv_string(to_export_dataframe(m, labels))
    fname = ((fname or "my-model.csv").strip())
    if not fname.lower().endswith(".csv"): fname += ".csv"

    # The pipeline looks the model up by the labels it writes into the pairwise
    # map filenames — warn loudly when a grouped view collapses them away.
    warning = []
    axis_labels = meta.get("labels") or []
    if axis_labels and labels != axis_labels:
        if meta.get("mah_fold") == "stim-wise-multiple-folds":
            warning = [f"Grouped export: {len(labels)} labels instead of the "
                       f"{len(axis_labels)} exact stimuli. The pipeline can resolve this only if "
                       "every label is a class name (e.g. DogA) that each stim_file maps to."]
        else:
            warning = [f"⚠ Grouped export: this CSV has {len(labels)} labels but "
                       f"dis_method={meta.get('dis_method')!r}"
                       + (f" / mah_fold={meta.get('mah_fold')!r}" if meta.get("mah_fold") else "")
                       + f" produces maps for {len(axis_labels)} labels — it will not match. "
                         "Switch Group by to none before exporting."]

    saved_msg = ""
    if folder:
        try:
            os.makedirs(folder, exist_ok=True)
            csv_path  = os.path.join(folder, fname)
            json_path = style_sidecar_path(csv_path)
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                f.write(csv_text)
            opts = style_to_summary(S, group_by, sep)
            opts["view_mode"]   = view_mode
            opts["exported_at"] = str(pd.Timestamp.now())
            opts["labels"]      = labels
            opts["build"]       = {k: meta.get(k) for k in
                                   ("config", "dataset", "specie", "glm_model",
                                    "dis_method", "mah_fold", "run_scope")}
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(opts, f, indent=2)
            saved_msg = f"Saved: {csv_path}  +  {os.path.basename(json_path)}"
            # read_model_dict() caches a .npy next to the CSV and prefers it —
            # a stale one would silently shadow this export.
            npy_path = csv_path[:-4] + ".npy"
            if os.path.exists(npy_path):
                os.remove(npy_path)
                saved_msg += f"  ·  removed stale {os.path.basename(npy_path)}"
        except Exception as e:
            saved_msg = f"Save failed: {e}"
    children = ([html.Div(w, style={"color": "#a33", "marginBottom": "4px"}) for w in warning]
                + [html.Div(saved_msg)])
    return dict(content=csv_text, filename=fname), children


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8051)

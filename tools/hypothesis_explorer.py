#!/usr/bin/env python
"""
hypothesis_explorer.py — EmoC RSA model explorer (standalone Dash app).

This app is a **row of self-contained model cards** — no hypothesis tree. You
**add** and **remove** cards, and each card is one RSA model you want to look at:

  * Pick a **Mahalanobis fold** (``mah_fold``: stim-wise / run-wise / …) first —
    that decides which model families and groupings the card offers. Then pick a
    **model** (a hypothesis *stem*) and a **grouping** (all / collapse / within /
    cross / dog / hum). Together they resolve to a concrete ``{stem}__{grouping}``
    model. The fold → models → groupings menu is driven by the dataset's central
    ``rsa_models/_models.csv`` manifest (built by ``tools/build_models_manifest.py``);
    edit that one file to add, retire, or re-group models. When the manifest is
    absent the card falls back to scanning the folder and offering every valid
    ``__{grouping}`` model under one synthetic fold.
  * Each card is **one species** — its own column: the **Species** control picks
    **Dog** or **Human** and the card draws that species' results map as a 2D atlas
    slice (put Dog and Human side by side in two cards). The map type defaults to
    the group **mean** and can be switched to the z-map or the cluster-corrected
    map; axis, slice position, a two-handle **range slider** (low/high threshold)
    and **colormap** are per-card. The colormap defaults to **Hot**: voxels below
    the range's low handle render transparent (alpha=0), everything at/above the
    high handle is painted the top color of the scale.
  * **Next to the brain** sits a **histogram of that map's values inside the
    search mask** — only the voxels the searchlight actually visited, read from
    the ROI/mask file itself (dog masks from ``Atlas/Dog/Nitzsche/``, human ones
    from ``{dataset}/ROI/H/``, tried against both the active data folder and the
    pipeline disk; a mask on a different grid is resampled onto the map's). It
    shares the brain view's height and the card's colormap: bars below the range
    slider's low handle are grey — exactly the voxels drawn transparent on the
    slice — and bars at/above it are coloured on the same [low, high] scale, with
    the two handles marked as vertical lines. Counts are on a log axis because a
    statistical map is overwhelmingly near-zero voxels. Toggle **📊 histogram**
    off to hide it and give the slice the full card width. The note under the
    slider reports ``supra-threshold / in-mask`` voxel counts.
  * Toggle **🔗 sync** to mirror the view (slice, axis, range, colormap)
    across every *other synced card of the same species*: move the slice on one and
    the matching-species cards follow, scales included. Dog and Human sync
    independently.
  * The card also shows the **model's dissimilarity matrix**, rendered exactly as
    in the RSA Model Builder. Toggle **show matrix** off to hide it.

Use **➕ Add model** to bring on another card and a card's ✕ to remove it (up to
6 slots). Toggle **✏️ Edit** in the top bar to **reorder** the row by dragging a
card's header (that is the only edit gesture — cards are not resizable); the
**Gap** box sets the spacing and **Reset order** restores the default.

Result source (top bar)
-----------------------
The maps can be read from **either** of two on-disk layouts:

  * **Drive (current-results)** — the flat Google-Drive mirror that
    ``sync_rsa_to_drive.py`` / ``create_tables`` write
    (``{root}/{dataset}/current-results/RSA/{specie}/{mask}/{specie}_{model}_z.nii.gz``).
  * **Raw (results/RSA)** — the pipeline's own nested output that
    ``pipeline_dashboard.py`` probes
    (``{root}/{dataset}/results/RSA/{glm_model}/{model}/mean/{mask}-{specie}-r-{radius}_{method}_{rsa_method}_z.nii.gz``),
    e.g. ``P:\\userdata\\raulh87\\data\\EmoC\\results\\RSA``. This lets you inspect
    results that exist on the pipeline disk before they are synced to the mirror.

Display + persistence
---------------------
Brain-view height is adjustable in the top bar; the source mode, data folder,
dataset, view height and the **card layout** (order + gap set in Edit mode) are
saved to ``~/.rsa_hypothesis_explorer_settings.json`` and restored on the next
launch. Each card's own selections (model, grouping, species, map type, axis,
colormap, max, sync, histogram + matrix show/hide, on/off) are persisted by
Dash's local persistence, so the cards come back as you left them.

Auto-update / manual update
----------------------------
The top-bar **Auto-update** toggle (on by default) controls whether changing a
card's fold/model/grouping/species/map-type/axis/threshold/colormap/max
re-renders that card's map immediately. Turn it off to batch several changes
and apply them together with **🔄 Update now**. The **slice** slider always
updates live regardless of this toggle (it's cheap and you want to scrub it),
as do card on/off, the histogram and matrix show/hide toggles, and the top-bar
source/ROI/reload/view-height controls. A gated card shows a "pending
changes" note until you click Update. The status line next to **Models** in
the header (normally the fold/model-family count) doubles as a general
feedback line: it also reports reloads, add/remove-card, layout-reset and
update actions as they happen.

Threshold (per card)
---------------------
A single two-handle **range slider** sets both bounds: the **low** handle
filters voxels below it out (rendered transparent, alpha=0); the **high**
handle caps the color scale — voxels at or above it are painted the top color
of the palette. The slider's meaning and limits depend on the card's map
type: for **Z-map** / **Cluster-corrected** it's a z-range (0-8, default
[3.1, 8]); for **Group average** it's an average-similarity range, typically
Kendall's tau (-1 to 1, default [0, 1]) since that's this pipeline's default
RSA method. Switching map type updates the slider's limits and marks; a
still-valid current [low, high] is kept, otherwise it resets to that mode's
default.

Standalone only (own port, default 8055):
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\hypothesis_explorer.py
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\hypothesis_explorer.py --port 8056

Reuses viz/datasource.py (result resolution), viz/niftiutil.py (atlas + 2D slices)
and viz/hypothesis_tree.py (result-set scan + per-model status).
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, ctx, dcc, html, no_update
from dash.dependencies import Input, Output, State

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)  # tools/ lives one level below the repo root
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from viz import datasource, niftiutil, hypothesis_tree as ht
from scheduler.paths import get_paths   # canonical model home (pipeline data disk)
import models_manifest as mm   # central _models.csv reader (fold → models → groupings)
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
DEFAULT_GLM_MODEL = "basic-block"
MODALITIES = ["RSA", "GLM"]
MAPTYPES = [("mean", "Group average"), ("z", "Z-map"), ("corrected", "Cluster-corrected")]
AXES = [("0", "Slice X"), ("1", "Slice Y"), ("2", "Slice Z")]
SOURCE_MODES = [("drive", "Drive (current-results)"), ("raw", "Raw (results/RSA)")]
# Maps display for a card: which single species' results map to draw. Each card
# is one species (one column) — "Both" is gone; put Dog and Human in two cards.
MAPS_OPTIONS = [("D", "Dog"), ("H", "Human")]
# Overlay colour maps offered per card. Default "Hot": sub-threshold voxels are
# transparent (alpha=0), at/above-threshold values ride the hot scale.
COLORMAPS = ["Hot", "YlOrRd", "Reds", "Viridis", "Cividis", "Jet", "Turbo", "Greys", "Blues"]
DEFAULT_CMAP = "Hot"
# Per-card view controls that the "sync" toggle shares across same-species cards.
SYNC_CONTROLS = ["axis", "frac", "range", "cmap"]
MAX_MODELS = 6             # total model-card slots, pre-registered; "Add model" turns the
                           # next one on and its own ✕ turns it off (add / remove)
DEFAULT_CARD_W = 360       # model-card base width (flex-basis, px); cards flex-grow to fill
DEFAULT_GAP = 10           # space between model cards, px
CORRECTED_ZT_TRIES = [3.1, 2.3, 3.9]
HIST_BINS = 60             # bins in the in-mask value histogram drawn beside the brain
HIST_SUB_COLOR = "#c9ced6"  # bar colour below the low handle (those voxels are transparent)

# status -> (colour, human label) for the per-card results-availability dot
STATUS_STYLE = {
    "both":     ("#1a7f37", "Dog + Human"),
    "D":        ("#3b7dd8", "Dog only"),
    "H":        ("#e08a1e", "Human only"),
    "none":     ("#cf4b4b", "No results"),
    "unlinked": ("#c9ced6", "No model"),
}

# --- caches (module-level; keyed so re-selecting is instant) --------------
_ATLAS = {}          # specie -> (hi, hi_aff, lo_aff, lo_shape)
_ATLAS_ON_GRID = {}  # (specie, shape, aff_hash) -> atlas resampled onto overlay grid
_MAP_CACHE = {}      # (source,datafolder,dataset,modality,roi,glm,specie,model,maptype,zt) -> (data,aff) | None
_RESULT_SETS_CACHE = {}  # source-keyed {'D':set,'H':set} of models with results
_MASK_CACHE = {}     # (datafolder,dataset,specie,roi) -> (data,aff,path) | None
_MASK_ON_GRID = {}   # (datafolder,dataset,specie,roi,shape,aff_hash) -> bool array | None


def _int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Persisted display / source settings
# ---------------------------------------------------------------------------
# Everything that shapes how the page comes up is written here on every change
# and restored at launch, so "next session loads like before". Kept per-user off
# the shared data disk (mirrors pipeline_dashboard's cache convention).
SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".rsa_hypothesis_explorer_settings.json")

# Canonical grouping order. "all" (stim-wise pooled) and "collapse" (run-wise
# pooled) both mean "everything together"; kept first so they lead the menus.
GROUPINGS = ["all", "collapse", "within", "cross", "dog", "hum"]
GROUPING_DESC = {
    "all":      "all stimulus pairs (Dog/Hum pooled)",
    "collapse": "collapsed across agent species (Dog/Hum pooled)",
    "within":   "within agent species only (Dog-Dog & Hum-Hum)",
    "cross":    "cross agent species only (Dog-Hum) — agent-invariant test",
    "dog":      "Dog-shown block only",
    "hum":      "Hum-shown block only",
}

# --- model-card layout (edit mode) ----------------------------------------
# In "Edit" mode the cards can be **reordered** by dragging a card's header (that
# is the only edit gesture — cards are not resizable). The gap between cards is a
# separate top-bar preference. The whole arrangement lives in this small dict, is
# the single source of truth for the cards' order + spacing, and is saved with the
# rest of the settings so it comes back exactly as left.
#   order : per-card-index CSS flex ``order`` (visual position, 0 = leftmost)
#   gap   : space between cards in px

def _default_layout():
    return {"order": list(range(MAX_MODELS)), "gap": DEFAULT_GAP}


def _clean_layout(layout):
    """Coerce a (possibly stale / partial / user-tampered) layout dict into a valid
    one: order is a length-``MAX_MODELS`` int list, gap clamped to a reasonable
    range. Tolerates and drops legacy keys (e.g. old per-card ``widths``). Never
    raises."""
    d = layout if isinstance(layout, dict) else {}
    seq = d.get("order")
    seq = seq if isinstance(seq, list) else []
    order = []
    for i in range(MAX_MODELS):
        try:
            order.append(int(seq[i]))
        except (IndexError, TypeError, ValueError):
            order.append(i)
    try:
        gap = max(0, min(80, int(d.get("gap"))))
    except (TypeError, ValueError):
        gap = DEFAULT_GAP
    return {"order": order, "gap": gap}


def _layout_get(layout):
    d = _clean_layout(layout)
    return d["order"], d["gap"]


DEFAULT_SETTINGS = {
    "source_mode": "drive",
    "datafolder": None,        # None -> resolved at load from the source mode
    "glm_model": DEFAULT_GLM_MODEL,
    "dataset": DEFAULT_DATASET,
    "modality": "RSA",
    "view_height": 230,        # brain-view (2D slice) graph height, px
    "layout": _default_layout(),    # model-card order + gap (edit mode)
}


def load_settings():
    s = dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            s.update({k: v for k, v in saved.items() if k in DEFAULT_SETTINGS})
    except Exception:
        pass
    s["layout"] = _clean_layout(s.get("layout"))   # order + gap always valid
    return s


def save_settings(s):
    try:
        tmp = SETTINGS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
        os.replace(tmp, SETTINGS_PATH)
    except Exception as e:
        print(f"[hypothesis_explorer] warning: could not save settings: {e}")


def _initial_datafolder(s):
    """The data folder to seed the top bar with, honouring a saved override then
    the source mode (raw -> pipeline disk, drive -> best current-results root)."""
    if s.get("datafolder"):
        return s["datafolder"]
    if s.get("source_mode") == "raw":
        try:
            return get_paths()[0]
        except Exception:
            pass
    return datasource.resolve_datafolder(s.get("dataset") or DEFAULT_DATASET)


SETTINGS = load_settings()


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


# --- The search mask (which voxels the searchlight actually covered) --------
# The card's "ROI / mask" selection names a mask *file*, and the histogram beside
# the brain is computed over exactly those voxels. Where that file lives depends
# on the species, mirroring how ``searchlight.py`` resolves ``mask_type``:
#   * Dog  -> the toolkit's own atlas tree, ``Atlas/Dog/Nitzsche/{mask}.nii.gz``
#   * Human-> the dataset's mask tree, ``{root}/{dataset}/ROI/H/{mask}.nii.gz``
# and ``cope13``-style masks live under ``ROI/{specie}/`` for both. Because the
# explorer may be pointed at the Google Drive mirror (which carries no ``ROI/``
# tree), every dataset-relative candidate is tried against the active data folder
# *and* the pipeline data disk, the same two roots ``_model_dirs`` searches.

def _data_roots(datafolder):
    """[active data folder, pipeline data disk], de-duplicated."""
    roots = []
    if datafolder:
        roots.append(datafolder)
    try:
        roots.append(get_paths()[0])
    except Exception:
        pass
    seen, out = set(), []
    for r in roots:
        key = os.path.normcase(os.path.abspath(r))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _mask_candidates(datafolder, dataset, specie, roi):
    """Ordered paths to try for the mask named ``roi``, species-appropriate first."""
    name = roi if roi.endswith((".nii", ".nii.gz")) else f"{roi}.nii.gz"
    cands = []
    if specie == "D":
        cands.append(os.path.join(_REPO_ROOT, "Atlas", "Dog", "Nitzsche", name))
    for root in _data_roots(datafolder):
        cands.append(os.path.join(root, dataset or "", "ROI", specie, name))
        cands.append(os.path.join(root, dataset or "", "ROI", name))
    if specie == "H":
        cands.append(os.path.join(_REPO_ROOT, "Atlas", "Hum", name))
    return cands


def _load_mask(datafolder, dataset, specie, roi):
    """(data, affine, path) for the ROI mask, or None when no candidate exists /
    loads. Cached per (root, dataset, species, mask) — this reads the network disk."""
    if not roi or roi in ("(none)",):
        return None
    key = (datafolder, dataset, specie, roi)
    if key not in _MASK_CACHE:
        found = None
        for p in _mask_candidates(datafolder, dataset, specie, roi):
            if not os.path.isfile(p):
                continue
            try:
                data, aff, _hdr = niftiutil.load_nifti(p)
            except Exception:
                continue          # unreadable — keep looking
            found = (data, aff, p)
            break
        _MASK_CACHE[key] = found
    return _MASK_CACHE[key]


def _mask_bool_for_grid(datafolder, dataset, specie, roi, shape, aff):
    """(in-mask boolean selector **on the map's own voxel grid**, resampled?) — or
    (None, False) when the mask file could not be resolved.

    A mask that does not already sit on the map's grid is nearest-neighbour
    resampled onto it (the same helper that puts the atlas there), so the
    histogram never mixes grids. The flag is passed on to the caller because a
    resample here is a *warning sign*, not routine: results and their search mask
    are supposed to share one voxel grid (see the "hard invariant" in CLAUDE.md),
    and when they don't, the map itself is suspect — not just this histogram."""
    key = (datafolder, dataset, specie, roi, tuple(shape), hash(aff.tobytes()))
    if key not in _MASK_ON_GRID:
        loaded = _load_mask(datafolder, dataset, specie, roi)
        out, resampled = None, False
        if loaded is not None:
            mdata, maff, _path = loaded
            if tuple(mdata.shape) != tuple(shape) or not np.allclose(maff, aff):
                mdata = niftiutil.resample_lowres_to_highres(mdata, maff, shape, aff)
                resampled = True
            out = np.isfinite(mdata) & (np.abs(mdata) > 0)
        _MASK_ON_GRID[key] = (out, resampled)
    return _MASK_ON_GRID[key]


def _mask_values(data, datafolder, dataset, specie, roi, aff):
    """(values, axis note) — the map's values inside the search mask, plus a short
    label saying where they came from. Falls back to the map's own non-zero voxels
    when no mask file can be found, and says so, rather than silently
    histogramming the whole (mostly empty) bounding box."""
    mb, resampled = _mask_bool_for_grid(datafolder, dataset, specie, roi, data.shape, aff)
    if mb is None:
        return data[np.abs(data) > 1e-6], "value · non-zero voxels (no mask file found)"
    if resampled:
        return data[mb], f"value · ⚠ {roi} resampled — mask and map are on different grids"
    return data[mb], f"value · voxels in {roi}"


# --- Drive (current-results) layout ---------------------------------------

def _drive_map_path(datafolder, dataset, modality, roi, specie, model, maptype, zt):
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


# --- Raw (results/RSA) layout ---------------------------------------------
# {datafolder}/{dataset}/results/RSA/{glm_model}/{model}/mean/
#     {mask}-{specie}-r-{radius}_{method}_{rsa_method}_{kind}.nii.gz
# The method/radius/mask vary per model (correlation vs mahalanobis, r-3 vs r-4),
# so instead of reconstructing the exact filename we glob the mean folder and pick
# the file for this species (matched on the "-{specie}-r-" segment).

# Matches a group map filename so we can read its mask prefix for the ROI menu.
_RAW_NAME_RE = re.compile(
    r"^(?:(?P<mask>.+)-)?(?P<specie>[DH])-r-\d+_[A-Za-z0-9]+_[A-Za-z0-9]+_"
    r"(?:mean|z|std)\.nii\.gz$"
)


def _raw_rsa_root(datafolder, dataset, glm_model):
    return os.path.join(datafolder, dataset, "results", "RSA", glm_model or DEFAULT_GLM_MODEL)


def _raw_mean_dir(datafolder, dataset, glm_model, model):
    return os.path.join(_raw_rsa_root(datafolder, dataset, glm_model), model, "mean")


def _raw_listdir(folder):
    try:
        return os.listdir(folder)
    except OSError:
        return []


def _raw_has_result(mean_dir, specie):
    """True if a group z-map (thresholded or not) exists for this species."""
    tag = f"-{specie}-r-"
    for fn in _raw_listdir(mean_dir):
        if tag in fn and (fn.endswith("_z.nii.gz") or fn.endswith("_corrected.nii.gz")):
            return True
    return False


def raw_models_with_results(datafolder, dataset, glm_model):
    """{'D': set(models), 'H': set(models)} — model folders under the raw RSA root
    whose ``mean/`` holds a group z-map for that species."""
    root = _raw_rsa_root(datafolder, dataset, glm_model)
    out = {"D": set(), "H": set()}
    for name in _raw_listdir(root):
        mean_dir = os.path.join(root, name, "mean")
        if not os.path.isdir(mean_dir):
            continue
        for sp in ("D", "H"):
            if _raw_has_result(mean_dir, sp):
                out[sp].add(name)
    return out


def raw_roi_options(datafolder, dataset, glm_model, limit=25):
    """Mask-type prefixes present in the raw group-map filenames (best-effort;
    scans up to ``limit`` model folders). Used to populate the ROI/mask menu."""
    root = _raw_rsa_root(datafolder, dataset, glm_model)
    masks, scanned = set(), 0
    for name in sorted(_raw_listdir(root)):
        mean_dir = os.path.join(root, name, "mean")
        files = _raw_listdir(mean_dir)
        if not files:
            continue
        for fn in files:
            m = _RAW_NAME_RE.match(fn)
            if m:
                masks.add(m.group("mask") or "(none)")
        scanned += 1
        if scanned >= limit and masks:
            break
    return sorted(masks)


def _raw_map_path(datafolder, dataset, glm_model, roi, specie, model, maptype, zt):
    mean_dir = _raw_mean_dir(datafolder, dataset, glm_model, model)
    files = _raw_listdir(mean_dir)
    if not files:
        return None
    tag = f"-{specie}-r-"
    mask = None if roi in (None, "", "(none)") else roi

    def pick(pred):
        cands = [f for f in files if tag in f and pred(f)]
        if mask:  # prefer files whose mask prefix matches the chosen ROI
            masked = [f for f in cands if f.startswith(f"{mask}-")]
            if masked:
                return sorted(masked)
        return sorted(cands)

    if maptype == "mean":
        c = pick(lambda f: f.endswith("_mean.nii.gz"))
    elif maptype == "z":
        c = pick(lambda f: f.endswith("_z.nii.gz"))
    elif maptype == "corrected":
        c = pick(lambda f, z=zt: f.endswith(f"_zt{z}_corrected.nii.gz"))
        if not c:
            for z in CORRECTED_ZT_TRIES:
                c = pick(lambda f, z=z: f.endswith(f"_zt{z}_corrected.nii.gz"))
                if c:
                    break
        if not c:
            c = pick(lambda f: f.endswith("_z_corrected.nii.gz"))
    else:
        c = []
    return os.path.join(mean_dir, c[0]) if c else None


# --- Source-aware dispatch ------------------------------------------------

def _map_path_for(source, datafolder, dataset, modality, roi, glm_model,
                  specie, model, maptype, zt):
    if source == "raw":
        return _raw_map_path(datafolder, dataset, glm_model, roi, specie, model, maptype, zt)
    return _drive_map_path(datafolder, dataset, modality, roi, specie, model, maptype, zt)


def resolve_result_sets(source, datafolder, dataset, modality, roi, glm_model):
    """{'D': set, 'H': set} of models that have results, for the active source.
    Cached (cleared by 'Reload results') because the raw scan touches many dirs."""
    if source == "raw":
        key = ("raw", datafolder, dataset, glm_model)
    else:
        key = ("drive", datafolder, dataset, modality, roi)
    if key not in _RESULT_SETS_CACHE:
        if source == "raw":
            _RESULT_SETS_CACHE[key] = raw_models_with_results(datafolder, dataset, glm_model)
        else:
            _RESULT_SETS_CACHE[key] = (ht.models_with_results(datafolder, dataset, modality, roi)
                                       if roi else {"D": set(), "H": set()})
    return _RESULT_SETS_CACHE[key]


def resolve_roi_options(source, datafolder, dataset, modality, glm_model):
    if source == "raw":
        return raw_roi_options(datafolder, dataset, glm_model)
    rois = set()
    for sp in ("D", "H"):
        try:
            rois.update(datasource.scan_roi_types(datafolder, dataset, modality, sp))
        except Exception:
            pass
    return sorted(rois)


def describe_source_mode(source, datafolder, dataset, glm_model):
    if source == "raw":
        root = _raw_rsa_root(datafolder, dataset, glm_model)
        kind = "found" if os.path.isdir(root) else "not mounted"
        return f"Raw results — {root}  [{kind}]"
    return datasource.describe_source(dataset)


def _load_map(source, datafolder, dataset, modality, roi, glm_model, specie, model, maptype, zt):
    key = (source, datafolder, dataset, modality, roi, glm_model, specie, model,
           maptype, round(float(zt), 2))
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    result = None
    if model:
        path = _map_path_for(source, datafolder, dataset, modality, roi, glm_model,
                             specie, model, maptype, zt)
        if path:
            try:
                data, aff, _hdr = niftiutil.load_nifti(path)
                result = (data, aff)
            except Exception:
                result = None
    _MAP_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# The model battery — only valid "{stem}__{grouping}" models, grouped by stem
# ---------------------------------------------------------------------------
# Battery models are named "{hypothesis}__{grouping}.csv"; the trailing token
# after "__" must be one of GROUPINGS. Suffix-less models (e.g. "agent-species-id")
# are NOT valid battery models and are ignored here. Each card offers the stems as
# its "model" menu and, under a chosen stem, only the groupings that exist on disk.
# The set of models is discovered by scanning the dataset's rsa_models folder; the
# manifest CSV, when present, only enriches them with curated ordering + descriptions.

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


def grouped_valid_models(datafolder, dataset):
    """{stem: {grouping: model_name}} over every battery model that ends in a valid
    ``__{grouping}`` suffix. Suffix-less models are excluded (not valid battery
    models)."""
    idx = {}
    for name in battery_models(datafolder, dataset):
        stem, grp = split_model(name)
        if grp:
            idx.setdefault(stem, {})[grp] = name
    return idx


def ordered_valid_stems(datafolder, dataset):
    """Stems to offer as a card's model menu: manifest hypotheses first (CSV row
    order, keeping the battery's curated order), then any other valid stems found
    on disk, sorted. Only stems that have at least one valid ``__{grouping}``
    variant."""
    valid = grouped_valid_models(datafolder, dataset)
    hyps, seen = [], set()
    for meta in _manifest(datafolder, dataset).values():        # 1. curated order
        h = meta.get("hypothesis") or ""
        if h in valid and h not in seen:
            seen.add(h)
            hyps.append(h)
    for h in sorted(valid):                                     # 2. any remaining
        if h not in seen:
            seen.add(h)
            hyps.append(h)
    return hyps


FALLBACK_FOLD = "(all models)"


def build_index(datafolder, dataset):
    """The fold-aware menu backing every card::

        {'folds': [fold, ...],
         'by_fold': {fold: {'stems': [...], 'index': {stem: {grouping: model}},
                            'why': {stem: why}, 'groupings': {stem: [...]}}}}

    Driven by the dataset's central ``rsa_models/_models.csv`` (via
    ``models_manifest``). Serialised into the ``ex-grouped`` store and rebuilt
    whenever the data folder / dataset changes or results are reloaded. When no
    manifest is present it falls back to scanning the folder and offering every
    valid ``__{grouping}`` model under one synthetic ``(all models)`` fold, so the
    explorer still works before ``_models.csv`` exists."""
    idx = mm.fold_index(_model_dirs(datafolder, dataset))
    if idx["folds"]:
        return idx
    # Fallback: no _models.csv — discover models by scanning the rsa_models folder.
    grouped = grouped_valid_models(datafolder, dataset)     # {stem: {grouping: model}}
    stems = ordered_valid_stems(datafolder, dataset)
    why = {s: model_description(datafolder, dataset, next(iter(grouped[s].values()), None))
           for s in stems}
    groupings = {s: [g for g in GROUPINGS if g in grouped[s]] for s in stems}
    return {"folds": [FALLBACK_FOLD],
            "by_fold": {FALLBACK_FOLD: {"stems": stems, "index": grouped,
                                        "why": why, "groupings": groupings}}}


def _fold_data(grouped, fold):
    """The ``by_fold`` entry for one fold, or an empty skeleton."""
    by_fold = (grouped or {}).get("by_fold", {}) if isinstance(grouped, dict) else {}
    return by_fold.get(fold or "", {}) if isinstance(by_fold, dict) else {}


def _resolve_model(grouped, fold, stem, grouping):
    """Concrete ``{stem}__{grouping}`` model for a card, or None if any part is
    unset / unknown in the current fold index."""
    if not fold or not stem or not grouping:
        return None
    return _fold_data(grouped, fold).get("index", {}).get(stem, {}).get(grouping)


# --- RSA model matrix, drawn with the RSA Model Builder's renderer ---------

def _model_heatmap(datafolder, dataset, model):
    """Return a Plotly figure of the linked model's dissimilarity matrix, rendered
    with rsa_model_builder.build_cell_heatmap and the model's saved _style.json
    (falls back to a compact default so a 40x40 matrix stays readable)."""
    if not model:
        return niftiutil.empty_fig("Pick a model + grouping to load its matrix.", height=200)
    path = _find_model_csv(datafolder, dataset, model)
    if not path:
        return niftiutil.empty_fig(f"No matrix CSV for '{model}'.", height=200)
    try:
        df = pd.read_csv(path, index_col=0)
        labels = [str(x) for x in df.index]
        matrix = rmb.enforce_invariants(df.values.astype(float))
    except Exception as e:
        return niftiutil.empty_fig(f"Matrix unreadable: {e}", height=200)
    sidecar = rmb.load_style_sidecar(path)
    fig_style = sidecar.get("figure_style") if isinstance(sidecar, dict) else None
    if fig_style:
        style = {**rmb.DEFAULT_STYLE, **fig_style}
    else:  # no saved style — compact so the full matrix fits the card
        style = {**rmb.DEFAULT_STYLE, "cell_size": 20, "val_font_size": 8, "label_font_size": 9}
    return rmb.build_cell_heatmap(matrix, labels, style)


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
           title="EmoC Model Explorer")
server = app.server

# --- Edit-mode layout: CSS affordances + pointer-driven drag-to-reorder ----
# All self-contained (no extra Dash components / packages). Edit mode enables only
# ONE gesture: reordering cards by dragging their headers — cards are not
# resizable. The JS never owns the layout: during a drag it reorders things in the
# DOM for smooth feedback, and on pointer-up it hands the final order (+ gap) back
# to Dash via ``dash_clientside.set_props('ex-layout', …)``. Dash then re-renders
# the card styles from that store (single source of truth) and persists them, so
# turning Edit off keeps the arrangement and the next launch restores it.
app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      .pl-row.edit-mode .pl-drag-handle{ cursor:move; background:#eef1f6; }
      .pl-row.edit-mode .pl-panel{ touch-action:none; }
      .pl-drag-hint{ display:none; }
      .pl-row.edit-mode .pl-drag-hint{ display:inline-block; }
      .pl-panel.pl-dragging{ opacity:0.55; z-index:5; box-shadow:0 6px 18px rgba(0,0,0,0.18); }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
    <script>
    (function(){
      var drag = null;                 // active reorder gesture state
      function row(){ return document.querySelector('.pl-row'); }
      function editing(){ var r=row(); return !!(r && r.classList.contains('edit-mode')); }
      function panels(){ return Array.prototype.slice.call(document.querySelectorAll('.pl-panel')); }
      function idxOf(el){ var i=parseInt(el.getAttribute('data-index')); return isNaN(i)?0:i; }
      function orderOf(el){ var o=parseInt(el.style.order); return isNaN(o)?idxOf(el):o; }
      function isInteractive(t){ return !!(t.closest &&
        t.closest('input,label,select,button,a,svg,.Select,.dash-dropdown,.rc-slider')); }
      // Hand the current DOM order back to Dash (authoritative + persisted).
      function commit(){
        var r=row(); if(!r) return;
        var ps=panels(); if(!ps.length) return;
        var order=[];
        for(var k=0;k<ps.length;k++){ order.push(0); }
        var sorted=ps.slice().sort(function(a,b){
          return (orderOf(a)-orderOf(b)) || (ps.indexOf(a)-ps.indexOf(b)); });
        sorted.forEach(function(el,i){ order[idxOf(el)]=i; });         // normalise 0..n-1
        var gap=parseFloat(getComputedStyle(r).gap); if(isNaN(gap)) gap=10;
        if(window.dash_clientside && window.dash_clientside.set_props){
          window.dash_clientside.set_props('ex-layout',
            {data:{order:order, gap:Math.round(gap)}});
        }
      }
      document.addEventListener('pointerdown', function(e){
        if(!editing() || !e.target.closest) return;
        var dh=e.target.closest('.pl-drag-handle');
        if(dh && !isInteractive(e.target)){   // empty header area only, not its controls
          var p=dh.closest('.pl-panel'); if(!p) return;
          e.preventDefault();
          drag={panel:p};
          p.classList.add('pl-dragging');
          document.body.style.userSelect='none';
        }
      });
      document.addEventListener('pointermove', function(e){
        if(!drag) return;
        var dragged=drag.panel;
        var vis=panels().filter(function(p){ return getComputedStyle(p).display!=='none'; });
        vis.sort(function(a,b){ return orderOf(a)-orderOf(b); });
        var idx=0;
        vis.forEach(function(p){ if(p===dragged) return;
          var r=p.getBoundingClientRect(); if(e.clientX > r.left+r.width/2) idx++; });
        var others=vis.filter(function(p){ return p!==dragged; });
        var seq=others.slice(0,idx).concat([dragged]).concat(others.slice(idx));
        seq.forEach(function(p,i){ p.style.order=i; });
      });
      document.addEventListener('pointerup', function(){
        if(drag){ drag.panel.classList.remove('pl-dragging'); drag=null;
                  document.body.style.userSelect=''; commit(); }
      });
    })();
    </script>
</body>
</html>"""


def _labeled(label, comp):
    return html.Div([html.Label(label, style={"fontSize": "11px", "color": MUTED}), comp])


def _num(id_, value, width="70px"):
    return dcc.Input(id=id_, value=value, type="number", debounce=True,
                     style={**INPUT_STYLE, "width": width})


def top_bar():
    return html.Div(style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "alignItems": "flex-end",
                    "padding": "10px 12px", "backgroundColor": PANEL, "borderRadius": "8px",
                    "border": f"1px solid {LINE}", "marginBottom": "8px"}, children=[
        _labeled("Result source", dcc.Dropdown(id="ex-source-mode",
                 options=[{"label": l, "value": v} for v, l in SOURCE_MODES],
                 value=SETTINGS["source_mode"], clearable=False, style={"width": "210px"})),
        _labeled("Data folder", dcc.Input(id="ex-datafolder", value=_initial_datafolder(SETTINGS),
                 type="text", debounce=True, style={**INPUT_STYLE, "width": "230px"})),
        _labeled("Dataset", dcc.Input(id="ex-dataset", value=SETTINGS["dataset"], type="text",
                 debounce=True, style={**INPUT_STYLE, "width": "80px"})),
        _labeled("GLM model (raw)", dcc.Input(id="ex-glm-model", value=SETTINGS["glm_model"],
                 type="text", debounce=True, style={**INPUT_STYLE, "width": "120px"})),
        _labeled("Modality", dcc.Dropdown(id="ex-modality", options=[{"label": m, "value": m} for m in MODALITIES],
                 value=SETTINGS["modality"], clearable=False, style={"width": "90px"})),
        _labeled("ROI / mask", dcc.Dropdown(id="ex-roi", options=[], value=None, style={"width": "190px"})),
        html.Button("Reload results", id="ex-reload", n_clicks=0, style=BTN2),
        html.Div(style={"width": "1px", "height": "34px", "background": LINE, "margin": "0 4px"}),
        _labeled("View h", _num("ex-view-height", SETTINGS["view_height"], "70px")),
        html.Div(style={"width": "1px", "height": "34px", "background": LINE, "margin": "0 4px"}),
        _labeled("Models", html.Div(style={"display": "flex", "gap": "8px", "alignItems": "center"}, children=[
            html.Button("➕ Add model", id="ex-addpanel", n_clicks=0, style=BTN),
            dcc.Checklist(id="ex-editmode", options=[{"label": " ✏️ Edit (drag to reorder)", "value": "edit"}],
                          value=[], labelStyle={"fontSize": "12px", "fontWeight": "bold"},
                          style={"paddingTop": "2px"}),
            _num("ex-gap", SETTINGS["layout"]["gap"], "56px"),
            html.Button("Reset order", id="ex-reset-layout", n_clicks=0, style=BTN2),
        ])),
        html.Div(style={"width": "1px", "height": "34px", "background": LINE, "margin": "0 4px"}),
        _labeled("Update", html.Div(style={"display": "flex", "gap": "8px", "alignItems": "center"}, children=[
            dcc.Checklist(id="ex-autoupdate", options=[{"label": " Auto-update", "value": "auto"}],
                          value=["auto"], persistence=True,
                          labelStyle={"fontSize": "12px", "fontWeight": "bold"},
                          style={"paddingTop": "2px"}),
            html.Button("🔄 Update now", id="ex-update-now", n_clicks=0, style=BTN),
        ])),
        html.Span(id="ex-source", style={"fontSize": "11px", "color": MUTED, "marginLeft": "auto"}),
    ])


def _card_block_style(i, enabled, layout, editing):
    """Inline style for card *i*'s outer block. Only the visual *order* comes from
    the shared ``ex-layout`` store (Dash stays the single source of truth even
    though the reorder drag is captured in JS); each enabled card is its own
    **column** — they sit side by side (model 1 → column 1, model 2 → column 2, …)
    and ``flex-grow`` to share the row width evenly. ``minWidth: 0`` is essential:
    without it a flex item's min-content (the Plotly graphs) stays full-width, so
    the cards wrap and stack instead of forming columns. Edit mode adds a dashed
    outline as a drag affordance."""
    order, _gap = _layout_get(layout)
    st = {"position": "relative", "boxSizing": "border-box",
          "backgroundColor": PANEL, "borderRadius": "8px", "padding": "8px 10px",
          "border": f"1px solid {LINE}",
          "flexGrow": 1, "flexShrink": 1, "flexBasis": f"{DEFAULT_CARD_W}px",
          "minWidth": 0, "order": int(order[i])}
    if not enabled:
        st["display"] = "none"
    if editing:
        st["outline"] = f"2px dashed {ACCENT}"
        st["outlineOffset"] = "-2px"
    return st


def card(i):
    vh = SETTINGS["view_height"]
    return html.Div(id=f"pl-{i}-block", className="pl-panel",
                    style=_card_block_style(i, i == 0, SETTINGS["layout"], False),
                    **{"data-index": str(i)}, children=[
        # --- header (also the drag handle in edit mode) ---
        html.Div(id=f"pl-{i}-head", className="pl-drag-handle",
                 style={"display": "flex", "gap": "6px", "alignItems": "center",
                        "flexWrap": "wrap", "padding": "2px 4px", "borderRadius": "6px"}, children=[
            html.B(f"Model {i + 1}", style={"color": INK}),
            dcc.Checklist(id=f"pl-{i}-enable", options=[{"label": " on", "value": "on"}],
                          value=(["on"] if i == 0 else []), persistence=True, style={"fontSize": "12px"}),
            html.Span(id=f"pl-{i}-title", style={"fontSize": "12px"}),
            html.Span("⠿ drag to reorder", className="pl-drag-hint",
                      style={"fontSize": "11px", "color": MUTED}),
            html.Button("✕", id=f"pl-{i}-remove", n_clicks=0, title="remove this model",
                        style={"border": "none", "background": "transparent", "color": MUTED,
                               "cursor": "pointer", "fontSize": "13px", "padding": "0 2px",
                               "marginLeft": "auto"}),
        ]),
        # --- fold + model + grouping selection (fold drives the other two) ---
        html.Div(style={"display": "flex", "gap": "6px", "flexWrap": "wrap", "margin": "6px 0"}, children=[
            dcc.Dropdown(id=f"pl-{i}-mahfold", options=[], value=None, placeholder="fold…",
                         clearable=False, persistence=True,
                         style={"flex": "1 1 120px", "minWidth": "110px"}),
            dcc.Dropdown(id=f"pl-{i}-stem", options=[], value=None, placeholder="model…",
                         persistence=True, style={"flex": "1 1 150px", "minWidth": "140px"}),
            dcc.Dropdown(id=f"pl-{i}-grouping", options=[], value=None, placeholder="grouping…",
                         persistence=True, style={"width": "130px"}),
        ]),
        # --- the model's "why" note from _models.csv ---
        html.Div(id=f"pl-{i}-why", style={"fontSize": "11px", "color": MUTED,
                 "fontStyle": "italic", "minHeight": "14px", "margin": "0 2px 4px"}),
        # --- display toggles: species + sync + show/hide the matrix ---
        html.Div(style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "alignItems": "flex-end",
                        "margin": "2px 0 6px"}, children=[
            _labeled("Species", dcc.Dropdown(id=f"pl-{i}-maps",
                     options=[{"label": l, "value": v} for v, l in MAPS_OPTIONS],
                     value="D", clearable=False, persistence=True, style={"width": "100px"})),
            dcc.Checklist(id=f"pl-{i}-sync", options=[{"label": " 🔗 sync", "value": "sync"}],
                          value=[], persistence=True,
                          labelStyle={"fontSize": "12px"}, style={"paddingBottom": "6px"}),
            dcc.Checklist(id=f"pl-{i}-showhist", options=[{"label": " 📊 histogram", "value": "on"}],
                          value=["on"], persistence=True,
                          labelStyle={"fontSize": "12px"}, style={"paddingBottom": "6px"}),
            dcc.Checklist(id=f"pl-{i}-showmodel", options=[{"label": " show matrix", "value": "on"}],
                          value=["on"], persistence=True,
                          labelStyle={"fontSize": "12px"}, style={"paddingBottom": "6px"}),
        ]),
        # --- map controls: type + axis + colormap ---
        html.Div(style={"display": "flex", "gap": "6px", "flexWrap": "wrap", "margin": "2px 0"}, children=[
            dcc.Dropdown(id=f"pl-{i}-maptype", options=[{"label": l, "value": v} for v, l in MAPTYPES],
                         value="mean", clearable=False, persistence=True, style={"width": "150px"}),
            dcc.Dropdown(id=f"pl-{i}-axis", options=[{"label": l, "value": v} for v, l in AXES],
                         value="2", clearable=False, persistence=True, style={"width": "95px"}),
            dcc.Dropdown(id=f"pl-{i}-cmap", options=[{"label": c, "value": c} for c in COLORMAPS],
                         value=DEFAULT_CMAP, clearable=False, persistence=True, style={"width": "110px"}),
        ]),
        html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center"}, children=[
            html.Span("slice", style={"fontSize": "11px", "color": MUTED}),
            html.Div(dcc.Slider(id=f"pl-{i}-frac", min=0, max=1, step=0.02, value=0.5,
                     marks=None, tooltip={"placement": "bottom"}), style={"flex": "1"}),
        ]),
        html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center"}, children=[
            html.Span(id=f"pl-{i}-zt-label", style={"fontSize": "11px", "color": MUTED, "minWidth": "70px"}),
            html.Div(dcc.RangeSlider(id=f"pl-{i}-range", min=-1, max=1, step=0.02, value=[0, 1],
                     allowCross=False, marks={-1: "-1", 0: "0", 1: "1"},
                     tooltip={"placement": "bottom", "always_visible": False}),
                     style={"flex": "1"}),
        ]),
        html.Div(id=f"pl-{i}-note", style={"fontSize": "11px", "color": ACCENT,
                 "minHeight": "16px", "marginTop": "4px"}),
        # --- results map + in-mask histogram, side by side ---
        # The histogram sits *next to* the brain and shares its height, so the
        # slice and the distribution behind it are read together. ``minWidth: 0``
        # on both halves keeps the Plotly graphs from forcing the row to wrap.
        html.Div(style={"display": "flex", "gap": "6px", "alignItems": "stretch"}, children=[
            html.Div(dcc.Graph(id=f"pl-{i}-map", style={"height": f"{vh}px"}),
                     style={"flex": "3 1 0", "minWidth": 0}),
            html.Div(id=f"pl-{i}-histwrap",
                     children=dcc.Graph(id=f"pl-{i}-hist", figure=niftiutil.empty_fig(height=vh),
                                        style={"height": f"{vh}px"},
                                        config={"displayModeBar": False}),
                     style={"flex": "2 1 0", "minWidth": 0}),
        ]),
        # --- model dissimilarity matrix (toggled by "show matrix") ---
        html.Div(id=f"pl-{i}-matrixwrap", children=[
            html.Div("Model matrix (builder view)", style={"fontSize": "11px", "color": MUTED,
                     "margin": "8px 0 2px"}),
            html.Div(dcc.Graph(id=f"pl-{i}-matrix", figure=niftiutil.empty_fig(height=200),
                     config={"displayModeBar": False}),
                     style={"maxHeight": "520px", "overflowY": "auto"}),
        ]),
    ])


app.layout = html.Div(style={"backgroundColor": BG, "color": INK, "minHeight": "100vh",
                      "padding": "10px 14px", "fontFamily": "'Segoe UI', Arial, sans-serif"}, children=[
    html.H2("EmoC Model Explorer", style={"textAlign": "center", "margin": "4px 0 8px"}),
    top_bar(),
    html.Div(style={"display": "flex", "alignItems": "baseline", "gap": "10px", "margin": "4px 2px 6px"},
             children=[
        html.H4("Models", style={"margin": "0", "color": INK}),
        html.Span(id="ex-models-title", style={"fontSize": "12px", "color": MUTED}),
    ]),
    html.Div(id="pl-row", className="pl-row",
             style={"display": "flex", "flexWrap": "wrap", "alignItems": "stretch",
                    "gap": f'{SETTINGS["layout"]["gap"]}px'},
             children=[card(i) for i in range(MAX_MODELS)]),

    dcc.Store(id="ex-dataver", data=0),
    dcc.Store(id="ex-update-trigger", data=0),
    dcc.Store(id="ex-grouped", data={"folds": [], "by_fold": {}}),
    dcc.Store(id="ex-layout", data=SETTINGS["layout"]),
    dcc.Store(id="ex-settings-status"),
    # Shared view state broadcast to synced same-species cards (see sync callbacks).
    dcc.Store(id="ex-sync-D", data={}),
    dcc.Store(id="ex-sync-H", data={}),
])


# ---------------------------------------------------------------------------
# Callbacks — data source
# ---------------------------------------------------------------------------

@app.callback(Output("ex-datafolder", "value"),
              Input("ex-source-mode", "value"), prevent_initial_call=True)
def cb_mode_datafolder(source):
    """When the user switches source mode, seed the data folder with that mode's
    default root (raw -> pipeline disk; drive -> best current-results root). Fires
    only on user changes (prevent_initial_call), so a saved folder is respected on
    load."""
    if source == "raw":
        try:
            return get_paths()[0]
        except Exception:
            return no_update
    return datasource.resolve_datafolder(DEFAULT_DATASET)


@app.callback(Output("ex-roi", "options"), Output("ex-roi", "value"), Output("ex-source", "children"),
              Input("ex-modality", "value"), Input("ex-datafolder", "value"), Input("ex-dataset", "value"),
              Input("ex-source-mode", "value"), Input("ex-glm-model", "value"),
              Input("ex-dataver", "data"))
def cb_rois(modality, datafolder, dataset, source, glm_model, _ver):
    rois = resolve_roi_options(source, datafolder, dataset, modality, glm_model)
    return ([{"label": r, "value": r} for r in rois], (rois[0] if rois else None),
            describe_source_mode(source, datafolder, dataset, glm_model))


@app.callback(Output("ex-grouped", "data"), Output("ex-models-title", "children"),
              Input("ex-datafolder", "value"), Input("ex-dataset", "value"), Input("ex-dataver", "data"))
def cb_build_index(datafolder, dataset, _ver):
    """Rebuild the fold → model → grouping menu whenever the data folder / dataset
    changes or results are reloaded, re-reading ``_models.csv`` fresh each time."""
    _MANIFEST_CACHE.pop((datafolder, dataset), None)   # legacy battery-manifest cache
    mm.clear_cache()                                   # re-read _models.csv from disk
    grouped = build_index(datafolder, dataset)
    folds = grouped.get("folds", [])
    nstems = sum(len(grouped["by_fold"][f]["stems"]) for f in folds)
    if folds == [FALLBACK_FOLD]:
        title = (f"no _models.csv — scanned {nstems} model families from disk; "
                 "add a card, pick a model + grouping")
    else:
        title = (f"{len(folds)} fold(s), {nstems} model families from _models.csv — "
                 "pick a fold, then a model + grouping per card")
    return grouped, title


@app.callback(Output("ex-dataver", "data"), Input("ex-reload", "n_clicks"),
              State("ex-dataver", "data"), prevent_initial_call=True)
def cb_reload(_n, ver):
    """Drop cached maps/masks/result-sets so freshly-synced results are re-read."""
    _MAP_CACHE.clear()
    _RESULT_SETS_CACHE.clear()
    _MASK_CACHE.clear()
    _MASK_ON_GRID.clear()
    return (ver or 0) + 1


# Per-card cascade: fold → model (stem) → grouping. Each level lists only what the
# level above allows, and each keeps a still-valid persisted value (so a card comes
# back exactly as left) while defaulting sensibly when the old value no longer fits.

def _register_panel_mahfold(i):
    @app.callback(Output(f"pl-{i}-mahfold", "options"), Output(f"pl-{i}-mahfold", "value"),
                  Input("ex-grouped", "data"), State(f"pl-{i}-mahfold", "value"))
    def _cb(grouped, cur):
        folds = (grouped or {}).get("folds", []) if isinstance(grouped, dict) else []
        opts = [{"label": f, "value": f} for f in folds]
        if cur in folds:
            return opts, cur
        return opts, (folds[0] if folds else None)
    return _cb


def _register_panel_stem(i):
    @app.callback(Output(f"pl-{i}-stem", "options"), Output(f"pl-{i}-stem", "value"),
                  Input(f"pl-{i}-mahfold", "value"), Input("ex-grouped", "data"),
                  State(f"pl-{i}-stem", "value"))
    def _cb(fold, grouped, cur):
        stems = _fold_data(grouped, fold).get("stems", [])
        opts = [{"label": s, "value": s} for s in stems]
        if cur in stems:                                     # keep a still-valid choice
            return opts, cur
        return opts, None                                    # else clear -> placeholder
    return _cb


def _register_panel_why(i):
    @app.callback(Output(f"pl-{i}-why", "children"),
                  Input(f"pl-{i}-mahfold", "value"), Input(f"pl-{i}-stem", "value"),
                  Input("ex-grouped", "data"))
    def _cb(fold, stem, grouped):
        return _fold_data(grouped, fold).get("why", {}).get(stem or "", "")
    return _cb


for _i in range(MAX_MODELS):
    _register_panel_mahfold(_i)
    _register_panel_stem(_i)
    _register_panel_why(_i)


# ---------------------------------------------------------------------------
# Callbacks — add / remove model cards
# ---------------------------------------------------------------------------
# Cards are pre-registered (``MAX_MODELS`` slots) and shown/hidden via their ``on``
# switch. "➕ Add model" turns on the next off slot; each card's ✕ button (and its
# ``on`` checkbox) turns it off. Grouped Output/State lists let one callback fan
# out over every slot.

@app.callback([Output(f"pl-{i}-enable", "value", allow_duplicate=True) for i in range(MAX_MODELS)],
              Output("ex-models-title", "children", allow_duplicate=True),
              Input("ex-addpanel", "n_clicks"),
              [State(f"pl-{i}-enable", "value") for i in range(MAX_MODELS)],
              prevent_initial_call=True)
def cb_add_panel(_n, *enables):
    """Turn on the first currently-off card; no-op when all are already on. (Dash
    passes the State list as separate positional args, hence ``*enables``.)"""
    out = [no_update] * MAX_MODELS
    for i, e in enumerate(enables):
        if "on" not in (e or []):
            out[i] = ["on"]
            return out, f"Added model {i + 1}."
    return out, f"All {MAX_MODELS} model slots are already in use."


def _register_panel_remove(i):
    @app.callback(Output(f"pl-{i}-enable", "value", allow_duplicate=True),
                  Output("ex-models-title", "children", allow_duplicate=True),
                  Input(f"pl-{i}-remove", "n_clicks"), prevent_initial_call=True)
    def _cb_remove(_n):
        return [], f"Removed model {i + 1}."       # clear the "on" value -> card hides
    return _cb_remove


for _i in range(MAX_MODELS):
    _register_panel_remove(_i)


# ---------------------------------------------------------------------------
# Callbacks — persist settings so the next session loads like this one
# ---------------------------------------------------------------------------

@app.callback(Output("ex-settings-status", "data"),
              Input("ex-source-mode", "value"), Input("ex-datafolder", "value"),
              Input("ex-glm-model", "value"), Input("ex-dataset", "value"),
              Input("ex-modality", "value"), Input("ex-view-height", "value"),
              Input("ex-layout", "data"), prevent_initial_call=True)
def cb_save_settings(source, datafolder, glm_model, dataset, modality, view_h, layout):
    s = {
        "source_mode": source or "drive",
        "datafolder": datafolder or None,
        "glm_model": (glm_model or DEFAULT_GLM_MODEL).strip(),
        "dataset": (dataset or DEFAULT_DATASET).strip(),
        "modality": modality or "RSA",
        "view_height": _int(view_h, DEFAULT_SETTINGS["view_height"]),
        "layout": _clean_layout(layout),   # model-card order + gap
    }
    save_settings(s)
    return s


# ---------------------------------------------------------------------------
# Callbacks — per-card grouping menu (depends on the chosen model stem)
# ---------------------------------------------------------------------------

def _register_panel_grouping(i):
    @app.callback(Output(f"pl-{i}-grouping", "options"), Output(f"pl-{i}-grouping", "value"),
                  Input(f"pl-{i}-mahfold", "value"), Input(f"pl-{i}-stem", "value"),
                  Input("ex-grouped", "data"), State(f"pl-{i}-grouping", "value"))
    def _cb(fold, stem, grouped, cur):
        variants = _fold_data(grouped, fold).get("index", {}).get(stem or "", {})
        groups = list(variants.keys())                       # already canonically ordered
        opts = [{"label": g, "value": g} for g in groups]
        if cur in groups:                                    # keep a still-valid choice
            return opts, cur
        return opts, (groups[0] if groups else None)
    return _cb


for _i in range(MAX_MODELS):
    _register_panel_grouping(_i)


# ---------------------------------------------------------------------------
# Callbacks — per-card threshold range (depends on the map type / "measure")
# ---------------------------------------------------------------------------
# The range slider's two handles mean different things for different map types:
# a z-threshold (0-8) for the z-map / cluster-corrected map, vs. an
# average-similarity threshold — Kendall's tau, this pipeline's default RSA
# method, ranging -1..1 — for the group-average map. Switching map type swaps
# the slider's limits/marks/label; a still-in-range current [lo, hi] survives
# the switch, otherwise it resets to that mode's default. The **low** handle
# filters voxels below it out (rendered transparent); the **high** handle caps
# the color scale — voxels at/above it are painted the palette's top color.
ZT_RANGE_Z = {"min": 0, "max": 8, "step": 0.1, "marks": {0: "0", 3.1: "3.1", 8: "8"},
              "default": [3.1, 8], "label": "z"}
ZT_RANGE_MEAN = {"min": -1, "max": 1, "step": 0.02, "marks": {-1: "-1", 0: "0", 1: "1"},
                 "default": [0.0, 1.0], "label": "Kendall τ"}


def _register_panel_zt_range(i):
    @app.callback(
        Output(f"pl-{i}-range", "min"), Output(f"pl-{i}-range", "max"),
        Output(f"pl-{i}-range", "step"), Output(f"pl-{i}-range", "marks"),
        Output(f"pl-{i}-range", "value", allow_duplicate=True),
        Output(f"pl-{i}-zt-label", "children"),
        Input(f"pl-{i}-maptype", "value"), State(f"pl-{i}-range", "value"),
        prevent_initial_call="initial_duplicate")
    def _cb(maptype, cur):
        r = ZT_RANGE_MEAN if maptype == "mean" else ZT_RANGE_Z
        try:
            lo, hi = float(cur[0]), float(cur[1])
            in_range = r["min"] <= lo <= hi <= r["max"]
        except (TypeError, ValueError, IndexError):
            in_range = False
        val = [lo, hi] if in_range else list(r["default"])
        return r["min"], r["max"], r["step"], r["marks"], val, r["label"]
    return _cb


for _i in range(MAX_MODELS):
    _register_panel_zt_range(_i)


# ---------------------------------------------------------------------------
# Callbacks — card rendering (one per card)
# ---------------------------------------------------------------------------

def _hist_fig(values, lo, hi, colorscale, height, xtitle):
    """Distribution of the map's values **inside the search mask**, drawn to sit
    beside the brain slice and read as its legend: bars below the range slider's
    low handle are grey (exactly the voxels the overlay renders transparent), bars
    at/above it carry the card's colormap over the same [low, high] scale.

    Counts are on a log axis on purpose — a statistical map is overwhelmingly
    near-zero voxels, and on a linear axis the supra-threshold tail you actually
    came to look at is a flat line on the baseline."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return niftiutil.empty_fig("no voxels in mask", height=height)
    lo, hi = float(lo), float(hi)
    # Always keep the low handle inside the axis so its cut line is visible, even
    # when it sits past the data (e.g. z-threshold 3.1 on an all-sub-threshold map).
    left, right = min(float(np.min(v)), lo), max(float(np.max(v)), lo)
    if right <= left:
        right = left + 1e-6
    # Put a bin boundary exactly on the low handle (splitting the bin budget in
    # proportion) so no single bar straddles the cut: with ``np.histogram``'s
    # half-open bins, "left edge >= lo" then selects precisely the voxels the
    # slice paints, and the coloured bars sum to the count in the title.
    if left < lo < right:
        n_below = int(round(HIST_BINS * (lo - left) / (right - left)))
        n_below = min(max(n_below, 1), HIST_BINS - 1)
        edges = np.concatenate([np.linspace(left, lo, n_below + 1),
                                np.linspace(lo, right, HIST_BINS - n_below + 1)[1:]])
    else:
        edges = np.linspace(left, right, HIST_BINS + 1)
    counts, edges = np.histogram(v, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    supra = edges[:-1] >= lo
    n_supra = int(np.sum(v >= lo))

    fig = go.Figure()
    if np.any(~supra):
        fig.add_trace(go.Bar(x=centers[~supra], y=counts[~supra], width=widths[~supra],
                             marker=dict(color=HIST_SUB_COLOR), showlegend=False,
                             hovertemplate="%{x:.3g} → %{y} vx<extra>below threshold</extra>"))
    if np.any(supra):
        fig.add_trace(go.Bar(
            x=centers[supra], y=counts[supra], width=widths[supra], showlegend=False,
            marker=dict(color=centers[supra], colorscale=(colorscale or DEFAULT_CMAP),
                        cmin=lo, cmax=max(hi, lo + 1e-6)),
            hovertemplate="%{x:.3g} → %{y} vx<extra>shown on the slice</extra>"))
    for x, dash in ((lo, "solid"), (hi, "dot")):
        if left <= x <= right:
            fig.add_vline(x=x, line=dict(color=ACCENT, width=1, dash=dash))
    fig.update_layout(
        title=dict(text=f"{v.size} vx in mask · {n_supra} ≥ {lo:g}",
                   font=dict(size=11, color=INK)),
        margin=dict(l=38, r=6, t=26, b=30), height=height, bargap=0,
        paper_bgcolor=PANEL, plot_bgcolor="#ffffff", font_color=INK,
        xaxis=dict(title=dict(text=xtitle, font=dict(size=9, color=MUTED)),
                   tickfont=dict(size=9), gridcolor=LINE,
                   zeroline=True, zerolinecolor=LINE, range=[left, right]),
        yaxis=dict(type="log", title=dict(text="voxels (log)", font=dict(size=9, color=MUTED)),
                   tickfont=dict(size=9), gridcolor=LINE))
    return fig


def _card_species_fig(source, datafolder, dataset, modality, roi, glm_model,
                      specie, model, maptype, axis, frac, zt, view_height,
                      colorscale=None, vmax_override=None, want_hist=True):
    """(slice figure, histogram figure | None, n supra-threshold in mask, n in mask).

    The histogram is computed from the *same* loaded volume as the slice, so the
    two always describe one map; it is skipped entirely (None) when the card has
    its histogram hidden."""
    loaded = _load_map(source, datafolder, dataset, modality, roi, glm_model,
                       specie, model, maptype, zt)
    label = {"D": "Dog", "H": "Human"}[specie]
    if loaded is None:
        empty = niftiutil.empty_fig(f"{label}: no {maptype} map", height=view_height)
        return empty, (niftiutil.empty_fig("no map", height=view_height) if want_hist else None), 0, 0
    data, aff = loaded
    atlas = _atlas_on_grid(specie, data.shape, aff)
    ax = int(axis)
    idx = int(round(float(frac) * (data.shape[ax] - 1)))
    nz = data[np.abs(data) > 1e-6]
    # ``zt`` / ``vmax_override`` are the card's range-slider low/high handles,
    # whose meaning depends on maptype (z-score for z/corrected, Kendall's-tau-
    # like average similarity for mean — see ZT_RANGE_Z / ZT_RANGE_MEAN). The
    # low handle filters voxels below it out (transparent); the high handle caps
    # the color scale (falls back to the data-driven max if unset).
    thr = float(zt)
    vmin = thr
    auto_max = float(np.max(np.abs(nz))) if nz.size else thr + 1
    try:
        vmax = float(vmax_override) if vmax_override not in (None, "") else auto_max
    except (TypeError, ValueError):
        vmax = auto_max
    if vmax <= vmin:
        vmax = vmin + 1e-6
    fig = niftiutil.make_slice_fig(atlas, data, ax, idx, opacity=0.8, z_threshold=thr,
                                   vmin=vmin, vmax=vmax, title=f"{label} · {model}",
                                   height=view_height, colorscale=colorscale)
    # Counts (and the histogram) are restricted to the search mask — the voxels the
    # searchlight actually visited — so "how many survive this threshold" is out of
    # a meaningful denominator instead of the whole bounding box.
    vals, xtitle = _mask_values(data, datafolder, dataset, specie, roi, aff)
    vals = vals[np.isfinite(vals)]
    supra = int(np.sum(vals >= thr))
    hist = _hist_fig(vals, thr, vmax, colorscale, view_height, xtitle) if want_hist else None
    return fig, hist, supra, int(vals.size)


def _register_panel(i):
    # Card *content* — title/status, single-species results map + note, model
    # matrix. The card block's own style (order, width, show/hide, edit outline) is
    # owned by the separate style callback below so the edit-mode arrangement is
    # never overwritten here.
    @app.callback(
        Output(f"pl-{i}-title", "children"),
        Output(f"pl-{i}-map", "figure"), Output(f"pl-{i}-map", "style"),
        Output(f"pl-{i}-hist", "figure"), Output(f"pl-{i}-hist", "style"),
        Output(f"pl-{i}-histwrap", "style"),
        Output(f"pl-{i}-matrix", "figure"), Output(f"pl-{i}-matrixwrap", "style"),
        Output(f"pl-{i}-note", "children"),
        Input(f"pl-{i}-enable", "value"), Input(f"pl-{i}-mahfold", "value"),
        Input(f"pl-{i}-stem", "value"),
        Input(f"pl-{i}-grouping", "value"), Input(f"pl-{i}-maps", "value"),
        Input(f"pl-{i}-showhist", "value"),
        Input(f"pl-{i}-showmodel", "value"), Input(f"pl-{i}-maptype", "value"),
        Input(f"pl-{i}-axis", "value"), Input(f"pl-{i}-frac", "value"), Input(f"pl-{i}-range", "value"),
        Input(f"pl-{i}-cmap", "value"),
        Input("ex-roi", "value"), Input("ex-dataver", "data"),
        Input("ex-source-mode", "value"), Input("ex-glm-model", "value"),
        Input("ex-view-height", "value"), Input("ex-grouped", "data"),
        Input("ex-update-trigger", "data"), State("ex-autoupdate", "value"),
        State("ex-datafolder", "value"), State("ex-dataset", "value"), State("ex-modality", "value"))
    def _cb(enable, mahfold, stem, grouping, maps, showhist, showmodel, maptype, axis, frac,
            rng, cmap, roi, _ver, source, glm_model, view_h, grouped, _update_trig, autoupdate,
            datafolder, dataset, modality):
        vh = _int(view_h, DEFAULT_SETTINGS["view_height"])
        gshow = {"height": f"{vh}px"}
        wrap_show = {}
        wrap_hide = {"display": "none"}
        show_matrix = "on" in (showmodel or [])
        show_hist = "on" in (showhist or [])
        hist_wrap = {"flex": "2 1 0", "minWidth": 0} if show_hist else wrap_hide
        if "on" not in (enable or []):        # card off — block hidden anyway
            return (no_update, no_update, no_update, no_update, no_update, hist_wrap,
                    no_update, wrap_hide, "")

        # Auto-update off: only the slice slider, card on/off, matrix show/hide and
        # the top-bar source/ROI/reload/view-height controls (plus the Update button
        # itself) re-render live; everything else just flags a pending change and
        # leaves the current map/matrix in place until Update is clicked.
        trig = ctx.triggered_id
        live_triggers = {f"pl-{i}-frac", f"pl-{i}-enable", f"pl-{i}-showmodel",
                         f"pl-{i}-showhist",
                         "ex-update-trigger", "ex-roi", "ex-dataver", "ex-source-mode",
                         "ex-glm-model", "ex-view-height", "ex-grouped"}
        if trig is not None and trig not in live_triggers and "auto" not in (autoupdate or []):
            return (no_update, no_update, no_update, no_update, no_update, hist_wrap,
                    no_update, no_update, "⏸ change pending — click 🔄 Update")

        model = _resolve_model(grouped, mahfold, stem, grouping)
        if not model:
            title = html.Span("— pick a fold, model + grouping —", style={"color": MUTED})
            empty = niftiutil.empty_fig("select a model + grouping", height=vh)
            mat = _model_heatmap(datafolder, dataset, None) if show_matrix else no_update
            hist = niftiutil.empty_fig("no model", height=vh) if show_hist else no_update
            return (title, empty, gshow, hist, gshow, hist_wrap, mat,
                    wrap_show if show_matrix else wrap_hide, "no model")

        # header: results-availability dot + resolved model name
        result_sets = resolve_result_sets(source, datafolder, dataset, modality, roi, glm_model)
        st = ht.node_status(model, result_sets)
        color, st_label = STATUS_STYLE.get(st, STATUS_STYLE["unlinked"])
        title = html.Span([status_dot(color), html.Span(model, style={"fontWeight": "bold", "color": INK}),
                           html.Span(f"  · {st_label}", style={"color": MUTED})])

        # single-species results map (this card's column). ``rng`` is the range
        # slider's [low, high]: low filters voxels out, high caps the color scale.
        try:
            zt, vmax = float(rng[0]), float(rng[1])
        except (TypeError, ValueError, IndexError):
            zt, vmax = 0.0, 1.0
        specie = maps if maps in ("D", "H") else "D"
        label = {"D": "Dog", "H": "Human"}[specie]
        fig, hist, n, n_mask = _card_species_fig(
            source, datafolder, dataset, modality, roi, glm_model,
            specie, model, maptype, axis, frac, zt, vh,
            colorscale=cmap, vmax_override=vmax, want_hist=show_hist)
        note = f"{label}: {n} / {n_mask} vx ≥ {zt:g} in mask"

        # model matrix (only re-rendered / shown when the toggle is on)
        mat = _model_heatmap(datafolder, dataset, model) if show_matrix else no_update
        return (title, fig, gshow, (hist if show_hist else no_update), gshow, hist_wrap,
                mat, wrap_show if show_matrix else wrap_hide, note)
    return _cb


def _register_panel_style(i):
    # Owns pl-{i}-block.style: applies the shared layout order, hides the block
    # when the card is switched off, and adds the edit-mode outline. Fires on
    # enable / layout / edit-mode changes.
    @app.callback(Output(f"pl-{i}-block", "style"),
                  Input(f"pl-{i}-enable", "value"), Input("ex-layout", "data"),
                  Input("ex-editmode", "value"))
    def _cb_style(enable, layout, edit):
        return _card_block_style(i, "on" in (enable or []), layout, "edit" in (edit or []))
    return _cb_style


for _i in range(MAX_MODELS):
    _register_panel(_i)
    _register_panel_style(_i)


# ---------------------------------------------------------------------------
# Callbacks — cross-card view sync (slice / axis / threshold / scale / colormap)
# ---------------------------------------------------------------------------
# Each card carries a "🔗 sync" toggle. Synced cards of the *same species* share one
# view: moving the slice/axis, or changing threshold / max / colormap on any of
# them mirrors to all the others. This is done with two tiny broadcast stores
# (``ex-sync-D`` / ``ex-sync-H``):
#   * a single **writer** callback watches every card's SYNC_CONTROLS + sync toggle;
#     when a synced card's control changes (or its sync turns on) it publishes that
#     card's control values into the store for the card's species.
#   * a per-card **reader** adopts its species store whenever the store (or the
#     card's own sync / species) changes, writing the shared values back onto its
#     controls. The loop is self-limiting: a reader only writes values equal to the
#     store, so the writer it re-triggers republishes the same data and Dash stops.

def _sync_params_from(vals, i):
    """The SYNC_CONTROLS snapshot for card *i* as a plain dict, ready to store."""
    return {p: vals[(i, p)] for p in SYNC_CONTROLS}


@app.callback(
    Output("ex-sync-D", "data", allow_duplicate=True),
    Output("ex-sync-H", "data", allow_duplicate=True),
    *[Input(f"pl-{i}-{p}", "value") for p in SYNC_CONTROLS for i in range(MAX_MODELS)],
    *[Input(f"pl-{i}-sync", "value") for i in range(MAX_MODELS)],
    *[State(f"pl-{i}-maps", "value") for i in range(MAX_MODELS)],
    prevent_initial_call=True)
def cb_sync_write(*args):
    n = MAX_MODELS
    nctl = len(SYNC_CONTROLS)
    # Reshape the flat Dash arg list back into addressable groups.
    vals = {}
    for pi, p in enumerate(SYNC_CONTROLS):
        for i in range(n):
            vals[(i, p)] = args[pi * n + i]
    syncs = args[nctl * n: (nctl + 1) * n]
    mapss = args[(nctl + 1) * n: (nctl + 2) * n]

    trig = ctx.triggered_id                      # e.g. "pl-2-frac"
    m = re.match(r"pl-(\d+)-(\w+)$", trig or "")
    if not m:
        return no_update, no_update
    i, prop = int(m.group(1)), m.group(2)
    if "sync" not in (syncs[i] or []):           # only synced cards publish
        return no_update, no_update
    if prop not in SYNC_CONTROLS and prop != "sync":
        return no_update, no_update

    params = _sync_params_from(vals, i)
    if mapss[i] == "H":
        return no_update, params
    return params, no_update


def _register_panel_sync_read(i):
    @app.callback(
        [Output(f"pl-{i}-{p}", "value", allow_duplicate=True) for p in SYNC_CONTROLS],
        Input("ex-sync-D", "data"), Input("ex-sync-H", "data"),
        Input(f"pl-{i}-sync", "value"), Input(f"pl-{i}-maps", "value"),
        prevent_initial_call=True)
    def _cb(store_d, store_h, sync, maps):
        if "sync" not in (sync or []):
            return [no_update] * len(SYNC_CONTROLS)
        store = store_h if maps == "H" else store_d
        if not isinstance(store, dict) or not store:
            return [no_update] * len(SYNC_CONTROLS)
        return [store.get(p, no_update) for p in SYNC_CONTROLS]
    return _cb


for _i in range(MAX_MODELS):
    _register_panel_sync_read(_i)


# ---------------------------------------------------------------------------
# Callbacks — model-card layout (edit mode: reorder / gap / reset)
# ---------------------------------------------------------------------------

@app.callback(Output("pl-row", "style"), Output("pl-row", "className"),
              Input("ex-layout", "data"), Input("ex-editmode", "value"))
def cb_row_layout(layout, edit):
    """Container style + class for the model-card row: the inter-card gap comes
    from the layout store, and the ``edit-mode`` class (added when the Edit toggle
    is on) is what the CSS/JS use to enable header dragging."""
    _order, gap = _layout_get(layout)
    style = {"display": "flex", "flexWrap": "wrap", "alignItems": "stretch", "gap": f"{gap}px"}
    cls = "pl-row edit-mode" if "edit" in (edit or []) else "pl-row"
    return style, cls


@app.callback(Output("ex-layout", "data", allow_duplicate=True),
              Input("ex-gap", "value"), State("ex-layout", "data"), prevent_initial_call=True)
def cb_gap(gap, layout):
    """Gap number box -> layout store (merged so it doesn't clobber the order)."""
    order, cur = _layout_get(layout)
    try:
        cur = int(gap)
    except (TypeError, ValueError):
        pass
    return _clean_layout({"order": order, "gap": cur})


@app.callback(Output("ex-layout", "data", allow_duplicate=True), Output("ex-gap", "value"),
              Output("ex-models-title", "children", allow_duplicate=True),
              Input("ex-reset-layout", "n_clicks"), prevent_initial_call=True)
def cb_reset_layout(_n):
    """Restore the default card order + gap and sync the gap box."""
    d = _default_layout()
    return d, d["gap"], "Layout reset to default order."


# ---------------------------------------------------------------------------
# Callbacks — manual "Update now" + auto-update status feedback
# ---------------------------------------------------------------------------

@app.callback(Output("ex-update-trigger", "data"), Output("ex-models-title", "children", allow_duplicate=True),
              Input("ex-update-now", "n_clicks"), State("ex-update-trigger", "data"),
              [State(f"pl-{i}-enable", "value") for i in range(MAX_MODELS)],
              prevent_initial_call=True)
def cb_update_now(_n, ver, *enables):
    """Bumps ex-update-trigger, which every card's render callback treats as a
    live trigger regardless of the auto-update toggle — applies any pending
    changes on enabled cards."""
    n_on = sum(1 for e in enables if "on" in (e or []))
    return (ver or 0) + 1, f"Updated {n_on} enabled card(s)."


@app.callback(Output("ex-models-title", "children", allow_duplicate=True),
              Input("ex-autoupdate", "value"), prevent_initial_call=True)
def cb_autoupdate_status(val):
    if "auto" in (val or []):
        return "Auto-update ON — cards refresh as you change settings."
    return "Auto-update OFF — change settings, then click 🔄 Update now to apply."


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="EmoC RSA model explorer")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("EXPLORER_PORT", os.environ.get("PORT", "8055"))))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(f"[hypothesis_explorer] settings : {SETTINGS_PATH}")
    print(f"[hypothesis_explorer] source   : {SETTINGS['source_mode']}  "
          f"datafolder={_initial_datafolder(SETTINGS)}")
    print(f"[hypothesis_explorer] open http://{args.host}:{args.port}")
    app.run(debug=args.debug, use_reloader=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

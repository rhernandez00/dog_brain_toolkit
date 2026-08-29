#!/usr/bin/env python
"""
hypothesis_explorer.py — RSA model explorer (standalone Dash app, any dataset).

This app is a **row of self-contained model cards** — no hypothesis tree. You
**add** and **remove** cards, and each card is one RSA model you want to look at:

  * The card's model is picked in **three visual steps**, and none of them is a
    list of names. The analysis is organised as ``distance method → (Mahalanobis
    fold) → model → grouping``, which is four dropdowns of jargon; the card asks
    the same four values as three pictures, in the same order.

      ① **Stimulus set.** One tile per (``dis_method``, ``mah_fold``) pair the
        dataset offers. That pair is not two settings — it is **which stimulus set
        the RDM is built over**, and therefore which models exist at all, since a
        model built for one pairwise-similarity method is not comparable with
        another's. So the tile *is* that set, drawn as its grid: one row of icons
        per **agent block** (🐕 Dog-shown / 🧑 Human-shown), one cell per
        **category** (🤩 positive anticipation, 😀 happiness, 😠 anger,
        😨 fear, 😐 neutral), and each cell carrying its **exemplar count**
        when it holds more than one. On EmoC that draws the difference exactly:
        ``mahalanobis · stim-wise`` is a 10x10 RDM over conditions (``DogP``),
        ``correlation · run-wise`` a 40x40 over four exemplars each
        (``DogP1``..``DogP4``, shown as ×4 on every icon). The method and fold name
        the tile underneath, with the number of models in the set. **Picking a set
        re-fills step ②.**
      ② **Which hypothesis.** A gallery of that set's models — each tile is the
        model's **own dissimilarity matrix**, drawn as a thumbnail, with its name,
        an ⓘ carrying the ``why`` column of ``_models.csv`` as its tooltip, and the
        groupings that hypothesis actually offers.
      ③ **Which pairs enter.** The same stimulus grid again, now *boxed*: one box
        around everything is **pooled** (every pair enters), a box per row is
        **within** (only pairs inside a row), two boxes joined by ⇅ is **cross**
        (only the pairs that bridge them), and a single lit row over a dimmed one is
        **Dog only** / **Human only**. Only the groupings the chosen hypothesis has
        a model file for are offered.

    Rows, columns and exemplar count are read off a model's own matrix, so another
    dataset draws *its* grid and an unknown label falls back to its initials.

    Steps are shown **one at a time**, in order: a fresh card opens ① alone,
    answering it moves to ②, then ③, and answering that collapses all three into a
    breadcrumb (``◈ set`` · ``▦ model`` · ``◧ grouping``) so the card only spends
    screen space on choices that are still open. Click a chip to reopen its step; a
    change that invalidates a level below clears it and re-opens that step by
    itself.

    Stem + grouping resolve to the concrete ``{stem}__{grouping}`` model, and the
    cascade's own values are mirrored into hidden inputs, so everything downstream
    still reads them as before. The whole picker is driven by the dataset's central
    ``rsa_models/_models.csv`` manifest (built by ``tools/build_models_manifest.py``),
    read from ``{data folder}/{dataset}/rsa_models/`` — both taken from the top bar,
    so pointing the app at another project reads that project's manifest and models;
    the resolved path is printed in the status line beside **Models**. Edit that one
    file to add, retire, or re-group models. When the manifest is absent the card
    falls back to scanning the folder and offering every valid ``__{grouping}``
    model under one synthetic set.
  * Each card is **one species** — its own column: the **Species** control picks
    **Dog** or **Human** and the card draws that species' results map as a 2D atlas
    slice (put Dog and Human side by side in two cards). **Which** map is drawn and
    **in which palette** are asked *on the picture*, by two floating switches in
    the slice's bottom corners — the way a map app puts its layer switcher in a
    corner rather than in a settings panel. Bottom-left is a three-segment
    **map-type** switch (``x̄`` group average · ``Z`` z-map · ``◉`` cluster-corrected,
    defaulting to the group mean); bottom-right is the **colour palette**,
    collapsed to a swatch of the palette in use and opening on hover into the full
    list, each entry drawn as itself rather than named. Axis, slice position and a
    two-handle **range slider** (low/high threshold) are per-card as well. The colormap defaults to **Hot**: voxels below
    the range's low handle render transparent (alpha=0), everything at/above the
    high handle is painted the top color of the scale. Slices are drawn
    radiology-style — **bright anatomy on a black canvas** — and are laid out from
    the map's *affine* rather than its array order, so **anterior is up** on an
    axial slice (Slice Z) and superior is up on a coronal/sagittal one, with the
    four edges marked **L/R · A/P · S/I**.
  * **Scroll the mouse wheel over a slice** to step through it one slice per
    notch (wheel up = higher slice index); the slice slider follows and the page
    itself does not scroll while the pointer is over the brain.
  * **Click a slice to plant a crosshair**, and the strip under the brain reads
    out that voxel the way any MR viewer does: **voxel index**, **world (mm)
    coordinate** and the map's **intensity** there — so clicking inside a cluster
    tells you exactly where you clicked and how strong it is. The crosshair is
    kept as a *voxel*, so it survives an axis switch, and its out-of-plane
    coordinate always follows the slice on screen: scroll through the slices and
    the read-out sweeps the same in-plane column, giving the value profile
    through the cluster. Values below the card's low threshold are still reported
    (flagged as not painted) — they are transparent on the slice, not absent.
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
  * **Under the histogram** is a second copy of that [low, high] pair, drawn as a
    slider **spanning the histogram's own value axis** and aligned to its bars, so
    a handle sits on the values it is cutting: drag it and watch the threshold
    move through the distribution. It and the card's range slider are the same two
    numbers — move either and the other follows. The card's slider keeps the map
    type's fixed scale (0..8+ for z, -1..1 for an average) so a value stays
    meaningful when 🔗 sync mirrors it onto another card; this one is as fine as
    the data, spending its whole track on values that exist in the map. A cap the
    histogram's range cannot reach (a max of 1.0 over data peaking at 0.35) shows
    as a handle parked at the edge; nudging it there sets the cap to the top of
    the distribution.
  * Toggle **🔗 sync** to mirror the view (slice, axis, range, colormap **and the
    crosshair**) across every *other synced card of the same species*: move the
    slice on one and the matching-species cards follow, scales included; click a
    voxel on one and every synced card of that species plants the crosshair on the
    same voxel, each reading out its *own* map's value there. Dog and Human sync
    independently.
  * The card also shows the **model's dissimilarity matrix**, rendered exactly as
    in the RSA Model Builder. Toggle **show matrix** off to hide it.
  * **Beside the matrix** sits the **model-comparison bar plot** — the transpose of
    the brain view. Where the slice asks "where does *this* model fit?", the bars
    ask "at *this voxel*, how do all the models compare?": one bar per model,
    sampled at the crosshair, in menu order (not sorted — a sorted plot would
    reshuffle every time you move the crosshair), with the card's own model in the
    accent colour and its low threshold marked. **Group average** maps carry
    **±SEM** error bars, computed from the ``_std.nii.gz`` map beside each mean and
    the number of maps averaged into it (``file_list`` in the ``_mean.json``
    sidecar); z-maps have no std twin, so they get no error bars. The scope menu
    chooses who takes part: the card's method/fold + grouping (one bar per
    hypothesis, the only fully controlled comparison), that whole method/fold, or
    every model of every distance method.

    It is computed **on demand** — press **📶 Compare**; a scope can be dozens of
    maps on a network disk, so it never rides along with a slider. Toggle **📶
    model bars** off to hide it and give the matrix the full card width.

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

Model bars: how the maps are read (top bar)
-------------------------------------------
The bar plot has to touch every model in scope, so the **Model bars** menu picks
how those maps are read — the difference is memory, not much else:

  * **On request (low memory)** — each map is opened and *only the crosshair
    voxel* is read through nibabel's array proxy; the volume never enters memory.
    Nothing is held between clicks, so every 📶 Compare re-reads.
  * **Preloaded (fast, uses RAM)** — the maps in scope are loaded once and every
    later sample comes from RAM, which is also what lets the plot **follow the
    crosshair live** instead of waiting for the button. The note under the plot
    reports how much is held; **🧹 Free** drops it, as does switching the mode or
    reloading results.

Measured on EmoC/Human (13 models, 2 mm MNI, over the ``P:`` network disk): ~18 s
per Compare on request, versus ~18 s once and then instant — for ~90 MB held.

Display + persistence
---------------------
Brain-view height is adjustable in the top bar; the source mode, data folder,
dataset, view height, model-bar read mode and the **card layout** (order + gap set
in Edit mode) are saved to ``~/.rsa_hypothesis_explorer_settings.json`` and
restored on the next launch. Each card's own selections (distance method, fold,
model, grouping, species, map type, axis, colormap, max, sync, histogram + matrix
show/hide, on/off) are persisted by Dash's local persistence, so the cards come
back as you left them.

Auto-update / manual update
----------------------------
Only the changes that make the app read a **file** are held back: a card's
distance method, fold, model, grouping, **species** and **map type**. Those are
what the top-bar **Auto-update** toggle (on by default) governs. Turn it off and
picking a different model costs nothing until you ask for it — with either

  * the card's own **🔄** button in its header (updates *that card only*), or
  * the top bar's **🔄 Update now** (updates every enabled card).

A gated card shows a "⏸ change pending" note in the meantime.

Everything else is **always live**, because it only re-draws a volume that is
already in RAM: the slice slider and wheel scroll, the axis, both threshold
sliders, the colormap, the crosshair, the view height, card on/off and the
histogram/matrix/bar show-hide toggles, plus the deliberate top-bar actions
(source, ROI, reload, GLM model). The status line next to **Models** in the
header (normally the method/fold/model-family count plus the resolved
``_models.csv`` path) doubles as a general feedback line: it also reports
reloads, add/remove-card, layout-reset and update actions as they happen.

What is redrawn, and what isn't
-------------------------------
Each re-render only rebuilds the parts the change can actually affect, which is
what keeps scrubbing through slices smooth over a network disk:

  * **slice / axis / crosshair** → the brain view only. The in-mask histogram and
    the model matrix do not depend on any of them.
  * **threshold / colormap** → brain + histogram. The matrix still isn't touched.
  * **model change** → everything, matrix included.

Underneath, volumes are cached **by resolved file path** rather than by the
selection that produced it, so moving a threshold handle never re-reads the map
(only a *cluster-corrected* map has the threshold in its file name, and that
correctly resolves to a different file). Model matrices are cached on the CSV's
mtime, mask-restricted value arrays on the map's path, and the raw layout's
directory listings on the folder — all dropped together by **Reload results**.

Terminal log
------------
The app narrates itself to the console it was started from: which card
re-rendered and what triggered it, which files were read off the disk and how
long they took, which renders were skipped and why, model-bar scans, index
rebuilds and cache drops. Werkzeug's one-line-per-request access log
(``POST /_dash-update-component ... 200 -``) is **off** — it fires several times
per slider notch and says nothing about what the app did. Flags:
``--verbose`` also logs cache hits and directory listings, ``--quiet`` logs
warnings only, and ``--http-log`` puts the access log back.

Threshold (per card)
---------------------
A single two-handle **range slider** sets both bounds: the **low** handle
filters voxels below it out (rendered transparent, alpha=0); the **high**
handle caps the color scale — voxels at or above it are painted the top color
of the palette. The slider's meaning and limits depend on the card's map type.

For **Z-map** / **Cluster-corrected** it is a z-range whose default is
**[3.1, the map's own maximum]** — the conventional cluster-forming threshold at
the low end, the brightest voxel actually in the image at the high end, so the
palette spends its full range on the data. That default is recomputed whenever
the image changes (map type, model, grouping, species, source, reload), and the
slider's upper limit grows with it when a map reaches past 8.

For **Group average** it is an average-similarity range, typically Kendall's tau
(-1 to 1, default [0, 1]) since that's this pipeline's default RSA method; a
still-valid current [low, high] survives a switch back to it.

The **slider under the histogram** is the same pair on the distribution's own
scale (see the card description above): the two are mirrored, and dragging either
re-renders the map.

Standalone only (own port, default 8055):
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\hypothesis_explorer.py
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\hypothesis_explorer.py --port 8056

Reuses viz/datasource.py (result resolution), viz/niftiutil.py (atlas + 2D slices)
and viz/hypothesis_tree.py (result-set scan + per-model status).
"""

import argparse
import base64
import json
import logging
import os
import re
import sys
import threading
import time

import numpy as np
import pandas as pd
import plotly.colors as pcolors
import plotly.graph_objects as go
from dash import Dash, ctx, dcc, html, no_update
from dash.dependencies import ALL, Input, Output, State

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

# The group-mean sidecar written by step 3 is named ``*_mean.json`` but is dumped
# with ``yaml.dump`` — so it is YAML text under a .json extension. PyYAML is a
# pipeline dependency, but this app is meant to run anywhere, so its absence only
# costs the error bars, not the app.
try:
    import yaml
except ImportError:
    yaml = None

# --- palette (matches the other viz apps) ---------------------------------
BG, PANEL, INK, MUTED, LINE, ACCENT = "#ffffff", "#f3f5f9", "#222222", "#667085", "#d5dbe5", "#4472C4"
INPUT_STYLE = {"backgroundColor": "#ffffff", "color": INK,
               "border": f"1px solid {LINE}", "borderRadius": "6px", "padding": "5px 8px"}
BTN = {"height": "32px", "padding": "0 14px", "backgroundColor": ACCENT, "color": "white",
       "border": "none", "borderRadius": "6px", "cursor": "pointer", "fontWeight": "bold"}
BTN2 = {**BTN, "backgroundColor": "#eef1f6", "color": INK, "border": f"1px solid {LINE}",
        "fontWeight": "normal"}
# Crosshair read-out under the brain view (voxel / mm / intensity), ITK-SNAP-ish.
CROSS_PANEL_STYLE = {"fontSize": "11px", "backgroundColor": "#ffffff",
                     "border": f"1px solid {LINE}", "borderRadius": "6px",
                     "padding": "4px 7px", "marginTop": "4px", "minHeight": "17px",
                     "display": "flex", "flexWrap": "wrap", "alignItems": "baseline"}

DEFAULT_DATASET = "EmoC"
DEFAULT_GLM_MODEL = "basic-block"
MODALITIES = ["RSA", "GLM"]
MAPTYPES = [("mean", "Group average"), ("z", "Z-map"), ("corrected", "Cluster-corrected")]
# Glyph + caption for the **floating map-type switch** drawn in the corner of the
# brain view (the Google-Maps "Map / Satellite" control). The long label above is
# kept as the tooltip, so a 3-segment pill 120 px wide still says what it is.
MAPTYPE_ICONS = {"mean": ("x̄", "average"), "z": ("Z", "z-map"),
                 "corrected": ("◉", "clusters")}
AXES = [("0", "Slice X"), ("1", "Slice Y"), ("2", "Slice Z")]
SOURCE_MODES = [("drive", "Drive (current-results)"), ("raw", "Raw (results/RSA)")]
# Maps display for a card: which single species' results map to draw. Each card
# is one species (one column) — "Both" is gone; put Dog and Human in two cards.
MAPS_OPTIONS = [("D", "Dog"), ("H", "Human")]
# Overlay colour maps offered per card. Default "Hot": sub-threshold voxels are
# transparent (alpha=0), at/above-threshold values ride the hot scale.
COLORMAPS = ["Hot", "YlOrRd", "Reds", "Viridis", "Cividis", "Jet", "Turbo", "Greys", "Blues"]
DEFAULT_CMAP = "Hot"
DEFAULT_MAPTYPE = "mean"
# Per-card view controls that the "sync" toggle shares across same-species cards.
# Two kinds, because they live on different props: ordinary dcc controls carry
# their state in ``value``, the crosshair in a dcc.Store's ``data``. The crosshair
# is synced too, so clicking a cluster in one card plants the same voxel in every
# other synced card of that species and their read-outs line up.
SYNC_CONTROLS = ["axis", "frac", "range", "cmap"]   # dcc controls -> .value
SYNC_STORES = ["cross"]                             # dcc.Store    -> .data
SYNC_KEYS = SYNC_CONTROLS + SYNC_STORES
MAX_MODELS = 6             # total model-card slots, pre-registered; "Add model" turns the
                           # next one on and its own ✕ turns it off (add / remove)
DEFAULT_CARD_W = 360       # model-card base width (flex-basis, px); cards flex-grow to fill
DEFAULT_GAP = 10           # space between model cards, px
# The matrix and the model-bar plot share one flex row at the bottom of a card:
# each takes half, and whichever is toggled off hands its half to the other.
# ``minWidth: 0`` keeps the Plotly graphs from forcing the row to wrap.
MATRIX_WRAP_SHOW = {"flex": "1 1 0", "minWidth": 0}
BAR_WRAP_SHOW = {"flex": "1 1 0", "minWidth": 0}
WRAP_HIDE = {"display": "none"}
CORRECTED_ZT_TRIES = [3.1, 2.3, 3.9]
HIST_BINS = 60             # bins in the in-mask value histogram drawn beside the brain
HIST_SUB_COLOR = "#c9ced6"  # bar colour below the low handle (those voxels are transparent)
# Plot-area inset of the histogram figure. A second range slider is drawn directly
# under the histogram, padded by these numbers so its track spans the *plotting
# area* rather than the graph div: handle and cut line then sit on the same pixel
# for the same value. They are only the first guess — Plotly grows the left margin
# to fit the y-axis title and tick labels (38 comes back as ~51), so the alignment
# is measured off the rendered figure in the page script and the padding corrected.
HIST_MARGIN_L = 38
HIST_MARGIN_R = 6

# --- model-comparison bar plot (beside the dissimilarity matrix) -----------
# "What does every model say at *this* voxel?" — one bar per model, sampled at the
# card's crosshair. Which models take part is the card's ``scope``:
BAR_SCOPES = [("fold-grouping", "this method/fold + grouping"),
              ("fold", "this method/fold, all groupings"),
              ("all", "every model, every method")]
# How the maps behind those bars are read. This is the difference between one
# network read per model per click and one read per model per *session*:
BAR_MODES = [("request", "On request (low memory)"),
             ("preload", "Preloaded (fast, uses RAM)")]
BAR_ROW_H = 15             # px of bar-plot height per model
BAR_MIN_H = 170
BAR_MAX_MODELS = 80        # cap, so the "every model" scope can't fire off a huge scan
BAR_COLOR = "#9db8e8"      # ordinary model bar
BAR_COLOR_CUR = ACCENT     # the card's own model, so it is findable at a glance

# status -> (colour, human label) for the per-card results-availability dot
STATUS_STYLE = {
    "both":     ("#1a7f37", "Dog + Human"),
    "D":        ("#3b7dd8", "Dog only"),
    "H":        ("#e08a1e", "Human only"),
    "none":     ("#cf4b4b", "No results"),
    "unlinked": ("#c9ced6", "No model"),
}

VERSION = "1.3.1"
LAST_CHANGE = ("Fix: a card's title and model matrix follow the picker even while a "
               "gated change waits for 🔄. The Auto-update gate exists to hold back the "
               "*result volume*, but it was also holding back the two things that depend "
               "only on the model name, so a card switched from correlation to mahalanobis "
               "kept the previous model's 40x40 matrix under a breadcrumb already reading "
               "mahalanobis · emotion_identity — which reads as a mis-drawn matrix, not as "
               "a pending fetch. The model is now resolved before the gate; the pending "
               "branch redraws the matrix from that model's own CSV and titles the card "
               "⏸ pending; the map and histogram still wait. Also: a log line can no "
               "longer take a callback down — the emoji in these messages raised "
               "UnicodeEncodeError on a cp1252 console, killing the very render that was "
               "reporting itself skipped.")


# ---------------------------------------------------------------------------
# Terminal log
# ---------------------------------------------------------------------------
# The app talks to one person watching one console, so it writes plain aligned
# lines rather than going through ``logging``. Werkzeug's per-request line
# ("POST /_dash-update-component ... 200 -") is silenced in ``main`` because it
# fires several times per slider notch and says nothing about what the app did;
# what is printed instead is the *work*: which card re-rendered and why, which
# file was read off the network disk and how long it took, which reads were
# served from cache, and which renders were skipped.
#
# ``--quiet`` drops everything but warnings; ``--http-log`` puts Werkzeug back.

LOG_LEVEL = 1            # 0 = warnings only, 1 = normal, 2 = verbose (cache hits)
_LOG_LOCK = threading.Lock()
_LOG_TAG_W = 8

def log(tag, msg, level=1):
    """One aligned line on stderr-free stdout: ``10:53:43.123  render   …``."""
    if level > LOG_LEVEL:
        return
    stamp = time.strftime("%H:%M:%S") + f".{int((time.time() % 1) * 1000):03d}"
    line = f"{stamp}  {tag:<{_LOG_TAG_W}} {msg}"
    with _LOG_LOCK:
        try:
            print(line, flush=True)
        except UnicodeEncodeError:
            # A console on a legacy code page (cp1252 is still the Windows default
            # without PYTHONIOENCODING) cannot encode the emoji these messages use.
            # A log line must never be able to take a callback down with it, so the
            # unencodable characters are dropped rather than raised.
            enc = getattr(sys.stdout, "encoding", None) or "ascii"
            print(line.encode(enc, "replace").decode(enc, "replace"), flush=True)


def logw(msg):
    log("warn", msg, level=0)


def _mb(arr):
    try:
        return arr.nbytes / 1e6
    except AttributeError:
        return 0.0


class _Timer:
    """``with _Timer() as t:`` … ``t.ms`` — used to say how long a disk read took."""

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.dt = time.time() - self.t0

    @property
    def ms(self):
        return self.dt * 1000.0


# --- caches (module-level; keyed so re-selecting is instant) --------------
_ATLAS = {}          # specie -> (hi, hi_aff, lo_aff, lo_shape)
_ATLAS_ON_GRID = {}  # (specie, shape, aff_hash) -> atlas resampled onto overlay grid
_PATH_CACHE = {}     # (source,…,maptype,zt-if-corrected) -> resolved map path | None
# Volumes are cached by **resolved path**, not by the selection that produced it.
# That is the difference between a threshold drag being free and it re-reading a
# ~50 MB file off the network disk: the card's [low, high] handles do not change
# which file a mean/z map lives in, so the same path comes back and the volume is
# already in RAM. (For a cluster-corrected map the threshold *is* part of the file
# name — a different path, a different entry, correctly.)
_VOL_CACHE = {}      # normalised map path -> (data, aff) | None
_LISTDIR_CACHE = {}  # raw-layout mean folder -> [file names]  (one network listing)
_RESULT_SETS_CACHE = {}  # source-keyed {'D':set,'H':set} of models with results
_MASK_CACHE = {}     # (datafolder,dataset,specie,roi) -> (data,aff,path) | None
_MASK_ON_GRID = {}   # (datafolder,dataset,specie,roi,shape,aff_hash) -> bool array | None
_MASK_VALS = {}      # (map path, datafolder, dataset, specie, roi) -> (values, axis note)
_HEATMAP_CACHE = {}  # (model CSV path, mtime, size) -> the built matrix figure
_MEANLOG_CACHE = {}  # path of a *_mean.json sidecar -> len(file_list) | None
_BAR_PRELOAD = {}    # bar-plot "preloaded" mode: ctx key -> {map path: (data, aff)}
# Per-card memo of what was last *pushed to the browser*, so a re-render that
# cannot have changed the matrix does not ship a 40x40 heatmap down the wire.


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
    "bar_mode": "request",     # model-comparison bar plot: read on demand vs preload
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


def _mask_values(data, datafolder, dataset, specie, roi, aff, map_key=None):
    """(values, axis note) — the map's values inside the search mask, plus a short
    label saying where they came from. Falls back to the map's own non-zero voxels
    when no mask file can be found, and says so, rather than silently
    histogramming the whole (mostly empty) bounding box.

    Cached on ``map_key`` (the map's resolved path) when one is given: pulling a
    few hundred thousand voxels out of a full volume is the same answer every time
    for one map + mask, and the histogram is rebuilt on every threshold move."""
    key = (map_key, datafolder, dataset, specie, roi) if map_key else None
    if key is not None and key in _MASK_VALS:
        return _MASK_VALS[key]
    mb, resampled = _mask_bool_for_grid(datafolder, dataset, specie, roi, data.shape, aff)
    if mb is None:
        out = (data[np.abs(data) > 1e-6], "value · non-zero voxels (no mask file found)")
    elif resampled:
        out = (data[mb], f"value · ⚠ {roi} resampled — mask and map are on different grids")
    else:
        out = (data[mb], f"value · voxels in {roi}")
    vals = out[0][np.isfinite(out[0])]
    out = (vals, out[1])
    if key is not None:
        _MASK_VALS[key] = out
    return out


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
    """Names in a raw-layout ``mean`` folder, cached per folder.

    This is a network listing and every model/species/maptype combination the bar
    plot resolves asks for the same one, so it is read once per folder per session
    and dropped by "Reload results"."""
    if folder in _LISTDIR_CACHE:
        return _LISTDIR_CACHE[folder]
    with _Timer() as t:
        try:
            names = os.listdir(folder)
        except OSError:
            names = []
    _LISTDIR_CACHE[folder] = names
    log("disk", f"listdir {len(names):>3} entries  {t.ms:6.0f} ms  {folder}", level=2)
    return names


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
    """Resolved file for one model's map, or None. Cached — resolving is not free
    on the network disk (the raw layout lists a directory, the drive layout stats
    several candidates), and the model-bar plot resolves *every model in scope* on
    every recompute, which is what that cost is actually paid for. Dropped by
    "Reload results" along with the loaded maps.

    ``zt`` only reaches the file name for a **cluster-corrected** map, so it is
    dropped from the key for every other map type. Without that, each notch of the
    threshold slider was a fresh cache miss and a fresh directory listing / stat
    storm on the network disk for a file that never moved."""
    zt_key = round(float(zt), 2) if maptype == "corrected" else None
    key = (source, datafolder, dataset, modality, roi, glm_model, specie, model,
           maptype, zt_key)
    if key not in _PATH_CACHE:
        if source == "raw":
            p = _raw_map_path(datafolder, dataset, glm_model, roi, specie, model, maptype, zt)
        else:
            p = _drive_map_path(datafolder, dataset, modality, roi, specie, model, maptype, zt)
        _PATH_CACHE[key] = p
        if p is None:
            log("path", f"no {maptype} map for {specie} · {model}", level=2)
    return _PATH_CACHE[key]


def resolve_result_sets(source, datafolder, dataset, modality, roi, glm_model):
    """{'D': set, 'H': set} of models that have results, for the active source.
    Cached (cleared by 'Reload results') because the raw scan touches many dirs."""
    if source == "raw":
        key = ("raw", datafolder, dataset, glm_model)
    else:
        key = ("drive", datafolder, dataset, modality, roi)
    if key not in _RESULT_SETS_CACHE:
        with _Timer() as t:
            if source == "raw":
                sets = raw_models_with_results(datafolder, dataset, glm_model)
            else:
                sets = (ht.models_with_results(datafolder, dataset, modality, roi)
                        if roi else {"D": set(), "H": set()})
        _RESULT_SETS_CACHE[key] = sets
        log("scan", f"results ({source}) — {len(sets.get('D', ()))} D / "
                    f"{len(sets.get('H', ()))} H models  {t.ms:.0f} ms")
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


def _load_volume(path):
    """``(data, affine)`` for one NIfTI, cached **by path** — or None.

    Keyed by the file rather than by the selection that resolved to it, so two
    cards showing the same map, and every threshold/colormap/slice change on one
    card, share a single read. This is the cache that matters: on ``P:`` a 2 mm MNI
    volume is a multi-second read, and the old key carried the threshold, so every
    notch of the slider paid it again."""
    if not path:
        return None
    key = os.path.normcase(os.path.abspath(path))
    if key in _VOL_CACHE:
        log("cache", f"hit  {os.path.basename(path)}", level=2)
        return _VOL_CACHE[key]
    result = None
    with _Timer() as t:
        try:
            data, aff, _hdr = niftiutil.load_nifti(path)
            result = (data, aff)
        except Exception as e:
            logw(f"could not read {path}: {e}")
    _VOL_CACHE[key] = result
    if result is not None:
        # Two models' maps often share a basename and differ only in the folder
        # above them, so the log shows the tail of the path, not just the name.
        tail = "/".join(path.replace("\\", "/").split("/")[-3:])
        log("disk", f"read {tail}  {'x'.join(str(s) for s in result[0].shape)}  "
                    f"{_mb(result[0]):.1f} MB  {t.ms:.0f} ms")
    return result


def _load_map(source, datafolder, dataset, modality, roi, glm_model, specie, model, maptype, zt):
    """The volume behind one card's selection, via the resolved-path cache."""
    if not model:
        return None
    path = _map_path_for(source, datafolder, dataset, modality, roi, glm_model,
                         specie, model, maptype, zt)
    return _load_volume(path)


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
    """``rsa_models`` folders to search for model CSVs / the manifest, for **this**
    data folder and **this** dataset: ``{datafolder}/{dataset}/rsa_models`` first,
    then the same folder on the canonical pipeline data disk (where
    ``build_rsa_models.py`` writes). Nothing here is project-specific — point the
    top bar at another dataset and its own ``_models.csv`` and model CSVs are what
    the menus are built from.

    The pipeline-disk entry is what lets models authored there show up in the
    explorer even when results are being viewed from the Google Drive mirror, which
    may not have those CSVs synced yet. ``models_manifest`` owns the resolution so
    every reader of the manifest looks in exactly the same places."""
    return mm.rsa_models_dirs(datafolder, dataset)


def manifest_file(datafolder, dataset):
    """The ``_models.csv`` actually in use for this (datafolder, dataset), or None.
    Shown in the header so it is never a guess which project's manifest is driving
    the menus."""
    return mm.manifest_path(_model_dirs(datafolder, dataset))


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


FALLBACK_DIS = "(all models)"
FOLD_ANY = mm.FOLD_ANY      # the pooled entry used whenever the fold level is skipped


def build_index(datafolder, dataset):
    """The menu backing every card, **distance method first**::

        {'dis_methods': [dis_method, ...],
         'by_dis': {dis_method: {'uses_fold': bool, 'folds': [fold, ...],
                                 'by_fold': {fold: {'stems': [...],
                                                    'index': {stem: {grouping: model}},
                                                    'why': {stem: why},
                                                    'groupings': {stem: [...]}},
                                             FOLD_ANY: {...}}}}}

    Driven by the dataset's central ``rsa_models/_models.csv`` (via
    ``models_manifest``), resolved from the *current* data folder + dataset — so
    another project's explorer session reads that project's manifest. Serialised
    into the ``ex-grouped`` store and rebuilt whenever the data folder / dataset
    changes or results are reloaded.

    When no manifest is present it falls back to scanning the folder and offering
    every valid ``__{grouping}`` model under one synthetic ``(all models)`` method,
    so the explorer still works before ``_models.csv`` exists."""
    idx = mm.dis_index(_model_dirs(datafolder, dataset))
    if idx["dis_methods"]:
        return idx
    # Fallback: no _models.csv — discover models by scanning the rsa_models folder.
    grouped = grouped_valid_models(datafolder, dataset)     # {stem: {grouping: model}}
    stems = ordered_valid_stems(datafolder, dataset)
    why = {s: model_description(datafolder, dataset, next(iter(grouped[s].values()), None))
           for s in stems}
    groupings = {s: [g for g in GROUPINGS if g in grouped[s]] for s in stems}
    return {"dis_methods": [FALLBACK_DIS],
            "by_dis": {FALLBACK_DIS: {
                "uses_fold": False, "folds": [],
                "by_fold": {FOLD_ANY: {"stems": stems, "index": grouped,
                                       "why": why, "groupings": groupings}}}}}


def _dis_data(grouped, dis):
    """The ``by_dis`` entry for one distance method, or an empty skeleton."""
    by_dis = (grouped or {}).get("by_dis", {}) if isinstance(grouped, dict) else {}
    return by_dis.get(dis or "", {}) if isinstance(by_dis, dict) else {}


def _dis_folds(grouped, dis):
    """(folds to offer, is the fold menu meaningful) for one distance method. A
    method that does not fold offers the single pooled ``FOLD_ANY`` entry, and its
    fold menu is hidden rather than shown with one meaningless option."""
    dd = _dis_data(grouped, dis)
    if dd.get("uses_fold") and dd.get("folds"):
        return list(dd["folds"]), True
    return [FOLD_ANY], False


def _fold_data(grouped, dis, fold):
    """The ``by_fold`` entry for one (distance method, fold), or an empty skeleton."""
    by_fold = _dis_data(grouped, dis).get("by_fold", {})
    if not isinstance(by_fold, dict):
        return {}
    return by_fold.get(fold or "", {})


def _resolve_model(grouped, dis, fold, stem, grouping):
    """Concrete ``{stem}__{grouping}`` model for a card, or None if any part is
    unset / unknown in the current index."""
    if not dis or not fold or not stem or not grouping:
        return None
    return _fold_data(grouped, dis, fold).get("index", {}).get(stem, {}).get(grouping)


# --- RSA model matrix, drawn with the RSA Model Builder's renderer ---------

def _model_heatmap(datafolder, dataset, model):
    """Return a Plotly figure of the linked model's dissimilarity matrix, rendered
    with rsa_model_builder.build_cell_heatmap and the model's saved _style.json
    (falls back to a compact default so a 40x40 matrix stays readable).

    Cached on (CSV path, mtime, size): the CSV and its style sidecar live on the
    network disk and building a 40x40 cell heatmap is not free, while the matrix
    itself only changes when the file does. Edit the model in the builder and the
    mtime moves, so the next render picks the new matrix up by itself."""
    if not model:
        return niftiutil.empty_fig("Pick a model + grouping to load its matrix.", height=200)
    path = _find_model_csv(datafolder, dataset, model)
    if not path:
        return niftiutil.empty_fig(f"No matrix CSV for '{model}'.", height=200)
    try:
        st = os.stat(path)
        key = (path, st.st_mtime, st.st_size)
    except OSError:
        key = (path, None, None)
    if key in _HEATMAP_CACHE:
        log("matrix", f"cached  {model}", level=2)
        return _HEATMAP_CACHE[key]
    with _Timer() as t:
        try:
            df = pd.read_csv(path, index_col=0)
            labels = [str(x) for x in df.index]
            matrix = rmb.enforce_invariants(df.values.astype(float))
        except Exception as e:
            logw(f"matrix unreadable for {model}: {e}")
            return niftiutil.empty_fig(f"Matrix unreadable: {e}", height=200)
        sidecar = rmb.load_style_sidecar(path)
        fig_style = sidecar.get("figure_style") if isinstance(sidecar, dict) else None
        if fig_style:
            style = {**rmb.DEFAULT_STYLE, **fig_style}
        else:  # no saved style — compact so the full matrix fits the card
            style = {**rmb.DEFAULT_STYLE, "cell_size": 20, "val_font_size": 8, "label_font_size": 9}
        fig = rmb.build_cell_heatmap(matrix, labels, style)
    _HEATMAP_CACHE[key] = fig
    log("matrix", f"built  {model}  {len(labels)}x{len(labels)}  {t.ms:.0f} ms")
    return fig


# ---------------------------------------------------------------------------
# The icon vocabulary the pickers are drawn from
# ---------------------------------------------------------------------------
# Everything a card asks about is a statement over the **stimulus set** an RDM is
# built on, and that set is a two-level categorisation: an **agent block** (who is
# shown -- Dog or Human) crossed with a **category** (which emotion), each cell
# holding one or more **exemplars**. Draw that grid and the questions answer
# themselves: which set am I in (how many exemplars per cell), and which pairs of
# it enter (the grouping). That is strictly more informative than the words
# ``mahalanobis / stim-wise`` and ``all / within / cross / dog / hum``, which only
# mean something once you already know the design.
#
# Nothing here is EmoC-specific by construction: rows, columns and exemplar count
# are read off a model's own dissimilarity matrix (``_set_taxonomy``), and a label
# this table has no icon for falls back to its own initials in a chip. Point the
# app at a dataset with different conditions and it draws that dataset's grid.
AGENT_ICONS = {"dog": ("\U0001f415", "Dog-shown"), "hum": ("\U0001f9d1", "Human-shown"),
               "human": ("\U0001f9d1", "Human-shown")}
CATEGORY_ICONS = {                    # EmoC's five conditions, in stimulus order
    "P": ("\U0001f929", "Positive anticipation"),
    "H": ("\U0001f600", "Happiness"),
    "A": ("\U0001f620", "Anger"),
    "F": ("\U0001f628", "Fear"),
    "N": ("\U0001f610", "Neutral"),
}
# ``DogP`` -> (Dog, P, ""), ``DogP1`` -> (Dog, P, "1"): the trailing digits are the
# exemplar index, and whether they are there is exactly what separates the two
# stimulus sets of this dataset.
_STIM_RE = re.compile(r"^(dog|hum(?:an)?)([A-Za-z]+?)(\d*)$", re.I)

# grouping -> (tile caption, agent rows that take part | None = every row,
#              how the participating stimuli are boxed)
#   "one"    : one box around everything   -- every pair inside it enters
#   "each"   : one box per agent row       -- only pairs *inside* a row enter
#   "bridge" : a box per agent row, joined -- only pairs that *cross* them enter
GROUPING_TILE = {
    "all":      ("Pooled",       None, "one"),
    "collapse": ("Pooled",       None, "one"),
    "within":   ("Within",       None, "each"),
    "cross":    ("Cross",        None, "bridge"),
    "dog":      ("Dog only",     [0],  "one"),
    "hum":      ("Human only",   [1],  "one"),
}

_TAXONOMY_CACHE = {}
_TAXONOMY_LOGGED = set()


def _split_stim_label(label):
    """``"DogP1"`` -> ``("Dog", "P", "1")``; unparseable -> ``(None, label, "")``."""
    m = _STIM_RE.match(str(label).strip())
    if not m:
        return None, str(label), ""
    return m.group(1), m.group(2), m.group(3)


def _set_taxonomy(datafolder, dataset, grouped, dis, fold):
    """``(agents, categories, exemplars_per_cell)`` for **one** (distance method,
    fold) set -- the stimulus set its RDMs are built over. ``agents`` and
    ``categories`` are ``(key, icon, title)`` triples.

    Read off one model's dissimilarity matrix from *that set*: every model in a set
    shares its row labels, so a single CSV names the whole set. This is per-set
    rather than per-dataset because the sets genuinely differ -- on EmoC,
    ``mahalanobis / stim-wise`` is a 10x10 RDM over conditions (``DogP``) while
    ``correlation / run-wise`` is a 40x40 over four exemplars each (``DogP1``..4),
    and that difference is the whole content of the first picker. Cached per
    (data folder, dataset, method, fold): a network read whose answer only moves
    when the battery does. Falls back to one unnamed block whose "categories" are
    the raw labels when they do not parse as ``{agent}{category}{n}``, so a dataset
    that is not organised that way still gets a (flat) picture."""
    key = (datafolder, dataset, dis, fold)
    if key in _TAXONOMY_CACHE:
        return _TAXONOMY_CACHE[key]
    labels = []
    for model in _set_models(grouped, dis, fold, limit=4):
        path = _model_csv_map(datafolder, dataset).get(model)
        if not path:
            continue
        try:
            labels = [str(x) for x in pd.read_csv(path, index_col=0, nrows=0).columns]
        except Exception as e:
            logw(f"taxonomy: could not read {path}: {e}")
            continue
        if labels:
            break
    agents, cats, per_cell = [], [], {}
    seen_a, seen_c = set(), set()
    for lab in labels:
        a, c, _idx = _split_stim_label(lab)
        if a is None:                                   # not {agent}{category}{n}
            agents, cats = [], []
            break
        if a.lower() not in seen_a:
            seen_a.add(a.lower())
            agents.append((a,) + AGENT_ICONS.get(a.lower(), ("\u25fc", a)))
        if c not in seen_c:
            seen_c.add(c)
            cats.append((c,) + CATEGORY_ICONS.get(c, (c[:2], c)))
        per_cell[(a.lower(), c)] = per_cell.get((a.lower(), c), 0) + 1
    reps = max(per_cell.values()) if per_cell else 1
    if not agents:                                      # flat / unreadable fallback
        agents = [("", "\u25fc", "all stimuli")]
        cats = [(l, (l[:2] or "?"), l) for l in labels[:8]] or [("?", "\u2022", "stimulus")]
        reps = 1
    out = (agents, cats, reps)
    _TAXONOMY_CACHE[key] = out
    if key not in _TAXONOMY_LOGGED:      # six cards ask at once and all miss the cache
        _TAXONOMY_LOGGED.add(key)
        log("index", f"stimulus set {dis} / {fold}: {len(agents)} agent block(s) x "
                     f"{len(cats)} categories x {reps} exemplar(s) = {len(labels)} stimuli")
    return out


def _set_size_note(agents, cats, reps, n):
    """The one-line "what is in this RDM" caption under a set's grid."""
    if reps > 1:
        return f"{n} stimuli \u00b7 {reps} per condition"
    return f"{n} conditions"


def _icon_cell(icon, title, reps):
    """One category icon, carrying its exemplar count when a cell holds more than
    one -- ``\U0001f620\u2074`` says "four anger clips", which is the difference between the
    two sets at a glance."""
    kids = [icon]
    if reps > 1:
        kids.append(html.Span(f"\u00d7{reps}", className="pl-ic-rep"))
    return html.Span(kids, className="pl-ic-cat", title=title)


def _icon_row(agent, cats, reps=1, dim=False):
    """One agent block: its own icon, then one cell per category."""
    cls = "pl-ic-row" + (" pl-ic-dim" if dim else "")
    return html.Div(className=cls, children=[
        html.Span(agent[1], className="pl-ic-agent", title=agent[2]),
        *[_icon_cell(ic, f"{title} ({agent[2]})" + (f" \u00d7{reps}" if reps > 1 else ""), reps)
          for _k, ic, title in cats],
    ])


def _set_tile(dis, fold, real, agents, cats, reps, n_models):
    """One tile of picker (1): the stimulus set a (distance method, fold) pair
    works over, drawn as its grid, with the method/fold naming it underneath."""
    n = len(agents) * len(cats) * reps
    return html.Span(className="pl-tile pl-tile-set",
                     title=f"{dis}" + (f" \u00b7 {fold}" if real else "") +
                           f" \u2014 {n}x{n} RDM over {n} stimuli, {n_models} models",
                     children=[
        html.Span(_set_size_note(agents, cats, reps, n), className="pl-tile-cap"),
        html.Span(className="pl-ic-stack", children=[
            html.Div(className="pl-ic-box", children=[
                _icon_row(a, cats, reps) for a in agents]),
        ]),
        html.Span(f"{dis}" + (f" \u00b7 {fold}" if real else ""), className="pl-tile-badge"),
        html.Span(f"{n_models} model{'' if n_models == 1 else 's'}",
                  className="pl-tile-badge"),
    ])


def _grouping_diagram(grouping, agents, cats, reps=1):
    """``(caption, [components])`` -- the picture of one grouping: which agent
    blocks take part and how they are boxed. The box **is** the claim: pairs
    inside a box enter the analysis, and a ``bridge`` says only the pairs spanning
    the two boxes do. Rows that do not take part stay on screen, dimmed, so a
    single-block grouping reads as "this half, not that one"."""
    caption, keep, mode = GROUPING_TILE.get(grouping, (grouping, None, "one"))
    idxs = list(range(len(agents))) if keep is None else [k for k in keep if k < len(agents)]
    if not idxs:                                   # e.g. "hum" on a one-block dataset
        idxs = [0]
    if mode == "one":
        body = [html.Div(className="pl-ic-box", children=[
            _icon_row(agents[k], cats, reps) for k in idxs])]
        body += [_icon_row(agents[k], cats, reps, dim=True)
                 for k in range(len(agents)) if k not in idxs]
    else:
        body = []
        for n, k in enumerate(idxs):
            if n and mode == "bridge":
                body.append(html.Div("\u21c5", className="pl-ic-link",
                                     title="only pairs that cross the two blocks"))
            body.append(html.Div(className="pl-ic-box",
                                 children=[_icon_row(agents[k], cats, reps)]))
    return caption, body


def _grouping_tile(grouping, agents, cats, reps=1):
    """One tile of picker (3): which pairs of the set's stimuli enter the RSA."""
    caption, body = _grouping_diagram(grouping, agents, cats, reps)
    return html.Span(className="pl-tile pl-tile-grp",
                     title=GROUPING_DESC.get(grouping, grouping), children=[
        html.Span(caption, className="pl-tile-cap"),
        html.Span(className="pl-ic-stack", children=body),
    ])


# --- model gallery: each model shown as its own dissimilarity matrix --------

_THUMB_CACHE = {}
_CSVMAP_CACHE = {}
THUMB_PX = 54                 # thumbnail box, px (cells shrink to fit an N x N matrix)
THUMB_LO, THUMB_HI = (247, 250, 253), (26, 51, 92)   # 0 -> near-white, 1 -> deep blue


def _model_csv_map(datafolder, dataset):
    """``{model name: absolute CSV path}`` for the dataset, built from **one**
    directory listing per model folder.

    The gallery needs a path per model to draw its matrix, and the model folder
    lives on a network disk where a single ``os.path.isfile`` costs tens of
    milliseconds -- eighteen models times two folders is seconds of stat traffic
    for information one listing already carries."""
    key = (datafolder, dataset)
    if key in _CSVMAP_CACHE:
        return _CSVMAP_CACHE[key]
    out = {}
    for folder in reversed(_model_dirs(datafolder, dataset)):   # first dir wins
        try:
            entries = os.listdir(folder)
        except Exception:
            continue
        for fn in entries:
            if fn.lower().endswith(".csv") and not fn.startswith("_"):
                out[fn[:-4]] = os.path.join(folder, fn)
    _CSVMAP_CACHE[key] = out
    return out


def _set_models(grouped, dis, fold, limit=4):
    """A few concrete model names from **one** (method, fold) set -- enough to find
    one readable matrix to read that set's stimulus labels off."""
    out = []
    for variants in _fold_data(grouped, dis, fold).get("index", {}).values():
        for name in variants.values():
            out.append(name)
            if len(out) >= limit:
                return out
    return out


def _thumb_color(v):
    t = 0.0 if not np.isfinite(v) else min(max(float(v), 0.0), 1.0)
    return "#%02x%02x%02x" % tuple(int(round(a + (b - a) * t))
                                   for a, b in zip(THUMB_LO, THUMB_HI))


def _matrix_thumb(datafolder, dataset, model):
    """A ``data:`` URI SVG of the model's dissimilarity matrix, small enough to be
    an icon -- the model's own picture standing in for its name in the gallery.

    Drawn as raw rects rather than through the builder's Plotly renderer: a
    gallery is a dozen-plus matrices at once, and a Plotly figure each would cost
    both the render and a dozen live graph divs in the page. Cached on
    (path, mtime, size) exactly like the full matrix, so editing a model in the
    builder refreshes its thumbnail by itself."""
    path = _model_csv_map(datafolder, dataset).get(model or "")
    if not path:
        return None
    try:
        st = os.stat(path)
        key = (path, st.st_mtime, st.st_size)
    except OSError:
        key = (path, None, None)
    if key in _THUMB_CACHE:
        return _THUMB_CACHE[key]
    try:
        m = pd.read_csv(path, index_col=0).values.astype(float)
    except Exception as e:
        logw(f"thumbnail unreadable for {model}: {e}")
        _THUMB_CACHE[key] = None
        return None
    n = int(min(m.shape)) if m.size else 0
    if n < 1:
        _THUMB_CACHE[key] = None
        return None
    finite = m[np.isfinite(m)]
    lo = float(np.min(finite)) if finite.size else 0.0
    hi = float(np.max(finite)) if finite.size else 1.0
    span = (hi - lo) or 1.0
    cell = max(2, THUMB_PX // n)
    side = cell * n
    rects = []
    for r in range(n):
        for c in range(n):
            rects.append(f'<rect x="{c * cell}" y="{r * cell}" width="{cell}" '
                         f'height="{cell}" fill="{_thumb_color((m[r, c] - lo) / span)}"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{side}" height="{side}" '
           f'viewBox="0 0 {side} {side}" shape-rendering="crispEdges">'
           + "".join(rects) +
           f'<rect x="0.5" y="0.5" width="{side - 1}" height="{side - 1}" fill="none" '
           f'stroke="{LINE}" stroke-width="1"/></svg>')
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")
    _THUMB_CACHE[key] = uri
    return uri


def _stem_tile(datafolder, dataset, model, stem, why, groupings):
    """One tile of picker (2): the hypothesis **is** its matrix, its name, and an
    (i) badge carrying the ``why`` column of ``_models.csv`` as its tooltip.

    ``model`` is a *representative* variant -- the grouping is picked in the next
    step, and a grouping only masks cells out of the same matrix, so the thumbnail
    shows the hypothesis itself. The groupings the stem offers are listed under the
    name, which is also the honest way to say that some hypotheses do not offer
    all of them."""
    uri = _matrix_thumb(datafolder, dataset, model)
    icon = (html.Img(src=uri, className="pl-thumb", alt=stem, title=stem) if uri
            else html.Span("\u25a6", className="pl-thumb pl-thumb-none", title=stem))
    caps = " ".join(GROUPING_TILE.get(g, (g,))[0] for g in groupings)
    return html.Span(className="pl-tile pl-tile-stem", children=[
        icon,
        html.Span(className="pl-tile-txt", children=[
            html.Span(stem, className="pl-tile-name"),
            html.Span("\u24d8", className="pl-why-ic",
                      title=why or "no description in _models.csv"),
            html.Span(caps, className="pl-tile-grps",
                      title="groupings this hypothesis offers"),
        ]),
    ])


# --- picker (1): one entry per stimulus set --------------------------------
# ``_models.csv`` keys its models on (``dis_method``, ``mah_fold``), and that pair
# is not a pair of settings -- it is **which stimulus set the RDM is built over**,
# and therefore which models exist at all. On EmoC the two combinations are a
# 10x10 RDM over conditions (mahalanobis / stim-wise) and a 40x40 over four
# exemplars per condition (correlation / run-wise); a model built for one is not
# comparable with the other's. So the card asks for the **set** first, as a picture
# of that set, and everything downstream is filtered by it.
SET_SEP = "\u0001"           # not a character a method or fold name can hold


def _set_key(dis, fold):
    return f"{dis}{SET_SEP}{fold}"


def _split_set(value):
    """``(dis, fold)`` for a picker value, or ``(None, None)``."""
    parts = (value or "").split(SET_SEP)
    if len(parts) != 2 or not all(parts):
        return None, None
    return parts[0], parts[1]


def _set_entries(grouped):
    """``[(dis, fold, fold-is-a-real-choice, n models), ...]`` -- every stimulus
    set the dataset offers, in manifest order. A method that does not fold
    contributes its single pooled ``FOLD_ANY`` entry, and its tile names only the
    method, since there is no fold to state."""
    out = []
    for dis in (grouped or {}).get("dis_methods", []):
        folds, real = _dis_folds(grouped, dis)
        for fold in folds:
            stems = _fold_data(grouped, dis, fold).get("stems", [])
            out.append((dis, fold, real, len(stems)))
    return out


def _set_summary(dis, fold):
    """The one line a chosen set collapses to in the card's breadcrumb."""
    tail = f" \u00b7 {fold}" if fold and fold != FOLD_ANY else ""
    return f"{dis}{tail}"


def _grouping_summary(grouping):
    return GROUPING_TILE.get(grouping, (grouping,))[0]


# --- floating colour switch: every palette shown as itself -----------------

_GRADIENT_CACHE = {}


def _cmap_gradient(name, stops=9):
    """A CSS ``linear-gradient`` for a Plotly colorscale, so the floating colour
    switch offers each palette as a swatch of itself rather than as its name."""
    key = (name, stops)
    if key in _GRADIENT_CACHE:
        return _GRADIENT_CACHE[key]
    try:
        cols = pcolors.sample_colorscale(
            name, [k / (stops - 1.0) for k in range(stops)], colortype="rgb")
    except Exception:
        cols = ["#ffffff", "#000000"]
    grad = "linear-gradient(to right, " + ", ".join(cols) + ")"
    _GRADIENT_CACHE[key] = grad
    return grad

# ---------------------------------------------------------------------------
# Model comparison at one voxel — the bar plot beside the matrix
# ---------------------------------------------------------------------------
# The card's map answers "where does *this* model fit the data?". The bar plot
# answers the transposed question — "at *this voxel*, how do all the models
# compare?" — by sampling every model's group map at the crosshair.
#
# It is deliberately **button-driven**: a card's scope can span dozens of models
# and each one is a separate file on a network disk, so it must never ride along
# with a slider drag. Two read strategies are offered (top bar, ``ex-barmode``):
#
#   * **on request** — each map is opened and the single crosshair voxel is read
#     through nibabel's array proxy, so the volume never enters memory. Cheap in
#     RAM, one disk read per model per click.
#   * **preloaded** — the maps in scope are loaded once into ``_BAR_PRELOAD`` and
#     every later click samples RAM. Fast and it lets the plot follow the
#     crosshair live, at the cost of holding the volumes (reported in the note).
#
# The two cost about the same wall time on a *cold* read — whichever runs first
# pays the network, the second is served from the OS page cache — so the choice is
# genuinely about memory and about whether repeat clicks are free. Measured on
# EmoC/Human (13 models, 2 mm MNI, over ``P:``): ~18 s per Compare on request, vs
# ~18 s once + ~0 s per click after, for ~90 MB held.
#
# Sampling goes through the **world (mm)** coordinate rather than the voxel index,
# so a model whose map happens to sit on another grid is still read at the same
# anatomical point instead of silently at the wrong one.

def _std_and_log_paths(mean_path):
    """(std map, mean-log sidecar) beside a ``*_mean.nii.gz`` group map — the two
    files step 3 writes next to it — or (None, None) for anything else (a z-map
    has no std twin)."""
    tail = "_mean.nii.gz"
    if not mean_path or not mean_path.endswith(tail):
        return None, None
    return mean_path[:-len(tail)] + "_std.nii.gz", mean_path[:-len(".nii.gz")] + ".json"


def _mean_log_n(path):
    """``len(file_list)`` from a group map's ``*_mean.json`` sidecar — how many
    participant/run maps were averaged into it, i.e. the *n* behind the standard
    error. None when the sidecar is missing or has no file list.

    Despite the extension the file is YAML (step 3 writes it with ``yaml.dump``),
    so JSON is tried first and YAML second. Cached per path: this is one network
    read per model, and the sidecar lists every input file."""
    if not path:
        return None
    if path not in _MEANLOG_CACHE:
        n = None
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError:
            text = None
        if text is not None:
            doc = None
            try:
                doc = json.loads(text)
            except ValueError:
                if yaml is not None:
                    try:
                        doc = yaml.safe_load(text)
                    except Exception:
                        doc = None
            if isinstance(doc, dict) and isinstance(doc.get("file_list"), (list, tuple)):
                n = len(doc["file_list"]) or None
        _MEANLOG_CACHE[path] = n
    return _MEANLOG_CACHE[path]


def _bar_model_list(grouped, dis, fold, grouping, scope):
    """(models to compare, label mode) for a card's bar plot.

    ``"fold-grouping"`` — one model per hypothesis stem in the card's distance
    method + fold, all at the card's own grouping: the comparison that holds
    everything but the hypothesis fixed, so the bars are directly comparable.
    Labels drop the shared ``__{grouping}`` suffix. ``"fold"`` adds every grouping
    of those stems and ``"all"`` spans every distance method; both label with the
    full model name, and both mix settings that are *not* held constant — read them
    accordingly. Note that models from two distance methods are not comparable at
    all (different pairwise-similarity maps underneath), which is why ``"all"`` is
    the widest and loosest scope."""
    if scope == "all":
        out = []
        for d in (grouped or {}).get("dis_methods", []) if isinstance(grouped, dict) else []:
            # the pooled entry already spans every fold of that method
            for variants in _fold_data(grouped, d, FOLD_ANY).get("index", {}).values():
                out.extend(variants.values())
        return sorted(dict.fromkeys(out)), "model"
    fd = _fold_data(grouped, dis, fold)
    index = fd.get("index", {}) or {}
    stems = fd.get("stems", []) or list(index)
    if scope == "fold":
        out = []
        for s in stems:
            out.extend((index.get(s) or {}).values())
        return list(dict.fromkeys(out)), "model"
    out = [(index.get(s) or {}).get(grouping) for s in stems]
    return [m for m in out if m], "stem"


def _preload_volume(store, path):
    """The volume at ``path`` held in the preload store (loaded on first ask),
    or None when it cannot be read. Kept as float32 — the maps are written as
    float64, and halving them halves the memory a scope costs."""
    if path not in store:
        try:
            data, aff, _hdr = niftiutil.load_nifti(path)
        except Exception:
            store[path] = None
        else:
            store[path] = (np.asarray(data, dtype=np.float32), aff)
    return store[path]


def _sample_at(path, world, store):
    """Value of the map at ``path`` at world (mm) point ``world``, or None.

    ``store`` None is the low-memory path: read the one voxel off disk and keep
    nothing. A dict is the preload store: the whole volume is held and sampled
    from RAM."""
    if store is None:
        val, _vox = niftiutil.sample_world_value(path, world)
        return val
    vol = _preload_volume(store, path)
    if vol is None:
        return None
    data, aff = vol
    vox = niftiutil.world_to_voxel(world, aff)
    if any(v < 0 or v >= s for v, s in zip(vox, data.shape)):
        return None
    v = float(data[vox[0], vox[1], vox[2]])
    return v if np.isfinite(v) else None


def _bar_series(source, datafolder, dataset, modality, roi, glm_model, specie,
                models, maptype, zt, world, store):
    """``[(model, value, sem, n), ...]`` — every model that actually has a map of
    this type, sampled at ``world``. Models without one are simply skipped: the
    plot shows the models "for which there is available data".

    For a **group-average** map the error bar is the standard error of that voxel's
    mean: the value in the ``_std`` map beside it over ``sqrt(n)``, with *n* the
    number of maps that went into the average (``file_list`` in the ``_mean.json``
    sidecar). Note the pipeline writes that std with ddof=0 and *n* counts averaged
    **maps** — participant runs, not participants — so the bars are within-map
    dispersion, not a between-subject CI. z-maps carry no std twin, so they get no
    error bars."""
    out = []
    for model in models:
        path = _map_path_for(source, datafolder, dataset, modality, roi, glm_model,
                             specie, model, maptype, zt)
        if not path:
            continue
        val = _sample_at(path, world, store)
        if val is None:
            continue
        sem, n = None, None
        std_p, log_p = _std_and_log_paths(path)
        if std_p:
            n = _mean_log_n(log_p)
            sd = _sample_at(std_p, world, store) if n else None
            if sd is not None and sd >= 0:
                sem = float(sd) / np.sqrt(float(n))
        out.append((model, float(val), sem, n))
    return out


def _bar_label(model, grouping, label_mode):
    """Axis label for one bar — the bare stem when the grouping is held constant
    across the whole plot (it is already in the plot's title), else the full name."""
    if label_mode == "stem" and grouping and model.endswith(f"__{grouping}"):
        return model[: -len(f"__{grouping}")]
    return model


def _bar_fig(rows, current, grouping, label_mode, lo, xtitle, subtitle):
    """(figure, height px) — horizontal bar chart of every model's value at the
    crosshair. The height is returned because it grows with the number of bars and
    the Graph's own style has to be set to match, or Plotly draws into a box of the
    wrong size.

    Bars keep the **menu order**, not value order: the crosshair moves constantly
    and a sorted plot would reshuffle under it, whereas a fixed order lets you
    watch one model's bar as you scroll through slices and compare the same row
    between two cards. The card's own model is drawn in the accent colour, the
    card's low threshold as a dotted line, so "does my model win here, and does it
    even clear threshold?" is one look."""
    if not rows:
        return niftiutil.empty_fig("no model has a map at this point", height=BAR_MIN_H), BAR_MIN_H
    labels = [_bar_label(m, grouping, label_mode) for m, _v, _s, _n in rows]
    values = [v for _m, v, _s, _n in rows]
    errs = [(s if s is not None else 0.0) for _m, _v, s, _n in rows]
    colors = [BAR_COLOR_CUR if m == current else BAR_COLOR for m, _v, _s, _n in rows]
    n_err = sum(1 for _m, _v, s, _n in rows if s is not None)
    hover = [(f"{m}<br>{v:.4g}" + (f" ± {s:.3g} SEM (n={n})" if s is not None else ""))
             for m, v, s, n in rows]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker=dict(color=colors),
        error_x=dict(type="data", array=errs, visible=n_err > 0,
                     color=MUTED, thickness=1, width=2),
        hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False))
    for x, dash in ((0.0, "solid"), (float(lo), "dot")):
        fig.add_vline(x=x, line=dict(color=LINE if dash == "solid" else ACCENT,
                                     width=1, dash=dash))
    height = max(BAR_MIN_H, 44 + BAR_ROW_H * len(rows))
    fig.update_layout(
        title=dict(text=subtitle, font=dict(size=11, color=INK)),
        margin=dict(l=6, r=10, t=28, b=30), height=height,
        paper_bgcolor=PANEL, plot_bgcolor="#ffffff", font_color=INK, bargap=0.25,
        xaxis=dict(title=dict(text=xtitle, font=dict(size=9, color=MUTED)),
                   tickfont=dict(size=9), gridcolor=LINE, zeroline=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=9),
                   automargin=True, gridcolor=LINE))
    return fig, height


def _bar_store_key(source, datafolder, dataset, modality, roi, glm_model, specie, maptype):
    """Key under which one context's preloaded volumes live. Everything that can
    change *which file* a model resolves to is in it, so a source / ROI / species
    switch gets its own store instead of quietly reusing the wrong brains."""
    return (source, datafolder, dataset, modality, roi, glm_model, specie, maptype)


def _bar_cache_mb():
    total = 0
    for store in _BAR_PRELOAD.values():
        for vol in store.values():
            if vol is not None:
                total += vol[0].nbytes
    return total / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# Crosshair: the voxel a click lands on, and the read-out under the slice
# ---------------------------------------------------------------------------
# The crosshair is stored as a **voxel of the map's own grid** (``[i, j, k]``), not
# as a screen position, so it survives an axis switch, a flip, a colormap change
# and a different map on the same grid. Only the in-plane part is honoured when
# drawing: the out-of-plane coordinate is always replaced by the slice currently
# on screen, so scrolling through slices sweeps the read-out down the same
# in-plane column — which is how you read an intensity profile through a cluster.

def _cross_voxel(cross, shape, axis, slice_idx):
    """The stored crosshair rebased onto the displayed slice, or None."""
    if not cross:
        return None
    try:
        vox = [int(c) for c in cross[:3]]
    except (TypeError, ValueError):
        return None
    if len(vox) != 3:
        return None
    vox[int(axis)] = int(slice_idx)
    if any(v < 0 or v >= s for v, s in zip(vox, shape)):
        return None
    return tuple(vox)


def _cross_field(label, value):
    return html.Span([
        html.Span(f"{label} ", style={"color": MUTED}),
        html.Span(value, style={"color": INK, "fontWeight": "bold",
                                "fontFamily": "Consolas, 'Courier New', monospace"}),
    ], style={"marginRight": "12px", "whiteSpace": "nowrap"})


def _cross_hint(text):
    return [html.Span(f"✛ {text}", style={"color": MUTED, "fontStyle": "italic"})]


def _cross_readout(vox, affine, value, axis, slice_idx, nslices, thr):
    """The read-out line: voxel index, world (mm) coordinate and the map's value
    at the crosshair — the "intensity" field of a normal MR viewer — plus which
    slice of how many is on screen. Sub-threshold voxels are still reported (that
    is the point of clicking one) but flagged, since they are drawn transparent."""
    if vox is None:
        return _cross_hint("click the slice to place the crosshair")
    mm = niftiutil.voxel_to_world(vox, affine)
    axis_name = {0: "X", 1: "Y", 2: "Z"}[int(axis)]
    out = [
        _cross_field("voxel", "(%d, %d, %d)" % vox),
        _cross_field("mm", "(%.1f, %.1f, %.1f)" % mm),
        _cross_field("intensity", "—" if value is None else f"{value:.4g}"),
        _cross_field(f"slice {axis_name}", f"{int(slice_idx)}/{int(nslices) - 1}"),
    ]
    if value is not None and abs(float(value)) < float(thr):
        out.append(html.Span("below threshold — not painted",
                             style={"color": MUTED, "fontStyle": "italic"}))
    return out


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
           title="RSA Model Explorer")
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
      /* the brain view eats the wheel (slice scrubbing), so say so on hover */
      .pl-map-wrap{ cursor:crosshair; position:relative; }

      /* ---- visual pickers (stimulus set, model gallery, grouping) --------
         Each tile is a plain button (see _tile_buttons for why not RadioItems),
         so the selected one is marked by a class rather than by a checked input
         and the styling is ours end to end. */
      /* A picker is temporary furniture -- it folds away as soon as it is
         answered -- so it is capped and scrolls rather than pushing the brain
         view off the screen while six cards are all being set up. */
      .pl-tiles{ display:flex; flex-wrap:wrap; gap:6px; align-content:flex-start;
                 max-height:290px; overflow-y:auto; padding-right:2px; }
      .pl-tile-btn{
        display:flex; align-items:flex-start; gap:4px; cursor:pointer;
        border:1px solid #d5dbe5; border-radius:8px; background:#ffffff;
        padding:4px 6px; line-height:1.15; font:inherit; color:inherit;
        text-align:left; transition:border-color .12s;
      }
      .pl-tile-btn:hover{ border-color:#4472C4; }
      .pl-tile-btn:focus-visible{ outline:2px solid #4472C4; outline-offset:1px; }
      .pl-tile-btn.pl-sel{ border-color:#4472C4; background:#f2f6fd;
                           box-shadow:inset 0 0 0 1px #4472C4; }
      .pl-tile{ display:flex; flex-direction:column; gap:2px; }
      .pl-tile-cap{ font-size:11px; font-weight:bold; color:#222222; }
      .pl-tile-badge{ font-size:9px; color:#667085; letter-spacing:.02em; }

      /* the stimulus grid: one row of icons per agent block, boxed. The same
         drawing serves picker (1) (what is in this set) and picker (3) (which of
         its pairs enter) -- only the boxing differs, which is the point. */
      .pl-ic-stack{ display:flex; flex-direction:column; gap:2px; }
      .pl-ic-box{ border:1px solid #4472C4; border-radius:5px; padding:1px 3px;
                  background:#fbfcfe; }
      .pl-ic-row{ display:flex; align-items:center; gap:2px; white-space:nowrap; }
      .pl-ic-dim{ opacity:.20; filter:grayscale(1); }
      .pl-ic-agent{ font-size:12px; margin-right:2px; }
      .pl-ic-cat{ font-size:11px; display:inline-flex; align-items:flex-start; }
      /* exemplars per cell -- "x4" is what separates a 10-condition set from a
         40-stimulus one, so it rides on the icon rather than in a caption */
      .pl-ic-rep{ font-size:7px; color:#4472C4; font-weight:bold;
                  margin-left:-1px; line-height:1; }
      .pl-ic-link{ font-size:11px; color:#4472C4; text-align:center;
                   line-height:.9; font-weight:bold; }

      /* the model gallery: a model is its own dissimilarity matrix */
      .pl-tiles-stem > .pl-tile-btn{ width:138px; }
      .pl-tile-stem{ flex-direction:row; align-items:center; gap:5px; }
      .pl-thumb{ width:34px; height:34px; flex:0 0 34px; border-radius:3px;
                 image-rendering:pixelated; }
      .pl-thumb-none{ display:flex; align-items:center; justify-content:center;
                      border:1px solid #d5dbe5; color:#667085; font-size:15px; }
      .pl-tile-txt{ display:flex; flex-wrap:wrap; align-items:baseline; gap:3px;
                    min-width:0; }
      .pl-tile-name{ font-size:10.5px; color:#222222; word-break:break-word; }
      .pl-why-ic{ font-size:11px; color:#4472C4; cursor:help; }
      /* which groupings this hypothesis offers -- step (3) will only show these */
      .pl-tile-grps{ font-size:8px; color:#667085; flex-basis:100%;
                     white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }

      /* the collapsed breadcrumb the three pickers fold into once all are set */
      .pl-crumb{ border:1px solid #d5dbe5; border-radius:999px; background:#ffffff;
                 color:#222222; font-size:11px; padding:2px 9px; cursor:pointer; }
      .pl-crumb:hover{ border-color:#4472C4; color:#4472C4; }
      .pl-step-h{ font-size:11px; color:#667085; margin:4px 2px 3px; }

      /* ---- floating switches over the brain view (Google-Maps corner) ----
         Absolutely positioned inside .pl-map-wrap, above the Plotly canvas. The
         colour switch is collapsed to the current palette and opens on hover or
         keyboard focus, which is what keeps two controls off a 230 px slice. */
      .pl-float{ position:absolute; z-index:4; bottom:6px;
                 font-family:'Segoe UI', Arial, sans-serif; }
      .pl-float-maptype{ left:6px; }
      .pl-float-cmap{ right:6px; }
      .pl-seg{ display:flex; gap:0; background:rgba(255,255,255,.92);
               border:1px solid rgba(0,0,0,.25); border-radius:6px;
               overflow:hidden; box-shadow:0 1px 4px rgba(0,0,0,.35); }
      .pl-seg > label{ display:flex; flex-direction:column; align-items:center;
                       justify-content:center; cursor:pointer; padding:2px 7px;
                       min-width:34px; line-height:1.05; color:#333333;
                       border-right:1px solid rgba(0,0,0,.12); }
      .pl-seg > label:last-child{ border-right:none; }
      .pl-panel .pl-seg > label{ color:#333333; }
      .pl-panel .pl-seg > label:hover{ background:#e8eefb; color:#333333; }
      .pl-panel .pl-seg > label.selected,
      .pl-panel .pl-seg > label:has(input:checked){ background:#4472C4; color:#ffffff; }
      /* visually hidden, not removed: a zero-sized input drops out of the tab
         order and the a11y tree, and these are the only handle the switch has */
      .pl-seg input[type="radio"], .pl-cmap-list input[type="radio"]{
        position:absolute; width:1px; height:1px; margin:-1px; padding:0;
        border:0; clip-path:inset(50%); overflow:hidden; white-space:nowrap;
      }
      .pl-panel .pl-seg > label:focus-within{ outline:2px solid #ffffff;
                                              outline-offset:-2px; }
      /* Dash wraps an option's label in its own span, so the flex context has to
         be re-established on our own node -- the label's flex only lays out
         Dash's wrapper + text spans, not what is inside them. */
      .pl-seg-item{ display:flex; flex-direction:column; align-items:center;
                    line-height:1.05; }
      .pl-seg-glyph{ font-size:13px; font-weight:bold; }
      .pl-seg-cap{ font-size:8px; letter-spacing:.02em; }

      .pl-cmap-face{ width:64px; height:22px; border-radius:6px;
                     border:1px solid rgba(0,0,0,.25); cursor:pointer;
                     box-shadow:0 1px 4px rgba(0,0,0,.35);
                     display:flex; align-items:flex-end; justify-content:center; }
      .pl-cmap-face span{ font-size:8px; color:#ffffff; text-shadow:0 0 3px #000000;
                          padding-bottom:1px; }
      .pl-cmap-list{ display:none; position:absolute; right:0; bottom:26px;
                     flex-direction:column; gap:2px; padding:4px;
                     background:rgba(255,255,255,.96); border-radius:6px;
                     border:1px solid rgba(0,0,0,.25);
                     box-shadow:0 2px 8px rgba(0,0,0,.35); max-height:210px;
                     overflow-y:auto; }
      .pl-float-cmap:hover .pl-cmap-list,
      .pl-float-cmap:focus-within .pl-cmap-list{ display:flex; }
      .pl-cmap-list > label{ display:flex; align-items:center; gap:4px;
                             cursor:pointer; border-radius:4px; padding:1px 3px; }
      .pl-panel .pl-cmap-list > label:hover{ background:#e8eefb; }
      .pl-panel .pl-cmap-list > label.selected,
      .pl-panel .pl-cmap-list > label:has(input:checked){ background:#dbe6fa; }
      .pl-panel .pl-cmap-list > label .pl-cmap-nm{ color:#333333; }

      .pl-cmap-item{ display:flex; align-items:center; gap:5px; }
      .pl-cmap-sw{ display:block; flex:0 0 52px; width:52px; height:12px;
                   border-radius:3px; border:1px solid rgba(0,0,0,.2); }
      .pl-cmap-nm{ font-size:9px; color:#333333; width:44px; }
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

      // --- wheel over a brain view = step through slices --------------------
      // The slice slider holds a *fraction* (0..1), but a wheel notch should move
      // exactly one slice, so the step is 1/(n-1) with n read from the hidden
      // ``pl-{i}-slices`` span the render callback keeps up to date. The new value
      // is handed to Dash the same way the reorder drag is (set_props), so the
      // slider, the sync stores and the card render all follow normally.
      //
      // Scrolling is faster than the server round trip, so consecutive notches are
      // accumulated locally: within WHEEL_HOLD ms of the last notch we step from
      // our own last value instead of re-reading the (still lagging) slider, and
      // after that we resync from the DOM so a drag / a 🔗 sync update wins.
      var WHEEL_HOLD = 700, wheelState = {};
      function fracOf(i){
        // '[role=slider]' + aria-valuenow is the one handle selector that holds
        // across dcc.Slider's DOM (Radix thumb in Dash 4, rc-slider before it);
        // the number box beside the slider is the fallback.
        var h=document.querySelector('#pl-'+i+'-frac [role="slider"]');
        var v=h?parseFloat(h.getAttribute('aria-valuenow')):NaN;
        if(isNaN(v)){
          var inp=document.querySelector('#pl-'+i+'-frac input');
          v=inp?parseFloat(inp.value):NaN;
        }
        return isNaN(v)?0.5:v;
      }
      function nSlices(i){
        var el=document.getElementById('pl-'+i+'-slices');
        var n=el?parseInt(el.textContent,10):NaN;
        return (isNaN(n)||n<2)?0:n;
      }
      document.addEventListener('wheel', function(e){
        if(!e.target.closest) return;
        var w=e.target.closest('.pl-map-wrap'); if(!w) return;
        if(e.target.closest('.pl-float')) return;   // the corner switches scroll themselves
        var i=w.getAttribute('data-index'); if(i===null) return;
        var n=nSlices(i); if(!n) return;                 // no map loaded -> let the page scroll
        e.preventDefault();                              // scrub slices, don't scroll the page
        var now=(window.performance?performance.now():Date.now());
        var st=wheelState[i];
        var base=(st && (now-st.t)<WHEEL_HOLD) ? st.v : fracOf(i);
        var dir=(e.deltaY>0)?-1:1;                       // wheel up -> higher slice index
        // Step the *slice index*, then convert back: stepping the fraction instead
        // would let the server's own frac -> index rounding swallow or double a
        // notch whenever the value lands on a half-slice.
        var slice=Math.min(n-1, Math.max(0, Math.round(base*(n-1)) + dir));
        var v=Math.round((slice/(n-1))*1e6)/1e6;
        wheelState[i]={v:v, t:now};
        if(window.dash_clientside && window.dash_clientside.set_props){
          window.dash_clientside.set_props('pl-'+i+'-frac', {value:v});
        }
      }, {passive:false});

      // --- keep the slider under a histogram spanning that histogram's bars ---
      // The slider's min/max are already the figure's x-axis range (the render
      // callback sets them), so the only thing left is pixels: a handle has to sit
      // on the bar holding its value. Two corrections, neither knowable in Python:
      //   * Plotly expands the requested left margin to fit the y-axis title and
      //     ticks (38 comes back as ~51), so the plot area is measured off the
      //     drawn SVG — the background rect against the graph div.
      //   * the slider keeps its handles *inside* its track, so a handle centre
      //     travels only track width minus one handle. The track is therefore run
      //     half a handle past the plot area at each end, which puts the centres
      //     exactly on it. That overhang can exceed the figure's right margin, so
      //     a negative inset is expressed as a margin (padding cannot be < 0) and
      //     the slider overhangs into the card's own padding.
      // Re-measured on any DOM change (a redraw rewrites the SVG) and on resize;
      // writing the inset is itself a mutation, so it is skipped when nothing moved.
      function setInset(el, side, v){
        if(el.dataset['ins'+side]===String(v)) return;
        el.style['padding'+side] = (v>0? v : 0)+'px';
        el.style['margin'+side]  = (v<0? v : 0)+'px';
        el.dataset['ins'+side] = v;
      }
      function alignHistSlider(wrap){
        var box=wrap.parentElement; if(!box) return;
        var gd=box.querySelector('.js-plotly-plot'); if(!gd) return;
        var bg=gd.querySelector('.bglayer rect'); if(!bg) return;
        var g=gd.getBoundingClientRect(), b=bg.getBoundingClientRect();
        if(!g.width || !b.width) return;
        var thumb=wrap.querySelector('[role="slider"]');
        var half=thumb? thumb.getBoundingClientRect().width/2 : 0;
        setInset(wrap, 'Left',  Math.round(b.left-g.left-half));
        setInset(wrap, 'Right', Math.round(g.right-b.right-half));
      }
      // setTimeout rather than requestAnimationFrame: rAF is suspended while the
      // tab is hidden, and a queue raised in that state would latch until the tab
      // came back — a second explorer window would sit misaligned until focused.
      var alignQueued=false;
      function queueAlign(){
        if(alignQueued) return;
        alignQueued=true;
        setTimeout(function(){
          alignQueued=false;
          var ws=document.querySelectorAll('.pl-histslider');
          for(var k=0;k<ws.length;k++){ alignHistSlider(ws[k]); }
        }, 16);
      }
      new MutationObserver(queueAlign).observe(document.body,
        {childList:true, subtree:true, attributes:true,
         attributeFilter:['style','width','height','transform']});
      window.addEventListener('resize', queueAlign);
      queueAlign();
    })();
    // Slice-slider tooltip: the value is a fraction, which is meaningless to read.
    window.dccFunctions = window.dccFunctions || {};
    window.dccFunctions.slicePct = function(v){ return Math.round(v*100) + '%'; };
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
        _labeled("Model bars", html.Div(style={"display": "flex", "gap": "6px", "alignItems": "center"},
                 children=[
            dcc.Dropdown(id="ex-barmode", options=[{"label": l, "value": v} for v, l in BAR_MODES],
                         value=SETTINGS["bar_mode"], clearable=False, style={"width": "200px"}),
            html.Button("🧹 Free", id="ex-barfree", n_clicks=0, title="drop preloaded maps",
                        style=BTN2),
        ])),
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


def _tile_buttons(level, card, options, value):
    """The tiles of one picker, as ordinary buttons.

    ``options`` is the familiar ``[{"label": component, "value": str}, ...]``; the
    chosen value lives in the hidden ``pl-{card}-{level}`` input, so the rest of
    the app still reads the cascade exactly as it always did. See the note above
    the picker callbacks for why this is not a ``dcc.RadioItems``."""
    return [html.Button(id={"pk": level, "card": card, "v": opt["value"]}, n_clicks=0,
                        className=_tile_class(opt["value"], value),
                        children=opt["label"])
            for opt in options]


def _tile_class(value, chosen):
    return "pl-tile-btn" + (" pl-sel" if value == chosen else "")


def _maptype_options():
    """Options for the floating map-type switch: a glyph over a caption, with the
    full name (``Group average`` …) as the tooltip."""
    return [{"label": html.Span(className="pl-seg-item", title=label, children=[
                html.Span(MAPTYPE_ICONS.get(v, ("?", v))[0], className="pl-seg-glyph"),
                html.Span(MAPTYPE_ICONS.get(v, ("?", v))[1], className="pl-seg-cap"),
             ]), "value": v}
            for v, label in MAPTYPES]


def _cmap_options():
    """Options for the floating colour switch: each palette drawn as itself."""
    return [{"label": html.Span(className="pl-cmap-item", children=[
                html.Span(className="pl-cmap-sw", style={"background": _cmap_gradient(c)}),
                html.Span(c, className="pl-cmap-nm"),
             ]), "value": c}
            for c in COLORMAPS]


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
            # Per-card manual refresh. The top bar's 🔄 Update now applies pending
            # changes on *every* enabled card; this one applies them on this card
            # only, which is the difference between re-reading one map and
            # re-reading six when five of them are already what you want.
            html.Button("🔄", id=f"pl-{i}-update", n_clicks=0,
                        title="update this card now (applies its pending changes)",
                        style={"border": f"1px solid {LINE}", "background": "#eef1f6",
                               "color": INK, "cursor": "pointer", "fontSize": "12px",
                               "borderRadius": "6px", "padding": "1px 6px",
                               "marginLeft": "auto"}),
            html.Button("✕", id=f"pl-{i}-remove", n_clicks=0, title="remove this model",
                        style={"border": "none", "background": "transparent", "color": MUTED,
                               "cursor": "pointer", "fontSize": "13px", "padding": "0 2px"}),
        ]),
        # --- picking the model, in three *visual* steps ---------------------
        # The analysis is organised as distance method → (Mahalanobis fold) →
        # model → grouping, which is four dropdowns of jargon. The card asks the
        # same four values as three pictures, in the same order:
        #
        #   ① set — the (distance method, fold) pair, drawn as **the stimulus set
        #     its RDM is built over**: a row of icons per agent block, one cell per
        #     category, each cell carrying its exemplar count. That pair decides
        #     which models exist at all, so picking it re-fills step ②.
        #   ② hypothesis — one tile per model stem of that set, showing the model's
        #     own dissimilarity matrix and an ⓘ carrying its ``why`` note.
        #   ③ grouping — which pairs of that set enter the RSA, drawn as the same
        #     grid boxed: one box (pooled), a box per row (within), two boxes joined
        #     by ⇅ (cross), or a single lit row (Dog only / Human only).
        #
        # They are shown one at a time, in order, and all three collapse into a
        # breadcrumb as soon as the card knows enough to fetch a result — which is
        # the point: a choice is on screen only while it is still open.
        html.Div(id=f"pl-{i}-step1", children=[
            html.Div("① stimulus set", className="pl-step-h"),
            html.Div(id=f"pl-{i}-setbox", className="pl-tiles pl-tiles-set"),
        ]),
        html.Div(id=f"pl-{i}-step2", children=[
            html.Div("② which hypothesis", className="pl-step-h"),
            html.Div(id=f"pl-{i}-stembox", className="pl-tiles pl-tiles-stem"),
        ]),
        html.Div(id=f"pl-{i}-step3", children=[
            html.Div("③ which pairs enter", className="pl-step-h"),
            html.Div(id=f"pl-{i}-groupbox", className="pl-tiles pl-tiles-grp"),
        ]),
        # Collapsed summary of the three steps: click a chip to reopen its step.
        html.Div(id=f"pl-{i}-crumbs", style={"display": "flex", "gap": "5px",
                 "flexWrap": "wrap", "alignItems": "center", "margin": "4px 0 2px"},
                 children=[
            html.Button("", id=f"pl-{i}-crumb-set", n_clicks=0, className="pl-crumb",
                        title="change the stimulus set (distance method / fold)"),
            html.Button("", id=f"pl-{i}-crumb-stem", n_clicks=0, className="pl-crumb",
                        title="change the hypothesis"),
            html.Button("", id=f"pl-{i}-crumb-grp", n_clicks=0, className="pl-crumb",
                        title="change which pairs enter"),
        ]),
        # which picker step is expanded ("set" / "stem" / "grp" / "")
        dcc.Store(id=f"pl-{i}-open", data="set"),
        # (data folder, dataset, model) of the matrix figure this *browser* holds.
        # Memory storage on purpose: a reload empties the graph and this together,
        # so the card knows to re-send it.
        dcc.Store(id=f"pl-{i}-matkey"),
        # --- the model's "why" note from _models.csv ---
        html.Div(id=f"pl-{i}-why", style={"fontSize": "11px", "color": MUTED,
                 "fontStyle": "italic", "minHeight": "14px", "margin": "0 2px 4px"}),
        # The two halves of the ① picker, mirrored into hidden inputs. Every
        # downstream callback (stem menu, matrix, map path, bar scope, sync) still
        # reads ``pl-{i}-dis`` / ``-mahfold`` exactly as it did when they were
        # dropdowns, so replacing the front end touched none of them.
        html.Div(style={"display": "none"}, children=[
            # What the three pickers above actually hold. The cascade's own two
            # values (``-dis`` / ``-mahfold``) are split out of the ① picker by a
            # tiny callback, so every downstream callback — stem menu, matrix, map
            # path, bar scope, sync — still reads the ids it always read.
            dcc.Input(id=f"pl-{i}-set", type="hidden", value="", persistence=True),
            dcc.Input(id=f"pl-{i}-stem", type="hidden", value="", persistence=True),
            dcc.Input(id=f"pl-{i}-grouping", type="hidden", value="", persistence=True),
            dcc.Input(id=f"pl-{i}-dis", type="hidden", value=""),
            dcc.Input(id=f"pl-{i}-mahfold", type="hidden", value=""),
        ]),
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
            dcc.Checklist(id=f"pl-{i}-showbar", options=[{"label": " 📶 model bars", "value": "on"}],
                          value=["on"], persistence=True,
                          labelStyle={"fontSize": "12px"}, style={"paddingBottom": "6px"}),
        ]),
        # --- map controls: only the slice axis lives here now. Map type and
        # colormap moved onto the brain view itself as floating switches (below),
        # where they sit next to what they change instead of above it.
        html.Div(style={"display": "flex", "gap": "6px", "flexWrap": "wrap", "margin": "2px 0"}, children=[
            dcc.Dropdown(id=f"pl-{i}-axis", options=[{"label": l, "value": v} for v, l in AXES],
                         value="2", clearable=False, persistence=True, style={"width": "95px"}),
        ]),
        # Slice position. The step is fine (0.001) rather than one-slice-sized
        # because the wheel handler steps by an exact 1/(n-1) — a coarse step would
        # snap those notches back onto its own grid and skip slices.
        html.Div(style={"display": "flex", "gap": "10px", "alignItems": "center"}, children=[
            html.Span("slice", style={"fontSize": "11px", "color": MUTED}),
            html.Div(dcc.Slider(id=f"pl-{i}-frac", min=0, max=1, step=0.001, value=0.5,
                     marks=None,
                     tooltip={"placement": "bottom", "transform": "slicePct"}),
                     style={"flex": "1"}),
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
            # ``pl-map-wrap`` + data-index is what the wheel handler hooks onto to
            # scrub slices; the hidden span carries the slice count it needs.
            html.Div(className="pl-map-wrap", **{"data-index": str(i)}, children=[
                dcc.Graph(id=f"pl-{i}-map", style={"height": f"{vh}px"}),
                html.Span(id=f"pl-{i}-slices", children="0", style={"display": "none"}),
                # --- floating switches, bottom corners of the slice ----------
                # Which map is on screen, and in which palette, are questions
                # about the picture — so they are asked *on* the picture, the way
                # a map app puts its layer switcher in the corner rather than in a
                # settings panel. Same component ids as the dropdowns they
                # replace, so the render callback and the 🔗 sync mirror are
                # untouched.
                html.Div(className="pl-float pl-float-maptype", children=[
                    dcc.RadioItems(id=f"pl-{i}-maptype", options=_maptype_options(),
                                   value=DEFAULT_MAPTYPE, className="pl-seg",
                                   persistence=True),
                ]),
                html.Div(className="pl-float pl-float-cmap", children=[
                    html.Div(id=f"pl-{i}-cmap-face", className="pl-cmap-face",
                             title="colour palette", tabIndex="0", role="button",
                             style={"background": _cmap_gradient(DEFAULT_CMAP)},
                             children=html.Span(DEFAULT_CMAP)),
                    dcc.RadioItems(id=f"pl-{i}-cmap", options=_cmap_options(),
                                   value=DEFAULT_CMAP, className="pl-cmap-list",
                                   persistence=True),
                ]),
            ], style={"flex": "3 1 0", "minWidth": 0}),
            html.Div(id=f"pl-{i}-histwrap", style={"flex": "2 1 0", "minWidth": 0}, children=[
                dcc.Graph(id=f"pl-{i}-hist", figure=niftiutil.empty_fig(height=vh),
                          style={"height": f"{vh}px"}, config={"displayModeBar": False}),
                # Second view of the *same* [low, high] the range slider above the
                # brain carries — this one spanning the histogram's own value axis,
                # sitting directly under it and inset by the figure's margins, so
                # its handles line up with the cut lines drawn in the bars. Move
                # either slider and the other follows. Because its range is the
                # data's rather than the map type's fixed scale, it is also the
                # finer of the two: the whole track is spent on values that exist.
                html.Div(className="pl-histslider",
                         style={"padding": f"0 {HIST_MARGIN_R}px 0 {HIST_MARGIN_L}px",
                                "marginTop": "-6px"}, children=[
                    dcc.RangeSlider(id=f"pl-{i}-hrange", min=-1, max=1, step=0.02,
                                    value=[0, 1], allowCross=False, marks=None,
                                    tooltip={"placement": "bottom", "always_visible": False}),
                ]),
                dcc.Store(id=f"pl-{i}-hecho"),   # last [low, high] pushed into it
            ]),
        ]),
        # --- crosshair read-out (voxel / mm / intensity at the clicked voxel) ---
        html.Div(id=f"pl-{i}-info", style=CROSS_PANEL_STYLE),
        dcc.Store(id=f"pl-{i}-cross"),   # crosshair voxel [i, j, k] on the map grid
        # --- model dissimilarity matrix + model-comparison bars, side by side ---
        # The matrix says what the selected model predicts; the bars say what every
        # model measured at the crosshair. Reading them together is the point of
        # putting them in one row, and either can be toggled off to give the other
        # the full card width.
        html.Div(style={"display": "flex", "gap": "8px", "alignItems": "flex-start"}, children=[
            html.Div(id=f"pl-{i}-matrixwrap", style=MATRIX_WRAP_SHOW, children=[
                html.Div("Model matrix (builder view)", style={"fontSize": "11px", "color": MUTED,
                         "margin": "8px 0 2px"}),
                html.Div(dcc.Graph(id=f"pl-{i}-matrix", figure=niftiutil.empty_fig(height=200),
                         config={"displayModeBar": False}),
                         style={"maxHeight": "520px", "overflowY": "auto"}),
            ]),
            html.Div(id=f"pl-{i}-barwrap", style=BAR_WRAP_SHOW, children=[
                html.Div("All models at the crosshair", style={"fontSize": "11px", "color": MUTED,
                         "margin": "8px 0 2px"}),
                html.Div(style={"display": "flex", "gap": "6px", "alignItems": "center",
                                "marginBottom": "3px"}, children=[
                    dcc.Dropdown(id=f"pl-{i}-bar-scope",
                                 options=[{"label": l, "value": v} for v, l in BAR_SCOPES],
                                 value="fold-grouping", clearable=False, persistence=True,
                                 style={"flex": "1 1 130px", "minWidth": "120px"}),
                    html.Button("📶 Compare", id=f"pl-{i}-bar-btn", n_clicks=0,
                                title="sample every model in scope at the crosshair", style=BTN),
                ]),
                html.Div(id=f"pl-{i}-barnote", style={"fontSize": "11px", "color": MUTED,
                         "minHeight": "15px", "margin": "0 2px 2px"}),
                html.Div(dcc.Graph(id=f"pl-{i}-bar",
                         figure=niftiutil.empty_fig("click ✛ then 📶 Compare", height=BAR_MIN_H),
                         style={"height": f"{BAR_MIN_H}px"}, config={"displayModeBar": False}),
                         style={"maxHeight": "520px", "overflowY": "auto"}),
            ]),
        ]),
    ])


app.layout = html.Div(style={"backgroundColor": BG, "color": INK, "minHeight": "100vh",
                      "padding": "10px 14px", "fontFamily": "'Segoe UI', Arial, sans-serif"}, children=[
    html.H2("RSA Model Explorer", style={"textAlign": "center", "margin": "4px 0 8px"}),
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
    dcc.Store(id="ex-grouped", data={"dis_methods": [], "by_dis": {}}),
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
    with _Timer() as t:
        rois = resolve_roi_options(source, datafolder, dataset, modality, glm_model)
    log("scan", f"ROI/mask options ({source}) - {len(rois)} found  {t.ms:.0f} ms")
    return ([{"label": r, "value": r} for r in rois], (rois[0] if rois else None),
            describe_source_mode(source, datafolder, dataset, glm_model))


@app.callback(Output("ex-grouped", "data"), Output("ex-models-title", "children"),
              Input("ex-datafolder", "value"), Input("ex-dataset", "value"), Input("ex-dataver", "data"))
def cb_build_index(datafolder, dataset, _ver):
    """Rebuild the distance-method → fold → model → grouping menu whenever the data
    folder / dataset changes or results are reloaded, re-reading that project's
    ``_models.csv`` fresh each time. The manifest path it actually resolved to is
    part of the status line, so switching dataset shows plainly which file the
    menus now come from."""
    _MANIFEST_CACHE.pop((datafolder, dataset), None)   # legacy battery-manifest cache
    mm.clear_cache()                                   # re-read _models.csv from disk
    with _Timer() as t:
        grouped = build_index(datafolder, dataset)
    dis = grouped.get("dis_methods", [])
    log("index", f"model menu rebuilt for {dataset} - {len(dis)} distance method(s)  "
                 f"{t.ms:.0f} ms")
    nstems = sum(len(_fold_data(grouped, d, FOLD_ANY).get("stems", [])) for d in dis)
    path = manifest_file(datafolder, dataset)
    if dis == [FALLBACK_DIS]:
        return grouped, (f"no _models.csv under {dataset or '?'}/rsa_models — scanned "
                         f"{nstems} model families from disk; add a card, pick a model "
                         "+ grouping")
    folds = sum(len(_dis_data(grouped, d).get("folds", [])) for d in dis)
    return grouped, (f"{len(dis)} distance method(s), {folds} fold(s), {nstems} model "
                     f"families — add a card and pick a grouping first · {path}")


@app.callback(Output("ex-dataver", "data"), Input("ex-reload", "n_clicks"),
              State("ex-dataver", "data"), prevent_initial_call=True)
def cb_reload(_n, ver):
    """Drop cached maps/masks/result-sets so freshly-synced results are re-read."""
    held = sum(_mb(v[0]) for v in _VOL_CACHE.values() if v is not None)
    _PATH_CACHE.clear()
    _VOL_CACHE.clear()
    _LISTDIR_CACHE.clear()
    _RESULT_SETS_CACHE.clear()
    _MASK_CACHE.clear()
    _MASK_ON_GRID.clear()
    _MASK_VALS.clear()
    _HEATMAP_CACHE.clear()
    _CSVMAP_CACHE.clear()      # re-list rsa_models: a model may have been added
    _THUMB_CACHE.clear()
    _TAXONOMY_CACHE.clear()
    _TAXONOMY_LOGGED.clear()
    _MEANLOG_CACHE.clear()
    _BAR_PRELOAD.clear()
    log("reload", f"caches dropped ({held:.0f} MB of volumes) — results will be re-read")
    return (ver or 0) + 1


@app.callback(Output("ex-models-title", "children", allow_duplicate=True),
              Input("ex-barmode", "value"), Input("ex-barfree", "n_clicks"),
              prevent_initial_call=True)
def cb_bar_memory(mode, _n):
    """Drop the model-bar preload store — on demand (🧹 Free) and whenever the read
    mode changes, since switching to "on request" is a request to stop holding
    brains in RAM and switching to "preloaded" should start from a clean, correctly
    keyed store."""
    freed = _bar_cache_mb()
    _BAR_PRELOAD.clear()
    what = "preloaded" if mode == "preload" else "read on request"
    tail = f" — {freed:.0f} MB freed" if freed >= 0.5 else ""
    if ctx.triggered_id == "ex-barfree":
        return f"Model-bar maps dropped{tail or ' — nothing was held'}."
    return f"Model bars: maps {what}{tail}."


# Per-card model choice, asked as **three visual steps** and mirrored back onto
# the cascade the rest of the app is written against.
#
#   ① ``pl-{i}-set`` — the (distance method, Mahalanobis fold) pair, one tile per
#      **stimulus set**. Its value carries both, and a tiny callback splits it
#      into the hidden ``-dis`` / ``-mahfold`` inputs, so every downstream callback
#      keeps reading exactly what it always read. Picking a set re-fills ②: the
#      pair decides which models exist at all, because a model built for one
#      pairwise-similarity method is not comparable with another's.
#   ② ``pl-{i}-stem`` — the hypotheses of that set, each shown as its own
#      dissimilarity matrix plus an ⓘ holding its ``why`` note.
#   ③ ``pl-{i}-grouping`` — the groupings *that stem declares*, drawn as the set's
#      grid boxed to show which pairs enter.
#
# Each level offers only what the level above allows, and each keeps a still-valid
# persisted value (so a card comes back exactly as left) while clearing one that no
# longer fits — which is also what re-opens the step that has become unanswered.

# Each picker is three callbacks, and the split is what keeps a rebuild cheap:
#
#   **build**  fills the box with tiles whenever the level above it moves, and
#              clears the level's own value when the old choice is no longer on
#              offer -- which is also what re-opens that step.
#   **click**  writes the clicked tile's value into the hidden input the cascade
#              reads. Rebuilding resets every ``n_clicks`` to 0, so it only acts on
#              a trigger that is a tile *and* actually carries a click.
#   **paint**  marks the chosen tile. Selection is a class, not a rebuild: a
#              re-render of eighteen thumbnails per click would put ~30 kB of
#              data-URI on the wire to move one outline.
#
# They are buttons rather than a ``dcc.RadioItems`` because Dash 4 does not
# re-render option labels that are components -- see ``_tile_buttons``.

def _register_panel_set(i):
    @app.callback(Output(f"pl-{i}-setbox", "children"),
                  Output(f"pl-{i}-set", "value", allow_duplicate=True),
                  Input("ex-grouped", "data"),
                  State("ex-datafolder", "value"), State("ex-dataset", "value"),
                  State(f"pl-{i}-set", "value"), prevent_initial_call=True)
    def _cb(grouped, datafolder, dataset, cur):
        opts, keys = [], []
        for dis, fold, real, n_models in _set_entries(grouped):
            agents, cats, reps = _set_taxonomy(datafolder, dataset, grouped, dis, fold)
            key = _set_key(dis, fold)
            keys.append(key)
            opts.append({"label": _set_tile(dis, fold, real, agents, cats, reps, n_models),
                         "value": key})
        # A single set is not a choice -- select it and let the card open at (2).
        value = cur if cur in keys else (keys[0] if len(keys) == 1 else "")
        return _tile_buttons("set", i, opts, value), value
    return _cb


def _register_panel_set_mirror(i):
    # One value in, the cascade's two out. Hidden dcc.Inputs rather than a Store on
    # purpose: a Store carries its state in ``data``, and switching prop would have
    # meant touching every callback that reads these.
    @app.callback(Output(f"pl-{i}-dis", "value"), Output(f"pl-{i}-mahfold", "value"),
                  Input(f"pl-{i}-set", "value"))
    def _cb(value):
        dis, fold = _split_set(value)
        return (dis or ""), (fold or "")
    return _cb


def _register_panel_stem(i):
    @app.callback(Output(f"pl-{i}-stembox", "children"),
                  Output(f"pl-{i}-stem", "value", allow_duplicate=True),
                  Input(f"pl-{i}-set", "value"), Input("ex-grouped", "data"),
                  State("ex-datafolder", "value"), State("ex-dataset", "value"),
                  State(f"pl-{i}-stem", "value"), prevent_initial_call=True)
    def _cb(value, grouped, datafolder, dataset, cur):
        dis, fold = _split_set(value)
        fd = _fold_data(grouped, dis, fold)
        index, why = fd.get("index", {}), fd.get("why", {})
        stems = [st for st in fd.get("stems", []) if index.get(st)]
        opts = []
        for st in stems:
            variants = index[st]                       # {grouping: model}, canonical order
            groupings = [g for g in GROUPINGS if g in variants]
            rep = variants[groupings[0]] if groupings else next(iter(variants.values()))
            opts.append({"label": _stem_tile(datafolder, dataset, rep, st,
                                             why.get(st, ""), groupings),
                         "value": st})
        chosen = cur if cur in stems else ""
        return _tile_buttons("stem", i, opts, chosen), chosen
    return _cb


def _register_panel_grouping(i):
    @app.callback(Output(f"pl-{i}-groupbox", "children"),
                  Output(f"pl-{i}-grouping", "value", allow_duplicate=True),
                  Input(f"pl-{i}-stem", "value"), Input(f"pl-{i}-set", "value"),
                  Input("ex-grouped", "data"),
                  State("ex-datafolder", "value"), State("ex-dataset", "value"),
                  State(f"pl-{i}-grouping", "value"), prevent_initial_call=True)
    def _cb(stem, value, grouped, datafolder, dataset, cur):
        dis, fold = _split_set(value)
        variants = _fold_data(grouped, dis, fold).get("index", {}).get(stem or "", {})
        groups = [g for g in GROUPINGS if g in variants]
        opts = []
        if groups:
            agents, cats, reps = _set_taxonomy(datafolder, dataset, grouped, dis, fold)
            opts = [{"label": _grouping_tile(g, agents, cats, reps), "value": g}
                    for g in groups]
        # Keep a still-valid choice; otherwise leave it unanswered so step (3) opens
        # rather than silently picking a different comparison than the one on screen.
        chosen = cur if cur in groups else ""
        return _tile_buttons("grouping", i, opts, chosen), chosen
    return _cb


def _register_panel_pick(i, level):
    """click -> value, and value -> which tile is painted, for one picker."""
    @app.callback(Output(f"pl-{i}-{level}", "value", allow_duplicate=True),
                  Input({"pk": level, "card": i, "v": ALL}, "n_clicks"),
                  prevent_initial_call=True)
    def _cb_click(clicks):
        trig = ctx.triggered_id
        if not isinstance(trig, dict) or not any(clicks or []):
            return no_update          # a rebuild zeroed every n_clicks, not a click
        return trig.get("v", "")

    @app.callback(Output({"pk": level, "card": i, "v": ALL}, "className"),
                  Input(f"pl-{i}-{level}", "value"),
                  State({"pk": level, "card": i, "v": ALL}, "id"))
    def _cb_paint(value, ids):
        return [_tile_class(d.get("v"), value) for d in (ids or [])]
    return _cb_click, _cb_paint


def _register_panel_why(i):
    @app.callback(Output(f"pl-{i}-why", "children"),
                  Input(f"pl-{i}-dis", "value"), Input(f"pl-{i}-mahfold", "value"),
                  Input(f"pl-{i}-stem", "value"), Input("ex-grouped", "data"))
    def _cb(dis, fold, stem, grouped):
        return _fold_data(grouped, dis, fold).get("why", {}).get(stem or "", "")
    return _cb


# --- progressive disclosure: show a step only while it is still open --------
# "Add a model" should not open six controls at once. A fresh card shows step ①
# alone, answering it swaps to ②, then ③, and answering that collapses all three
# into the breadcrumb while the card gets on with drawing the map. Clicking a chip
# reopens its step, and ``pl-{i}-open`` is the single place that state lives.

_STEPS = ("set", "stem", "grp")


def _register_panel_open(i):
    @app.callback(Output(f"pl-{i}-open", "data"),
                  Input(f"pl-{i}-set", "value"), Input(f"pl-{i}-stem", "value"),
                  Input(f"pl-{i}-grouping", "value"),
                  Input(f"pl-{i}-crumb-set", "n_clicks"),
                  Input(f"pl-{i}-crumb-stem", "n_clicks"),
                  Input(f"pl-{i}-crumb-grp", "n_clicks"))
    def _cb(value, stem, grouping, _ns, _nh, _ng):
        trig = ctx.triggered_id
        for step in _STEPS:
            if trig == f"pl-{i}-crumb-{step}":
                return step
        # Otherwise: whatever is still missing is what is on screen, and nothing
        # missing means nothing on screen. Stated as a function of the three values
        # (rather than of which one just moved) it is the same answer on a fresh
        # card, on a persisted reload, and after a change further up that left the
        # levels below it valid — in that case nothing re-opens, because there is
        # nothing left to ask. A change that *invalidates* a level clears it in that
        # level's own callback, and this re-runs on that and opens it.
        if not value:
            return "set"
        if not stem:
            return "stem"
        if not grouping:
            return "grp"
        return ""
    return _cb


def _register_panel_steps(i):
    @app.callback(Output(f"pl-{i}-step1", "style"), Output(f"pl-{i}-step2", "style"),
                  Output(f"pl-{i}-step3", "style"), Output(f"pl-{i}-crumbs", "style"),
                  Output(f"pl-{i}-crumb-set", "children"),
                  Output(f"pl-{i}-crumb-stem", "children"),
                  Output(f"pl-{i}-crumb-grp", "children"),
                  Output(f"pl-{i}-crumb-grp", "style"),
                  Input(f"pl-{i}-open", "data"), Input(f"pl-{i}-set", "value"),
                  Input(f"pl-{i}-stem", "value"), Input(f"pl-{i}-grouping", "value"))
    def _cb(open_step, value, stem, grouping):
        dis, fold = _split_set(value)
        show, crumbs = {}, {"display": "flex", "gap": "5px", "flexWrap": "wrap",
                            "alignItems": "center", "margin": "4px 0 2px"}
        # Exactly the open step is on screen. ``pl-{i}-open`` already resolves to
        # the first unanswered step when nothing was clicked, so this does not need
        # to re-derive that -- and must not, or re-opening ① from the breadcrumb
        # would leave ② showing underneath it.
        s1 = show if open_step == "set" else WRAP_HIDE
        s2 = show if open_step == "stem" else WRAP_HIDE
        s3 = show if open_step == "grp" else WRAP_HIDE
        c_set = ("◈ " + _set_summary(dis, fold)) if value else "◈ pick a stimulus set"
        c_stem = ("▦ " + stem) if stem else "▦ pick a hypothesis"
        c_grp = ("◧ " + _grouping_summary(grouping)) if grouping else "◧ pick a grouping"
        # The grouping chip only means something once a hypothesis names the
        # groupings on offer, so it stays off the row until then.
        return (s1, s2, s3, (crumbs if value else WRAP_HIDE), c_set, c_stem, c_grp,
                {} if stem else WRAP_HIDE)
    return _cb


def _register_panel_cmap_face(i):
    # The collapsed face of the floating colour switch *is* the current palette.
    @app.callback(Output(f"pl-{i}-cmap-face", "style"),
                  Output(f"pl-{i}-cmap-face", "children"),
                  Input(f"pl-{i}-cmap", "value"))
    def _cb(cmap):
        name = cmap or DEFAULT_CMAP
        return {"background": _cmap_gradient(name)}, html.Span(name)
    return _cb


for _i in range(MAX_MODELS):
    _register_panel_set(_i)
    _register_panel_set_mirror(_i)
    _register_panel_stem(_i)
    _register_panel_grouping(_i)
    for _level in ("set", "stem", "grouping"):
        _register_panel_pick(_i, _level)
    _register_panel_why(_i)
    _register_panel_open(_i)
    _register_panel_steps(_i)
    _register_panel_cmap_face(_i)


# ---------------------------------------------------------------------------
# Callbacks — add / remove model cards
# ---------------------------------------------------------------------------
# Cards are pre-registered (``MAX_MODELS`` slots) and shown/hidden via their ``on``
# switch. "➕ Add model" turns on the next off slot; each card's ✕ button (and its
# ``on`` checkbox) turns it off. Grouped Output/State lists let one callback fan
# out over every slot.

@app.callback(*[Output(f"pl-{i}-enable", "value", allow_duplicate=True) for i in range(MAX_MODELS)],
              Output("ex-models-title", "children", allow_duplicate=True),
              Input("ex-addpanel", "n_clicks"),
              [State(f"pl-{i}-enable", "value") for i in range(MAX_MODELS)],
              prevent_initial_call=True)
def cb_add_panel(_n, *enables):
    """Turn on the first currently-off card; no-op when all are already on. (Dash
    passes the State list as separate positional args, hence ``*enables``.)

    The outputs are spread (``*[...]``) rather than passed as one list because a
    grouped list *plus* a further positional Output flattens to MAX_MODELS + 1
    separate outputs — so the return has to be that many flat values too, not
    ``(list, message)``. Getting that wrong raises SchemaLengthValidationError
    inside Dash and the click 500s silently."""
    out = [no_update] * MAX_MODELS
    for i, e in enumerate(enables):
        if "on" not in (e or []):
            out[i] = ["on"]
            log("ui", f"card {i + 1} added")
            return (*out, f"Added model {i + 1}.")
    log("ui", f"add ignored - all {MAX_MODELS} card slots already in use")
    return (*out, f"All {MAX_MODELS} model slots are already in use.")


def _register_panel_remove(i):
    @app.callback(Output(f"pl-{i}-enable", "value", allow_duplicate=True),
                  Output("ex-models-title", "children", allow_duplicate=True),
                  Input(f"pl-{i}-remove", "n_clicks"), prevent_initial_call=True)
    def _cb_remove(_n):
        log("ui", f"card {i + 1} removed")
        return [], f"Removed model {i + 1}."      # clear the "on" value -> card hides
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
              Input("ex-layout", "data"), Input("ex-barmode", "value"),
              prevent_initial_call=True)
def cb_save_settings(source, datafolder, glm_model, dataset, modality, view_h, layout, barmode):
    s = {
        "source_mode": source or "drive",
        "datafolder": datafolder or None,
        "glm_model": (glm_model or DEFAULT_GLM_MODEL).strip(),
        "dataset": (dataset or DEFAULT_DATASET).strip(),
        "modality": modality or "RSA",
        "view_height": _int(view_h, DEFAULT_SETTINGS["view_height"]),
        "bar_mode": barmode if barmode in dict(BAR_MODES) else DEFAULT_SETTINGS["bar_mode"],
        "layout": _clean_layout(layout),   # model-card order + gap
    }
    save_settings(s)
    log("settings", f"saved - {s['source_mode']} / {s['dataset']} / {s['datafolder']}", level=2)
    return s


# ---------------------------------------------------------------------------
# Callbacks — per-card threshold range (depends on the map type / "measure")
# ---------------------------------------------------------------------------
# The range slider's two handles mean different things for different map types:
# a z-threshold for the z-map / cluster-corrected map, vs. an average-similarity
# threshold — Kendall's tau, this pipeline's default RSA method, ranging -1..1 —
# for the group-average map. The **low** handle filters voxels below it out
# (rendered transparent); the **high** handle caps the color scale — voxels
# at/above it are painted the palette's top color.
#
# For **Z-map / Cluster-corrected** the default window is not a fixed pair but
# **3.1 → the map's own maximum**: the conventional cluster-forming threshold at
# one end, the brightest voxel actually present at the other, so the palette
# spends its whole range on the data instead of on headroom that no voxel reaches.
# That means the default is recomputed whenever the *image* changes (map type,
# model, grouping, species, source, reload) — and the slider's own max grows with
# it, since a z of 12 has to be reachable by the handle. For **Group average** the
# fixed -1..1 scale stands and a still-valid current [lo, hi] survives a switch.
ZT_RANGE_Z = {"min": 0, "max": 8, "step": 0.1, "marks": {0: "0", 3.1: "3.1", 8: "8"},
              "default": [3.1, 8], "label": "z"}
ZT_RANGE_MEAN = {"min": -1, "max": 1, "step": 0.02, "marks": {-1: "-1", 0: "0", 1: "1"},
                 "default": [0.0, 1.0], "label": "Kendall τ"}


def _map_abs_max(source, datafolder, dataset, modality, roi, glm_model,
                 specie, model, maptype, zt):
    """Largest |value| in the map a card is about to draw, or None when it cannot
    be loaded / is empty. Goes through ``_load_map``'s cache, so once the card has
    rendered that map this costs nothing."""
    loaded = _load_map(source, datafolder, dataset, modality, roi, glm_model,
                       specie, model, maptype, zt)
    if loaded is None:
        return None
    finite = loaded[0][np.isfinite(loaded[0])]
    if finite.size == 0:
        return None
    top = float(np.max(np.abs(finite)))
    return top if top > 0 else None


def _register_panel_zt_range(i):
    @app.callback(
        Output(f"pl-{i}-range", "min"), Output(f"pl-{i}-range", "max"),
        Output(f"pl-{i}-range", "step"), Output(f"pl-{i}-range", "marks"),
        Output(f"pl-{i}-range", "value", allow_duplicate=True),
        Output(f"pl-{i}-zt-label", "children"),
        Input(f"pl-{i}-maptype", "value"), Input(f"pl-{i}-dis", "value"),
        Input(f"pl-{i}-mahfold", "value"),
        Input(f"pl-{i}-stem", "value"), Input(f"pl-{i}-grouping", "value"),
        Input(f"pl-{i}-maps", "value"), Input("ex-roi", "value"),
        Input("ex-source-mode", "value"), Input("ex-glm-model", "value"),
        Input("ex-grouped", "data"), Input("ex-dataver", "data"),
        Input(f"pl-{i}-update", "n_clicks"), Input("ex-update-trigger", "data"),
        Input(f"pl-{i}-enable", "value"), State("ex-autoupdate", "value"),
        State(f"pl-{i}-range", "value"), State("ex-datafolder", "value"),
        State("ex-dataset", "value"), State("ex-modality", "value"),
        prevent_initial_call="initial_duplicate")
    def _cb(maptype, dis, mahfold, stem, grouping, maps, roi, source, glm_model, grouped,
            _ver, _card_update, _update_trig, enable, autoupdate, cur, datafolder, dataset,
            modality):
        # A switched-off card is not on screen, and this callback *reads the map* to
        # find its peak — so on a page load with five cards off it was five network
        # reads for sliders nobody can see.
        if "on" not in (enable or []):
            return (no_update,) * 6
        # Reading the map is also why Auto-update off has to gate it exactly like
        # the render callback does — otherwise picking a model in "manual" mode
        # would still pay the network read that manual mode exists to defer.
        trig = ctx.triggered_id
        gated = {f"pl-{i}-dis", f"pl-{i}-mahfold", f"pl-{i}-stem",
                 f"pl-{i}-grouping", f"pl-{i}-maps", f"pl-{i}-maptype"}
        if trig in gated and "auto" not in (autoupdate or []):
            return (no_update,) * 6
        if maptype == "mean":
            r = ZT_RANGE_MEAN
            try:
                lo, hi = float(cur[0]), float(cur[1])
                in_range = r["min"] <= lo <= hi <= r["max"]
            except (TypeError, ValueError, IndexError):
                in_range = False
            val = [lo, hi] if in_range else list(r["default"])
            return r["min"], r["max"], r["step"], r["marks"], val, r["label"]

        r = ZT_RANGE_Z
        lo = float(r["default"][0])                 # 3.1 — cluster-forming threshold
        model = _resolve_model(grouped, dis, mahfold, stem, grouping)
        specie = maps if maps in ("D", "H") else "D"
        top = (_map_abs_max(source, datafolder, dataset, modality, roi, glm_model,
                            specie, model, maptype, lo) if model else None)
        hi = round(float(top), 2) if top else float(r["max"])
        if hi <= lo:                                # nothing above 3.1 in this map
            hi = lo + float(r["step"])
        smax = max(float(r["max"]), float(np.ceil(hi * 10) / 10))
        marks = {r["min"]: f"{r['min']:g}", lo: f"{lo:g}", smax: f"{smax:g}"}
        return r["min"], smax, r["step"], marks, [lo, hi], r["label"]
    return _cb


for _i in range(MAX_MODELS):
    _register_panel_zt_range(_i)


# ---------------------------------------------------------------------------
# Callbacks — the histogram's own [low, high] slider, mirrored with the card's
# ---------------------------------------------------------------------------
# The slider under the histogram and the range slider above the brain are two
# views of one pair of numbers, so moving either moves the other. They are *not*
# the same scale: the card's slider spans the map type's fixed range (0..8+ for z,
# -1..1 for a group average) because a value has to survive being mirrored onto a
# synced card whose data tops out elsewhere, while the histogram's spans the values
# this map actually holds — which is what lines its handles up with the bars.
#
# The two directions are deliberately owned by different callbacks:
#   * **card slider -> histogram slider** rides along with the render callback
#     below, which is the one that knows the figure the slider has to match. It
#     travels with the redraw, so a card gated by Auto-update keeps its slider on
#     the histogram still on screen instead of on a map it isn't showing yet.
#   * **histogram slider -> card slider** is this callback.
# A value pushed the first way can be outside the histogram's range (a max cap of
# 1.0 over data that peaks at 0.35), so it is clamped and the handle pins at the
# edge rather than rendering off the track. That clamped push comes straight back
# here as a change of the histogram slider, and propagating it would quietly
# rewrite the value the user set — so the pushed pair is remembered in
# ``pl-{i}-hecho`` and ignored when it returns.

def _pair(val):
    """[float, float] from a slider value, or None if it isn't one."""
    try:
        return [float(val[0]), float(val[1])]
    except (TypeError, ValueError, IndexError):
        return None


def _pair_clamped(val, lo, hi):
    """``val`` moved inside [lo, hi], keeping low <= high."""
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return val
    a = min(max(val[0], lo), hi)
    b = min(max(val[1], lo), hi)
    return [a, max(a, b)]


def _pair_eq(a, b):
    return a is not None and b is not None and abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9


def _register_panel_hrange(i):
    @app.callback(
        Output(f"pl-{i}-range", "value", allow_duplicate=True),
        Input(f"pl-{i}-hrange", "value"),
        State(f"pl-{i}-range", "value"),
        State(f"pl-{i}-range", "min"), State(f"pl-{i}-range", "max"),
        State(f"pl-{i}-hecho", "data"), prevent_initial_call=True)
    def _cb(hrng, rng, rmin, rmax, echo):
        hrng, rng = _pair(hrng), _pair(rng)
        if hrng is None or rng is None:
            return no_update
        if _pair_eq(hrng, _pair(echo)):           # our own push coming back
            return no_update
        want = _pair_clamped(hrng, rmin, rmax)    # the card's slider can't show more
        return no_update if _pair_eq(want, rng) else want
    return _cb


for _i in range(MAX_MODELS):
    _register_panel_hrange(_i)


# ---------------------------------------------------------------------------
# Callbacks — card rendering (one per card)
# ---------------------------------------------------------------------------

def _hist_axis_range(values, lo):
    """[left, right] the histogram's value axis spans, or None for no data.

    The low handle is always inside it, so its cut line stays visible even when it
    sits past the data (e.g. z-threshold 3.1 on an all-sub-threshold map). This is
    a helper rather than a few lines inside ``_hist_fig`` because the slider drawn
    *under* the histogram takes its min/max from exactly the same numbers — that
    is what makes a handle sit on the value its cut line marks."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    lo = float(lo)
    left, right = min(float(np.min(v)), lo), max(float(np.max(v)), lo)
    if right <= left:
        right = left + 1e-6
    return [left, right]


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
    bounds = _hist_axis_range(v, lo)
    if bounds is None:
        return niftiutil.empty_fig("no voxels in mask", height=height)
    lo, hi = float(lo), float(hi)
    left, right = bounds
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
        margin=dict(l=HIST_MARGIN_L, r=HIST_MARGIN_R, t=26, b=30), height=height, bargap=0,
        paper_bgcolor=PANEL, plot_bgcolor="#ffffff", font_color=INK,
        xaxis=dict(title=dict(text=xtitle, font=dict(size=9, color=MUTED)),
                   tickfont=dict(size=9), gridcolor=LINE,
                   zeroline=True, zerolinecolor=LINE, range=[left, right]),
        yaxis=dict(type="log", title=dict(text="voxels (log)", font=dict(size=9, color=MUTED)),
                   tickfont=dict(size=9), gridcolor=LINE))
    return fig


def _card_species_fig(source, datafolder, dataset, modality, roi, glm_model,
                      specie, model, maptype, axis, frac, zt, view_height,
                      colorscale=None, vmax_override=None, want_hist=True, cross=None):
    """(slice figure, histogram figure | None, [left, right] of the histogram's
    value axis | None, n supra-threshold in mask, n in mask, crosshair read-out,
    n slices along the displayed axis).

    The histogram is computed from the *same* loaded volume as the slice, so the
    two always describe one map; it is skipped entirely (None) when the card has
    its histogram hidden. Its axis bounds ride along because the slider drawn under
    it is given exactly that range. The slice count is handed back because the
    wheel-scroll handler needs it to move the slider by exactly one slice per
    notch."""
    map_path = _map_path_for(source, datafolder, dataset, modality, roi, glm_model,
                             specie, model, maptype, zt) if model else None
    loaded = _load_volume(map_path)
    label = {"D": "Dog", "H": "Human"}[specie]
    if loaded is None:
        empty = niftiutil.empty_fig(f"{label}: no {maptype} map", height=view_height, dark=True)
        return (empty, (niftiutil.empty_fig("no map", height=view_height) if want_hist else None),
                None, 0, 0, _cross_hint("no map loaded"), 0)
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
    # Crosshair: stored in voxels, rebased onto this slice, then converted into the
    # rendered picture's own (col, row) frame — the same frame a click reports back.
    orient = niftiutil.slice_orientation(aff, ax)
    vox = _cross_voxel(cross, data.shape, ax, idx)
    cross_rc, value = None, None
    if vox is not None:
        r, c = niftiutil.voxel_to_slice_rc(data.shape, ax, vox, orient)
        cross_rc = (c, r)
        v = float(data[vox])
        value = v if np.isfinite(v) else None
    info = _cross_readout(vox, aff, value, ax, idx, data.shape[ax], thr)
    # ``aff`` is the map's affine; the atlas has been resampled onto that same
    # grid, so it orients both volumes and drives the L/R · A/P · S/I labels.
    fig = niftiutil.make_slice_fig(atlas, data, ax, idx, opacity=0.8, z_threshold=thr,
                                   vmin=vmin, vmax=vmax, title=f"{label} · {model}",
                                   height=view_height, colorscale=colorscale, affine=aff,
                                   show_crosshair=cross_rc is not None, cross=cross_rc)
    # Counts (and the histogram) are restricted to the search mask — the voxels the
    # searchlight actually visited — so "how many survive this threshold" is out of
    # a meaningful denominator instead of the whole bounding box.
    vals, xtitle = _mask_values(data, datafolder, dataset, specie, roi, aff,
                                map_key=map_path)
    supra = int(np.sum(vals >= thr))
    hist = _hist_fig(vals, thr, vmax, colorscale, view_height, xtitle) if want_hist else None
    hbounds = _hist_axis_range(vals, thr) if want_hist else None
    return fig, hist, hbounds, supra, int(vals.size), info, int(data.shape[ax])


def _hslider_props(hbounds, val):
    """(min, max, step, value, echo) for the slider under the histogram: the
    figure's own value axis, with the card's [low, high] clamped into it. The step
    is a four-hundredth of the range rather than the card slider's fixed one — the
    span here can be a few hundredths of a tau, and a 0.02 step would quantise it
    to three positions. ``no_update``s when there is no histogram to align to
    (hidden, or nothing loaded), leaving the slider as the last map left it."""
    if not hbounds:
        return (no_update,) * 5
    lo, hi = float(hbounds[0]), float(hbounds[1])
    push = _pair_clamped(val, lo, hi)
    return lo, hi, max((hi - lo) / 400.0, 1e-9), push, push


def _register_panel(i):
    # Card *content* — title/status, single-species results map + note, model
    # matrix. The card block's own style (order, width, show/hide, edit outline) is
    # owned by the separate style callback below so the edit-mode arrangement is
    # never overwritten here. It also owns the histogram slider's *scale* (the
    # figure it just drew is what that slider has to line up with); the value
    # mirroring itself lives in the callback above.
    @app.callback(
        Output(f"pl-{i}-title", "children"),
        Output(f"pl-{i}-map", "figure"), Output(f"pl-{i}-map", "style"),
        Output(f"pl-{i}-hist", "figure"), Output(f"pl-{i}-hist", "style"),
        Output(f"pl-{i}-histwrap", "style"),
        Output(f"pl-{i}-matrix", "figure"), Output(f"pl-{i}-matrixwrap", "style"),
        Output(f"pl-{i}-note", "children"),
        Output(f"pl-{i}-info", "children"), Output(f"pl-{i}-slices", "children"),
        Output(f"pl-{i}-hrange", "min"), Output(f"pl-{i}-hrange", "max"),
        Output(f"pl-{i}-hrange", "step"),
        Output(f"pl-{i}-hrange", "value", allow_duplicate=True),
        Output(f"pl-{i}-hecho", "data", allow_duplicate=True),
        Output(f"pl-{i}-matkey", "data"),
        Input(f"pl-{i}-enable", "value"), Input(f"pl-{i}-dis", "value"),
        Input(f"pl-{i}-mahfold", "value"), Input(f"pl-{i}-stem", "value"),
        Input(f"pl-{i}-grouping", "value"), Input(f"pl-{i}-maps", "value"),
        Input(f"pl-{i}-showhist", "value"),
        Input(f"pl-{i}-showmodel", "value"), Input(f"pl-{i}-maptype", "value"),
        Input(f"pl-{i}-axis", "value"), Input(f"pl-{i}-frac", "value"), Input(f"pl-{i}-range", "value"),
        Input(f"pl-{i}-cmap", "value"), Input(f"pl-{i}-cross", "data"),
        Input(f"pl-{i}-update", "n_clicks"),
        Input("ex-roi", "value"), Input("ex-dataver", "data"),
        Input("ex-source-mode", "value"), Input("ex-glm-model", "value"),
        Input("ex-view-height", "value"), Input("ex-grouped", "data"),
        Input("ex-update-trigger", "data"), State("ex-autoupdate", "value"),
        State("ex-datafolder", "value"), State("ex-dataset", "value"), State("ex-modality", "value"),
        State(f"pl-{i}-matkey", "data"),
        prevent_initial_call="initial_duplicate")
    def _cb(enable, dis, mahfold, stem, grouping, maps, showhist, showmodel, maptype, axis,
            frac, rng, cmap, cross, _card_update, roi, _ver, source, glm_model, view_h, grouped,
            _update_trig, autoupdate, datafolder, dataset, modality, matkey):
        t_start = time.time()
        vh = _int(view_h, DEFAULT_SETTINGS["view_height"])
        gshow = {"height": f"{vh}px"}
        wrap_show = MATRIX_WRAP_SHOW
        wrap_hide = WRAP_HIDE
        show_matrix = "on" in (showmodel or [])
        show_hist = "on" in (showhist or [])
        hist_wrap = {"flex": "2 1 0", "minWidth": 0} if show_hist else wrap_hide
        hs_keep = (no_update,) * 5            # histogram slider: leave scale alone
        trig = ctx.triggered_id
        if "on" not in (enable or []):        # card off — block hidden anyway
            return (no_update, no_update, no_update, no_update, no_update, hist_wrap,
                    no_update, wrap_hide, "", no_update, no_update) + hs_keep + (no_update,)

        # --- what this trigger can actually change --------------------------
        # Three tiers, and keeping them apart is most of what makes the app feel
        # quick. The old code redrew *everything* — including re-reading the model
        # CSV off the network disk — for a mouse-wheel notch on the slice.
        #
        #   GATED  — changes *which file* is read (model cascade, species, map
        #            type). Only these wait for Auto-update / 🔄, because only
        #            these can cost a network read.
        #   VIEW   — re-draws what is already in RAM (slice, axis, threshold,
        #            colormap, crosshair, view height). Always live: with the
        #            volume cached by path these are pure re-renders.
        #   FULL   — the deliberate top-bar actions and the two Update buttons.
        gated_triggers = {f"pl-{i}-dis", f"pl-{i}-mahfold", f"pl-{i}-stem",
                          f"pl-{i}-grouping", f"pl-{i}-maps", f"pl-{i}-maptype"}

        # Which model the card is on is settled the moment the picker is clicked:
        # it is a lookup in the in-memory index, not a read of anything. Resolve it
        # *before* the gate so the pending branch below can say which model is
        # waiting, and draw that model's matrix.
        model = _resolve_model(grouped, dis, mahfold, stem, grouping)

        if trig in gated_triggers and "auto" not in (autoupdate or []):
            # The gate protects the *result volume* — the expensive read. The title
            # and the model matrix are functions of the model **name** alone (a small
            # CSV, cached on its mtime), so holding them back only made the card
            # contradict its own breadcrumb: pick ``mahalanobis · emotion_identity``
            # over a ``correlation`` card and the matrix stayed on the previous
            # model's 40x40 grid, which reads as a mis-drawn matrix rather than as a
            # pending fetch. They follow the picker; only the map/histogram wait.
            log("skip", f"card {i + 1}  {trig} changed — gated, waiting for 🔄")
            pend_title = html.Span([
                status_dot(STATUS_STYLE["unlinked"][0]),
                html.Span(model or "— pick a set, a hypothesis and a grouping —",
                          style={"fontWeight": "bold", "color": MUTED}),
                html.Span("  · ⏸ pending", style={"color": MUTED})])
            mkey = [datafolder, dataset, model]
            if show_matrix and list(matkey or []) != mkey:
                mat, mkey_out = _model_heatmap(datafolder, dataset, model), mkey
            else:
                mat, mkey_out = no_update, no_update
            return (pend_title, no_update, no_update, no_update, no_update, hist_wrap,
                    mat, wrap_show if show_matrix else wrap_hide,
                    "⏸ change pending — click 🔄 on this card",
                    no_update, no_update) + hs_keep + (mkey_out,)
        # A click only moves the crosshair, and neither the in-mask histogram nor
        # the model matrix depend on it — nor do they depend on which slice or axis
        # is on screen. Only the slice figure has to be rebuilt for those.
        map_only = trig in (f"pl-{i}-cross", f"pl-{i}-frac", f"pl-{i}-axis")

        if not model:
            title = html.Span("— pick a set, a hypothesis and a grouping —",
                              style={"color": MUTED})
            empty = niftiutil.empty_fig("select a model + grouping", height=vh, dark=True)
            mat = _model_heatmap(datafolder, dataset, None) if show_matrix else no_update
            hist = niftiutil.empty_fig("no model", height=vh) if show_hist else no_update
            return (title, empty, gshow, hist, gshow, hist_wrap, mat,
                    wrap_show if show_matrix else wrap_hide, "no model",
                    _cross_hint("no model selected"), "0") + hs_keep + (None,)

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
        fig, hist, hbounds, n, n_mask, info, nsl = _card_species_fig(
            source, datafolder, dataset, modality, roi, glm_model,
            specie, model, maptype, axis, frac, zt, vh,
            colorscale=cmap, vmax_override=vmax,
            want_hist=show_hist and not map_only, cross=cross)
        note = f"{label}: {n} / {n_mask} vx ≥ {zt:g} in mask"

        # Model matrix. It depends on nothing but (data folder, dataset, model), so
        # it is only shipped when one of those actually moved since this card last
        # pushed one — otherwise a slice scrub would send a 40x40 heatmap down the
        # wire per notch. What "last pushed" means is a fact about the *browser*, so
        # the key rides in a memory Store next to the graph: reload the page and
        # both are empty together, and the matrix comes back instead of the card
        # trusting a server-side note that the browser never saw. (JSON round-trips
        # the tuple as a list, hence the list comparison.)
        mkey = [datafolder, dataset, model]
        if not show_matrix:
            mat, mkey_out = no_update, no_update
        elif list(matkey or []) != mkey or trig == f"pl-{i}-showmodel":
            mat, mkey_out = _model_heatmap(datafolder, dataset, model), mkey
        else:
            mat, mkey_out = no_update, no_update

        drew = ["map"]
        if hist is not None:
            drew.append("histogram")
        if mat is not no_update:
            drew.append("matrix")
        log("render", f"card {i + 1}  {model} · {specie} · {maptype}  "
                      f"[{trig or 'page load'}]  {'+'.join(drew)}  "
                      f"{(time.time() - t_start) * 1000:.0f} ms")
        return (title, fig, gshow, (hist if hist is not None else no_update), gshow, hist_wrap,
                mat, wrap_show if show_matrix else wrap_hide, note, info, str(nsl)) + \
            _hslider_props(hbounds, [zt, vmax]) + (mkey_out,)
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


def _register_panel_click(i):
    # Click on the brain view -> crosshair voxel. Plotly reports the click in the
    # *rendered slice's* frame (x = column, y = row), so it has to be run back
    # through the same orientation the slice was drawn with to name a voxel. The
    # map is re-resolved here rather than passed along, which is free: it is the
    # very volume the render callback just loaded, so `_load_map` is a cache hit.
    # ``allow_duplicate`` because the sync reader writes this same store when a
    # crosshair arrives from another card of the species.
    @app.callback(Output(f"pl-{i}-cross", "data", allow_duplicate=True),
                  Input(f"pl-{i}-map", "clickData"),
                  State(f"pl-{i}-dis", "value"), State(f"pl-{i}-mahfold", "value"),
                  State(f"pl-{i}-stem", "value"),
                  State(f"pl-{i}-grouping", "value"), State(f"pl-{i}-maps", "value"),
                  State(f"pl-{i}-maptype", "value"), State(f"pl-{i}-axis", "value"),
                  State(f"pl-{i}-frac", "value"), State(f"pl-{i}-range", "value"),
                  State("ex-grouped", "data"), State("ex-roi", "value"),
                  State("ex-source-mode", "value"), State("ex-glm-model", "value"),
                  State("ex-datafolder", "value"), State("ex-dataset", "value"),
                  State("ex-modality", "value"), prevent_initial_call=True)
    def _cb_click(click, dis, mahfold, stem, grouping, maps, maptype, axis, frac, rng,
                  grouped, roi, source, glm_model, datafolder, dataset, modality):
        pt = ((click or {}).get("points") or [{}])[0]
        if pt.get("x") is None or pt.get("y") is None:
            return no_update
        model = _resolve_model(grouped, dis, mahfold, stem, grouping)
        if not model:
            return no_update
        try:
            zt = float(rng[0])
        except (TypeError, ValueError, IndexError):
            zt = 0.0
        specie = maps if maps in ("D", "H") else "D"
        loaded = _load_map(source, datafolder, dataset, modality, roi, glm_model,
                           specie, model, maptype, zt)
        if loaded is None:
            return no_update
        data, aff = loaded
        ax = int(axis)
        idx = int(round(float(frac) * (data.shape[ax] - 1)))
        orient = niftiutil.slice_orientation(aff, ax)
        return list(niftiutil.slice_rc_to_voxel(data.shape, ax, idx,
                                                pt["y"], pt["x"], orient))
    return _cb_click


def _register_panel_barwrap(i):
    # Show/hide the model-bar half of the bottom row. Kept out of the big render
    # callback so toggling it never re-reads a map.
    @app.callback(Output(f"pl-{i}-barwrap", "style"),
                  Input(f"pl-{i}-showbar", "value"), Input(f"pl-{i}-enable", "value"))
    def _cb(showbar, enable):
        on = "on" in (enable or []) and "on" in (showbar or [])
        return BAR_WRAP_SHOW if on else WRAP_HIDE
    return _cb


def _register_panel_bar(i):
    # "What does every model say at this voxel?" — see the section above the data
    # helpers for the two read modes. This is **button-driven on purpose**: the
    # scope can be dozens of maps on a network disk. A crosshair move is allowed to
    # recompute only in *preloaded* mode and only once this context's store is
    # already warm (so the volumes are in RAM and it costs nothing); otherwise it
    # just marks the plot stale and waits for the button. That way clicking around
    # the brain can never kick off a multi-model disk scan by accident.
    @app.callback(
        Output(f"pl-{i}-bar", "figure"), Output(f"pl-{i}-bar", "style"),
        Output(f"pl-{i}-barnote", "children"),
        Input(f"pl-{i}-bar-btn", "n_clicks"), Input(f"pl-{i}-cross", "data"),
        State(f"pl-{i}-bar-scope", "value"), State(f"pl-{i}-dis", "value"),
        State(f"pl-{i}-mahfold", "value"),
        State(f"pl-{i}-stem", "value"), State(f"pl-{i}-grouping", "value"),
        State(f"pl-{i}-maps", "value"), State(f"pl-{i}-maptype", "value"),
        State(f"pl-{i}-axis", "value"), State(f"pl-{i}-frac", "value"),
        State(f"pl-{i}-range", "value"), State(f"pl-{i}-showbar", "value"),
        State(f"pl-{i}-enable", "value"),
        State("ex-grouped", "data"), State("ex-roi", "value"),
        State("ex-source-mode", "value"), State("ex-glm-model", "value"),
        State("ex-datafolder", "value"), State("ex-dataset", "value"),
        State("ex-modality", "value"), State("ex-barmode", "value"),
        prevent_initial_call=True)
    def _cb(_n, cross, scope, dis, mahfold, stem, grouping, maps, maptype, axis, frac, rng,
            showbar, enable, grouped, roi, source, glm_model, datafolder, dataset,
            modality, barmode):
        # Nothing to draw for a hidden plot or a switched-off card — and a card can
        # be off and still receive a crosshair, via 🔗 sync.
        if "on" not in (showbar or []) or "on" not in (enable or []):
            return no_update, no_update, no_update
        specie = maps if maps in ("D", "H") else "D"
        key = _bar_store_key(source, datafolder, dataset, modality, roi, glm_model,
                             specie, maptype)
        warm = bool(_BAR_PRELOAD.get(key))
        if ctx.triggered_id == f"pl-{i}-cross" and not (barmode == "preload" and warm):
            return no_update, no_update, "✛ moved — click 📶 Compare to resample"

        gshow = {"height": f"{BAR_MIN_H}px"}
        try:
            zt = float(rng[0])
        except (TypeError, ValueError, IndexError):
            zt = 0.0
        # The crosshair belongs to *this card's* map, so that map is what turns it
        # into a world coordinate. It is already loaded (the render callback just
        # drew it), so this is a cache hit.
        model_cur = _resolve_model(grouped, dis, mahfold, stem, grouping)
        loaded = _load_map(source, datafolder, dataset, modality, roi, glm_model,
                           specie, model_cur, maptype, zt) if model_cur else None
        if loaded is None:
            return (niftiutil.empty_fig("no map on this card", height=BAR_MIN_H), gshow,
                    "this card has no map to take a crosshair from")
        data, aff = loaded
        ax = int(axis)
        idx = int(round(float(frac) * (data.shape[ax] - 1)))
        vox = _cross_voxel(cross, data.shape, ax, idx)
        if vox is None:
            return (niftiutil.empty_fig("click ✛ then 📶 Compare", height=BAR_MIN_H), gshow,
                    "no crosshair yet — click the slice first")
        world = niftiutil.voxel_to_world(vox, aff)

        models, label_mode = _bar_model_list(grouped, dis, mahfold, grouping, scope)
        n_scope = len(models)
        models = models[:BAR_MAX_MODELS]
        store = _BAR_PRELOAD.setdefault(key, {}) if barmode == "preload" else None
        log("bars", f"card {i + 1}  sampling {len(models)} model(s) [{scope}] at "
                    f"({world[0]:.0f}, {world[1]:.0f}, {world[2]:.0f}) mm - {barmode} mode")
        t0 = time.time()
        rows = _bar_series(source, datafolder, dataset, modality, roi, glm_model,
                           specie, models, maptype, zt, world, store)
        dt = time.time() - t0
        log("bars", f"card {i + 1}  {len(rows)} model(s) had a map  {dt:.1f} s")

        label = {"D": "Dog", "H": "Human"}[specie]
        maptype_label = dict(MAPTYPES).get(maptype, maptype)
        xtitle = "group mean (Kendall τ)" if maptype == "mean" else "z"
        subtitle = ("✛ (%.1f, %.1f, %.1f) mm · %s · %s"
                    % (world[0], world[1], world[2], label, maptype_label))
        fig, height = _bar_fig(rows, model_cur, grouping, label_mode, zt, xtitle, subtitle)

        shown = min(n_scope, BAR_MAX_MODELS)
        note = f"{len(rows)}/{shown} models with a map · {dt:.1f}s"
        if n_scope > BAR_MAX_MODELS:
            note += f" (scope capped at {BAR_MAX_MODELS} of {n_scope})"
        n_err = sum(1 for r in rows if r[2] is not None)
        if n_err:
            note += f" · ±SEM on {n_err}"
        elif maptype == "mean":
            note += " · no ±SEM (no _std / _mean.json beside the maps)"
        if barmode == "preload":
            note += f" · {_bar_cache_mb():.0f} MB held"
        return fig, {"height": f"{height}px"}, note
    return _cb


for _i in range(MAX_MODELS):
    _register_panel(_i)
    _register_panel_style(_i)
    _register_panel_click(_i)
    _register_panel_barwrap(_i)
    _register_panel_bar(_i)


# ---------------------------------------------------------------------------
# Callbacks — cross-card view sync (slice / axis / threshold / scale / colormap)
# ---------------------------------------------------------------------------
# Each card carries a "🔗 sync" toggle. Synced cards of the *same species* share one
# view: moving the slice/axis, changing threshold / max / colormap, **or clicking a
# new crosshair** on any of them mirrors to all the others. This is done with two
# tiny broadcast stores (``ex-sync-D`` / ``ex-sync-H``):
#   * a single **writer** callback watches every card's SYNC_KEYS + sync toggle;
#     when a synced card's control changes (or its sync turns on) it publishes that
#     card's values into the store for the card's species. It *merges* into what is
#     already there rather than replacing it, so a card that has no crosshair of its
#     own does not wipe the shared one — it just leaves that key as it found it.
#   * a per-card **reader** adopts its species store whenever the store (or the
#     card's own sync / species) changes, writing the shared values back onto its
#     controls. The loop is self-limiting: a reader only writes values equal to the
#     store, so the writer it re-triggers republishes the same data and Dash stops.
#
# The crosshair travels as a **voxel index**, which is what makes it meaningful in
# another card at all: same-species maps sit on one voxel grid (the pipeline's hard
# invariant), so voxel (i, j, k) is the same anatomy in every card of that species.

def _sync_params_from(vals, i, prev):
    """Card *i*'s SYNC_KEYS snapshot, merged over the species store's current
    contents. A ``None`` crosshair is dropped rather than published: "this card has
    no crosshair" must not clear everyone else's."""
    params = dict(prev) if isinstance(prev, dict) else {}
    params.update({p: vals[(i, p)] for p in SYNC_KEYS})
    if params.get("cross") is None:
        params.pop("cross", None)
    return params


@app.callback(
    Output("ex-sync-D", "data", allow_duplicate=True),
    Output("ex-sync-H", "data", allow_duplicate=True),
    *[Input(f"pl-{i}-{p}", "value") for p in SYNC_CONTROLS for i in range(MAX_MODELS)],
    *[Input(f"pl-{i}-{p}", "data") for p in SYNC_STORES for i in range(MAX_MODELS)],
    *[Input(f"pl-{i}-sync", "value") for i in range(MAX_MODELS)],
    *[State(f"pl-{i}-maps", "value") for i in range(MAX_MODELS)],
    State("ex-sync-D", "data"), State("ex-sync-H", "data"),
    prevent_initial_call=True)
def cb_sync_write(*args):
    n = MAX_MODELS
    nkeys = len(SYNC_KEYS)
    # Reshape the flat Dash arg list back into addressable groups. Order follows
    # the declaration above: control values, store data, sync toggles, then the
    # States (species per card, then the two broadcast stores).
    vals = {}
    for pi, p in enumerate(SYNC_KEYS):
        for i in range(n):
            vals[(i, p)] = args[pi * n + i]
    syncs = args[nkeys * n: (nkeys + 1) * n]
    mapss = args[(nkeys + 1) * n: (nkeys + 2) * n]
    store_d, store_h = args[(nkeys + 2) * n], args[(nkeys + 2) * n + 1]

    trig = ctx.triggered_id                      # e.g. "pl-2-frac"
    m = re.match(r"pl-(\d+)-(\w+)$", trig or "")
    if not m:
        return no_update, no_update
    i, prop = int(m.group(1)), m.group(2)
    if "sync" not in (syncs[i] or []):           # only synced cards publish
        return no_update, no_update
    if prop not in SYNC_KEYS and prop != "sync":
        return no_update, no_update

    if mapss[i] == "H":
        return no_update, _sync_params_from(vals, i, store_h)
    return _sync_params_from(vals, i, store_d), no_update


def _register_panel_sync_read(i):
    @app.callback(
        [Output(f"pl-{i}-{p}", "value", allow_duplicate=True) for p in SYNC_CONTROLS] +
        [Output(f"pl-{i}-{p}", "data", allow_duplicate=True) for p in SYNC_STORES],
        Input("ex-sync-D", "data"), Input("ex-sync-H", "data"),
        Input(f"pl-{i}-sync", "value"), Input(f"pl-{i}-maps", "value"),
        prevent_initial_call=True)
    def _cb(store_d, store_h, sync, maps):
        if "sync" not in (sync or []):
            return [no_update] * len(SYNC_KEYS)
        store = store_h if maps == "H" else store_d
        if not isinstance(store, dict) or not store:
            return [no_update] * len(SYNC_KEYS)
        return [store.get(p, no_update) for p in SYNC_KEYS]
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
    log("update", f"Update now - refreshing {n_on} enabled card(s)")
    return (ver or 0) + 1, f"Updated {n_on} enabled card(s)."


@app.callback(Output("ex-models-title", "children", allow_duplicate=True),
              Input("ex-autoupdate", "value"), prevent_initial_call=True)
def cb_autoupdate_status(val):
    on = "auto" in (val or [])
    log("update", f"auto-update {'ON' if on else 'OFF'}")
    if on:
        return "Auto-update ON — cards refresh as you change settings."
    return ("Auto-update OFF — change the model/species/map type, then click 🔄 on a card "
            "(or 🔄 Update now for all).")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="RSA model explorer (any dataset)")
    ap.add_argument("--port", type=int,
                    default=int(os.environ.get("EXPLORER_PORT", os.environ.get("PORT", "8055"))))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--verbose", action="store_true",
                    help="also log cache hits, directory listings and settings saves")
    ap.add_argument("--quiet", action="store_true",
                    help="log warnings only")
    ap.add_argument("--http-log", action="store_true",
                    help="restore Werkzeug's one-line-per-request access log "
                         "(off by default: it fires several times per slider notch "
                         "and says nothing about what the app did)")
    args = ap.parse_args()

    global LOG_LEVEL
    LOG_LEVEL = 0 if args.quiet else (2 if args.verbose else 1)

    # Werkzeug logs every POST /_dash-update-component at INFO. A single drag of a
    # slider produces a wall of those and buries the app's own messages, so the
    # access log is muted and what the app prints instead is the work it did.
    # ERROR (not WARNING) is deliberate: the "Running on http://…" banner and the
    # development-server warning are logged at those levels too, and we print our
    # own startup lines below.
    if not args.http_log:
        logging.getLogger("werkzeug").setLevel(logging.ERROR)
        app.logger.setLevel(logging.ERROR)
        try:    # Flask's own " * Serving Flask app …" banner goes through click, not
                # the logger, so muting the logger alone leaves it behind.
            import flask.cli
            flask.cli.show_server_banner = lambda *a, **k: None
        except Exception:
            pass

    print()
    log("boot", f"RSA Model Explorer v{VERSION}")
    log("boot", f"settings   {SETTINGS_PATH}")
    log("boot", f"source     {SETTINGS['source_mode']}  "
                f"datafolder={_initial_datafolder(SETTINGS)}")
    log("boot", f"dataset    {SETTINGS['dataset']}  glm={SETTINGS['glm_model']}")
    log("boot", f"log level  {['warnings only', 'normal', 'verbose'][LOG_LEVEL]}"
                f"{'' if args.http_log else '  (HTTP access log muted)'}")
    log("boot", f"open       http://{args.host}:{args.port}")
    print()
    app.run(debug=args.debug, use_reloader=False, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

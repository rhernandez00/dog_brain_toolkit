#!/usr/bin/env python
"""
models_manifest.py — read the centralized RSA model manifest ``_models.csv``.

``_models.csv`` is the single source of truth for **which models exist, which
distance method and Mahalanobis fold each belongs to, and which groupings each
offers**. It lives in a dataset's ``rsa_models`` folder and is (re)built by
``tools/build_models_manifest.py``:

    {datafolder}/{dataset}/rsa_models/_models.csv

Both parts of that path are the caller's — the same file is read for any dataset
on any data folder, so a new project only has to drop its own ``_models.csv`` in
its own ``rsa_models`` folder. Nothing here knows about EmoC.

Columns (one row per model *family* × distance method × fold):

    dis_method          pairwise-similarity method (mahalanobis / correlation / ...)
    mah_fold            folding strategy the family was built for (stim-wise / run-wise / ...)
    model               model family stem — matches the "{model}__{grouping}.csv" filenames
    groupings_possible  python-list literal of grouping suffixes, e.g. "['all', 'within', ...]"
    why                 one-line rationale, shown in the dashboards

``dis_method`` is the **outermost** filter: it decides which models exist at all.
The fold is only a real choice under ``mahalanobis`` (it names how the crossnobis
folds are cut); for every other method the manifest's ``mah_fold`` cell is
bookkeeping, so ``dis_index`` pools those rows under the synthetic ``FOLD_ANY``
and callers skip the fold menu — see ``uses_fold``.

Both ``pipeline_dashboard.py`` and ``hypothesis_explorer.py`` read models through
here, so a model can be added / edited / re-grouped in one place. The reader is
tolerant: a missing file yields an empty manifest (callers fall back to scanning
the folder), and a stale ``groupings_possible`` cell is parsed leniently.

This module only reads data; nothing in the core toolkit imports it.
"""

import ast
import os

import pandas as pd

try:  # canonical pipeline data disk, used to locate the manifest when a caller's
    from scheduler.paths import get_paths   # datafolder is a Drive mirror without it
except Exception:  # pragma: no cover - scheduler always importable in this repo
    get_paths = None

MANIFEST_FILENAME = "_models.csv"

# Canonical display order for grouping suffixes across folds. "all" (stim-wise
# pooled) and "collapse" (run-wise pooled) both mean "everything together"; the rest
# follow. Unknown groupings are appended (sorted) after these.
GROUPING_ORDER = ["all", "collapse", "within", "cross", "dog", "hum"]

# Manifest rows without a ``dis_method`` cell predate the column; they are the
# Mahalanobis battery.
DEFAULT_DIS_METHOD = "mahalanobis"

# The only distance method for which the Mahalanobis fold is a real choice — it is
# what ``--mah_fold`` cuts the crossnobis folds by. Every other method (correlation,
# ...) has no folding step, so its rows are pooled and the fold menu is skipped.
FOLD_METHODS = {"mahalanobis"}

# Synthetic fold key holding *every* row of a distance method, regardless of the
# ``mah_fold`` cell. Always present, and it is the entry callers use when the fold
# level does not apply.
FOLD_ANY = "(any fold)"

_ROWS_CACHE = {}   # tuple(normalised dirs) -> parsed rows


# ---------------------------------------------------------------------------
# Folder / path helpers
# ---------------------------------------------------------------------------
def rsa_models_dirs(datafolder, dataset):
    """Candidate ``rsa_models`` folders to search for the manifest and model CSVs:
    the caller's data folder first, then the canonical pipeline data disk
    (``get_paths``), de-duplicated. The pipeline-disk entry lets the manifest and
    model CSVs authored there be found even when results are viewed from a Google
    Drive mirror that has not synced them."""
    dirs = []
    if datafolder:
        dirs.append(os.path.join(datafolder, dataset or "", "rsa_models"))
    if get_paths is not None:
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


def manifest_path(dirs):
    """First existing ``_models.csv`` among ``dirs``, else None."""
    for d in dirs:
        p = os.path.join(d, MANIFEST_FILENAME)
        if os.path.isfile(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _parse_groupings(raw):
    """Grouping list from a ``groupings_possible`` cell. Accepts a python-list
    literal ("['all', 'cross']"), or a plain comma/semicolon separated string."""
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]
    s = str(raw or "").strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str) and v.strip():
            return [v.strip()]
    except (ValueError, SyntaxError):
        pass
    for sep in (";", ","):
        if sep in s:
            return [p.strip().strip("[]'\" ") for p in s.split(sep) if p.strip().strip("[]'\" ")]
    return [s.strip("[]'\" ")]


def normalize_dis_method(raw):
    """A manifest ``dis_method`` cell as a clean string, defaulting to
    ``mahalanobis`` when the column is missing or blank (pre-column rows)."""
    return str(raw or "").strip() or DEFAULT_DIS_METHOD


def uses_fold(dis_method):
    """True when the Mahalanobis fold is a meaningful choice for this distance
    method — i.e. when the menus should offer a fold level at all."""
    return normalize_dis_method(dis_method).lower() in FOLD_METHODS


def order_groupings(groupings):
    """Groupings in canonical order, de-duplicated; unknown ones sorted at the end."""
    seen, uniq = set(), []
    for g in groupings:
        g = str(g).strip()
        if g and g not in seen:
            seen.add(g)
            uniq.append(g)
    known = [g for g in GROUPING_ORDER if g in uniq]
    extra = sorted(g for g in uniq if g not in GROUPING_ORDER)
    return known + extra


def load_rows(dirs, use_cache=True):
    """Parsed manifest rows as a list of dicts ``{mah_fold, model, groupings, why}``,
    in file order. Empty list when no ``_models.csv`` is found. Cached per dir-set;
    call ``clear_cache()`` after the file changes on disk."""
    key = tuple(os.path.normcase(os.path.abspath(d)) for d in dirs)
    if use_cache and key in _ROWS_CACHE:
        return _ROWS_CACHE[key]
    rows = []
    path = manifest_path(dirs)
    if path:
        try:
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                model = str(r.get("model", "") or "").strip()
                fold = str(r.get("mah_fold", "") or "").strip()
                if not model or not fold:
                    continue
                rows.append({
                    "dis_method": normalize_dis_method(r.get("dis_method")),
                    "mah_fold": fold,
                    "model": model,
                    "groupings": order_groupings(_parse_groupings(r.get("groupings_possible"))),
                    "why": str(r.get("why", "") or "").strip(),
                })
        except Exception:
            rows = []
    _ROWS_CACHE[key] = rows
    return rows


def clear_cache():
    """Drop the parsed-manifest cache (call after editing ``_models.csv``)."""
    _ROWS_CACHE.clear()


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def folds(dirs):
    """Distinct ``mah_fold`` values, in manifest order."""
    out, seen = [], set()
    for r in load_rows(dirs):
        f = r["mah_fold"]
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def dis_methods(dirs):
    """Distinct ``dis_method`` values, in manifest order. This is the *first* menu
    level: it decides which models exist for the analysis being looked at."""
    out, seen = [], set()
    for r in load_rows(dirs):
        d = r["dis_method"]
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def folds_for_dis_method(dirs, dis_method):
    """Distinct ``mah_fold`` values within one distance method, in manifest order.
    Empty for a method that does not fold (see ``uses_fold``)."""
    if not uses_fold(dis_method):
        return []
    want = normalize_dis_method(dis_method)
    out, seen = [], set()
    for r in load_rows(dirs):
        if r["dis_method"] != want:
            continue
        if r["mah_fold"] not in seen:
            seen.add(r["mah_fold"])
            out.append(r["mah_fold"])
    return out


def models_for_fold(dirs, mah_fold):
    """Ordered ``[(model, [groupings], why)]`` for one fold (manifest order)."""
    return [(r["model"], r["groupings"], r["why"])
            for r in load_rows(dirs) if r["mah_fold"] == mah_fold]


def concrete_model_name(dirs, model, grouping):
    """The concrete CSV stem for a (family, grouping): ``"{model}__{grouping}"``
    normally, but the suffix-less ``"{model}"`` when only that file exists on disk
    (e.g. ``agent-species-id``). Prefers an on-disk match; defaults to the suffixed
    form so not-yet-built models still get a stable name."""
    suffixed = f"{model}__{grouping}"
    for d in dirs:
        if os.path.isfile(os.path.join(d, suffixed + ".csv")):
            return suffixed
    for d in dirs:
        if os.path.isfile(os.path.join(d, model + ".csv")):
            return model
    return suffixed


def _blank_fold():
    return {"stems": [], "index": {}, "why": {}, "groupings": {}}


def _add_row(fd, dirs, row):
    """Fold one manifest row into a ``by_fold`` entry (merging when the stem is
    already there, which is what pooling several folds into ``FOLD_ANY`` does)."""
    stem = row["model"]
    if stem not in fd["index"]:
        fd["stems"].append(stem)
        fd["index"][stem] = {}
        fd["why"][stem] = row["why"]
        fd["groupings"][stem] = []
    for g in row["groupings"]:
        fd["index"][stem][g] = concrete_model_name(dirs, stem, g)
    fd["groupings"][stem] = order_groupings(fd["groupings"][stem] + list(row["groupings"]))
    # keep the {grouping: model} dict itself in canonical order — the grouping menu
    # is built from its key order
    fd["index"][stem] = {g: fd["index"][stem][g] for g in fd["groupings"][stem]}


def dis_index(dirs):
    """The structure backing the model dashboards, **distance method first**::

        {'dis_methods': [dis_method, ...],
         'by_dis': {dis_method: {
             'uses_fold': bool,              # is the fold menu meaningful here?
             'folds':     [mah_fold, ...],   # empty when uses_fold is False
             'by_fold':   {mah_fold: {'stems':     [stem, ...],
                                      'index':     {stem: {grouping: concrete_model}},
                                      'why':       {stem: why},
                                      'groupings': {stem: [grouping, ...]}},
                           FOLD_ANY: {...}}}}}   # every row of the method, pooled

    ``FOLD_ANY`` is always present for every method, so a caller that skips the
    fold level (any non-Mahalanobis method) has one entry to read instead of a
    special case. Empty ``dis_methods`` when there is no ``_models.csv`` (callers
    fall back to scanning the folder)."""
    out = {"dis_methods": [], "by_dis": {}}
    for r in load_rows(dirs):
        dis = r["dis_method"]
        if dis not in out["by_dis"]:
            out["dis_methods"].append(dis)
            out["by_dis"][dis] = {"uses_fold": uses_fold(dis), "folds": [],
                                  "by_fold": {FOLD_ANY: _blank_fold()}}
        dd = out["by_dis"][dis]
        if dd["uses_fold"]:
            fold = r["mah_fold"]
            if fold not in dd["by_fold"]:
                dd["folds"].append(fold)
                dd["by_fold"][fold] = _blank_fold()
            _add_row(dd["by_fold"][fold], dirs, r)
        _add_row(dd["by_fold"][FOLD_ANY], dirs, r)
    return out


def concrete_models_for_fold(dirs, mah_fold):
    """Flat, ordered, de-duplicated list of concrete model names for a fold
    (each family expanded over its groupings)."""
    seen, out = set(), []
    for model, groupings, _why in models_for_fold(dirs, mah_fold):
        for g in groupings:
            name = concrete_model_name(dirs, model, g)
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def concrete_models_for(dirs, dis_method, mah_fold=None):
    """Flat, ordered, de-duplicated concrete model names for one ``dis_method``,
    optionally restricted to one Mahalanobis fold. Each family is expanded over
    its groupings; rows without a ``dis_method`` column default to 'mahalanobis'.

    This is the query a model menu wants: a fold name is not unique across
    distance methods (EmoC's correlation rows are 'run-wise', its mahalanobis rows
    'stim-wise'), so filtering by fold alone mixes methods together, and filtering
    by neither offers models that cannot be run with the selected method at all.

    ``mah_fold`` is honoured only where ``uses_fold(dis_method)`` says the fold is
    a real choice (mahalanobis); for every other method it is ignored, because
    there the fold decides nothing about which models exist."""
    want = normalize_dis_method(dis_method)
    by_fold = uses_fold(want) and mah_fold not in (None, "", FOLD_ANY)
    seen, out = set(), []
    for r in load_rows(dirs):
        if r["dis_method"] != want:
            continue
        if by_fold and r["mah_fold"] != mah_fold:
            continue
        for g in r["groupings"]:
            name = concrete_model_name(dirs, r["model"], g)
            if name not in seen:
                seen.add(name)
                out.append(name)
    return out


def concrete_models_for_dis_method(dirs, dis_method):
    """Every concrete model name of one ``dis_method``, all folds pooled."""
    return concrete_models_for(dirs, dis_method)


def classified_models(dirs):
    """Set of every concrete model name the manifest classifies, across all folds.
    Used to tell manifest-managed models apart from unclassified/legacy ones."""
    out = set()
    for r in load_rows(dirs):
        for g in r["groupings"]:
            out.add(concrete_model_name(dirs, r["model"], g))
    return out


def all_concrete_models(dirs):
    """Flat, ordered, de-duplicated list of every concrete model name the manifest
    classifies, across all folds (manifest fold order, then per-fold order). Use
    this when the Mahalanobis fold does not apply (e.g. a non-mahalanobis distance
    method) so every manifest model is offered regardless of fold."""
    seen, out = set(), []
    for f in folds(dirs):
        for m in concrete_models_for_fold(dirs, f):
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def family_of(dirs, concrete):
    """The manifest row matching a concrete model name (``{'mah_fold','model',
    'groupings','why'}``), or None. Matches on exact stem or ``{stem}__`` prefix."""
    for r in load_rows(dirs):
        if concrete == r["model"] or concrete.startswith(r["model"] + "__"):
            return r
    return None


def why_for(dirs, concrete):
    """The 'why' note for a concrete model name (matched by family), or ''."""
    r = family_of(dirs, concrete)
    return r["why"] if r else ""

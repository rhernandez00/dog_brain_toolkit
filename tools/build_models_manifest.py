#!/usr/bin/env python
"""
build_models_manifest.py — (re)build the centralized RSA model manifest `_models.csv`.

`_models.csv` is the single source of truth that both `pipeline_dashboard.py` and
`hypothesis_explorer.py` read to decide **which models exist, which Mahalanobis
fold each belongs to, and which groupings each offers**. It lives in the dataset's
``rsa_models`` folder:

    {datafolder}/{dataset}/rsa_models/_models.csv

Columns (one row per model *family* × fold):

    mah_fold            folding strategy the family was built for (stim-wise / run-wise / ...)
    model               model family stem — matches the "{model}__{grouping}.csv" filenames
    groupings_possible  python-list literal of grouping suffixes, e.g. "['all', 'within', ...]"
    why                 one-line rationale, shown in the dashboards

This script assembles that file from the two upstream battery manifests:

  * **stim-wise** battery — ``model_manifest_EmoC_RSA_model_battery.csv`` (built by
    the Obsidian battery generator). Every row is ``mah_fold='stim-wise'``; the
    ``why`` is copied from its ``why_test`` column, groupings come from ``scope``.
  * **run-wise** battery — ``_MODEL_BATTERY_MANIFEST.csv`` (on the data disk). Every
    row is ``mah_fold='run-wise'``; the ``why`` is the ``description`` column with
    its trailing " | {grouping clause}" stripped, groupings come from ``grouping``.

Run it whenever either upstream manifest changes:

    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\build_models_manifest.py

Pass ``--stimwise`` / ``--runwise`` / ``--out`` to point at other files. Nothing
here is imported by the core toolkit; it only writes data.
"""

import argparse
import ast
import os

import pandas as pd

# Canonical display order for grouping suffixes; the reader re-applies this, but we
# also emit groupings in this order so the CSV reads naturally. "all" (stim-wise
# pooled) and "collapse" (run-wise pooled) both mean "everything together".
GROUPING_ORDER = ["all", "collapse", "within", "cross", "dog", "hum"]

# Default locations on this machine (Windows). Override with CLI flags elsewhere.
DEFAULT_STIMWISE = (r"C:\Users\raul_\Documents\Obsidian\Nexus\output"
                    r"\EmoC_RSA_model_battery\model_manifest_EmoC_RSA_model_battery.csv")
DEFAULT_RUNWISE = r"P:\userdata\raulh87\data\EmoC\rsa_models\_MODEL_BATTERY_MANIFEST.csv"
DEFAULT_OUT = r"P:\userdata\raulh87\data\EmoC\rsa_models\_models.csv"


def _order(groupings):
    """Groupings in canonical order, de-duplicated, with any unknown ones appended."""
    seen, uniq = set(), []
    for g in groupings:
        g = str(g).strip()
        if g and g not in seen:
            seen.add(g)
            uniq.append(g)
    known = [g for g in GROUPING_ORDER if g in uniq]
    extra = [g for g in uniq if g not in GROUPING_ORDER]
    return known + extra


def _groupings_literal(groupings):
    """A python-list literal string, e.g. ['all', 'within', 'cross'] (matches the
    format read back by ast.literal_eval in models_manifest.py)."""
    return "[" + ", ".join(repr(g) for g in groupings) + "]"


def rows_from_stimwise(path):
    """[(family, [groupings], why)] from the stim-wise battery manifest, one entry
    per family, preserving first-seen family order. groupings from ``scope``, why
    copied verbatim from ``why_test``."""
    df = pd.read_csv(path)
    order, acc = [], {}
    for _, r in df.iterrows():
        fam = str(r["family"]).strip()
        scope = str(r["scope"]).strip()
        why = str(r.get("why_test", "") or "").strip()
        if fam not in acc:
            acc[fam] = {"groupings": [], "why": why}
            order.append(fam)
        if scope and scope not in acc[fam]["groupings"]:
            acc[fam]["groupings"].append(scope)
        if not acc[fam]["why"] and why:
            acc[fam]["why"] = why
    return [(fam, _order(acc[fam]["groupings"]), acc[fam]["why"]) for fam in order]


def rows_from_runwise(path):
    """[(family, [groupings], why)] from the run-wise battery manifest, one entry per
    family, preserving first-seen order. The family stem is the ``hypothesis`` column
    (the ``model`` column holds concrete "{family}__{grouping}" names); groupings come
    from ``grouping``, why from ``description`` with the trailing
    ' | {grouping clause}' removed."""
    df = pd.read_csv(path)
    order, acc = [], {}
    for _, r in df.iterrows():
        # Prefer the family stem (hypothesis); fall back to the family part of the
        # concrete model name for any row missing a hypothesis.
        fam = str(r.get("hypothesis", "") or "").strip() or str(r["model"]).strip().split("__", 1)[0]
        grouping = str(r["grouping"]).strip()
        desc = str(r.get("description", "") or "").strip()
        why = desc.split(" | ", 1)[0].strip()   # drop the per-grouping clause
        if fam not in acc:
            acc[fam] = {"groupings": [], "why": why}
            order.append(fam)
        if grouping and grouping not in acc[fam]["groupings"]:
            acc[fam]["groupings"].append(grouping)
        if not acc[fam]["why"] and why:
            acc[fam]["why"] = why
    return [(fam, _order(acc[fam]["groupings"]), acc[fam]["why"]) for fam in order]


def build(stimwise_path, runwise_path):
    """Assemble the full manifest DataFrame from both source batteries."""
    out_rows = []
    for fam, groupings, why in rows_from_stimwise(stimwise_path):
        out_rows.append({"mah_fold": "stim-wise", "model": fam,
                         "groupings_possible": _groupings_literal(groupings), "why": why})
    for model, groupings, why in rows_from_runwise(runwise_path):
        out_rows.append({"mah_fold": "run-wise", "model": model,
                         "groupings_possible": _groupings_literal(groupings), "why": why})
    return pd.DataFrame(out_rows, columns=["mah_fold", "model", "groupings_possible", "why"])


def main():
    ap = argparse.ArgumentParser(description="Build the centralized _models.csv manifest")
    ap.add_argument("--stimwise", default=DEFAULT_STIMWISE, help="stim-wise battery manifest CSV")
    ap.add_argument("--runwise", default=DEFAULT_RUNWISE, help="run-wise battery manifest CSV")
    ap.add_argument("--out", default=DEFAULT_OUT, help="destination _models.csv")
    args = ap.parse_args()

    missing = [p for p in (args.stimwise, args.runwise) if not os.path.isfile(p)]
    if missing:
        raise SystemExit("source manifest(s) not found:\n  " + "\n  ".join(missing))

    df = build(args.stimwise, args.runwise)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[build_models_manifest] wrote {len(df)} model rows -> {args.out}")
    for fold in df["mah_fold"].unique():
        n = int((df["mah_fold"] == fold).sum())
        print(f"    {fold:12s}: {n} families")


if __name__ == "__main__":
    main()

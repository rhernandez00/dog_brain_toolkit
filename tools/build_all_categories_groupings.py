"""
build_all_categories_groupings.py — derive agent-species grouping variants of the
hand-built ``all-categories_*`` RSA models (categorical / bipolar / emotionality).

The ``all-categories_*`` models are 10x10 dissimilarity matrices over the 10 EmoC
conditions (5 emotions x 2 species-shown, exemplars collapsed), with condition
labels ``HumH, DogH, HumP, ...``. Unlike the factorial battery in
``build_rsa_models.py`` these were authored directly (builder UI / by hand), so
their values are NOT re-derived here — we only *mask* which condition pairs enter
the RSA correlation, exactly the way ``build_rsa_models.py`` applies its groupings.

For each source model and each grouping we write a full 10x10 matrix that keeps
the original values on the pairs the grouping allows and sets every other
off-diagonal cell to ``NaN`` (excluded from the correlation; the diagonal stays
0). Grouping predicates and descriptions are imported from ``build_rsa_models``
so the two stay in lock-step.

Groupings produced (subset of ``build_rsa_models.GROUPINGS``):
    collapse  all pairs kept                      -> identical values to the source
    cross     only Dog-shown x Hum-shown pairs    -> agent-invariant emotion test
    dog       only Dog-shown x Dog-shown pairs    -> structure within dog stimuli
    hum       only Hum-shown x Hum-shown pairs    -> structure within human stimuli

Output: ``{name}__{grouping}.csv`` next to the sources, in
``{datafolder}/EmoC/rsa_models/`` (or ``--out_dir``). Any stale ``{name}.npy``
cache for a written model is removed so ``read_model_dict`` re-reads the matrix.

Run:
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\build_all_categories_groupings.py
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\build_all_categories_groupings.py --dry-run
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)  # tools/ lives one level below the repo root
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scheduler.paths import get_paths
# Reuse the exact grouping predicates/descriptions the battery uses so the two
# generators can never drift apart.
from build_rsa_models import _grouping, GROUPINGS

# Source models to derive variants from.
SOURCE_MODELS = [
    "all-categories_categorical",
    "all-categories_bipolar",
    "all-categories_emotionality",
]

# Which groupings to emit (build_rsa_models also defines "within"; the request
# asked for these four).
GROUPING_NAMES = ["collapse", "cross", "dog", "hum"]

NAN = np.nan


def species_of(label):
    """Map a condition label (e.g. 'DogH', 'HumN') to its agent species-shown."""
    if label.startswith("Dog"):
        return "Dog"
    if label.startswith("Hum"):
        return "Hum"
    raise ValueError(f"Cannot determine species from condition label {label!r}")


def read_source_matrix(csv_path):
    """Return (labels, matrix) for a source model CSV (first column = row labels)."""
    table = pd.read_csv(csv_path, index_col=0)
    labels = table.columns.tolist()
    if list(table.index) != labels:
        raise ValueError(
            f"{csv_path}: row labels {list(table.index)} do not match column "
            f"labels {labels}; expected a square, symmetrically-labelled matrix."
        )
    return labels, table.to_numpy(dtype=np.float64)


def apply_grouping(labels, matrix, grouping_ok):
    """Copy `matrix`, NaN-ing every off-diagonal pair the grouping excludes."""
    species = [species_of(l) for l in labels]
    n = len(labels)
    out = matrix.copy()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue  # diagonal stays as-is (0)
            if not grouping_ok(species[i], species[j]):
                out[i, j] = NAN
    return out


def matrix_stats(labels, m):
    n = len(labels)
    iu = np.triu_indices(n, k=1)
    vals = m[iu]
    finite = vals[~np.isnan(vals)]
    return {
        "pairs_used": int(finite.size),
        "n_excluded": int(iu[0].size - finite.size),
        "has_variance": bool(finite.size > 0 and np.ptp(finite) > 0),
    }


def write_model(out_dir, name, labels, m, dry_run=False):
    df = pd.DataFrame(m, index=labels, columns=labels)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    if not dry_run:
        df.to_csv(csv_path, na_rep="NaN")
        npy = os.path.join(out_dir, f"{name}.npy")
        if os.path.exists(npy):
            os.remove(npy)  # drop stale read_model_dict cache
    return csv_path


def main():
    ap = argparse.ArgumentParser(
        description="Derive collapse/cross/dog/hum grouping variants of the "
                    "all-categories_* RSA models."
    )
    ap.add_argument("--dataset", default="EmoC")
    ap.add_argument("--out_dir", default=None,
                    help="Override output folder (default: {datafolder}/{dataset}/rsa_models)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and stats without writing files.")
    args = ap.parse_args()

    if args.out_dir:
        model_dir = args.out_dir
    else:
        root = get_paths()[0]  # pipeline data disk (P:\ on Windows / mount on Linux)
        model_dir = os.path.join(root, args.dataset, "rsa_models")

    print(f"Model dir: {model_dir}")
    print(f"Dry run:   {args.dry_run}\n")

    rows = []
    for source in SOURCE_MODELS:
        src_path = os.path.join(model_dir, f"{source}.csv")
        if not os.path.exists(src_path):
            print(f"SKIP: source model not found: {src_path}")
            continue
        labels, matrix = read_source_matrix(src_path)
        for grouping in GROUPING_NAMES:
            name = f"{source}__{grouping}"
            m = apply_grouping(labels, matrix, _grouping(grouping))
            st = matrix_stats(labels, m)
            write_model(model_dir, name, labels, m, dry_run=args.dry_run)
            rows.append({
                "model": name, "source": source, "grouping": grouping,
                "description": GROUPINGS[grouping], **st,
            })

    man_df = pd.DataFrame(rows)
    bad = man_df[~man_df["has_variance"]]
    if len(bad):
        print("WARNING: models with no variance (cannot correlate):")
        print(bad[["model"]].to_string(index=False))

    print(f"\nGenerated {len(rows)} models "
          f"({len(SOURCE_MODELS)} sources x {len(GROUPING_NAMES)} groupings).\n")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(man_df[["model", "grouping", "pairs_used", "n_excluded"]].to_string(index=False))

    if not args.dry_run:
        print(f"\nWrote {len(rows)} .csv files to {model_dir}")


if __name__ == "__main__":
    main()

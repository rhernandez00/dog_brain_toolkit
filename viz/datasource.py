"""Result-folder resolution and scanning, shared by every dashboard tab.

Resolution order for the viewer's "current-results" tree (first hit wins):

    1. $DBT_RESULTS_ROOT          explicit override (env var)
    2. Google Drive               G:\\My Drive\\Results        (Windows)
    3. Network share              P:\\userdata\\raulh87\\data  (Windows)
                                  /home/.../userdata/raulh87/data (Linux)

A "results root" is the folder that directly contains
``{dataset}/current-results/{modality}/{specie}/{roi_type}/...`` — the flat
layout that step 10 of searchlight.py mirrors to Google Drive.

Both the Normal-mode Dash app and the Failsafe static exporter import from
here so the two stay in lockstep.
"""

import os
import glob


# --- Candidate roots ------------------------------------------------------

def _windows_candidates():
    return [
        r"G:\My Drive\Results",
        r"P:\userdata\raulh87\data",
    ]


def _linux_candidates():
    base = os.path.join("/home", "raulh87", "mnt", "a471", "userdata", "raulh87", "data")
    return [base]


def candidate_roots():
    """Ordered list of result-root candidates for this machine.

    An explicit ``$DBT_RESULTS_ROOT`` always takes priority.
    """
    roots = []
    override = os.environ.get("DBT_RESULTS_ROOT")
    if override:
        roots.append(override)
    roots.extend(_windows_candidates() if os.name == "nt" else _linux_candidates())
    # De-duplicate while preserving order.
    seen, out = set(), []
    for r in roots:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _has_current_results(root, dataset):
    return os.path.isdir(os.path.join(root, dataset, "current-results"))


def resolve_datafolder(dataset, must_have_results=True):
    """Return the best result root for ``dataset``.

    If ``must_have_results`` is True, prefer the first candidate that actually
    contains ``{dataset}/current-results``. Falls back to the first existing
    candidate directory, then to the first candidate regardless of existence
    (so the UI can still show a sensible default path to edit).
    """
    cands = candidate_roots()
    if must_have_results:
        for root in cands:
            if _has_current_results(root, dataset):
                return root
    for root in cands:
        if os.path.isdir(root):
            return root
    return cands[0] if cands else ""


def describe_source(dataset):
    """Human-readable note about which root was chosen and why."""
    root = resolve_datafolder(dataset)
    if _has_current_results(root, dataset):
        kind = "results found"
    elif os.path.isdir(root):
        kind = "folder exists, no current-results yet"
    else:
        kind = "not mounted"
    label = "Google Drive" if "My Drive" in root else ("Network (P:)" if root.startswith("P:") else root)
    return f"{label} — {root}  [{kind}]"


# --- Scanning the current-results tree ------------------------------------
# Layout:  {root}/{dataset}/current-results/{modality}/{specie}/{roi_type}/
#          {specie}_{model}_z_corrected.nii.gz   (+ .xlsx / .csv table)

def _current_results_dir(datafolder, dataset, modality, specie, roi_type):
    return os.path.join(datafolder, dataset, "current-results", modality, specie, roi_type)


def scan_roi_types(datafolder, dataset, modality, specie):
    base = os.path.join(datafolder, dataset, "current-results", modality, specie)
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))


def scan_models(datafolder, dataset, modality, specie, roi_type):
    """Model names available for a species, from either the unthresholded
    ``_z.nii.gz`` map or a thresholded ``_z_corrected.nii.gz`` map."""
    d = _current_results_dir(datafolder, dataset, modality, specie, roi_type)
    prefix = f"{specie}_"
    models = set()
    for f in glob.glob(os.path.join(d, f"{specie}_*_z.nii.gz")):
        models.add(os.path.basename(f)[len(prefix):-len("_z.nii.gz")])
    for f in glob.glob(os.path.join(d, f"{specie}_*_z_corrected.nii.gz")):
        models.add(os.path.basename(f)[len(prefix):-len("_z_corrected.nii.gz")])
    return sorted(models)


def overlay_path(datafolder, dataset, modality, specie, roi_type, model_name):
    """Path to the map used for the interactive (continuous) overlay.

    Prefers the unthresholded z-map so the threshold slider can move freely;
    falls back to the cluster-corrected map (already thresholded) if the
    unthresholded one has not been mirrored yet. Returns (path, kind) where
    kind is "unthresholded" or "corrected"; path is None if neither exists.
    """
    d = _current_results_dir(datafolder, dataset, modality, specie, roi_type)
    unthr = os.path.join(d, f"{specie}_{model_name}_z.nii.gz")
    if os.path.exists(unthr):
        return unthr, "unthresholded"
    corr = os.path.join(d, f"{specie}_{model_name}_z_corrected.nii.gz")
    if os.path.exists(corr):
        return corr, "corrected"
    return None, None


def corrected_path(datafolder, dataset, modality, specie, roi_type, model_name, z_threshold=None):
    """Path to the cluster-corrected map for a given threshold (None = any)."""
    d = _current_results_dir(datafolder, dataset, modality, specie, roi_type)
    if z_threshold is not None:
        p = os.path.join(d, f"{specie}_{model_name}_zt{z_threshold}_corrected.nii.gz")
        if os.path.exists(p):
            return p
    p = os.path.join(d, f"{specie}_{model_name}_z_corrected.nii.gz")
    return p if os.path.exists(p) else None


def table_path(datafolder, dataset, modality, specie, roi_type, model_name, z_threshold=None):
    """Path to the cluster table (xlsx/csv) for a given threshold (None = any)."""
    d = _current_results_dir(datafolder, dataset, modality, specie, roi_type)
    candidates = []
    if z_threshold is not None:
        candidates += [f"{specie}_{model_name}_zt{z_threshold}.xlsx",
                       f"{specie}_{model_name}_zt{z_threshold}.csv"]
    candidates += [f"{specie}_{model_name}.xlsx", f"{specie}_{model_name}.csv"]
    for name in candidates:
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def scan_datasets():
    """Datasets that have a current-results tree on any candidate root."""
    found = set()
    for root in candidate_roots():
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            if os.path.isdir(os.path.join(root, name, "current-results")):
                found.add(name)
    return sorted(found)

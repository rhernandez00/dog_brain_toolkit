#!/usr/bin/env python
"""Summarise a whole RSA model battery from its group maps.

Reads the model registry (``rsa_models/_models.csv``) as the list of hypotheses
to look for, then walks the step-3/7/9/10 outputs on disk and turns them into
one per-map table plus a set of cross-map analyses and a written report.

The registry is the source of truth for *what should exist*; disk decides what
*does*. Every model is therefore reported in one of three states -- ``z`` (the
permutation z-map is on disk), ``mean-only`` (step 3 ran, step 7 did not, so a
parametric t-map stands in) and ``missing`` -- and nothing is silently dropped.
That matters because an absent file and a null result look identical in a plain
directory listing, and only one of them is a finding.

What it computes
----------------
per-map      z / tau / t descriptives inside the searchlight mask, uncorrected
             suprathreshold volume, the cluster-corrected outcome (from the
             step-9 sidecar or recomputed from the map), and the labelled peak.
ranking      models ordered by corrected volume, per species.
species      rank agreement between dogs and humans across the shared
             hypotheses -- "do the same models win in both species?".
similarity   model x model correlation of the z-maps inside one species, with
             an average-linkage ordering, so redundant hypotheses are visible.
methods      the nine hypotheses that exist in both the mahalanobis/stim-wise
             and the correlation/run-wise family, compared map against map, and
             their model RDMs compared after collapsing the run-wise 40x40 to
             the stim-wise 10x10 (a check that the pair really is one
             hypothesis, not two).
groupings    within vs cross vs dog vs hum for each family that offers them --
             the species-general claim.
conjunction  per species, a NIfTI counting how many models call each voxel
             significant, plus how distinctive each model's own cluster is.
calibration  in-mask mean/SD of every z-map and its negative-tail volume. A
             permutation z-map should be ~N(0,1) under the null; systematic
             departures are a pipeline problem, not a result.

Outputs land in ``--out_dir`` (default ``tools/zmap_summary_out/{dataset}``):
CSVs for every table, the conjunction NIfTIs, PNG figures, and
``report.md`` tying them together.

Usage
-----
  # everything, both species, default parameters
  python tools/zmap_summary.py

  # one distance method, no figures (faster)
  python tools/zmap_summary.py --dis_method mahalanobis --no_figures

  # a different threshold, dogs only
  python tools/zmap_summary.py --species D --z_threshold 2.3

Reading ~500 maps off the network disk is the slow part; it is threaded
(``--workers``) and the whole run is well under an hour on EmoC.
"""
import argparse
import contextlib
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import nibabel as nib

# Resolve the repo root by walking up until scheduler/paths.py appears, then put
# both this folder and the root on sys.path -- the tools/ convention.
_here = os.path.dirname(os.path.abspath(__file__))
_root = _here
while _root != os.path.dirname(_root):
    if os.path.exists(os.path.join(_root, 'scheduler', 'paths.py')):
        break
    _root = os.path.dirname(_root)
for _p in (_here, _root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scheduler.paths import get_paths  # noqa: E402
import models_manifest as mm  # noqa: E402

# ---------------------------------------------------------------------------
# Fixed knowledge about the two species. Radius and atlas differ; everything
# else in the pipeline is shared.
# ---------------------------------------------------------------------------
SPECIES_DEFAULTS = {
    'D': {'radius': 3, 'label': 'Dog', 'atlas_for_labels': 'Czeibert', 'atlas_type': 'Nitzsche'},
    'H': {'radius': 4, 'label': 'Hum', 'atlas_for_labels': 'AAL', 'atlas_type': 'MNI'},
}

# *Candidate* pairs: hypotheses that look like the same claim under two names in
# the two model families (mahalanobis/stim-wise works on a 10x10 condition RDM,
# correlation/run-wise on the 40x40 exemplar RDM of the same conditions). The
# names are a guess; the `rdm_r` column in the method-comparison table is what
# decides. On EmoC it refutes one of them -- action_tendency puts anger with
# approach, approach-avoid puts it with avoid -- so read rdm_r before reading
# map_r for any pair.
METHOD_PAIRS = [
    ('emotion_identity', 'emo-id'),
    ('valence3', 'val3'),
    ('valence_binary_emotional', 'val-bin'),
    ('emotionality', 'emo-vs-neu'),
    ('negative_vs_other', 'threat'),
    ('action_tendency', 'approach-avoid'),
    ('valence_bipolar', 'grad-val'),
    ('arousal_graded', 'grad-arousal'),
    ('species_identity', 'agent-species-id'),
]
# 'all' (stim-wise) and 'collapse' (run-wise) are the same scope under two names.
GROUPING_ALIASES = {'all': 'collapse', 'collapse': 'all'}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ts(t):
    return time.strftime('%Y-%m-%d %H:%M', time.localtime(t)) if t else None


def _days(older, newer):
    """How many days ``newer`` lags behind ``older`` (positive = out of date)."""
    if not older or not newer:
        return None
    return round((older - newer) / 86400.0, 2)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def rsa_dir(datafolder, dataset, glm_model, rsa_model):
    return os.path.join(datafolder, dataset, 'results', 'RSA', glm_model, rsa_model, 'mean')


def rnd_dir(datafolder, dataset, glm_model):
    return os.path.join(datafolder, dataset, 'results', 'RSA_rnd', glm_model)


def map_stem(mask_type, specie, radius, dis_method, rsa_method):
    """The filename prefix every step-3/7/9/10 output for one map shares."""
    core = f"{specie}-r-{radius}_{dis_method}_{rsa_method}"
    return f"{mask_type}-{core}" if mask_type else core


def mask_path(datafolder, dataset, specie, mask_type):
    """Where the searchlight mask lives -- the Atlas tree for dogs, the dataset's
    own ROI folder for humans, exactly as ``searchlight.py`` resolves it."""
    if specie == 'D':
        return os.path.join(_root, 'Atlas', 'Dog', SPECIES_DEFAULTS['D']['atlas_type'],
                            mask_type + '.nii.gz')
    return os.path.join(datafolder, dataset, 'ROI', specie, mask_type + '.nii.gz')


def load_label_atlas(specie):
    """(label_data, label_affine, label_dict) for peak naming, or (None,)*3."""
    try:
        if specie == 'D':
            img = nib.load(os.path.join(_root, 'Atlas', 'Dog', 'Nitzsche',
                                        'Czeibert_labels2mm.nii.gz'))
            dic = pd.read_csv(os.path.join(_root, 'Atlas', 'Dog', 'Czeibert_dictionary.csv'))
        else:
            img = nib.load(os.path.join(_root, 'Atlas', 'Hum', 'AAL3.nii.gz'))
            dic = pd.read_csv(os.path.join(_root, 'Atlas', 'Hum', 'AAL_dictionary.csv'))
        return img.get_fdata(), img.affine, dic
    except Exception as e:
        log(f"  WARNING: no label atlas for {specie} ({e.__class__.__name__}: {e})")
        return None, None, None


# ---------------------------------------------------------------------------
# Inventory -- one scandir per model folder, never one stat per file. On a
# network mount a stat costs ~56 ms; a scandir returns every name at once.
# ---------------------------------------------------------------------------
def build_inventory(args, datafolder):
    dirs = mm.rsa_models_dirs(datafolder, args.dataset)
    rows = mm.load_rows(dirs)
    if not rows:
        raise SystemExit(f"No _models.csv found under {dirs}")

    wanted = []
    for r in rows:
        if args.dis_method and r['dis_method'] != args.dis_method:
            continue
        for g in r['groupings']:
            wanted.append({
                'dis_method': r['dis_method'],
                'mah_fold': r['mah_fold'],
                'stem': r['model'],
                'grouping': g,
                'why': r['why'],
                'rsa_model': mm.concrete_model_name(dirs, r['model'], g),
            })
    log(f"registry: {len(rows)} hypotheses -> {len(wanted)} runnable models")

    # One listing per model folder, in parallel: the folders are independent and
    # the cost is latency, not bandwidth.
    def listing(w):
        d = rsa_dir(datafolder, args.dataset, args.model, w['rsa_model'])
        try:
            return w['rsa_model'], {e.name: (e.stat().st_size, e.stat().st_mtime)
                                    for e in os.scandir(d)}
        except OSError:
            return w['rsa_model'], {}

    uniq = {w['rsa_model']: w for w in wanted}.values()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        listings = dict(ex.map(listing, uniq))

    # The permutation null lives in a single flat folder -- one listing covers
    # every model and species.
    rd = rnd_dir(datafolder, args.dataset, args.model)
    try:
        rnd_names = {e.name: (e.stat().st_size, e.stat().st_mtime)
                     for e in os.scandir(rd) if e.is_file()}
    except OSError:
        rnd_names = {}
        log(f"WARNING: no RSA_rnd folder at {rd}; null diagnostics unavailable")

    inv = []
    for w in wanted:
        names = listings.get(w['rsa_model'], {})
        for specie in args.species:
            radius = SPECIES_DEFAULTS[specie]['radius']
            stem = map_stem(args.mask_type, specie, radius, w['dis_method'], args.rsa_method)
            zt = args.z_threshold
            files = {
                'mean': f"{stem}_mean.nii.gz",
                'std': f"{stem}_std.nii.gz",
                'mean_json': f"{stem}_mean.json",
                'z': f"{stem}_z.nii.gz",
                'corrected': f"{stem}_zt{zt}_corrected.nii.gz",
                'corrected_json': f"{stem}_zt{zt}_corrected.json",
                'table': f"{stem}_zt{zt}.csv",
            }
            d = rsa_dir(datafolder, args.dataset, args.model, w['rsa_model'])
            rec = dict(w)
            rec.update({'specie': specie, 'radius': radius, 'dir': d})
            for k, fn in files.items():
                # A zero-byte file is a half-written one: treat it as absent.
                size, mtime = names.get(fn, (0, None))
                rec[f'path_{k}'] = os.path.join(d, fn)
                rec[f'has_{k}'] = size > 0
                rec[f'mtime_{k}'] = mtime
            for k, fn in (('null_mean', f"{specie}-{w['rsa_model']}_mean.nii.gz"),
                          ('null_std', f"{specie}-{w['rsa_model']}_std.nii.gz")):
                size, mtime = rnd_names.get(fn, (0, None))
                rec[f'path_{k}'] = os.path.join(rd, fn)
                rec[f'has_{k}'] = size > 0
                rec[f'mtime_{k}'] = mtime
            inv.append(rec)
    return inv


# ---------------------------------------------------------------------------
# Per-map statistics
# ---------------------------------------------------------------------------
def _n_participants(path):
    """Participant count and availability from the step-3 sidecar."""
    try:
        import yaml
        with open(path) as f:
            d = yaml.safe_load(f)
        return len(d.get('file_list') or []), d.get('perc_available')
    except Exception:
        return np.nan, np.nan


def _read_vec(path, mask_bool):
    """In-mask float32 vector of a map, non-finite mapped to 0."""
    arr = np.asarray(nib.load(path).dataobj, dtype=np.float32)
    if arr.shape != mask_bool.shape:
        raise ValueError(f"{os.path.basename(path)} is {arr.shape}, mask is {mask_bool.shape}")
    v = arr[mask_bool]
    v[~np.isfinite(v)] = 0.0
    return v


def describe(vec, zt=None):
    """The summary of one statistical map inside the mask.

    ``zt`` only makes sense for a z-scaled map; pass None for the tau maps,
    where "voxels above 3.1" would be a column of zeros pretending to mean
    something (Kendall tau is bounded by 1).
    """
    n = vec.size
    if n == 0:
        return {}
    d = {
        'n_in_mask': int(n),
        'n_nonzero': int(np.count_nonzero(vec)),
        'mean': float(vec.mean()),
        'sd': float(vec.std(ddof=1)),
        'max': float(vec.max()),
        'min': float(vec.min()),
        'p99': float(np.percentile(vec, 99)),
        'p95': float(np.percentile(vec, 95)),
    }
    if zt is not None:
        d['n_supra'] = int((vec >= zt).sum())
        d['n_neg_supra'] = int((vec <= -zt).sum())
    return d


def robust_z(zvec):
    """Re-express a z-map against the rest of the same brain.

    The pipeline's z asks "is tau here bigger than a model-permuted null?",
    voxel by voxel and independently. That is the right question only if the
    null removes everything the model shares with the whole brain. When the real
    group tau carries a *global* positive offset -- the model fitting every
    searchlight a little -- permuting the model cannot remove it, and z comes
    out positive nearly everywhere, which cluster correction then blesses as one
    brain-sized cluster.

    ``zr = (z - median(z)) / (1.4826 * MAD(z))`` asks the localisation question
    instead: does this voxel stand out from the rest of *this* brain for *this*
    model. Median/MAD rather than mean/SD so a genuine cluster does not set the
    baseline it is measured against. It is a descriptive re-centring, not a
    second significance test -- there is no null behind it, so read it as a
    ranking, not as a p-value.
    """
    med = float(np.median(zvec))
    mad = float(np.median(np.abs(zvec - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= 0:
        return None, med, np.nan
    return (zvec - med) / scale, med, scale


def load_species_maps(inv, specie, args, mask_bool, mask_affine=None, out_dir=None):
    """Load every available z / mean / std map for one species.

    Returns (stats_rows, z_vectors, tau_vectors). Vectors are in-mask float32,
    keyed by rsa_model, and are what every cross-map analysis runs on.
    """
    recs = [r for r in inv if r['specie'] == specie]
    zt = args.z_threshold
    zr_dir = os.path.join(out_dir, 'zr', specie) if (out_dir and args.write_zr) else None
    if zr_dir:
        os.makedirs(zr_dir, exist_ok=True)

    def one(rec):
        out = dict(rec)
        out.pop('dir', None)
        zvec = tauvec = None
        try:
            if rec['has_mean']:
                tauvec = _read_vec(rec['path_mean'], mask_bool)
                for k, v in describe(tauvec).items():
                    out[f'tau_{k}'] = v
            if rec['has_z']:
                zvec = _read_vec(rec['path_z'], mask_bool)
                for k, v in describe(zvec, zt).items():
                    out[f'z_{k}'] = v
                out['z_frac_positive'] = float((zvec > 0).mean())
                # Percentages, not counts, whenever dogs and humans are put side
                # by side: the human mask is ~19x the dog mask, so a raw voxel
                # count compares brain sizes as much as it compares effects.
                out['z_pct_supra'] = 100.0 * (zvec >= zt).mean()
                zr, med, scale = robust_z(zvec)
                out['zr_median'] = med
                out['zr_scale'] = scale
                if zr is not None:
                    out['zr_max'] = float(zr.max())
                    out['zr_n_supra'] = int((zr >= zt).sum())
                    out['zr_pct_supra'] = 100.0 * (zr >= zt).mean()
                    out['zr_n_neg_supra'] = int((zr <= -zt).sum())
                    if zr_dir:
                        vol = np.zeros(mask_bool.shape, dtype=np.float32)
                        vol[mask_bool] = zr
                        nib.save(nib.Nifti1Image(vol, mask_affine),
                                 os.path.join(zr_dir, f"{rec['rsa_model']}_zr.nii.gz"))
            # The two halves of the z: how far the whole brain has shifted, and
            # how wide the permutation null is. A z-map that is uniformly
            # positive is these two numbers, not a localisation.
            if rec['has_null_mean'] and rec['has_null_std']:
                nm = _read_vec(rec['path_null_mean'], mask_bool)
                ns = _read_vec(rec['path_null_std'], mask_bool)
                out['null_mean_mean'] = float(nm.mean())
                out['null_sd_mean'] = float(ns[ns > 0].mean()) if (ns > 0).any() else np.nan
                out['null_sd_zero_vox'] = int((ns <= 0).sum())
                if tauvec is not None and np.isfinite(out['null_sd_mean']):
                    out['tau_global_offset'] = float(tauvec.mean())
                    out['z_from_offset'] = float(
                        (tauvec.mean() - out['null_mean_mean']) / out['null_sd_mean'])
            if rec['has_mean_json']:
                n, perc = _n_participants(rec['path_mean_json'])
                out['n_participants'], out['perc_available'] = n, perc
            # Parametric stand-in. The pipeline writes the across-participant SD
            # with ddof=0, so t = mean * sqrt(n-1) / sd0. It is not the
            # permutation test and is only reported as a fallback ranking for
            # models whose step 7 has not run.
            if rec['has_std'] and tauvec is not None and out.get('n_participants', 0) > 1:
                sd0 = _read_vec(rec['path_std'], mask_bool)
                n = float(out['n_participants'])
                with np.errstate(divide='ignore', invalid='ignore'):
                    t = tauvec * np.sqrt(n - 1.0) / sd0
                t[~np.isfinite(t)] = 0.0
                out['t_max'] = float(t.max())
                out['t_n_supra'] = int((t >= zt).sum())
            if rec['has_corrected_json']:
                with open(rec['path_corrected_json']) as f:
                    cj = json.load(f)
                out['corr_n_clusters'] = cj.get('n_clusters')
                out['corr_n_voxels'] = cj.get('n_voxels')
                out['corr_min_cluster_size'] = cj.get('minimal_cluster_size')
                out['corr_empty'] = cj.get('empty')
            if rec['has_corrected']:
                carr = np.asarray(nib.load(rec['path_corrected']).dataobj, dtype=np.float32)
                cvec = carr[mask_bool]
                cvec[~np.isfinite(cvec)] = 0.0
                out['corr_n_voxels_map'] = int((cvec > 0).sum())
                out['corr_peak_z'] = float(cvec.max()) if cvec.size else np.nan
                if out.get('corr_n_clusters') is None and out['corr_n_voxels_map']:
                    # Runs that predate the step-9 sidecar still have the map;
                    # 26-connectivity, the same structure apply_cluster_correction
                    # used, so the count matches what step 9 would have written.
                    from scipy.ndimage import label as _lab, generate_binary_structure as _gbs
                    _, nc = _lab(np.isfinite(carr) & (carr > 0) & mask_bool, structure=_gbs(3, 3))
                    out['corr_n_clusters'] = int(nc)
                if 'corr_n_voxels' not in out:
                    # Older runs predate the sidecar; the map still states the
                    # result, so read it off the map rather than reporting a gap.
                    out['corr_n_voxels'] = out['corr_n_voxels_map']
                    out['corr_empty'] = out['corr_n_voxels_map'] == 0
        except Exception as e:
            out['error'] = f"{e.__class__.__name__}: {e}"
        return out, zvec, tauvec

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(one, recs))

    stats, zvecs, tauvecs = [], {}, {}
    for out, zvec, tauvec in results:
        stats.append(out)
        if zvec is not None:
            zvecs[out['rsa_model']] = zvec
        if tauvec is not None:
            tauvecs[out['rsa_model']] = tauvec

    for s in stats:
        s['state'] = ('z' if s['has_z'] else 'mean-only' if s['has_mean'] else 'missing')
    return stats, zvecs, tauvecs


# ---------------------------------------------------------------------------
# Peak labelling
# ---------------------------------------------------------------------------
def label_peaks(inv, specie, args, out_dir):
    """Top labelled peak per corrected map, reusing the step-10 table when it
    exists and recomputing from the map when it does not."""
    import rsa_utils
    ldata, laff, ldict = load_label_atlas(specie)
    rows = []
    for rec in [r for r in inv if r['specie'] == specie and r['has_corrected']]:
        got = None
        if rec['has_table']:
            try:
                t = pd.read_csv(rec['path_table'])
                if len(t) and 'subpeak_Z' in t.columns:
                    t = t.sort_values('subpeak_Z', ascending=False)
                    got = [{
                        'cluster_id': int(r0.cluster_id), 'size_vox': int(r0.cluster_size_vox),
                        'peak_Z': float(r0.subpeak_Z), 'region': r0.region,
                        'x_mm': r0.subpeak_x_mm, 'y_mm': r0.subpeak_y_mm, 'z_mm': r0.subpeak_z_mm,
                        'source': 'step10',
                    } for r0 in t.head(args.max_peaks).itertuples()]
            except Exception:
                got = None
        if got is None:
            try:
                # extract_clusters_and_peaks narrates every peak it labels; the
                # narration is for a pipeline log, not for this table.
                with contextlib.redirect_stdout(io.StringIO()):
                    res = rsa_utils.extract_clusters_and_peaks(
                        rec['path_corrected'], stat_thresh=None,
                        min_dist_mm=args.min_dist_mm, max_peaks_per_cluster=1,
                        label_dict=ldict, label_nii_data=ldata, label_affine=laff)
                res = sorted(res, key=lambda c: -(c['peak_Z'] or 0))[:args.max_peaks]
                got = [{
                    'cluster_id': c['cluster_id'], 'size_vox': c['size_vox'],
                    'peak_Z': c['peak_Z'],
                    'region': (c['peaks'][0].get('region') if c['peaks'] else None),
                    'x_mm': c['peak_xyz_mm'][0] if c['peak_xyz_mm'] is not None else None,
                    'y_mm': c['peak_xyz_mm'][1] if c['peak_xyz_mm'] is not None else None,
                    'z_mm': c['peak_xyz_mm'][2] if c['peak_xyz_mm'] is not None else None,
                    'source': 'recomputed',
                } for c in res]
            except Exception as e:
                got = [{'error': f"{e.__class__.__name__}: {e}", 'source': 'failed'}]
        for g in got:
            rows.append({'specie': specie, 'rsa_model': rec['rsa_model'],
                         'dis_method': rec['dis_method'], 'stem': rec['stem'],
                         'grouping': rec['grouping'], **g})
    return rows


# ---------------------------------------------------------------------------
# Cross-map analyses
# ---------------------------------------------------------------------------
def similarity_matrix(vecs, keep):
    """Model x model Pearson r over the analysis voxels, plus a linkage order."""
    names = sorted(vecs)
    if len(names) < 2:
        return None, names
    M = np.vstack([vecs[n][keep] for n in names])
    M = M - M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1)
    sd[sd == 0] = 1.0
    M = M / sd[:, None]
    R = (M @ M.T) / M.shape[1]
    np.fill_diagonal(R, 1.0)
    return pd.DataFrame(np.clip(R, -1, 1), index=names, columns=names), names


def linkage_order(R):
    """Average-linkage leaf order on 1-r, so the heatmap groups redundant models."""
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
        D = 1.0 - R.values
        np.fill_diagonal(D, 0.0)
        D = (D + D.T) / 2.0
        Z = linkage(squareform(D, checks=False), method='average')
        return [R.index[i] for i in leaves_list(Z)], Z
    except Exception:
        return list(R.index), None


def species_agreement(per_map, metric):
    """Paired D/H values of one metric across the hypotheses both species ran."""
    key = ['dis_method', 'stem', 'grouping']
    d = per_map[per_map.specie == 'D'].set_index(key)[metric]
    h = per_map[per_map.specie == 'H'].set_index(key)[metric]
    both = pd.concat([d.rename('D'), h.rename('H')], axis=1).dropna()
    if len(both) < 3:
        return both, np.nan, np.nan
    from scipy.stats import spearmanr, pearsonr
    rho = spearmanr(both.D, both.H).statistic
    r = pearsonr(both.D, both.H).statistic
    return both, float(rho), float(r)


def collapse_runwise_rdm(df):
    """Average a 40x40 run-wise model RDM down to the 10x10 condition RDM, so it
    can be compared with its stim-wise twin. Labels are ``DogP1..DogP4``; the
    condition is the label minus its trailing exemplar digit."""
    labels = [str(c) for c in df.columns]
    cond = [re.sub(r'\d+$', '', l) for l in labels]
    order, seen = [], set()
    for c in cond:
        if c not in seen:
            seen.add(c)
            order.append(c)
    A = df.values.astype(float)
    out = np.full((len(order), len(order)), np.nan)
    idx = {c: [i for i, x in enumerate(cond) if x == c] for c in order}
    for a, ca in enumerate(order):
        for b, cb in enumerate(order):
            block = A[np.ix_(idx[ca], idx[cb])]
            if a == b:
                # The diagonal block holds within-condition, between-exemplar
                # pairs; its own diagonal is the self-comparison and is not a
                # modelled dissimilarity.
                m = ~np.eye(block.shape[0], dtype=bool)
                block = block[m]
            vals = block[np.isfinite(block)]
            if vals.size:
                out[a, b] = vals.mean()
    return pd.DataFrame(out, index=order, columns=order)


def compare_model_rdms(datafolder, dataset, stim_name, run_name):
    """Correlate a stim-wise model RDM with its collapsed run-wise twin."""
    dirs = mm.rsa_models_dirs(datafolder, dataset)
    def find(n):
        for d in dirs:
            p = os.path.join(d, n + '.csv')
            if os.path.isfile(p):
                return p
        return None
    ps, pr = find(stim_name), find(run_name)
    if not ps or not pr:
        return None
    A = pd.read_csv(ps, index_col=0)
    B = collapse_runwise_rdm(pd.read_csv(pr, index_col=0))
    common = [c for c in A.index if c in B.index]
    if len(common) < 3:
        return None
    a = A.loc[common, common].values.astype(float)
    b = B.loc[common, common].values.astype(float)
    iu = np.triu_indices(len(common), k=1)
    x, y = a[iu], b[iu]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return {'n_pairs': int(ok.sum()), 'r': np.nan, 'identical': bool(np.allclose(x[ok], y[ok]))}
    return {'n_pairs': int(ok.sum()),
            'r': float(np.corrcoef(x[ok], y[ok])[0, 1]),
            'identical': bool(np.allclose(x[ok], y[ok]))}


def conjunction(inv, specie, mask_bool, args, out_dir):
    """Count, per voxel, how many models call it significant after correction."""
    recs = [r for r in inv if r['specie'] == specie and r['has_corrected']]
    if not recs:
        return None, pd.DataFrame()
    ref = nib.load(recs[0]['path_corrected'])
    count = np.zeros(mask_bool.shape, dtype=np.int16)
    binmaps = {}
    for rec in recs:
        try:
            a = np.asarray(nib.load(rec['path_corrected']).dataobj, dtype=np.float32)
            b = np.isfinite(a) & (a > 0) & mask_bool
            if b.any():
                count += b.astype(np.int16)
                binmaps[rec['rsa_model']] = b[mask_bool]
        except Exception as e:
            log(f"  conjunction: skipped {rec['rsa_model']} ({e.__class__.__name__})")
    path = os.path.join(out_dir, f'conjunction_{specie}.nii.gz')
    nib.save(nib.Nifti1Image(count, ref.affine), path)

    # How much of each model's own cluster is its own? A model whose voxels are
    # all shared with a dozen others is not adding an independent finding.
    cvec = count[mask_bool]
    rows = []
    for name, b in binmaps.items():
        n = int(b.sum())
        overl = cvec[b]
        rows.append({'specie': specie, 'rsa_model': name, 'n_sig': n,
                     'mean_n_models_sharing': float(overl.mean()),
                     'frac_unique': float((overl <= 1).mean()),
                     'frac_shared_ge3': float((overl >= 3).mean())})
    return path, pd.DataFrame(rows).sort_values('n_sig', ascending=False)


def dice_matrix(inv, specie, mask_bool):
    """Dice overlap between the corrected maps that actually found something."""
    recs = [r for r in inv if r['specie'] == specie and r['has_corrected']]
    bins = {}
    for rec in recs:
        try:
            a = np.asarray(nib.load(rec['path_corrected']).dataobj, dtype=np.float32)
            b = (np.isfinite(a) & (a > 0))[mask_bool]
            if b.sum() > 0:
                bins[rec['rsa_model']] = b
        except Exception:
            pass
    names = sorted(bins)
    if len(names) < 2:
        return pd.DataFrame()
    B = np.vstack([bins[n] for n in names])
    inter = B.astype(np.float32) @ B.astype(np.float32).T
    sizes = B.sum(axis=1).astype(np.float32)
    denom = sizes[:, None] + sizes[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        D = 2.0 * inter / denom
    D[~np.isfinite(D)] = 0.0
    return pd.DataFrame(D, index=names, columns=names)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(out_dir, per_map, sim, order, agree, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Drop last run's figures first. Figure filenames encode which metric they
    # plot, so a renamed metric would otherwise leave an orphan PNG that the
    # report no longer links and nobody knows is stale.
    for f in os.listdir(out_dir):
        if f.startswith('fig_') and f.endswith('.png'):
            try:
                os.remove(os.path.join(out_dir, f))
            except OSError:
                pass

    made = []
    for specie in args.species:
        d = per_map[(per_map.specie == specie) & per_map.corr_n_voxels.notna()]
        d = d[d.corr_n_voxels > 0].sort_values('corr_n_voxels')
        if len(d) == 0:
            continue
        fig, ax = plt.subplots(figsize=(9, max(3, 0.28 * len(d) + 1.2)))
        colors = ['#3b7dd8' if m == 'mahalanobis' else '#d8733b' for m in d.dis_method]
        ax.barh(d.rsa_model, d.corr_n_voxels, color=colors)
        ax.set_xscale('log')
        ax.set_xlabel(f'cluster-corrected voxels (z>={args.z_threshold}, log scale)')
        ax.set_title(f'{args.dataset} {specie}: models with surviving clusters')
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ('#3b7dd8', '#d8733b')]
        ax.legend(handles, ['mahalanobis / stim-wise', 'correlation / run-wise'],
                  fontsize=8, loc='lower right')
        fig.tight_layout()
        p = os.path.join(out_dir, f'fig_ranking_{specie}.png')
        fig.savefig(p, dpi=140)
        plt.close(fig)
        made.append(p)

        R = sim.get(specie)
        if R is not None and len(R) > 1:
            o = order[specie]
            Rm = R.loc[o, o]
            fig, ax = plt.subplots(figsize=(max(6, 0.16 * len(o) + 3),) * 2)
            im = ax.imshow(Rm.values, vmin=-1, vmax=1, cmap='RdBu_r')
            ax.set_xticks(range(len(o)))
            ax.set_xticklabels(o, rotation=90, fontsize=5)
            ax.set_yticks(range(len(o)))
            ax.set_yticklabels(o, fontsize=5)
            ax.set_title(f'{specie}: z-map similarity (Pearson r, in-mask), clustered')
            fig.colorbar(im, ax=ax, shrink=0.7)
            fig.tight_layout()
            p = os.path.join(out_dir, f'fig_similarity_{specie}.png')
            fig.savefig(p, dpi=160)
            plt.close(fig)
            made.append(p)

    # The offset diagnostic: how much of each z-map is a whole-brain shift.
    if 'z_from_offset' in per_map.columns:
        d = per_map[per_map.state == 'z'].dropna(subset=['z_from_offset'])
        if len(d):
            fig, ax = plt.subplots(figsize=(7, 5))
            for specie, c in zip(args.species, ('#3b7dd8', '#d8733b')):
                s = d[d.specie == specie]
                ax.scatter(s.z_from_offset, s.z_pct_supra.clip(lower=0.01), s=26,
                           alpha=0.75, color=c, label=specie)
            ax.axvline(args.z_threshold, ls='--', lw=1, color='crimson')
            ax.axvline(-args.z_threshold, ls='--', lw=1, color='crimson')
            ax.set_yscale('log')
            ax.set_xlabel('z the whole brain gets from the global tau offset alone')
            ax.set_ylabel(f'% of mask with z >= {args.z_threshold} (log)')
            ax.set_title('Suprathreshold volume is largely explained by the offset')
            ax.legend()
            fig.tight_layout()
            p = os.path.join(out_dir, 'fig_global_offset.png')
            fig.savefig(p, dpi=150)
            plt.close(fig)
            made.append(p)

    for metric, both in agree.items():
        if both is None or len(both) < 3:
            continue
        fig, ax = plt.subplots(figsize=(5.6, 5.4))
        ax.scatter(both.D, both.H, s=26, alpha=0.8, color='#3b7dd8')
        for (dm, st, g), r in both.iterrows():
            ax.annotate(f'{st}__{g}', (r.D, r.H), fontsize=4.5,
                        xytext=(3, 2), textcoords='offset points')
        ax.set_xlabel(f'Dog {metric}')
        ax.set_ylabel(f'Human {metric}')
        ax.set_title(f'{metric}: dog vs human, one point per hypothesis')
        fig.tight_layout()
        p = os.path.join(out_dir, f'fig_species_{metric}.png')
        fig.savefig(p, dpi=150)
        plt.close(fig)
        made.append(p)
    return made


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(v):
    """Voxel counts are counts: print them as integers, not as ``2.58e+03``.
    Everything else gets three significant digits."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ''
    if isinstance(v, (int, np.integer)):
        return f'{int(v):,}'
    if isinstance(v, (float, np.floating)):
        if float(v).is_integer() and abs(v) < 1e12:
            return f'{int(v):,}'
        return f'{v:.3g}'
    return str(v)


def md_table(df, cols=None, n=None):
    d = df if cols is None else df[cols]
    if n:
        d = d.head(n)
    if len(d) == 0:
        return '_(nothing to show)_\n'
    d = d.copy()
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].map(_fmt)
    head = '| ' + ' | '.join(str(c) for c in d.columns) + ' |'
    sep = '| ' + ' | '.join('---' for _ in d.columns) + ' |'
    body = '\n'.join(
        '| ' + ' | '.join('' if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
                          for v in row) + ' |'
        for row in d.itertuples(index=False))
    return f'{head}\n{sep}\n{body}\n'


def write_report(out_dir, args, per_map, peaks, sim, order, agree, method_cmp,
                 grouping_tbl, conj_tbl, calib, provenance, figures, mask_sizes,
                 mask_notes, elapsed):
    L = []
    A = L.append
    A(f"# {args.dataset} RSA battery — z-map summary\n")
    A(f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `tools/zmap_summary.py` "
      f"in {elapsed/60:.1f} min.\n")
    A(f"GLM model `{args.model}` · mask `{args.mask_type}` · rsa_method "
      f"`{args.rsa_method}` · z threshold **{args.z_threshold}** · "
      f"radius D=3 / H=4.\n")
    A("Searchlight mask: " + ', '.join(f"{s} {n:,} voxels" for s, n in mask_sizes.items())
      + ". Binarised as `!= 0`, matching `rsa_utils`' own `.astype(bool)`.\n")
    for _s, _n in mask_notes.items():
        if _n:
            A(f"\n> **Mask warning ({_s}):** {_n:,} of those voxels hold `-1`, not `1`. "
              f"The pipeline's `.astype(bool)` counts them as in-mask, so they are "
              f"included here too, but a searchlight mask is meant to be 0/1 and this "
              f"one is not.\n")

    # ---- headline ---------------------------------------------------------
    A("\n## 0. Headline\n")
    for specie in args.species:
        d = per_map[per_map.specie == specie]
        n_tot, n_z = len(d), int((d.state == 'z').sum())
        n_corr = int(d.has_corrected.sum())
        hit = d[d.corr_n_voxels.fillna(0) > 0]
        best = hit.sort_values('corr_n_voxels', ascending=False).head(3)
        A(f"\n**{specie}** — {n_tot} models in the registry; {n_z} have a z-map, "
          f"{n_corr} reached cluster correction, **{len(hit)} survived it**.\n")
        for r in best.itertuples():
            A(f"  - `{r.rsa_model}` — {_fmt(r.corr_n_voxels)} voxels in "
              f"{_fmt(r.corr_n_clusters)} cluster(s), peak z = {_fmt(r.corr_peak_z)}"
              + (f", but its whole-brain offset alone is worth z = "
                 f"{_fmt(r.z_from_offset)}" if 'z_from_offset' in per_map.columns
                 and pd.notna(getattr(r, 'z_from_offset', np.nan))
                 and abs(r.z_from_offset) >= args.z_threshold * 0.5 else "") + "\n")
        if len(hit) == 0:
            A("  - nothing survived correction in this species.\n")
        frac = d[d.state == 'z']
        if 'z_from_offset' in frac.columns and len(frac):
            n_off = int((frac.z_from_offset.abs() >= args.z_threshold * 0.5).sum())
            if n_off:
                A(f"  - ⚠ {n_off} of {len(frac)} z-maps carry a whole-brain "
                  f"offset worth at least half the threshold on its own "
                  f"(see §2c).\n")

    A("\n## 1. Coverage\n")
    cov = (per_map.groupby(['specie', 'dis_method', 'state']).size()
           .rename('n').reset_index().pivot_table(index=['specie', 'dis_method'],
                                                  columns='state', values='n', fill_value=0)
           .reset_index())
    A(md_table(cov))
    A("\n`z` = permutation z-map on disk · `mean-only` = step 3 ran but step 7 "
      "did not (a parametric t-map is reported instead) · `missing` = no group "
      "map at all.\n")
    pipe = (per_map.assign(corrected=per_map.has_corrected, table=per_map.has_table)
            .groupby(['specie', 'dis_method'])[['has_mean', 'has_z', 'corrected', 'table']]
            .sum().reset_index())
    A("\nHow far each family got down the pipeline (step 3 / step 7 / step 9 / step 10):\n\n")
    A(md_table(pipe))
    miss = per_map[per_map.state == 'missing'][['specie', 'dis_method', 'rsa_model']]
    if len(miss):
        A(f"\n<details><summary>{len(miss)} missing model×species combinations</summary>\n\n")
        A(md_table(miss.sort_values(['specie', 'rsa_model'])))
        A("\n</details>\n")

    A("\n## 2. What survived cluster correction\n")
    for specie in args.species:
        d = per_map[(per_map.specie == specie)].copy()
        hit = d[(d.corr_n_voxels.fillna(0) > 0)].sort_values('corr_n_voxels', ascending=False)
        tested = int(d.has_corrected.sum())
        A(f"\n### {specie} — {len(hit)} of {tested} corrected maps are non-empty "
          f"({int((d.state == 'z').sum()) - tested} z-maps have not reached step 9)\n")
        cols = ['rsa_model', 'dis_method', 'grouping', 'corr_n_clusters', 'corr_n_voxels',
                'corr_peak_z', 'corr_min_cluster_size', 'z_max', 'z_n_supra', 'tau_max']
        A(md_table(hit, [c for c in cols if c in hit.columns]))
        # The registry's own rationale for each surviving model, so the result
        # is read against the claim it was built to test.
        if len(hit):
            A("\n**What the surviving models claim** (`why`, from `_models.csv`)\n\n")
            seen = set()
            for r in hit.itertuples():
                if r.stem in seen:
                    continue
                seen.add(r.stem)
                A(f"- **{r.stem}** — {r.why}\n")
        pk = peaks[(peaks.specie == specie)]
        pk = pk[pk.rsa_model.isin(hit.rsa_model)]
        if len(pk):
            A(f"\n**Labelled peaks ({specie})**\n\n")
            A(md_table(pk.sort_values('peak_Z', ascending=False),
                       [c for c in ['rsa_model', 'size_vox', 'peak_Z', 'region',
                                    'x_mm', 'y_mm', 'z_mm'] if c in pk.columns], n=40))

    A("\n## 2b. Models with no z-map — the parametric stand-in\n")
    A("For a model that finished step 3 but not step 7, the group mean and SD "
      "maps still support a one-sample t across participants "
      "(`t = tau * sqrt(n-1) / sd`, the pipeline writes SD with ddof=0). "
      "This is **not** the permutation test the pipeline reports — it assumes "
      "normality of tau across participants and carries no cluster correction — "
      "so treat it only as a ranking of which missing models are worth running.\n\n")
    mo = per_map[(per_map.state == 'mean-only') & per_map.t_max.notna()] \
        if 't_max' in per_map.columns else per_map.iloc[0:0]
    if len(mo):
        A(md_table(mo.sort_values('t_n_supra', ascending=False),
                   [c for c in ['specie', 'rsa_model', 'dis_method', 'n_participants',
                                'tau_max', 't_max', 't_n_supra'] if c in mo.columns], n=40))
    else:
        A("_(every model with a mean map also has a z-map)_\n")

    A("\n## 2c. Global offset — read this before believing section 2\n")
    A("A permutation z-map is `(real tau − null mean) / null SD`, computed at "
      "each voxel independently. Permuting the model destroys any systematic "
      "relation, so the null centres on zero — but it cannot remove an offset "
      "the model shares with the **whole brain**. If the real group tau is "
      "positive everywhere, z is positive everywhere, and cluster correction "
      "then reports one brain-sized cluster.\n\n")
    A("`tau_global_offset` is the mean of the real group tau inside the mask; "
      "`z_from_offset` is that offset expressed in null SDs — i.e. the z you "
      "would get at a voxel with **no** local effect at all. When "
      "`z_from_offset` approaches the threshold, the map is an offset, not a "
      "localisation.\n\n")
    off_cols = ['specie', 'rsa_model', 'dis_method', 'tau_global_offset',
                'null_mean_mean', 'null_sd_mean', 'z_from_offset', 'z_mean',
                'z_frac_positive', 'z_n_supra', 'zr_n_supra']
    off = per_map[per_map.state == 'z'][[c for c in off_cols if c in per_map.columns]]
    if 'z_from_offset' in off.columns:
        off = off.reindex(off.z_from_offset.abs().sort_values(ascending=False).index)
    A(md_table(off, n=40))
    if 'z_from_offset' in per_map.columns:
        bad = per_map[(per_map.state == 'z')
                      & (per_map.z_from_offset.abs() >= args.z_threshold * 0.5)]
        A(f"\n**{len(bad)} of {int((per_map.state == 'z').sum())} z-maps have an "
          f"offset worth at least half the {args.z_threshold} threshold on its "
          f"own** — "
          + ', '.join(f"{s}: {int((bad.specie == s).sum())}" for s in args.species) + ".\n")

    A("\n## 2d. Offset-robust ranking (`zr`)\n")
    A("`zr = (z − median z) / (1.4826 · MAD z)`, computed inside the mask. It "
      "asks the localisation question — does this voxel stand out from the rest "
      "of *this* brain for *this* model — and is immune to the global offset "
      "above. There is no null behind it, so it is a **ranking, not a test**; "
      "the count below is voxels with `zr >= " + str(args.z_threshold) + "`.\n\n")
    if 'zr_n_supra' in per_map.columns:
        for specie in args.species:
            d = per_map[(per_map.specie == specie) & per_map.zr_n_supra.notna()]
            d = d.sort_values('zr_n_supra', ascending=False)
            A(f"\n### {specie}\n")
            A(md_table(d, [c for c in ['rsa_model', 'dis_method', 'grouping',
                                       'zr_n_supra', 'zr_pct_supra', 'zr_max',
                                       'z_n_supra', 'z_pct_supra',
                                       'corr_n_voxels', 'z_from_offset']
                           if c in d.columns], n=25))

    A("\n## 3. Do dogs and humans favour the same hypotheses?\n")
    A("The two species sit on different templates, so their maps cannot be "
      "compared voxel to voxel. What *can* be compared is the ranking: one "
      "point per hypothesis, its value in dogs against its value in humans. "
      "Volumes are percentages of each species' own mask — the human mask is "
      "about 19× the dog mask, so raw counts would compare brain sizes.\n")
    for metric, (both, rho, r) in agree.items():
        A(f"\n**{metric}** — {len(both)} hypotheses run in both species; "
          f"Spearman rho = {rho:.3f}, Pearson r = {r:.3f}.\n")
        if len(both):
            b = both.reset_index()
            b['ratio_H_over_D'] = b.H / b.D.replace(0, np.nan)
            A(md_table(b.sort_values('H', ascending=False), n=25))

    A("\n## 4. Which hypotheses produce the same map?\n")
    for specie in args.species:
        R = sim.get(specie)
        if R is None:
            continue
        o = order[specie]
        iu = np.triu_indices(len(R), k=1)
        vals = R.loc[o, o].values[iu]
        A(f"\n### {specie} — {len(R)} z-maps compared voxelwise inside the mask\n")
        A(f"Off-diagonal r: median {np.median(vals):.3f}, "
          f"10th–90th pct {np.percentile(vals,10):.3f} – {np.percentile(vals,90):.3f}.\n")
        pairs = [(R.index[i], R.columns[j], R.values[i, j])
                 for i, j in zip(*np.triu_indices(len(R), k=1))]
        pairs.sort(key=lambda t: -t[2])
        A("\nMost redundant pairs:\n\n")
        A(md_table(pd.DataFrame(pairs[:15], columns=['model_a', 'model_b', 'r'])))
        A("\nMost dissimilar pairs:\n\n")
        A(md_table(pd.DataFrame(pairs[-10:], columns=['model_a', 'model_b', 'r'])))

    A("\n## 5. Mahalanobis/stim-wise vs correlation/run-wise\n")
    A("Nine hypotheses *look like* the same claim under two names. `rdm_r` "
      "compares the model matrices themselves (the run-wise 40×40 collapsed to "
      "10×10) and is what decides whether the pair really is one hypothesis; "
      "`map_r` compares the resulting z-maps voxelwise. **Read `rdm_r` first** — "
      "where it is well below 1 the two models encode different claims and "
      "`map_r` is comparing apples to oranges.\n\n")
    A(md_table(method_cmp))

    A("\n## 6. Grouping: within-species vs cross-species\n")
    A("For every family that offers them. `cross` is the species-general test — "
      "only stimulus pairs that bridge dog-shown and human-shown agents.\n\n")
    A(md_table(grouping_tbl))

    A("\n## 7. How distinctive is each finding?\n")
    A("`frac_unique` is the share of a model's own surviving voxels that no "
      "other model claims. A low value means the model is re-detecting a "
      "cluster the battery finds many ways to reach.\n")
    for specie in args.species:
        c = conj_tbl.get(specie)
        if c is None or len(c) == 0:
            continue
        A(f"\n### {specie}\n")
        A(md_table(c, n=25))

    A("\n## 8. Null calibration\n")
    A("A permutation z-map should be about N(0,1) under the null, and its "
      "negative tail should be about as heavy as a null positive tail. "
      "`z_mean` far from 0, `z_sd` far from 1, or a `neg_supra` that dwarfs "
      "`n_supra` is a pipeline problem, not a result.\n\n")
    A(md_table(calib, n=200))

    A("\n## 8b. Provenance — is each z-map newer than its own inputs?\n")
    A("A z-map divides step 3's real group mean by step 6's null. Re-running "
      "either without re-running step 7 leaves a z-map that mixes two "
      "generations of the analysis, and nothing downstream notices. Positive "
      "numbers below are days out of date.\n\n"
      "A gap of hours usually means the queue is *still running* — step 6 has "
      "just landed and step 7 has not been re-run yet — rather than that "
      "anything is wrong. A gap of days is the real warning.\n\n")
    if len(provenance):
        n_stale = int(provenance.stale.sum())
        A(f"**{n_stale} of {len(provenance)} z-maps are stale** "
          + ', '.join(f"({s}: {int(provenance[(provenance.specie==s)].stale.sum())}"
                      f"/{int((provenance.specie==s).sum())})" for s in args.species)
          + ".\n\n")
        st = provenance[provenance.stale].sort_values(
            ['z_older_than_null_days', 'z_older_than_mean_days'], ascending=False)
        A(md_table(st, ['specie', 'rsa_model', 'dis_method', 'step3_mean',
                        'step6_null', 'step7_z', 'z_older_than_mean_days',
                        'z_older_than_null_days'], n=60))
        nold = provenance[provenance.null_older_than_mean_days.fillna(0) > 1]
        if len(nold):
            A(f"\nSeparately, **{len(nold)} z-maps use a null computed more than "
              f"a day before the real group mean it is divided by** — the z-map "
              f"itself may be current, but its two halves are not from the same "
              f"run of the pipeline.\n\n")
            A(md_table(nold.sort_values('null_older_than_mean_days', ascending=False),
                       ['specie', 'rsa_model', 'step3_mean', 'step6_null', 'step7_z',
                        'null_older_than_mean_days'], n=40))
    else:
        A("_(no z-maps to check)_\n")

    A("\n## 9. What is still missing\n")
    gaps = []
    for specie in args.species:
        d = per_map[per_map.specie == specie]
        gaps.append({'specie': specie,
                     'need_step_3': int((~d.has_mean).sum()),
                     'need_step_7': int((d.has_mean & ~d.has_z).sum()),
                     'need_step_9': int((d.has_z & ~d.has_corrected).sum()),
                     'need_step_10': int((d.has_corrected & ~d.has_table).sum())})
    A(md_table(pd.DataFrame(gaps)))
    A("\nQueue the gaps with, e.g.:\n\n```bash\n"
      "python tools/schedule_steps.py --steps 7,8,9,10 --dry_run\n```\n")

    if figures:
        A("\n## Figures\n")
        for f in figures:
            A(f"- ![{os.path.basename(f)}]({os.path.basename(f)})\n")

    path = os.path.join(out_dir, 'report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    return path


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='EmoC')
    ap.add_argument('--model', default='basic-block', help='GLM model folder')
    ap.add_argument('--mask_type', default='b_GreyMatter2mmB')
    ap.add_argument('--rsa_method', default='kendall')
    ap.add_argument('--dis_method', default=None,
                    help='restrict to one distance method (default: all in the registry)')
    ap.add_argument('--z_threshold', type=float, default=3.1)
    ap.add_argument('--species', nargs='+', default=['D', 'H'])
    ap.add_argument('--datafolder', default=None, help='override the machine default')
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--min_dist_mm', type=float, default=8.0)
    ap.add_argument('--max_peaks', type=int, default=5, help='clusters listed per map')
    ap.add_argument('--coverage_frac', type=float, default=0.9,
                    help='a voxel joins the analysis mask if this fraction of maps covers it')
    ap.add_argument('--no_figures', action='store_true')
    ap.add_argument('--write_zr', action='store_true',
                    help='also save the offset-robust zr maps as NIfTI (one per z-map)')
    ap.add_argument('--skip_similarity', action='store_true')
    ap.add_argument('--skip_conjunction', action='store_true')
    ap.add_argument('--skip_peaks', action='store_true')
    args = ap.parse_args()

    t0 = time.time()
    datafolder = args.datafolder or get_paths()[0]
    out_dir = args.out_dir or os.path.join(_here, 'zmap_summary_out', args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    log(f"datafolder {datafolder}")
    log(f"out_dir    {out_dir}")

    inv = build_inventory(args, datafolder)
    log(f"inventory: {len(inv)} model x species combinations")

    per_map_rows, sim, order, conj_tbl, peaks_rows = [], {}, {}, {}, []
    mask_sizes, analysis_masks, mask_notes = {}, {}, {}

    for specie in args.species:
        mp = mask_path(datafolder, args.dataset, specie, args.mask_type)
        mimg = nib.load(mp)
        mask_raw = mimg.get_fdata()
        # Match the pipeline exactly: rsa_utils binarises the searchlight mask
        # with .astype(bool), i.e. "!= 0". The EmoC human mask is not 0/1 -- it
        # carries -1 voxels, which the pipeline therefore *includes*. Using
        # "> 0" here would analyse a different mask than the one the maps were
        # computed on, so the oddity is reported rather than quietly cleaned up.
        mask_bool = mask_raw.astype(bool)
        n_neg = int((mask_raw < 0).sum())
        mask_sizes[specie] = int(mask_bool.sum())
        mask_notes[specie] = n_neg
        log(f"{specie}: mask {os.path.basename(mp)} shape {mask_bool.shape} "
            f"{mask_sizes[specie]:,} voxels"
            + (f"  (WARNING: {n_neg:,} of them are -1, not 1)" if n_neg else ""))

        log(f"{specie}: reading group maps ...")
        stats, zvecs, tauvecs = load_species_maps(inv, specie, args, mask_bool,
                                                  mimg.affine, out_dir)
        per_map_rows += stats
        log(f"{specie}: {len(zvecs)} z-maps, {len(tauvecs)} mean maps loaded")

        # Analysis voxels: those the searchlight actually reached in nearly every
        # map. Voxels that are zero everywhere would otherwise inflate every
        # between-model correlation towards agreement about nothing.
        if tauvecs:
            cover = np.zeros(mask_sizes[specie], dtype=np.int32)
            for v in tauvecs.values():
                cover += (v != 0)
            keep = cover >= args.coverage_frac * len(tauvecs)
        else:
            keep = np.ones(mask_sizes[specie], dtype=bool)
        analysis_masks[specie] = keep
        log(f"{specie}: analysis voxels {int(keep.sum()):,} / {mask_sizes[specie]:,}")

        if not args.skip_similarity and len(zvecs) > 1:
            R, _ = similarity_matrix(zvecs, keep)
            o, _ = linkage_order(R)
            sim[specie], order[specie] = R, o
            R.to_csv(os.path.join(out_dir, f'similarity_z_{specie}.csv'))
            Rt, _ = similarity_matrix(tauvecs, keep)
            if Rt is not None:
                Rt.to_csv(os.path.join(out_dir, f'similarity_tau_{specie}.csv'))
            log(f"{specie}: similarity matrices written")

        if not args.skip_conjunction:
            p, ctbl = conjunction(inv, specie, mask_bool, args, out_dir)
            conj_tbl[specie] = ctbl
            if p:
                log(f"{specie}: conjunction map -> {os.path.basename(p)}")
            D = dice_matrix(inv, specie, mask_bool)
            if len(D):
                D.to_csv(os.path.join(out_dir, f'dice_{specie}.csv'))

        if not args.skip_peaks:
            log(f"{specie}: labelling peaks ...")
            peaks_rows += label_peaks(inv, specie, args, out_dir)

    per_map = pd.DataFrame(per_map_rows)
    drop = [c for c in per_map.columns if c.startswith('path_')]
    per_map.drop(columns=drop).to_csv(os.path.join(out_dir, 'per_map.csv'), index=False)
    peaks = pd.DataFrame(peaks_rows) if peaks_rows else pd.DataFrame(
        columns=['specie', 'rsa_model', 'size_vox', 'peak_Z', 'region'])
    peaks.to_csv(os.path.join(out_dir, 'peaks.csv'), index=False)

    # --- species agreement -------------------------------------------------
    agree = {}
    for metric in ('z_pct_supra', 'zr_pct_supra', 'z_max', 'tau_max'):
        if metric in per_map.columns:
            agree[metric] = species_agreement(per_map, metric)
    pd.concat({k: v[0] for k, v in agree.items()}, names=['metric']).to_csv(
        os.path.join(out_dir, 'species_agreement.csv'))

    # --- method comparison -------------------------------------------------
    idx = per_map.set_index(['dis_method', 'stem', 'grouping', 'specie'])
    rows = []
    for stim, run in METHOD_PAIRS:
        stim_g = sorted(per_map[(per_map.stem == stim)].grouping.unique())
        for g in stim_g:
            gr = GROUPING_ALIASES.get(g, g)
            rdm = compare_model_rdms(datafolder, args.dataset,
                                     mm.concrete_model_name(
                                         mm.rsa_models_dirs(datafolder, args.dataset), stim, g),
                                     mm.concrete_model_name(
                                         mm.rsa_models_dirs(datafolder, args.dataset), run, gr))
            for specie in args.species:
                try:
                    a = idx.loc[('mahalanobis', stim, g, specie)]
                    b = idx.loc[('correlation', run, gr, specie)]
                except KeyError:
                    continue
                a = a.iloc[0] if isinstance(a, pd.DataFrame) else a
                b = b.iloc[0] if isinstance(b, pd.DataFrame) else b
                row = {'stem_stimwise': stim, 'stem_runwise': run, 'grouping': g,
                       'specie': specie,
                       'rdm_r': (rdm or {}).get('r'), 'rdm_identical': (rdm or {}).get('identical'),
                       'mah_n_supra': a.get('z_n_supra'), 'corr_n_supra': b.get('z_n_supra'),
                       'mah_corr_vox': a.get('corr_n_voxels'), 'corr_corr_vox': b.get('corr_n_voxels'),
                       'mah_z_max': a.get('z_max'), 'corr_z_max': b.get('z_max')}
                R = sim.get(specie)
                if R is not None and a['rsa_model'] in R.index and b['rsa_model'] in R.index:
                    row['map_r'] = float(R.loc[a['rsa_model'], b['rsa_model']])
                rows.append(row)
    method_cmp = pd.DataFrame(rows)
    method_cmp.to_csv(os.path.join(out_dir, 'method_comparison.csv'), index=False)

    # --- grouping contrast -------------------------------------------------
    g_rows = []
    for (dm, stem, specie), d in per_map.groupby(['dis_method', 'stem', 'specie']):
        piv = d.set_index('grouping')
        row = {'dis_method': dm, 'stem': stem, 'specie': specie}
        for g in ('all', 'collapse', 'within', 'cross', 'dog', 'hum'):
            if g in piv.index:
                r = piv.loc[g]
                r = r.iloc[0] if isinstance(r, pd.DataFrame) else r
                row[f'{g}_supra'] = r.get('z_n_supra')
                row[f'{g}_corr'] = r.get('corr_n_voxels')
        if 'cross_supra' in row and 'within_supra' in row:
            try:
                row['cross_minus_within_supra'] = row['cross_supra'] - row['within_supra']
            except TypeError:
                pass
        g_rows.append(row)
    grouping_tbl = pd.DataFrame(g_rows).sort_values(['specie', 'dis_method', 'stem'])
    grouping_tbl.to_csv(os.path.join(out_dir, 'grouping_contrast.csv'), index=False)

    for s, c in conj_tbl.items():
        if len(c):
            c.to_csv(os.path.join(out_dir, f'distinctiveness_{s}.csv'), index=False)

    # --- provenance --------------------------------------------------------
    # A z-map is real-data / null. Both halves are files with dates, and the
    # z must be newer than both -- otherwise it divides one generation of the
    # analysis by another. Re-running step 3 or step 6 without re-running step 7
    # leaves exactly that, and nothing downstream notices.
    prov = []
    for rec in inv:
        if not rec['has_z']:
            continue
        tm, tn, tz = rec.get('mtime_mean'), rec.get('mtime_null_mean'), rec.get('mtime_z')
        row = {'specie': rec['specie'], 'rsa_model': rec['rsa_model'],
               'dis_method': rec['dis_method'],
               'step3_mean': _ts(tm), 'step6_null': _ts(tn), 'step7_z': _ts(tz),
               'step9_corrected': _ts(rec.get('mtime_corrected'))}
        row['z_older_than_mean_days'] = _days(tm, tz)
        row['z_older_than_null_days'] = _days(tn, tz)
        row['null_older_than_mean_days'] = _days(tm, tn)
        row['stale'] = bool((row['z_older_than_mean_days'] or 0) > 0
                            or (row['z_older_than_null_days'] or 0) > 0)
        prov.append(row)
    provenance = pd.DataFrame(prov)
    provenance.to_csv(os.path.join(out_dir, 'provenance.csv'), index=False)

    # --- calibration -------------------------------------------------------
    calib_cols = ['specie', 'rsa_model', 'dis_method', 'z_mean', 'z_sd', 'z_max',
                  'z_min', 'z_n_supra', 'z_n_neg_supra', 'n_participants']
    calib = per_map[per_map.state == 'z'][[c for c in calib_cols if c in per_map.columns]].copy()
    if 'z_sd' in calib:
        calib['sd_dev'] = (calib.z_sd - 1).abs()
        calib = calib.sort_values('sd_dev', ascending=False).drop(columns='sd_dev')
    calib.to_csv(os.path.join(out_dir, 'calibration.csv'), index=False)

    figures = []
    if not args.no_figures:
        try:
            figures = make_figures(out_dir, per_map, sim, order, {k: v[0] for k, v in agree.items()}, args)
            log(f"figures: {len(figures)} written")
        except Exception as e:
            log(f"figures failed: {e.__class__.__name__}: {e}")

    rp = write_report(out_dir, args, per_map, peaks, sim, order, agree, method_cmp,
                      grouping_tbl, conj_tbl, calib, provenance, figures, mask_sizes,
                      mask_notes, time.time() - t0)
    log(f"report -> {rp}")
    log(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()

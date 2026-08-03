#!/usr/bin/env python
"""gpu_group.py -- GPU (PyTorch) reimplementation of RSA pipeline steps 3, 5, 6, 7.

Companion to ``gpu_rsa.py`` (steps 1, 2, 4). Where that module works on *one*
participant, this one works on the *group*: it consumes the per-participant maps
that a Colab run already produced -- the ``result_{model}_{specie}-sub-NN.zip``
files sitting in OUT_DIR -- and reduces them to the group maps steps 8-10 expect:

  * Step 3 -- ``calculate_group_model_similarity_map``   (mean/std of the real maps)
  * Step 5 -- ``calculate_group_model_similarity_map_rnd`` (reps_group group perms)
  * Step 6 -- ``calculate_voxelwise_rnd_distribution``   (mean/std across those)
  * Step 7 -- ``calculate_z_maps_rnd`` + ``calculate_z_map_real_data``

Why the GPU helps here at all: step 5 is a *sampling* reduction. For every one of
``reps_group`` group permutations it draws one of each participant's ``reps``
permutation maps and averages them. On the CPU that is
``reps_group x n_participants`` NIfTI loads (1000 x 16 = 16 000 file reads of the
same 1600 files) followed by a second full pass for the std in step 6. Here the
1600 maps are read **once** into an ``(n_maps, n_voxels)`` matrix and the whole
of steps 5-7 becomes an index-gather plus a mean along the participant axis, so
the run is bounded by reading the result zips rather than by arithmetic.

Faithfulness
------------
Everything is float64, like the CPU (numpy) pipeline, and every output volume,
filename and dtype matches what ``rsa_utils`` writes -- including the details
that are easy to get wrong:

  * the group *rnd* maps carry **no** ``{mask_type}-`` prefix while the group
    *real* maps do (``calculate_group_model_similarity_map_rnd`` vs
    ``calculate_group_model_similarity_map``);
  * ``mah_fold`` sub-foldering applies to the **participant** paths only, never
    to the group ``mean/`` folder;
  * step 3 multiplies mean and std by the mask; step 7's rnd z maps do **not**
    clean up non-finite values (a voxel with zero null-std stays NaN, exactly as
    ``calculate_z_maps_rnd`` leaves it), while the real z map zeroes them and is
    cast to float32 under the mean map's float64 header, as
    ``calculate_z_map_real_data`` does.

The one deliberate difference is the same one the step-4 port makes: the CPU
draws its per-participant permutation with an unseeded ``random.choice``, so a
rerun never reproduces itself. Here the draw uses a deterministic seed derived
from the model name, which is a valid sample from the same null but not
bit-identical to a CPU run.

Dependency-light (torch, numpy, nibabel, plus ``gpu_rsa`` from the same folder)
so it runs on a stock Colab runtime.
"""

import contextlib
import glob
import gzip
import io
import json
import os
import sys
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import nibabel as nib
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gpu_rsa  # noqa: E402  -- check_same_space / pick_device / load_reference_mask

DTYPE = torch.float64
STEPS_ALL = (3, 5, 6, 7)


class MissingMapsError(RuntimeError):
    """Not enough participant maps to build a group map."""


class OffMaskError(ValueError):
    """A participant map has non-zero values outside the searchlight mask."""


# ===========================================================================
# manifest + pipeline paths
#
# Every path below is returned *relative to the data root* and in posix form,
# because that is exactly the arcname convention the result zips use and what
# tools/unpack_results.py merges onto the pipeline disk.
# ===========================================================================
def load_manifest(pkg_root):
    with open(os.path.join(pkg_root, "manifest.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def is_per_run(manifest):
    """True when participant maps live in per-run folders.

    Mirrors the ``per_run`` flag in ``_model_similarity_map_file`` /
    ``calculate_group_model_similarity_map``: everything except Mahalanobis is
    per-run, and Mahalanobis is too under the ``stim-wise-all-runs`` fold.
    """
    return (manifest["dis_method"] != "mahalanobis"
            or manifest.get("mah_fold") == "stim-wise-all-runs")


def units(manifest):
    """The averaging units of the group steps: ``(sub_N, session, run_N)``.

    One per participant for Mahalanobis stim-wise, one per participant-run
    otherwise -- the same enumeration ``calculate_group_model_similarity_map``
    walks to build ``files_list``.
    """
    out = []
    runs_by_sub = manifest.get("runs_by_sub") or {}
    for sub_N in manifest["participants"]:
        if not is_per_run(manifest):
            out.append((int(sub_N), None, None))
            continue
        entries = runs_by_sub.get(str(sub_N)) or runs_by_sub.get(sub_N)
        if not entries:
            raise ValueError(
                f"Per-run layout needs runs_by_sub for sub-{sub_N}; rebuild the "
                "group package with tools/create_group_package.py.")
        for e in entries:
            out.append((int(sub_N), int(e["session"]), int(e["run_N"])))
    return out


def unit_label(unit):
    sub_N, session, run_N = unit
    if session is None:
        return f"sub-{sub_N:02d}"
    return f"sub-{sub_N:02d} ses-{session:02d} run-{run_N:02d}"


def _participant_stem(manifest):
    """``{mask_type}-r-{radius}_{dis_method}_{rsa_method}`` -- the participant stem."""
    stem = (f"r-{manifest['radius']}_{manifest['dis_method']}"
            f"_{manifest['rsa_method']}")
    if manifest.get("mask_type"):
        stem = f"{manifest['mask_type']}-{stem}"
    return stem


def _group_stem(manifest):
    """``{specie}-r-{radius}_{dis_method}_{rsa_method}`` -- the group rnd stem.

    Note the missing ``{mask_type}-``: the group permutation files really are
    named without it (see ``calculate_group_model_similarity_map_rnd``), which is
    why step 8's glob finds them and a mask-prefixed guess would not.
    """
    return (f"{manifest['specie']}-r-{manifest['radius']}"
            f"_{manifest['dis_method']}_{manifest['rsa_method']}")


def _participant_root_rel(manifest, rsa_model, rnd):
    """Fold-isolated participant root -- mirrors ``_rsa_model_output_dir``."""
    parts = [manifest["dataset"], "results", "RSA_rnd" if rnd else "RSA",
             manifest["model"], rsa_model]
    mah_fold = manifest.get("mah_fold")
    if manifest["dis_method"] == "mahalanobis" and mah_fold not in (None, "stim-wise"):
        parts.append(mah_fold)
    return "/".join(parts)


def participant_map_rel(manifest, rsa_model, unit, rnd=False, rnd_index=None):
    """Data-root-relative path of one participant model-similarity map.

    Mirrors ``rsa_utils._model_similarity_map_file``.
    """
    sub_N, session, run_N = unit
    parts = [_participant_root_rel(manifest, rsa_model, rnd),
             f"{manifest['specie']}-sub-{sub_N:02d}"]
    if is_per_run(manifest):
        parts.append(f"ses-{int(session):02d}_task-{manifest['task']}"
                     f"_run-{int(run_N):02d}")
    name = _participant_stem(manifest)
    if rnd_index is not None:
        name = f"{name}_{rnd_index:04d}"
    parts.append(f"{name}.nii.gz")
    return "/".join(parts)


def group_real_rel(manifest, rsa_model, kind):
    """Step 3 / step 7-real outputs. ``kind`` in {'mean', 'std', 'z'}."""
    stem = f"r-{manifest['radius']}_{manifest['dis_method']}_{manifest['rsa_method']}"
    if manifest.get("mask_type"):
        stem = f"{manifest['mask_type']}-{manifest['specie']}-{stem}"
    else:
        stem = f"{manifest['specie']}-{stem}"
    return "/".join([manifest["dataset"], "results", "RSA", manifest["model"],
                     rsa_model, "mean", f"{stem}_{kind}.nii.gz"])


def group_rnd_rel(manifest, rsa_model, kind, index):
    """Step 5 (kind='mean') / step 7-rnd (kind='z') permutation outputs."""
    return "/".join([manifest["dataset"], "results", "RSA_rnd", manifest["model"],
                     rsa_model, "mean",
                     f"{_group_stem(manifest)}_{kind}_{index:05d}.nii.gz"])


def group_rnd_log_rel(manifest, rsa_model, kind):
    """``..._{kind}_log.txt`` next to the permutation maps (step 7 writes one)."""
    return "/".join([manifest["dataset"], "results", "RSA_rnd", manifest["model"],
                     rsa_model, "mean", f"{_group_stem(manifest)}_{kind}_log.txt"])


def distribution_rel(manifest, rsa_model, kind):
    """Step 6 voxelwise null distribution. ``kind`` in {'mean', 'std'}."""
    return "/".join([manifest["dataset"], "results", "RSA_rnd", manifest["model"],
                     f"{manifest['specie']}-{rsa_model}_{kind}.nii.gz"])


def mask_rel(manifest):
    return "/".join([manifest["dataset"], "ROI", manifest["specie"],
                     f"{manifest['mask_type']}.nii.gz"])


def target_path(manifest, rel):
    """Render a data-root-relative path as it will look on the pipeline disk.

    Only used for the *contents* of the logs step 3 writes, so a merged run is
    indistinguishable from one computed on the workstation. ``datafolder`` is
    recorded by ``tools/create_group_package.py``.
    """
    datafolder = manifest.get("datafolder")
    if not datafolder:
        return rel
    sep = "\\" if "\\" in datafolder else "/"
    return datafolder.rstrip("/\\") + sep + rel.replace("/", sep)


# ===========================================================================
# Reading participant maps out of Colab result zips (or an unpacked tree)
# ===========================================================================
def _list_dir(dirpath):
    """Names of the files in one folder, or an empty set if it does not exist.

    One ``scandir`` is one round-trip on the network data disk and answers every
    "does this map exist?" question about that folder at once -- the same reason
    ``tools/unpack_results.py`` is built around listings instead of ``exists``.
    """
    try:
        with os.scandir(dirpath) as it:
            return {e.name for e in it if e.is_file()}
    except (FileNotFoundError, NotADirectoryError):
        return set()
    except OSError as exc:
        print(f"WARNING: cannot list {dirpath}: {exc}")
        return set()


def _img_from_bytes(raw):
    """Decode a ``.nii.gz`` blob into a nibabel image without touching disk."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    fh = nib.FileHolder(fileobj=io.BytesIO(raw))
    return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})


class ResultStore:
    """Locate participant maps across result zips and/or unpacked data trees.

    A source is either

    * a folder of ``result_*.zip`` files (what a Colab step-1/2/4 run writes to
      OUT_DIR) -- matched by filename, so only the zips of the model being
      processed are ever opened; or
    * a data root, i.e. any folder containing ``{dataset}/results/`` -- what you
      get after ``tools/unpack_results.py``, or after unzipping by hand.

    Both may be mixed; the tree wins when a map exists in both, because an
    unpacked file is cheaper to read than a zip member.
    """

    def __init__(self, sources, dataset=None, verbose=True):
        if isinstance(sources, (str, os.PathLike)):
            sources = [sources]
        self.trees, self.zip_dirs = [], []
        self._zip_names = {}     # dir -> [basenames]
        self._members = {}       # zip path -> {arcname: member name}
        self.verbose = verbose
        for src in sources:
            src = os.path.abspath(str(src))
            if not os.path.isdir(src):
                raise FileNotFoundError(f"Result source not found: {src}")
            looks_like_tree = (
                dataset and os.path.isdir(os.path.join(src, dataset, "results")))
            if not looks_like_tree:
                # also accept a root holding exactly one dataset folder
                looks_like_tree = any(
                    os.path.isdir(os.path.join(src, d, "results"))
                    for d in os.listdir(src)
                    if os.path.isdir(os.path.join(src, d)))
            if looks_like_tree:
                self.trees.append(src)
            zips = [n for n in os.listdir(src) if n.lower().endswith(".zip")]
            if zips:
                self.zip_dirs.append(src)
                self._zip_names[src] = zips
        if verbose:
            print(f"[store] {len(self.trees)} data tree(s), "
                  f"{sum(len(v) for v in self._zip_names.values())} zip(s) in "
                  f"{len(self.zip_dirs)} folder(s)")

    # -- zip indexing -------------------------------------------------------
    def zips_for(self, manifest, rsa_model):
        """Result zips whose *name* says they hold this model's participant maps.

        ``gpu_rsa.zip_model_result`` names them
        ``result_{rsa_model}_{specie}-sub-NN.zip``, so this is a prefix test on a
        single directory listing -- no zip is opened to find out.
        """
        prefix = f"result_{rsa_model}_{manifest['specie']}-sub-".lower()
        out = []
        for d in self.zip_dirs:
            for name in self._zip_names[d]:
                if name.lower().startswith(prefix):
                    out.append(os.path.join(d, name))
        return sorted(out)

    def _index_zip(self, zip_path):
        idx = self._members.get(zip_path)
        if idx is None:
            with zipfile.ZipFile(zip_path) as zf:
                idx = {n.replace("\\", "/").lstrip("./"): n for n in zf.namelist()}
            self._members[zip_path] = idx
        return idx

    def index_model(self, manifest, rsa_model):
        """Build ``{data-root-relative path: ref}`` for one model.

        A ``ref`` is ``('file', abspath)`` or ``('zip', zip_path, member)``.

        The tree side is deliberately **one listing per participant folder**
        rather than a recursive glob: a recursive glob under the model root also
        walks the ``mean/`` folder, which holds the thousands of group
        permutation maps this step is about to write, and on the network data
        disk each round-trip costs ~56 ms. Listing only the folders that can hold
        participant maps turns that into ``2 x n_units`` round-trips.
        """
        refs = {}
        for zip_path in self.zips_for(manifest, rsa_model):
            for rel, member in self._index_zip(zip_path).items():
                if rel.endswith(".nii.gz"):
                    refs[rel] = ("zip", zip_path, member)
        if not self.trees:
            return refs

        wanted_dirs = set()
        for unit in units(manifest):
            for rnd in (False, True):
                rel = participant_map_rel(manifest, rsa_model, unit, rnd=rnd,
                                          rnd_index=0 if rnd else None)
                wanted_dirs.add(rel.rsplit("/", 1)[0])
        jobs = [(tree, reldir) for tree in self.trees for reldir in sorted(wanted_dirs)]
        with ThreadPoolExecutor(max_workers=max(1, min(16, len(jobs)))) as ex:
            listings = list(ex.map(
                lambda j: _list_dir(os.path.join(j[0], j[1].replace("/", os.sep))),
                jobs))
        for (tree, reldir), names in zip(jobs, listings):
            d = os.path.join(tree, reldir.replace("/", os.sep))
            for name in names:
                if name.endswith(".nii.gz"):
                    refs[f"{reldir}/{name}"] = ("file", os.path.join(d, name))
        return refs


def _read_ref(ref, open_zips):
    if ref[0] == "file":
        return nib.load(ref[1])
    zf = open_zips[ref[1]]
    return _img_from_bytes(zf.read(ref[2]))


def _load_unit(manifest, rsa_model, unit, refs, mask_img, mask_bool, mask_flat,
               reps, want_real, strict_space, strict_mask):
    """Load one unit's real map and its available permutation maps.

    Returns ``(real_vec | None, {rep_index: vec})`` with each ``vec`` restricted
    to the mask voxels. Every image is checked against the mask's voxel grid and
    verified to be zero outside the mask -- the group steps combine these by
    array index and write the result on the mask's grid, so both have to hold.
    """
    wanted = {}
    if want_real:
        rel = participant_map_rel(manifest, rsa_model, unit)
        if rel in refs:
            wanted[("real", None)] = refs[rel]
    for i in range(reps):
        rel = participant_map_rel(manifest, rsa_model, unit, rnd=True, rnd_index=i)
        if rel in refs:
            wanted[("rnd", i)] = refs[rel]
    if not wanted:
        return None, {}

    zip_paths = sorted({r[1] for r in wanted.values() if r[0] == "zip"})
    real_vec, rnd = None, {}
    loaded = []
    with contextlib.ExitStack() as stack:
        open_zips = {p: stack.enter_context(zipfile.ZipFile(p)) for p in zip_paths}
        for (kind, i), ref in sorted(wanted.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
            img = _read_ref(ref, open_zips)
            label = f"{unit_label(unit)} {kind}{'' if i is None else f'-{i:04d}'}"
            loaded.append((label, img))
            flat = np.asarray(img.dataobj, dtype=np.float64).reshape(-1)
            if flat.size != mask_bool.size:
                raise gpu_rsa.SpaceMismatchError(
                    f"{label}: {flat.size} voxels != mask's {mask_bool.size}")
            off = flat[~mask_bool.reshape(-1)]
            if off.any():
                message = (
                    f"{label} has {int((off != 0).sum())} non-zero voxel(s) outside "
                    f"the mask {manifest.get('mask_type')!r}.\n"
                    "The group steps only carry mask voxels through the GPU, so a "
                    "map with support outside the mask would be silently truncated. "
                    "Most likely the participant maps were computed against a "
                    "different mask than the one in this package.")
                if strict_mask:
                    raise OffMaskError(message)
                print(f"WARNING: {message}")
            vec = flat[mask_flat]
            if kind == "real":
                real_vec = vec
            else:
                rnd[i] = vec
    gpu_rsa.check_same_space(
        ("mask", mask_img), loaded,
        context=f"GPU group steps for {rsa_model} {unit_label(unit)}",
        strict=strict_space)
    return real_vec, rnd


def load_participant_maps(manifest, rsa_model, store, mask_img, mask_bool,
                          workers=8, want_real=True, want_rnd=True, verbose=True):
    """Read every unit's maps once into memory.

    Availability is tracked **separately per step**, as on the CPU: step 3 averages
    every unit that has a real map, step 5 every unit that has at least one
    permutation map, and a unit can qualify for one and not the other. Merging the
    two would quietly drop a participant from the group mean because their step-4
    job had not finished.

    ``rnd`` concatenates each qualifying unit's permutation maps; ``offsets[u]``
    and ``counts[u]`` delimit unit ``u``'s block.
    """
    refs = store.index_model(manifest, rsa_model)
    mask_flat = np.flatnonzero(mask_bool.reshape(-1))
    all_units = units(manifest)
    reps = manifest["reps"]
    strict_space = not manifest.get("allow_space_mismatch", False)
    strict_mask = not manifest.get("allow_off_mask", False)

    t0 = time.time()
    n_workers = max(1, min(workers, len(all_units)))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(
            lambda u: _load_unit(manifest, rsa_model, u, refs, mask_img, mask_bool,
                                 mask_flat, reps, want_real, strict_space, strict_mask),
            all_units))

    real_units, real_rows = [], []
    rnd_units, rnd_blocks = [], []
    missing_real, missing_rnd = [], []
    for unit, (real_vec, rnd) in zip(all_units, results):
        if want_real:
            if real_vec is None:
                missing_real.append(unit_label(unit))
            else:
                real_units.append(unit)
                real_rows.append(real_vec)
        if want_rnd:
            if rnd:
                rnd_units.append(unit)
                rnd_blocks.append([rnd[i] for i in sorted(rnd)])
            else:
                missing_rnd.append(unit_label(unit))

    counts = np.array([len(b) for b in rnd_blocks], dtype=np.int64)
    offsets = (np.concatenate([[0], np.cumsum(counts)])[:-1]
               if len(counts) else np.array([], dtype=np.int64))
    rnd_flat = (np.stack([v for b in rnd_blocks for v in b], axis=0)
                if counts.sum() else np.zeros((0, mask_flat.size), dtype=np.float64))
    real = np.stack(real_rows, axis=0) if real_rows else None

    if verbose:
        print(f"[load] {rsa_model}: {len(real_units)}/{len(all_units)} real map(s), "
              f"{len(rnd_units)}/{len(all_units)} unit(s) with permutations "
              f"({int(counts.sum())} map(s)), {time.time() - t0:.1f}s")
        if missing_rnd:
            print(f"[load]   no permutation maps: {', '.join(missing_rnd[:8])}"
                  + (" ..." if len(missing_rnd) > 8 else ""))
        if missing_real:
            print(f"[load]   no real map: {', '.join(missing_real[:8])}"
                  + (" ..." if len(missing_real) > 8 else ""))
    return {
        "units": rnd_units, "real_units": real_units, "all_units": all_units,
        "real": real, "rnd": rnd_flat, "offsets": offsets, "counts": counts,
        "mask_flat": mask_flat, "n_real": len(real_units),
        "missing_real": missing_real, "missing_rnd": missing_rnd,
    }


# ===========================================================================
# writing volumes
# ===========================================================================
def _volume(vec, mask_flat, shape, fill=0.0):
    vol = np.full(int(np.prod(shape)), fill, dtype=np.float64)
    vol[mask_flat] = vec
    return vol.reshape(shape)


def save_volume(vec, mask_flat, shape, affine, path, fill=0.0, header=None,
                astype=None):
    """Scatter a masked vector into a volume and write it as ``.nii.gz``.

    ``fill`` is the value outside the mask, and it is not cosmetic: the CPU
    divides whole volumes, so off-mask voxels come out 0 for a mean and NaN for a
    z map (0-0)/0. ``astype``/``header`` reproduce ``calculate_z_map_real_data``,
    which casts to float32 but saves under the mean map's float64 header.
    """
    data = _volume(vec, mask_flat, shape, fill=fill)
    if astype is not None:
        data = data.astype(astype)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = nib.Nifti1Image(data, affine, header=header)
    nib.save(img, path)
    return path


def _save_many(jobs, workers=8):
    """Write many volumes in parallel (zlib releases the GIL, so threads help)."""
    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        return list(ex.map(lambda kw: save_volume(**kw), jobs))


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ===========================================================================
# Step 5 -- group permutations (the part the GPU is here for)
# ===========================================================================
def draw_group_indices(counts, reps_group, seed):
    """Pick one permutation map per unit for each of ``reps_group`` group perms.

    Returns an ``(reps_group, n_units)`` int64 array of **flat** row indices into
    the concatenated permutation matrix. Same scheme as
    ``calculate_group_model_similarity_map_rnd`` -- an independent uniform draw
    per unit, with replacement, over the permutation indices that actually exist
    on disk -- but seeded, so the run reproduces itself.
    """
    rng = np.random.default_rng(seed)
    offsets = np.concatenate([[0], np.cumsum(counts)])[:-1]
    cols = np.empty((reps_group, len(counts)), dtype=np.int64)
    for u, n in enumerate(counts):
        cols[:, u] = offsets[u] + rng.integers(0, int(n), size=reps_group)
    return cols


def group_permutation_stats(rnd_flat, cols, device=None, vox_batch=20000,
                            g_batch=64, want_group_means=True, verbose=False):
    """Steps 5+6 in one voxel-chunked pass.

    ``rnd_flat`` is the ``(T, V)`` matrix of every unit's permutation maps and
    ``cols`` the ``(G, U)`` draw from :func:`draw_group_indices`.

    Returns ``(group_means (G, V) | None, dist_mean (V,), dist_std (V,))`` in
    float64. The group mean is a plain mean over units, matching ``nifti_mean``,
    and the distribution std is the population std over the G group maps
    (``sqrt(sum((x-mean)^2)/G)``), matching ``nifti_mean``'s second pass.

    A voxel chunk holds all G group maps at once, which is what lets step 6 and
    step 7 ride along on the same pass instead of re-reading 1000 volumes.
    """
    device = device or gpu_rsa.pick_device()
    T, V = rnd_flat.shape
    G, U = cols.shape
    cols_t = torch.as_tensor(cols, device=device)
    group_means = np.empty((G, V), dtype=np.float64) if want_group_means else None
    dist_mean = np.empty(V, dtype=np.float64)
    dist_std = np.empty(V, dtype=np.float64)

    vox_batch = vox_batch if vox_batch and vox_batch > 0 else V
    g_batch = g_batch if g_batch and g_batch > 0 else G
    t0 = time.time()
    for v0 in range(0, V, vox_batch):
        v1 = min(v0 + vox_batch, V)
        block = torch.as_tensor(rnd_flat[:, v0:v1], dtype=DTYPE, device=device)
        gm = torch.empty(G, v1 - v0, dtype=DTYPE, device=device)
        for g0 in range(0, G, g_batch):
            g1 = min(g0 + g_batch, G)
            sel = block.index_select(0, cols_t[g0:g1].reshape(-1))
            gm[g0:g1] = sel.view(g1 - g0, U, v1 - v0).mean(dim=1)
        mu = gm.mean(dim=0)
        sd = torch.sqrt(((gm - mu) ** 2).mean(dim=0))
        dist_mean[v0:v1] = mu.cpu().numpy()
        dist_std[v0:v1] = sd.cpu().numpy()
        if want_group_means:
            group_means[:, v0:v1] = gm.cpu().numpy()
        del block, gm
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if verbose:
            print(f"[step5]   voxels {v1}/{V}  ({time.time() - t0:.1f}s)")
    return group_means, dist_mean, dist_std


def mean_std(rows, device=None):
    """Voxelwise mean/std over a ``(N, V)`` stack -- the ``nifti_mean`` reduction."""
    device = device or gpu_rsa.pick_device()
    x = torch.as_tensor(rows, dtype=DTYPE, device=device)
    mu = x.mean(dim=0)
    sd = torch.sqrt(((x - mu) ** 2).mean(dim=0))
    return mu.cpu().numpy(), sd.cpu().numpy()


# ===========================================================================
# Steps 3 / 5 / 6 / 7 -- driver
# ===========================================================================
def _step3_log(manifest, rsa_model, maps, mask_file_rel):
    """Reproduce the ``.json`` sidecar ``calculate_group_model_similarity_map``
    writes -- it is read back on the next run to decide whether the map has to be
    recomputed, so the paths in it must look like workstation paths."""
    file_list = [target_path(manifest, participant_map_rel(manifest, rsa_model, u))
                 for u in maps["real_units"]]
    total = len(maps["all_units"])
    return {
        "datafolder": manifest.get("datafolder", ""),
        "dataset": manifest["dataset"],
        "specie": manifest["specie"],
        "model": manifest["model"],
        "mask_type": manifest.get("mask_type"),
        "task": manifest["task"],
        "radius": manifest["radius"],
        "rsa_model": rsa_model,
        "dis_method": manifest["dis_method"],
        "mah_fold": manifest.get("mah_fold"),
        "replace_file": True,
        "min_percentage_available": manifest.get("min_percentage_available", 1.0),
        "participants": [int(p) for p in manifest["participants"]],
        "file_list": file_list,
        "perc_available": (len(file_list) / total) if total else 0.0,
        "output_mean_file": target_path(manifest, group_real_rel(manifest, rsa_model, "mean")),
        "output_std_file": target_path(manifest, group_real_rel(manifest, rsa_model, "std")),
        "mask_file": target_path(manifest, mask_file_rel),
        "notes": ["computed on GPU by tools/colab_gpu/gpu_group.py"],
    }


def _dump_log_json(path, payload):
    """``yaml.dump`` like the CPU does, falling back to JSON (valid YAML) if
    PyYAML is unavailable -- ``yaml.safe_load`` reads either back."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        import yaml
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(payload, f)
    except ImportError:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    return path


def run_group_model(pkg_root, work_root, manifest, rsa_model, store,
                    steps=STEPS_ALL, device=None, seed=None, vox_batch=20000,
                    g_batch=64, write_group_means=True, workers=8, verbose=True):
    """Run steps 3/5/6/7 for one RSA model. Returns the written paths.

    Outputs land under ``{work_root}/data/`` with pipeline-relative paths, ready
    to be zipped by :func:`zip_group_result` and merged with
    ``tools/unpack_results.py``.
    """
    steps = tuple(sorted(set(int(s) for s in steps)))
    device = device or gpu_rsa.pick_device()
    data_root = os.path.join(pkg_root, "data")
    out_root = os.path.join(work_root, "data")
    reps_group = manifest["reps_group"]
    min_pct = manifest.get("min_percentage_available", 1.0)

    mask_img, mask_bool = gpu_rsa.load_reference_mask(data_root, manifest)
    shape, affine = mask_bool.shape, mask_img.affine

    need_rnd = any(s in steps for s in (5, 6, 7))
    need_real = 3 in steps
    maps = load_participant_maps(manifest, rsa_model, store, mask_img, mask_bool,
                                 workers=workers, want_real=need_real,
                                 want_rnd=need_rnd, verbose=verbose)
    mask_flat = maps["mask_flat"]
    total_units = len(maps["all_units"])
    written = []

    def out(rel):
        return os.path.join(out_root, rel.replace("/", os.sep))

    # ---- Step 3: group mean/std of the real maps --------------------------
    real_mean = None
    if 3 in steps:
        if maps["real"] is None:
            raise MissingMapsError(
                f"{rsa_model}: no real (step-2) maps found -- cannot run step 3.")
        pct = maps["n_real"] / total_units
        if pct < min_pct:
            raise MissingMapsError(
                f"{rsa_model}: only {pct*100:.1f}% of the real maps are available "
                f"({maps['n_real']}/{total_units}); min_percentage_available is "
                f"{min_pct*100:.1f}%.")
        mu, sd = mean_std(maps["real"], device=device)
        # step 3 multiplies mean and std by the mask; inside the mask that is a
        # no-op, so it is enough to leave the off-mask fill at 0
        real_mean = mu
        written.append(save_volume(mu, mask_flat, shape, affine,
                                   out(group_real_rel(manifest, rsa_model, "mean"))))
        written.append(save_volume(sd, mask_flat, shape, affine,
                                   out(group_real_rel(manifest, rsa_model, "std"))))
        log_path = out(group_real_rel(manifest, rsa_model, "mean")).replace(
            ".nii.gz", ".json")
        written.append(_dump_log_json(
            log_path, _step3_log(manifest, rsa_model, maps, mask_rel(manifest))))
        if verbose:
            print(f"[step3] {rsa_model}: mean/std over {maps['n_real']} map(s)")

    if not need_rnd:
        return written

    # ---- Steps 5 + 6: group permutations and their voxelwise distribution --
    n_units = len(maps["units"])
    if n_units == 0:
        raise MissingMapsError(
            f"{rsa_model}: no permutation (step-4) maps found for any unit.")
    pct = n_units / total_units
    if pct < min_pct:
        raise MissingMapsError(
            f"{rsa_model}: only {pct*100:.1f}% of the units have permutation maps "
            f"({n_units}/{total_units}); min_percentage_available is "
            f"{min_pct*100:.1f}%.")
    if seed is None:
        seed = zlib.crc32(
            f"group-{rsa_model}-{manifest['specie']}-{reps_group}".encode())
    cols = draw_group_indices(maps["counts"], reps_group, seed)

    want_gm = write_group_means and 5 in steps
    t0 = time.time()
    group_means, dist_mean, dist_std = group_permutation_stats(
        maps["rnd"], cols, device=device, vox_batch=vox_batch, g_batch=g_batch,
        want_group_means=want_gm or (7 in steps), verbose=False)
    if verbose:
        print(f"[step5] {rsa_model}: {reps_group} group permutation(s) over "
              f"{n_units} unit(s) in {time.time() - t0:.1f}s (seed={seed})")

    if 5 in steps and write_group_means:
        t0 = time.time()
        jobs = [dict(vec=group_means[g], mask_flat=mask_flat, shape=shape,
                     affine=affine,
                     path=out(group_rnd_rel(manifest, rsa_model, "mean", g)))
                for g in range(reps_group)]
        written += _save_many(jobs, workers=workers)
        if verbose:
            print(f"[step5] wrote {reps_group} group mean map(s) in "
                  f"{time.time() - t0:.1f}s")
    elif 5 in steps and verbose:
        print("[step5] group mean maps not written (write_group_means=False); "
              "they are inputs to steps 6-7 only, both of which run here.")

    if 6 in steps:
        written.append(save_volume(dist_mean, mask_flat, shape, affine,
                                   out(distribution_rel(manifest, rsa_model, "mean"))))
        written.append(save_volume(dist_std, mask_flat, shape, affine,
                                   out(distribution_rel(manifest, rsa_model, "std"))))
        log = [f"Found {reps_group} available rnd mean files.",
               "Missing 0 rnd mean files.",
               f"Calculating distribution mean map: "
               f"{target_path(manifest, distribution_rel(manifest, rsa_model, 'mean'))}",
               f"Calculating distribution std map: "
               f"{target_path(manifest, distribution_rel(manifest, rsa_model, 'std'))}"]
        written.append(write_text(
            out(distribution_rel(manifest, rsa_model, "mean")).replace(
                ".nii.gz", "_log.txt"),
            "\n".join(log)))
        if verbose:
            print(f"[step6] {rsa_model}: voxelwise null mean/std written")

    # ---- Step 7: z maps ----------------------------------------------------
    if 7 in steps:
        t0 = time.time()
        with np.errstate(divide="ignore", invalid="ignore"):
            z_rnd = (group_means - dist_mean[None, :]) / dist_std[None, :]
        # off-mask voxels are (0-0)/0 on the CPU, i.e. NaN -- keep them NaN so a
        # map written here is byte-comparable with one written by the pipeline
        jobs = [dict(vec=z_rnd[g], mask_flat=mask_flat, shape=shape, affine=affine,
                     path=out(group_rnd_rel(manifest, rsa_model, "z", g)),
                     fill=np.nan)
                for g in range(reps_group)]
        written += _save_many(jobs, workers=workers)
        written.append(write_text(
            out(group_rnd_log_rel(manifest, rsa_model, "z")),
            "\n".join([
                f"Loaded distribution mean map: "
                f"{target_path(manifest, distribution_rel(manifest, rsa_model, 'mean'))}",
                f"Loaded distribution std map: "
                f"{target_path(manifest, distribution_rel(manifest, rsa_model, 'std'))}",
                f"Calculated z maps for {reps_group} available rnd mean files.",
                "Missing 0 rnd mean files."])))

        if real_mean is None:
            real_mean = _load_existing_real_mean(work_root, store, manifest,
                                                 rsa_model, mask_flat, mask_bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            z_real = (real_mean - dist_mean) / dist_std
        z_real[~np.isfinite(z_real)] = 0.0
        # the CPU casts to float32 but keeps the mean map's (float64) header, so
        # the file on disk is float64 carrying float32-rounded values
        mean_header = nib.Nifti1Image(
            np.zeros(shape, dtype=np.float64), affine).header.copy()
        written.append(save_volume(
            z_real, mask_flat, shape, affine,
            out(group_real_rel(manifest, rsa_model, "z")),
            fill=0.0, header=mean_header, astype=np.float32))
        if verbose:
            print(f"[step7] {rsa_model}: {reps_group} rnd z map(s) + real z map "
                  f"in {time.time() - t0:.1f}s")
    return written


def _load_existing_real_mean(work_root, store, manifest, rsa_model, mask_flat,
                             mask_bool):
    """Find a step-3 group mean map when step 3 was not part of this run."""
    rel = group_real_rel(manifest, rsa_model, "mean")
    candidates = [os.path.join(work_root, "data", rel.replace("/", os.sep))]
    for tree in store.trees:
        candidates.append(os.path.join(tree, rel.replace("/", os.sep)))
    for path in candidates:
        if os.path.exists(path):
            flat = np.asarray(nib.load(path).dataobj, dtype=np.float64).reshape(-1)
            return flat[mask_flat]
    raise MissingMapsError(
        f"{rsa_model}: step 7 needs the step-3 group mean map\n  {rel}\n"
        "It was not produced in this run and is not in the result sources. "
        "Add step 3 to --steps, or drop the file into the results folder.")


# ===========================================================================
# result zip
# ===========================================================================
def group_output_globs(manifest, rsa_model):
    """Data-root-relative glob patterns covering every step 3/5/6/7 output."""
    d, m, s = manifest["dataset"], manifest["model"], manifest["specie"]
    return [
        f"{d}/results/RSA/{m}/{rsa_model}/mean/*",
        f"{d}/results/RSA_rnd/{m}/{rsa_model}/mean/*",
        f"{d}/results/RSA_rnd/{m}/{s}-{rsa_model}_mean.nii.gz",
        f"{d}/results/RSA_rnd/{m}/{s}-{rsa_model}_std.nii.gz",
        f"{d}/results/RSA_rnd/{m}/{s}-{rsa_model}_mean_log.txt",
    ]


def zip_group_result(work_root, manifest, rsa_model, out_dir):
    """Zip one model's group outputs -> ``result_group_{rsa_model}_{specie}.zip``.

    Arcnames are pipeline-relative to the data folder, the same convention
    ``gpu_rsa.zip_model_result`` uses, so ``tools/unpack_results.py`` merges this
    zip exactly like a step-1/2/4 one.
    """
    data_root = os.path.join(work_root, "data")
    paths = []
    for pattern in group_output_globs(manifest, rsa_model):
        paths += glob.glob(os.path.join(data_root, pattern.replace("/", os.sep)))
    paths = sorted(p for p in dict.fromkeys(paths) if os.path.isfile(p))
    name = f"result_group_{rsa_model}_{manifest['specie']}.zip"
    zip_path = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for p in paths:
            zf.write(p, arcname=os.path.relpath(p, data_root).replace(os.sep, "/"))
    return zip_path, len(paths)

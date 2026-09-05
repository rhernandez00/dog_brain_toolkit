#!/usr/bin/env python
"""gpu_step5.py -- RSA pipeline step 5 on a GPU, straight out of the Colab result zips.

Step 5 (``rsa_utils.calculate_group_model_similarity_map_rnd``) builds the group
null: for each of ``reps_group`` group permutations it draws **one** of every
participant's ``reps`` step-4 permutation maps and averages them voxelwise. The
CPU does that with ``reps_group x n_units`` NIfTI loads -- 1000 x 32 = 32 000
reads of the same 3 200 files.

Here every map is read **exactly once**. A map that was drawn for group
permutations ``g1, g2, ...`` is scatter-added into those rows of a running
``(reps_group, n_voxels)`` accumulator on the GPU, so the whole of step 5 is one
pass over the data and the run is bounded by reading the zips, not by arithmetic.

Why this file needs neither a package nor the network disk
----------------------------------------------------------
The sibling ``gpu_group.py`` runs steps 3/5/6/7 and needs ``pkg_group_*.zip``
from ``tools/create_group_package.py`` -- a manifest and, crucially, the
searchlight mask. Step 5 on its own needs neither:

* **No manifest.** Every parameter step 5 uses -- dataset, GLM model, RSA model,
  specie, radius, ``dis_method``, ``rsa_method``, ``mah_fold``, the per-run
  layout, which participants exist and which permutation indices each of them
  has -- is written into the arcnames inside the result zips. :func:`scan_results`
  reads them back out.

* **No mask.** Step 5 calls ``nifti_mean(files_list, result_map_path=...)``
  *without* ``mask_img`` (rsa_utils.py), so it is a plain voxelwise mean over
  full volumes. A voxel that is 0 in every input is therefore 0 in the output,
  which means restricting the arithmetic to the union of the inputs' support is
  **exact**, not an approximation -- and that union is discovered from the maps
  themselves. It matters for memory: the EmoC human grid is 91x109x91 = 902 629
  voxels, of which ~154 000 carry a searchlight value, so the accumulator is
  1.2 GB instead of 7.2 GB.

  The support is grown as maps arrive rather than taken from the first map,
  because a Kendall tau really can come out exactly 0.0: on EmoC H-sub-01 about
  1 000 voxels differ in support between any two of that participant's maps.
  Growing it is safe -- a voxel appearing for the first time in map *j* was 0 in
  every map before it, so the column inserted into the accumulator is zero.

Faithfulness to the CPU step
----------------------------
* float64 throughout, and the output volume is ``nib.Nifti1Image(data, affine)``
  with a default header, exactly what ``nifti_mean`` writes.
* The draw is uniform with replacement over the permutation indices that
  actually exist for that unit, like ``random.choice(indices)`` on the CPU, and
  a unit with no permutation maps at all is skipped -- also like the CPU.
* Filenames, folders and the availability gate follow
  ``calculate_group_model_similarity_map_rnd``: the group maps carry **no**
  ``{mask_type}-`` prefix (only the participant maps do), and ``mah_fold``
  sub-foldering applies to the participant paths only, never to ``mean/``.
* The one deliberate difference is ``gpu_group.py``'s: the draw is seeded (from
  the model name), so a rerun reproduces itself where the CPU's unseeded
  ``random.choice`` does not. Same seed formula as ``gpu_group.run_group_model``,
  so a step-5-only run and a 3/5/6/7 run draw the same permutations.

Availability gate
-----------------
The CPU compares the number of units it found maps for against the number of
units in the *config's* participant list. There is no config here, so the
denominator is stated explicitly:

* ``expected_participants=<int>``  -- the number you know the config has;
* ``expected_participants='auto'`` -- every ``sub-NN`` that appears in *any*
  ``result_*_{specie}-sub-NN.zip`` in the results folder (so a participant who
  finished other models but not this one still counts as expected);
* ``expected_participants='found'`` -- the units found, i.e. always 100%. The
  gate is then vacuous and :func:`run_step5` says so.

Availability is measured in **participants**, not participant-runs: maps arrive
one zip per participant, so a missing participant is missing all of their runs.

Dependency-light (torch, numpy, nibabel) and self-contained, so it can simply be
dropped next to the notebook on Drive.

CLI (local testing; on Colab call :func:`run_step5_all` from the notebook):

    python tools/colab_gpu/gpu_step5.py --results "G:/My Drive/rsa_colab/results" \
        --out "G:/My Drive/rsa_colab/results" --specie H \
        --models action_tendency__all --reps_group 1000 \
        --min_percentage_available 0.5 --cpu
"""

import argparse
import collections
import glob
import gzip
import io
import itertools
import os
import re
import shutil
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import nibabel as nib
import torch

DTYPE = torch.float64


# ===========================================================================
# Voxel-grid validation
#
# Mirror of ``rsa_utils.check_same_space`` / ``gpu_rsa.check_same_space``,
# duplicated rather than imported because this file is meant to travel to Colab
# on its own. Keep the three in sync.
#
# Step 5 averages maps by array index and writes the result under a single
# affine, so every map going in must sit on the same voxel grid -- same shape
# AND same affine. Matching shapes prove nothing (see CLAUDE.md).
# ===========================================================================
SPACE_TOLERANCE_MM = 0.5


class SpaceMismatchError(ValueError):
    """Raised when maps that must share a voxel grid do not."""


class MissingMapsError(RuntimeError):
    """Not enough participant maps to build the group null."""


def grid_offset_mm(affine_a, affine_b, shape):
    """Worst-case displacement in mm between two voxel grids over a volume."""
    shape = tuple(int(s) for s in shape[:3])
    corners = np.array(list(itertools.product(*[(0, s - 1) for s in shape])), dtype=float)
    centre = (np.array(shape, dtype=float) - 1.0) / 2.0
    points = np.vstack([corners, centre])
    hom = np.c_[points, np.ones(len(points))]
    pos_a = (np.asarray(affine_a, dtype=float) @ hom.T).T[:, :3]
    pos_b = (np.asarray(affine_b, dtype=float) @ hom.T).T[:, :3]
    return float(np.linalg.norm(pos_a - pos_b, axis=1).max())


def check_grid(label, shape, affine, ref_shape, ref_affine, strict=True):
    """Compare one map's grid against the reference established by the first map."""
    if tuple(shape[:3]) != tuple(ref_shape[:3]):
        problem = f"{label}: shape {tuple(shape[:3])} != {tuple(ref_shape[:3])}"
    else:
        offset = grid_offset_mm(affine, ref_affine, ref_shape)
        if offset <= SPACE_TOLERANCE_MM:
            return True
        problem = f"{label}: same shape but grid is {offset:.1f} mm away"
    message = ("Step 5 averages these maps by array index, so they must share a "
               f"voxel grid.\n  {problem}")
    if strict:
        raise SpaceMismatchError(message)
    print(f"WARNING: {message}")
    return False


def pick_device(prefer_gpu=True):
    """Return a torch device: cuda if available and requested, else cpu."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ===========================================================================
# Reading the result zips
# ===========================================================================
# EmoC/results/RSA_rnd/basic-block/action_tendency__all[/{mah_fold}]/H-sub-01
#     [/ses-01_task-EmoC_run-02]/b_GreyMatter2mmB-r-4_mahalanobis_kendall_0000.nii.gz
_SUB_RE = re.compile(r"^(?P<specie>[DH])-sub-(?P<sub>\d+)$")
_RUN_RE = re.compile(r"^ses-(?P<session>\d+)_task-(?P<task>.+)_run-(?P<run>\d+)$")
_STEM_RE = re.compile(
    r"^(?:(?P<mask_type>.+?)-)?r-(?P<radius>\d+)"
    r"_(?P<dis_method>[^_]+)_(?P<rsa_method>[^_]+)"
    r"(?:_(?P<rnd_index>\d+))?\.nii\.gz$")


def parse_arcname(rel):
    """Decode one participant-map arcname into its pipeline parameters.

    Returns a dict, or ``None`` when the member is not a participant
    model-similarity map (a group ``mean/`` map, a log, anything else).
    """
    parts = rel.replace("\\", "/").lstrip("./").split("/")
    if len(parts) < 6 or parts[1] != "results" or parts[2] not in ("RSA", "RSA_rnd"):
        return None
    dataset, _results, kind, glm_model, rsa_model = parts[:5]
    rest = parts[5:]

    mah_fold = None
    if not _SUB_RE.match(rest[0]):
        mah_fold, rest = rest[0], rest[1:]     # fold-isolated participant root
    if not rest:
        return None
    sub_m = _SUB_RE.match(rest[0])
    if not sub_m:
        return None                            # e.g. the group 'mean/' folder
    rest = rest[1:]

    session = run_N = task = None
    if len(rest) == 2:
        run_m = _RUN_RE.match(rest[0])
        if not run_m:
            return None
        session, run_N = int(run_m.group("session")), int(run_m.group("run"))
        task = run_m.group("task")
        rest = rest[1:]
    if len(rest) != 1:
        return None
    stem_m = _STEM_RE.match(rest[0])
    if not stem_m:
        return None

    rnd = kind == "RSA_rnd"
    idx = stem_m.group("rnd_index")
    if rnd != (idx is not None):
        return None                            # rnd maps are indexed, real ones are not
    return {
        "dataset": dataset, "model": glm_model, "rsa_model": rsa_model,
        "mah_fold": mah_fold, "specie": sub_m.group("specie"),
        "sub_N": int(sub_m.group("sub")), "session": session, "run_N": run_N,
        "task": task, "mask_type": stem_m.group("mask_type"),
        "radius": int(stem_m.group("radius")),
        "dis_method": stem_m.group("dis_method"),
        "rsa_method": stem_m.group("rsa_method"),
        "rnd": rnd, "rnd_index": None if idx is None else int(idx),
    }


class ModelSpec:
    """Everything step 5 needs for one (rsa_model, specie), read off the zips.

    ``units`` is the ordered list of averaging units -- ``(sub_N, session, run_N)``
    with ``session``/``run_N`` ``None`` for the Mahalanobis stim-wise layout, the
    same enumeration ``calculate_group_model_similarity_map_rnd`` walks.
    ``refs[unit]`` maps a permutation index to ``(zip_path, member)``.
    """

    def __init__(self, rsa_model, specie, params, refs, participants_seen):
        self.rsa_model = rsa_model
        self.specie = specie
        self.params = params                        # dataset/model/radius/dis/rsa/...
        self.refs = refs                            # {unit: {rnd_index: ref}}
        self.units = sorted(refs)
        self.participants = sorted({u[0] for u in refs})
        self.participants_seen = participants_seen  # every sub-NN in the folder
        self.counts = np.array([len(refs[u]) for u in self.units], dtype=np.int64)

    # -- output paths (mirror rsa_utils / gpu_group: the group maps carry no
    #    mask prefix, and mah_fold never reaches the group 'mean/' folder) ----
    @property
    def group_stem(self):
        p = self.params
        return f"{self.specie}-r-{p['radius']}_{p['dis_method']}_{p['rsa_method']}"

    def group_mean_rel(self, index):
        p = self.params
        return "/".join([p["dataset"], "results", "RSA_rnd", p["model"],
                         self.rsa_model, "mean",
                         f"{self.group_stem}_mean_{index:05d}.nii.gz"])

    def describe(self):
        p = self.params
        fold = f"/{p['mah_fold']}" if p.get("mah_fold") else ""
        return (f"{self.rsa_model} [{self.specie}]  {p['dataset']}/{p['model']}  "
                f"r-{p['radius']}  {p['dis_method']}{fold}/{p['rsa_method']}  "
                f"mask={p['mask_type']}  units={len(self.units)}  "
                f"perm maps={int(self.counts.sum())}")


def scan_results(results_dir, specie=None, models=None, verbose=True):
    """Index the ``result_*.zip`` in a folder into one :class:`ModelSpec` per model.

    Only zips whose *name* says they hold a wanted model are opened -- the naming
    convention is ``gpu_rsa.zip_model_result``'s
    ``result_{rsa_model}_{specie}-sub-NN.zip`` -- so selecting one model out of a
    92-model battery costs one directory listing plus that model's own zips.
    """
    results_dir = os.path.abspath(str(results_dir))
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results folder not found: {results_dir}")
    names = [n for n in os.listdir(results_dir) if n.lower().endswith(".zip")]

    name_re = re.compile(
        r"^result_(?P<model>.+)_(?P<specie>[DH])-sub-(?P<sub>\d+)\.zip$", re.IGNORECASE)
    seen = collections.defaultdict(set)          # specie -> {sub_N}
    wanted = []
    for n in names:
        if n.startswith("result_group_") or n.startswith("result_step5_"):
            continue                             # our own / the 3-5-6-7 run's output
        m = name_re.match(n)
        if not m:
            continue
        sp = m.group("specie").upper()
        seen[sp].add(int(m.group("sub")))
        if specie and sp != specie:
            continue
        if models and m.group("model") not in models:
            continue
        wanted.append(os.path.join(results_dir, n))

    if verbose:
        tally = ", ".join(f"{k}:{len(v)}" for k, v in sorted(seen.items())) or "none"
        print(f"[scan] {results_dir}: {len(names)} zip(s), {len(wanted)} match the "
              f"selection (participants seen -- {tally})")
    if not wanted:
        raise MissingMapsError(
            "No result zips matched. Expected files named "
            f"result_{{rsa_model}}_{{specie}}-sub-NN.zip in {results_dir}")

    acc = {}                                     # (rsa_model, specie) -> params/refs
    conflicts = collections.defaultdict(set)
    for zip_path in sorted(wanted):
        with zipfile.ZipFile(zip_path) as zf:
            members = zf.namelist()
        for member in members:
            info = parse_arcname(member)
            if info is None or not info["rnd"]:
                continue                         # step 5 reads only the rnd maps
            key = (info["rsa_model"], info["specie"])
            params = {k: info[k] for k in ("dataset", "model", "task", "radius",
                                           "mask_type", "dis_method", "rsa_method",
                                           "mah_fold")}
            entry = acc.setdefault(key, {"params": params, "refs": {}})
            for k, v in params.items():
                if k == "task" and v is None:
                    continue                     # only the per-run layout names it
                if entry["params"].get(k) != v:
                    conflicts[key].add(k)
            unit = (info["sub_N"], info["session"], info["run_N"])
            entry["refs"].setdefault(unit, {})[info["rnd_index"]] = (zip_path, member)

    if conflicts:
        detail = "; ".join(f"{m} [{s}]: {sorted(k)}" for (m, s), k in conflicts.items())
        raise ValueError(
            "Result zips for the same model disagree on pipeline parameters "
            f"({detail}). Mixing settings would build one null distribution out of "
            "two analyses -- move the odd zips out of the results folder.")

    specs = {key: ModelSpec(key[0], key[1], entry["params"], entry["refs"],
                            sorted(seen[key[1]]))
             for key, entry in acc.items()}
    if verbose:
        for key in sorted(specs):
            print(f"[scan]   {specs[key].describe()}")
    return specs


# ===========================================================================
# The draw
# ===========================================================================
def draw_group_indices(counts, reps_group, seed):
    """Pick one permutation map per unit for each of ``reps_group`` group perms.

    Returns a ``(reps_group, n_units)`` array of indices **local to each unit**,
    i.e. positions into that unit's sorted list of available permutation indices.
    Uniform with replacement over the permutations that actually exist, like
    ``random.choice(indices)`` in the CPU step -- but seeded. The rng is consumed
    one unit at a time, in the same order as ``gpu_group.draw_group_indices``, so
    both ports draw identically for the same seed.
    """
    rng = np.random.default_rng(seed)
    cols = np.empty((reps_group, len(counts)), dtype=np.int64)
    for u, n in enumerate(counts):
        if n <= 0:
            raise ValueError(f"unit {u} has no permutation maps")
        cols[:, u] = rng.integers(0, int(n), size=reps_group)
    return cols


def default_seed(rsa_model, specie, reps_group):
    """Same formula as ``gpu_group.run_group_model``, so the two ports agree."""
    return zlib.crc32(f"group-{rsa_model}-{specie}-{reps_group}".encode())


# ===========================================================================
# The accumulator
# ===========================================================================
def _img_from_bytes(raw):
    """Decode a ``.nii.gz`` blob into a nibabel image without touching disk."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    fh = nib.FileHolder(fileobj=io.BytesIO(raw))
    return nib.Nifti1Image.from_file_map({"header": fh, "image": fh})


class GroupAccumulator:
    """``(reps_group, n_support_voxels)`` running sum of the drawn maps.

    Maps are added one at a time: map *j* of unit *u* is scatter-added into the
    rows that drew it. The support grows as maps arrive -- a voxel first seen in
    map *j* was 0 in every earlier map, so the column inserted for it is zero and
    the result is exactly what a full-volume sum would have given.
    """

    def __init__(self, reps_group, n_voxels, device):
        self.reps_group = int(reps_group)
        self.n_voxels = int(n_voxels)
        self.device = device
        self.support = np.zeros(0, dtype=np.int64)   # sorted flat voxel indices
        self._in_support = np.zeros(self.n_voxels, dtype=bool)
        self.acc = torch.zeros((self.reps_group, 0), dtype=DTYPE, device=device)
        self.n_expansions = 0

    def _grow(self, new_voxels):
        """Insert zero columns for voxels seen for the first time."""
        merged = np.union1d(self.support, new_voxels)
        grown = torch.zeros((self.reps_group, merged.size), dtype=DTYPE,
                            device=self.device)
        if self.support.size:
            old_pos = torch.as_tensor(np.searchsorted(merged, self.support),
                                      device=self.device)
            grown.index_copy_(1, old_pos, self.acc)
        del self.acc
        self.acc = grown
        self.support = merged
        self._in_support[new_voxels] = True
        self.n_expansions += 1

    def add(self, flat, rows):
        """Add one map (a flat float64 volume) into the given accumulator rows."""
        if rows.size == 0:
            return
        active = np.flatnonzero(~np.isfinite(flat) | (flat != 0.0))
        new = active[~self._in_support[active]]
        if new.size:
            self._grow(new)
        vec = torch.as_tensor(flat[self.support], dtype=DTYPE, device=self.device)
        idx = torch.as_tensor(rows, device=self.device)
        # a map is typically drawn by reps_group/reps rows, so the repeat is small
        self.acc.index_add_(0, idx, vec.repeat(idx.numel(), 1))

    def means(self, n_units):
        """The group means: the running sum divided by the number of units."""
        return (self.acc / float(n_units)).cpu().numpy()


class _ZipReader:
    """Thread-safe reader that keeps one open ``ZipFile`` per thread.

    ``zipfile.ZipFile`` is not safe for concurrent reads from one handle, and
    reopening per map would pay the central-directory read (and, on Drive, the
    open latency) 3 200 times. Jobs arrive zip-major, so one cached handle per
    thread is almost always the right one.
    """

    def __init__(self):
        self._local = threading.local()
        self._all = []
        self._lock = threading.Lock()

    def read(self, zip_path, member):
        cache = getattr(self._local, "cache", None)
        if cache is None:
            cache = self._local.cache = {}
        zf = cache.get(zip_path)
        if zf is None:
            zf = cache[zip_path] = zipfile.ZipFile(zip_path)
            with self._lock:
                self._all.append(zf)
        return zf.read(member)

    def close(self):
        with self._lock:
            for zf in self._all:
                try:
                    zf.close()
                except Exception:
                    pass
            self._all = []


def _stream_maps(spec, cols, orders, accum, ref_grid, strict_space,
                 read_workers=8, prefetch=None, progress=None):
    """Read every drawn map once and add it into the accumulator.

    Only the maps some group permutation actually drew are read; with
    ``reps_group=1000`` over 100 permutations that is all of them, but with a
    small ``reps_group`` it is many fewer.

    Reads run ahead on a thread pool -- they are latency-bound on Drive, and
    gzip releases the GIL -- but the results are **applied in job order**, so the
    run stays bit-identical to a serial one. Order matters twice over: the
    accumulator's float64 sums are order-dependent, and a support expansion has
    to see the maps in a fixed sequence. Within one unit it would not matter
    (each accumulator row takes exactly one map per unit), but across units it
    does, and applying in order costs nothing.
    """
    jobs = []                                   # (unit_index, zip, member, rows)
    for u, unit in enumerate(spec.units):
        col = cols[:, u]
        for j in np.unique(col):
            zip_path, member = spec.refs[unit][orders[u][int(j)]]
            jobs.append((u, zip_path, member, np.flatnonzero(col == j)))
    # zip-major within a unit, so a thread's cached handle stays the right one
    jobs.sort(key=lambda t: (t[0], t[1], t[2]))

    reader = _ZipReader()
    workers = max(1, int(read_workers))
    prefetch = prefetch or max(2 * workers, 4)
    pending = collections.deque()
    it = iter(jobs)
    n_read = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            def submit_next():
                try:
                    u, zip_path, member, rows = next(it)
                except StopIteration:
                    return False
                pending.append((u, member, rows,
                                ex.submit(reader.read, zip_path, member)))
                return True

            for _ in range(prefetch):
                if not submit_next():
                    break
            while pending:
                u, member, rows, fut = pending.popleft()
                img = _img_from_bytes(fut.result())
                check_grid(member, img.shape, img.affine, ref_grid["shape"],
                           ref_grid["affine"], strict=strict_space)
                flat = np.asarray(img.dataobj, dtype=np.float64).reshape(-1)
                if flat.size != accum.n_voxels:
                    raise SpaceMismatchError(
                        f"{member}: {flat.size} voxels != {accum.n_voxels}")
                accum.add(flat, rows)
                n_read += 1
                if progress is not None:
                    progress(u, n_read, len(jobs))
                submit_next()
    finally:
        reader.close()
    return n_read, len(jobs)


# ===========================================================================
# Step 5
# ===========================================================================
def save_volume(vec, support, shape, affine, path):
    """Scatter a support vector into a full volume and write it as ``.nii.gz``.

    Off-support voxels are 0, which is what ``nifti_mean`` produces there (a mean
    of zeros), and the image is float64 under a default header -- exactly
    ``nib.Nifti1Image(mean_data, img_affine)``.
    """
    vol = np.zeros(int(np.prod(shape)), dtype=np.float64)
    vol[support] = vec
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(vol.reshape(shape), affine), path)
    return path


def resolve_expected_participants(spec, expected_participants):
    """Denominator of the availability check -- see the module docstring."""
    if expected_participants in (None, "found"):
        return len(spec.participants), "the participants found (gate is vacuous)"
    if expected_participants == "auto":
        return (len(spec.participants_seen),
                f"every {spec.specie}-sub-NN seen in the results folder")
    n = int(expected_participants)
    if n <= 0:
        raise ValueError("expected_participants must be positive")
    return n, "given explicitly"


def run_step5(spec, out_dir, reps_group=1000, work_root=None, device=None, seed=None,
              min_percentage_available=1.0, expected_participants="auto", workers=8,
              read_workers=8, allow_space_mismatch=False, keep_work=False,
              verbose=True):
    """Run step 5 for one model and write ``result_step5_{rsa_model}_{specie}.zip``.

    Returns ``(zip_path, n_files)``.
    """
    device = device or pick_device()
    work_root = work_root or os.path.join(os.path.abspath(out_dir), "_step5_work")
    n_units = len(spec.units)
    if n_units == 0:
        raise MissingMapsError(
            f"{spec.rsa_model}: no step-4 permutation maps found in any zip.")

    expected, how = resolve_expected_participants(spec, expected_participants)
    pct = len(spec.participants) / expected if expected else 0.0
    if verbose:
        print(f"=== step 5: {spec.describe()} ===")
        print(f"    participants {len(spec.participants)}/{expected} "
              f"({pct * 100:.1f}%) -- denominator: {how}")
        print(f"    permutations per unit: min={int(spec.counts.min())} "
              f"max={int(spec.counts.max())}   reps_group={reps_group}   "
              f"device={device}   read_workers={read_workers}")
    if pct < min_percentage_available:
        raise MissingMapsError(
            f"{spec.rsa_model}: only {pct * 100:.1f}% of the participants have "
            f"permutation maps ({len(spec.participants)}/{expected}); "
            f"min_percentage_available is {min_percentage_available * 100:.1f}%.")
    if expected_participants in (None, "found") and min_percentage_available > 0:
        print("    NOTE: expected_participants resolves to the number found, so "
              "the availability check cannot fail. Pass 'auto' or a count to make "
              "it meaningful.")

    if seed is None:
        seed = default_seed(spec.rsa_model, spec.specie, reps_group)
    cols = draw_group_indices(spec.counts, reps_group, seed)
    # sorted available permutation indices per unit; cols holds positions into these
    orders = [sorted(spec.refs[u]) for u in spec.units]

    # the first map fixes the grid, and with it the accumulator's width
    first_zip, first_member = spec.refs[spec.units[0]][orders[0][0]]
    with zipfile.ZipFile(first_zip) as zf:
        first_img = _img_from_bytes(zf.read(first_member))
    ref_grid = {"shape": first_img.shape, "affine": first_img.affine}
    n_voxels = int(np.prod(first_img.shape))
    accum = GroupAccumulator(reps_group, n_voxels, device)

    t0 = time.time()
    state = {"unit": -1}

    def progress(u, n_read, n_total):
        if not verbose or u == state["unit"]:
            return
        state["unit"] = u
        unit = spec.units[u]
        label = (f"sub-{unit[0]:02d}" if unit[1] is None
                 else f"sub-{unit[0]:02d} ses-{unit[1]:02d} run-{unit[2]:02d}")
        print(f"[step5] unit {u + 1}/{n_units} {label}: {n_read}/{n_total} "
              f"map(s) read, support={accum.support.size}, {time.time() - t0:.1f}s")

    n_read, n_jobs = _stream_maps(spec, cols, orders, accum, ref_grid,
                                  strict_space=not allow_space_mismatch,
                                  read_workers=read_workers, progress=progress)

    group_means = accum.means(n_units)
    if verbose:
        print(f"[step5] {reps_group} group permutation(s) over {n_units} unit(s) "
              f"from {n_read}/{n_jobs} map(s) in {time.time() - t0:.1f}s "
              f"(seed={seed}, support={accum.support.size}/{n_voxels} voxels, "
              f"{accum.n_expansions} support expansion(s))")

    data_root = os.path.join(work_root, "data")
    shutil.rmtree(data_root, ignore_errors=True)
    t1 = time.time()
    jobs = [dict(vec=group_means[g], support=accum.support, shape=ref_grid["shape"],
                 affine=ref_grid["affine"],
                 path=os.path.join(data_root,
                                   spec.group_mean_rel(g).replace("/", os.sep)))
            for g in range(reps_group)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        paths = list(ex.map(lambda kw: save_volume(**kw), jobs))
    if verbose:
        print(f"[step5] wrote {len(paths)} group mean map(s) in "
              f"{time.time() - t1:.1f}s")

    zip_path, n_files = zip_step5_result(data_root, spec, out_dir)
    if not keep_work:
        shutil.rmtree(data_root, ignore_errors=True)
    if verbose:
        print(f"[step5] {n_files} file(s), "
              f"{os.path.getsize(zip_path) / 1e6:.1f} MB -> {zip_path}")
    return zip_path, n_files


def zip_step5_result(data_root, spec, out_dir):
    """Zip the group mean maps with pipeline-relative arcnames.

    Named ``result_step5_...`` rather than ``result_group_...``:
    ``tools/unpack_results.py`` merges any ``result_*.zip``, but the 3/5/6/7
    notebook treats ``result_group_{model}_{specie}.zip`` as "this model is
    done" and would skip the steps 6-7 a step-5-only run has not produced.
    """
    paths = sorted(p for p in glob.glob(os.path.join(data_root, "**", "*.nii.gz"),
                                        recursive=True) if os.path.isfile(p))
    name = f"result_step5_{spec.rsa_model}_{spec.specie}.zip"
    os.makedirs(out_dir, exist_ok=True)
    zip_path = os.path.join(out_dir, name)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for p in paths:
            zf.write(p, arcname=os.path.relpath(p, data_root).replace(os.sep, "/"))
    return zip_path, len(paths)


def run_step5_all(results_dir, out_dir, specie=None, models=None, reps_group=1000,
                  work_root=None, device=None, min_percentage_available=1.0,
                  expected_participants="auto", workers=8, read_workers=8,
                  allow_space_mismatch=False, force=False, verbose=True):
    """Run step 5 for every model found. Resumable: skips models already zipped."""
    specs = scan_results(results_dir, specie=specie, models=models, verbose=verbose)
    device = device or pick_device()
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for i, key in enumerate(sorted(specs), 1):
        spec = specs[key]
        zip_path = os.path.join(
            out_dir, f"result_step5_{spec.rsa_model}_{spec.specie}.zip")
        if os.path.exists(zip_path) and not force:
            if verbose:
                print(f"[{i}/{len(specs)}] {spec.rsa_model} [{spec.specie}]: "
                      "result exists -- skipping.")
            continue
        try:
            path, _ = run_step5(
                spec, out_dir, reps_group=reps_group, work_root=work_root,
                device=device, min_percentage_available=min_percentage_available,
                expected_participants=expected_participants, workers=workers,
                read_workers=read_workers,
                allow_space_mismatch=allow_space_mismatch, verbose=verbose)
        except MissingMapsError as exc:
            print(f"[{i}/{len(specs)}] {spec.rsa_model} [{spec.specie}]: "
                  f"SKIPPED -- {exc}")
            continue
        written.append(path)
    if verbose:
        print(f"=== finished: {len(written)} new step-5 zip(s) in {out_dir} ===")
    return written


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run RSA step 5 on a GPU from the Colab result zips.")
    ap.add_argument("--results", required=True,
                    help="Folder holding result_{model}_{specie}-sub-NN.zip")
    ap.add_argument("--out", required=True,
                    help="Output folder for result_step5_*.zip (can be the same)")
    ap.add_argument("--specie", choices=["D", "H"], default=None)
    ap.add_argument("--models", nargs="*", default=None,
                    help="RSA models to run (default: every model in the folder)")
    ap.add_argument("--reps_group", type=int, default=1000,
                    help="Group permutations to build (default 1000)")
    ap.add_argument("--work", default=None, help="Scratch dir for the maps")
    ap.add_argument("--min_percentage_available", type=float, default=1.0,
                    help="Required fraction of participants (default 1.0)")
    ap.add_argument("--expected_participants", default="auto",
                    help="'auto' (every sub-NN in the folder), an integer, or "
                         "'found' to disable the check")
    ap.add_argument("--workers", type=int, default=8,
                    help="Threads for writing the output niftis")
    ap.add_argument("--read_workers", type=int, default=8,
                    help="Threads prefetching maps out of the result zips. Reads "
                         "are latency-bound on Drive and dominate the run; the "
                         "maps are still applied in order, so the result is "
                         "unchanged")
    ap.add_argument("--allow_space_mismatch", action="store_true",
                    help="Downgrade the voxel-grid check to a warning")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if the result zip exists")
    return ap.parse_args()


def main():
    a = parse_args()
    expected = a.expected_participants
    if expected not in ("auto", "found"):
        expected = int(expected)
    run_step5_all(a.results, a.out, specie=a.specie, models=a.models,
                  reps_group=a.reps_group, work_root=a.work,
                  device=torch.device("cpu") if a.cpu else pick_device(),
                  min_percentage_available=a.min_percentage_available,
                  expected_participants=expected, workers=a.workers,
                  read_workers=a.read_workers,
                  allow_space_mismatch=a.allow_space_mismatch, force=a.force)


if __name__ == "__main__":
    main()

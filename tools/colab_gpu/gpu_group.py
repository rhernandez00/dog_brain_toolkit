#!/usr/bin/env python
"""gpu_group.py -- GPU (PyTorch) reimplementation of RSA pipeline steps 3, 5, 6, 7, 8.

Companion to ``gpu_rsa.py`` (steps 1, 2, 4). Where that module works on *one*
participant, this one works on the *group*: it consumes the per-participant maps
that a Colab run already produced -- the ``result_{model}_{specie}-sub-NN.zip``
files sitting in OUT_DIR -- and reduces them to what steps 9-10 expect:

  * Step 3 -- ``calculate_group_model_similarity_map``   (mean/std of the real maps)
  * Step 5 -- ``calculate_group_model_similarity_map_rnd`` (reps_group group perms)
  * Step 6 -- ``calculate_voxelwise_rnd_distribution``   (mean/std across those)
  * Step 7 -- ``calculate_z_maps_rnd`` + ``calculate_z_map_real_data``
  * Step 8 -- ``calculate_cluster_size_distribution``    (cluster sizes per z map)

Why step 8 belongs here rather than on the workstation: steps 5 and 7 each write
``reps_group`` whole-brain maps, and they are read by steps 6-7 and step 8
respectively -- all of which now run in this same pass. Measured on EmoC humans
at ``reps_group=1000`` that is 727 MB + 1273 MB per model of pure intermediate.
With ``write_group_means``/``write_z_maps`` off (the default) a model ships about
4 MB instead of 2 GB, and step 8 works on the z maps while they are still in
memory. Step 8 also computes **several thresholds in one pass**, since labelling
is cheap next to reading the participant maps and the z maps are not kept.

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

import collections
import contextlib
import csv
import datetime
import glob
import gzip
import io
import json
import os
import re
import shutil
import sys
import threading
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

# ===========================================================================
# Version -- bump VERSION and rewrite LAST_CHANGE on every edit to this file.
#
# This half of the toolkit runs somewhere else: three files are copied to Drive
# by hand (gpu_group.py, run_colab_group.py, colab_rsa_group.ipynb) and Colab
# caches imported modules across cell runs, so "am I running the code I just
# fixed?" is a real and recurring question. A single number cannot answer it,
# because any one of the three can be stale on its own -- hence
# ``check_versions()``, which compares all three and says which to re-copy.
#
# Patch bump for a fix or tweak, minor for a new parameter or behaviour, major
# for anything that changes what a run produces or what it needs as input.
# ===========================================================================
VERSION = "4.4.0"
LAST_CHANGE = (
    "No kernel change here -- bumped to stay in lockstep with "
    "run_colab_group.py, which now writes a .started marker per model in "
    "out_dir before processing begins and removes it on a clean finish. A "
    "marker surviving with no matching result zip means the runtime died "
    "mid-model; the next run skips that model instead of retrying it into the "
    "same crash. Delete the marker, or pass force=True, to retry deliberately."
)

DTYPE = torch.float64
STEPS_ALL = (3, 5, 6, 7, 8)
DEFAULT_Z_THRESHOLDS = (3.1, 3.5, 4.0, 4.5, 5.0)


def version_banner():
    """One line naming the version, for the top of a run."""
    return f"gpu_group v{VERSION}"


def check_versions(notebook_version=None, strict=True, verbose=True):
    """Confirm the files that travel to Drive are all from the same release.

    Colab keeps an imported module across cell runs, and the three files are
    copied over separately, so the common failure is running a fixed
    ``gpu_group.py`` against a stale notebook, or the reverse. Compares
    ``gpu_group.VERSION``, ``run_colab_group.VERSION`` and the notebook's own
    constant, and names whichever is behind.

    Returns True when they agree. With ``strict`` it raises instead of warning,
    because a silent mismatch is what wastes the next hour.
    """
    found = {"gpu_group.py": VERSION}
    try:
        import run_colab_group
        found["run_colab_group.py"] = getattr(run_colab_group, "VERSION", "?")
    except ImportError:
        found["run_colab_group.py"] = "(not importable)"
    if notebook_version is not None:
        found["colab_rsa_group.ipynb"] = str(notebook_version)

    if verbose:
        print(f"  {version_banner()}  --  {LAST_CHANGE[:96]}...")
        for name, v in found.items():
            print(f"    {name:24s} v{v}")
        print(f"    module loaded from       {os.path.abspath(__file__)}")

    versions = set(found.values())
    if len(versions) == 1:
        return True
    behind = [n for n, v in found.items() if v != VERSION]
    message = (
        f"Version mismatch across the Colab files: {found}.\n"
        f"These are behind gpu_group.py v{VERSION}: {', '.join(behind)}.\n"
        "Re-copy them to Drive from the workstation:\n"
        "  copy \\github\\dog_brain_toolkit\\tools\\colab_gpu\\gpu_group.py "
        "\"G:\\My Drive\\rsa_colab\\\"\n"
        "  copy \\github\\dog_brain_toolkit\\tools\\colab_gpu\\run_colab_group.py "
        "\"G:\\My Drive\\rsa_colab\\\"\n"
        "  copy \\github\\dog_brain_toolkit\\tools\\colab_gpu\\colab_rsa_group.ipynb "
        "\"G:\\My Drive\\rsa_colab\\\"\n"
        "then RESTART THE COLAB RUNTIME (Runtime -> Restart session) so the old "
        "module is dropped, and run from the top.")
    if strict:
        raise RuntimeError(message)
    print(f"WARNING: {message}")
    return False


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


# ---------------------------------------------------------------------------
# Manifest without a package: recover it from the result zips + a refs folder
#
# ``tools/create_group_package.py`` builds the manifest from the dataset config
# on the network disk, which makes every group run depend on that share being
# up. Almost none of it has to come from there: the arcnames inside the result
# zips already state the dataset, GLM model, RSA model, specie, radius,
# dis_method, rsa_method, mah_fold, mask_type, the per-run layout, and which
# permutation indices each participant has.
#
# Exactly three things do not survive in an arcname, and they live in
# ``tools/colab_gpu/refs/`` (built once by ``refs/build_refs.py``, committed, a
# few tens of kB):
#
#   * the mask -- needed as THE reference voxel grid, and to verify that every
#     participant map is zero outside it. The group arithmetic itself would be
#     fine without one (see gpu_step5's argument: off-mask voxels are 0 in every
#     input, so they stay 0), but the grid and that check are worth keeping;
#   * the participant list -- the denominator of the availability check. A
#     participant who has produced nothing leaves no trace on Drive, so "32
#     participants have maps" cannot become a percentage without it;
#   * ``runs_by_sub`` and ``task``, which only the per-run layouts need.
# ---------------------------------------------------------------------------
_SUB_RE = re.compile(r"^(?P<specie>[DH])-sub-(?P<sub>\d+)$")
_RUN_RE = re.compile(r"^ses-(?P<session>\d+)_task-(?P<task>.+)_run-(?P<run>\d+)$")
_STEM_RE = re.compile(
    r"^(?:(?P<mask_type>.+?)-)?r-(?P<radius>\d+)"
    r"_(?P<dis_method>[^_]+)_(?P<rsa_method>[^_]+)"
    r"(?:_(?P<rnd_index>\d+))?\.nii\.gz$")
_ZIP_NAME_RE = re.compile(
    r"^result_(?P<model>.+)_(?P<specie>[DH])-sub-(?P<sub>\d+)\.zip$", re.IGNORECASE)
# result_step1_mah_H-sub-03.zip matches _ZIP_NAME_RE with model="step1_mah", but
# it holds step-1 pairwise distance maps, not model-similarity maps -- "step1_mah"
# is not an RSA model. result_group_/result_step5_ are a previous group run's own
# output. All three have to be excluded by name, because discovery deliberately
# does not open zips to find out what is inside them.
_NOT_A_MODEL = ("result_step1_", "result_group_", "result_step5_")


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


def load_refs(refs_dir, dataset):
    """Read ``{dataset}_refs.json`` from a refs folder (see refs/build_refs.py)."""
    path = os.path.join(refs_dir, f"{dataset}_refs.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No reference snapshot at {path}. Build one on the workstation while "
            "the data share is up:\n"
            "  python \\github\\dog_brain_toolkit\\tools\\colab_gpu\\refs\\build_refs.py")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def refs_mask_path(refs_dir, dataset, specie, mask_type):
    """``refs/{dataset}/{specie}_{mask_type}.nii.gz``, the mask carried with the refs."""
    return os.path.join(refs_dir, dataset, f"{specie}_{mask_type}.nii.gz")


def load_group_mask(manifest, pkg_root=None, refs_dir=None):
    """The reference voxel grid, from a package's ``data/`` or from the refs folder."""
    if pkg_root:
        return gpu_rsa.load_reference_mask(os.path.join(pkg_root, "data"), manifest)
    path = refs_mask_path(refs_dir, manifest["dataset"], manifest["specie"],
                          manifest["mask_type"])
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Mask not found at {path}. The refs folder must carry the mask the "
            f"participant maps were computed against ({manifest['mask_type']!r}, "
            f"read from the result zips' arcnames).")
    img = nib.load(path)
    return img, np.asarray(img.dataobj).astype(bool)


def _open_zip_resilient(path, attempts=3):
    """Open a zip, retrying once or twice if the Drive mount hiccups.

    Colab's ``drive.mount`` FUSE endpoint drops under sustained access and every
    subsequent call raises ``OSError: [Errno 107] Transport endpoint is not
    connected`` until it is remounted. A short retry rides out a blip; a real
    disconnect is re-raised with an instruction, because nothing here can fix it.
    """
    last = None
    for i in range(attempts):
        try:
            return zipfile.ZipFile(path)
        except OSError as exc:
            last = exc
            if getattr(exc, "errno", None) != 107:
                raise
            time.sleep(1.0 + i)
    raise OSError(
        f"{last}\n\nThe Google Drive mount dropped while reading\n  {path}\n"
        "Re-run the mount cell (`drive.mount('/content/drive', force_remount=True)`) "
        "and run this cell again -- finished models are skipped, so it resumes.")


def scan_result_zip_names(results_dir, specie=None, models=None, verbose=True):
    """Which models and participants exist, from the **file names only**.

    ``gpu_rsa.zip_model_result`` names them
    ``result_{rsa_model}_{specie}-sub-NN.zip``, so the model and the participant
    are both in the name and nothing has to be opened. That matters: a full EmoC
    human battery is ~3 700 zips, and opening every one of them to read a
    namelist is what makes Colab's Drive FUSE endpoint drop
    (``OSError: [Errno 107]``). Returns
    ``({(rsa_model, specie): {sub_N: path}}, {specie: {sub_N}})``.
    """
    results_dir = os.path.abspath(str(results_dir))
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results folder not found: {results_dir}")
    by_model, seen = {}, {}
    n_zips = 0
    for n in os.listdir(results_dir):
        if not n.lower().endswith(".zip"):
            continue
        n_zips += 1
        if n.startswith(_NOT_A_MODEL):
            continue          # step-1 output, or a previous group run's
        m = _ZIP_NAME_RE.match(n)
        if not m:
            continue
        sp, sub = m.group("specie").upper(), int(m.group("sub"))
        seen.setdefault(sp, set()).add(sub)
        if specie and sp != specie:
            continue
        if models and m.group("model") not in models:
            continue
        by_model.setdefault((m.group("model"), sp), {})[sub] = \
            os.path.join(results_dir, n)
    if verbose:
        tally = ", ".join(f"{k}:{len(v)}" for k, v in sorted(seen.items())) or "none"
        n_sel = sum(len(v) for v in by_model.values())
        print(f"[scan] {results_dir}: {n_zips} zip(s) -> {len(by_model)} model(s), "
              f"{n_sel} zip(s) selected (participants seen -- {tally})")
    if not by_model:
        raise MissingMapsError(
            "No result zips matched. Expected files named "
            f"result_{{rsa_model}}_{{specie}}-sub-NN.zip in {results_dir}")
    return by_model, seen


def scan_result_zips(results_dir, specie=None, models=None, verbose=True):
    """Index a folder of ``result_*.zip`` into ``{(rsa_model, specie): info}``.

    Opens every selected zip, so it is only for the per-model paths that need
    the member list. Discovery uses :func:`scan_result_zip_names` instead.
    """
    results_dir = os.path.abspath(str(results_dir))
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Results folder not found: {results_dir}")
    names = [n for n in os.listdir(results_dir) if n.lower().endswith(".zip")]

    seen = {}                                    # specie -> {sub_N}
    wanted = []
    for n in names:
        if n.startswith(_NOT_A_MODEL):
            continue          # step-1 output, or a previous group run's
        m = _ZIP_NAME_RE.match(n)
        if not m:
            continue
        sp = m.group("specie").upper()
        seen.setdefault(sp, set()).add(int(m.group("sub")))
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

    found, conflicts = {}, {}
    for zip_path in sorted(wanted):
        with _open_zip_resilient(zip_path) as zf:
            members = zf.namelist()
        for member in members:
            info = parse_arcname(member)
            if info is None:
                continue
            key = (info["rsa_model"], info["specie"])
            params = {k: info[k] for k in ("dataset", "model", "task", "radius",
                                           "mask_type", "dis_method", "rsa_method",
                                           "mah_fold")}
            entry = found.setdefault(key, {"params": params, "units": {},
                                           "reps": 0, "zips": set()})
            entry["zips"].add(zip_path)
            for k, v in params.items():
                if k == "task" and v is None:
                    continue                     # only the per-run layout names it
                if entry["params"].get(k) != v:
                    conflicts.setdefault(key, set()).add(k)
            unit = (info["sub_N"], info["session"], info["run_N"])
            u = entry["units"].setdefault(unit, {"real": False, "rnd": set()})
            if info["rnd"]:
                u["rnd"].add(info["rnd_index"])
                entry["reps"] = max(entry["reps"], info["rnd_index"] + 1)
            else:
                u["real"] = True

    if conflicts:
        detail = "; ".join(f"{m} [{s}]: {sorted(k)}" for (m, s), k in conflicts.items())
        raise ValueError(
            "Result zips for the same model disagree on pipeline parameters "
            f"({detail}). Mixing settings would build one null distribution out of "
            "two analyses -- move the odd zips out of the results folder.")
    return found


def mem_now():
    """``(rss_gb, available_gb)`` on Linux, ``(None, None)`` elsewhere.

    Read from /proc rather than psutil so it works on a bare Colab runtime.
    ``MemAvailable`` is the number that matters: Colab kills the kernel when it
    hits zero, and it accounts for reclaimable page cache -- which a Drive-backed
    run generates a great deal of.
    """
    try:
        rss = avail = None
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1e6      # kB -> GB
                    break
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1e6
                    break
        return rss, avail
    except (OSError, ValueError, IndexError):
        return None, None


def mem_report(label, device=None):
    """One line of memory state. Silent where /proc is unavailable (Windows)."""
    rss, avail = mem_now()
    if rss is None:
        return
    gpu = ""
    if device is not None and getattr(device, "type", None) == "cuda":
        try:
            gpu = (f"  gpu={torch.cuda.memory_allocated() / 1e9:.2f}/"
                   f"{torch.cuda.memory_reserved() / 1e9:.2f} GB")
        except Exception:
            gpu = ""
    print(f"[mem] {label:<34s} rss={rss:6.2f} GB  avail={avail:6.2f} GB{gpu}",
          flush=True)


def _safe_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return float("inf")


def model_stem(rsa_model):
    """``valence3__cross`` -> ``valence3``; a model with no grouping is its own stem.

    Groupings are appended with a double underscore by
    ``tools/build_rsa_models.py``, and ``_models.csv`` lists the stems.
    """
    return rsa_model.split("__", 1)[0]


def load_models_manifest(refs_dir, dataset=None, verbose=False):
    """``{stem: (dis_method, mah_fold)}`` from ``refs/{dataset}/_models.csv``.

    This is how a model's battery is known *without opening its zip*, which on
    Drive costs a full download of a 576 MB file. Returns ``{}`` when the CSV is
    absent, and the caller falls back to probing.
    """
    hits = []
    if dataset:
        hits.append(os.path.join(refs_dir, dataset, "_models.csv"))
    else:
        try:
            for name in sorted(os.listdir(refs_dir)):
                p = os.path.join(refs_dir, name, "_models.csv")
                if os.path.isfile(p):
                    hits.append(p)
        except OSError:
            pass
    for path in hits:
        try:
            out = {}
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    stem = (row.get("model") or "").strip()
                    if stem:
                        out[stem] = ((row.get("dis_method") or "").strip(),
                                     (row.get("mah_fold") or "").strip() or None)
            if out:
                if verbose:
                    print(f"[discover] {len(out)} model stem(s) from "
                          f"{os.path.basename(os.path.dirname(path))}/_models.csv")
                return out
        except (OSError, csv.Error) as exc:
            print(f"WARNING: could not read {path}: {exc}")
    if verbose:
        print("[discover] no _models.csv in refs; probing one zip per model "
              "(slow on Drive)")
    return {}


def probe_zip_params(zip_path):
    """Read one zip's arcnames for the pipeline parameters and its max rnd index."""
    with _open_zip_resilient(zip_path) as zf:
        members = zf.namelist()
    params, reps, per_run = None, 0, False
    for member in members:
        info = parse_arcname(member)
        if info is None:
            continue
        if params is None:
            params = {k: info[k] for k in ("dataset", "model", "task", "radius",
                                           "mask_type", "dis_method",
                                           "rsa_method", "mah_fold")}
        if info["rnd"]:
            reps = max(reps, info["rnd_index"] + 1)
        per_run |= info["session"] is not None
    if params is None:
        raise MissingMapsError(
            f"{zip_path} holds no participant model-similarity maps.")
    return params, reps, per_run


def _build_manifest(params, reps, refs, sp, model_names, reps_group,
                    min_percentage_available, allow_space_mismatch,
                    allow_off_mask):
    """Assemble a manifest dict from probed parameters plus the refs snapshot."""
    sp_refs = refs.get("species", {}).get(sp)
    if sp_refs is None:
        raise KeyError(
            f"The reference snapshot has no entry for specie {sp!r} "
            f"(has {sorted(refs.get('species', {}))}). Rebuild it with "
            f"refs/build_refs.py --species {sp}")
    return {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "kind": "group-discovered",
        "datafolder": refs.get("datafolder", ""),
        "dataset": params["dataset"], "model": params["model"], "specie": sp,
        "task": params["task"] or sp_refs.get("task", params["dataset"]),
        "radius": params["radius"], "mask_type": params["mask_type"],
        "dis_method": params["dis_method"],
        "mah_fold": params["mah_fold"] or "stim-wise",
        "rsa_method": params["rsa_method"],
        "reps": reps, "reps_group": int(reps_group),
        "min_percentage_available": float(min_percentage_available),
        "participants": [int(x) for x in sp_refs["participants"]],
        "runs_by_sub": sp_refs.get("runs_by_sub") or {},
        "models": list(model_names),
        "allow_space_mismatch": bool(allow_space_mismatch),
        "allow_off_mask": bool(allow_off_mask),
    }


def manifest_signature(m):
    """The parameters that decide where a model's participant maps live."""
    return (m["dataset"], m["model"], m["dis_method"], m["mah_fold"],
            m["rsa_method"], m["radius"], m["mask_type"], is_per_run(m))


def discover_model_manifests(results_dir, refs_dir, specie=None, models=None,
                             reps_group=1000, min_percentage_available=1.0,
                             allow_space_mismatch=False, allow_off_mask=False,
                             verbose=True):
    """One manifest **per model**, each carrying that model's own parameters.

    A results folder can hold more than one analysis, and EmoC's does: 50 models
    are ``mahalanobis`` stim-wise (one map per participant) and 41 are
    ``correlation`` (one map per participant *run* -- a different folder layout
    entirely). Deriving a single ``dis_method`` from a sample of zips and
    applying it to every model is therefore wrong: for whichever half loses the
    vote, the group steps build paths that do not exist and every one of those
    models fails to load. That is the bug this replaced, and it was silent --
    the failure looked like missing data rather than a mislabelled analysis.

    Costs one zip open per model (91 for a full EmoC human battery, a couple of
    seconds) instead of one per zip (3 640, which is what made Colab's Drive
    endpoint drop). Each model is described by its own data rather than a guess.
    """
    by_model, _seen = scan_result_zip_names(results_dir, specie=specie,
                                            models=models, verbose=verbose)
    species = {k[1] for k in by_model}
    if len(species) > 1:
        raise ValueError(
            f"Result zips for more than one specie ({sorted(species)}). Pass "
            "specie= to pick one.")
    sp = species.pop()
    refs, out, skipped = None, {}, []
    t0 = time.time()

    # Which battery each model belongs to comes from refs/{dataset}/_models.csv,
    # NOT from opening its zip. That matters enormously on Drive: reading a zip's
    # central directory means seeking to the END of the file, and Drive's FUSE
    # layer serves that by pulling the whole file down. A correlation
    # participant zip is 576 MB against 45 MB for a mahalanobis one, so probing
    # one zip per model was ~27 GB of downloads and over an hour before this
    # function printed anything.
    #
    # So: group by dis_method from the CSV, then probe ONE zip per group -- the
    # smallest, since only the parameters are wanted and every zip in a group
    # carries the same ones. Two opens instead of 94.
    by_stem = load_models_manifest(refs_dir, verbose=verbose)
    groups, ungrouped = {}, []
    for (model, _sp) in sorted(by_model):
        key = by_stem.get(model_stem(model))
        (groups.setdefault(key, []) if key else ungrouped).append(model)
    if ungrouped and verbose:
        print(f"[discover] not in _models.csv, probed individually: "
              f"{', '.join(ungrouped[:6])}"
              + (" ..." if len(ungrouped) > 6 else ""))

    def smallest_zip(model):
        paths = by_model[(model, sp)]
        return min(paths.values(), key=lambda p: _safe_size(p))

    probe_of = {}                      # model -> (params, reps) to use
    todo = [(g, ms) for g, ms in groups.items()] + [(None, [m]) for m in ungrouped]
    for i, (_key, ms) in enumerate(todo, 1):
        pick = min(ms, key=lambda m: _safe_size(smallest_zip(m)))
        path = smallest_zip(pick)
        if verbose:
            print(f"[discover] probing {i}/{len(todo)}: {pick} "
                  f"({_safe_size(path) / 1e6:.0f} MB) for {len(ms)} model(s)...",
                  flush=True)
        try:
            params, reps, _pr = probe_zip_params(path)
        except MissingMapsError as exc:
            skipped.extend((m, str(exc)) for m in ms)
            continue
        for m in ms:
            probe_of[m] = (params, reps)

    for (model, _sp) in sorted(by_model):
        if model not in probe_of:
            continue
        subs = sorted(by_model[(model, _sp)])
        params, reps = probe_of[model]
        if refs is None:
            refs = load_refs(refs_dir, params["dataset"])
        m = _build_manifest(params, reps, refs, sp, [model], reps_group,
                            min_percentage_available, allow_space_mismatch,
                            allow_off_mask)
        m["zips_per_model"] = {model: subs}
        out[model] = m
    if not out:
        raise MissingMapsError(
            "No model had readable participant maps:\n  "
            + "\n  ".join(f"{m}: {w}" for m, w in skipped))
    if verbose:
        print(f"[discover] {len(out)} model(s) described from {len(todo)} zip "
              f"probe(s) in {time.time() - t0:.1f}s")
        groups = {}
        for model, m in out.items():
            groups.setdefault(manifest_signature(m), []).append(model)
        for sig, ms in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            (_ds, _glm, dis, fold, rsa, rad, mask, per_run) = sig
            fold_s = f"/{fold}" if dis == "mahalanobis" else ""
            layout = "per-run" if per_run else "per-participant"
            print(f"[discover]   {len(ms):3d} model(s): {dis}{fold_s}/{rsa}  "
                  f"r-{rad}  mask={mask}  {layout}   e.g. {sorted(ms)[0]}")
        if skipped:
            print(f"[discover]   skipped {len(skipped)}: "
                  f"{', '.join(m for m, _ in skipped[:5])}")
    return out


def discover_manifest(results_dir, refs_dir, specie=None, models=None,
                      reps_group=1000, min_percentage_available=1.0,
                      allow_space_mismatch=False, allow_off_mask=False,
                      sample=None, verbose=True):
    """A single manifest, for a selection that really is all one analysis.

    Thin wrapper over :func:`discover_model_manifests`: it probes every selected
    model and **raises** when they disagree, rather than silently adopting one
    model's parameters for all of them. ``sample`` is accepted and ignored -- it
    existed when this guessed from a few zips, which is what went wrong.
    """
    per_model = discover_model_manifests(
        results_dir, refs_dir, specie=specie, models=models,
        reps_group=reps_group, min_percentage_available=min_percentage_available,
        allow_space_mismatch=allow_space_mismatch, allow_off_mask=allow_off_mask,
        verbose=verbose)
    groups = {}
    for model, m in per_model.items():
        groups.setdefault(manifest_signature(m), []).append(model)
    if len(groups) > 1:
        detail = "\n  ".join(
            f"{len(ms):3d} model(s): dis={sig[2]} fold={sig[3]} rsa={sig[4]} "
            f"r={sig[5]} per_run={sig[7]}  e.g. {sorted(ms)[0]}"
            for sig, ms in groups.items())
        raise ValueError(
            "This results folder holds more than one analysis:\n  " + detail +
            "\n\nA single manifest cannot describe them. Use "
            "discover_model_manifests() -- run_group_package() does that for "
            "you -- or narrow the run with models=.")
    manifest = next(iter(per_model.values()))
    manifest["models"] = sorted(per_model)
    manifest["zips_per_model"] = {m: v["zips_per_model"][m]
                                  for m, v in per_model.items()}
    if verbose:
        fold = (f"/{manifest['mah_fold']}"
                if manifest["dis_method"] == "mahalanobis" else "")
        print(f"[discover] {manifest['specie']}  {manifest['dataset']}/"
              f"{manifest['model']}  r-{manifest['radius']}  "
              f"{manifest['dis_method']}{fold}/{manifest['rsa_method']}  "
              f"mask={manifest['mask_type']}  reps={manifest['reps']}  "
              f"reps_group={manifest['reps_group']}")
        print(f"[discover] {len(manifest['models'])} model(s); config lists "
              f"{len(manifest['participants'])} participant(s)")
    return manifest


def coverage(manifest, results_dir=None, verbose=False):
    """``{rsa_model: (n_participants_with_a_zip, n_participants_expected)}``.

    Counted from **file names** -- one zip per participant per model, so a
    missing participant is a missing name. Nothing is opened, which is what makes
    the preflight cheap enough to run over a 3 700-zip folder on Drive without
    knocking the mount over. Falls back to re-listing the folder only if the
    manifest predates ``zips_per_model`` (i.e. came from a package).
    """
    per_model = manifest.get("zips_per_model")
    if per_model is None:
        by_model, _seen = scan_result_zip_names(
            results_dir, specie=manifest["specie"], models=manifest["models"],
            verbose=verbose)
        per_model = {k[0]: sorted(v) for k, v in by_model.items()}
    total = len(manifest["participants"])
    return {m: (len(subs), total) for m, subs in sorted(per_model.items())}


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


def cluster_dist_rel(manifest, rsa_model):
    """Step 8's cluster-size distribution ``.npy``.

    Mirrors ``calculate_cluster_size_distribution``: it lives under **RSA**, not
    ``RSA_rnd``, in a ``dist/`` folder, and carries no ``{mask_type}-`` prefix --
    ``get_minimal_cluster_size`` (step 9) rebuilds this exact path.
    """
    return "/".join([manifest["dataset"], "results", "RSA", manifest["model"],
                     rsa_model, "dist",
                     f"{_group_stem(manifest)}_dist.npy"])


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

    def __init__(self, sources, dataset=None, verbose=True, index_workers=8):
        if isinstance(sources, (str, os.PathLike)):
            sources = [sources]
        self.trees, self.zip_dirs = [], []
        self._zip_names = {}     # dir -> [basenames]
        self._members = {}       # zip path -> {arcname: member name}
        self._members_lock = threading.Lock()
        self.index_workers = index_workers
        self.verbose = verbose
        for src in sources:
            src = os.path.abspath(str(src))
            if not os.path.isdir(src):
                raise FileNotFoundError(f"Result source not found: {src}")
            # ONE scandir for the whole folder. This used to be os.listdir plus an
            # os.path.isdir per entry -- and the generator evaluated it twice per
            # entry, so a 3 700-zip results folder cost ~7 400 stat calls before
            # anything was read. On Colab's Drive FUSE that is ten silent minutes.
            # entry.is_dir() reuses the dirent the listing already returned.
            t0 = time.time()
            subdirs, zips = [], []
            with os.scandir(src) as it:
                for e in it:
                    try:
                        if e.is_dir():
                            subdirs.append(e.name)
                        elif e.name.lower().endswith(".zip"):
                            zips.append(e.name)
                    except OSError:
                        continue
            # a data root is one holding {dataset}/results/ -- only subdirectories
            # can qualify, and a results folder has approximately none
            looks_like_tree = False
            for d in ([dataset] if dataset and dataset in subdirs else subdirs):
                if os.path.isdir(os.path.join(src, d, "results")):
                    looks_like_tree = True
                    break
            if looks_like_tree:
                self.trees.append(src)
            if zips:
                self.zip_dirs.append(src)
                self._zip_names[src] = zips
            if verbose:
                print(f"[store] listed {len(zips)} zip(s) + {len(subdirs)} "
                      f"subdir(s) in {time.time() - t0:.1f}s: {src}")
        if verbose:
            print(f"[store] {len(self.trees)} data tree(s), "
                  f"{sum(len(v) for v in self._zip_names.values())} zip(s) in "
                  f"{len(self.zip_dirs)} folder(s)")

    def forget(self):
        """Drop the cached per-zip namelists.

        They are only reused within one model -- each model has its own zips --
        so holding them for a 91-model battery accumulates every listing of
        every zip touched so far for no benefit. Correlation zips carry ~600
        members each, 40 zips per model.
        """
        with self._members_lock:
            self._members.clear()

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
            # read outside the lock: the read is the slow part and two threads
            # racing on the same zip is wasteful but harmless
            with _open_zip_resilient(zip_path) as zf:
                idx = {n.replace("\\", "/").lstrip("./"): n for n in zf.namelist()}
            with self._members_lock:
                self._members.setdefault(zip_path, idx)
                idx = self._members[zip_path]
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
        # Read the zips' central directories IN PARALLEL, with progress.
        #
        # This is latency-bound, not bandwidth-bound: opening a zip seeks to the
        # end of the file, and on Drive each seek is a slow round trip. Serially
        # over 40 correlation zips (322 MB each) that measured 35+ minutes of
        # total silence, which is indistinguishable from a hang -- and was
        # repeatedly misdiagnosed as an out-of-memory, when RSS never exceeded
        # 0.7 GB. The same pool that reads the maps handles this.
        zip_paths = self.zips_for(manifest, rsa_model)
        todo = [p for p in zip_paths if p not in self._members]
        if todo:
            t0 = time.time()
            if self.verbose:
                print(f"[store] listing {len(todo)} zip(s) for {rsa_model} "
                      f"({sum(_safe_size(p) for p in todo) / 1e9:.1f} GB) ...",
                      flush=True)
            done = {"n": 0}
            lock = threading.Lock()

            def index_one(p):
                out = self._index_zip(p)
                if self.verbose:
                    with lock:
                        done["n"] += 1
                        n = done["n"]
                        if n == 1 or n == len(todo) or n % max(1, len(todo) // 5) == 0:
                            dt = time.time() - t0
                            print(f"[store]   {n}/{len(todo)} listed, {dt:.0f}s",
                                  flush=True)
                return out

            with ThreadPoolExecutor(
                    max_workers=max(1, min(self.index_workers, len(todo)))) as ex:
                list(ex.map(index_one, todo))
            if self.verbose:
                print(f"[store] listed {len(todo)} zip(s) in "
                      f"{time.time() - t0:.1f}s", flush=True)

        refs = {}
        for zip_path in zip_paths:
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


def prefetch_model_zips(store, manifest, rsa_model, cache_dir, workers=8,
                        verbose=True):
    """Copy one model's result zips to local disk and return a store reading them.

    **Measured to be a net loss on a Windows Drive mount, and off by default.**
    Two untouched models, 38 zips / ~1.6 GB each, cold cache, 8 threads:

        direct   (read members off Drive)      128.0 s   12.6 MB/s
        prefetch (copy 119.8 s + read 56.1 s)  175.9 s    9.3 MB/s

    The premise behind prefetching was that the mount is *latency*-bound on many
    small member reads, so one bulk sequential copy would beat them. It is not:
    direct member reads already sustain the same MB/s as a bulk copy (12.6 vs
    14), so the mount is **bandwidth**-bound at this thread count, and prefetch
    just adds a full local re-read (56 s) for nothing.

    Kept as an option because it is mount-dependent -- it can only win where
    per-member latency really does dominate, which a different FUSE layer might.
    Measure before turning it on; ``load_participant_maps`` reads every member of
    every zip, so prefetch is always the same bytes twice.

    Returns ``(store, cache_dir)``; the original store comes back untouched when
    there is nothing to copy, so the caller can always use the result.
    """
    zips = store.zips_for(manifest, rsa_model)
    if not zips:
        return store, None
    os.makedirs(cache_dir, exist_ok=True)

    def one(src):
        dst = os.path.join(cache_dir, os.path.basename(src))
        size = os.path.getsize(src)
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            return 0                     # already cached by an earlier attempt
        shutil.copyfile(src, dst)
        return size

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(zips)))) as ex:
        sizes = list(ex.map(one, zips))
    total = sum(sizes)
    copied = sum(1 for s in sizes if s)
    if verbose and copied:
        dt = time.time() - t0
        print(f"[prefetch] {rsa_model}: copied {copied} zip(s), "
              f"{total / 1e6:.0f} MB in {dt:.1f}s "
              f"({total / 1e6 / max(dt, 1e-6):.0f} MB/s, {workers} threads)")
    return ResultStore([cache_dir], dataset=manifest["dataset"], verbose=False), cache_dir


def _read_ref(ref, open_zips):
    if ref[0] == "file":
        return nib.load(ref[1])
    zf = open_zips[ref[1]]
    return _img_from_bytes(zf.read(ref[2]))


def _load_unit(manifest, rsa_model, unit, refs, mask_img, mask_bool, mask_flat,
               reps, want_real, strict_space, strict_mask, want_rnd=True):
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
    # Only when the caller actually wants them. Since steps 5-8 stream their own
    # reads, this loop was fetching all 100 permutation maps per unit and having
    # them thrown away immediately -- every map in the model read twice, ~560 s
    # of pure waste per correlation model, and enough allocator churn that RSS
    # climbed ~2.4 GB per model across a battery.
    if want_rnd:
        for i in range(reps):
            rel = participant_map_rel(manifest, rsa_model, unit, rnd=True,
                                      rnd_index=i)
            if rel in refs:
                wanted[("rnd", i)] = refs[rel]
    if not wanted:
        return None, {}

    zip_paths = sorted({r[1] for r in wanted.values() if r[0] == "zip"})
    real_vec, rnd = None, {}
    loaded = []
    with contextlib.ExitStack() as stack:
        open_zips = {p: stack.enter_context(_open_zip_resilient(p))
                     for p in zip_paths}
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
    mem_report(f"before index_model")
    refs = store.index_model(manifest, rsa_model)
    mem_report(f"after index_model ({len(refs)} refs)")
    mask_flat = np.flatnonzero(mask_bool.reshape(-1))
    all_units = units(manifest)
    # Re-derive reps from what is actually on disk rather than trusting the
    # manifest. A discovered manifest reads it off a small sample of zips, and a
    # participant who happens to have more permutations than the sample would
    # otherwise have the extras silently ignored.
    observed = 0
    for rel in refs:
        info = parse_arcname(rel)
        if info and info["rnd"]:
            observed = max(observed, info["rnd_index"] + 1)
    reps = max(int(manifest.get("reps") or 0), observed)
    strict_space = not manifest.get("allow_space_mismatch", False)
    strict_mask = not manifest.get("allow_off_mask", False)

    t0 = time.time()
    n_workers = max(1, min(workers, len(all_units)))
    if verbose:
        per_unit = (reps if want_rnd else 0) + (1 if want_real else 0)
        print(f"[load] {rsa_model}: reading up to {len(all_units) * per_unit} "
              f"map(s) from {len(all_units)} unit(s) "
              f"({'real+perms' if want_rnd else 'real only'}), "
              f"{n_workers} thread(s)...", flush=True)

    # Progress as units land, not just a summary at the end. This is the longest
    # phase by far -- on Colab's Drive it is minutes per model -- and reporting
    # only on completion makes a slow run indistinguishable from a hung one.
    done = {"n": 0}
    lock = threading.Lock()

    # Plan the layout BEFORE reading anything. The refs index already says which
    # permutation indices each unit has, so every map's destination row is known
    # up front and the maps can be written straight into one preallocated array.
    #
    # This used to collect per-unit lists and then np.stack them, which held the
    # same data twice for the duration of the copy: 30.4 GB became a 61 GB peak
    # for a 239-unit correlation model and would not fit any Colab runtime below
    # an A100. The arrays produced are identical -- this is only how they are
    # built.
    avail, has_real = {}, {}
    for u in all_units:
        avail[u] = [i for i in range(reps)
                    if participant_map_rel(manifest, rsa_model, u, rnd=True,
                                           rnd_index=i) in refs]
        has_real[u] = (want_real and
                       participant_map_rel(manifest, rsa_model, u) in refs)

    rnd_units = [u for u in all_units if avail[u]] if want_rnd else []
    real_units = [u for u in all_units if has_real[u]]
    missing_rnd = [unit_label(u) for u in all_units
                   if want_rnd and not avail[u]]
    missing_real = [unit_label(u) for u in all_units
                    if want_real and not has_real[u]]

    counts = np.array([len(avail[u]) for u in rnd_units], dtype=np.int64)
    offsets = (np.concatenate([[0], np.cumsum(counts)])[:-1]
               if len(counts) else np.array([], dtype=np.int64))
    n_rows = int(counts.sum())
    rnd_flat = np.empty((n_rows, mask_flat.size), dtype=np.float64)
    real = (np.empty((len(real_units), mask_flat.size), dtype=np.float64)
            if real_units else None)
    row_of_rnd = {u: int(offsets[k]) for k, u in enumerate(rnd_units)}
    row_of_real = {u: k for k, u in enumerate(real_units)}

    def one(u):
        out = _load_unit(manifest, rsa_model, u, refs, mask_img, mask_bool,
                         mask_flat, reps, want_real, strict_space, strict_mask,
                         want_rnd=want_rnd)
        real_vec, rnd = out
        # threads write disjoint row ranges, so no lock is needed here
        if real_vec is not None and u in row_of_real:
            real[row_of_real[u]] = real_vec
        if u in row_of_rnd:
            base = row_of_rnd[u]
            for j, i in enumerate(avail[u]):
                rnd_flat[base + j] = rnd[i]
        # drop this unit's copies now rather than at the end of the pool
        out = real_vec = rnd = None
        if verbose:
            with lock:
                done["n"] += 1
                n, total = done["n"], len(all_units)
                if n == 1 or n == total or n % max(1, total // 10) == 0:
                    dt = time.time() - t0
                    # no ETA off the first unit: the pool is still spinning up,
                    # so extrapolating from it reads ~7x too pessimistic
                    eta = (f", ~{dt / n * (total - n):.0f}s left"
                           if n_workers < n < total else "")
                    print(f"[load]   {n}/{total} unit(s), {dt:.0f}s elapsed{eta}",
                          flush=True)
        return None

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        list(ex.map(one, all_units))
    mem_report("after real maps")

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
        "mask_flat": mask_flat, "n_real": len(real_units), "refs": refs,
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


def stream_permutation_stats(manifest, rsa_model, store, mask_img, mask_bool,
                             mask_flat, reps_group, seed, device=None,
                             workers=8, want_group_means=True, refs=None,
                             verbose=True):
    """Steps 5+6 **without ever holding all the permutation maps**.

    ``group_permutation_stats`` needs the whole ``(T, V)`` matrix in RAM: 30 GB
    for a 239-unit correlation model, which no Colab runtime survives once the
    reader threads and the group-mean arrays are added. But step 5 is a *sum*:

        group_mean[g] = (1/U) * sum over units u of  X[u, drawn(g, u)]

    so a map only has to be present long enough to be added into the rows that
    drew it. Streaming unit by unit into a running ``(reps_group, V)``
    accumulator turns the peak from ``n_units * reps`` maps into **one**, plus
    the accumulator itself -- 30.4 GB becomes ~1.3 GB, independent of how many
    units the model has.

    Reads are prefetched on a thread pool but **applied in job order**, so the
    float64 sums accumulate in a fixed sequence and the result is reproducible.
    Within one unit the order is irrelevant anyway: each accumulator row takes
    exactly one map per unit.

    Returns ``(group_means | None, dist_mean, dist_std, n_units, counts)``.
    """
    device = device or gpu_rsa.pick_device()
    # reuse the caller's index; building it again means re-listing 40 zips
    refs = store.index_model(manifest, rsa_model) if refs is None else refs
    all_units = units(manifest)
    strict_space = not manifest.get("allow_space_mismatch", False)
    strict_mask = not manifest.get("allow_off_mask", False)

    observed = 0
    for rel in refs:
        info = parse_arcname(rel)
        if info and info["rnd"]:
            observed = max(observed, info["rnd_index"] + 1)
    reps = max(int(manifest.get("reps") or 0), observed)

    avail = {u: [i for i in range(reps)
                 if participant_map_rel(manifest, rsa_model, u, rnd=True,
                                        rnd_index=i) in refs]
             for u in all_units}
    rnd_units = [u for u in all_units if avail[u]]
    counts = np.array([len(avail[u]) for u in rnd_units], dtype=np.int64)
    if not len(counts):
        return None, None, None, 0, counts

    # draw_group_indices returns indices into the CONCATENATED (T, V) matrix, so
    # each column is offset by where that unit's block starts. Streaming has no
    # such matrix, so subtract the offset to get a position within the unit's own
    # list. Same function and same seed as the bulk path, hence the same draw.
    cols = draw_group_indices(counts, reps_group, seed)
    offsets = np.concatenate([[0], np.cumsum(counts)])[:-1]
    V = mask_flat.size
    acc = torch.zeros((reps_group, V), dtype=DTYPE, device=device)
    mem_report("after accumulator alloc", device)

    # (unit_index, rnd_index, rows) in the order they must be applied
    jobs = []
    for u_i, u in enumerate(rnd_units):
        col = cols[:, u_i] - offsets[u_i]
        for j in np.unique(col):
            jobs.append((u_i, int(avail[u][int(j)]), np.flatnonzero(col == j)))

    def read(job):
        u_i, idx, rows = job
        rel = participant_map_rel(manifest, rsa_model, rnd_units[u_i], rnd=True,
                                  rnd_index=idx)
        ref = refs[rel]
        if ref[0] == "file":
            img = nib.load(ref[1])
        else:
            with _open_zip_resilient(ref[1]) as zf:
                img = _img_from_bytes(zf.read(ref[2]))
        gpu_rsa.check_same_space(("mask", mask_img), [(rel, img)],
                                 context=f"step 5 stream for {rsa_model}",
                                 strict=strict_space)
        flat = np.asarray(img.dataobj, dtype=np.float64).reshape(-1)
        off = flat[~mask_bool.reshape(-1)]
        if off.any() and strict_mask:
            raise OffMaskError(
                f"{rel} has non-zero voxels outside the mask "
                f"{manifest.get('mask_type')!r}.")
        return rows, flat[mask_flat]

    t0, n_done, last_unit = time.time(), 0, -1
    n_workers = max(1, min(workers, len(jobs)))
    pending = collections.deque()
    it = iter(jobs)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        def submit():
            try:
                job = next(it)
            except StopIteration:
                return False
            pending.append((job[0], ex.submit(read, job)))
            return True

        for _ in range(max(2 * n_workers, 4)):
            if not submit():
                break
        while pending:
            u_i, fut = pending.popleft()
            rows, vec = fut.result()
            acc.index_add_(0, torch.as_tensor(rows, device=device),
                           torch.as_tensor(vec, dtype=DTYPE,
                                           device=device).repeat(rows.size, 1))
            n_done += 1
            if verbose and u_i != last_unit:
                last_unit = u_i
                n_u = len(rnd_units)
                if u_i == 0 or u_i + 1 == n_u or (u_i + 1) % max(1, n_u // 10) == 0:
                    dt = time.time() - t0
                    eta = (f", ~{dt / n_done * (len(jobs) - n_done):.0f}s left"
                           if n_done > n_workers else "")
                    print(f"[step5]   unit {u_i + 1}/{n_u}, {n_done}/{len(jobs)} "
                          f"map(s), {dt:.0f}s{eta}", flush=True)
                    mem_report(f"  streaming unit {u_i + 1}", device)
            submit()

    acc /= float(len(rnd_units))
    mu = acc.mean(dim=0)
    sd = torch.sqrt(((acc - mu) ** 2).mean(dim=0))
    dist_mean, dist_std = mu.cpu().numpy(), sd.cpu().numpy()
    group_means = acc.cpu().numpy() if want_group_means else None
    del acc
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if verbose:
        print(f"[step5] {rsa_model}: {reps_group} group permutation(s) over "
              f"{len(rnd_units)} unit(s) from {len(jobs)} map read(s) in "
              f"{time.time() - t0:.1f}s (streamed, seed={seed})")
    return group_means, dist_mean, dist_std, len(rnd_units), counts


def mean_std(rows, device=None):
    """Voxelwise mean/std over a ``(N, V)`` stack -- the ``nifti_mean`` reduction."""
    device = device or gpu_rsa.pick_device()
    x = torch.as_tensor(rows, dtype=DTYPE, device=device)
    mu = x.mean(dim=0)
    sd = torch.sqrt(((x - mu) ** 2).mean(dim=0))
    return mu.cpu().numpy(), sd.cpu().numpy()


# ===========================================================================
# Step 8 -- cluster-size distribution, computed on the z maps still in memory
#
# This is the step that lets the rnd z maps stay on Colab. They are step 8's
# only consumer, they are the bulk of the output (measured on EmoC H: 1273 kB
# each, so 1.27 GB per model at reps_group=1000), and step 8 reduces all of them
# to one small .npy. Computing it here turns ~1.3 GB per model crossing Drive
# into a few kB.
#
# Faithful to ``rsa_utils.calculate_cluster_size_distribution`` ->
# ``count_clusters_sizes`` -> ``_count_on_3d``:
#   * ``np.nan_to_num`` first (so the NaNs step 7 leaves off-mask become 0, and
#     an infinity from a zero null-std becomes a huge finite value that does
#     cross the threshold -- same as on the CPU);
#   * a strict ``> threshold`` test, positives only (``two_sided=False``);
#   * ``generate_binary_structure(rank=3, connectivity=3)``, i.e. 26-connected,
#     matching FSL's default;
#   * sizes from ``np.bincount(labels.ravel())[1:]``, sorted descending, as a
#     plain Python list -- what ``get_minimal_cluster_size`` iterates over.
# ===========================================================================
def _mask_bbox(mask_flat, shape):
    """Smallest box containing every mask voxel, as a tuple of slices.

    Labelling inside this box is exact rather than an approximation: every voxel
    outside the searchlight mask is 0 in a group map, so no cluster can reach
    beyond the box, and connectivity within it is unchanged. It is worth doing --
    on the EmoC human grid the box is a little over a third of the volume, and
    step 8 labels ``reps_group x len(thresholds)`` times.
    """
    idx = np.unravel_index(mask_flat, shape)
    return tuple(slice(int(i.min()), int(i.max()) + 1) for i in idx)


def cluster_size_distribution(z_rnd, mask_flat, shape, thresholds,
                              connectivity=3, workers=8, verbose=True):
    """Cluster sizes of every rnd z map, at every threshold, without writing them.

    ``z_rnd`` is the ``(reps_group, n_mask_voxels)`` array step 7 produced.
    Returns ``{f"z{threshold}": {"number_of_images": G, "cluster_sizes": ...}}``
    -- the dict layout ``get_minimal_cluster_size`` expects, with
    ``cluster_sizes`` an object array of one list per permutation.
    """
    try:
        from scipy.ndimage import label, generate_binary_structure
    except ImportError as exc:                    # pragma: no cover
        raise ImportError(
            "Step 8 needs scipy (preinstalled on Colab): pip install scipy") from exc
    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity must be 1, 2, or 3 for 3D images.")

    thresholds = [float(t) for t in thresholds]
    if not thresholds:
        raise ValueError("At least one z threshold is required for step 8.")
    structure = generate_binary_structure(rank=3, connectivity=connectivity)
    box = _mask_bbox(mask_flat, shape)
    n_vox = int(np.prod(shape))
    G = z_rnd.shape[0]
    out = {t: np.zeros((G,), dtype=object) for t in thresholds}

    def one(g):
        vol = np.full(n_vox, np.nan, dtype=np.float64)   # step 7's off-mask fill
        vol[mask_flat] = z_rnd[g]
        vol = np.nan_to_num(vol.reshape(shape))[box]
        sizes = {}
        for t in thresholds:
            labels, n = label(vol > t, structure=structure)
            if n == 0:
                sizes[t] = []
            else:
                counts = np.bincount(labels.ravel())[1:]
                sizes[t] = np.sort(counts)[::-1].tolist()
        return g, sizes

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for g, sizes in ex.map(one, range(G)):
            for t in thresholds:
                out[t][g] = sizes[t]

    dist = {}
    for t in thresholds:
        # the key format get_minimal_cluster_size builds from a float
        # --z_threshold, so "z3.1" and "z4.0"
        dist[f"z{t}"] = {"number_of_images": G, "cluster_sizes": out[t]}
    if verbose:
        for t in thresholds:
            per_perm = np.array([max(s) if len(s) else 0 for s in out[t]])
            print(f"[step8]   z>{t}: max cluster per permutation "
                  f"median={int(np.median(per_perm))} "
                  f"p95={int(np.percentile(per_perm, 95))} "
                  f"max={int(per_perm.max())}")
        print(f"[step8] {G} map(s) x {len(thresholds)} threshold(s) in "
              f"{time.time() - t0:.1f}s")
    return dist


def merge_cluster_distribution(path, dist):
    """Fold new thresholds into an existing ``_dist.npy`` instead of replacing it.

    The file is keyed by threshold and the CPU step appends to it one key at a
    time, so a run that computes z4.0 must not drop a z3.1 someone already
    computed. Only matters when a scratch tree is reused; the keys computed here
    win on a collision, being the ones from this run's z maps.
    """
    if not os.path.exists(path):
        return dist
    try:
        existing = np.load(path, allow_pickle=True).item()
    except Exception as exc:
        print(f"WARNING: could not read {path} ({exc}); writing a fresh one.")
        return dist
    merged = dict(existing)
    merged.update(dist)
    return merged


def write_cluster_distribution(path, dist, manifest, rsa_model, thresholds):
    """Save the dict as ``.npy`` plus the ``_log.txt`` the CPU step writes."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = merge_cluster_distribution(path, dist)
    with open(path, "wb") as f:
        np.save(f, payload)
    lines = ["\n" + "=" * 50 + "\n",
             f"Log date and time: "
             f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
    for t in thresholds:
        key = f"z{float(t)}"
        n = dist[key]["number_of_images"]
        lines.append(f"Z threshold: {t}")
        lines.append(f"Processed {n} z map files for cluster size distribution.")
        lines.append(f"Data added to cluster_sizes_dict under key {key}.")
    lines.append(f"Computed on GPU by tools/colab_gpu/gpu_group.py v{VERSION} "
                 "from the z maps in memory; the rnd z maps themselves were "
                 "not written.")
    lines.append(f"Keys now in the file: {sorted(payload)}")
    write_text(path.replace(".npy", "_log.txt"), "\n".join(lines))
    return path


# ===========================================================================
# Steps 3 / 5 / 6 / 7 / 8 -- driver
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
        "notes": [f"computed on GPU by tools/colab_gpu/gpu_group.py v{VERSION}"],
        "gpu_group_version": VERSION,
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
                    g_batch=64, write_group_means=False, write_z_maps=False,
                    z_thresholds=DEFAULT_Z_THRESHOLDS, connectivity=3, workers=8,
                    refs_dir=None, verbose=True):
    """Run steps 3/5/6/7/8 for one RSA model. Returns the written paths.

    Outputs land under ``{work_root}/data/`` with pipeline-relative paths, ready
    to be zipped by :func:`zip_group_result` and merged with
    ``tools/unpack_results.py``.
    """
    steps = tuple(sorted(set(int(s) for s in steps)))
    device = device or gpu_rsa.pick_device()
    out_root = os.path.join(work_root, "data")
    reps_group = manifest["reps_group"]
    min_pct = manifest.get("min_percentage_available", 1.0)
    if 8 in steps and 7 not in steps:
        raise ValueError(
            "Step 8 here reads the rnd z maps out of memory rather than off disk, "
            "so step 7 has to run in the same pass. Add 7 to steps (or run step 8 "
            "on the workstation against z maps you have already merged).")

    mask_img, mask_bool = load_group_mask(manifest, pkg_root=pkg_root,
                                          refs_dir=refs_dir)
    shape, affine = mask_bool.shape, mask_img.affine

    need_rnd = any(s in steps for s in (5, 6, 7, 8))
    need_real = 3 in steps
    # want_rnd=False: the permutation maps are streamed below rather than loaded
    # in bulk, so nothing here holds more than one of them at a time.
    maps = load_participant_maps(manifest, rsa_model, store, mask_img, mask_bool,
                                 workers=workers, want_real=need_real,
                                 want_rnd=False, verbose=verbose)
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
    if seed is None:
        seed = zlib.crc32(
            f"group-{rsa_model}-{manifest['specie']}-{reps_group}".encode())

    want_gm = write_group_means and 5 in steps
    group_means, dist_mean, dist_std, n_units, _counts = stream_permutation_stats(
        manifest, rsa_model, store, mask_img, mask_bool, mask_flat, reps_group,
        seed, device=device, workers=workers, refs=maps.get("refs"),
        want_group_means=want_gm or (7 in steps) or (8 in steps), verbose=verbose)
    if n_units == 0:
        raise MissingMapsError(
            f"{rsa_model}: no permutation (step-4) maps found for any unit.")
    pct = n_units / total_units
    if pct < min_pct:
        raise MissingMapsError(
            f"{rsa_model}: only {pct*100:.1f}% of the units have permutation maps "
            f"({n_units}/{total_units}); min_percentage_available is "
            f"{min_pct*100:.1f}%.")

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
        if write_z_maps:
            jobs = [dict(vec=z_rnd[g], mask_flat=mask_flat, shape=shape,
                         affine=affine,
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
                f"Calculated z maps for {reps_group} available rnd mean files."
                + ("" if write_z_maps else
                   " Not written to disk (write_z_maps=False); consumed in memory "
                   "by step 8 in the same run."),
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
            how = "written" if write_z_maps else "kept in memory for step 8"
            print(f"[step7] {rsa_model}: {reps_group} rnd z map(s) ({how}) + "
                  f"real z map in {time.time() - t0:.1f}s")

    # ---- Step 8: cluster-size distribution ---------------------------------
    if 8 in steps:
        dist = cluster_size_distribution(
            z_rnd, mask_flat, shape, z_thresholds, connectivity=connectivity,
            workers=workers, verbose=verbose)
        path = out(cluster_dist_rel(manifest, rsa_model))
        written.append(write_cluster_distribution(
            path, dist, manifest, rsa_model, z_thresholds))
        written.append(path.replace(".npy", "_log.txt"))
        if verbose:
            print(f"[step8] {rsa_model}: thresholds "
                  f"{', '.join(str(float(t)) for t in z_thresholds)} -> "
                  f"{os.path.basename(path)}")
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
        f"{d}/results/RSA/{m}/{rsa_model}/dist/*",
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

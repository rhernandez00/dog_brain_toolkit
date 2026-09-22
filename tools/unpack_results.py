#!/usr/bin/env python
"""unpack_results.py -- merge Colab result zips back onto the pipeline data disk.

The Colab GPU run (see tools/colab_gpu/) writes one ``result_*.zip`` per finished
part, each holding files whose arc-paths are already pipeline-relative to the data
folder (e.g. ``EmoC/results/RSA/basic-block/H-sub-40/r-4_mahalanobis_DogA_DogF.nii.gz``).
Unpacking is therefore a validated merge: extract each member to
``{datafolder}/{arcname}``. Afterwards the remaining steps of ``searchlight.py``
run exactly as if the maps had been computed on the workstation -- steps 3-10
after a per-participant run (``result_step1_*.zip`` / ``result_<model>_*.zip``),
steps 8-10 after a group run (``result_group_<model>_<specie>.zip``).

Usage (from the repo root, full Anaconda interpreter -- see CLAUDE.md):

    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\unpack_results.py DOWNLOADS_DIR
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\unpack_results.py result_step1_mah_H-sub-40.zip --dry-run

Accepts any mix of ``result_*.zip`` files, directories containing them, and
already-unzipped ``result_*`` folders (e.g. extracted by hand, or left over from a
partial unzip) -- all three can sit side by side in the same downloads folder and
are merged the same way, since an unzipped folder holds exactly the same
pipeline-relative tree a zip's members would extract to. Existing files are left
untouched unless ``--replace`` is given; ``--dry-run`` reports the planned copies
without writing anything.

``--no_step4_files`` leaves the step-4 permutation maps in the zip. They are the
bulkiest thing a run produces (``--reps`` maps per participant per run) and step 5
is their only reader, so when step 5 already ran on the GPU -- its group means are
in the same zip -- writing them to the data disk only fills it with something
``tools/bulk_check.py --delete_step4`` would reclaim afterwards. What counts as a
step-4 file is the participant level of ``results/RSA_rnd/``; the group ``mean/``
outputs of steps 5 and 7 live under the same tree and are always merged.

Resuming an interrupted merge is the normal case, so the script is built around
making "is this zip already unpacked?" cheap on a network data folder:

* the target tree is probed **one directory listing per folder**, never one stat
  per member -- a listing is a single SMB round-trip that returns every name *and
  its size*, so it answers all N questions about that folder at once. Measured on
  ``P:`` (2026-08-01, 1560 maps in one step-1 run folder): one ``scandir`` with
  sizes 1.3 s, versus 87 s for 1560 ``os.path.exists`` calls and 150 s for
  ``exists`` + ``getsize`` -- roughly 56 ms per round-trip, so the per-file probe
  costs 66x more than the listing that replaces it;
* listings are cached for the whole run and shared across zips, and are updated
  in place as files are written, so no folder is ever enumerated twice;
* a zip whose members are all present at the right size is reported complete and
  skipped without touching a single output path;
* the listings and the extractions both run on a thread pool, because the cost
  here is round-trip latency rather than bandwidth.

At that latency the writes are latency-bound too, which is why they share the
thread pool; raise ``--workers`` above the default if the link tolerates it.

Because the listing hands back sizes for free, an existing file only counts as
present if its size matches the zip entry. A file left half-written by a killed
run is therefore detected and rewritten instead of being skipped forever
(``--no-verify-size`` restores the old existence-only check). New files are
written to a ``.part`` temp file and atomically renamed, so interrupting this
script cannot create a truncated map in the first place.
"""

import argparse
import os
import re
import shutil
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from scheduler.paths import get_paths  # noqa: E402

DEFAULT_WORKERS = 16
COPY_CHUNK = 1 << 20  # 1 MiB -- keep the SMB pipe full on large niftis

# What a result zip is allowed to put on the data disk. Everything the pipeline
# itself writes for these steps, and nothing else.
MERGEABLE_SUFFIXES = (".nii.gz", ".json", ".txt", ".npy", ".csv")

# The listing is a dict lookup where the old code called os.path.exists, so it
# has to reproduce that call's case rules: Windows resolves case-variant
# filenames, Linux does not (the same split make_mask.py warns about). Without
# this a member cased differently from the file on disk would read as missing
# and be rewritten on every run.
_FOLD_CASE = os.name == "nt"


def _key(name):
    return name.lower() if _FOLD_CASE else name


def _is_result_name(name):
    """True for the ``result_...`` naming convention shared by zips and their
    unzipped folders (``result_step1_*``, ``result_group_<model>_<specie>``, ...)."""
    return name.lower().startswith("result_")


class ZipSource:
    """A ``result_*.zip`` file, read member-by-member with ``zipfile``."""

    kind = "zip"

    def __init__(self, path):
        self.path = path

    @property
    def label(self):
        return os.path.basename(self.path)

    def iter_members(self):
        """Yield ``(arcname, size, info)`` for every member."""
        with zipfile.ZipFile(self.path) as zf:
            for info in zf.infolist():
                yield info.filename, info.file_size, info

    def open(self, info, local):
        # One ZipFile handle per worker thread: a single handle is not safe to
        # read from concurrently, and reopening a local zip is cheap next to an
        # SMB write.
        zf = getattr(local, "zf", None)
        if zf is None:
            zf = local.zf = zipfile.ZipFile(self.path)
        return zf.open(info)


class DirSource:
    """An already-unzipped ``result_*`` folder: same tree a zip would extract to."""

    kind = "dir"

    def __init__(self, path):
        self.path = os.path.normpath(path)

    @property
    def label(self):
        return os.path.basename(self.path)

    def iter_members(self):
        """Yield ``(arcname, size, full_path)`` for every file under the root."""
        for root, _dirs, files in os.walk(self.path):
            for fname in files:
                full = os.path.join(root, fname)
                arcname = os.path.relpath(full, self.path).replace(os.sep, "/")
                try:
                    size = os.path.getsize(full)
                except OSError:
                    continue
                yield arcname, size, full

    def open(self, full_path, local):
        return open(full_path, "rb")


def collect_sources(inputs):
    """Expand files/dirs into a sorted list of ``ZipSource``/``DirSource`` objects.

    A directory input is either a *container* -- searched one level down for
    ``*.zip`` files and ``result_*`` subfolders -- or, if its own name already
    follows the ``result_*`` convention, an unzipped result folder in its own
    right (so passing that folder directly also works).
    """
    sources = []
    for item in inputs:
        if os.path.isdir(item):
            base = os.path.basename(os.path.normpath(item))
            if _is_result_name(base):
                sources.append(DirSource(item))
                continue
            for name in sorted(os.listdir(item)):
                full = os.path.join(item, name)
                if os.path.isfile(full) and name.lower().endswith(".zip"):
                    sources.append(ZipSource(full))
                elif os.path.isdir(full) and _is_result_name(name):
                    sources.append(DirSource(full))
        elif os.path.isfile(item) and item.lower().endswith(".zip"):
            sources.append(ZipSource(item))
        else:
            print(f"WARNING: skipping {item!r} (not a .zip or directory)")
    return sources


def _safe_member(name):
    """Reject absolute paths and parent-directory escapes; keep pipeline outputs.

    Steps 1/2/4 emit only niftis, but the group steps (3/5/6/7) also write the
    sidecars ``rsa_utils`` writes next to them: step 3's ``*_mean.json`` -- which
    ``calculate_group_model_similarity_map`` reads back to decide whether the map
    must be recomputed -- and the ``*_log.txt`` files of steps 6 and 7. Dropping
    those would leave a merged run subtly different from a local one.
    """
    norm = name.replace("\\", "/")
    if norm.endswith("/"):
        return None
    if os.path.isabs(norm) or ".." in norm.split("/"):
        raise ValueError(f"Unsafe path in zip: {name!r}")
    if not norm.endswith(MERGEABLE_SUFFIXES):
        return None
    return norm


# A step-4 map is a *participant* map under RSA_rnd:
#   {dataset}/results/RSA_rnd/{model}/{rsa_model}[/{mah_fold}]/{specie}-sub-{NN}/
#       [ses-{ss}_task-{task}_run-{rr}/]{stem}_{index:04d}.nii.gz
# Steps 5 and 7 write into the same RSA_rnd tree but under a group ``mean/``
# folder with no participant component, which is what tells the two apart here.
_SUB_DIR_RE = re.compile(r"^[DH]-sub-\d+$")


def _is_step4_member(norm):
    """True for a per-participant permutation map (step 4's output)."""
    parts = norm.split("/")
    if len(parts) < 6 or parts[1] != "results" or parts[2] != "RSA_rnd":
        return False
    return any(_SUB_DIR_RE.match(p) for p in parts[5:-1])


def _list_dir(dirpath):
    """Return ``{filename: size}`` for one folder, or None if it does not exist.

    One ``scandir`` is one round-trip on a network share and the sizes come back
    inside it, so this replaces every ``os.path.exists``/``getsize`` we would
    otherwise issue for the files in this folder.
    """
    try:
        listing = {}
        with os.scandir(dirpath) as it:
            for entry in it:
                try:
                    if entry.is_file():
                        listing[_key(entry.name)] = entry.stat().st_size
                except OSError:
                    continue  # vanished mid-scan; treat as absent
        return listing
    except FileNotFoundError:
        return None
    except NotADirectoryError:
        return None
    except OSError as exc:
        print(f"WARNING: cannot list {dirpath}: {exc}")
        return None


class DirIndex:
    """Cached view of the target tree, one listing per directory, run-wide."""

    def __init__(self, workers=DEFAULT_WORKERS):
        self._dirs = {}  # dirpath -> {name: size} | None (missing)
        self._lock = threading.Lock()
        self._workers = max(1, workers)

    def prime(self, dirpaths):
        """List every not-yet-known directory, in parallel."""
        todo = [d for d in dict.fromkeys(dirpaths) if d not in self._dirs]
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=min(self._workers, len(todo))) as ex:
            for d, listing in zip(todo, ex.map(_list_dir, todo)):
                self._dirs[d] = listing

    def size_of(self, path):
        """Size of ``path`` on disk, or None if absent. Never hits the network
        for a directory that has already been listed."""
        d, name = os.path.split(path)
        if d not in self._dirs:
            self._dirs[d] = _list_dir(d)
        listing = self._dirs[d]
        return None if listing is None else listing.get(_key(name))

    def ensure_dir(self, dirpath):
        """makedirs only for directories the listing proved to be missing."""
        with self._lock:
            if self._dirs.get(dirpath) is not None:
                return
            os.makedirs(dirpath, exist_ok=True)
            self._dirs[dirpath] = {}

    def record(self, path, size):
        """Fold a freshly written file into the cache so a later zip that ships
        the same map sees it without re-listing the folder."""
        d, name = os.path.split(path)
        with self._lock:
            listing = self._dirs.get(d)
            if listing is None:
                listing = self._dirs[d] = {}
            listing[_key(name)] = size


def plan_zip(source, datafolder, index, dataset=None, replace=False,
             verify_size=True, verbose=False, skip_step4=False):
    """Decide what this source (zip or unzipped folder) still owes the data folder.

    Returns ``(todo, present, stale, excluded)`` where ``todo`` is a list of
    ``(raw, member, size, dst)`` quadruples still to write -- ``raw`` is whatever
    ``source.open()`` needs (a ``ZipInfo`` for a zip, a file path for a folder) --
    ``present`` counts members already on disk, ``stale`` counts members that exist
    at the wrong size (half-written by an interrupted run) and are therefore in
    ``todo``, and ``excluded`` counts step-4 maps left out by ``skip_step4``.
    """
    members = []
    excluded = 0
    for name, size, raw in source.iter_members():
        member = _safe_member(name)
        if member is None:
            continue
        if dataset and member.split("/")[0] != dataset:
            if verbose:
                print(f"  (skip {member}: not dataset {dataset})")
            continue
        if skip_step4 and _is_step4_member(member):
            excluded += 1
            if verbose:
                print(f"  (skip {member}: step-4 map)")
            continue
        members.append((raw, member, size,
                        os.path.join(datafolder, member.replace("/", os.sep))))

    if replace:
        return members, 0, 0, excluded

    index.prime(os.path.dirname(dst) for _, _, _, dst in members)

    todo, present, stale = [], 0, 0
    for raw, member, size, dst in members:
        existing = index.size_of(dst)
        if existing is not None:
            if not verify_size or existing == size:
                present += 1
                if verbose:
                    print(f"  exists, skip: {member}")
                continue
            stale += 1
            if verbose:
                print(f"  size {existing} != {size}, rewrite: {member}")
        todo.append((raw, member, size, dst))
    return todo, present, stale, excluded


def _extract(source, raw, dst, size, index, local):
    """Write one member via a .part temp file + atomic rename."""
    tmp = dst + ".part"
    try:
        with source.open(raw, local) as src, open(tmp, "wb") as out:
            shutil.copyfileobj(src, out, COPY_CHUNK)
        os.replace(tmp, dst)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    index.record(dst, size)


def unpack_zip(source, datafolder, index=None, dataset=None, replace=False,
               dry_run=False, verbose=False, workers=DEFAULT_WORKERS,
               verify_size=True, skip_step4=False):
    """Merge one source (a result zip or an already-unzipped result folder) into
    ``datafolder``. ``source`` may be a ``ZipSource``/``DirSource`` or, for
    backwards compatibility, a plain zip path string. Returns (written, skipped)."""
    if isinstance(source, str):
        source = ZipSource(source)
    if index is None:
        index = DirIndex(workers=workers)
    todo, present, stale, excluded = plan_zip(
        source, datafolder, index, dataset=dataset, replace=replace,
        verify_size=verify_size, verbose=verbose, skip_step4=skip_step4)
    if excluded:
        print(f"  left {excluded} step-4 file(s) in the zip (--no_step4_files)")
    if not todo:
        if present or not excluded:
            print(f"  already complete ({present} file(s)) -- skipped")
        else:
            print("  nothing to merge -- every member was a step-4 file")
        return 0, present
    if stale:
        print(f"  {stale} file(s) present but truncated -- rewriting")
    if dry_run:
        if verbose:
            for _, member, _, _ in todo:
                print(f"  would write: {member}")
        print(f"  would write {len(todo)} file(s), {present} already present")
        return len(todo), present

    for dirpath in dict.fromkeys(os.path.dirname(dst) for _, _, _, dst in todo):
        index.ensure_dir(dirpath)

    local = threading.local()

    def worker(item):
        raw, member, size, dst = item
        _extract(source, raw, dst, size, index, local)
        if verbose:
            print(f"  wrote: {member}")
        return 1

    n = max(1, min(workers, len(todo)))
    with ThreadPoolExecutor(max_workers=n) as ex:
        written = sum(ex.map(worker, todo))
    print(f"  wrote {written} file(s), skipped {present} existing")
    return written, present


def parse_args():
    ap = argparse.ArgumentParser(description="Merge Colab result zips onto the data disk.")
    ap.add_argument("inputs", nargs="+",
                    help="result_*.zip files, already-unzipped result_* folders, "
                         "and/or directories containing either")
    ap.add_argument("--dataset", default=None, help="Only unpack members of this dataset")
    ap.add_argument("--datafolder", default=None,
                    help="Target data folder (default: machine's pipeline data disk)")
    ap.add_argument("--replace", action="store_true", help="Overwrite existing files")
    ap.add_argument("--dry-run", action="store_true", help="Report without writing")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"Parallel listings/copies (default {DEFAULT_WORKERS}; 1 = serial)")
    ap.add_argument("--no-verify-size", dest="verify_size", action="store_false",
                    help="Treat any existing file as done, without comparing its size")
    ap.add_argument("--no_step4_files", "--no-step4-files", dest="skip_step4",
                    action="store_true",
                    help="Do not unpack the step-4 permutation maps (the per-participant "
                         "maps under results/RSA_rnd/; the group mean/ outputs of steps "
                         "5 and 7 are merged either way)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print one line per member instead of one per zip")
    return ap.parse_args()


def main():
    a = parse_args()
    datafolder = a.datafolder or get_paths()[0]
    sources = collect_sources(a.inputs)
    if not sources:
        print("No result zips or unzipped result folders found.")
        return
    print(f"Target datafolder: {datafolder}")
    print(f"{'DRY RUN -- ' if a.dry_run else ''}unpacking {len(sources)} source(s)\n")
    index = DirIndex(workers=a.workers)  # shared across sources: list each folder once
    tot_w = tot_s = 0
    complete = 0
    for source in sources:
        print(source.label)
        w, s = unpack_zip(source, datafolder, index=index, dataset=a.dataset,
                          replace=a.replace, dry_run=a.dry_run, verbose=a.verbose,
                          workers=a.workers, verify_size=a.verify_size,
                          skip_step4=a.skip_step4)
        tot_w += w
        tot_s += s
        if w == 0:
            complete += 1
    verb = "would write" if a.dry_run else "wrote"
    print(f"\nDone: {verb} {tot_w} file(s), skipped {tot_s} existing "
          f"({complete}/{len(sources)} source(s) already complete).")


if __name__ == "__main__":
    main()

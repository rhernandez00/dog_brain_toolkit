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

Accepts any mix of ``result_*.zip`` files and directories containing them. Existing
files are left untouched unless ``--replace`` is given; ``--dry-run`` reports the
planned copies without writing anything.

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
MERGEABLE_SUFFIXES = (".nii.gz", ".json", ".txt", ".npy")

# The listing is a dict lookup where the old code called os.path.exists, so it
# has to reproduce that call's case rules: Windows resolves case-variant
# filenames, Linux does not (the same split make_mask.py warns about). Without
# this a member cased differently from the file on disk would read as missing
# and be rewritten on every run.
_FOLD_CASE = os.name == "nt"


def _key(name):
    return name.lower() if _FOLD_CASE else name


def collect_zips(inputs):
    """Expand files/dirs into a sorted list of result_*.zip paths."""
    zips = []
    for item in inputs:
        if os.path.isdir(item):
            for name in sorted(os.listdir(item)):
                if name.lower().endswith(".zip"):
                    zips.append(os.path.join(item, name))
        elif os.path.isfile(item) and item.lower().endswith(".zip"):
            zips.append(item)
        else:
            print(f"WARNING: skipping {item!r} (not a .zip or directory)")
    return zips


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


def plan_zip(zip_path, datafolder, index, dataset=None, replace=False,
             verify_size=True, verbose=False):
    """Decide what this zip still owes the data folder.

    Returns ``(todo, present, stale)`` where ``todo`` is a list of
    ``(zip_info, member, dst)`` triples still to write, ``present`` counts members
    already on disk, and ``stale`` counts members that exist at the wrong size
    (half-written by an interrupted run) and are therefore in ``todo``.
    """
    members = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            member = _safe_member(info.filename)
            if member is None:
                continue
            if dataset and member.split("/")[0] != dataset:
                if verbose:
                    print(f"  (skip {member}: not dataset {dataset})")
                continue
            members.append((info, member,
                            os.path.join(datafolder, member.replace("/", os.sep))))

    if replace:
        return members, 0, 0

    index.prime(os.path.dirname(dst) for _, _, dst in members)

    todo, present, stale = [], 0, 0
    for info, member, dst in members:
        size = index.size_of(dst)
        if size is not None:
            if not verify_size or size == info.file_size:
                present += 1
                if verbose:
                    print(f"  exists, skip: {member}")
                continue
            stale += 1
            if verbose:
                print(f"  size {size} != {info.file_size}, rewrite: {member}")
        todo.append((info, member, dst))
    return todo, present, stale


def _extract(zf, info, dst, index):
    """Write one member via a .part temp file + atomic rename."""
    tmp = dst + ".part"
    try:
        with zf.open(info) as src, open(tmp, "wb") as out:
            shutil.copyfileobj(src, out, COPY_CHUNK)
        os.replace(tmp, dst)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    index.record(dst, info.file_size)


def unpack_zip(zip_path, datafolder, index=None, dataset=None, replace=False,
               dry_run=False, verbose=False, workers=DEFAULT_WORKERS,
               verify_size=True):
    """Extract one result zip into ``datafolder``. Returns (written, skipped)."""
    if index is None:
        index = DirIndex(workers=workers)
    todo, present, stale = plan_zip(zip_path, datafolder, index, dataset=dataset,
                                    replace=replace, verify_size=verify_size,
                                    verbose=verbose)
    if not todo:
        print(f"  already complete ({present} file(s)) -- skipped")
        return 0, present
    if stale:
        print(f"  {stale} file(s) present but truncated -- rewriting")
    if dry_run:
        if verbose:
            for _, member, _ in todo:
                print(f"  would write: {member}")
        print(f"  would write {len(todo)} file(s), {present} already present")
        return len(todo), present

    for dirpath in dict.fromkeys(os.path.dirname(dst) for _, _, dst in todo):
        index.ensure_dir(dirpath)

    # One ZipFile handle per worker: a single handle is not safe to read from
    # concurrently, and reopening a local zip is cheap next to an SMB write.
    local = threading.local()

    def worker(item):
        zf = getattr(local, "zf", None)
        if zf is None:
            zf = local.zf = zipfile.ZipFile(zip_path)
        info, member, dst = item
        _extract(zf, info, dst, index)
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
    ap.add_argument("inputs", nargs="+", help="result_*.zip files and/or directories")
    ap.add_argument("--dataset", default=None, help="Only unpack members of this dataset")
    ap.add_argument("--datafolder", default=None,
                    help="Target data folder (default: machine's pipeline data disk)")
    ap.add_argument("--replace", action="store_true", help="Overwrite existing files")
    ap.add_argument("--dry-run", action="store_true", help="Report without writing")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"Parallel listings/copies (default {DEFAULT_WORKERS}; 1 = serial)")
    ap.add_argument("--no-verify-size", dest="verify_size", action="store_false",
                    help="Treat any existing file as done, without comparing its size")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="Print one line per member instead of one per zip")
    return ap.parse_args()


def main():
    a = parse_args()
    datafolder = a.datafolder or get_paths()[0]
    zips = collect_zips(a.inputs)
    if not zips:
        print("No result zips found.")
        return
    print(f"Target datafolder: {datafolder}")
    print(f"{'DRY RUN -- ' if a.dry_run else ''}unpacking {len(zips)} zip(s)\n")
    index = DirIndex(workers=a.workers)  # shared across zips: list each folder once
    tot_w = tot_s = 0
    complete = 0
    for z in zips:
        print(os.path.basename(z))
        w, s = unpack_zip(z, datafolder, index=index, dataset=a.dataset,
                          replace=a.replace, dry_run=a.dry_run, verbose=a.verbose,
                          workers=a.workers, verify_size=a.verify_size)
        tot_w += w
        tot_s += s
        if w == 0:
            complete += 1
    verb = "would write" if a.dry_run else "wrote"
    print(f"\nDone: {verb} {tot_w} file(s), skipped {tot_s} existing "
          f"({complete}/{len(zips)} zip(s) already complete).")


if __name__ == "__main__":
    main()

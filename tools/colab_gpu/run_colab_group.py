#!/usr/bin/env python
"""run_colab_group.py -- orchestrate RSA steps 3/5/6/7/8 for many models on a Colab GPU.

The step-1/2/4 companion (``run_colab.py``) is *per participant*: you feed it a
package of one subject's betas and it writes ``result_{model}_{specie}-sub-NN.zip``
per model into OUT_DIR. This script picks up from there. Its input is **that same
OUT_DIR** -- it reads the participant maps straight out of the result zips, so
nothing has to be downloaded, merged and re-uploaded between the two halves.

For each RSA model it:

  1. reads every participant's step-2 real map and step-4 permutation maps once
     into an ``(n_maps, n_mask_voxels)`` matrix;
  2. runs step 3 (group mean/std), step 5 (``reps_group`` group permutations),
     step 6 (voxelwise null mean/std), step 7 (rnd z maps + real z map) and
     step 8 (cluster-size distribution at several thresholds) on the GPU in a
     single voxel-chunked pass;
  3. writes ``result_group_{rsa_model}_{specie}.zip`` into OUT_DIR, with the same
     pipeline-relative arcnames as every other result zip, and clears its scratch
     files.

By default the two bulky intermediates -- step 5's group mean maps and step 7's
rnd z maps -- are **not** written. Their only consumers (steps 6-7 and step 8)
run here, and on EmoC humans they are 727 MB and 1273 MB per model against about
4 MB of actual results. ``write_group_means``/``write_z_maps`` bring them back.

Models whose result zip already exists are skipped, so a disconnected Colab
session resumes by re-running the cell. Each model also gets a ``.started``
marker in ``out_dir`` the moment it begins and loses it on a clean finish --
a marker surviving with no matching result zip means the runtime died
mid-model, so the next run skips that model too instead of retrying it into
the same crash. Delete the marker (or pass ``force=True``) to retry a model
deliberately.

Usable from the notebook (call ``run_group_package``) or as a CLI for local
testing:

    python run_colab_group.py --pkg /content/pkg_group --results /content/drive/MyDrive/rsa_out \
                              --out /content/drive/MyDrive/rsa_out --cpu
"""

import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gpu_group  # noqa: E402
import gpu_rsa    # noqa: E402

# Version -- keep in lockstep with gpu_group.VERSION and the notebook's
# NOTEBOOK_VERSION; gpu_group.check_versions() compares all three and says which
# file on Drive is stale. Bump on every edit to this file.
VERSION = "4.4.0"
LAST_CHANGE = (
    "Per-model .started marker written to out_dir before a model's heavy work "
    "begins, removed on a clean finish (or a clean MissingMapsError skip). A "
    "marker with no matching result zip means the runtime died mid-model; the "
    "next run now skips that model instead of retrying it into the same crash "
    "-- delete the marker, or pass force=True, to retry it deliberately.")


def _clear_model_outputs(work_root, manifest, rsa_model):
    """Drop one model's scratch files once they are safely inside the result zip."""
    import glob
    data_root = os.path.join(work_root, "data")
    for pattern in gpu_group.group_output_globs(manifest, rsa_model):
        for path in glob.glob(os.path.join(data_root, pattern.replace("/", os.sep))):
            try:
                os.remove(path)
            except OSError:
                pass
    # prune the now-empty mean/ and dist/ folders, leaving the rest of the tree intact
    for rel in (f"{manifest['dataset']}/results/RSA/{manifest['model']}/{rsa_model}",
                f"{manifest['dataset']}/results/RSA_rnd/{manifest['model']}/{rsa_model}"):
        d = os.path.join(data_root, rel.replace("/", os.sep))
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


def run_group_package(pkg_root, results_dir, out_dir, work_root=None, models=None,
                      steps=gpu_group.STEPS_ALL, device=None, batch=20000,
                      g_batch=64, write_group_means=False, write_z_maps=False,
                      z_thresholds=gpu_group.DEFAULT_Z_THRESHOLDS, connectivity=3,
                      workers=8, refs_dir=None, specie=None, reps_group=1000,
                      min_percentage_available=1.0, prefetch_zips=False,
                      cache_root=None, manifests=None, force=False,
                      keep_work=False, verbose=True):
    """Run steps 3/5/6/7/8 for every model. Returns written zips.

    ``pkg_root`` is optional. Pass it to use a ``pkg_group_*.zip`` built by
    ``tools/create_group_package.py``; pass ``refs_dir`` instead (and leave
    ``pkg_root`` as ``None``) to recover the manifest from the result zips
    themselves plus the committed reference snapshot, which needs no package and
    no network disk. ``reps_group``/``min_percentage_available``/``specie`` apply
    only to the second form -- with a package they come from its manifest.
    """
    device = device or gpu_rsa.pick_device()
    # One manifest PER MODEL. A results folder can hold more than one analysis --
    # EmoC's mixes 50 mahalanobis models with 41 correlation ones, and the two use
    # different folder layouts -- so a single guessed dis_method silently breaks
    # whichever half it does not describe.
    if pkg_root:
        manifest = gpu_group.load_manifest(pkg_root)
        per_model = None
    else:
        if not refs_dir:
            raise ValueError(
                "Pass either pkg_root (a group package) or refs_dir (the committed "
                "reference snapshot, tools/colab_gpu/refs).")
        # Probing costs one zip open per model -- 1.7 s locally but ~175 s on
        # Colab's Drive. The notebook's preflight has already done it, so accept
        # the result rather than repeating it.
        per_model = manifests
        if per_model is None:
            per_model = gpu_group.discover_model_manifests(
                results_dir if isinstance(results_dir, str) else results_dir[0],
                refs_dir, specie=specie, models=models, reps_group=reps_group,
                min_percentage_available=min_percentage_available,
                verbose=verbose)
        elif verbose:
            print(f"[discover] reusing {len(per_model)} probed manifest(s)")
        manifest = next(iter(per_model.values()))
    anchor = pkg_root or (results_dir if isinstance(results_dir, str)
                          else results_dir[0])
    work_root = work_root or os.path.join(os.path.dirname(os.path.abspath(anchor)),
                                          "group_work")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work_root, exist_ok=True)
    cache_root = cache_root or os.path.join(work_root, "zip_cache")

    sources = results_dir if isinstance(results_dir, (list, tuple)) else [results_dir]
    store = gpu_group.ResultStore(sources, dataset=manifest["dataset"],
                                  verbose=verbose, index_workers=workers)

    specie = manifest["specie"]
    steps = tuple(sorted(set(int(s) for s in steps)))
    if per_model is not None:
        # honour the caller's order (a second instance may run the list reversed)
        model_list = [m for m in (models or sorted(per_model)) if m in per_model]
        unknown = [m for m in (models or []) if m not in per_model]
        if unknown and verbose:
            print(f"    no result zips for: {', '.join(unknown[:5])}"
                  + (" ..." if len(unknown) > 5 else ""))
    else:
        model_list = list(models or manifest["models"])
    written = []

    if verbose:
        print(f"=== group v{VERSION} | {specie}  "
              f"{manifest['dataset']}/{manifest['model']}  "
              f"reps_group={manifest['reps_group']} ===")
        print(f"    device={device}  steps={steps}  models={len(model_list)}")
        print(f"    group_means={'written' if write_group_means else 'skipped'}  "
              f"rnd_z_maps={'written' if write_z_maps else 'skipped'}"
              + (f"  z_thresholds={list(z_thresholds)}" if 8 in steps else ""))

    for i, rsa_model in enumerate(model_list, 1):
        zip_path = os.path.join(out_dir, f"result_group_{rsa_model}_{specie}.zip")
        started_path = os.path.join(
            out_dir, f"result_group_{rsa_model}_{specie}.started")
        if os.path.exists(zip_path) and not force:
            if verbose:
                print(f"[{i}/{len(model_list)}] {rsa_model}: result exists -- skipping.")
            continue
        if os.path.exists(started_path) and not force:
            if verbose:
                print(f"[{i}/{len(model_list)}] {rsa_model}: a previous attempt "
                      f"started this and never finished (crash, or still running "
                      f"in another session) -- skipping. Delete "
                      f"{os.path.basename(started_path)} to retry it, or pass "
                      f"force=True.")
            continue
        # this model's OWN parameters, not the batch's
        m_manifest = per_model[rsa_model] if per_model is not None else manifest
        if verbose:
            fold = (f"/{m_manifest['mah_fold']}"
                    if m_manifest["dis_method"] == "mahalanobis" else "")
            print(f"[{i}/{len(model_list)}] {rsa_model}: "
                  f"{m_manifest['dis_method']}{fold}/{m_manifest['rsa_method']}  "
                  f"r-{m_manifest['radius']}  reps={m_manifest['reps']}  "
                  f"{len(gpu_group.units(m_manifest))} unit(s)")
        t0 = time.time()
        # Written before the heavy work and removed on a clean finish. If the
        # runtime gets OOM-killed mid-model, this is what survives on Drive to
        # tell the next run which model to skip instead of retrying.
        with open(started_path, "w") as f:
            json.dump({"started": t0, "pid": os.getpid()}, f)
        # the namelist cache only helps within one model; keeping it for the
        # whole battery just accumulates listings of zips already finished
        store.forget()
        gpu_group.mem_report(f"start {rsa_model}", device)
        model_store, cache_dir = store, None
        if prefetch_zips:
            model_store, cache_dir = gpu_group.prefetch_model_zips(
                store, m_manifest, rsa_model, os.path.join(cache_root, rsa_model),
                workers=workers, verbose=verbose)
        try:
            gpu_group.run_group_model(
                pkg_root, work_root, m_manifest, rsa_model, model_store, steps=steps,
                device=device, vox_batch=batch, g_batch=g_batch,
                write_group_means=write_group_means, write_z_maps=write_z_maps,
                z_thresholds=z_thresholds, connectivity=connectivity,
                workers=workers, refs_dir=refs_dir, verbose=verbose)
        except gpu_group.MissingMapsError as exc:
            print(f"[{i}/{len(model_list)}] {rsa_model}: SKIPPED -- {exc}")
            _clear_model_outputs(work_root, m_manifest, rsa_model)
            os.remove(started_path)  # not a crash -- let a later run retry it
            continue
        finally:
            # the cached zips are this model's alone; keeping them would grow
            # local scratch by ~1.4 GB per model across a battery
            if cache_dir:
                shutil.rmtree(cache_dir, ignore_errors=True)
        zip_path, n_files = gpu_group.zip_group_result(work_root, m_manifest,
                                                       rsa_model, out_dir)
        os.remove(started_path)
        if not keep_work:
            _clear_model_outputs(work_root, m_manifest, rsa_model)
        written.append(zip_path)
        gpu_group.mem_report(f"done {rsa_model}", device)
        if verbose:
            size_mb = os.path.getsize(zip_path) / 1e6
            print(f"[{i}/{len(model_list)}] {rsa_model}: {n_files} file(s), "
                  f"{size_mb:.1f} MB in {time.time() - t0:.1f}s "
                  f"-> {os.path.basename(zip_path)}")

    if verbose:
        print(f"=== finished: {len(written)} new group result zip(s) in {out_dir} ===")
    return written


def parse_args():
    ap = argparse.ArgumentParser(
        description="Run RSA group steps 3/5/6/7 for a Colab group package.")
    ap.add_argument("--pkg", default=None,
                    help="Unzipped group package root (has manifest.json). "
                         "Optional -- omit it and pass --refs to recover the "
                         "manifest from the result zips instead.")
    ap.add_argument("--refs", default=os.path.join(HERE, "refs"),
                    help="Reference snapshot folder (default: tools/colab_gpu/refs). "
                         "Used when --pkg is omitted; supplies the mask and the "
                         "config's participant list.")
    ap.add_argument("--specie", choices=["D", "H"], default=None,
                    help="Required with --refs when the results folder holds both")
    ap.add_argument("--reps_group", type=int, default=1000,
                    help="Group permutations (--refs mode only; with a package it "
                         "comes from the manifest)")
    ap.add_argument("--min_percentage_available", type=float, default=1.0,
                    help="Required fraction of the config's participants "
                         "(--refs mode only)")
    ap.add_argument("--prefetch_zips", action="store_true",
                    help="Copy each model's result zips to local disk before "
                         "reading their members. Measured a NET LOSS on a "
                         "Windows Drive mount (cold, 8 threads: direct 128 s vs "
                         "prefetch 176 s) because the mount is bandwidth-bound, "
                         "not latency-bound. Kept because it is mount-dependent; "
                         "measure before using it.")
    ap.add_argument("--results", required=True, nargs="+",
                    help="Folder(s) with result_*.zip from the step-1/2/4 run, "
                         "and/or an unpacked data root")
    ap.add_argument("--out", required=True,
                    help="Output dir for result_group_*.zip (a Drive folder)")
    ap.add_argument("--work", default=None, help="Scratch dir for the maps being built")
    ap.add_argument("--steps", type=int, nargs="+", default=list(gpu_group.STEPS_ALL),
                    help="Subset of 3 5 6 7 8 (default: all five; 7 needs 3's "
                         "output, 8 needs 7 in the same run)")
    ap.add_argument("--models", nargs="*", default=None, help="Subset of models to run")
    ap.add_argument("--batch", type=int, default=20000, help="Voxel chunk size")
    ap.add_argument("--g_batch", type=int, default=64,
                    help="Group permutations gathered per GPU pass")
    ap.add_argument("--workers", type=int, default=8,
                    help="Threads for reading result zips / writing niftis")
    ap.add_argument("--write_group_means", action="store_true",
                    help="Write step 5's reps_group group mean maps. Off by "
                         "default: they are inputs to steps 6-7 only, both of "
                         "which run here (measured 727 kB each on the EmoC human "
                         "grid, so 727 MB per model at reps_group=1000).")
    ap.add_argument("--write_z_maps", action="store_true",
                    help="Write step 7's reps_group rnd z maps. Off by default: "
                         "step 8 is their only consumer and it runs here (1273 kB "
                         "each on the EmoC human grid = 1.27 GB per model). Turn "
                         "on only to run step 8 on the workstation instead.")
    ap.add_argument("--z_thresholds", type=float, nargs="+",
                    default=list(gpu_group.DEFAULT_Z_THRESHOLDS),
                    help="Step 8 cluster-forming thresholds. All are computed in "
                         "one pass over the z maps and stored under separate keys, "
                         "so step 9 can pick any of them later with --z_threshold.")
    ap.add_argument("--connectivity", type=int, default=3, choices=[1, 2, 3],
                    help="Cluster connectivity for step 8 (3 = 26-connected, "
                         "FSL's default and the CPU step's)")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (default: GPU if available)")
    ap.add_argument("--force", action="store_true", help="Recompute even if results exist")
    ap.add_argument("--keep_work", action="store_true",
                    help="Keep the scratch tree after zipping (debugging)")
    return ap.parse_args()


def main():
    import torch
    a = parse_args()
    device = torch.device("cpu") if a.cpu else gpu_rsa.pick_device()
    run_group_package(a.pkg, a.results, a.out, work_root=a.work, models=a.models,
                      steps=a.steps, device=device, batch=a.batch, g_batch=a.g_batch,
                      write_group_means=a.write_group_means,
                      write_z_maps=a.write_z_maps, z_thresholds=a.z_thresholds,
                      connectivity=a.connectivity, workers=a.workers,
                      refs_dir=None if a.pkg else a.refs, specie=a.specie,
                      reps_group=a.reps_group,
                      min_percentage_available=a.min_percentage_available,
                      prefetch_zips=a.prefetch_zips,
                      force=a.force, keep_work=a.keep_work)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""run_colab_group.py -- orchestrate RSA steps 3/5/6/7 for many models on a Colab GPU.

The step-1/2/4 companion (``run_colab.py``) is *per participant*: you feed it a
package of one subject's betas and it writes ``result_{model}_{specie}-sub-NN.zip``
per model into OUT_DIR. This script picks up from there. Its input is **that same
OUT_DIR** -- it reads the participant maps straight out of the result zips, so
nothing has to be downloaded, merged and re-uploaded between the two halves.

For each RSA model it:

  1. reads every participant's step-2 real map and step-4 permutation maps once
     into an ``(n_maps, n_mask_voxels)`` matrix;
  2. runs step 3 (group mean/std), step 5 (``reps_group`` group permutations),
     step 6 (voxelwise null mean/std) and step 7 (rnd z maps + real z map) on the
     GPU in a single voxel-chunked pass;
  3. writes ``result_group_{rsa_model}_{specie}.zip`` into OUT_DIR, with the same
     pipeline-relative arcnames as every other result zip, and clears its scratch
     files.

Models whose result zip already exists are skipped, so a disconnected Colab
session resumes by re-running the cell.

Usable from the notebook (call ``run_group_package``) or as a CLI for local
testing:

    python run_colab_group.py --pkg /content/pkg_group --results /content/drive/MyDrive/rsa_out \
                              --out /content/drive/MyDrive/rsa_out --cpu
"""

import argparse
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gpu_group  # noqa: E402
import gpu_rsa    # noqa: E402


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
    # prune the now-empty mean/ folders, leaving the rest of the tree intact
    for rel in (f"{manifest['dataset']}/results/RSA/{manifest['model']}/{rsa_model}",
                f"{manifest['dataset']}/results/RSA_rnd/{manifest['model']}/{rsa_model}"):
        d = os.path.join(data_root, rel.replace("/", os.sep))
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)


def run_group_package(pkg_root, results_dir, out_dir, work_root=None, models=None,
                      steps=gpu_group.STEPS_ALL, device=None, batch=20000,
                      g_batch=64, write_group_means=True, workers=8, force=False,
                      keep_work=False, verbose=True):
    """Run steps 3/5/6/7 for every model in the package. Returns written zips."""
    manifest = gpu_group.load_manifest(pkg_root)
    device = device or gpu_rsa.pick_device()
    work_root = work_root or os.path.join(os.path.dirname(os.path.abspath(pkg_root)),
                                          "group_work")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work_root, exist_ok=True)

    sources = results_dir if isinstance(results_dir, (list, tuple)) else [results_dir]
    store = gpu_group.ResultStore(sources, dataset=manifest["dataset"], verbose=verbose)

    specie = manifest["specie"]
    steps = tuple(sorted(set(int(s) for s in steps)))
    model_list = list(models or manifest["models"])
    written = []

    if verbose:
        print(f"=== group {specie}  {manifest['dataset']}/{manifest['model']}  "
              f"r-{manifest['radius']}  {manifest['dis_method']}/{manifest['rsa_method']}  "
              f"reps={manifest['reps']}  reps_group={manifest['reps_group']} ===")
        print(f"    device={device}  steps={steps}  models={len(model_list)}  "
              f"units={len(gpu_group.units(manifest))}  "
              f"group_means={'written' if write_group_means else 'skipped'}")

    for i, rsa_model in enumerate(model_list, 1):
        zip_path = os.path.join(out_dir, f"result_group_{rsa_model}_{specie}.zip")
        if os.path.exists(zip_path) and not force:
            if verbose:
                print(f"[{i}/{len(model_list)}] {rsa_model}: result exists -- skipping.")
            continue
        t0 = time.time()
        try:
            gpu_group.run_group_model(
                pkg_root, work_root, manifest, rsa_model, store, steps=steps,
                device=device, vox_batch=batch, g_batch=g_batch,
                write_group_means=write_group_means, workers=workers,
                verbose=verbose)
        except gpu_group.MissingMapsError as exc:
            print(f"[{i}/{len(model_list)}] {rsa_model}: SKIPPED -- {exc}")
            _clear_model_outputs(work_root, manifest, rsa_model)
            continue
        zip_path, n_files = gpu_group.zip_group_result(work_root, manifest,
                                                       rsa_model, out_dir)
        if not keep_work:
            _clear_model_outputs(work_root, manifest, rsa_model)
        written.append(zip_path)
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
    ap.add_argument("--pkg", required=True,
                    help="Unzipped group package root (has manifest.json)")
    ap.add_argument("--results", required=True, nargs="+",
                    help="Folder(s) with result_*.zip from the step-1/2/4 run, "
                         "and/or an unpacked data root")
    ap.add_argument("--out", required=True,
                    help="Output dir for result_group_*.zip (a Drive folder)")
    ap.add_argument("--work", default=None, help="Scratch dir for the maps being built")
    ap.add_argument("--steps", type=int, nargs="+", default=list(gpu_group.STEPS_ALL),
                    help="Subset of 3 5 6 7 (default: all four; 7 needs 3's output)")
    ap.add_argument("--models", nargs="*", default=None, help="Subset of models to run")
    ap.add_argument("--batch", type=int, default=20000, help="Voxel chunk size")
    ap.add_argument("--g_batch", type=int, default=64,
                    help="Group permutations gathered per GPU pass")
    ap.add_argument("--workers", type=int, default=8,
                    help="Threads for reading result zips / writing niftis")
    ap.add_argument("--skip_group_means", action="store_true",
                    help="Do not write step 5's reps_group group mean maps. They "
                         "are inputs to steps 6-7 only, both of which run here, so "
                         "this halves the output with no effect on steps 8-10.")
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
                      write_group_means=not a.skip_group_means, workers=a.workers,
                      force=a.force, keep_work=a.keep_work)


if __name__ == "__main__":
    main()

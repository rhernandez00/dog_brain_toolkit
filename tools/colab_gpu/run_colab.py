#!/usr/bin/env python
"""run_colab.py -- orchestrate RSA steps 1/2/4 for one package on a Colab GPU.

Given an unzipped package directory (``pkg_root`` -- contains ``manifest.json``,
``data/``, ``code/``) and an output directory (a Google Drive folder), this:

  1. Runs step 1 once (unless the manifest says ``step1_done`` or the maps are
     already present), then writes ``result_step1_{specie}-sub-NN.zip`` to OUT_DIR.
  2. For each model in the manifest, computes steps 2 + 4 and writes
     ``result_{model}_{specie}-sub-NN.zip`` to OUT_DIR -- skipping any model whose
     result zip already exists (so a disconnected/rerun session resumes cleanly).

The step-1 maps are loaded once into a per-voxel matrix and reused across every
model (see gpu_rsa.load_meta_similarity), so the per-model cost is a few matmuls.

Usable from the notebook (call ``run_package``) or as a CLI for local testing:

    python run_colab.py --pkg /content/pkg --out /content/drive/MyDrive/rsa_out
"""

import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gpu_rsa  # noqa: E402


def _step1_present(pkg_root, manifest):
    """True if all step-1 maps are already in the package (bundled or computed).

    Mahalanobis expects ``n_pairs`` maps under the subject folder; correlation
    expects ``n_pairs`` per-run maps in every run folder (recursive glob)."""
    data_root = os.path.join(pkg_root, "data")
    base = os.path.join(data_root, manifest["dataset"], "results", "RSA",
                        manifest["model"], f"{manifest['specie']}-sub-{manifest['sub_N']:02d}")
    n_pairs = len(manifest["pairs"])
    radius = manifest["radius"]
    if manifest.get("dis_method", "mahalanobis") == "correlation":
        found = glob.glob(os.path.join(base, "**", f"r-{radius}_correlation_*.nii.gz"),
                          recursive=True)
        return len(found) >= n_pairs * len(manifest["runs"])
    found = glob.glob(os.path.join(base, f"r-{radius}_mahalanobis_*.nii.gz"))
    return len(found) >= n_pairs


def run_package(pkg_root, out_dir, device=None, batch=1024, models=None,
                force=False, verbose=True):
    """Run step 1 (if needed) + every model for one package. Returns written zips."""
    manifest = gpu_rsa.load_manifest(pkg_root)
    device = device or gpu_rsa.pick_device()
    os.makedirs(out_dir, exist_ok=True)
    specie, sub_N = manifest["specie"], manifest["sub_N"]
    written = []

    if verbose:
        print(f"=== {specie}-sub-{sub_N:02d}  {manifest['dataset']}/{manifest['model']}  "
              f"r-{manifest['radius']}  {manifest['rsa_method']}  reps={manifest['reps']} ===")
        print(f"    device={device}  models={len(manifest['models'])}  "
              f"step1_done={manifest.get('step1_done')}")

    # ---- Step 1 (once) ----
    step1_zip = os.path.join(out_dir, f"result_step1_{specie}-sub-{sub_N:02d}.zip")
    if _step1_present(pkg_root, manifest) and not force:
        if verbose:
            print("[step1] maps already present in package -- skipping compute.")
    else:
        gpu_rsa.run_step1(pkg_root, manifest, device=device, batch=batch, verbose=verbose)
    if not os.path.exists(step1_zip) or force:
        # only publish a step-1 result zip when we actually computed it here
        if not manifest.get("step1_done"):
            gpu_rsa.zip_step1_result(pkg_root, manifest, out_dir)
            written.append(step1_zip)
            if verbose:
                print(f"[step1] -> {step1_zip}")

    # ---- Steps 2 + 4 (per model), meta loaded once ----
    meta = gpu_rsa.load_meta(os.path.join(pkg_root, "data"), manifest, device)
    model_list = models or manifest["models"]
    for i, rsa_model in enumerate(model_list, 1):
        zip_path = os.path.join(out_dir, f"result_{rsa_model}_{specie}-sub-{sub_N:02d}.zip")
        if os.path.exists(zip_path) and not force:
            if verbose:
                print(f"[{i}/{len(model_list)}] {rsa_model}: result exists -- skipping.")
            continue
        t0 = time.time()
        gpu_rsa.run_model(pkg_root, manifest, rsa_model, meta=meta, device=device,
                          verbose=False)
        gpu_rsa.zip_model_result(pkg_root, manifest, rsa_model, out_dir)
        written.append(zip_path)
        if verbose:
            print(f"[{i}/{len(model_list)}] {rsa_model}: done in {time.time()-t0:.1f}s "
                  f"-> {os.path.basename(zip_path)}")

    if verbose:
        print(f"=== finished: {len(written)} new result zip(s) in {out_dir} ===")
    return written


def parse_args():
    ap = argparse.ArgumentParser(description="Run RSA steps 1/2/4 for a Colab package.")
    ap.add_argument("--pkg", required=True, help="Unzipped package root (has manifest.json)")
    ap.add_argument("--out", required=True, help="Output dir for result_*.zip (Drive folder)")
    ap.add_argument("--batch", type=int, default=1024, help="Searchlight voxel batch size")
    ap.add_argument("--cpu", action="store_true", help="Force CPU (default: GPU if available)")
    ap.add_argument("--force", action="store_true", help="Recompute even if results exist")
    ap.add_argument("--models", nargs="*", default=None, help="Subset of models to run")
    return ap.parse_args()


def main():
    import torch
    a = parse_args()
    device = torch.device("cpu") if a.cpu else gpu_rsa.pick_device()
    run_package(a.pkg, a.out, device=device, batch=a.batch, models=a.models, force=a.force)


if __name__ == "__main__":
    main()

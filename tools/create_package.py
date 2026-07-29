#!/usr/bin/env python
"""create_package.py -- build a self-contained Colab package for RSA steps 1, 2, 4.

Assembles one participant's data + a set of RSA models + the GPU code + a Colab
notebook into a single ``.zip`` you drop into a Google Drive folder. Colab unzips
it, runs steps 1/2/4 on the GPU (see ``tools/colab_gpu/gpu_rsa.py``), and emits one
result ``.zip`` per finished part; ``tools/unpack_results.py`` merges those back
onto the pipeline data disk.

Usage (from the repo root, with the full Anaconda interpreter -- see CLAUDE.md):

    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\create_package.py D 1 \\
        --models valence3__all valence3__cross --reps 10

    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\create_package.py H 40 \\
        --all-stim-wise

``--all-stim-wise`` expands to every stim-wise model x grouping in the dataset's
``rsa_models/_models.csv`` (via tools/models_manifest.py) -- 50 models for EmoC.

If the participant's 45 step-1 pairwise maps already exist on disk, they are
bundled and the manifest flags ``step1_done`` so Colab skips step 1.

Only the Mahalanobis ``stim-wise`` path is supported (the pipeline default and the
one the GPU port accelerates).
"""

import argparse
import datetime
import json
import os
import sys
import zipfile

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

import models_manifest as mm          # noqa: E402
import rsa_utils                      # noqa: E402
from scheduler.paths import get_paths  # noqa: E402

COLAB_DIR = os.path.join(HERE, "colab_gpu")
DEFAULT_OUT = os.path.join(COLAB_DIR, "packages")


# ---------------------------------------------------------------------------
def stimwise_category(stim, dataset):
    if dataset == "EmoB":
        return stim.split("-")[0]
    if dataset == "EmoC":
        return stim[:-1]
    raise ValueError(f"Dataset {dataset!r} not supported for stim-wise categories.")


def get_runs(datafolder, dataset, specie, sub_N):
    """Session/run list for a participant (mirrors rsa_utils.get_session_and_run_dict)."""
    db = pd.read_csv(os.path.join(datafolder, dataset, "BIDS", f"{specie}_database-details.csv"))
    db = db[db["sub_N"] == sub_N].reset_index(drop=True)
    return [{"session": int(r["session"]), "run_N": int(r["run_N"])} for _, r in db.iterrows()]


def beta_relpath(dataset, model, specie, sub_N, session, run_N, task, stim):
    """Package-relative path (under data/) of one aligned beta map.

    Mirrors ``rsa_utils.beta_map_path``: the step-0.5 layout, not the raw FEAT
    ``stats/pe*`` files. The GPU port masks and folds these by array index just
    like the CPU pipeline, so it must be fed maps that are known to share one
    voxel grid.
    """
    return os.path.join(
        "data", dataset, "results", "GLM", model, f"{specie}-sub-{sub_N:02d}",
        f"ses-{int(session):02d}_task-{task}_run-{int(run_N):02d}",
        f"beta_{stim}.nii.gz",
    ).replace(os.sep, "/")


def resolve_models(datafolder, dataset, explicit, all_flag, all_stim_wise, dis_method):
    """Ordered, de-duplicated model list from --models, --all (by dis_method) and
    --all-stim-wise (mahalanobis alias)."""
    dirs = mm.rsa_models_dirs(datafolder, dataset)
    names = []
    if all_stim_wise:
        names += mm.concrete_models_for_dis_method(dirs, "mahalanobis")
    if all_flag:
        names += mm.concrete_models_for_dis_method(dirs, dis_method)
    names += list(explicit or [])
    seen, out = set(), []
    rsa_dir = os.path.join(datafolder, dataset, "rsa_models")
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        if not os.path.exists(os.path.join(rsa_dir, f"{n}.csv")):
            raise FileNotFoundError(f"RSA model CSV not found: {os.path.join(rsa_dir, n + '.csv')}")
        out.append(n)
    if not out:
        raise ValueError("No models selected. Pass --models, --all, and/or --all-stim-wise.")
    return out


def step1_maps_on_disk(datafolder, dataset, model, specie, sub_N, radius, pairs):
    """Mahalanobis: return the 45 existing step-1 map paths (either orientation) as
    ``(arcname, src)`` if ALL present, else None."""
    base = os.path.join(datafolder, dataset, "results", "RSA", model, f"{specie}-sub-{sub_N:02d}")
    sub = f"{specie}-sub-{sub_N:02d}"
    found = []
    for c1, c2 in pairs:
        a = os.path.join(base, f"r-{radius}_mahalanobis_{c1}_{c2}.nii.gz")
        b = os.path.join(base, f"r-{radius}_mahalanobis_{c2}_{c1}.nii.gz")
        src = a if os.path.exists(a) else (b if os.path.exists(b) else None)
        if src is None:
            return None
        arc = f"data/{dataset}/results/RSA/{model}/{sub}/{os.path.basename(src)}"
        found.append((arc, src))
    return found


def step1_corr_maps_on_disk(datafolder, dataset, model, specie, sub_N, radius, task,
                            runs, pairs):
    """Correlation: return all per-run step-1 map paths (one orientation each) as
    ``(arcname, src)`` if ALL 780 pairs exist in EVERY run folder, else None."""
    sub = f"{specie}-sub-{sub_N:02d}"
    found = []
    for entry in runs:
        session = f"{int(entry['session']):02d}"
        run_N = int(entry["run_N"])
        run_folder = f"ses-{session}_task-{task}_run-{run_N:02d}"
        base = os.path.join(datafolder, dataset, "results", "RSA", model, sub, run_folder)
        for s1, s2 in pairs:
            a = os.path.join(base, f"r-{radius}_correlation_{s1}_{s2}.nii.gz")
            b = os.path.join(base, f"r-{radius}_correlation_{s2}_{s1}.nii.gz")
            src = a if os.path.exists(a) else (b if os.path.exists(b) else None)
            if src is None:
                return None
            arc = (f"data/{dataset}/results/RSA/{model}/{sub}/{run_folder}/"
                   f"{os.path.basename(src)}")
            found.append((arc, src))
    return found


# ---------------------------------------------------------------------------
def build_package(specie, sub_N, models, all_flag, all_stim_wise, dataset, model,
                  radius, dis_method, mah_fold, rsa_method, reps, mask_type, out_dir,
                  verbose=True, allow_space_mismatch=False):
    if specie not in ("D", "H"):
        raise ValueError("specie must be 'D' or 'H'")
    if dis_method not in ("mahalanobis", "correlation"):
        raise ValueError("dis_method must be 'mahalanobis' or 'correlation'")
    if dis_method == "mahalanobis" and mah_fold != "stim-wise":
        raise ValueError("Only mah_fold='stim-wise' is supported for mahalanobis packages.")

    datafolder, _git, _py = get_paths()
    if radius is None:
        radius = 3 if specie == "D" else 4

    cfg_path = os.path.join(datafolder, dataset, "config_files", f"{specie}_{model}.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    task = cfg.get("task", dataset)
    stim_types = cfg["stim_types"]

    # categories & pairs differ by dis_method: mahalanobis collapses to stim-wise
    # categories (10), correlation uses the full per-stimulus set (40 -> 780 pairs).
    if dis_method == "correlation":
        categories = list(stim_types)
    else:
        categories = sorted({stimwise_category(s, dataset) for s in stim_types})
    pairs = [[a, b] for i, a in enumerate(categories) for b in categories[i + 1:]]

    model_list = resolve_models(datafolder, dataset, models, all_flag, all_stim_wise,
                                dis_method)
    runs = get_runs(datafolder, dataset, specie, sub_N)
    if not runs:
        raise ValueError(f"No runs found for {specie}-sub-{sub_N:02d} in the database.")

    if dis_method == "correlation":
        existing_step1 = step1_corr_maps_on_disk(
            datafolder, dataset, model, specie, sub_N, radius, task, runs, pairs)
    else:
        existing_step1 = step1_maps_on_disk(
            datafolder, dataset, model, specie, sub_N, radius, pairs)
    step1_done = existing_step1 is not None

    mask_src = os.path.join(datafolder, dataset, "ROI", specie, f"{mask_type}.nii.gz")
    if not os.path.exists(mask_src):
        raise FileNotFoundError(f"Mask not found: {mask_src}")

    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset, "model": model, "specie": specie, "sub_N": sub_N,
        "task": task, "radius": radius, "mask_type": mask_type,
        "dis_method": dis_method, "mah_fold": mah_fold, "rsa_method": rsa_method,
        "reps": reps, "stim_types": stim_types, "categories": categories,
        "pairs": pairs, "runs": runs, "models": model_list, "step1_done": step1_done,
        # gpu_rsa runs the same voxel-grid check on Colab; without this the package
        # would build here and then fail there
        "allow_space_mismatch": bool(allow_space_mismatch),
    }

    os.makedirs(out_dir, exist_ok=True)
    zip_name = f"pkg_{specie}-sub-{sub_N:02d}_{dataset}_{model}_{dis_method}.zip"
    zip_path = os.path.join(out_dir, zip_name)

    # The GPU port masks and folds these betas by array index exactly as the CPU
    # pipeline does, so the same voxel-grid invariant applies -- check before
    # shipping gigabytes to Colab rather than after.
    beta_srcs = []
    for entry in runs:
        for stim in stim_types:
            rel = beta_relpath(dataset, model, specie, sub_N,
                               entry["session"], entry["run_N"], task, stim)
            src = os.path.join(datafolder, rel[len("data/"):].replace("/", os.sep))
            if not os.path.exists(src):
                raise FileNotFoundError(
                    f"Aligned beta map not found: {src}\n"
                    f"Run step 0.5 for {specie}-sub-{sub_N:02d} first:\n"
                    f"  python searchlight.py --dataset {dataset} --model {model} "
                    f"--specie {specie} --steps_to_run 0.5\n"
                    "Packages deliberately ship only step-0.5 output -- the raw FEAT "
                    "pe maps are scanner-native for humans and would make the GPU "
                    "results anatomically meaningless."
                )
            beta_srcs.append((rel, src))
    rsa_utils.check_same_space(
        (f"mask {os.path.basename(mask_src)}", mask_src),
        [(os.path.basename(os.path.dirname(src)) + "/" + os.path.basename(src), src)
         for _, src in beta_srcs],
        context=f"packaging {specie}-sub-{sub_N:02d} of {dataset} for Colab",
        strict=not allow_space_mismatch,
    )

    n_betas = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        # betas (only the odd pe maps actually used)
        for rel, src in beta_srcs:
            zf.write(src, arcname=rel)
            n_betas += 1
        # mask
        zf.write(mask_src, arcname=f"data/{dataset}/ROI/{specie}/{mask_type}.nii.gz")
        # model CSVs
        for m in model_list:
            zf.write(os.path.join(datafolder, dataset, "rsa_models", f"{m}.csv"),
                     arcname=f"data/{dataset}/rsa_models/{m}.csv")
        # config
        zf.write(cfg_path, arcname=f"data/{dataset}/config_files/{specie}_{model}.yaml")
        # bundled step-1 maps (if already computed) -- arcnames come from the helper
        if step1_done:
            for arc, src in existing_step1:
                zf.write(src, arcname=arc)
        # code + notebook
        zf.write(os.path.join(COLAB_DIR, "gpu_rsa.py"), arcname="code/gpu_rsa.py")
        zf.write(os.path.join(COLAB_DIR, "run_colab.py"), arcname="code/run_colab.py")
        zf.write(os.path.join(COLAB_DIR, "colab_rsa.ipynb"), arcname="colab_rsa.ipynb")
        # manifest + readme
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("README.txt", _package_readme(manifest, zip_name))

    if verbose:
        size_mb = os.path.getsize(zip_path) / 1e6
        print(f"Package: {zip_path}")
        print(f"  {specie}-sub-{sub_N:02d}  {dataset}/{model}  {dis_method}  r-{radius}  "
              f"{rsa_method}  reps={reps}")
        print(f"  runs={len(runs)}  betas={n_betas}  models={len(model_list)}  "
              f"pairs={len(pairs)}  step1_done={step1_done}  size={size_mb:.1f} MB")
    return zip_path


def _package_readme(m, zip_name):
    return (
        f"Colab RSA package: {zip_name}\n"
        f"{m['specie']}-sub-{m['sub_N']:02d}  {m['dataset']}/{m['model']}  "
        f"dis_method={m['dis_method']}  radius={m['radius']}  "
        f"rsa_method={m['rsa_method']}  reps={m['reps']}\n"
        f"models: {len(m['models'])}   step1_done: {m['step1_done']}\n\n"
        "To run on Colab:\n"
        "  1. Upload this .zip to a Google Drive folder.\n"
        "  2. Open colab_rsa.ipynb in Colab (GPU runtime: L4 or T4, High-RAM).\n"
        "  3. Set PKG_ZIP to this file's Drive path and OUT_DIR to a Drive output folder.\n"
        "  4. Run all cells. One result_*.zip is written to OUT_DIR per finished part\n"
        "     (result_step1_*.zip once, then result_<model>_*.zip per model; resumable).\n"
        "  5. Back on the workstation: tools/unpack_results.py <OUT_DIR downloads> to\n"
        "     merge results onto the data disk, then run pipeline steps 3-10 as usual.\n"
    )


def parse_args():
    ap = argparse.ArgumentParser(description="Build a Colab GPU package for RSA steps 1/2/4.")
    ap.add_argument("specie", choices=["D", "H"], help="'D' (dog) or 'H' (human)")
    ap.add_argument("sub_N", type=int, help="Subject number, e.g. 40")
    ap.add_argument("--models", nargs="*", default=[], help="Explicit RSA model names (CSV stems)")
    ap.add_argument("--dis_method", default="mahalanobis",
                    choices=["mahalanobis", "correlation"],
                    help="Distance method / model family (default mahalanobis)")
    ap.add_argument("--all", dest="all_flag", action="store_true",
                    help="Add every model for --dis_method from _models.csv")
    ap.add_argument("--all-stim-wise", action="store_true",
                    help="Add every mahalanobis stim-wise model x grouping (alias)")
    ap.add_argument("--dataset", default="EmoC")
    ap.add_argument("--model", default="basic-block", help="GLM model")
    ap.add_argument("--radius", type=int, default=None, help="Searchlight radius (default 3 D / 4 H)")
    ap.add_argument("--mah_fold", default="stim-wise")
    ap.add_argument("--rsa_method", default="kendall", choices=["kendall", "pearson", "correlation"])
    ap.add_argument("--reps", type=int, default=100, help="Permutations for step 4")
    ap.add_argument("--mask_type", default="b_GreyMatter2mmB")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output folder for the .zip")
    ap.add_argument("--allow_space_mismatch", action="store_true",
                    help=("Downgrade the voxel-grid check to a warning and package "
                          "anyway. Recorded in the manifest so the Colab run honours "
                          "it too. The betas are then masked and folded by array "
                          "index regardless of their affines."))
    return ap.parse_args()


def main():
    a = parse_args()
    build_package(a.specie, a.sub_N, a.models, a.all_flag, a.all_stim_wise, a.dataset,
                  a.model, a.radius, a.dis_method, a.mah_fold, a.rsa_method, a.reps,
                  a.mask_type, a.out, allow_space_mismatch=a.allow_space_mismatch)


if __name__ == "__main__":
    main()

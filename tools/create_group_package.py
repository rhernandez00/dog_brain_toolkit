#!/usr/bin/env python
"""create_group_package.py -- build a Colab package for RSA steps 3, 5, 6, 7.

The per-participant sibling, ``tools/create_package.py``, ships gigabytes of beta
maps because steps 1/2/4 start from the raw data. The **group** steps do not: they
start from the maps a Colab run has *already written* into OUT_DIR
(``result_{model}_{specie}-sub-NN.zip``). So this package carries no imaging data
beyond the searchlight mask -- it is a few tens of kilobytes of manifest, mask and
code, and the Colab run reads the participant maps straight out of the result zips
next to it on Drive.

What ends up on Colab:

    manifest.json           parameters, participant list, model list
    data/{dataset}/ROI/{specie}/{mask_type}.nii.gz
    code/gpu_rsa.py         (voxel-grid check, device pick, mask loader)
    code/gpu_group.py       steps 3/5/6/7 kernels
    code/run_colab_group.py orchestrator
    colab_rsa_group.ipynb   the notebook

Usage (from the repo root, full Anaconda interpreter -- see CLAUDE.md):

    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\create_group_package.py D --all-stim-wise
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\create_group_package.py H \\
        --models valence3__all valence3__cross --reps_group 1000

``--all-stim-wise`` expands to every stim-wise model x grouping in the dataset's
``rsa_models/_models.csv`` (via tools/models_manifest.py), exactly like
``create_package.py``.
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
from create_package import resolve_models  # noqa: E402  -- same selection rules
from scheduler.paths import get_paths  # noqa: E402

COLAB_DIR = os.path.join(HERE, "colab_gpu")
DEFAULT_OUT = os.path.join(COLAB_DIR, "packages")


def runs_by_sub(datafolder, dataset, specie, participants):
    """``{sub_N: [{'session','run_N'}, ...]}`` -- mirrors get_session_and_run_dict.

    Only the per-run layouts (correlation, ``stim-wise-all-runs``) actually need
    this, but it is a few hundred bytes and it keeps the manifest self-describing.
    """
    db = pd.read_csv(os.path.join(datafolder, dataset, "BIDS",
                                  f"{specie}_database-details.csv"))
    out = {}
    for sub_N in participants:
        rows = db[db["sub_N"] == sub_N]
        out[str(int(sub_N))] = [{"session": int(r["session"]), "run_N": int(r["run_N"])}
                                for _, r in rows.iterrows()]
    return out


def build_group_package(specie, models, all_flag, all_stim_wise, dataset, model,
                        radius, dis_method, mah_fold, rsa_method, reps, reps_group,
                        mask_type, out_dir, participants=None,
                        min_percentage_available=1.0, verbose=True,
                        allow_space_mismatch=False, allow_off_mask=False):
    if specie not in ("D", "H"):
        raise ValueError("specie must be 'D' or 'H'")

    datafolder, _git, _py = get_paths()
    if radius is None:
        radius = 3 if specie == "D" else 4

    cfg_path = os.path.join(datafolder, dataset, "config_files", f"{specie}_{model}.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    task = cfg.get("task", dataset)
    if participants:
        participant_list = [int(p) for p in participants]
    else:
        participant_list = [int(p) for p in cfg["participants"]]

    model_list = resolve_models(datafolder, dataset, models, all_flag, all_stim_wise,
                                dis_method)

    mask_src = os.path.join(datafolder, dataset, "ROI", specie, f"{mask_type}.nii.gz")
    if not os.path.exists(mask_src):
        raise FileNotFoundError(f"Mask not found: {mask_src}")

    manifest = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "kind": "group",
        # recorded so the logs written on Colab carry workstation paths, exactly
        # like the ones rsa_utils writes (step 3 reads its own .json back)
        "datafolder": datafolder,
        "dataset": dataset, "model": model, "specie": specie,
        "task": task, "radius": radius, "mask_type": mask_type,
        "dis_method": dis_method, "mah_fold": mah_fold, "rsa_method": rsa_method,
        "reps": reps, "reps_group": reps_group,
        "min_percentage_available": min_percentage_available,
        "participants": participant_list,
        "runs_by_sub": runs_by_sub(datafolder, dataset, specie, participant_list),
        "models": model_list,
        # both checks also run on Colab; recording them here means a package that
        # builds does not then fail on the GPU for a policy reason
        "allow_space_mismatch": bool(allow_space_mismatch),
        "allow_off_mask": bool(allow_off_mask),
    }

    os.makedirs(out_dir, exist_ok=True)
    zip_name = f"pkg_group_{specie}_{dataset}_{model}_{dis_method}.zip"
    zip_path = os.path.join(out_dir, zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        zf.write(mask_src, arcname=f"data/{dataset}/ROI/{specie}/{mask_type}.nii.gz")
        zf.write(os.path.join(COLAB_DIR, "gpu_rsa.py"), arcname="code/gpu_rsa.py")
        zf.write(os.path.join(COLAB_DIR, "gpu_group.py"), arcname="code/gpu_group.py")
        zf.write(os.path.join(COLAB_DIR, "run_colab_group.py"),
                 arcname="code/run_colab_group.py")
        zf.write(os.path.join(COLAB_DIR, "colab_rsa_group.ipynb"),
                 arcname="colab_rsa_group.ipynb")
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("README.txt", _package_readme(manifest, zip_name))

    if verbose:
        size_kb = os.path.getsize(zip_path) / 1e3
        print(f"Group package: {zip_path}")
        print(f"  {specie}  {dataset}/{model}  {dis_method}  r-{radius}  {rsa_method}")
        print(f"  participants={len(participant_list)}  models={len(model_list)}  "
              f"reps={reps}  reps_group={reps_group}  size={size_kb:.1f} kB")
        print(f"  expects result_<model>_{specie}-sub-NN.zip for every participant "
              "in the results folder")
    return zip_path


def _package_readme(m, zip_name):
    return (
        f"Colab RSA group package: {zip_name}\n"
        f"{m['specie']}  {m['dataset']}/{m['model']}  dis_method={m['dis_method']}  "
        f"radius={m['radius']}  rsa_method={m['rsa_method']}\n"
        f"participants: {len(m['participants'])}   models: {len(m['models'])}   "
        f"reps: {m['reps']}   reps_group: {m['reps_group']}\n\n"
        "This package holds no imaging data except the mask. The group steps read\n"
        "the participant maps out of the step-1/2/4 result zips already on Drive.\n\n"
        "To run on Colab:\n"
        "  1. Upload this .zip to the Drive folder that holds your result_*.zip files.\n"
        "  2. Open colab_rsa_group.ipynb in Colab (GPU runtime; High-RAM for humans).\n"
        "  3. Set PKG_ZIP, RESULTS_DIR (where the result_*.zip live) and OUT_DIR.\n"
        "  4. Run all cells. One result_group_<model>_<specie>.zip per model appears\n"
        "     in OUT_DIR; re-running skips models already done.\n"
        "  5. Back on the workstation: tools/unpack_results.py <downloads> merges them,\n"
        "     then run pipeline steps 8-10 as usual.\n"
    )


def parse_args():
    ap = argparse.ArgumentParser(
        description="Build a Colab GPU package for RSA group steps 3/5/6/7.")
    ap.add_argument("specie", choices=["D", "H"], help="'D' (dog) or 'H' (human)")
    ap.add_argument("--models", nargs="*", default=[],
                    help="Explicit RSA model names (CSV stems)")
    ap.add_argument("--dis_method", default="mahalanobis",
                    choices=["mahalanobis", "correlation"])
    ap.add_argument("--all", dest="all_flag", action="store_true",
                    help="Add every model for --dis_method from _models.csv")
    ap.add_argument("--all-stim-wise", action="store_true",
                    help="Add every mahalanobis stim-wise model x grouping (alias)")
    ap.add_argument("--dataset", default="EmoC")
    ap.add_argument("--model", default="basic-block", help="GLM model")
    ap.add_argument("--radius", type=int, default=None,
                    help="Searchlight radius (default 3 D / 4 H)")
    ap.add_argument("--mah_fold", default="stim-wise")
    ap.add_argument("--rsa_method", default="kendall",
                    choices=["kendall", "pearson", "correlation"])
    ap.add_argument("--reps", type=int, default=100,
                    help="Per-participant permutations written by step 4")
    ap.add_argument("--reps_group", type=int, default=1000,
                    help="Group permutations for step 5")
    ap.add_argument("--mask_type", default="b_GreyMatter2mmB")
    ap.add_argument("--participants", type=int, nargs="*", default=None,
                    help="Override the config's participant list")
    ap.add_argument("--min_percentage_available", type=float, default=1.0)
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output folder for the .zip")
    ap.add_argument("--allow_space_mismatch", action="store_true",
                    help="Downgrade the voxel-grid check to a warning on Colab too")
    ap.add_argument("--allow_off_mask", action="store_true",
                    help="Warn instead of failing when a participant map has "
                         "non-zero voxels outside the mask")
    return ap.parse_args()


def main():
    a = parse_args()
    build_group_package(a.specie, a.models, a.all_flag, a.all_stim_wise, a.dataset,
                        a.model, a.radius, a.dis_method, a.mah_fold, a.rsa_method,
                        a.reps, a.reps_group, a.mask_type, a.out,
                        participants=a.participants,
                        min_percentage_available=a.min_percentage_available,
                        allow_space_mismatch=a.allow_space_mismatch,
                        allow_off_mask=a.allow_off_mask)


if __name__ == "__main__":
    main()

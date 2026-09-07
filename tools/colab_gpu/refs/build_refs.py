#!/usr/bin/env python
"""build_refs.py -- snapshot the few dataset facts a package-free Colab run needs.

Almost everything the group steps need is written into the arcnames inside the
result zips (see ``gpu_group.discover_manifest``). Three things are not, because
they live in the dataset config on the network disk:

  * the **participant list** -- the denominator of the availability check. Without
    it, "32 participants have maps" cannot be turned into a percentage, because a
    participant who has produced nothing leaves no trace on Drive.
  * ``runs_by_sub`` -- only the per-run layouts need it, and even then the run
    folders are discoverable; kept because it is a few hundred bytes and makes a
    missing run visible rather than invisible.
  * ``task``, for datasets where it differs from the dataset name.

This writes them to ``refs/{dataset}_refs.json``, a few kB that can be committed
and carried to Colab (or pasted into the notebook), so a group run never has to
touch ``P:``. Re-run it whenever the config's participant list changes.

    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\colab_gpu\\refs\\build_refs.py
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\colab_gpu\\refs\\build_refs.py --dataset EmoC --species H D
"""

import argparse
import datetime
import json
import os
import sys

import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from scheduler.paths import get_paths  # noqa: E402


def runs_by_sub(datafolder, dataset, specie, participants):
    """``{sub_N: [{'session','run_N'}, ...]}`` -- mirrors get_session_and_run_dict."""
    path = os.path.join(datafolder, dataset, "BIDS", f"{specie}_database-details.csv")
    if not os.path.exists(path):
        print(f"  WARNING: no database details at {path}; runs_by_sub left empty")
        return {}
    db = pd.read_csv(path)
    out = {}
    for sub_N in participants:
        rows = db[db["sub_N"] == sub_N]
        out[str(int(sub_N))] = [{"session": int(r["session"]), "run_N": int(r["run_N"])}
                                for _, r in rows.iterrows()]
    return out


def build(dataset, species, model, out_dir, verbose=True):
    datafolder, _git, _py = get_paths()
    if not os.path.isdir(datafolder):
        raise FileNotFoundError(
            f"Data folder not reachable: {datafolder}. This script is the one thing "
            "that still needs the network disk -- run it while the share is up.")
    refs = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "datafolder": datafolder,     # only used to render workstation paths in logs
        "model": model,
        "species": {},
    }
    for specie in species:
        cfg_path = os.path.join(datafolder, dataset, "config_files",
                                f"{specie}_{model}.yaml")
        if not os.path.exists(cfg_path):
            print(f"  skipping {specie}: no config at {cfg_path}")
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        participants = [int(p) for p in cfg["participants"]]
        refs["species"][specie] = {
            "task": cfg.get("task", dataset),
            "participants": participants,
            "runs_by_sub": runs_by_sub(datafolder, dataset, specie, participants),
        }
        if verbose:
            print(f"  {specie}: {len(participants)} participant(s), "
                  f"task={refs['species'][specie]['task']}")

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{dataset}_refs.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(refs, f, indent=2)
    print(f"wrote {path} ({os.path.getsize(path) / 1e3:.1f} kB)")
    return path


def parse_args():
    ap = argparse.ArgumentParser(
        description="Snapshot the dataset facts a package-free Colab group run needs.")
    ap.add_argument("--dataset", default="EmoC")
    ap.add_argument("--species", nargs="+", default=["H", "D"])
    ap.add_argument("--model", default="basic-block", help="GLM model")
    ap.add_argument("--out", default=HERE)
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    build(a.dataset, a.species, a.model, a.out)

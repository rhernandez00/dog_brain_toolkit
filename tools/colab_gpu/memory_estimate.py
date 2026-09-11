#!/usr/bin/env python
"""memory_estimate.py -- what a group run will actually allocate, and where.

Answers "which Colab runtime do I need?" from the numbers that decide it: how
many units a model has, how many permutations each, and how many mask voxels.
Every figure below is read off a specific allocation in ``gpu_group.py`` rather
than guessed, so if one of those changes this file has to change with it.

    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\colab_gpu\\memory_estimate.py
    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\colab_gpu\\memory_estimate.py \\
        --results "G:\\My Drive\\rsa_colab\\results" --specie H

With ``--results`` it probes the real folder and reports per battery. Without it,
it uses the EmoC human numbers as a worked example.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

F8 = 8  # bytes per float64; the whole pipeline is float64 to match the CPU
# set by main() before host_peak runs; the CUDA chunk copy scales with it
vox_batch_global = [20000]


def host_peak(n_units, reps, n_vox, reps_group, want_group_means, want_z,
              workers=8):
    """Host (system) RAM, in bytes, broken down by allocation.

    Since v3.2.0 ``load_participant_maps`` preallocates ``rnd_flat`` and writes
    each map straight into its row, so the loading peak is one copy of the data
    plus whatever the reader threads hold transiently (one unit each). Before
    that it collected per-unit lists and ``np.stack``ed them, which held the same
    data twice and doubled the peak.
    """
    T = n_units * reps                       # permutation maps held at once
    vec = n_vox * F8                         # one map, restricted to the mask

    # Since v4.0.0 steps 5/6 STREAM: one map is read, added into the rows that
    # drew it, and dropped. Nothing holds T maps any more -- the accumulator is
    # (reps_group, V) regardless of how many units the model has.
    rnd_flat = 0
    in_flight = workers * vec                # one map per reader thread
    acc = reps_group * n_vox * F8            # the streaming accumulator
    real = n_units * vec                     # step 3's real maps
    gmeans = reps_group * n_vox * F8 if (want_group_means or want_z) else 0
    z_rnd = reps_group * n_vox * F8 if want_z else 0
    # group_permutation_stats slices rnd_flat by COLUMN (rnd_flat[:, v0:v1]).
    # That is a non-contiguous view: measured on CPU, torch.as_tensor shares it
    # with no copy, but a CUDA transfer cannot DMA a strided buffer, so torch
    # materialises a contiguous one first. Counted here because it scales with
    # vox_batch and lands on the host, which is the side that runs out.
    chunk = T * min(vox_batch_global[0], n_vox) * F8 if vox_batch_global[0] else 0

    load_peak = acc + in_flight + real
    compute_peak = acc + real + gmeans + z_rnd
    return {
        "streaming accumulator (G x V)": acc,
        f"in flight ({workers} reader threads)": in_flight,
        "real maps": real,
        "group means (G x V)": gmeans,
        "rnd z maps (G x V)": z_rnd,
        "PEAK while loading": load_peak,
        "PEAK while computing": compute_peak,
        "PEAK overall": max(load_peak, compute_peak),
    }



def device_peak(n_units, reps, n_vox, reps_group, vox_batch, g_batch):
    """GPU RAM, in bytes, for the streaming accumulator in v4.0.0+.

    ``vox_batch``/``g_batch`` no longer size anything here: streaming holds the
    whole accumulator resident and adds one map at a time, so the footprint is
    flat in both the number of units and the chunk settings. The second entry is
    the ``acc - mu`` temporary in the std pass, which is the real peak.
    """
    acc = reps_group * n_vox * F8            # the running accumulator
    centred = reps_group * n_vox * F8        # acc - mu, in the std reduction
    per_map = max(1, reps_group // max(1, reps)) * n_vox * F8  # rows.repeat(...)
    return {
        "accumulator (G x V)": acc,
        "acc - mu temporary (G x V)": centred,
        "one map, repeated per drawing row": per_map,
        "PEAK on device": acc + centred + per_map,
    }


def gb(n):
    return n / 1e9


def report(label, n_units, reps, n_vox, reps_group, vox_batch, g_batch,
           want_group_means=False, want_z=True):
    T = n_units * reps
    print(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    print(f"  {n_units} unit(s) x {reps} perms = {T} maps, "
          f"{n_vox} mask voxels, reps_group={reps_group}")

    vox_batch_global[0] = vox_batch
    h = host_peak(n_units, reps, n_vox, reps_group, want_group_means, want_z)
    print("\n  host RAM")
    for k, v in h.items():
        if k.startswith("PEAK"):
            print(f"    {'-' * 56}")
        print(f"    {k:<40s} {gb(v):8.2f} GB")

    d = device_peak(n_units, reps, n_vox, reps_group, vox_batch, g_batch)
    print("\n  GPU RAM (one voxel chunk)")
    for k, v in d.items():
        if k.startswith("PEAK"):
            print(f"    {'-' * 56}")
        print(f"    {k:<40s} {gb(v):8.2f} GB")
    return h["PEAK overall"], d["PEAK on device"]


# Colab runtimes, approximate and subject to change -- treat as a shortlist to
# check in the runtime picker, not as a spec sheet.
RUNTIMES = [
    ("CPU, standard RAM",       12.7,  0.0, "free"),
    ("T4 GPU, standard RAM",    12.7, 15.0, "free"),
    ("T4 GPU, High-RAM",        51.0, 15.0, "Pro"),
    ("L4 GPU, High-RAM",        53.0, 22.5, "Pro"),
    ("A100 40GB, High-RAM",     83.0, 40.0, "Pro+"),
]


def recommend(host_bytes, dev_bytes, label):
    print(f"\n  runtimes that fit {label} "
          f"(needs {gb(host_bytes):.1f} GB host, {gb(dev_bytes):.1f} GB GPU):")
    any_fit = False
    for name, ram, vram, tier in RUNTIMES:
        # leave 15% headroom: the interpreter, torch, nibabel and Drive caching
        ok_h = gb(host_bytes) <= ram * 0.85
        ok_d = vram == 0 or gb(dev_bytes) <= vram * 0.85
        fits = ok_h and ok_d
        any_fit |= fits
        why = "" if fits else ("  <- host RAM too small" if not ok_h
                               else "  <- GPU RAM too small")
        print(f"    [{'YES' if fits else ' no'}] {name:<24s} "
              f"{ram:5.1f} GB RAM / {vram:4.1f} GB VRAM  ({tier}){why}")
    if not any_fit:
        print("    NOTHING FITS -- the run has to allocate less, not rent more.")
    return any_fit


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default=None,
                    help="Probe a real results folder instead of the example")
    ap.add_argument("--refs", default=os.path.join(HERE, "refs"))
    ap.add_argument("--specie", default="H", choices=["H", "D"])
    ap.add_argument("--reps_group", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=20000, help="vox_batch")
    ap.add_argument("--g_batch", type=int, default=64)
    ap.add_argument("--write_group_means", action="store_true")
    return ap.parse_args()


def main():
    a = parse_args()
    cases = []
    if a.results:
        import gpu_group
        import numpy as np
        per_model = gpu_group.discover_model_manifests(
            a.results, a.refs, specie=a.specie, reps_group=a.reps_group,
            verbose=True)
        _img, mask_bool = gpu_group.load_group_mask(
            next(iter(per_model.values())), refs_dir=a.refs)
        n_vox = int(mask_bool.sum())
        groups = {}
        for name, m in per_model.items():
            groups.setdefault(gpu_group.manifest_signature(m), []).append(name)
        for sig, names in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            m = per_model[names[0]]
            lbl = (f"{len(names)} model(s): {sig[2]}"
                   + (f"/{sig[3]}" if sig[2] == "mahalanobis" else "")
                   + f"/{sig[4]}  " + ("per-run" if sig[7] else "per-participant"))
            cases.append((lbl, len(gpu_group.units(m)), m["reps"], n_vox))
    else:
        n_vox = 159254                       # EmoC H, b_GreyMatter2mmB
        cases = [
            ("EmoC H mahalanobis (per-participant)", 40, 100, n_vox),
            ("EmoC H correlation (per-run)", 239, 100, n_vox),
        ]

    for label, n_units, reps, nv in cases:
        h, d = report(label, n_units, reps, nv, a.reps_group, a.batch,
                      a.g_batch, want_group_means=a.write_group_means)
        recommend(h, d, label)


if __name__ == "__main__":
    main()

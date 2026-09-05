#!/usr/bin/env python
"""validate_step5.py -- correctness harness for ``gpu_step5.py``.

Builds a synthetic results folder of ``result_{model}_{specie}-sub-NN.zip``,
runs step 5 over it, and compares every output map against the reduction the CPU
performs -- a plain **full-volume** voxelwise mean over the same drawn files,
which is what ``nifti_mean(files_list, result_map_path=...)`` does when step 5
calls it without a mask.

That comparison is the point of the harness. ``gpu_step5`` never materialises a
full volume while accumulating: it works on the union of the inputs' support and
scatters back at the end. The two agree exactly only if the support logic is
right, which is why the synthetic maps deliberately contain voxels that are
**exactly 0.0 inside the mask** -- a Kendall tau really does land on zero, and on
real EmoC data ~1000 voxels differ in support between two maps of the same
participant. A single map's support is therefore not the mask.

Also covered: the per-run layout and its ``mah_fold`` sub-folder, ragged
permutation counts across participants, a stem with no ``{mask_type}-`` prefix,
arcname parsing (including the members step 5 must ignore), the availability
gate, and that the result zip merges through ``tools/unpack_results.py``.

Exits non-zero if anything diverges.

    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\colab_gpu\\validate_step5.py

Needs ``KMP_DUPLICATE_LIB_OK=TRUE`` on this machine, like the other harnesses
(Anaconda and torch each ship an OpenMP runtime).
"""

import gzip
import io
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
import zlib

import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import gpu_step5 as g5  # noqa: E402

SHAPE = (7, 8, 9)
NVOX = int(np.prod(SHAPE))
AFFINE = np.diag([-2.0, 2.0, 2.0, 1.0])
AFFINE[:3, 3] = [6.0, -8.0, -9.0]

FAILURES = []


def report(ok, label, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)
    return ok


def _seed(*parts):
    return zlib.crc32("|".join(str(p) for p in parts).encode())


# the synthetic searchlight mask: the voxels that ever carry a value
MASK = np.random.default_rng(0).random(NVOX) < 0.4


def make_map(*key, zero_frac=0.05):
    """A map supported on MASK, with some voxels landing exactly on 0.0."""
    rng = np.random.default_rng(_seed(*key))
    flat = np.zeros(NVOX, dtype=np.float64)
    vals = rng.normal(size=int(MASK.sum()))
    vals[rng.random(vals.size) < zero_frac] = 0.0     # exact-zero taus
    flat[MASK] = vals
    return flat


def write_zip(path, members):
    """Write {arcname: flat volume} as gzip-compressed niftis inside one zip."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, flat in members.items():
            img = nib.Nifti1Image(flat.reshape(SHAPE), AFFINE)
            buf = io.BytesIO()
            gz = gzip.GzipFile(fileobj=buf, mode="wb")
            file_map = img.make_file_map()
            file_map["image"].fileobj = gz
            img.to_file_map(file_map)
            gz.close()
            zf.writestr(arc, buf.getvalue())


def build_results(root, subs, reps_by_sub, per_run=False,
                  mask_type="b_GreyMatter2mmB", rsa_model="mymodel"):
    """Write one results folder. Returns (folder, {arcname: flat}) for the reference."""
    results = os.path.join(root, "results")
    os.makedirs(results, exist_ok=True)
    truth = {}
    stem = f"{mask_type + '-' if mask_type else ''}r-4_mahalanobis_kendall"
    for sub in subs:
        members = {}
        runs = [(1, 1), (1, 2)] if per_run else [(None, None)]
        for ses, run in runs:
            parts = ["EmoC", "results", "RSA_rnd", "basic-block", rsa_model]
            if per_run:
                parts.append("stim-wise-all-runs")   # fold-isolated participant root
            parts.append(f"H-sub-{sub:02d}")
            if per_run:
                parts.append(f"ses-{ses:02d}_task-EmoC_run-{run:02d}")
            base = "/".join(parts)
            for i in range(reps_by_sub[sub]):
                arc = f"{base}/{stem}_{i:04d}.nii.gz"
                members[arc] = truth[arc] = make_map(rsa_model, sub, ses, run, i)
            real = list(parts)
            real[2] = "RSA"                           # step 5 must ignore the real map
            members["/".join(real) + f"/{stem}.nii.gz"] = make_map(rsa_model, sub, "real")
        write_zip(os.path.join(results, f"result_{rsa_model}_H-sub-{sub:02d}.zip"),
                  members)
    return results, truth


def full_volume_reference(spec, cols, orders, truth):
    """The CPU reduction: a plain mean of full volumes, no mask, no support."""
    out = np.zeros((cols.shape[0], NVOX), dtype=np.float64)
    for u, unit in enumerate(spec.units):
        for g in range(cols.shape[0]):
            out[g] += truth[spec.refs[unit][orders[u][cols[g, u]]][1]]
    return out / len(spec.units)


def case(label, subs, reps_by_sub, reps_group, per_run=False,
         mask_type="b_GreyMatter2mmB"):
    tmp = tempfile.mkdtemp(prefix="val_step5_")
    try:
        results, truth = build_results(tmp, subs, reps_by_sub, per_run, mask_type)
        specs = g5.scan_results(results, specie="H", verbose=False)
        spec = specs[("mymodel", "H")]

        report(spec.params["mask_type"] == mask_type, f"{label}: mask_type parsed",
               repr(spec.params["mask_type"]))
        report(spec.params["mah_fold"] == ("stim-wise-all-runs" if per_run else None),
               f"{label}: mah_fold parsed", repr(spec.params["mah_fold"]))
        expected_units = len(list(subs)) * (2 if per_run else 1)
        report(len(spec.units) == expected_units, f"{label}: units discovered",
               f"{len(spec.units)} == {expected_units}")

        zip_path, n_files = g5.run_step5(
            spec, os.path.join(tmp, "out"), reps_group=reps_group,
            work_root=os.path.join(tmp, "work"), device=g5.pick_device(False),
            expected_participants="found", verbose=False)
        report(n_files == reps_group, f"{label}: one map per group permutation",
               f"{n_files} == {reps_group}")

        cols = g5.draw_group_indices(
            spec.counts, reps_group, g5.default_seed("mymodel", "H", reps_group))
        orders = [sorted(spec.refs[u]) for u in spec.units]
        ref = full_volume_reference(spec, cols, orders, truth)

        worst, name_ok, dtype_ok, affine_ok = 0.0, True, True, True
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
            for g in range(reps_group):
                arc = spec.group_mean_rel(g)
                if arc not in names:
                    name_ok = False
                    continue
                img = g5._img_from_bytes(zf.read(arc))
                dtype_ok &= img.get_data_dtype() == np.dtype("<f8")
                affine_ok &= bool(np.allclose(img.affine, AFFINE))
                got = np.asarray(img.dataobj, dtype=np.float64).reshape(-1)
                worst = max(worst, float(np.abs(got - ref[g]).max()))
        report(name_ok, f"{label}: output arcnames match the pipeline convention")
        report(dtype_ok, f"{label}: float64, as nifti_mean writes")
        report(affine_ok, f"{label}: affine preserved")
        report(worst == 0.0, f"{label}: GPU == full-volume CPU mean",
               f"max abs diff {worst:.3e}")
        return zip_path if worst == 0.0 else None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_read_threads_are_deterministic():
    """Prefetching reads must not change a single bit of the result.

    Maps are read on a thread pool but applied in job order, so the float64 sums
    accumulate in the same sequence however the reads finish. This is the check
    that keeps that true: same data, ``read_workers`` 1 vs 16, byte-comparable
    output.
    """
    print("\nthreaded reads vs serial reads")
    tmp = tempfile.mkdtemp(prefix="val_step5_thr_")
    try:
        results, _ = build_results(tmp, range(1, 7), {s: 9 for s in range(1, 7)})
        spec = g5.scan_results(results, specie="H", verbose=False)[("mymodel", "H")]
        outs = {}
        for workers in (1, 16):
            zip_path, _ = g5.run_step5(
                spec, os.path.join(tmp, f"out{workers}"), reps_group=50,
                work_root=os.path.join(tmp, f"work{workers}"),
                device=g5.pick_device(False), expected_participants="found",
                read_workers=workers, verbose=False)
            with zipfile.ZipFile(zip_path) as zf:
                outs[workers] = {n: zf.read(n) for n in sorted(zf.namelist())}
        same_names = sorted(outs[1]) == sorted(outs[16])
        same_bytes = same_names and all(outs[1][n] == outs[16][n] for n in outs[1])
        report(same_bytes, "read_workers=1 and read_workers=16 agree byte for byte")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_parsing():
    print("\narcname parsing")
    good = {
        "EmoC/results/RSA_rnd/basic-block/m/H-sub-01/"
        "b_GM-r-4_mahalanobis_kendall_0007.nii.gz":
            dict(rnd=True, rnd_index=7, mask_type="b_GM", mah_fold=None, radius=4,
                 specie="H", sub_N=1),
        "EmoC/results/RSA/basic-block/m/H-sub-01/"
        "b_GM-r-4_mahalanobis_kendall.nii.gz":
            dict(rnd=False, rnd_index=None),
        "EmoC/results/RSA_rnd/basic-block/m/stim-wise-all-runs/D-sub-03/"
        "ses-02_task-EmoC_run-05/r-3_correlation_pearson_0012.nii.gz":
            dict(rnd=True, rnd_index=12, mask_type=None, radius=3, session=2,
                 run_N=5, task="EmoC", specie="D", mah_fold="stim-wise-all-runs"),
    }
    for rel, expect in good.items():
        got = g5.parse_arcname(rel)
        ok = got is not None and all(got[k] == v for k, v in expect.items())
        report(ok, f"parses {rel.rsplit('/', 1)[1]}")
    ignore = [
        # the group outputs, which live under the same model root
        "EmoC/results/RSA_rnd/basic-block/m/mean/"
        "H-r-4_mahalanobis_kendall_mean_00000.nii.gz",
        "EmoC/results/RSA_rnd/basic-block/m/mean/H-r-4_mahalanobis_kendall_z_log.txt",
        "EmoC/results/RSA_rnd/basic-block/H-mymodel_mean.nii.gz",
        "EmoC/ROI/H/b_GreyMatter2mmB.nii.gz",
    ]
    for rel in ignore:
        report(g5.parse_arcname(rel) is None, f"ignores {rel.rsplit('/', 1)[1]}")


def check_gate():
    """The availability gate, with the 'auto' denominator."""
    print("\navailability gate")
    tmp = tempfile.mkdtemp(prefix="val_step5_gate_")
    try:
        # 4 of 10 participants have this model; the other 6 exist under another one
        results, _ = build_results(tmp, range(1, 5), {s: 5 for s in range(1, 5)})
        for sub in range(5, 11):
            write_zip(
                os.path.join(results, f"result_othermodel_H-sub-{sub:02d}.zip"),
                {f"EmoC/results/RSA_rnd/basic-block/othermodel/H-sub-{sub:02d}/"
                 f"b_GreyMatter2mmB-r-4_mahalanobis_kendall_0000.nii.gz":
                 make_map("othermodel", sub)})
        spec = g5.scan_results(results, specie="H", models=["mymodel"],
                               verbose=False)[("mymodel", "H")]
        n, _how = g5.resolve_expected_participants(spec, "auto")
        report(n == 10, "'auto' counts every sub-NN in the folder", f"{n} == 10")
        report(g5.resolve_expected_participants(spec, "found")[0] == 4,
               "'found' counts only the participants of this model")
        report(g5.resolve_expected_participants(spec, 12)[0] == 12,
               "an explicit denominator is used verbatim")

        for gate, should_run in [(0.4, True), (0.5, False), (1.0, False)]:
            out = os.path.join(tmp, f"out{gate}")
            try:
                g5.run_step5(spec, out, reps_group=5,
                             work_root=os.path.join(tmp, "work"),
                             device=g5.pick_device(False),
                             min_percentage_available=gate,
                             expected_participants="auto", verbose=False)
                ran = True
            except g5.MissingMapsError:
                ran = False
            report(ran == should_run,
                   f"gate {gate:.2f} at 40% availability -> "
                   f"{'runs' if ran else 'refuses'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_unpack(zip_path):
    """The result zip must merge onto a data disk via tools/unpack_results.py."""
    print("\nmerge back with unpack_results.py")
    if zip_path is None or not os.path.exists(zip_path):
        return report(False, "a result zip was produced to merge")
    tmp = tempfile.mkdtemp(prefix="val_step5_unpack_")
    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(REPO, "tools", "unpack_results.py"),
             zip_path, "--datafolder", tmp, "--dataset", "EmoC"],
            capture_output=True, text=True)
        landed = os.path.join(
            tmp, "EmoC", "results", "RSA_rnd", "basic-block", "mymodel", "mean",
            "H-r-4_mahalanobis_kendall_mean_00000.nii.gz")
        ok = proc.returncode == 0 and os.path.exists(landed)
        report(ok, "unpack_results.py merges the zip onto the data disk",
               "" if ok else (proc.stdout + proc.stderr)[-400:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print(f"device: {g5.pick_device()} (the harness runs on CPU torch for "
          f"determinism)\nmask voxels: {int(MASK.sum())}/{NVOX}\n")
    print("step 5 vs the CPU reduction")
    zip_path = case("stim-wise, uniform reps", range(1, 9),
                    {s: 20 for s in range(1, 9)}, 40)
    case("stim-wise, ragged reps", range(1, 7),
         {1: 20, 2: 3, 3: 11, 4: 1, 5: 20, 6: 7}, 200)
    case("per-run layout", range(1, 5), {s: 12 for s in range(1, 5)}, 60,
         per_run=True)
    case("no mask_type prefix", range(1, 5), {s: 10 for s in range(1, 5)}, 30,
         mask_type=None)
    check_read_threads_are_deterministic()
    check_parsing()
    check_gate()

    # rebuild one zip outside the temp teardown so unpack has something to merge
    tmp = tempfile.mkdtemp(prefix="val_step5_keep_")
    try:
        results, _ = build_results(tmp, range(1, 4), {s: 6 for s in range(1, 4)})
        spec = g5.scan_results(results, specie="H", verbose=False)[("mymodel", "H")]
        zip_path, _ = g5.run_step5(spec, os.path.join(tmp, "out"), reps_group=3,
                                   work_root=os.path.join(tmp, "work"),
                                   device=g5.pick_device(False),
                                   expected_participants="found", verbose=False)
        check_unpack(zip_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s)")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

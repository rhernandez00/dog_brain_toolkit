#!/usr/bin/env python
"""validate_group.py -- prove the GPU group steps match the CPU pipeline.

Run on the workstation (Anaconda Python, torch CPU is fine):

    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\colab_gpu\\validate_group.py

Unlike ``validate_gpu.py``, which can lean on real maps already sitting on the
data disk, the group steps need a whole synthetic *dataset* -- participants,
permutation maps, a mask, a config -- so this script builds one in a temp folder
(small volume, few participants, few permutations), runs both paths over it, and
compares:

  1. step 3   -- ``calculate_group_model_similarity_map``      (mean, std)
  2. step 5   -- the group-permutation mean, against an explicit ``nifti_mean``
                 over the files the same seeded draw selects
  3. step 6   -- ``calculate_voxelwise_rnd_distribution``       (mean, std)
  4. step 7   -- ``calculate_z_maps_rnd`` + ``calculate_z_map_real_data``

Steps 6 and 7 are fed the *GPU's own* step-5 maps on the CPU side, so the
comparison is exact rather than a comparison of two independent random draws --
which is the only part of step 5 that cannot be bit-matched (the CPU picks its
permutation with an unseeded ``random.choice``).

Exits non-zero if any check exceeds tolerance.
"""

import json
import os
import re
import shutil
import sys
import tempfile
import zipfile

import numpy as np
import nibabel as nib
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import gpu_group      # noqa: E402
import run_colab_group  # noqa: E402

FAIL = []
DEV = torch.device("cpu")

DATASET = "TestSet"
GLM_MODEL = "basic-block"
SPECIE = "D"
MASK_TYPE = "testmask"
RSA_MODEL = "toy-model"
RADIUS = 3
DIS_METHOD = "mahalanobis"
RSA_METHOD = "kendall"
PARTICIPANTS = [1, 3, 7, 9]
REPS = 8
REPS_GROUP = 12
SHAPE = (7, 8, 9)
# Low on purpose: the synthetic z maps are roughly standard normal on a 7x8x9
# grid, so the real 3.1+ thresholds would find nothing and the comparison would
# be vacuously equal. These produce actual clusters to disagree about.
Z_THRESHOLDS = (0.5, 1.0, 1.5)


def report(name, err, tol):
    ok = err <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name:52s} max|err|={err:.3e}  (tol {tol:.0e})")
    if not ok:
        FAIL.append(name)


def report_bool(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name:52s} {detail}")
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------------------
# synthetic dataset
# ---------------------------------------------------------------------------
def build_dataset(root):
    """Write a mask, a config and one real + REPS permutation maps per participant.

    Maps are zero outside the mask, exactly like the ones the pipeline writes.
    """
    rng = np.random.default_rng(12345)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = [-7.0, -8.0, -9.0]

    mask = np.zeros(SHAPE, dtype=np.float64)
    mask[1:6, 1:7, 1:8] = 1.0
    mask[3, 3, 3] = 0.0                       # a hole, to catch mask/index slips
    mask_bool = mask > 0
    mask_dir = os.path.join(root, DATASET, "ROI", SPECIE)
    os.makedirs(mask_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(mask, affine),
             os.path.join(mask_dir, f"{MASK_TYPE}.nii.gz"))

    cfg_dir = os.path.join(root, DATASET, "config_files")
    os.makedirs(cfg_dir, exist_ok=True)
    with open(os.path.join(cfg_dir, f"{SPECIE}_{GLM_MODEL}.yaml"), "w") as f:
        json.dump({"task": DATASET, "participants": PARTICIPANTS,
                   "stim_types": ["a1", "a2"]}, f)

    stem = f"{MASK_TYPE}-r-{RADIUS}_{DIS_METHOD}_{RSA_METHOD}"
    n_mask = int(mask_bool.sum())
    for sub_N in PARTICIPANTS:
        sub = f"{SPECIE}-sub-{sub_N:02d}"
        real_dir = os.path.join(root, DATASET, "results", "RSA", GLM_MODEL,
                                RSA_MODEL, sub)
        rnd_dir = os.path.join(root, DATASET, "results", "RSA_rnd", GLM_MODEL,
                               RSA_MODEL, sub)
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(rnd_dir, exist_ok=True)
        for name, path in ([(None, os.path.join(real_dir, f"{stem}.nii.gz"))] +
                           [(i, os.path.join(rnd_dir, f"{stem}_{i:04d}.nii.gz"))
                            for i in range(REPS)]):
            vol = np.zeros(SHAPE, dtype=np.float64)
            vol[mask_bool] = rng.normal(0.02 if name is None else 0.0, 0.3, n_mask)
            nib.save(nib.Nifti1Image(vol, affine), path)
    return mask_bool, affine


def build_result_zips(data_root, zip_dir):
    """Repackage the synthetic dataset as step-2/4 result zips.

    Same naming and arcname convention as ``gpu_rsa.zip_model_result``, so this
    exercises the path an actual Colab run takes: participant maps read out of
    the zips in OUT_DIR, never unpacked.
    """
    import glob
    import zipfile
    os.makedirs(zip_dir, exist_ok=True)
    for sub_N in PARTICIPANTS:
        sub = f"{SPECIE}-sub-{sub_N:02d}"
        paths = []
        for kind in ("RSA", "RSA_rnd"):
            paths += sorted(glob.glob(os.path.join(
                data_root, DATASET, "results", kind, GLM_MODEL, RSA_MODEL, sub,
                "*.nii.gz")))
        name = f"result_{RSA_MODEL}_{sub}.zip"
        with zipfile.ZipFile(os.path.join(zip_dir, name), "w",
                             zipfile.ZIP_STORED) as zf:
            for p in paths:
                zf.write(p, arcname=os.path.relpath(p, data_root).replace(os.sep, "/"))
    return zip_dir


def build_package(pkg_root, data_root):
    """A group package pointing at the synthetic dataset (mask + manifest only)."""
    os.makedirs(pkg_root, exist_ok=True)
    shutil.copytree(os.path.join(data_root, DATASET, "ROI"),
                    os.path.join(pkg_root, "data", DATASET, "ROI"))
    manifest = {
        "kind": "group", "datafolder": data_root,
        "dataset": DATASET, "model": GLM_MODEL, "specie": SPECIE, "task": DATASET,
        "radius": RADIUS, "mask_type": MASK_TYPE, "dis_method": DIS_METHOD,
        "mah_fold": "stim-wise", "rsa_method": RSA_METHOD,
        "reps": REPS, "reps_group": REPS_GROUP, "min_percentage_available": 1.0,
        "participants": PARTICIPANTS,
        "runs_by_sub": {str(p): [{"session": 1, "run_N": 1}] for p in PARTICIPANTS},
        "models": [RSA_MODEL], "allow_space_mismatch": False, "allow_off_mask": False,
    }
    with open(os.path.join(pkg_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def load(path):
    return np.asarray(nib.load(path).dataobj, dtype=np.float64)


def check_step3(manifest, gpu_root, data_root):
    import rsa_utils
    session_and_run_all = {p: [{"session": 1, "run_N": 1}] for p in PARTICIPANTS}
    cpu_root = os.path.join(data_root, "_cpu")
    # rsa_utils writes next to the inputs, so run it against a copy of the tree
    shutil.copytree(os.path.join(data_root, DATASET),
                    os.path.join(cpu_root, DATASET), dirs_exist_ok=True)
    rsa_utils.calculate_group_model_similarity_map(
        cpu_root, DATASET, session_and_run_all, SPECIE, GLM_MODEL, DATASET, RADIUS,
        rsa_model=RSA_MODEL, rsa_method=RSA_METHOD, dis_method=DIS_METHOD,
        replace_file=True, verbose=False, min_percentage_available=1.0,
        mask_type=MASK_TYPE, mah_fold="stim-wise")

    worst = 0.0
    for kind in ("mean", "std"):
        rel = gpu_group.group_real_rel(manifest, RSA_MODEL, kind)
        a = load(os.path.join(gpu_root, rel.replace("/", os.sep)))
        b = load(os.path.join(cpu_root, rel.replace("/", os.sep)))
        worst = max(worst, np.max(np.abs(a - b)))
    report("step 3 group mean/std vs rsa_utils", worst, 1e-12)
    return cpu_root


def check_zip_source(pkg_root, zip_dir, tree_root, tmp):
    """The zip-backed read must produce byte-for-byte what the tree-backed one did.

    This is the path a real run takes -- the participant maps stay inside the
    ``result_*.zip`` files on Drive and are never unpacked.
    """
    out_dir = os.path.join(tmp, "out_zip")
    work_root = os.path.join(tmp, "work_zip")
    # both intermediates on, to match the reference run this compares against
    run_colab_group.run_group_package(
        pkg_root, zip_dir, out_dir, work_root=work_root, steps=(3, 5, 6, 7),
        device=DEV, batch=64, g_batch=5, workers=4, write_group_means=True,
        write_z_maps=True, keep_work=True, verbose=False)
    zip_gpu_root = os.path.join(work_root, "data")

    worst, checked = 0.0, 0
    for dirpath, _dirs, files in os.walk(tree_root):
        for name in files:
            if not name.endswith(".nii.gz"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), tree_root)
            other = os.path.join(zip_gpu_root, rel)
            if not os.path.exists(other):
                report_bool("zip source produces the same file set", False,
                            f"missing {rel}")
                return
            a, b = load(os.path.join(tree_root, rel)), load(other)
            both = np.isfinite(a) & np.isfinite(b)
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                report_bool("zip source NaN pattern matches", False, rel)
                return
            worst = max(worst, float(np.max(np.abs(a[both] - b[both]))))
            checked += 1
    report(f"zip source == tree source ({checked} maps)", worst, 0.0)


def check_step5(manifest, gpu_root, data_root):
    """The group mean of permutation g must equal an explicit mean of the files
    the same seeded draw picks -- i.e. the gather+mean kernel really is
    ``nifti_mean`` over one map per participant."""
    import zlib

    import rsa_utils
    # reproduce the seed run_group_model derives when none is passed
    seed = zlib.crc32(f"group-{RSA_MODEL}-{SPECIE}-{REPS_GROUP}".encode())
    counts = np.array([REPS] * len(PARTICIPANTS), dtype=np.int64)
    cols = gpu_group.draw_group_indices(counts, REPS_GROUP, seed)

    stem = f"{MASK_TYPE}-r-{RADIUS}_{DIS_METHOD}_{RSA_METHOD}"
    worst = 0.0
    for g in (0, REPS_GROUP // 2, REPS_GROUP - 1):
        files = []
        for u, sub_N in enumerate(PARTICIPANTS):
            rep = int(cols[g, u] - u * REPS)
            files.append(os.path.join(
                data_root, DATASET, "results", "RSA_rnd", GLM_MODEL, RSA_MODEL,
                f"{SPECIE}-sub-{sub_N:02d}", f"{stem}_{rep:04d}.nii.gz"))
        expected, _ = rsa_utils.nifti_mean(files)
        got = load(os.path.join(
            gpu_root,
            gpu_group.group_rnd_rel(manifest, RSA_MODEL, "mean", g).replace("/", os.sep)))
        worst = max(worst, np.max(np.abs(expected - got)))
    report("step 5 group perm mean vs nifti_mean", worst, 1e-12)


def check_steps_6_7(manifest, gpu_root, cpu_root):
    """Feed the CPU functions the GPU's own step-5 maps, then compare 6 and 7."""
    import rsa_utils
    # copy the GPU permutation maps into the CPU tree so both start from the same
    # null sample (the CPU's own draw is unseeded and cannot be reproduced)
    for g in range(REPS_GROUP):
        rel = gpu_group.group_rnd_rel(manifest, RSA_MODEL, "mean", g)
        dst = os.path.join(cpu_root, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(gpu_root, rel.replace("/", os.sep)), dst)

    rsa_utils.calculate_voxelwise_rnd_distribution(
        cpu_root, DATASET, SPECIE, GLM_MODEL, DATASET, RADIUS,
        dis_method=DIS_METHOD, rsa_method=RSA_METHOD, rsa_model=RSA_MODEL,
        reps_group=REPS_GROUP, verbose=False)
    worst = 0.0
    for kind in ("mean", "std"):
        rel = gpu_group.distribution_rel(manifest, RSA_MODEL, kind)
        a = load(os.path.join(gpu_root, rel.replace("/", os.sep)))
        b = load(os.path.join(cpu_root, rel.replace("/", os.sep)))
        worst = max(worst, np.max(np.abs(a - b)))
    report("step 6 null distribution mean/std vs rsa_utils", worst, 1e-12)

    rsa_utils.calculate_z_maps_rnd(
        cpu_root, DATASET, SPECIE, GLM_MODEL, DATASET, RADIUS,
        dis_method=DIS_METHOD, rsa_method=RSA_METHOD, rsa_model=RSA_MODEL,
        verbose=False, reps_group=REPS_GROUP, replace_file=True)
    rsa_utils.calculate_z_map_real_data(
        cpu_root, DATASET, SPECIE, GLM_MODEL, RADIUS, dis_method=DIS_METHOD,
        rsa_method=RSA_METHOD, rsa_model=RSA_MODEL, verbose=False,
        mask_type=MASK_TYPE)

    worst, nan_ok = 0.0, True
    for g in range(REPS_GROUP):
        rel = gpu_group.group_rnd_rel(manifest, RSA_MODEL, "z", g)
        a = load(os.path.join(gpu_root, rel.replace("/", os.sep)))
        b = load(os.path.join(cpu_root, rel.replace("/", os.sep)))
        nan_ok &= bool(np.array_equal(np.isnan(a), np.isnan(b)))
        both = np.isfinite(a) & np.isfinite(b)
        worst = max(worst, np.max(np.abs(a[both] - b[both])))
    report("step 7 rnd z maps vs rsa_utils", worst, 1e-12)
    report_bool("step 7 rnd z NaN pattern matches (off-mask stays NaN)", nan_ok)

    rel = gpu_group.group_real_rel(manifest, RSA_MODEL, "z")
    a = load(os.path.join(gpu_root, rel.replace("/", os.sep)))
    b = load(os.path.join(cpu_root, rel.replace("/", os.sep)))
    report("step 7 real z map vs rsa_utils", float(np.max(np.abs(a - b))), 1e-12)


def check_step8(manifest, gpu_root, cpu_root):
    """Compare the in-memory cluster-size distribution against the CPU's.

    The CPU step reads the rnd z maps off disk; the GPU one never writes them and
    works on the same array in memory. To make this a test of *step 8* rather
    than of z-map rounding, the CPU is pointed at the GPU's own z maps.
    """
    import rsa_utils
    for g in range(REPS_GROUP):
        rel = gpu_group.group_rnd_rel(manifest, RSA_MODEL, "z", g)
        dst = os.path.join(cpu_root, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.join(gpu_root, rel.replace("/", os.sep)), dst)

    for t in Z_THRESHOLDS:
        rsa_utils.calculate_cluster_size_distribution(
            cpu_root, DATASET, GLM_MODEL, RSA_MODEL, RADIUS, SPECIE,
            DIS_METHOD, RSA_METHOD, z_threshold=t, verbose=False,
            replace_file=False)

    rel = gpu_group.cluster_dist_rel(manifest, RSA_MODEL)
    gpu_dist = np.load(os.path.join(gpu_root, rel.replace("/", os.sep)),
                       allow_pickle=True).item()
    cpu_dist = np.load(os.path.join(cpu_root, rel.replace("/", os.sep)),
                       allow_pickle=True).item()

    report_bool("step 8 writes the path step 9 rebuilds",
                os.path.exists(os.path.join(gpu_root, rel.replace("/", os.sep))),
                rel)
    keys_ok = all(f"z{float(t)}" in gpu_dist for t in Z_THRESHOLDS)
    report_bool("step 8 keys match get_minimal_cluster_size's f\"z{threshold}\"",
                keys_ok, str(sorted(gpu_dist)))

    same, n_clusters, detail = True, 0, ""
    for t in Z_THRESHOLDS:
        key = f"z{float(t)}"
        if key not in cpu_dist or key not in gpu_dist:
            same, detail = False, f"{key} missing"
            continue
        a, b = gpu_dist[key], cpu_dist[key]
        if a["number_of_images"] != b["number_of_images"]:
            same, detail = False, f"{key} n_images {a['number_of_images']} != {b['number_of_images']}"
        for g in range(REPS_GROUP):
            ga, cb = list(a["cluster_sizes"][g]), list(b["cluster_sizes"][g])
            n_clusters += len(ga)
            if ga != cb:
                same = False
                detail = detail or f"{key} perm {g}: {ga} != {cb}"
    report_bool("step 8 cluster sizes vs rsa_utils (exact)", same,
                detail or f"{n_clusters} clusters compared over "
                          f"{len(Z_THRESHOLDS)} threshold(s)")
    report_bool("step 8 found clusters to compare (test is not vacuous)",
                n_clusters > 0, f"{n_clusters} clusters")

    # step 9 must be able to derive a minimal cluster size from what we wrote
    try:
        sizes = rsa_utils.get_minimal_cluster_size(
            os.path.join(gpu_root, rel.replace("/", os.sep)),
            float(Z_THRESHOLDS[0]), 0.05)
        ok, why = isinstance(sizes, (int, np.integer)), f"minimal size {sizes}"
    except Exception as exc:
        ok, why = False, str(exc)
    report_bool("step 9's get_minimal_cluster_size reads the file", ok, why)


def check_no_bulk_output(manifest, out_dir):
    """With the defaults, neither the group means nor the rnd z maps are shipped."""
    zip_path = os.path.join(out_dir, f"result_group_{RSA_MODEL}_{SPECIE}.zip")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    means = [n for n in names if "_mean_" in n and "/mean/" in n]
    zmaps = [n for n in names if "_z_" in n and n.endswith(".nii.gz")]
    dist = [n for n in names if "/dist/" in n and n.endswith(".npy")]
    report_bool("default run ships no group mean maps", not means,
                f"{len(means)} found")
    report_bool("default run ships no rnd z maps", not zmaps, f"{len(zmaps)} found")
    report_bool("default run ships the cluster-size distribution", len(dist) == 1,
                f"{len(dist)} found")
    report_bool("default run still ships steps 3/6/7-real",
                len(names) >= 6, f"{len(names)} member(s): {sorted(names)[:8]}")


def build_refs_dir(tmp, data_root):
    """The committed-refs equivalent for the synthetic dataset."""
    refs_dir = os.path.join(tmp, "refs")
    os.makedirs(os.path.join(refs_dir, DATASET), exist_ok=True)
    shutil.copyfile(
        os.path.join(data_root, DATASET, "ROI", SPECIE, f"{MASK_TYPE}.nii.gz"),
        os.path.join(refs_dir, DATASET, f"{SPECIE}_{MASK_TYPE}.nii.gz"))
    with open(os.path.join(refs_dir, f"{DATASET}_refs.json"), "w") as f:
        json.dump({"dataset": DATASET, "datafolder": "", "model": GLM_MODEL,
                   "species": {SPECIE: {"task": DATASET,
                                        "participants": PARTICIPANTS,
                                        "runs_by_sub": {}}}}, f)
    return refs_dir


def check_package_free(pkg_root, refs_dir, zip_dir, pkg_gpu_root, tmp):
    """A run with no package must reproduce the packaged run exactly.

    This is the path the notebook takes: the manifest is recovered from the
    arcnames inside the result zips, and only the mask and the config's
    participant list come from the refs folder. Everything downstream --
    filenames, grids, values -- has to be identical to the packaged run.
    """
    pkg_manifest = gpu_group.load_manifest(pkg_root)
    got = gpu_group.discover_manifest(
        zip_dir, refs_dir, specie=SPECIE, reps_group=REPS_GROUP, verbose=False)

    same = {k: (got.get(k), pkg_manifest.get(k)) for k in
            ("dataset", "model", "specie", "task", "radius", "mask_type",
             "dis_method", "mah_fold", "rsa_method", "reps", "reps_group",
             "participants")
            if got.get(k) != pkg_manifest.get(k)}
    report_bool("discovered manifest == packaged manifest", not same, str(same))
    report_bool("discovered model list == packaged model list",
                got["models"] == [RSA_MODEL], str(got["models"]))

    out_dir = os.path.join(tmp, "out_free")
    work_root = os.path.join(tmp, "work_free")
    run_colab_group.run_group_package(
        None, zip_dir, out_dir, work_root=work_root, refs_dir=refs_dir,
        specie=SPECIE, reps_group=REPS_GROUP, steps=(3, 5, 6, 7, 8), device=DEV,
        batch=64, g_batch=5, workers=4, write_group_means=True, write_z_maps=True,
        z_thresholds=Z_THRESHOLDS, prefetch_zips=True, keep_work=True,
        verbose=False)
    free_root = os.path.join(work_root, "data")

    worst, checked, missing = 0.0, 0, []
    for dirpath, _dirs, files in os.walk(pkg_gpu_root):
        for name in files:
            if not name.endswith(".nii.gz"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, name), pkg_gpu_root)
            other = os.path.join(free_root, rel)
            if not os.path.exists(other):
                missing.append(rel)
                continue
            a, b = load(os.path.join(pkg_gpu_root, rel)), load(other)
            if not np.array_equal(np.isnan(a), np.isnan(b)):
                report_bool("package-free NaN pattern matches", False, rel)
                return
            both = np.isfinite(a) & np.isfinite(b)
            worst = max(worst, float(np.max(np.abs(a[both] - b[both]))))
            checked += 1
    report_bool("package-free produces the same file set", not missing,
                f"{len(missing)} missing: {missing[:3]}")
    report(f"package-free == packaged ({checked} maps)", worst, 0.0)

    rel = gpu_group.cluster_dist_rel(got, RSA_MODEL).replace("/", os.sep)
    a = np.load(os.path.join(pkg_gpu_root, rel), allow_pickle=True).item()
    b = np.load(os.path.join(free_root, rel), allow_pickle=True).item()
    same8 = sorted(a) == sorted(b) and all(
        [list(x) for x in a[k]["cluster_sizes"]] ==
        [list(x) for x in b[k]["cluster_sizes"]] for k in a)
    report_bool("package-free step-8 distribution matches", same8, str(sorted(b)))

    # the prefetch cache is per-model scratch and must not survive the run
    cache = os.path.join(work_root, "zip_cache", RSA_MODEL)
    report_bool("prefetch cache is cleared after each model",
                not os.path.isdir(cache), cache)



def check_versions_agree():
    """The three files that travel to Drive must declare the same version."""
    print("\nversion stamps")
    import run_colab_group as rcg
    report_bool("gpu_group.py and run_colab_group.py agree",
                gpu_group.VERSION == getattr(rcg, "VERSION", None),
                f"{gpu_group.VERSION} vs {getattr(rcg, 'VERSION', None)}")

    nb = os.path.join(HERE, "colab_rsa_group.ipynb")
    nb_ver = None
    if os.path.exists(nb):
        blob = "".join(x for c in json.load(open(nb, encoding="utf-8"))["cells"]
                       for x in c["source"])
        m = re.search(r'NOTEBOOK_VERSION\s*=\s*"([^"]+)"', blob)
        nb_ver = m.group(1) if m else None
    report_bool("colab_rsa_group.ipynb declares the same version",
                nb_ver == gpu_group.VERSION,
                f"notebook {nb_ver} vs gpu_group {gpu_group.VERSION}")
    report_bool("check_versions() accepts a matching set",
                gpu_group.check_versions(gpu_group.VERSION, strict=False,
                                         verbose=False))
    raised = False
    try:
        gpu_group.check_versions("0.0.1-stale", strict=True, verbose=False)
    except RuntimeError:
        raised = True
    report_bool("check_versions() raises on a stale notebook", raised)


def check_version_in_outputs(manifest, gpu_root):
    """A merged result must still say which code produced it."""
    print("\nversion recorded in the outputs")
    rel = gpu_group.group_real_rel(manifest, RSA_MODEL, "mean").replace(
        ".nii.gz", ".json").replace("/", os.sep)
    path = os.path.join(gpu_root, rel)
    ok = False
    if os.path.exists(path):
        import yaml
        with open(path, encoding="utf-8") as f:
            ok = yaml.safe_load(f).get("gpu_group_version") == gpu_group.VERSION
    report_bool("step 3's .json records gpu_group_version", ok,
                f"v{gpu_group.VERSION}")

    log = os.path.join(gpu_root, gpu_group.cluster_dist_rel(
        manifest, RSA_MODEL).replace(".npy", "_log.txt").replace("/", os.sep))
    ok8 = os.path.exists(log) and f"v{gpu_group.VERSION}" in open(
        log, encoding="utf-8").read()
    report_bool("step 8's log records the version", ok8)


def check_split_availability(pkg_root, data_root, tmp):
    """A unit with a real map but no permutation maps must still count for step 3.

    Step 3 and step 5 have independent availability on the CPU -- a participant
    whose step-4 job has not finished is still in the group mean. Merging the two
    would drop them from both.
    """
    tree = os.path.join(tmp, "partial")
    shutil.copytree(os.path.join(data_root, DATASET), os.path.join(tree, DATASET))
    victim = PARTICIPANTS[1]
    shutil.rmtree(os.path.join(tree, DATASET, "results", "RSA_rnd", GLM_MODEL,
                               RSA_MODEL, f"{SPECIE}-sub-{victim:02d}"))
    manifest = gpu_group.load_manifest(pkg_root)
    store = gpu_group.ResultStore([tree], dataset=DATASET, verbose=False)
    mask_path = os.path.join(pkg_root, "data", DATASET, "ROI", SPECIE,
                             f"{MASK_TYPE}.nii.gz")
    mask_img = nib.load(mask_path)
    mask_bool = np.asarray(mask_img.dataobj).astype(bool)
    maps = gpu_group.load_participant_maps(manifest, RSA_MODEL, store, mask_img,
                                           mask_bool, workers=4, verbose=False)
    report_bool("real/perm availability are tracked separately",
                maps["n_real"] == len(PARTICIPANTS)
                and len(maps["units"]) == len(PARTICIPANTS) - 1,
                f"real={maps['n_real']}/{len(PARTICIPANTS)}, "
                f"perm units={len(maps['units'])}/{len(PARTICIPANTS)}")


def check_zip_roundtrip(manifest, out_dir, tmp):
    """The result zip must merge with tools/unpack_results.py unchanged."""
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import unpack_results

    zip_path = os.path.join(out_dir, f"result_group_{RSA_MODEL}_{SPECIE}.zip")
    target = os.path.join(tmp, "merged")
    os.makedirs(target, exist_ok=True)
    written, _ = unpack_results.unpack_zip(zip_path, target, verify_size=True)

    expected = [
        gpu_group.group_real_rel(manifest, RSA_MODEL, "mean"),
        gpu_group.group_real_rel(manifest, RSA_MODEL, "std"),
        gpu_group.group_real_rel(manifest, RSA_MODEL, "mean").replace(".nii.gz", ".json"),
        gpu_group.group_real_rel(manifest, RSA_MODEL, "z"),
        gpu_group.distribution_rel(manifest, RSA_MODEL, "mean"),
        gpu_group.distribution_rel(manifest, RSA_MODEL, "std"),
        gpu_group.group_rnd_rel(manifest, RSA_MODEL, "mean", 0),
        gpu_group.group_rnd_rel(manifest, RSA_MODEL, "z", REPS_GROUP - 1),
        gpu_group.group_rnd_log_rel(manifest, RSA_MODEL, "z"),
    ]
    missing = [r for r in expected
               if not os.path.exists(os.path.join(target, r.replace("/", os.sep)))]
    report_bool("unpack_results merges the group zip",
                not missing and written > 0,
                f"{written} file(s)" + (f", missing {missing}" if missing else ""))

    # step 8 has to be able to glob the rnd z maps by the exact name it expects
    import glob
    pattern = os.path.join(
        target, DATASET, "results", "RSA_rnd", GLM_MODEL, RSA_MODEL, "mean",
        f"{SPECIE}-r-{RADIUS}_{DIS_METHOD}_{RSA_METHOD}_z_*.nii.gz")
    found = glob.glob(pattern)
    report_bool("step 8's glob finds the rnd z maps",
                len(found) == REPS_GROUP, f"{len(found)}/{REPS_GROUP}")


# ---------------------------------------------------------------------------
def main():
    tmp = tempfile.mkdtemp(prefix="rsa_group_val_")
    try:
        data_root = os.path.join(tmp, "data")
        os.makedirs(data_root, exist_ok=True)
        build_dataset(data_root)
        pkg_root = os.path.join(tmp, "pkg")
        manifest = build_package(pkg_root, data_root)

        # Run 1 keeps every intermediate on disk, because the fidelity checks
        # below compare them against what rsa_utils writes.
        out_dir = os.path.join(tmp, "out")
        work_root = os.path.join(tmp, "work")
        run_colab_group.run_group_package(
            pkg_root, data_root, out_dir, work_root=work_root,
            steps=(3, 5, 6, 7, 8), device=DEV, batch=64, g_batch=5, workers=4,
            write_group_means=True, write_z_maps=True, z_thresholds=Z_THRESHOLDS,
            keep_work=True, verbose=False)
        gpu_root = os.path.join(work_root, "data")

        zip_dir = build_result_zips(data_root, os.path.join(tmp, "zips"))
        check_zip_source(pkg_root, zip_dir, gpu_root, tmp)
        check_package_free(pkg_root, build_refs_dir(tmp, data_root), zip_dir,
                           gpu_root, tmp)
        check_split_availability(pkg_root, data_root, tmp)
        cpu_root = check_step3(manifest, gpu_root, data_root)
        check_step5(manifest, gpu_root, data_root)
        check_steps_6_7(manifest, gpu_root, cpu_root)
        check_step8(manifest, gpu_root, cpu_root)
        check_versions_agree()
        check_version_in_outputs(manifest, gpu_root)
        check_zip_roundtrip(manifest, out_dir, tmp)

        # Run 2 uses the shipping defaults: the bulky intermediates stay here.
        out2 = os.path.join(tmp, "out_default")
        run_colab_group.run_group_package(
            pkg_root, data_root, out2, work_root=os.path.join(tmp, "work_default"),
            steps=(3, 5, 6, 7, 8), device=DEV, batch=64, g_batch=5, workers=4,
            z_thresholds=Z_THRESHOLDS, verbose=False)
        check_no_bulk_output(manifest, out2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAIL:
        print(f"{len(FAIL)} check(s) FAILED: {', '.join(FAIL)}")
        sys.exit(1)
    print("All group-step checks passed.")


if __name__ == "__main__":
    main()

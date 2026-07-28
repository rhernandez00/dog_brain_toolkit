#!/usr/bin/env python
"""validate_gpu.py -- prove the GPU kernels match the CPU pipeline.

Run on the workstation (Anaconda Python, torch CPU is fine):

    & "C:\\ProgramData\\anaconda3\\python.exe" tools\\colab_gpu\\validate_gpu.py

Checks, in order:
  1. batched_ledoit_wolf  vs  sklearn.covariance.LedoitWolf        (random matrices)
  2. batched_crossnobis   vs  rsa_utils.crossnobis                 (random patterns)
  3. _kendall_taua        vs  rsa_utils.kendall_tau_a              (random + NaN model)
  4. crossnobis_searchlight vs the existing D-sub-01 r-3 maps on the data disk
     (the real end-to-end ground truth for step 1)

Exits non-zero if any check exceeds tolerance.
"""

import os
import sys

import numpy as np
import torch
import yaml
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)                       # so rsa_utils / utils import

import gpu_rsa                                  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)
DEV = torch.device("cpu")
FAIL = []


def report(name, err, tol):
    ok = err <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name:44s} max|err|={err:.3e}  (tol {tol:.0e})")
    if not ok:
        FAIL.append(name)


# ---------------------------------------------------------------------------
# 1. Ledoit-Wolf
# ---------------------------------------------------------------------------
def check_ledoit_wolf():
    from sklearn.covariance import LedoitWolf
    worst = 0.0
    for P in (5, 20, 60, 123, 257):
        n = 60
        R = np.random.randn(8, n, P) * np.random.rand(8, 1, P)   # varied scales
        cov_gpu = gpu_rsa.batched_ledoit_wolf(
            torch.as_tensor(R, dtype=gpu_rsa.DTYPE, device=DEV)).cpu().numpy()
        for b in range(R.shape[0]):
            cov_sk = LedoitWolf().fit(R[b]).covariance_
            worst = max(worst, np.max(np.abs(cov_gpu[b] - cov_sk)))
    report("batched_ledoit_wolf vs sklearn", worst, 1e-10)


# ---------------------------------------------------------------------------
# 2. crossnobis
# ---------------------------------------------------------------------------
def check_crossnobis():
    import rsa_utils
    M, C, exemplars, P = 6, 10, 4, 123
    conds = [f"c{c:02d}" for c in range(C)]
    Y, labels, partitions = [], [], []
    for m in range(M):
        for ci, c in enumerate(conds):
            for _ in range(exemplars):
                Y.append(np.random.randn(P) + ci * 0.1)
                labels.append(c)
                partitions.append(m)
    Y = np.array(Y); labels = np.array(labels); partitions = np.array(partitions)
    D_cpu = rsa_utils.crossnobis(Y, labels, partitions, return_rdm=True)

    # build U[m, c] the same way crossnobis does, then batch
    U = np.zeros((M, C, P))
    for m in range(M):
        for ci, c in enumerate(conds):
            U[m, ci] = Y[(partitions == m) & (labels == c)].mean(0)
    D_gpu = gpu_rsa.batched_crossnobis(
        torch.as_tensor(U[None], dtype=gpu_rsa.DTYPE, device=DEV))[0].cpu().numpy()
    report("batched_crossnobis vs rsa_utils", np.max(np.abs(D_cpu - D_gpu)), 1e-9)


# ---------------------------------------------------------------------------
# 3. kendall tau-a (incl. NaN-masked model)
# ---------------------------------------------------------------------------
def check_kendall():
    import rsa_utils
    k = 45
    Nv, Nm = 20, 6
    data = np.random.randn(Nv, k)
    model = np.random.randn(Nm, k)
    model[1, [3, 7, 11, 20]] = np.nan          # grouping-masked pairs
    model[2, :] = np.nan                        # degenerate row -> all-NaN
    tau_gpu = gpu_rsa._kendall_taua(
        torch.as_tensor(data, dtype=gpu_rsa.DTYPE, device=DEV),
        torch.as_tensor(model, dtype=gpu_rsa.DTYPE, device=DEV)).cpu().numpy()
    worst = 0.0
    for v in range(Nv):
        for m in range(Nm):
            cpu = rsa_utils.kendall_tau_a(data[v], model[m])
            g = tau_gpu[v, m]
            if np.isnan(cpu) and np.isnan(g):
                continue
            worst = max(worst, abs(cpu - g))
    report("_kendall_taua vs rsa_utils.kendall_tau_a", worst, 1e-10)


# ---------------------------------------------------------------------------
# 4. full step-1 searchlight vs existing D-sub-01 r-3 maps on disk
# ---------------------------------------------------------------------------
def check_step1_realdata():
    datafolder = r"P:\userdata\raulh87\data"
    dataset, model, specie, sub_N, radius = "EmoC", "basic-block", "D", 1, 3
    cfg_path = os.path.join(datafolder, dataset, "config_files", f"{specie}_{model}.yaml")
    if not os.path.exists(cfg_path):
        print("[SKIP] step-1 realdata check (data disk not available)")
        return
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    stim_types = cfg["stim_types"]
    categories = gpu_rsa.stimwise_categories(stim_types, dataset)

    import pandas as pd
    db = pd.read_csv(os.path.join(datafolder, dataset, "BIDS", f"{specie}_database-details.csv"))
    db = db[db["sub_N"] == sub_N]
    runs = [{"session": int(r.session), "run_N": int(r.run_N)} for r in db.itertuples()]

    manifest = dict(dataset=dataset, model=model, specie=specie, sub_N=sub_N,
                    task=cfg["task"], stim_types=stim_types, categories=categories,
                    runs=runs, radius=radius, mask_type="b_GreyMatter2mmB")
    mask_path = os.path.join(datafolder, dataset, "ROI", specie, "b_GreyMatter2mmB.nii.gz")
    mask_bool = np.asarray(nib.load(mask_path).dataobj).astype(bool)

    print(f"[step1 realdata] {len(runs)} runs, {int(mask_bool.sum())} voxels -- computing ...")
    means, affine, cats, runs = gpu_rsa._load_category_means(datafolder, manifest, mask_bool)
    dist_maps = gpu_rsa.crossnobis_searchlight(means, mask_bool, cats, runs, radius,
                                               device=DEV, batch=2048, verbose=False)

    base = os.path.join(datafolder, dataset, "results", "RSA", model, f"{specie}-sub-{sub_N:02d}")
    worst, checked = 0.0, 0
    for (c1, c2), vol in dist_maps.items():
        a = os.path.join(base, f"r-{radius}_mahalanobis_{c1}_{c2}.nii.gz")
        b = os.path.join(base, f"r-{radius}_mahalanobis_{c2}_{c1}.nii.gz")
        ref_path = a if os.path.exists(a) else (b if os.path.exists(b) else None)
        if ref_path is None:
            print(f"   (no reference map for {c1} vs {c2})")
            continue
        ref = np.asarray(nib.load(ref_path).dataobj, dtype=np.float64)
        both = np.isfinite(ref) & np.isfinite(vol)
        if both.any():
            worst = max(worst, np.max(np.abs(ref[both] - vol[both])))
        # NaN pattern should match (same searchlight support)
        checked += 1
    report(f"step1 searchlight vs disk ({checked} maps)", worst, 1e-5)


def check_step2_realdata():
    """Full step-2 path: model CSV -> vector, meta assembly, kendall, vs a fresh
    CPU reference (rsa_utils.kendall_tau_a) on the same 45 disk maps. Also checks a
    grouping model with NaN-masked pairs."""
    import rsa_utils
    datafolder = r"P:\userdata\raulh87\data"
    dataset, model, specie, sub_N, radius = "EmoC", "basic-block", "D", 1, 3
    base = os.path.join(datafolder, dataset, "results", "RSA", model, f"{specie}-sub-{sub_N:02d}")
    if not os.path.isdir(base) or not glob_any(base):
        print("[SKIP] step-2 realdata check (no step-1 maps on disk)")
        return
    cfg = yaml.safe_load(open(os.path.join(datafolder, dataset, "config_files",
                                           f"{specie}_{model}.yaml")))
    categories = gpu_rsa.stimwise_categories(cfg["stim_types"], dataset)
    manifest = dict(dataset=dataset, model=model, specie=specie, sub_N=sub_N,
                    radius=radius, mask_type="b_GreyMatter2mmB", categories=categories)
    meta_t, mask_flat, shape, _aff = gpu_rsa.load_meta_similarity(datafolder, manifest, DEV)
    meta_np = meta_t.cpu().numpy()                              # (n_vox, 45) canonical order

    worst = 0.0
    for rsa_model in ("valence3__all", "valence3__dog"):
        csv = os.path.join(datafolder, dataset, "rsa_models", f"{rsa_model}.csv")
        if not os.path.exists(csv):
            continue
        Mmat = gpu_rsa.read_model_matrix(csv, categories)
        mvec = gpu_rsa._upper_tri_vector(Mmat)                  # canonical order
        # cross-check model vector against rsa_utils._build_mahalanobis_model_vector
        rmd = rsa_utils.read_model_dict(csv)
        cpu_pairs, cpu_vals = rsa_utils._build_mahalanobis_model_vector(
            list(rmd["categories"]), {c: c for c in rmd["categories"]}, rmd)
        cpu_lookup = {tuple(sorted(p)): v for p, v in zip(cpu_pairs, cpu_vals)}
        pairs = gpu_rsa.canonical_pairs(categories)
        vec_err = max(
            (0.0 if (np.isnan(mvec[k]) and np.isnan(cpu_lookup[tuple(sorted(pairs[k]))]))
             else abs(mvec[k] - cpu_lookup[tuple(sorted(pairs[k]))]))
            for k in range(len(pairs)))
        report(f"  model vector {rsa_model} vs CPU", vec_err, 1e-12)

        sim_gpu = gpu_rsa.compute_similarity(
            meta_t, torch.as_tensor(mvec[None], dtype=gpu_rsa.DTYPE, device=DEV),
            "kendall")[:, 0].cpu().numpy()
        step = max(1, meta_np.shape[0] // 200)
        for v in range(0, meta_np.shape[0], step):
            cpu = rsa_utils.kendall_tau_a(meta_np[v], mvec)
            g = sim_gpu[v]
            if np.isnan(cpu):
                cpu = 0.0                                       # CPU writes 0 where skipped
            if np.isnan(g):
                g = 0.0
            worst = max(worst, abs(cpu - g))
    report("step2 kendall (real models) vs rsa_utils", worst, 1e-9)


def glob_any(base):
    import glob as _g
    return bool(_g.glob(os.path.join(base, "r-3_mahalanobis_*.nii.gz")))


# ---------------------------------------------------------------------------
# 5-6. correlation path: Pearson-RDM step 1 (vs disk) + step-2 kendall over 780
# ---------------------------------------------------------------------------
def check_correlation():
    import rsa_utils
    datafolder = r"P:\userdata\raulh87\data"
    dataset, model, specie, sub_N, radius = "EmoC", "basic-block", "D", 1, 3
    cfg_path = os.path.join(datafolder, dataset, "config_files", f"{specie}_{model}.yaml")
    if not os.path.exists(cfg_path):
        print("[SKIP] correlation checks (data disk not available)")
        return
    cfg = yaml.safe_load(open(cfg_path))
    stim_types, task = cfg["stim_types"], cfg["task"]
    entry = {"session": 1, "run_N": 5}                     # D-sub-01 ses-01 run-05 exists
    manifest = dict(dataset=dataset, model=model, specie=specie, sub_N=sub_N, task=task,
                    stim_types=stim_types, categories=list(stim_types), radius=radius,
                    mask_type="b_GreyMatter2mmB")
    mask_path = os.path.join(datafolder, dataset, "ROI", specie, "b_GreyMatter2mmB.nii.gz")
    mask_bool = np.asarray(nib.load(mask_path).dataobj).astype(bool)

    betas, _aff = gpu_rsa._load_run_betas(datafolder, manifest, entry, mask_bool)
    dist, mask_flat = gpu_rsa.pearson_rdm_searchlight(betas, mask_bool, radius,
                                                      device=DEV, batch=4096)
    pairs = gpu_rsa.canonical_pairs(stim_types)            # 780
    shape = mask_bool.shape

    # 5. step-1 vs disk maps for D-sub-01 ses-01 run-05
    base = os.path.join(datafolder, dataset, "results", "RSA", model,
                        f"{specie}-sub-{sub_N:02d}", "ses-01_task-EmoC_run-05")
    worst1, checked = 0.0, 0
    for k, (s1, s2) in enumerate(pairs):
        a = os.path.join(base, f"r-{radius}_correlation_{s1}_{s2}.nii.gz")
        b = os.path.join(base, f"r-{radius}_correlation_{s2}_{s1}.nii.gz")
        ref_path = a if os.path.exists(a) else (b if os.path.exists(b) else None)
        if ref_path is None:
            continue
        ref = np.asarray(nib.load(ref_path).dataobj, dtype=np.float64).reshape(-1)[mask_flat]
        both = np.isfinite(ref) & np.isfinite(dist[:, k])
        if both.any():
            worst1 = max(worst1, float(np.max(np.abs(ref[both] - dist[:, k][both]))))
        checked += 1
    report(f"correlation step1 vs disk ({checked} maps)", worst1, 1e-5)

    # 6. model vector + step-2 kendall over 780 items
    rsa_model = "emo-id__collapse"
    csv = os.path.join(datafolder, dataset, "rsa_models", f"{rsa_model}.csv")
    Mmat = gpu_rsa.read_model_matrix(csv, stim_types)
    mvec = gpu_rsa._upper_tri_vector(Mmat)
    rmd = rsa_utils.read_model_dict(csv)
    cpu_lookup = {tuple(sorted(p)): rmd["model"][p[0]][p[1]] for p in rmd["pairs"]}
    vec_err = max(abs(mvec[k] - cpu_lookup[tuple(sorted(pairs[k]))]) for k in range(len(pairs)))
    report(f"  model vector {rsa_model} vs CPU (780)", vec_err, 1e-12)

    dist64 = dist.astype(np.float64)
    sample = np.arange(0, dist.shape[0], max(1, dist.shape[0] // 200))
    meta_t = torch.as_tensor(dist64[sample], dtype=gpu_rsa.DTYPE, device=DEV)
    sim_gpu = gpu_rsa.compute_similarity(
        meta_t, torch.as_tensor(mvec[None], dtype=gpu_rsa.DTYPE, device=DEV),
        "kendall", vox_batch=1024)[:, 0].cpu().numpy()
    worst2 = 0.0
    for i, v in enumerate(sample):
        cpu = rsa_utils.kendall_tau_a(dist64[v], mvec)
        g = sim_gpu[i]
        worst2 = max(worst2, abs((0.0 if np.isnan(cpu) else cpu) - (0.0 if np.isnan(g) else g)))
    report("correlation step2 kendall (780) vs rsa_utils", worst2, 1e-9)


if __name__ == "__main__":
    check_ledoit_wolf()
    check_crossnobis()
    check_kendall()
    check_step1_realdata()
    check_step2_realdata()
    check_correlation()
    print("\n" + ("ALL CHECKS PASSED" if not FAIL else f"FAILURES: {FAIL}"))
    sys.exit(1 if FAIL else 0)

# colab_gpu/ — GPU acceleration for RSA steps 1, 2, 4

Run the compute-heavy RSA steps for **one participant × many models** on a Colab
GPU (L4/T4, High-RAM), then drop the results back onto the pipeline data disk so
steps 3–10 of `searchlight.py` continue unchanged on the workstation.

Two distance methods are supported (`--dis_method`, read from `_models.csv`):

| `dis_method` | fold | step 1 | pairs | models (EmoC) |
|---|---|---|---|---|
| `mahalanobis` (default) | stim-wise | cross-run crossnobis, 10 categories → 45 maps per subject | 45 | 50 |
| `correlation` | run-wise | per-run Pearson RDM, 40 stimuli → 780 maps per run | 780 | 41 |

In both cases step 1 is model-independent (run once) and steps 2 & 4 reuse its maps.
For correlation, Kendall's tau-a over the 780-item RDM uses a **float32 sign matmul**
(exact — sums of ±1 stay far below 2²⁴) with float64 division, voxel-batched for
memory.

## Why it's fast

- **Step 1** (crossnobis searchlight) is the only heavy part and is
  model-independent — it runs once. It's a per-voxel eigendecomposition + whitening
  of a covariance matrix, which batches cleanly on the GPU (`torch.linalg.eigh`).
- **Steps 2 & 4** load the 45 maps once into an `(n_voxels, 45)` matrix and reduce
  to matmuls. Kendall's tau-a factors into a signed upper-triangle matmul
  `data_sign(n_vox, 990) @ model_sign(990, n_perms)`, so a 50-model × 100-permutation
  battery is a handful of matmuls.

Everything runs in **float64** to match the CPU (numpy) pipeline. Validated against
the CPU to ~1e-12 (see `validate_gpu.py`).

## Faithfulness note

Step-1 and step-2 (real) maps match the CPU pipeline to ~1e-12, including the CPU
quirk that crossnobis **partitions by `run_N`** — two sessions sharing a run number
collapse into one partition (see `_load_category_means`). Step-4 permutations use
the same *scheme* as `rsa_utils.shuffle_vector` (permute category labels) with their
own deterministic seed, so they are a valid draw from the same null, not
bit-identical to a CPU rerun.

## Files

| File | What it is |
|---|---|
| `gpu_rsa.py` | PyTorch step 1/2/4 kernels: `batched_ledoit_wolf`, `batched_crossnobis`, `crossnobis_searchlight`, `run_step1`, `run_model`, plus per-part zip helpers. Copied into every package. Depends only on torch/numpy/nibabel. |
| `run_colab.py` | Orchestrator: step 1 once + steps 2/4 per model, one `result_*.zip` per part, resumable. Importable (`run_package`) or a CLI. |
| `colab_rsa.ipynb` | The Colab notebook — check GPU, mount Drive, unzip package, run. |
| `validate_gpu.py` | Correctness harness vs the CPU pipeline (LW, crossnobis, kendall, step-1 vs disk maps, step-2). Run on the workstation. |
| `packages/` | Default output folder for `tools/create_package.py` (git-ignored contents). |

## Workflow

```powershell
# 1. Build a package (workstation). H-sub-40 with the whole mahalanobis battery:
& "C:\ProgramData\anaconda3\python.exe" tools\create_package.py H 40 --all-stim-wise
#    ...or the whole correlation battery (41 models, run-wise):
& "C:\ProgramData\anaconda3\python.exe" tools\create_package.py H 40 --dis_method correlation --all

# 2. Upload the pkg_*.zip to a Google Drive folder, open colab_rsa.ipynb in Colab
#    (GPU runtime), set PKG_ZIP + OUT_DIR, run all cells.
#    -> result_step1_*.zip once, then result_<model>_*.zip per model, in OUT_DIR.

# 3. Download OUT_DIR and merge back onto the data disk (workstation):
& "C:\ProgramData\anaconda3\python.exe" tools\unpack_results.py DOWNLOADS_DIR

# 4. Continue the pipeline (steps 3-10) as usual, e.g. via the scheduler.
```

For a quick local smoke test without Colab (CPU torch is fine):

```powershell
& "C:\ProgramData\anaconda3\python.exe" tools\create_package.py D 1 --models valence3__all valence3__cross --reps 10
& "C:\ProgramData\anaconda3\python.exe" tools\colab_gpu\run_colab.py --pkg <unzipped-pkg> --out <scratch-out> --cpu
```

## Validate

```powershell
& "C:\ProgramData\anaconda3\python.exe" tools\colab_gpu\validate_gpu.py
```
Exits non-zero if any kernel diverges from the CPU beyond tolerance.

# colab_gpu/ — GPU acceleration for RSA steps 1, 2, 4 and 3, 5, 6, 7

Two halves that chain, plus a step-5-only shortcut:

| Half | Scope | Package | Notebook | Writes |
|---|---|---|---|---|
| **participant** — steps 1, 2, 4 | one participant × many models | `tools/create_package.py` | `colab_rsa.ipynb` | `result_step1_*.zip`, `result_<model>_<specie>-sub-NN.zip` |
| **group** — steps 3, 5, 6, 7 | all participants × many models | `tools/create_group_package.py` | `colab_rsa_group.ipynb` | `result_group_<model>_<specie>.zip` |
| **step 5 only** | all participants × many models | *none needed* | `colab_rsa_step5.ipynb` | `result_step5_<model>_<specie>.zip` |

The group half's input is the participant half's **output folder**: it reads the
per-participant maps straight out of the `result_*.zip` files already sitting in
OUT_DIR, so nothing has to come down to the workstation and go back up in
between. Afterwards `tools/unpack_results.py` merges either kind of zip and
`searchlight.py` continues — from step 3 after a participant run, from step 8
after a group run.

The third row is for when you want **step 5 and nothing else** — see
[Step 5 on its own](#step-5-on-its-own) below.

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
- **Steps 3, 5, 6, 7** are I/O-bound on the CPU, not arithmetic-bound. Step 5 draws
  one of each participant's `reps` permutation maps for each of `reps_group` group
  permutations, i.e. `reps_group × n_participants` NIfTI loads (1000 × 16 = 16 000
  reads of the same 1600 files), and step 6 then walks all 1000 group maps twice for
  a mean and an std. Here those 1600 maps are read **once** into an
  `(n_maps, n_mask_voxels)` matrix and the whole of steps 5–7 becomes one
  voxel-chunked index-gather plus a mean along the participant axis. What is left is
  writing the output volumes.

Everything runs in **float64** to match the CPU (numpy) pipeline. Validated against
the CPU to ~1e-12 (`validate_gpu.py` for steps 1/2/4, `validate_group.py` for
steps 3/5/6/7).

## Faithfulness note

Step-1 and step-2 (real) maps match the CPU pipeline to ~1e-12, including the CPU
quirk that crossnobis **partitions by `run_N`** — two sessions sharing a run number
collapse into one partition (see `_load_category_means`). Step-4 permutations use
the same *scheme* as `rsa_utils.shuffle_vector` (permute category labels) with their
own deterministic seed, so they are a valid draw from the same null, not
bit-identical to a CPU rerun.

The group steps are the same story one level up. Steps 3, 6 and 7 are exact; step 5
draws its per-participant permutation with a seeded RNG where the CPU uses an
unseeded `random.choice`, so it is a valid draw from the same null but not a
reproduction of a particular CPU run. Feed `validate_group.py`'s CPU side the GPU's
own step-5 maps and steps 6–7 agree to ~1e-16.

Details that are easy to get wrong and are reproduced deliberately:

- the group **rnd** maps carry **no** `{mask_type}-` prefix while the group **real**
  maps do — step 8's glob depends on it;
- `mah_fold` sub-foldering applies to the **participant** paths only, never to the
  group `mean/` folder;
- step 3 multiplies mean and std by the mask;
- step 7's rnd z maps keep their non-finite values (outside the mask the CPU
  computes `(0-0)/0`, so those voxels are NaN on disk), while the real z map zeroes
  them and is cast to float32 *under the mean map's float64 header* — exactly what
  `calculate_z_map_real_data` does;
- step 3's `.json` sidecar is written too, because
  `calculate_group_model_similarity_map` reads it back to decide whether the map has
  to be recomputed. Its paths are rendered as **workstation** paths, from the
  `datafolder` the package records.

## Output geometry

Every output volume is written with the **mask's affine** — the mask is the single
reference voxel grid, and `check_same_space()` verifies that every beta map and
every step-1 map sits on it before anything is computed.

This used to be "whichever input image was loaded first", while the CPU used
"whichever was loaded last". On a compliant dataset those are the same grid and it
made no difference; on EmoC humans — where each run's betas are in a different
scanner-native space — GPU and CPU maps came out numerically identical but with
headers 5.2 mm apart, which looks like a shift or artifact in a viewer. The check
now refuses such a dataset outright, on both paths. `check_same_space` here is a
deliberate copy of `rsa_utils.check_same_space` (packages ship only `gpu_rsa.py`,
so it cannot import the toolkit) — **keep the two in sync**.

Check a dataset before packaging:

```bash
python tools/check_space.py --dataset EmoC --specie H --model basic-block
```

## Files

| File | What it is |
|---|---|
| `gpu_rsa.py` | PyTorch step 1/2/4 kernels: `batched_ledoit_wolf`, `batched_crossnobis`, `crossnobis_searchlight`, `run_step1`, `run_model`, plus per-part zip helpers. Copied into every package. Depends only on torch/numpy/nibabel. |
| `run_colab.py` | Orchestrator for steps 1/2/4: step 1 once + steps 2/4 per model, one `result_*.zip` per part, resumable. Importable (`run_package`) or a CLI. `run_package(..., calculate_step1=True)` ignores any step-1 maps bundled in the package (or a manifest that claims `step1_done`) and recomputes on the GPU regardless — for a package built before you decided the bundled maps shouldn't have been reused, without rebuilding and re-uploading the (often multi-GB) zip. `delete_step1=True` additionally deletes any step-1 maps already unpacked into the package before deciding, which on its own also forces the recompute; pair it with `calculate_step1` when the bundled maps might sit under the *other* pair orientation (`catB_catA` instead of `catA_catB`) from what a fresh compute writes, since a stale file in the other orientation would not simply get overwritten. Both flags default to `False` and are exposed in `colab_rsa.ipynb` as `CALCULATE_STEP1`/`DELETE_STEP1`, and on the CLI as `--calculate_step1`/`--delete_step1`. See also `tools/purge_step1_from_package.py`, which strips bundled step-1 maps out of already-uploaded package zips in place, for when you'd rather shrink the zip on Drive than have Colab discard them at unpack time. |
| `colab_rsa.ipynb` | The Colab notebook for steps 1/2/4 — check GPU, mount Drive, unzip package, run. |
| `gpu_group.py` | PyTorch step 3/5/6/7 kernels: `ResultStore` (reads participant maps out of the result zips), `draw_group_indices`, `group_permutation_stats`, `run_group_model`, `zip_group_result`. Also the pipeline path builders for every group output. Imports `gpu_rsa` for the voxel-grid check. |
| `run_colab_group.py` | Orchestrator for steps 3/5/6/7: one `result_group_<model>_<specie>.zip` per model, resumable. Importable (`run_group_package`) or a CLI. |
| `colab_rsa_group.ipynb` | The Colab notebook for steps 3/5/6/7. |
| `gpu_step5.py` | **Step 5 alone**, with no package and no mask: `scan_results` recovers every pipeline parameter from the arcnames inside the result zips, and `run_step5`/`run_step5_all` build the group null. Self-contained (torch/numpy/nibabel only, imports nothing else in this folder) so it can be dropped next to the notebook on Drive. Importable or a CLI. |
| `colab_rsa_step5.ipynb` | The Colab notebook for step 5 on its own. Needs only `gpu_step5.py` and the folder of `result_*.zip`. |
| `validate_gpu.py` | Correctness harness vs the CPU pipeline (LW, crossnobis, kendall, step-1 vs disk maps, step-2). Run on the workstation. |
| `validate_group.py` | Correctness harness for the group steps: builds a synthetic dataset, runs both paths, compares steps 3/5/6/7, and checks that the result zip merges via `unpack_results.py` and that step 8's glob finds the z maps. |
| `validate_step5.py` | Correctness harness for `gpu_step5.py`: compares every output map against a plain **full-volume** mean of the same drawn files (the reduction `nifti_mean` performs), over the stim-wise and per-run layouts, ragged permutation counts and a stem with no `{mask_type}-` prefix. Also checks arcname parsing, the availability gate, that prefetching reads changes nothing byte-wise, and that the zip merges via `unpack_results.py`. |
| `packages/` | Default output folder for `tools/create_package.py` and `tools/create_group_package.py` (git-ignored contents). |

## Workflow

```powershell
# 1. Build a per-participant package (workstation). H-sub-40, whole mahalanobis battery:
& "C:\ProgramData\anaconda3\python.exe" tools\create_package.py H 40 --all-stim-wise
#    ...or the whole correlation battery (41 models, run-wise):
& "C:\ProgramData\anaconda3\python.exe" tools\create_package.py H 40 --dis_method correlation --all

# 2. Upload the pkg_*.zip to a Google Drive folder, open colab_rsa.ipynb in Colab
#    (GPU runtime), set PKG_ZIP + OUT_DIR, run all cells. Repeat per participant.
#    -> result_step1_*.zip once, then result_<model>_*.zip per model, in OUT_DIR.

# 3. Once EVERY participant is done, build the group package (34 kB -- mask + manifest):
& "C:\ProgramData\anaconda3\python.exe" tools\create_group_package.py H --all-stim-wise

# 4. Upload it to the SAME Drive folder, open colab_rsa_group.ipynb, set
#    PKG_ZIP + RESULTS_DIR + OUT_DIR, run all cells.
#    -> result_group_<model>_<specie>.zip per model, in OUT_DIR.

# 5. Download OUT_DIR and merge back onto the data disk (workstation):
& "C:\ProgramData\anaconda3\python.exe" tools\unpack_results.py DOWNLOADS_DIR

# 6. Continue the pipeline (steps 8-10) as usual, e.g. via the scheduler.
```

## Group steps — what to know before running them

**Everyone has to be finished.** Step 5 averages one permutation map per
participant, so the group half needs every participant's `result_<model>_*.zip` in
`RESULTS_DIR`. A participant with no permutation maps is dropped and reported; if
that pushes availability below `min_percentage_available` (default 1.0) the model
is skipped with a message rather than silently averaged over fewer subjects.

**Output volume.** Step 5 writes `reps_group` group mean maps and step 7 writes
`reps_group` z maps — 2000 files per model at the default `reps_group=1000`. On
the data disk a dog group map measures ~43 kB, so that is **~86 MB per model per
species** (humans more). Only the z maps are read downstream,
by step 8; the group means are inputs to steps 6 and 7, both of which run here. Set
`WRITE_GROUP_MEANS = False` in the notebook (`--skip_group_means` on the CLI) to
leave them out and roughly halve what you download.

**`RESULTS_DIR` may also be an unpacked data root**, which is handy for a local
test — but reading maps one file at a time off `P:` is *slow* (measured 2026-08-03:
~7 s per map). The zip path reads each participant's 101 maps out of a single file
and is what a real run should use.

**Steps 3 and 7 travel together**: the real z map is `(group mean − null mean) /
null std`, so step 7 needs step 3's output. Keep `3` in `STEPS`, or make sure the
step-3 mean map is already in `RESULTS_DIR`.

## Step 5 on its own

`colab_rsa_step5.ipynb` + `gpu_step5.py` run **step 5 and nothing else**, and
unlike everything else here they need **no package** — just the folder of
`result_<model>_<specie>-sub-NN.zip` on Drive. Copy the one file across and go:

```
REM workstation, Anaconda Prompt
copy \github\dog_brain_toolkit\tools\colab_gpu\gpu_step5.py "G:\My Drive\rsa_colab\"
```

Two things make the package unnecessary:

* **No manifest.** Dataset, GLM model, radius, `dis_method`, `rsa_method`,
  `mah_fold`, the per-run layout, who has which permutation indices — all of it
  is in the arcnames inside the zips, and `scan_results` reads it back out. It
  refuses to proceed if two zips of one model disagree on any of it, since that
  would build one null out of two analyses.
* **No mask.** Step 5 calls `nifti_mean(files_list, result_map_path=...)`
  *without* `mask_img`, so it is a plain voxelwise mean over full volumes. A
  voxel that is 0 in every input is 0 in the output, so restricting the
  arithmetic to the union of the inputs' support is exact — and that union comes
  from the maps themselves. It is what keeps the accumulator to 1.2 GB instead
  of 7.2 GB on the human grid (159 198 of 902 629 voxels, measured on EmoC).

  The support is *grown* as maps arrive rather than taken from the first map,
  because a Kendall tau does land on exactly 0.0: on EmoC H-sub-01 about 1 000
  voxels differ in support between two maps of the same participant, and a real
  run expanded 18 times before settling.

**When to use it instead of `colab_rsa_group.ipynb`.** Only when you want step 5
by itself. Its output is bulky — 1000 group mean maps are ~450 MB per model on
the human grid — and those maps feed steps 6 and 7 and nothing else. If you want
6 and 7 too, `colab_rsa_group.ipynb` with `WRITE_GROUP_MEANS = False` does
3/5/6/7 in one pass and brings back a few megabytes.

**The availability gate needs a denominator.** The CPU compares against the
config's participant list; there is no config on Drive, so
`EXPECTED_PARTICIPANTS` says what to compare against: `'auto'` (every
`<specie>-sub-NN` appearing in *any* result zip in the folder — so a participant
who finished other models but not this one still counts as expected), an integer,
or `'found'` to disable the check. Availability is counted in **participants**,
not participant-runs, because maps arrive one zip per participant.

**Reads dominate.** Each map is read exactly once (the CPU re-reads each one
`reps_group × units / maps` times — 10× at EmoC human scale), and the reads are
prefetched on a thread pool while being *applied in order*, so the result stays
bit-identical to a serial run. Measured on `action_tendency__all` [H], 32
participants off a local Drive mount: `--read_workers 1` took 118 s, the default
8 took 11 s.

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
& "C:\ProgramData\anaconda3\python.exe" tools\colab_gpu\gpu_step5.py `
    --results "G:\My Drive\rsa_colab\results" --out <scratch-out> `
    --specie H --models action_tendency__all --reps_group 1000 `
    --min_percentage_available 0.5 --cpu
```

For a quick local smoke test without Colab (CPU torch is fine):

```powershell
& "C:\ProgramData\anaconda3\python.exe" tools\create_package.py D 1 --models valence3__all valence3__cross --reps 10
& "C:\ProgramData\anaconda3\python.exe" tools\colab_gpu\run_colab.py --pkg <unzipped-pkg> --out <scratch-out> --cpu

& "C:\ProgramData\anaconda3\python.exe" tools\create_group_package.py D --models valence3__all --reps_group 20 --participants 1 3 4
& "C:\ProgramData\anaconda3\python.exe" tools\colab_gpu\run_colab_group.py --pkg <unzipped-pkg> --results <dir-of-result-zips> --out <scratch-out> --cpu
```

## Validate

```powershell
& "C:\ProgramData\anaconda3\python.exe" tools\colab_gpu\validate_gpu.py
& "C:\ProgramData\anaconda3\python.exe" tools\colab_gpu\validate_group.py
& "C:\ProgramData\anaconda3\python.exe" tools\colab_gpu\validate_step5.py
```
Each exits non-zero if any kernel diverges from the CPU beyond tolerance. Both need
`KMP_DUPLICATE_LIB_OK=TRUE` on this machine (Anaconda and torch each ship an OpenMP
runtime).

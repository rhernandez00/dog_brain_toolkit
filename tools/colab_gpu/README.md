# colab_gpu/ — GPU acceleration for RSA steps 1, 2, 4 and 3, 5, 6, 7, 8

Two halves that chain:

| Half | Scope | Package | Notebook | Writes |
|---|---|---|---|---|
| **participant** — steps 1, 2, 4 | one participant × many models | `tools/create_package.py` | `colab_rsa.ipynb` | `result_step1_*.zip`, `result_<model>_<specie>-sub-NN.zip` |
| **group** — steps 3, 5, 6, 7, 8 | all participants × many models | *none* (`refs/`), or `tools/create_group_package.py` | `colab_rsa_group.ipynb` | `result_group_<model>_<specie>.zip` |

The group half's input is the participant half's **output folder**: it reads the
per-participant maps straight out of the `result_*.zip` files already sitting in
OUT_DIR, so nothing has to come down to the workstation and go back up in
between. Afterwards `tools/unpack_results.py` merges either kind of zip and
`searchlight.py` continues — from step 3 after a participant run, from step 9
after a group run.

Two distance methods are supported (`--dis_method`, read from `_models.csv`):

| `dis_method` | fold | step 1 | pairs | models (EmoC) |
|---|---|---|---|---|
| `mahalanobis` (default) | stim-wise | cross-run crossnobis, 10 categories → 45 maps per subject | 45 | 50 |
| `correlation` | run-wise | per-run Pearson RDM, 40 stimuli → 780 maps per run | 780 | 41 |

**Run-dependent models** — `visual1`, `visual2`, `flow` — have no `{model}.csv`;
they are one predicted RDM per run, `{model}-run-{run_N}.csv`, because they
describe the video actually shown in that run. Name them on `--models` like any
other model and `create_package.py` bundles every run's CSV and lists them in the
manifest's `run_dependent_models`; `gpu_rsa` then reads the matching matrix inside
its per-run loop. They require `--dis_method correlation`, the only path whose
step-2/4 output is per run — under `mahalanobis` the runs are already pooled into
one crossnobis RDM, and `create_package.py` refuses the combination.

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
- **Steps 3, 5, 6, 7, 8** are I/O-bound on the CPU, not arithmetic-bound. Step 5 draws
  one of each participant's `reps` permutation maps for each of `reps_group` group
  permutations, i.e. `reps_group × n_participants` NIfTI loads (1000 × 16 = 16 000
  reads of the same 1600 files), and step 6 then walks all 1000 group maps twice for
  a mean and an std. Here those 1600 maps are read **once** into an
  `(n_maps, n_mask_voxels)` matrix and the whole of steps 5–7 becomes one
  voxel-chunked index-gather plus a mean along the participant axis. What is left is
  writing the output volumes.

Everything runs in **float64** to match the CPU (numpy) pipeline. Validated against
the CPU to ~1e-12 (`validate_gpu.py` for steps 1/2/4, `validate_group.py` for
steps 3/5/6/7/8).

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
| `gpu_group.py` | PyTorch step 3/5/6/7/8 kernels: `ResultStore` (reads participant maps out of the result zips), `draw_group_indices`, `group_permutation_stats`, `cluster_size_distribution` (step 8, over the z maps still in memory, several thresholds per pass, labelling cropped to the mask's bounding box), `run_group_model`, `zip_group_result`. Also the pipeline path builders for every group output. Imports `gpu_rsa` for the voxel-grid check, and `scipy.ndimage` for step 8's labelling. |
| `run_colab_group.py` | Orchestrator for steps 3/5/6/7/8: one `result_group_<model>_<specie>.zip` per model, resumable. Importable (`run_group_package`) or a CLI. |
| `colab_rsa_group.ipynb` | The Colab notebook for steps 3/5/6/7/8. |
| `validate_gpu.py` | Correctness harness vs the CPU pipeline (LW, crossnobis, kendall, step-1 vs disk maps, step-2). Run on the workstation. |
| `validate_group.py` | Correctness harness for the group steps: builds a synthetic dataset, runs both paths, compares steps 3/5/6/7 and step 8's cluster sizes against `rsa_utils` exactly, checks that `get_minimal_cluster_size` can read the `.npy` step 8 writes, that a default run ships neither the group means nor the rnd z maps, and that the result zip merges via `unpack_results.py`. |
| `memory_estimate.py` | Works out what a group run will allocate and which Colab runtime fits, from the numbers that decide it: units x reps x mask voxels. Reports host RAM and GPU RAM separately, and separates the **loading** peak from the **computing** peak — `load_participant_maps` builds a list of per-map vectors and then `np.stack`s them, so both exist at once and the loading peak is 2x the steady state. `--results` probes a real folder and reports per battery. |
| `refs/` | The committed reference snapshot that replaces the package: `{dataset}_refs.json` (participant list, task, `runs_by_sub`) and `{dataset}/{specie}_{mask_type}.nii.gz` (the searchlight mask). ~80 kB for EmoC. `refs/build_refs.py` builds it and is the **only** thing here that touches the data share; `refs/check_refs_mask.py` then verifies the mask against the maps actually inside the result zips (grid, affine, and that nothing has support outside it) — worth running after every rebuild, because `ROI/H/` holds several same-named-looking masks on the wrong grid. |
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

# 6. Continue the pipeline (steps 9-10; step 8 already ran on Colab), e.g. via the scheduler.
```

## Group steps — what to know before running them

**Crash recovery.** `run_colab_group.py` writes a `result_group_{model}_{specie}.started`
marker into `OUT_DIR` the instant a model's processing begins, and removes it on a
clean finish (or a clean `MissingMapsError` skip). A marker surviving with no
matching `.zip` means the runtime died mid-model — the next run (`force=False`,
the default) skips that model too instead of retrying it into the same crash, so
a battery with one consistently-crashing model can still finish the rest in one
pass. Delete the marker file, or pass `force=True`, to retry a specific model
deliberately.

**Known issue (2026-09-08, EmoC H correlation battery):** a model has been
observed to crash the Colab runtime with an apparent host-RAM OOM at the exact
same point on repeated attempts — same model, same position in a sorted run
order, near-identical `avail=` RAM reported by `mem_report` just before it dies
— across both a fresh session and a session that had already processed 40+
other models cleanly. Lowering `BATCH` from 20000 to 2000 (a 10x cut to the
per-chunk GPU-transfer allocation) made **no difference** to when or how it
crashed, which rules out that term and a simple cross-model RAM leak as the
sole cause; something appears specific to that particular model's data.
Not yet root-caused. If you hit this, the `.started`-marker skip above is the
practical workaround; if you're debugging it, start from `mem_report` logging
around `run_group_model` for the specific model, not the battery-wide loop.

**Everyone has to be finished.** Step 5 averages one permutation map per
participant, so the group half needs every participant's `result_<model>_*.zip` in
`RESULTS_DIR`. A participant with no permutation maps is dropped and reported; if
that pushes availability below `min_percentage_available` (default 1.0) the model
is skipped with a message rather than silently averaged over fewer subjects.

**Output volume — the reason step 8 runs here too.** Step 5 writes `reps_group`
group mean maps and step 7 writes `reps_group` z maps: 2000 whole-brain volumes
per model at the default `reps_group=1000`. Measured on the EmoC human grid
(91×109×91, 159 198 mask voxels) that is **727 MB + 1273 MB per model**. Neither
is a result:

| Output | Read by | Runs where |
|---|---|---|
| step 5 group means | steps 6 and 7 | here |
| step 7 rnd z maps | step 8 | **here**, since v2 |
| step 3 + 6 + real z + step 8's `.npy` (~4 MB) | steps 9–10 | workstation |

So `WRITE_GROUP_MEANS` and `WRITE_Z_MAPS` both default to **False** and a model
ships ~4 MB instead of ~2 GB — for a ~90-model human battery, roughly 0.4 GB
instead of 180 GB, and no multi-gigabyte upload per model. Turn `WRITE_Z_MAPS`
back on only if you want to run step 8 on the workstation instead.

**Step 8 and thresholds.** `Z_THRESHOLDS` (CLI `--z_thresholds`) defaults to
`[3.1, 3.5, 4.0, 4.5, 5.0]`. All of them are computed in one pass — labelling is
cheap next to reading the participant maps — and each is stored under its own
`z{threshold}` key, exactly as `get_minimal_cluster_size` rebuilds it from step
9's float `--z_threshold`. **List every threshold you might want up front:** the
z maps are not kept, so adding one later means re-running the model. Labelling is
cropped to the mask's bounding box, which is exact (nothing outside the mask is
non-zero, so no cluster can cross the box) and about 3× less volume to walk on
the human grid.

**Merging step 8's output.** `unpack_results.py` skips existing files unless
given `--replace`. A model that already has `dist/..._dist.npy` on the data disk
from an earlier workstation run will therefore *not* pick up the Colab one —
delete it first or unpack with `--replace`. The Colab file carries every
threshold in `Z_THRESHOLDS`, so replacing loses nothing unless the old file held
a threshold you did not list.

**Leaving the step-4 maps behind.** A per-participant `result_<model>_*.zip`
carries `--reps` permutation maps per participant per run, and step 5 is their
only reader. Once the group half has run them on Colab, unpacking them writes
the bulkiest thing on the disk for nothing — `tools/bulk_check.py --delete_step4`
would only reclaim it again. `unpack_results.py --no_step4_files` skips them:
it excludes the *participant* level of `results/RSA_rnd/`, so the group `mean/`
outputs of steps 5 and 7 in the same tree are still merged.

**`RESULTS_DIR` may also be an unpacked data root**, which is handy for a local
test — but reading maps one file at a time off `P:` is *slow* (measured 2026-08-03:
~7 s per map). The zip path reads each participant's 101 maps out of a single file
and is what a real run should use.

**Steps 3 and 7 travel together**: the real z map is `(group mean − null mean) /
null std`, so step 7 needs step 3's output. Keep `3` in `STEPS`, or make sure the
step-3 mean map is already in `RESULTS_DIR`.

## Running without a package

The group half does **not** need `pkg_group_*.zip`. Leave `PKG_ZIP` out and point
the notebook at `refs/` instead: `discover_manifest()` rebuilds the manifest from
the arcnames inside the result zips, which already state the dataset, GLM model,
RSA model, specie, radius, `dis_method`, `rsa_method`, `mah_fold`, `mask_type`,
the per-run layout, and which permutation indices each participant has. It
refuses to proceed when two zips of one model disagree on any of it, since that
would build one null distribution out of two analyses.

Three things are genuinely not in an arcname, and they live in the committed
`refs/` folder (~80 kB for EmoC, built by `refs/build_refs.py`):

* **the mask** — the reference voxel grid, and what the off-mask-zero check on
  every participant map is made against;
* **the participant list** — the denominator of the availability check. A
  participant who has produced nothing leaves no trace on Drive, so "32
  participants have maps" cannot become a percentage without it;
* **`runs_by_sub` and `task`**, which only the per-run layouts need.

**Verify the mask, don't trust its name.** `EmoC/ROI/H/` holds five masks and
three of them are on the scanner-native EPI grid left over from the alignment
problem:

| file | grid | voxels | |
|---|---|---|---|
| `b_GreyMatter2mmB.nii.gz` | (91,109,91) | 159 254 | **the right one** |
| `b_GreyMatter2mmB_pre-alignment.nii.gz` | (96,96,52) | 108 433 | wrong grid |
| `b_greyMatter2mm.nii.gz` | (96,96,52) | 114 280 | wrong grid |
| `results_space.nii.gz` | (96,96,52) | 114 280 | wrong grid |
| `original_atlas_space.nii.gz` | (91,109,91) | 171 094 | right grid, wrong extent |

`refs/check_refs_mask.py` decides by evidence instead of by filename: it compares
the refs copy against every candidate on the share, then against participant maps
sampled straight out of the result zips — shape, affine within 0.5 mm, and that
no map has non-zero voxels outside the mask. Run it after every `build_refs.py`.
Measured on the current EmoC H zips: 24 maps from 6 models, all on grid, **0**
voxels outside, and a full-model read covers 159 198 of the mask's 159 254 voxels
(99.96%).

`refs/build_refs.py` is now the *only* thing in this folder that touches the data
share, and it needs to run again only when a dataset's config participant list
changes. Everything else — including a whole battery on Colab — works from the
committed snapshot. `validate_group.py` checks that a package-free run reproduces
a packaged one byte for byte.

The result zips themselves are always read **in place**, packaged or not: nothing
unpacks them and nothing rebuilds them.

**Prefetching — measured, and off by default.** `PREFETCH_ZIPS` (CLI
`--prefetch_zips`) copies a model's result zips to local disk before reading
their members. The idea was that the mount is *latency*-bound on many small
member reads, so one bulk copy would beat them. Measured on two untouched models
(38 zips / ~1.6 GB each, cold cache, 8 threads, Windows Drive mount):

| | time | throughput |
|---|---|---|
| direct member reads | **128.0 s** | 12.6 MB/s |
| prefetch (copy 119.8 s + local re-read 56.1 s) | 175.9 s | 9.3 MB/s |

Direct reads already sustain the same MB/s as a bulk copy, so the mount is
**bandwidth**-bound at this thread count and prefetching just reads the same
bytes twice. `load_participant_maps` reads every member of every zip, so there is
no subset to win back either. Kept as an option because it is mount-dependent and
Colab's FUSE layer may behave differently — time one model each way before
turning it on for a battery. Both paths produce byte-identical arrays; the cache
is per-model and cleared between models (~1.6 GB of local scratch).

Do not read the earlier "234 s" figure as the cold cost of prefetching — that run
copied serially. The copy is parallel now; it still loses.

```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
& "C:\ProgramDatanaconda3\python.exe" tools\colab_gpu
un_colab_group.py `
    --results "G:\My Drive
sa_colab
esults" --out <scratch-out> `
    --refs tools\colab_gpu
efs --specie H --models action_tendency__all `
    --reps_group 1000 --min_percentage_available 0.8 --cpu
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
```
Each exits non-zero if any kernel diverges from the CPU beyond tolerance. Both need
`KMP_DUPLICATE_LIB_OK=TRUE` on this machine (Anaconda and torch each ship an OpenMP
runtime).

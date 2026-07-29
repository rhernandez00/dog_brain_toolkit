# `searchlight.py` — RSA Pipeline Usage Guide

`searchlight.py` is the command-line runner for the Representational Similarity
Analysis (RSA) pipeline. It processes fMRI BOLD data from **dogs (`D`)** and
**humans (`H`)** through a sequence of numbered steps, ending in thresholded,
cluster-corrected z-maps and an Excel table of significant brain regions.

All heavy lifting lives in `rsa_utils.py`; `searchlight.py` is a thin driver that
parses CLI arguments, loads the dataset config, and dispatches each requested
step to the matching `rsa_utils` function.

---

## 1. Quick start

```bash
# Linux (remote) — `python` resolves to the correct Anaconda interpreter
python searchlight.py \
    --dataset EmoC \
    --model basic-block \
    --rsa_model test-model \
    --specie D \
    --steps_to_run 1 2 3 4 5 6 7 8 9 10
```

```powershell
# Windows (local) — always use the full Anaconda path
& "C:\ProgramData\anaconda3\python.exe" searchlight.py `
    --dataset EmoC --model basic-block --rsa_model test-model `
    --specie D --steps_to_run 1 2 3 4 5 6 7 8 9 10
```

For a fast smoke test, reduce the permutation counts:

```bash
python searchlight.py --dataset EmoC --model basic-block --rsa_model test-model \
    --specie D --steps_to_run 2 3 4 5 6 7 8 9 10 --reps 10 --reps_group 50
```

> **Tip:** In production you normally do **not** call `searchlight.py` by hand.
> The scheduler (`schedule_rsa.py` → `run_jobs.py`) builds the correct dependency
> graph and invokes `searchlight.py` one step at a time. See
> [§7 Running via the scheduler](#7-running-via-the-scheduler).

---

## 2. Prerequisites

Before running, the following must already exist on the shared data disk:

| Requirement | Location |
|---|---|
| Dataset config YAML | `{datafolder}/{dataset}/config_files/{specie}_{model}.yaml` |
| RSA model CSV (steps 2+) | `{datafolder}/{dataset}/rsa_models/{rsa_model}.csv` |
| Brain mask | see [§5 Masks](#5-masks) |
| Beta maps | Dogs: produced by step 0. Humans: FSL first-level output must pre-exist. |
| Aligned beta maps | Produced by step 0.5 from the above — required by steps 1+ and by Colab packages. See [§5.1](#51-voxel-grid-invariant). |

The data folder is resolved automatically from the OS:

| Machine | `datafolder` |
|---|---|
| Windows | `P:\userdata\raulh87\data` |
| Linux | `/home/raulh87/mnt/a471/userdata/raulh87/data` |

The config YAML supplies dataset-level parameters — `participants`, `runs`,
`sessions`, `stim_types`, `radius_fwd`, `threshold_fwd`, `smooth`, `img_type`,
and optionally `model_dict`.

---

## 3. Pipeline steps

Steps run in the order given by `--steps_to_run`. Each step reads the outputs of
its dependencies from disk, so you can run steps individually as long as the
upstream outputs already exist.

| Step | Label | `rsa_utils` function | Notes |
|---|---|---|---|
| **0** | Beta maps | `calculate_beta_maps` | **Dogs only**; humans have pre-existing beta maps. |
| **0.5** | Aligned beta maps | `calculate_aligned_beta_maps` | Writes `beta_{stim}.nii.gz` on the template grid. **Humans need FSL → Linux only.** Run once per dataset; see [§5.1](#51-voxel-grid-invariant). |
| **1** | Pairwise similarity maps | `calculate_pairwise_similarity_maps2` | Most expensive step; independent of the RSA model. |
| **2** | Model similarity (real) | `compare_with_model2(rnd=False)` | Requires `--rsa_model`. |
| **3** | Group model similarity map | `calculate_group_model_similarity_map` | Averages step 2 across participants. |
| **4** | Permuted model similarity (RND) | `compare_with_model2(rnd=True)` | Controlled by `--reps` (default 100). |
| **5** | Group permutations (RND) | `calculate_group_model_similarity_map_rnd` | Controlled by `--reps_group` (default 1000). |
| **6** | Voxelwise RND distribution | `calculate_voxelwise_rnd_distribution` | Per-voxel mean & std of the null. |
| **7** | Z-maps | `calculate_z_maps_rnd` + `calculate_z_map_real_data` | Requires steps **3 and 6**. |
| **7.5** (`75`) | Z-map for real data only | `calculate_z_map_real_data` | Optional; run the real-data z-map separately. |
| **8** | Cluster size distribution | `calculate_cluster_size_distribution` | Controlled by `--z_threshold`. |
| **9** | Cluster correction | `apply_cluster_correction` | Requires steps **7 and 8**. |
| **10** | Create tables | `create_tables` | Output: `.xlsx` report. |
| **11** | Cross-participant similarity | `calculate_cross_participant_similarity` | Out of scheduler scope. |
| **12** | DSM extraction | `calculate_similarity_across_all_pairs` | Per significant cluster; uses `roi_database.csv`. |
| **13** | Movement (`.par`) parameters | `calculate_movement_parameters` + `preprocess_functions.fwd` | Runs mcflirt to derive framewise displacement. |

### Dependency graph (steps 0–10)

```
0 → 0.5 → 1 → 2 → 3 ↘
              ↓        7 → 8 → 9 → 10
              4 → 5 → 6 ↗
```

- Step **7** requires step **3** (real data) **and** step **6** (null distribution).
- Step **9** requires step **7** (z-map) **and** step **8** (cluster distribution).
- Step **0.5** is a one-time migration per dataset, not part of the per-RSA-model
  loop. Once it has run, steps 1+ read its output and the `.feat` directories can
  be deleted.

> **Note on step 7.5:** Passing `--steps_to_run 75` triggers the "step 7.5" block
> (real-data z-map only). This is a naming quirk — `75` is the literal integer the
> code checks for.

---

## 4. Command-line arguments

### Core selection

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `EmoC` | Dataset name (also the default task). |
| `--task` | = dataset | Task name, if it differs from the dataset. |
| `--model` | `basic` | GLM model. Selects the config YAML and beta maps. |
| `--rsa_model` | `None` | RSA model CSV name. **Required for steps 2+.** |
| `--specie` | `H` | `D` = Dog, `H` = Human. |
| `--steps_to_run` | `1 2 3 4 5 6 7 8 9 10` | Space-separated list of step numbers. |

### Similarity / RSA methods

| Argument | Default | Description |
|---|---|---|
| `--method` | `mahalanobis` | Pairwise similarity method (step 1). Also `pearson`, `correlation`, `euclidean`, `kendall`. |
| `--rsa_method` | `kendall` | Method comparing similarity maps to the model (step 2). |
| `--mah_fold` | `stim-wise` | Mahalanobis folding: `stim-wise`, `stim-wise-multiple-folds`, `stim-wise-all-runs`, or `run-wise`. `stim-wise-multiple-folds` uses EmoC `stim_file` labels and repeated metadata partitions, producing direct-subject maps for exact repeatable stimulus pairs. EmoC-only `stim-wise-all-runs` computes each run independently, collapses exemplars such as `DogA1`--`DogA4` to `DogA`, and uses their exemplar IDs as within-run cross-validation folds. Steps 2/4 retain the legacy direct model-output path for `stim-wise`; other folds use a fold-specific result directory (with run folders for `stim-wise-all-runs`) before steps 3/5 aggregate to the usual group paths. |

### Searchlight / masking

| Argument | Default | Description |
|---|---|---|
| `--radius` | `None` → 3 (dog) / 4 (human) | Searchlight sphere radius in voxels. |
| `--mask_type` | `b_GreyMatter2mmB` | Brain mask selector. See [§5](#5-masks). |
| `--atlas_type` | `Nitzsche` | Atlas (dogs). Humans are forced to `MNI`. |

### Statistics / thresholding

| Argument | Default | Description |
|---|---|---|
| `--z_threshold` | `3.1` | Z-score threshold for cluster forming (steps 8–10). |
| `--cluster_threshold` | `0.05` | Cluster-level p threshold for correction (step 9). |
| `--reps` | `100` | Permutations per participant (step 4). |
| `--reps_group` | `1000` | Group-level permutations (step 5). |
| `--min_percentage_available` | `0.8` | Minimum fraction of the DB required to process. |
| `--min_dist_mm` | `8.0` | Minimum distance between peaks (step 10 tables). |

### Behaviour flags

| Argument | Default | Description |
|---|---|---|
| `--replace_file` | off | Recompute even when output files exist. |
| `--replace_rnd_files` | off | Recompute existing RND output files. |
| `--shuffle_participants` | off | Shuffle participant order in permutations. |
| `--shuffle_runs` | off | Shuffle run order (step 1 only). |
| `--participants_forced` | `[]` | Restrict to a subset of participant numbers. |
| `--skip_prefile_check` | off | Skip the input-file existence check. |
| `--overwrite_movement` | off | Overwrite existing movement files. |
| `--verbose` | off | Verbose logging. |
| `--wait_time` | `3000` | Seconds to wait between certain steps. |
| `--coords` | `None` | Voxel-space coords `x,y,z` for similarity files. |
| `--peak_id` | `None` | Peak id for step 12 (else derived from `roi_database.csv`). |
| `--job_marker_dir` | `None` | Directory for step completion markers (scheduler). |
| `--allow_space_mismatch` | off | Downgrade the voxel-grid check to a warning. See [§5.1](#51-voxel-grid-invariant). |

---

## 5. Masks

The `--mask_type` argument selects which voxels enter the searchlight:

- **Default (`b_GreyMatter2mmB` / `b_GreyMatter2mm`)** — grey-matter mask shipped
  in the atlas folder (dogs) or under `ROI/{specie}/` (humans).
- **`cope13`** — a functional mask from GLM results
  (`{datafolder}/{dataset}/ROI/{specie}/cope13.nii.gz`), matched to beta-map (GLM)
  space.
- **Any other value** — treated as a named mask under
  `{datafolder}/{dataset}/ROI/{specie}/{mask_type}.nii.gz`.

For **step 1** specifically, if the mask is not a grey-matter mask it is
automatically swapped for `b_GreyMatter2mmB`, because the searchlight must run over
the full grey-matter volume.

### 5.1 Voxel-grid invariant

The mask and **every beta map of every run of every participant must sit on the
same voxel grid** — same shape *and* same affine. Steps 1–3 combine images by
array index (`data[mask_bool]`, crossnobis folds across runs, group averaging)
and never resample, so a grid mismatch yields a map that looks normal but is
anatomically meaningless.

Matching shapes are **not** sufficient. FSL first-level output is the usual
offender: `fmri(regstandard_yn) 1` only *estimates* the transform into template
space and stores it in `reg/`; `stats/pe*.nii.gz` remain in scanner-native space,
a different grid per run. Dogs comply because their data is pre-normalised into
Nitzsche space — humans generally do not.

Measured on EmoC: dogs are clean (98 runs, one shape, one affine). Humans were
not — 239 runs all 96×96×52, but **111 distinct affines**, spread up to 72 mm
across the dataset and up to **58 mm within a single participant**.

Steps 1–3 raise `rsa_utils.SpaceMismatchError` on any mismatch. Check a dataset
before committing compute:

```bash
python tools/check_space.py --dataset EmoC --specie H --model basic-block
```

**To comply, run step 0.5, rebuild the mask, then re-run from step 1.** Step-1
outputs produced on mismatched grids are already wrong — restarting at step 2
does not fix them.

```bash
python searchlight.py --dataset EmoC --model basic-block --specie H --steps_to_run 0.5
```

Step 0.5 applies the transform FEAT already computed
(`reg/example_func2standard.mat` onto `reg/standard.nii.gz`, via
`flirt -applyxfm`) and writes `beta_{stim}.nii.gz` into a run folder beside the
`.feat`. It needs FSL, so for humans it runs on the **Linux machine only**; the
output lands on the shared disk, so Windows can run steps 1+ afterwards. For dogs
it copies the pe maps unchanged, since they are already in Nitzsche space.

Then put the mask on that same grid:

```bash
python tools/make_mask.py --dataset EmoC --specie H
```

Watch out for the mask filename: `ROI/D/` holds `b_GreyMatter2mmB.nii.gz` while
`ROI/H/` holds `b_greyMatter2mmB.nii.gz`, and `--mask_type` defaults to the
capital-G spelling. Windows resolves either; **Linux does not**. `make_mask.py`
warns when it finds a case-variant.

`--allow_space_mismatch` downgrades the error to a warning. It exists only to
reproduce legacy runs; results produced with it are not valid.

---

## 6. Outputs

Results are written under `{datafolder}/{dataset}/results/`:

- `results/RSA/{model}/{rsa_model}/mean/` — group mean maps, z-maps, corrected
  maps, and Excel tables.
- `results/RSA/{model}/{rsa_model}/dist/` — cluster-size distributions.
- `results/RSA_rnd/{model}/` — permutation (null) mean/std maps.

**Filenames encode the parameters** so runs with different settings do not
overwrite each other:

- Step 9: `{specie}-r-{radius}_{method}_{rsa_method}_zt{z_threshold}_corrected.nii.gz`
- Step 10: `{specie}-r-{radius}_{method}_{rsa_method}_zt{z_threshold}.xlsx`

---

## 7. Running via the scheduler

The recommended way to run a full analysis is through the job scheduler, which
encodes the dependency graph and lets both machines process jobs in parallel.

```powershell
# 1. Schedule — creates all job JSON files in pending/ and waiting/
python schedule_rsa.py --dataset EmoC --model basic-block --rsa_model MyModel

# 2. Run workers (either or both machines)
python run_jobs.py --max_jobs 0 --loop

# 3. Check progress
python job_status.py --dataset EmoC --rsa_model MyModel
```

The scheduler passes `--job_marker_dir` to `searchlight.py`. After each step, if
the step function reports success, `searchlight.py` writes a
`{step}.done` marker; the worker checks for that marker before marking the job
`completed`. See `CLAUDE.md` for the full scheduler description.

---

## 8. Species differences at a glance

| Property | Dogs (`D`) | Humans (`H`) |
|---|---|---|
| Starts at | Step 0 | Step 1 (beta maps pre-exist) |
| Default radius | 3 voxels | 4 voxels |
| Atlas (masks) | Nitzsche | MNI |
| Atlas (labels) | Czeibert / Johnson | AAL |
| Coord transform in step 10 | `False` | `True` |
| Design template | `FSL_designs/basic_DHRF.fsf` | `{dataset}/FSL_designs/H_{model}.fsf` |

---

## 9. Common pitfalls

- **Missing `--rsa_model`** — steps 2 and later need it; step 1 does not.
- **Running step 7 without steps 3 and 6** — z-maps need both the real group map
  and the null distribution.
- **Changing `--method` mid-analysis** — the method is baked into filenames and
  job IDs; a mismatched method silently reads/writes the wrong files.
- **`--z_threshold` mismatch** — steps 8, 9, 10 must share the same threshold, or
  step 9/10 will not find the step 8 distribution.

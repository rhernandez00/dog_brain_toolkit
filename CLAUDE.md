# dog_brain_toolkit — Codebase Guide

## What this repo is

A Python toolkit for running Representational Similarity Analysis (RSA) on neuroimaging data from two species: dogs (`D`) and humans (`H`). The pipeline processes fMRI BOLD data through ~11 sequential steps to produce thresholded, cluster-corrected z-maps and Excel tables of significant brain regions.

---

## Python environment

**Always use the full Anaconda path — never bare `python`.**

| Machine | Python path |
|---|---|
| Windows (local) | `C:\ProgramData\anaconda3\python.exe` |
| Linux (remote) | `python` (resolves correctly in that environment) |

Run with `& "C:\ProgramData\anaconda3\python.exe" script.py` in PowerShell.

---

## Shared data storage

Both machines mount the same network disk at different paths:

| Machine | Data folder |
|---|---|
| Windows | `P:\userdata\raulh87\data` |
| Linux | `/home/raulh87/mnt/a471/userdata/raulh87/data` |

Path selection is handled automatically by `scheduler/paths.py` via `os.name`.

---

## Key files

| File | Purpose |
|---|---|
| `searchlight.py` | Main pipeline runner — CLI, steps 0–13 |
| `rsa_utils.py` | All core RSA functions (~5000 lines) |
| `rsa_model_builder.py` | Dash web app (port 8051) to build RSA model CSVs |
| `brain_viewer.py` | Dash web app (port 8050) to visualize results |
| `utils.py` | Coordinate conversions, FSL template helpers |
| `preprocess_functions.py` | FSL preprocessing wrappers |
| `schedule_rsa.py` | **Scheduler CLI** — create job DAG for an RSA model |
| `run_jobs.py` | **Worker** — claim and execute pending jobs |
| `job_status.py` | **Status** — print step × species progress table |
| `scheduler/paths.py` | Machine-aware path resolution |
| `scheduler/dag.py` | Step dependency graph + job ID builder |
| `scheduler/jobs.py` | Atomic file-based job state machine |

---

## RSA pipeline steps (searchlight.py)

| Step | Label | Key function in rsa_utils.py | Notes |
|---|---|---|---|
| 0 | Beta maps | `calculate_beta_maps` | Dogs only; humans have pre-existing beta maps |
| 1 | Pairwise similarity | `calculate_pairwise_similarity_maps2` | Most expensive step; model-independent |
| 2 | Model similarity | `compare_with_model2(rnd=False)` | Requires `--rsa_model` |
| 3 | Group similarity map | `calculate_group_model_similarity_map` | |
| 4 | RND permuted model | `compare_with_model2(rnd=True)` | Controlled by `--reps` (default 100) |
| 5 | RND group permutations | `calculate_group_model_similarity_map_rnd` | Controlled by `--reps_group` (default 1000) |
| 6 | Voxelwise RND distribution | `calculate_voxelwise_rnd_distribution` | |
| 7 | Z-maps | `calculate_z_maps_rnd` + `calculate_z_map_real_data` | Requires steps 3 AND 6 |
| 8 | Cluster size distribution | `calculate_cluster_size_distribution` | Controlled by `--z_threshold` |
| 9 | Cluster correction | `apply_cluster_correction` | |
| 10 | Create tables | `create_tables` | Output: `.xlsx` Excel report |

**Dependency graph:**
```
0 → 1 → 2 → 3 ↘
        ↓         7 → 8 → 9 → 10
        4 → 5 → 6 ↗
```
Step 7 requires both step 3 (real data) and step 6 (null distribution).
Step 9 requires both step 7 (z-map) and step 8 (cluster distribution).

Steps 11–13 exist but are out of scope for the scheduler.

---

## Species differences

| Property | Dogs (D) | Humans (H) |
|---|---|---|
| Starts at | Step 0 | Step 1 (beta maps pre-exist) |
| Default radius | 3 voxels | 4 voxels |
| Atlas | Nitzsche/Czeibert/Johnson | MNI/AAL |
| Coords transform (step 10) | False | True |

---

## Key CLI parameters (searchlight.py)

```
--dataset       Dataset name, e.g. EmoC (required)
--model         GLM model, e.g. basic-block (default)
--rsa_model     RSA model CSV name, e.g. test-model (required for steps 2+)
--specie        H or D
--steps_to_run  One or more step numbers, e.g. --steps_to_run 1 2 3
--z_threshold   Z-score threshold for clustering (default 3.1)
--reps          Permutations for step 4 (default 100)
--reps_group    Permutations for step 5 (default 1000)
--replace_file  If set, recompute even if output files exist
```

---

## Output file naming

Step 9 and 10 outputs include `z_threshold` in the filename to prevent silent overwrites:
- Step 9: `{specie}-r-{radius}_{method}_{rsa_method}_zt{z_threshold}_corrected.nii.gz`
- Step 10: `{specie}-r-{radius}_{method}_{rsa_method}_zt{z_threshold}.xlsx`

Config files live at: `{datafolder}/{dataset}/config_files/{specie}_{model}.yaml`
RSA model CSVs live at: `{datafolder}/{dataset}/rsa_models/{rsa_model}.csv`

---

## Job scheduling system

Jobs queue lives on the shared network disk at `{datafolder}/job_queue/` with subfolders: `pending/`, `waiting/`, `running/`, `completed/`, `failed/`, `logs/`.

### Job ID format
```
{dataset}__{model}__{rsa_model}__{specie}__step{step:02d}__zt{z_threshold}__r{reps}__rg{reps_group}__m{method}
```
The trailing `__m{method}` (e.g. `__mcorrelation`, `__mmahalanobis`) keeps runs of the
same model under different pairwise-similarity methods from colliding in the queue.
`--method` defaults to `mahalanobis` and is threaded through `schedule_rsa.py` →
`scheduler/dag.py` → `run_jobs.py` (passed to `searchlight.py --method`). Pass `--method`
to `job_status.py` too, or it will look up the wrong (default) IDs.

### Workflow
```powershell
# 1. Schedule — creates all job JSON files in pending/ and waiting/
& "C:\ProgramData\anaconda3\python.exe" schedule_rsa.py `
    --dataset EmoC --model basic-block --rsa_model MyModel

# 2. Run (either machine, simultaneously if desired)
& "C:\ProgramData\anaconda3\python.exe" run_jobs.py --max_jobs 0 --loop

# 3. Check progress
& "C:\ProgramData\anaconda3\python.exe" job_status.py `
    --dataset EmoC --rsa_model MyModel
```

For fast test runs: `--reps 10 --reps_group 50`

### Worker mechanics
- Jobs are claimed by atomically renaming `pending/{id}.json` → `running/{id}.json`
- On success: moved to `completed/`, dependent waiting jobs promoted to `pending/`
- On failure: moved to `failed/` with exit code and log path recorded
- Both machines can run `run_jobs.py` simultaneously without conflict

### Success/failure signalling — Option B (completion markers)

The pipeline functions were not originally designed to signal success to a scheduler.
The chosen solution is **completion marker files** written by `searchlight.py` after each step.

**How it works:**
1. `run_jobs.py` creates a per-job marker directory: `{queue_dir}/markers/{job_id}/`
   and passes it to `searchlight.py` via `--job_marker_dir`
2. `searchlight.py` receives `--job_marker_dir`. After each step block, if the step
   function returned `True` (success), it writes `{job_marker_dir}/{step}.done`
3. Key `rsa_utils` functions are updated to explicitly `return True` on success
   (they already `raise` on failure, so no return = exception = no marker written)
4. After `searchlight.py` exits 0, `run_jobs.py` checks for `{job_marker_dir}/{step}.done`
   before calling `complete_job()`. Missing marker → job moved to `failed/` instead.

**Files changed:**
- `rsa_utils.py` — key step functions get explicit `return True` at the end
- `searchlight.py` — new `--job_marker_dir` arg; writes `{step}.done` after each step
- `run_jobs.py` — passes `--job_marker_dir`; checks marker before `complete_job()`
- `scheduler/jobs.py` — `complete_job()` accepts optional `marker_path` to verify

**Marker directory:** `{datafolder}/job_queue/markers/{job_id}/`
**Marker filename:** `{step}.done` (e.g., `9.done`, `10.done`)

### Planned UI (next session)
A UI is planned for the scheduler. Key features to design:
- View all jobs across all states (pending/waiting/running/completed/failed)
- Schedule a new RSA model (form: dataset, model, rsa_model, specie, z_threshold, reps, reps_group)
- Cancel or retry failed jobs
- Display log file content for a selected job
- Test RSA model: `test-model` on dataset `EmoC` (D dogs currently running steps 2–10)

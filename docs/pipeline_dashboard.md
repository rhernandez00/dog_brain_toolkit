# `pipeline_dashboard.py` — RSA pipeline status dashboard (Dash)

A browser dashboard that shows how far each RSA model has progressed through the
pipeline (steps 0–10) by checking the actual output files on disk. It is the web
version of `pipeline_console.py` and reuses the same file-probing logic.

## Run it

```bash
python tools/pipeline_dashboard.py                 # serves http://127.0.0.1:8060
python tools/pipeline_dashboard.py --port 8062     # choose a different port
RSA_DASHBOARD_PORT=8062 python tools/pipeline_dashboard.py
```

On Windows use the full interpreter path:

```powershell
& "C:\ProgramData\anaconda3\python.exe" tools\pipeline_dashboard.py
```

Then open **http://127.0.0.1:8060** in a browser.

### It will not interfere with your other consoles

- It runs as its **own process on its own port** (default **8060**, distinct from
  the `8050` unified dashboard and the `8051` model builder).
- It binds to **127.0.0.1 only** and runs **single-process with the reloader
  disabled**, so it never spawns child processes.
- Pressing **Ctrl+C in this app's terminal stops only this app** — your other
  Dash apps keep running in their own terminals.
- If port 8060 is already taken, pass a different `--port`.

## Using it

1. **Set the parameters** at the top: `dataset`, `model`, `rsa_model`
   (dropdown, populated from `{datafolder}/{dataset}/rsa_models/`), `specie`,
   `method`, `mah_fold` (dropdown; see below), `rsa_method`, `radius`
   (blank = auto: 3 dog / 4 human), `z_threshold`, `mask_type`, `reps`,
   `reps_group`. Press **⟳ reload models** if you add a new model CSV while the
   app is open.
2. **Check on demand.** Scanning the disk is slow, so **nothing is checked
   automatically**. Press **▶ Check all steps**, or the **Check** button on an
   individual step, to run that probe.
3. **Read the status.** Each step shows a coloured badge:
   green = all files present, orange = partial (e.g. some participants or
   permutations missing), red = missing, grey = N/A (step 0 for humans) or
   unknown (probe inputs unavailable, e.g. disk not mounted).
4. **Drill in.** Press **Details** on a step to see the per-participant breakdown
   (which `sub-NN` passed/failed) and, when a scheduler job failed, the recorded
   error message and log path — the "why" a participant couldn't pass.

## Memory

Results are remembered per **parameter set** in a per-user cache file:

```
~/.rsa_pipeline_dashboard_cache.json
```

- Once a step is checked, its verdict and detail are stored and shown on the next
  launch — no re-scan needed.
- The cache is keyed by the full parameter signature, so switching `rsa_model`,
  `method`, `z_threshold`, etc. shows the remembered results for *that* set (or
  "NOT CHECKED" if you've never checked it).
- The cache lives in your home folder (not on the shared data disk), so two
  machines don't overwrite each other's view.

### Clearing memory (when you redo a step)

- **Clear** on a step removes just that step's remembered result, so the next
  **Check** re-scans it. Use this after you recompute a step.
- **🗑 Clear all (this parameter set)** forgets every step for the current
  parameters.

## `mah_fold` — telling mahalanobis folds apart

`mah_fold` is the Mahalanobis folding strategy (`searchlight.py --mah_fold`):
`stim-wise` (default), `stim-wise-multiple-folds`, `stim-wise-all-runs`,
`run-wise`. It only matters when `method = mahalanobis`, but there it matters a
lot for the status check:

- **Step 1 (pairwise similarity) is model-independent and shared** — every
  mahalanobis model drops its per-pair maps (`r-{radius}_mahalanobis_{catA}_{catB}.nii.gz`)
  into the *same* `{specie}-sub-NN/` folder. A model with few categories
  (`agent-species-id`) and one with many (`all-categories_bipolar`) sit side by
  side. The dashboard now counts **only the pairs the selected model's own
  categories imply**, instead of globbing `r-{radius}_mahalanobis_*` — so one
  model's (or one fold's) files are never credited to another. This is what
  stopped step 1 falsely reading DONE for every model.
- `mah_fold` also decides *where* those maps live and *how many* there are:
  - `stim-wise` — one map per model-category pair, **directly** under the
    subject folder. `C(n_categories, 2)`, run-independent.
  - `run-wise` — one map per *stim-type* pair, directly under the subject
    folder. Run-independent.
  - `stim-wise-multiple-folds` — one map per model-category pair **per run**, in
    each `ses-*_run-*` subfolder → `C(n_categories, 2) × n_runs`.
  - `stim-wise-all-runs` — one map per pair of *that run's* stimuli (from
    `config['model_dict']`) per run.

  The probe follows the selected fold's layout and counts only the exact
  expected filenames, so leftover/garbage files from other runs are ignored.
- **Expected counts are per participant.** The number of runs is read from
  `{dataset}/BIDS/{specie}_database-details.csv` (`get_session_and_run_dict`) —
  the same source `searchlight.py` loops over to create the files — so a
  participant with 6 runs and one with 7 get different expected totals for the
  per-run folds and the correlation method, instead of one run-blind number for
  everyone.
- `mah_fold` is part of the **cache signature**, so checking the same model under
  two different folds keeps two independent verdicts instead of overwriting one.

**Worked example.** `agent-species-id` was run `stim-wise` (40 categories →
`C(40,2)=780` maps directly under each subject, run-independent).
`all-categories_bipolar` was run `stim-wise-multiple-folds` (10 categories →
`C(10,2)=45` maps per run): a 6-run dog expects `45×6=270` maps across its run
subfolders, a 7-run dog expects `45×7=315`. Select the matching `mah_fold` for
each model or its status will look wrong.

Scheduling (**Sched missing** / **Sched from here** / per-map **Schedule**) also
threads the selected `mah_fold` through to `searchlight.py`, replacing the old
hard-coded `stim-wise`.

## Important: parameters must match the run

The pipeline encodes `method`, `mah_fold`, `rsa_method`, `radius`, `mask_type`,
and `z_threshold` into the output filenames (or their layout). If the dashboard's
parameters don't match how the model was actually run, a completed step will look
**missing**. The defaults mirror `searchlight.py`'s defaults (`mahalanobis` /
`stim-wise` / `kendall` / `b_GreyMatter2mmB` / `zt 3.1`).

## Relationship to the other tools

- `tools/pipeline_console.py` — same checks, terminal version (interactive + `--report`).
- `job_status.py` — reports the scheduler's job-queue state (not disk files).

This dashboard checks disk files (source of truth for "did the output get
written") and additionally surfaces the scheduler's recorded failure reason.

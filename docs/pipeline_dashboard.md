# `pipeline_dashboard.py` — RSA pipeline status dashboard (Dash)

A browser dashboard that shows how far each RSA model has progressed through the
pipeline (steps 0–10) by checking the actual output files on disk. It is the web
version of `pipeline_console.py` and reuses the same file-probing logic.

## Run it

```bash
python pipeline_dashboard.py                 # serves http://127.0.0.1:8060
python pipeline_dashboard.py --port 8062     # choose a different port
RSA_DASHBOARD_PORT=8062 python pipeline_dashboard.py
```

On Windows use the full interpreter path:

```powershell
& "C:\ProgramData\anaconda3\python.exe" pipeline_dashboard.py
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
   `method`, `rsa_method`, `radius` (blank = auto: 3 dog / 4 human),
   `z_threshold`, `mask_type`, `reps`, `reps_group`. Press **⟳ reload models**
   if you add a new model CSV while the app is open.
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

## Important: parameters must match the run

The pipeline encodes `method`, `rsa_method`, `radius`, `mask_type`, and
`z_threshold` into the output filenames. If the dashboard's parameters don't
match how the model was actually run, a completed step will look **missing**. The
defaults mirror `searchlight.py`'s defaults (`mahalanobis` / `kendall` /
`b_GreyMatter2mmB` / `zt 3.1`).

## Relationship to the other tools

- `pipeline_console.py` — same checks, terminal version (interactive + `--report`).
- `job_status.py` — reports the scheduler's job-queue state (not disk files).

This dashboard checks disk files (source of truth for "did the output get
written") and additionally surfaces the scheduler's recorded failure reason.

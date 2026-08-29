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
   (dropdown, filtered by `dis_method` — see below), `specie`,
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

## Full verbose mode — diagnosing file-detection mismatches

If a step's verdict looks wrong (e.g. files you know exist show as MISSING, or
vice versa), tick **🔍 Full verbose mode** next to **▶ Check all steps**. With it
on, every subsequent **Check** / **Check all steps** / **Sched missing** prints
each file the probe actually checked to the terminal the app was launched from
(not the browser), tagged:

- `FOUND` — this exact file exists on disk.
- `MISSING` — this exact file was expected but doesn't exist.
- `PATTERN` — a glob pattern that was searched instead of an exact name, for
  steps whose filenames can't be predicted in advance (step 0's `pe*.nii.gz`,
  and steps 1/2/4's permutation-indexed fallbacks). The `FOUND` lines right
  after a `PATTERN` line are what that pattern matched.

Each step's block ends with a found/missing total. This is a diagnostic aid —
it doesn't change any verdict, cache entry, or scheduling behaviour; it only
adds console output. Leave it off for routine checks, since it can print
hundreds of lines for a step with many participants/pairs (step 1 especially).

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

### ⟳ Live — following the cache file

The table is drawn from the cache file, and the dashboard is not its only
writer: [`tools/bulk_check.py`](../tools/bulk_check.py) running in a terminal, a
second dashboard on another port, or a copy of the file from the other machine
all write the same file. Whether the page notices is a switch, **⟳ Live**, next
to the verbose toggle. The line under the buttons always says which mode you are
in.

**Off** (the default, remembered per browser):

```
⏸ live off — the table shows the cache as of your last check, parameter change or reload
```

The page behaves exactly as it always has — it re-reads the cache only when you
press a button or change a parameter. The poll interval is `disabled`, so the
page makes no requests at all while it sits there.

**On**:

```
⟳ live — showing the cache file as written at 2026-08-25 07:30:11
```

The page re-stats the file every 3 s (`CACHE_POLL_MS`) and redraws only when its
modification time actually moved, so a bulk check fills the table in front of
you. Switching it on also picks up whatever was written while it was off.

Either way it does **not** re-probe anything: the poll is a local `stat` on a
file in your home folder, never a scan of the data disk. Job completions still
do not change the table until something checks the disk again.

### Bulk-filling the cache

Checking a 50-model battery by hand is one button press per model per step.
[`tools/bulk_check.py`](../tools/bulk_check.py) runs the same probes over every
model in `_models.csv` and writes the same cache:

```powershell
& "C:\ProgramData\anaconda3\python.exe" tools\bulk_check.py `
    --dataset EmoC --specie D --dis_method correlation --steps 5,7
```

Switch **⟳ Live** on and leave the dashboard open while it runs — the table
fills in as it goes. Every
parameter it does not take from you defaults to the dashboard's own defaults,
because a cached result is keyed by the *full* parameter signature: change
`z_threshold` in the panel and the page reads a different entry.

### Steps 0 and 1 are shared between models (⇄)

Two steps do **not** write into the rsa_model's own folder, so their result does
not depend on which model is selected — the table marks them `⇄ shared`:

| Step | Files | Shared over |
|---|---|---|
| 0 — beta maps | `GLM/{model}/{specie}-sub-NN/…/stats/pe*.nii.gz` | `dataset`, GLM `model`, `specie` |
| 1 — pairwise similarity | `RSA/{model}/{specie}-sub-NN/r-{radius}_{dis_method}_{a}_{b}.nii.gz` | the above **+** `dis_method`, `radius`, and (mahalanobis only) `mah_fold` |

Checking one of them once fills it in for **every** other model that would look
for exactly the same files, so you don't re-scan the same 15 subject folders
once per model. `rsa_method`, `mask_type`, `z_threshold`, `reps` and
`reps_group` appear nowhere in those filenames, so they never split the shared
result either.

One caveat, and it is why the key is not simply "everything except rsa_model":
under `mahalanobis` + the `stim-wise` fold the pairwise maps are named after the
**model's categories** (`searchlight.py` passes `categories=rsa_model_dict['categories']`),
so two models with different category sets expect different files. For that
fold the shared key therefore carries a fingerprint of the model's category set,
and the result is shared only among models over the same categories. (For the
whole EmoC battery that is one group: all 50 stim-wise models use the same 10
categories, so step 1 is checked once for all of them.) If the model CSV can't
be read, the step falls back to being cached per parameter set, so a model whose
categories are unknown never inherits another model's verdict.

The detail panel names the model the shared scan was run under ("Checked while
rsa_model=… was selected").

### Clearing memory (when you redo a step)

- **Clear** on a step removes just that step's remembered result, so the next
  **Check** re-scans it. Use this after you recompute a step. On a `⇄ shared`
  step this forgets it for every model it covers — which is what you want after
  recomputing pairwise maps.
- **🗑 Clear all (this parameter set)** forgets every step for the current
  parameters, including the shared steps 0 and 1.

Entries written before sharing existed (one step-1 result per full parameter
set) are still read as a fallback, so nothing already cached was lost — the
first re-**Check** of a step promotes it to the shared key.

## Which models the `rsa_model` menu offers

The menu is driven by the dataset's central manifest,
`{datafolder}/{dataset}/rsa_models/_models.csv`, read through
[`models_manifest.py`](../tools/models_manifest.py). It lists **only models that
manifest classifies under the parameters you have selected** — never a loose scan
of the folder — intersected with the model CSVs that actually exist on disk:

| Selected `dis_method` | The menu offers |
|---|---|
| `mahalanobis` | that method's models **for the selected `mah_fold`** (fold dropdown enabled, listing that method's folds) |
| anything else | that method's models, **all folds** (fold dropdown greyed out — there the fold decides nothing about which models exist) |
| a method with no rows in the manifest | nothing; the `dis_method` menu marks which methods the manifest knows with *— in \_models.csv* |

The distance method is the filter that matters, because **a fold name is not
unique across methods**: EmoC classifies its 41 correlation models as `run-wise`
and its 50 mahalanobis models as `stim-wise`. Filtering by fold alone (what the
dashboard did until v2.5.0) mixed the two together — selecting `correlation`
offered all 91 models, and `mahalanobis` + `run-wise` offered the correlation
ones. The status line under the buttons reports what the filter produced, e.g.
`50 model(s) from _models.csv — mahalanobis / stim-wise`.

Press **⟳ reload models** after editing `_models.csv` or adding a model CSV; it
re-reads the manifest and the on-disk file list.
[`tools/bulk_check.py`](../tools/bulk_check.py) selects models exactly the same
way (`--dis_method`, optional `--mah_fold`), so the CLI and the page always agree
on what exists.

## `mah_fold` — telling mahalanobis folds apart

`mah_fold` is the Mahalanobis folding strategy (`searchlight.py --mah_fold`):
`stim-wise` (default), `stim-wise-multiple-folds`, `stim-wise-all-runs`,
`run-wise`. It only matters when `method = mahalanobis`, but there it matters a
lot for the status check:

- **Step 1 (pairwise similarity) is model-independent and shared** — most
  mahalanobis folds write per-pair maps (`r-{radius}_mahalanobis_{catA}_{catB}.nii.gz`)
  directly into `{specie}-sub-NN/`; `stim-wise-all-runs` instead writes into
  each run folder. The dashboard counts **only the pairs the selected model and
  fold imply**, instead of globbing `r-{radius}_mahalanobis_*`, so one model's
  (or one fold's) files are never credited to another.
- `mah_fold` also decides *where* those maps live and *how many* there are:
  - `stim-wise` — one map per model-category pair, **directly** under the
    subject folder. `C(n_categories, 2)`, run-independent.
  - `run-wise` — one map per *stim-type* pair, directly under the subject
    folder. Run-independent.
  - `stim-wise-multiple-folds` — one map per pair of exact repeatable EmoC
    `config['model_dict']` `stim_file` conditions, directly under the subject
    folder. Stimuli without two common metadata partitions are excluded.
  - `stim-wise-all-runs` — EmoC-only; one map per pair of *that run's* classes.
    It collapses exemplars such as `DogA1`--`DogA4` to `DogA`, uses the final
    exemplar IDs as within-run cross-validation folds, and writes maps in the
    run folder (for example, `r-{radius}_mahalanobis_DogA_DogH.nii.gz`).

  The probe follows the selected fold's layout and counts only the exact
  expected filenames, so leftover/garbage files from other runs are ignored.
- **Steps 2 and 4 (model comparison and its randomized versions) are also
  fold-specific.** `stim-wise` retains its legacy direct-subject location under
  `{rsa_model}/{specie}-sub-NN/`. To prevent a run with another fold from
  overwriting it, `stim-wise-multiple-folds` writes under
  `{rsa_model}/stim-wise-multiple-folds/{specie}-sub-NN/`; `stim-wise-all-runs`
  uses its corresponding fold directory and then each run folder. The
  dashboard expects one real map and `--reps` randomized maps for each of those
  folders. Steps 3 and 5 aggregate these inputs into the unchanged canonical
  group `mean/` paths consumed by steps 6--10.
- **Expected counts are per participant.** The participant run list is read from
  `{dataset}/BIDS/{specie}_database-details.csv` (`get_session_and_run_dict`) —
  the same source `searchlight.py` loops over to create the files. For
  `stim-wise-multiple-folds`, the probe derives the exact repeatable stimuli and
  their common partitions from that run list. For `stim-wise-all-runs`, it
  derives each run's class labels from those same condition keys.
- `mah_fold` is part of the **cache signature**, so checking the same model under
  two different folds keeps two independent verdicts instead of overwriting one.

**Worked example.** `agent-species-id` run with `stim-wise` (40 categories →
`C(40,2)=780` maps) writes directly under each subject. For EmoC
`stim-wise-multiple-folds`, `R1DogP1` and `R1DogP2` each occur in partitions 1
and 2, so their direct-subject map is
`r-{radius}_mahalanobis_R1DogP1_R1DogP2.nii.gz`. Select the matching
`mah_fold` for each model or its status will look wrong.

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

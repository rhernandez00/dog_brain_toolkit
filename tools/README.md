# tools/

Standalone apps and one-off scripts built on top of the core `dog_brain_toolkit`
pipeline (`rsa_utils.py`, `searchlight.py`, `scheduler/`, `rsa_model_builder.py`,
`viz/`, etc.). Everything here is a *consumer* of the core toolkit — nothing in
the core toolkit imports from `tools/`.

Each script still lives directly in `tools/` (no subpackage) and adds both its
own folder and the repo root to `sys.path` at import time, so it can do a plain
`import rsa_utils`, `from scheduler.paths import ...`, `from viz import ...`, or
`import rsa_model_builder` exactly as it did when it lived at the repo root.
Run every script with the full Anaconda interpreter path from the repo root
(see `CLAUDE.md`), e.g.:

```powershell
& "C:\ProgramData\anaconda3\python.exe" tools\pipeline_dashboard.py
```

## Tools

| Tool | What it is |
|---|---|
| [`dashboard.py`](dashboard.py) | Unified entry point — one Flask server (port 8050) that mounts the Brain Viewer, RSA Model Builder, Scheduler, and Planner sub-apps under one URL with a tabbed landing page. Useful for sharing everything through a single tunnel. |
| [`pipeline_dashboard.py`](pipeline_dashboard.py) | Browser dashboard (Dash, port 8060) showing how far each RSA model has progressed through pipeline steps 0–10, by checking output files on disk. Editable run parameters, on-demand checks, per-parameter-set result cache. The **dis_method** (dropdown: pearson/kendall/euclidean/mahalanobis/correlation), **mah_fold**, and **rsa_model** menus are driven by the central `rsa_models/_models.csv` manifest — the rsa_model list shows **only** models in that manifest. `mahalanobis` is the special case that uses **mah_fold**: it filters the model list to the selected fold's models (and enables the fold dropdown); any other distance method drops the fold filter, greys out mah_fold, and offers every manifest model. The selected model's **why** note is shown. Reuses the probe logic in `pipeline_console.py`. Docs: [`../docs/pipeline_dashboard.md`](../docs/pipeline_dashboard.md). |
| [`pipeline_console.py`](pipeline_console.py) | Terminal version of the same progress probe — read-only, interactive menu or a non-interactive `--report` mode for scripting. Docs: [`../docs/pipeline_console.md`](../docs/pipeline_console.md). |
| [`hypothesis_explorer.py`](hypothesis_explorer.py) | Standalone Dash app (port 8055): a row of self-contained **model cards** (no hypothesis tree). Each card is one RSA model — first pick a **Mahalanobis fold** (`mah_fold`: stim-wise / run-wise / …), which decides the available **models** (hypothesis *stems*) and **groupings** (all/collapse/within/cross/dog/hum); pick a stem and a grouping and they resolve to a concrete `{stem}__{grouping}` model (or a suffix-less model when that is the only file, e.g. `agent-species-id`). The fold → model → grouping menu and each model's **why** note come from the dataset's central [`rsa_models/_models.csv`](../tools/models_manifest.py) manifest (built by `build_models_manifest.py`); when it is absent the card falls back to **scanning the `rsa_models` folder(s)** and offering every valid `__{grouping}` model under one synthetic fold. It searches both the active results root and the pipeline data disk (`P:\userdata\raulh87\data\...\rsa_models`). Each card is **one species** (its own **column** — model 1 → column 1, model 2 → column 2, …): the **Species** control picks **Dog** or **Human** and the card draws that species' results map as a 2D atlas slice (put Dog and Human side by side in two cards). The map type defaults to the group **mean** (switchable to z-map / cluster-corrected); axis / slice / z-threshold / **colormap** / an optional **max** (scale ceiling) are per-card. The colormap defaults to **Hot**: voxels below the threshold render transparent (alpha=0), everything at/above rides the hot scale. Toggle **🔗 sync** to mirror the view (slice, axis, threshold, max, colormap) across all *other synced cards of the same species* — move the slice on one and the matching-species cards follow, scales included; Dog and Human sync independently. The card also shows the model's **dissimilarity matrix** (RSA Model Builder view); toggle **show matrix** off to hide it. A per-card status dot flags whether computed results exist (dog / human / both / none). Maps can be read from **either** source: **Drive (current-results)** flat mirror, or **Raw (results/RSA)** nested pipeline output (`.../results/RSA/{glm_model}/{model}/mean/`, same paths `pipeline_dashboard.py` probes) — switch via the *Result source* menu; it auto-seeds the data folder. Use **➕ Add model** / per-card ✕ remove (up to 6 slots), and an **✏️ Edit** toggle: while on, drag a card's header to **reorder** it (cards are not resizable; a **Gap** box sets spacing and **Reset order** restores the default). Brain-view height is adjustable; the source mode / data folder / dataset / view height / card layout are saved to `~/.rsa_hypothesis_explorer_settings.json`, and each card's own selections (model, grouping, species, map type, axis, colormap, max, sync, on/off) are kept by Dash local persistence, so the cards come back next launch. |
| [`models_manifest.py`](models_manifest.py) | Shared reader for the central `rsa_models/_models.csv` manifest — the single source of truth for **which models exist, which Mahalanobis fold each belongs to, which groupings each offers, and its `why` note**. Both `pipeline_dashboard.py` and `hypothesis_explorer.py` read models through it (fold list, per-fold model/grouping index, concrete-name resolution incl. suffix-less models, `why` lookup), so models are added/edited/re-grouped in **one file**. Tolerant of a missing/stale manifest (callers fall back to scanning the folder). |
| [`build_models_manifest.py`](build_models_manifest.py) | (Re)builds `rsa_models/_models.csv` from the two upstream battery manifests: the **stim-wise** `model_manifest_EmoC_RSA_model_battery.csv` (`why` ← its `why_test`, groupings ← `scope`) and the **run-wise** `_MODEL_BATTERY_MANIFEST.csv` (`why` ← `description` minus the grouping clause, groupings ← `grouping`). One row per model family × fold. Run it whenever either upstream manifest changes. |
| [`build_rsa_models.py`](build_rsa_models.py) | Generates the full factorial battery of EmoC RSA model CSVs (41 models) ready to feed into `searchlight.py` / the scheduler. Docs: [`../docs/RSA_model_battery_plan.md`](../docs/RSA_model_battery_plan.md). |
| [`build_all_categories_groupings.py`](build_all_categories_groupings.py) | Derives agent-species grouping variants (`__collapse`, `__cross`, `__dog`, `__hum`) of the hand-built `all-categories_{categorical,bipolar,emotionality}` models. Keeps the source 10×10 values and only NaN-masks the excluded condition pairs, reusing `build_rsa_models`' grouping predicates. Writes 12 CSVs next to the sources in `{datafolder}/EmoC/rsa_models/`. |
| [`set_live.py`](set_live.py) | Writes `docs/live.json` so the GitHub Pages landing page points at the current laptop tunnel URL (cloudflared/ngrok). Run once per session after starting the tunnel, then commit & push `docs/live.json`. |
| [`make_qr.py`](make_qr.py) | Generates a printable QR code (`docs/qr.png`) encoding the stable GitHub Pages landing URL. |
| [`create_package.py`](create_package.py) | Builds a self-contained `.zip` for **one participant × many RSA models** so the compute-heavy steps **1, 2, 4 run on a Colab GPU**. Supports both `--dis_method mahalanobis` (stim-wise, 45 category pairs) and `--dis_method correlation` (run-wise, 780 per-stimulus pairs). Bundles just the needed GLM betas, mask, model CSVs, config, the GPU code and a Colab notebook, plus a `manifest.json`. Checks whether the step-1 pairwise maps already exist on disk and, if so, bundles them so Colab skips step 1. `--all` expands to every model of the chosen `dis_method` in `rsa_models/_models.csv` (`--all-stim-wise` is the mahalanobis alias), via [`models_manifest.py`](models_manifest.py). See [`colab_gpu/README.md`](colab_gpu/README.md). |
| [`unpack_results.py`](unpack_results.py) | Merges the `result_*.zip` files a Colab run writes (step-1 maps, and per-model step-2/step-4 maps) back onto the pipeline data disk. Arc-paths are already pipeline-relative, so it's a validated merge (with `--dry-run` / `--replace` guards); afterwards `searchlight.py` steps 3–10 run as if the maps were computed locally. |
| [`colab_gpu/`](colab_gpu/README.md) | The GPU subsystem: PyTorch reimplementation of RSA steps 1/2/4 (`gpu_rsa.py`), the Colab orchestrator/notebook (`run_colab.py`, `colab_rsa.ipynb`), and a CPU-vs-GPU fidelity harness (`validate_gpu.py`, validated to ~1e-12). |

## Adding a new tool

New standalone tools/scripts belong in this folder, not the repo root. When you
add one:

1. Put the script (and any folder it creates for its own data/cache, if that
   data isn't already going to the shared network disk) inside `tools/`.
2. Add both the repo root and `tools/` itself to `sys.path` at the top, following
   the pattern already used by the scripts above, so imports of core modules
   (`rsa_utils`, `scheduler.*`, `viz.*`, `rsa_model_builder`) keep resolving.
3. Add a row for it to the table above.

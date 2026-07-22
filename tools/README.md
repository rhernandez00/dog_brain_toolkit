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
| [`pipeline_dashboard.py`](pipeline_dashboard.py) | Browser dashboard (Dash, port 8060) showing how far each RSA model has progressed through pipeline steps 0–10, by checking output files on disk. Editable run parameters, on-demand checks, per-parameter-set result cache. Reuses the probe logic in `pipeline_console.py`. Docs: [`../docs/pipeline_dashboard.md`](../docs/pipeline_dashboard.md). |
| [`pipeline_console.py`](pipeline_console.py) | Terminal version of the same progress probe — read-only, interactive menu or a non-interactive `--report` mode for scripting. Docs: [`../docs/pipeline_console.md`](../docs/pipeline_console.md). |
| [`hypothesis_explorer.py`](hypothesis_explorer.py) | Standalone Dash app (port 8055) for authoring an RSA hypothesis tree and browsing linked 2D atlas-slice result panels side by side (dog / human / both). |
| [`build_rsa_models.py`](build_rsa_models.py) | Generates the full factorial battery of EmoC RSA model CSVs (41 models) ready to feed into `searchlight.py` / the scheduler. Docs: [`../docs/RSA_model_battery_plan.md`](../docs/RSA_model_battery_plan.md). |
| [`set_live.py`](set_live.py) | Writes `docs/live.json` so the GitHub Pages landing page points at the current laptop tunnel URL (cloudflared/ngrok). Run once per session after starting the tunnel, then commit & push `docs/live.json`. |
| [`make_qr.py`](make_qr.py) | Generates a printable QR code (`docs/qr.png`) encoding the stable GitHub Pages landing URL. |

## Adding a new tool

New standalone tools/scripts belong in this folder, not the repo root. When you
add one:

1. Put the script (and any folder it creates for its own data/cache, if that
   data isn't already going to the shared network disk) inside `tools/`.
2. Add both the repo root and `tools/` itself to `sys.path` at the top, following
   the pattern already used by the scripts above, so imports of core modules
   (`rsa_utils`, `scheduler.*`, `viz.*`, `rsa_model_builder`) keep resolving.
3. Add a row for it to the table above.

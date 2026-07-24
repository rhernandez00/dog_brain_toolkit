# `pipeline_console.py` — RSA progress console

A read-only console that tells you **how far a model has progressed** through the
RSA pipeline by checking the actual output files on disk. It discovers runnable
models from the dataset's `rsa_models/` folder, and for a model you pick it
probes each step (0–10), reports whether that step's files exist, names the
participants that are still missing on the per-participant steps (0, 1, 2, 4), and
surfaces the recorded error from the job queue when a step failed.

It never computes anything — "done" means the files `searchlight.py` would have
written are present.

## Run it

Interactive (menu-driven):

```bash
python tools/pipeline_console.py --dataset EmoC
```

You get a numbered list of the RSA models found in
`{datafolder}/{dataset}/rsa_models/`. Pick one by number to see the step table;
type `.7` to drill into step 7 (per-participant detail + failure reasons); `s` to
change species/method/threshold/etc.; `d` to switch dataset; `r` to reprint the
report; `q` to quit.

One-shot report (non-interactive, scriptable):

```bash
python tools/pipeline_console.py --dataset EmoC --rsa_model test-model --specie D --report
```

Single-step detail:

```bash
python tools/pipeline_console.py --dataset EmoC --rsa_model test-model --specie D --report --step 5
```

## Options

| Option | Default | Notes |
|---|---|---|
| `--dataset` | (required) | Dataset name. |
| `--model` | `basic-block` | GLM model. |
| `--rsa_model` | picker | RSA model CSV name; omit for the interactive picker. |
| `--specie` | `D` | `D` or `H`. |
| `--method` | `mahalanobis` | Pairwise method — part of every filename. |
| `--mah_fold` | `stim-wise` | Mahalanobis folding (`stim-wise`, `stim-wise-multiple-folds`, `stim-wise-all-runs`, `run-wise`). Sets which/where the step-1 pairwise maps are expected, so models/folds sharing a subject folder aren't mixed. |
| `--rsa_method` | `kendall` | Model-comparison method. |
| `--radius` | 3 (D) / 4 (H) | Searchlight radius. |
| `--z_threshold` | `3.1` | Threshold baked into step 9/10 filenames. |
| `--mask_type` | `b_GreyMatter2mmB` | Mask selector; use `none` for the no-mask naming. |
| `--reps` / `--reps_group` | `100` / `1000` | Expected permutation counts (steps 4/5). |
| `--report` / `--step` | off | Non-interactive report; optional single step. |

> **Important:** the parameters must match how the model was actually run — the
> pipeline encodes `method`, `mah_fold`, `rsa_method`, `radius`, `mask_type`, and
> `z_threshold` into the output filenames (or their on-disk layout), so a
> mismatch makes a completed step look missing. The defaults mirror
> `searchlight.py`'s defaults.

## What each step checks

| Step | Verdict source |
|---|---|
| 0 Beta maps (D only) | GLM `pe*.nii.gz` per participant/run |
| 1 Pairwise similarity | per-participant maps (`r-{radius}_{method}_*`) |
| 2 Model similarity | one map per direct-subject fold result, or one per run for `stim-wise-all-runs` |
| 3 Group similarity map | `mean/{...}_mean.nii.gz` |
| 4 RND permuted model | `--reps` permutations per direct result, or per run for `stim-wise-all-runs` |
| 5 RND group permutations | count of `RSA_rnd/.../mean_{NNNNN}.nii.gz` vs `--reps_group` |
| 6 Voxelwise RND distribution | `RSA_rnd/{model}/{specie}-{rsa_model}_mean/_std.nii.gz` |
| 7 Z-maps | `mean/{...}_z.nii.gz` |
| 8 Cluster size distribution | `dist/{...}_dist.npy` |
| 9 Cluster correction | `mean/{...}_zt{z}_corrected.nii.gz` |
| 10 Create tables | `mean/{...}_zt{z}.xlsx` / `.csv` |

Verdicts: **DONE** (all expected files present), **PARTIAL** (some present —
e.g. 22/40 participants, or 640/1000 permutations), **MISSING** (none),
**N/A** (step 0 for humans), **UNKNOWN** (probe inputs unavailable, e.g. the
data disk is not mounted).

The per-participant "expected" counts read the participant list from the config
YAML and the session/run layout via `rsa_utils.get_session_and_run_dict`, so the
data disk must be mounted for those steps to move beyond UNKNOWN. The
group/final steps only need the result files themselves.

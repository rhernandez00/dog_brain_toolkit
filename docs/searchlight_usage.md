# `searchlight.py` — RSA Pipeline Usage Guide

Step **16.2** computes a compact dog-by-human searchlight Kendall tau-b matrix
from existing Mahalanobis stim-wise maps. It supports the local CLI and a GPU
Colab notebook; see [step 16.2](cross_species_rsa.md) for inputs and usage.

`searchlight.py` is the command-line runner for the Representational Similarity
Analysis (RSA) pipeline. It processes fMRI BOLD data from **dogs (`D`)** and
**humans (`H`)** through a sequence of numbered steps, ending in thresholded,
cluster-corrected z-maps, a cluster CSV, and publication tables and Results text.

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

### Export a coordinate as a run-dependent RSA model (step 12.1)

After step 1 has produced the pairwise searchlight maps, use:

```powershell
& "C:\ProgramData\anaconda3\python.exe" searchlight.py `
    --dataset EmoB --model basic --specie H --steps_to_run 12.1 `
    --rsa_model visual1 --output_rsa_model H-roi-example `
    --coords 40,50,30 --radius 4 --dis_method correlation
```

Replace the example coordinates with your voxel indices and use the GLM model,
radius, distance method and `--mah_fold` used to generate the maps. Coordinates
are zero-based voxel indices in the mask/map grid, not millimeters. As in step
12, extraction samples the existing searchlight value at that center; it does
not average neighboring searchlight centers again.

Alternatively, replace `--coords 40,50,30` with `--coords_mm=10,-20,30` to
provide world coordinates in millimeters. The options are mutually exclusive.
Use the `=` form when the first coordinate is negative, e.g.
`--coords_mm=-10,-20,30`. Conversion uses the selected mask's inverse affine
(including orientation, voxel size and origin) and rounds to the nearest voxel
with NumPy's nearest-even rule for exact half-voxel ties. The step prints both
the zero-based voxel indices and their mm coordinates. With mm input, it also
prints the requested mm coordinates so rounding is visible. Coordinates must
be in the mask's world space (e.g. the appropriate human or dog template).

`--rsa_model` supplies stimulus labels and pair ordering from an existing CSV
(or its `-run-N.csv` variants). `--output_rsa_model` is a new filename stem.
The example writes `EmoB/rsa_models/H-roi-example-run-1.csv` through
`H-roi-example-run-4.csv` when the participant metadata contains runs 1–4.
Each CSV is a symmetric labeled matrix with a zero diagonal, readable later
with `--rsa_model H-roi-example`. Values retain the input maps' distance or
similarity convention, including negative values.

To include only selected stimuli, add `--stimuli A-1 A-2 H-1 H-2`.
Use exact labels from the input model, separated by spaces (quote labels that
contain spaces). Only pairs among these stimuli are loaded and exported, in
the order you specify. Provide at least two distinct labels; each must exist
in every run's input model. Omitting `--stimuli` retains all stimuli.

Configured participants of `--specie` with a session for a run contribute equally.
Repeated sessions for a participant/run are averaged before averaging across
participants. Participants with no session for that run are excluded from its
average; a warning lists their IDs, the run number, and the contributor count.
Runs are taken from the union of the selected participants' session/run metadata.
`--participants_forced` explicitly restricts the cohort. Missing maps for listed
sessions, non-finite pair values, and coordinates outside the mask are errors;
all contributing maps are validated before any CSV is written. Use `--replace_file` to
overwrite existing models (also invalidates their cached `.npy` model data).
Step 12 does not need to be run first.

With `--dis_method mahalanobis --mah_fold stim-wise`, step 12.1 writes a single
`<output_rsa_model>.csv` in the same `rsa_models` directory. It loads each
selected participant's map once and averages participants equally, independently
of their run/session metadata. Labels come from `<rsa_model>.csv`, falling back
to `<rsa_model>-run-1.csv`; `--stimuli` still selects a subset. Missing participant
maps remain errors. Existing run-specific output files are not deleted.

Other folding modes retain their current output naming. Mahalanobis modes other
than `stim-wise` and `stim-wise-all-runs` also share maps across runs and may
produce identical run CSVs; step 12.1 warns about this.

Add `--individual` to export separate participant models instead of a group
average. In this mode, participants and sessions/runs come from the unique
`sub_N` entries in `BIDS/{specie}_database-details.csv`, rather than the config's
participant list. `--participants_forced` restricts these manifest participants;
IDs absent from the manifest are errors.

For example, `--output_rsa_model LcSSG_mah --specie D --dis_method mahalanobis
--mah_fold stim-wise --stimuli A H C --individual` writes
`rsa_models/LcSSG_mah_D-sub-01.csv`, `LcSSG_mah_D-sub-02.csv`, etc.
For run-specific exports, the names include both identifiers:
`LcSSG_mah_D-sub-01-run-1.csv`. Repeated sessions are averaged within each
participant/run; participants are never averaged together. Absent runs are
skipped with a warning. Missing maps for expected inputs remain errors.
Coordinate options, stimulus selection, and `--replace_file` work as before.

### Average maps across individual source-species models (steps 3 and 5)

Use `--model_specie` when `--rsa_model` is a prefix for individual models. For
example:

```powershell
& "C:\ProgramData\anaconda3\python.exe" searchlight.py `
    --dataset EmoB --model basic-block --specie H --steps_to_run 3 `
    --rsa_model LcSSG_mah --dis_method mahalanobis `
    --mah_fold stim-wise --model_specie D
```

This expects `LcSSG_mah_D-sub-01.csv`, `LcSSG_mah_D-sub-02.csv`, and one
individual model for every D participant in the BIDS manifest. A missing model
CSV is an error. Step 3 reads each model's H
participant maps from `results/RSA/basic-block/<individual-model>/H-sub-XX/`
and averages every available source-model × H-participant map into one H mean
and standard deviation under `results/RSA/basic-block/LcSSG_mah_D/mean/`. Its
JSON log records the exact source models and input maps. `--specie` names the
species whose brain maps are averaged; `--model_specie` names the species in
the individual model filenames. The base model CSV is not required.

Step 5 with the same options reads the permutation maps under each individual
model's `results/RSA_rnd/` folder. For each group permutation it samples one
available participant permutation map for every source-model × H-participant
unit and writes a combined map under `results/RSA_rnd/basic-block/LcSSG_mah_D/mean/`.
The usual `--reps` and `--reps_group` options apply. Both steps honor
`--min_percentage_available` over the full set of expected maps; its default
of `1.0` requires every source-model × participant unit. Step 5 checks which
permutation maps actually exist when `--model_specie` is set. Step 3 will not
overwrite an existing combined map unless `--replace_file` is supplied.

Without `--model_specie`, steps 3 and 5 retain their usual single-model behavior.
The option is accepted only for these two steps.

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
| **4** | Permuted model similarity (RND) | `compare_with_model2(rnd=True)` | Controlled by `--reps` (default 100); visits output indices in random order and skips existing maps unless `--replace_rnd_files` is set. |
| **5** | Group permutations (RND) | `calculate_group_model_similarity_map_rnd` | Controlled by `--reps_group` (default 1000). |
| **6** | Voxelwise RND distribution | `calculate_voxelwise_rnd_distribution` | Per-voxel mean & std of the null. |
| **7** | Z-maps | `calculate_z_maps_rnd` + `calculate_z_map_real_data` | Requires steps **3 and 6**. |
| **7.5** (`75`) | Z-map for real data only | `calculate_z_map_real_data` | Optional; run the real-data z-map separately. |
| **8** | Cluster size distribution | `calculate_cluster_size_distribution` | Controlled by `--z_threshold`. |
| **9** | Cluster correction | `apply_cluster_correction` | Requires steps **7 and 8**. |
| **10** | Publication reports | `create_tables` | CSV, editable Word, LaTeX, Results text and provenance JSON. |
| **11** | Cross-participant similarity | `calculate_cross_participant_similarity` | Out of scheduler scope. |
| **11.2** | Participant leave-one-out similarity | `calculate_leave_one_out_cross_participant_similarity` | Correlates each participant's pairwise maps with the average of all other participants. |
| **11.3** | Average leave-one-out similarity | `calculate_mean_leave_one_out_cross_participant_similarity_map` | Averages step 11.2 with equal participant weight; writes mean/std and JSON provenance. |
| **11.13** | Paired class difference | `calculate_group_level_difference` | Subtracts `--target_class` from `--rsa_class` in each participant's step-11.2 maps, then averages participants equally. Writes mean/std and JSON under `class_<rsa_class>/leave-one-out/[run-wise/]class_<target_class>/mean/`. Requires complete maps for both classes; `--min_percentage_available` controls participant completeness. |
| **11.15** | Permuted paired class difference | `calculate_group_level_difference_rnd` | Independently samples one available participant permutation per class from `RSA_same_rnd` (step 11.4), subtracts target from source within participant, and averages participants equally. Repeats `--reps_group` times; writes mean/std and sampled-file JSON under the step-11.13 hierarchy in `RSA_same_rnd`, with five-digit group indices. Reuses complete outputs unless `--replace_rnd_files` is set. `--shuffle_participants` randomizes the order of group permutation output indices for concurrent instances, preserving participant order and existing completeness checks. |
| **11.4** | Permuted participant leave-one-out similarity | `calculate_leave_one_out_cross_participant_similarity` (`rnd=True`) | Shuffles held-out participant category labels for each of `--reps` maps; reference participants stay fixed. |
| **11.5** | Permuted group leave-one-out similarity | `calculate_mean_leave_one_out_cross_participant_similarity_map_rnd` | Samples step 11.4 maps for `--reps_group` group mean/std maps. |
| **11.6** | Leave-one-out voxelwise null | `calculate_leave_one_out_inference` | Population mean/std across step 11.5 group means. |
| **11.7** | Leave-one-out z-maps | `calculate_leave_one_out_inference` | Z-scores the real 11.3 mean and permuted 11.5 means using 11.6. |
| **11.8** | Leave-one-out cluster null | `calculate_leave_one_out_inference` | Positive-tail, 26-neighbour cluster sizes from permuted z-maps. |
| **11.9** | Leave-one-out cluster correction | `calculate_leave_one_out_inference` | Applies the empirical maximum-cluster cutoff to the real z-map. |
| **11.10 / 11.101** | Leave-one-out reports | `calculate_leave_one_out_inference` | CSV and publication reports, including empty results. |
| **11.16** | Class-difference voxelwise null | `calculate_leave_one_out_inference` | Population mean/std across step-11.15 group difference means. |
| **11.17** | Class-difference z-maps | `calculate_leave_one_out_inference` | Z-scores the real 11.13 difference and permuted 11.15 differences using 11.16. |
| **11.18** | Class-difference cluster null | `calculate_leave_one_out_inference` | Separate positive and negative 26-neighbour clusters; one maximum across both signs per recorded permutation. |
| **11.19** | Class-difference correction | `calculate_leave_one_out_inference` | Applies the combined two-sided maximum-cluster cutoff, preserving signed z values. |
| **11.20** | Class-difference reports | `calculate_leave_one_out_inference` | CSV, Word where supported, LaTeX, Results text and provenance for both directions, including empty results. |
| **12** | DSM extraction | `calculate_similarity_across_all_pairs` | Per significant cluster; uses `roi_database.csv`. |
| **13** | Movement (`.par`) parameters | `calculate_movement_parameters` + `preprocess_functions.fwd` | Runs mcflirt to derive framewise displacement. |
| **15** | Multiple regression RSA | `calculate_multiple_regression_rsa` | Target model coefficient controlling for `--regression_model`; needs step 1. |
| **15.3** | Mean regression beta maps | `calculate_group_regression_maps` | Mean/std of step-15 beta maps. |
| **15.4** | Permuted regression RSA | `calculate_multiple_regression_rsa(rnd=True)` | `--reps` target-model permutations; neural data and controls stay fixed. |
| **15.5** | Group regression permutations | `calculate_group_regression_maps(rnd=True)` | `--reps_group` mean/std maps sampled from step 15.4. |
| **15.6** | Regression voxelwise null | `calculate_regression_inference` | Mean and population std across step-15.5 **beta mean** maps. |
| **15.7** | Regression z-maps | `calculate_regression_inference` | Standardizes every null mean and the step-15.3 observed beta mean using step 15.6. |
| **15.8** | Regression cluster null | `calculate_regression_inference` | Positive-tail, 26-neighbour cluster sizes, with a maximum per permutation for correction. |
| **15.9** | Corrected regression z-map | `calculate_regression_inference` | Uses `--z_threshold` and `--cluster_threshold`; writes a correction JSON sidecar. |
| **15.10** | Regression publication reports | `calculate_regression_inference` | CSV, Word, LaTeX, Results text and provenance; valid empty reports if no clusters survive. |

### Class-difference inference (11.16 through 11.20)

Inference receipts use schema version 2 with dataset-relative input/output paths
and forward slashes, so the same dataset can be mounted on Linux or Windows.
Masks are identified by voxel-content hash, shape, and affine rather than a path;
repository atlas masks can live on a separate drive from the dataset. Absolute-path
receipts from earlier versions are not supported: rebuild steps 11.16 through
11.20 (or 11.6 through 11.10 for similarity). Input size and modification-time
checks still apply; copying data must preserve timestamps to reuse receipts.

Run step 11.13 for the observed `rsa_class - target_class` difference and step
11.15 for its group permutations first, using the same participants and analysis
parameters. The inference sequence requires both class arguments and uses
`--reps_group`, `--min_percentage_available`, `--z_threshold` and
`--cluster_threshold`, as in the existing inference steps:

```powershell
& "C:\ProgramData\anaconda3\python.exe" searchlight.py `
  --dataset EmoC --model basic-block --specie H --radius 4 `
  --rsa_class all-mahalanobis --target_class dog `
  --dis_method mahalanobis --rsa_method kendall --mah_fold stim-wise `
  --steps_to_run 11.16 11.17 11.18 11.19 11.20 --reps_group 1000
```

Outputs stay under `class_<rsa_class>/leave-one-out/[run-wise/]class_<target_class>/`:
null moments and their input receipt in `RSA_same_rnd/.../dist/`, permutation
z-maps in `RSA_same_rnd/.../mean/`, cluster nulls in `RSA_same/.../dist/`, and real
z-maps, corrected maps and reports in `RSA_same/.../mean/`. The standard deviation
is across permutation **means**, rather than the within-group standard deviation
files. Class-difference inference tests both tails of `rsa_class - target_class`:
clusters form separately at `z > z_threshold` and `z < -z_threshold` (default
`+3.1` and `-3.1`), using 26-neighbour connectivity. Opposite-sign clusters are
never merged. Each permutation contributes its largest cluster across both
signs to the extent cutoff, controlling the search across both tails together.
Corrected maps retain signed z values, and reports include negative minima as
well as positive maxima. Signs describe deviations from the voxelwise
permutation mean, which need not be exactly zero. Steps 11.8--11.10 remain
positive-tail similarity inference. After updating, rerun steps 11.18--11.20;
legacy positive-only cluster nulls and correction receipts are rejected.
Undefined voxels and zero null variance yield zero in both real and permuted
z-maps. At least two available group permutations are required. Later steps use
the recorded ensemble, reject changed inputs or parameters, and ignore unrelated
leftover permutation files. Pass report step `11.20` literally; the parser keeps
it distinct from participant step `11.2`.

### Leave-one-out cross-participant similarity (11.2 and 11.3)

Step **11.31** repeats participant leave-one-out similarity and equal participant
averaging after excluding each unique `cat1`/`cat2` label in the selected class
CSV from both target and peer RDMs. It supports Mahalanobis and correlation,
preserving pooled or per-run input units. See [the Colab notebook and packager instructions](stimulus_contribution_colab.md).

Create a class containing every available comparison from a GLM config with
`rsa_utils.create_rsa_class(config_path, rsa_class, dis_method, mah_fold='stim-wise')`.
The function writes only `cat1,cat2` columns, with one row per unordered pair,
and returns the saved path. No RSA model is required. A command-line wrapper
is also available:

```powershell
& "C:\ProgramData\anaconda3\python.exe" tools/create_rsa_class.py `
    --config "P:\userdata\raulh87\data\EmoC\config_files\H_basic-block.yaml" `
    --rsa_class all_mahalanobis --dis_method mahalanobis --mah_fold stim-wise
```

This creates `EmoC/rsa_models/by_class/class_all_mahalanobis.csv`: the 45
comparisons among the config's 10 category labels. `stim-wise` is the default
and follows the pipeline's category collapse (EmoC removes the final exemplar
digit; EmoB uses the category before the first hyphen). Correlation uses all
40 `stim_types` in this config, producing 780 comparisons; use
`--rsa_class all_correlation --dis_method correlation` for that class.
Mahalanobis `--mah_fold run-wise` also uses all 40 raw stimulus labels and
produces 780 comparisons, so give that class a separate name such as
`all_mahalanobis_runwise`. Choose a class generated for the same fold as the
pairwise maps used in step 11.2.

Optional `--stimuli` restricts the class to selected available labels.
`--output_dir` overrides the default `rsa_models/by_class` directory;
`--replace_file` allows overwriting. Mahalanobis `stim-wise-multiple-folds`
uses the pipeline's repeated exact-stimulus labels; `stim-wise-all-runs` takes
the union of comparisons that coexist within a run, excluding comparisons
between labels that never occur together.

Run both steps after the pairwise maps from step 1 exist:

```powershell
& "C:\ProgramData\anaconda3\python.exe" searchlight.py `
    --dataset EmoB --model basic-block --specie H `
    --steps_to_run 11.2 11.3 --rsa_class all `
    --dis_method mahalanobis --mah_fold run-wise --rsa_method kendall
```

`--rsa_class` is required. For example, `--rsa_class dog` reads exactly the
comparison rows in `rsa_models/by_class/class_dog.csv`, whose two columns are
`cat1` and `cat2`. The `class_dog` stem is also accepted. These steps ignore
`--rsa_model`: it cannot select pairs or substitute for a missing class. When
running only steps 11.2/11.3, no RSA model file is loaded or validated. For correlation
inputs, replace `--dis_method mahalanobis --mah_fold run-wise` with
`--dis_method correlation`. `--rsa_method` independently selects the RDM
correlation statistic: Kendall tau-b, Spearman, or Pearson.

Step 11.2 correlates a participant's RDM with the arithmetic mean RDM of the
other participants. Mahalanobis run-wise maps already pool runs, so each
participant contributes once. Correlation maps are compared only within the
same run number; repeated sessions are averaged within each reference
participant before averaging the peers. Missing pairwise maps or incompatible
image grids stop the calculation. Absent runs are excluded with a warning;
every run needs at least two participants.

The output root follows the existing cross-participant `results/RSA_same`
hierarchy, with a separate leave-one-out directory:

```text
results/RSA_same/<model>/class_<rsa_class>/leave-one-out/[run-wise/]
  <specie>-sub-NN[_ses-SS_task-TASK_run-RR]/
    <mask_type>-r-R_<dis_method>_<rsa_method>_task-TASK.nii.gz
  <mask_type>-<specie>-r-R_<dis_method>_<rsa_method>_task-TASK_mean.nii.gz
  <mask_type>-<specie>-r-R_<dis_method>_<rsa_method>_task-TASK_std.nii.gz
  <mask_type>-<specie>-r-R_<dis_method>_<rsa_method>_task-TASK_mean.json
```

The `run-wise/` directory applies to Mahalanobis only; session/run suffixes
apply to correlation only. Step 11.3 averages sessions within each run, runs
within each participant, and finally participants equally. It writes the
arithmetic mean and population standard deviation across participant maps,
plus an input inventory. Participants with incomplete step-11.2 outputs are
excluded with a warning; `--min_percentage_available 1` requires all selected
participants, and at least two complete participants are always required.
Constant RDMs yield NaN inside the mask; these propagate through the group
average. Outside-mask voxels are zero. Existing outputs are reused unless
`--replace_file` is supplied. These steps run through `searchlight.py` directly
and are outside scheduler scope.

With `--verbose`, steps 11.2/11.3 announce input counts and each loading,
averaging, correlation, and saving phase. Loading messages include participant
progress, the map count within that participant, the overall file count, and
the actual input path. Existing outputs that are reused are reported explicitly.

### Leave-one-out permutations (11.4 and 11.5)

Run after step 1, using the same class and folding as 11.2/11.3:

```powershell
& "C:\ProgramData\anaconda3\python.exe" searchlight.py `
    --dataset EmoB --specie H --model basic `
    --steps_to_run 11.4 11.5 --rsa_class all `
    --dis_method mahalanobis --mah_fold run-wise --rsa_method kendall `
    --reps 100 --reps_group 1000
```

Step 11.4 permutes category labels of the held-out participant, using one
label mapping across all voxels of each map. The other participants' RDMs and
their class-pair selection remain fixed. For a sparse class definition, target
permutations can require any pair among the labels in the class: those step-1
pairwise maps must also exist. No RSA model file is required or loaded when
running only steps 11.2 through 11.5.

Outputs mirror 11.2/11.3 under `RSA_same_rnd`, keeping the class, task, mask,
method and folding in the paths:

```text
results/RSA_same_rnd/<model>/class_<rsa_class>/leave-one-out/[run-wise/]
  <specie>-sub-NN[_ses-SS_task-TASK_run-RR]/
    <mask_type>-r-R_<dis_method>_<rsa_method>_0000.nii.gz
  mean/
    <mask_type>-<specie>-r-R_<dis_method>_<rsa_method>_task-TASK_mean_00000.nii.gz
    <mask_type>-<specie>-r-R_<dis_method>_<rsa_method>_task-TASK_std_00000.nii.gz
    <mask_type>-<specie>-r-R_<dis_method>_<rsa_method>_task-TASK_mean_00000.json
```

Participant indices run from zero to `--reps - 1`; group indices run from zero
to `--reps_group - 1`. As in step 5, each group draw independently samples one
available permutation per participant/session/run. As in 11.3, it averages
sessions within runs, runs within participants, then participants equally.
The std map is the population standard deviation across participant averages;
the JSON records the sampled files. Participants need at least one permutation
for every expected unit. Missing participants trigger a warning and obey
`--min_percentage_available`; at least two complete participants are required.
Both steps reuse existing outputs; use `--replace_rnd_files` to replace them.
Both steps create permutation file numbers in randomized order and recheck
outputs before calculating and before saving, so concurrent instances can skip
work completed by another instance.

### Regression inference

Run regression inference after steps 15.3 and 15.5, for example:

```powershell
& "C:\ProgramData\anaconda3\python.exe" searchlight.py `
    --dataset EmoC --model basic-block --specie D `
    --regression_model visual_3 --rsa_models_list emo-id__collapse `
    --steps_to_run 15.6 15.7 15.8 15.9 15.10 --reps_group 1000
```

Use the same mask, radius, distance method and group permutation count as in
15.3/15.5. Each target has its own outputs beneath
`results/RSA_regression[_rnd]/{model}/{regression_model}/{target}/`.
The null mean/std and its input manifest live in the random tree's `dist/`;
z-maps live in the corresponding `mean/` folders. Cluster distributions live in
the real tree's `dist/`, and corrected maps and CSV reports in its `mean/`.
Corrected filenames include both z and cluster probability thresholds.
These inference steps rebuild their outputs when invoked; after changing the
group maps, rerun 15.6 through 15.10 in order. At least two null maps and the
requested `--min_percentage_available` coverage are required. Zero-variance
voxels receive z=0. Inference tests positive effects with strict `z > threshold`,
consistently for real and null maps, using the same empirical maximum-cluster
cutoff helper as steps 8/9. `15.10` is retained as a textual step ID because
floating-point conversion would turn it into `15.1`.

### Leave-one-out inference (11.6–11.10)

After steps 11.3 and 11.5, use the same class, task, mask, folding, radius,
distance method and RSA method:

```powershell
& "C:\ProgramData\anaconda3\python.exe" searchlight.py `
    --dataset EmoB --model basic-block --specie H --rsa_class all `
    --steps_to_run 11.6 11.7 11.8 11.9 11.101 --reps_group 1000
```

`11.101` is an alias for `11.10`. Both are preserved as textual report step
IDs, so they do not collide with step 11.1. These steps do not need an RSA
model CSV; `--rsa_class` selects the leave-one-out inputs.

Let `root` be `results/RSA_same/{model}/class_{class}/leave-one-out/`, with
`run-wise/` appended for Mahalanobis run-wise folding, and let `stem` be
`{mask_type-}{specie}-r-{radius}_{dis_method}_{rsa_method}_task-{task}`.
The corresponding permutation root uses `RSA_same_rnd`.

- **11.6** reads permutation `mean/{stem}_mean_{index:05d}.nii.gz` files and
  writes `dist/{stem}_mean.nii.gz`, `dist/{stem}_std.nii.gz` and an exact input
  inventory under the permutation root. It uses `--reps_group` and
  `--min_percentage_available`, requiring at least two group permutations.
- **11.7** reads the real `{stem}_mean.nii.gz` directly from the real root
  (the existing step-11.3 layout), writes real `mean/{stem}_z.nii.gz` and
  permutation `mean/{stem}_z_{index:05d}.nii.gz` files. Zero null variance
  yields zero z; values outside the mask are zero. Voxels with a non-finite
  value in any group permutation or in the real mean are excluded from all
  z-maps. Step 11.6 records non-finite exclusion counts in its distribution
  receipt; step 11.7 records the final valid voxel count beside the real
  z-map. Inconsistent voxel grids are rejected.
- **11.8** writes real-root `dist/{stem}_dist.npy`, using exactly the null
  inventory from 11.6. Clusters use the strict positive threshold
  `z > --z_threshold` and 26-neighbour connectivity.
- **11.9** writes real `mean/{stem}_zt{z_threshold}_corrected.nii.gz` and a
  JSON receipt, using `--cluster_threshold` for maximum-cluster correction.
- **11.10 / 11.101** writes real `mean/{stem}_zt{z_threshold}.csv` and the
  usual publication companions. Word export and optional Google Drive
  mirroring follow the same runtime rules as step 10; mirrored files retain
  the class, fold, task and mask paths under `RSA_same`.

Derived outputs are rebuilt whenever their step runs. After changing 11.3
or 11.5 inputs, rerun 11.6 through 11.10 in order. Use the same z threshold
for 11.8, 11.9 and 11.10. As in steps 6–10, changing the cluster probability
replaces the corrected map and report for that z threshold.

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

### Multiple regression RSA (steps 15–15.5)

```bash
python searchlight.py --dataset EmoC --model basic-block --specie H \
    --dis_method correlation --regression_model visual_3 \
    --rsa_models_list YOUR_TARGET_MODEL \
    --steps_to_run 15 15.3 15.4 15.5 --reps 100 --reps_group 1000
```

Replace `YOUR_TARGET_MODEL` with an RSA model name, or pass several names.
`--rsa_model` also works for a single target. The control names come from
`rsa_models/regression_models/{regression_model}.csv`; their real model matrices
are resolved separately for each run. All four steps require `--regression_model`.
Steps can be run individually after their inputs exist: `1 → 15 → 15.3` and
`1 → 15.4 → 15.5`.

Step **15.3 averages standardized beta maps**, because these describe the target
model's effect after controlling for the other models. A t-map also depends on
the uncertainty of the individual fit, so an average t-map is not the group
effect estimate. Passing coefficient estimates into higher-level analysis is
also the convention described in the [FSL group-analysis guide](https://fsl.fmrib.ox.ac.uk/fsl/docs/task_fmri/feat/user_guide.html).
As in step 3, each available participant/session/run map has equal weight;
participants with more runs contribute more to the mean. The std map describes
the spread across these input maps, not a standard error or a group significance
test. Runs are not first averaged within participants.

Step **15.4** jointly permutes the target RDM's category labels (the same
row/column relabeling used by step 4), then refits the same voxelwise regression
as step 15. The neural pairwise maps and every control-model vector stay fixed.
It does not consume the RSA z-maps from steps 7/7.6. Each run produces `--reps`
beta/t/p maps plus JSON receipts recording the permuted target vectors.
Pairwise maps are loaded once per run and shared by all targets and permutations.

Step **15.5** generates `--reps_group` group draws. For each draw it independently
selects one available permutation (index below `--reps`) per participant/session/
run, then averages the selected beta maps with the same weighting as 15.3.
It preserves separate group draws for a subsequent null distribution, rather
than collapsing every permutation into a single mean. Each std map again
describes variation across the selected run maps.
With `--shuffle_participants`, group permutation files are produced in random
index order to reduce collisions between concurrent instances. Completed files
with matching cache signatures are still skipped.
Progress messages show the current loop position and the shuffled file index,
for example `Permutation 1/1000, map 00427 running`; cached maps say `skipping`.

Outputs, relative to `{datafolder}/{dataset}/results/`:

```text
RSA_regression/{model}/{regression_model}/{target}/
  {specie}-sub-NN/ses-NN_task-{task}_run-NN/r-{radius}_{dis_method}_beta_map.nii.gz
  mean/{mask_type}-{specie}-r-{radius}_{dis_method}_beta_mean.nii.gz
  mean/{mask_type}-{specie}-r-{radius}_{dis_method}_beta_std.nii.gz
RSA_regression_rnd/{model}/{regression_model}/{target}/
  {specie}-sub-NN/ses-NN_task-{task}_run-NN/r-{radius}_{dis_method}_beta_map_0000.nii.gz
  mean/{mask_type}-{specie}-r-{radius}_{dis_method}_beta_mean_00000.nii.gz
  mean/{mask_type}-{specie}-r-{radius}_{dis_method}_beta_std_00000.nii.gz
```

Per-run fits also have `_t_map`, `_p_map`, and `_regression.json` files with the
same permutation suffix, if applicable. Group mean JSON receipts list the exact
input maps and available fraction. Group steps honor `--min_percentage_available`
and validate image grids against the mask. They write completion markers only
when all requested targets have sufficient inputs. `--replace_file` recomputes
observed outputs; `--replace_rnd_files` recomputes permutation outputs. Use the
respective replacement flag when source maps have been changed in place.

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
| `--dis_method` | `mahalanobis` | Pairwise similarity method (step 1). Also `pearson`, `correlation`, `euclidean`, `kendall`. |
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
  maps, cluster CSVs, and publication reports.
- `results/RSA/{model}/{rsa_model}/dist/` — cluster-size distributions.
- `results/RSA_rnd/{model}/` — permutation (null) mean/std maps.

**Filenames encode the parameters** so runs with different settings do not
overwrite each other:

- Step 9: `{specie}-r-{radius}_{dis_method}_{rsa_method}_zt{z_threshold}_corrected.nii.gz`
- Step 10: `{specie}-r-{radius}_{dis_method}_{rsa_method}_zt{z_threshold}.csv`
  plus `_publication.docx`, `_publication.tex`, `_results.txt`, and `_report.json`
  appended to the same filename stem (including the mask prefix when present).

### Step 10 and 15.10 publication reports

On Windows or Google Colab, install the additional Word-export dependency in the
pipeline environment: `python -m pip install -r requirements-reporting.txt`.
No Word installation, LaTeX compiler, network service or LLM is needed to export.

On any **non-Windows, non-Colab computer**, steps 10 and 15.10 print a warning
and skip Word generation completely, without importing `python-docx` or attempting
Google Drive operations. CSV, LaTeX, Results text and provenance JSON still run
and the step can complete successfully. The JSON records the skipped Word file;
an older Word file, if present, is not refreshed. These remote workers do not
need `requirements-reporting.txt`. Colab detection uses its runtime environment
markers (`COLAB_RELEASE_TAG` or `COLAB_BACKEND_VERSION`), not installed libraries.
Even in Colab, the Windows `G:` mirror is never accessed.

Running step 10 or 15.10 writes the original CSV plus these companion files
(Word is omitted on unsupported platforms):

- `_publication.docx`: an editable journal-style table with all reported peaks,
  grouped cluster extents, coordinates in mm, two-decimal Z values, caption,
  statistical notes, and deterministic Results prose: an overview followed by
  one paragraph per cluster describing its extent and every reported peak's
  ID, anatomical label, Z value, and coordinates. Repeated and unlabelled peak
  locations are retained; coverage is of all CSV rows, not unreported maxima.
- `_publication.tex`: the same Results text and a multipage-capable table. Include
  it with `\input{...}` after loading `booktabs`, `longtable`, and `array`.
- `_results.txt`: the Results prose alone, ready to paste into a manuscript.
- `_report.json`: full-precision facts, source paths and SHA-256 hashes, saved
  correction settings, peak-selection settings, and metadata/anatomy warnings.

Use `--report_title "Action tendency"` to supply a manuscript-friendly model
name; by default, underscores in `--rsa_model` become spaces. All exports are
rebuilt when step 10 runs; existing CSV names and columns are unchanged.

Reports validate the CSV against the corrected NIfTI. Sample size is derived
from actual group input files in the saved mean-map metadata, not the current
participant configuration. Correction settings come from the corrected-map
sidecar, not current CLI defaults. Legacy YAML-formatted `.json` files are
supported. Missing metadata produces warnings and omits unsupported statements;
contradictory metadata or mismatched peaks/extents fails the report. No surviving
clusters is a valid result and produces a header-only CSV and explicit prose.
On Windows/Colab, missing `python-docx` fails with an installation instruction, leaving the CSV
available but not marking step 10 complete.

Anatomical names describe peak locations, not full cluster coverage. Unknown
labels remain unlabelled; Results text does not invent interpretations, effect
sizes, exact corrected p-values, or comparisons between models. Author review
is still required before manuscript submission.

The Google Drive mirror runs **only on Windows and only when
`G:\My Drive\Results` already exists**. Otherwise, it skips copying without
creating any Drive-style directories. When available, the existing dataset,
species and mask destination layout is preserved; corrected/unthresholded maps,
CSV, publication companions, and the correction sidecar (when present) are
copied. A copy failure is a warning and does not invalidate the primary outputs.

Step **15.10** uses the same exporter for each target model. Its captions and
Results text identify regression RSA and the control-model specification.
Z describes the group target coefficient standardized against the permutation
null, not the beta coefficient or its regression t statistic. The report uses
the strict `Z > threshold` rule of step 15.9, including for older sidecars that
did not record the comparison explicitly. Sample size comes from the actual
step-15.3 input list, and correction settings from the step-15.9 sidecar.

Regression reports retain the existing CSV stem (including `_beta`, z and
cluster-probability thresholds). Their optional Windows mirror is under
`G:\My Drive\Results\{dataset}\current-results\RSA_regression\{model}\{regression_model}\{target}\mean\`,
separate from ordinary RSA and other control/target combinations. The same
Windows/existing-root guard applies. `--report_title` also applies to 15.10.
Rebuild the Colab inference support ZIP to include the exporter; the notebook
installs `python-docx`, and downloaded result ZIPs retain `.docx` and `.tex` files.

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
- **Changing `--dis_method` mid-analysis** — the value is baked into filenames and
  job IDs; a mismatched method silently reads/writes the wrong files.
- **`--z_threshold` mismatch** — steps 8, 9, 10 must share the same threshold, or
  step 9/10 will not find the step 8 distribution.


### Concurrent regression runs (steps 15 and 15.4)

Multiple instances can process different target models or runs of the same
participant. Regression workers atomically claim each target/participant/session/run
before checking outputs or loading pairwise maps. Claimed target runs are skipped,
while other target models for that run can proceed. All workers must use this
updated version and the same output storage. Locks live under
`results/RSA_regression_rnd/<model>/<regression_model>/<target_model>/.locks/`
(or `RSA_regression` for step 15), with participant folders and session/task/run
filenames. They cover all settings that write that target's run files. Output
existence is checked after claiming the target run, preserving resumable fits.
A worker that skips a claimed target run does not write a whole-step completion marker;
rerun after the workers finish to confirm all requested outputs are complete.

Locks are removed on normal completion and Python exceptions. A forcibly killed
worker may leave a `.lock` file containing its host, PID, and creation time.
After confirming that worker has stopped, remove that specific lock and rerun.
Locks never expire automatically, so long-running fits remain protected.

Step 15.4 caches masked neural responses and their standardization once per target
and run when the design and responses are finite. Missing-value cases retain the
original fitting path, since shuffled exclusions can change the fitted rows.
Beta, t, and p maps and the target-label permutation scheme are unchanged.

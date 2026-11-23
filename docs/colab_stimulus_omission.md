# Colab: leave-one-stimulus-out maps and cross-participant similarity

## Active supervised Drive workflow

The current notebook can request Google Drive sign-in and run both existing
cohorts from `rsa_colab/omission_D` and `rsa_colab/omission_H`. Current support
modules live in `rsa_colab/omission_code`; the large participant ZIPs do not need
rebuilding when these small launcher/scheduler files are updated.

`launch_stimulus_omission.py` launches one monitored supervisor per output folder.
`supervise_stimulus_omission.py` starts OS workers with private scratch directories
and logs on the same VM, capped at two concurrent processes. It checks CPU,
available RAM, GPU utilization, VRAM and disk before admitting another worker.
Measured small dog jobs use four CPU cores while human jobs use the GPU.
The scheduler waits for every participant job of a species before its group step.

For the uploaded EmoC cohorts, the user-approved setting is
`ALLOWED_MISSING_RUNS = ['H:20:1']`. Group run 1 uses the 39 humans with that run;
human 20 remains included in every other run. All 17 dogs remain included.
Preflight verifies that the actual absent participant/run pairs match this
exact list; any additional absence raises an error. The effective inclusion
lists are saved in `omission_analysis/analysis_policy.json`, and every group
result records its included participants.

Live progress is in `omission_analysis/status.json`, resources in
`resources.jsonl`, and per-worker output in `logs/`. Results remain separated
under `results_omission_D` and `results_omission_H`. Completed ZIPs are reused
after a restart. Code-sensitive checkpoint identities cover the neural compute
and input-reading implementation; scheduling-only changes preserve those ZIPs.

`tools/create_package.py` supports **steps 1 and 11.2** for beta omissions with
`--leave_one_stimulus_out`. The Colab implementation uses float64 PyTorch for
the searchlight covariance, crossnobis distances and cross-participant
correlations. It supports the package builder's **Mahalanobis stim-wise** fold.
The notebook and all code are bundled in each participant package, so Colab
does not need a repository checkout or access to the workstation data share.

## Build participant packages

Run this on the workstation. Step 0.5 aligned beta maps and the selected
`rsa_models/by_class/class_all.csv` must already exist.

```powershell
& C:\ProgramData\anaconda3\python.exe C:\github\dog_brain_toolkit\tools\create_package.py D `
    --all-participants --dataset EmoC --model basic-block --radius 4 `
    --leave_one_stimulus_out --rsa_class all --rsa_method kendall `
    --out 'G:\My Drive\rsa_colab\pkg_omission_D'
```

Use `H` and a separate output folder for humans. The model name must match the
config and beta folders exactly. `--datafolder PATH` overrides the usual data
parent. To build specific packages, replace `--all-participants` with subject
numbers, for example `D 1 2 3`. The package still records the full configured
cohort; an intentional smaller analysis is selected with `PARTICIPANTS` in the
notebook. `--models`, `--all`, and `--all-stim-wise` are unnecessary in omission
mode. Step 1 derives all categories and category pairs from the config's
`stim_types`, using the same rule as `calculate_pairwise_similarity_maps2`
(`DogA1` → `DogA` for EmoC; the prefix before `-` for EmoB). `--rsa_class`
names the output directory and selects comparisons only for step 11.2.

The class CSV can contain either category labels or individual beta labels.
Beta labels are mapped through the config's category rule, within-category
pairs are dropped, and duplicate category pairs are counted only once. Thus
EmoC's 780-pair beta-level `class_all.csv` resolves to 45 category comparisons.
The workstation CSV is preserved: packages contain a resolved category CSV
plus the original under `source/class_all.csv`. A custom class subset never
reduces the step-1 maps or the categories used to estimate covariance.

The output directory contains:

- One `pkg_..._leave-one-stimulus-out_class-all.zip` per participant.
- `colab_leave_one_stimulus_out.ipynb`, ready to open in Colab.

Each ZIP includes aligned betas, mask, config, class pairs, participant/run
metadata, GPU code and the notebook. Existing ordinary step-1 maps are excluded:
they do not represent beta omissions. Standard packages for steps 1/2/4 retain
their existing workflow and filenames.

## Upload and run

Upload/sync the participant ZIPs into one Google Drive folder. Open
`tools/colab_gpu/colab_leave_one_stimulus_out.ipynb` (or its copy beside the
packages) in Colab, select a GPU runtime and edit:

```python
PACKAGES_DIR = '/content/drive/MyDrive/rsa_colab/pkg_omission_D'
OUT_DIR = '/content/drive/MyDrive/rsa_colab/results_omission_D'
SPECIE = 'D'
DATASET = 'EmoC'
MODEL = 'basic-block'
PARTICIPANTS = None
STEPS = [1, 11.2]
```

Run all cells. The notebook verifies the cohort, matching analysis parameters,
class CSVs, code versions and identical masks before computing. By default all
configured participants must have packages. `PARTICIPANTS = [1, 2, 3]` explicitly
selects a smaller reference cohort; at least two are required. Radius, class,
mask and correlation method come from the packages, rather than separate
notebook overrides.

Step 1 removes one exact beta condition from one session/run at a time, retaining
all other observations. Category means and Ledoit–Wolf covariance are recomputed
for each omission. Individual session/run entries remain separate CV partitions,
matching the workstation omission implementation. Pairs with fewer than two
paired partitions remain NaN.

Step 11.2 retains the class CSV pairs and compares each participant's omission
RDM against the equally weighted mean RDM of the other participants with the
**same omitted beta/run**. Repeated omission sessions are averaged within each
reference participant. Target sessions are correlated individually; their
correlations are averaged within participant before averaging participants.
NaN/Inf values are excluded from means and correlations. Kendall uses tau-b,
including ties; Pearson and Spearman are also supported.

The group output has exactly this pipeline-relative layout:

```text
EmoC/results/RSA_same/basic-block/class_all/leave-one-out/by_stim/run-01/
  b_GreyMatter2mmB-D-r-4_kendall_DogA1.nii.gz
```

`run-01` identifies the run from which `DogA1` was removed, even though the
Mahalanobis RDM pools multiple runs. Participant maps retain the step-1 layout
`by_stim/D-sub-XX/run-N/DogA1/r-4_mahalanobis_<cat1>_<cat2>.nii.gz`, with a
`ses-XX` component when run numbers repeat across sessions.

## Checkpoints, memory and resuming

Every completed participant omission produces a `result_step1_omission_*.zip`
on Drive. Each completed group map and JSON sidecar produce a
`result_step11_2_omission_*.zip`. Files publish through `.part` names and are
reused only when their input/version fingerprints and expected members match.
Rerun the notebook with `FORCE = False` after disconnecting; completed omissions
are skipped. `[1]` runs only step 1; `[11.2]` uses the existing step-1 checkpoints
in the same `OUT_DIR`. `FORCE = True` recomputes the selected steps.

`STEP1_BATCH`, `VOXEL_BATCH` and `COMPARISON_BATCH` control compute batch sizes,
not analysis parameters. Lower the first for searchlight GPU memory pressure;
lower the other two for correlation memory pressure. A CPU fallback is available
for testing, while the notebook uses CUDA whenever available.

Only one participant's betas are extracted locally at a time. Step 11.2 stages
one omitted beta/run's masked RDM bank at a time. Scratch files are removed after
each stage completes. Drive retains all pair-map sets, so output size grows with
participants × runs × beta conditions × class pairs. Cohort members without an
omitted run are excluded only if at least two and `MIN_PERCENTAGE_AVAILABLE` of
the selected cohort remain; missing checkpoint files are errors.

## Restore results to the workstation

Download/sync the output folder, then merge both kinds of ZIP with the existing
unpacker:

```powershell
& C:\ProgramData\anaconda3\python.exe C:\github\dog_brain_toolkit\tools\unpack_results.py `
    'G:\My Drive\rsa_colab\results_omission_D'
```

The ZIP members are already relative to the data parent. Add `--datafolder PATH`
for another destination or `--replace` when intentionally replacing local maps.
Step 11.2 already writes the requested group maps; no step 11.3 is needed.

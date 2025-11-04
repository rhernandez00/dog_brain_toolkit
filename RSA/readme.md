````markdown
# Representational Similarity Analysis (RSA) Pipeline

This section documents the full RSA pipeline implemented in `rsa_utils.py`, mirroring the example analysis in `running2.ipynb`. It walks through:

- the **logic** of each step (0–10),
- the **functions** involved,
- the **input/output files** and **filename conventions**,
- and minimal **usage examples**.

The pipeline assumes a dataset organized roughly as:

```text
{datafolder}/
  {dataset}/
    config_files/
      {model}.yaml
    rsa_models/
      {rsa_model}.xlsx
    results/
      GLM/
      RSA/
      RSA_rnd/
    normalized/
    models/
````

Throughout this document we’ll use these placeholders:

* `datafolder` – root data directory
* `dataset` – dataset name (e.g. `EmoB`)
* `model` – GLM model name (e.g. `basic`)
* `rsa_model` – RSA model name (e.g. `emotion-valence-basic`)
* `specie` – `D` for dog, `H` for human, etc.
* `task` – task name (e.g. `EmoB`)
* `sub_N` – integer subject number (e.g. `1`)
* `session` – integer session (e.g. `1`)
* `run_N` – integer run (e.g. `1`)
* `radius` – searchlight radius in voxels
* `method` – searchlight similarity (e.g. `pearson`, `mahalanobis`)
* `rsa_method` – RSA comparison (e.g. `kendall`, `pearson`, `spearman`)

---

## Quick high-level picture

Very roughly, the pipeline does:

* **Step 0** — GLM, beta maps. Fit GLM, get **beta maps per condition**.
* **Step 1** — Searchlight similarity **between condition betas**.
* **Step 2** — Voxelwise RSA. For each voxel, correlate empirical pairwise pattern vector with **RSA model vector** → **per-run RSA map**.
* **Step 3** — Group average. Average per-run RSA maps over subjects/sessions/runs → **group mean/std maps**.
* **Step 4** — Permutation-based null. Repeat Step 2 with **shuffled model labels** → subject-level permutation maps.
* **Step 5** — Voxelwise null mean/std. Randomly sample permutations per subject/run and average → **group-level permutation maps**.
* **Step 6** — Calculate distribution mean and std. From permutations on group average null, compute voxelwise **null mean/std**.
* **Step 7** — Z-maps. Use voxelwise null mean/std to calculate Z-maps. Turn both real and permuted group maps into **z-maps** using null mean/std.
* **Step 8** — Cluster-size null distribution. From permuted z-maps, build **null cluster-size distribution**.
* **Step 9** — Cluster-extent correction. Apply **cluster-extent correction** to the real z-map.
* **Step 10** — Cluster table export. Extract clusters & peaks → **Excel table** for reporting.

1. **GLM → beta maps** (per condition, per run)
2. **Searchlight similarity** between condition betas
3. **Voxelwise RSA**: compare empirical pattern relations to a model
4. **Group average** across runs/subjects
5. **Permutation-based null** via label-shuffling
6. **Voxelwise null mean/std**
7. **Z-maps** (real + permutations)
8. **Cluster-size null distribution**
9. **Cluster-extent correction**
10. **Cluster table export** (Excel)

If you just want the TL;DR mental model, skip to the end.
If you want to actually run this thing without crying, read on.

---

## Step 0 – Fit GLMs and write beta maps

**Goal:** Fit a GLM per run and get one **beta (PE) map per condition**, which will be the input for RSA.

**Main function**

```python
from rsa_utils import calculate_beta_maps
```

**What it does**

* Takes a design template (`.fsf`) and fills in paths, TR, number of volumes, and stimulus onsets.
* Runs FSL FEAT (via the generated `.fsf` file).
* Produces `pe*.nii.gz` beta maps (one per condition regressor).

**Key inputs**

* `datafolder`, `dataset`, `model`, `task`, `specie`
* `sub_N`, `session`, `run_N`
* `stim_types`: list of condition names (e.g. `['dog', 'human']`)
* `design_template`: path to template FSF file
* `atlas_file`, `smooth`, motion parameters, etc.

**Input files (expected)**

* Normalized BOLD NIfTI:

  ```text
  normalized/{specie}-sub-{sub:02d}/{specie}-sub-{sub:02d}_ses-{session:02d}_task-{task}_run-{run:02d}.nii.gz
  ```

* Condition onset files (one per `stim_type`):

  ```text
  models/{model}/{specie}-sub-{sub:02d}/
    ses-{session:02d}_task-{task}_run-{run:02d}/{stim_type}.txt
  ```

**Output files**

* FEAT directory:

  ```text
  results/GLM/{model}/{specie}-sub-{sub:02d}/
      ses-{session:02d}_task-{task}_run-{run:02d}.feat/
  ```

* Beta maps (within `stats/`):

  ```text
  results/GLM/{model}/{specie}-sub-{sub:02d}/
      ses-{session:02d}_task-{task}_run-{run:02d}.feat/stats/pe{K}.nii.gz
  ```

**Example**

```python
calculate_beta_maps(
    datafolder="/path/to/data",
    dataset="EmoB",
    model="basic",
    specie="D",
    sub_N=1,
    session=1,
    run_N=1,
    task="EmoB",
    stim_types=["dog", "human", "object"],
    design_template="/path/to/basic_template.fsf",
    atlas_file="/path/to/atlas.nii.gz",
    smooth=4,
    radius_fwd=1.0,
    threshold_fwd=0.5,
    redo_if_exists=False,
    overwrite_movement=False,
)
```

---

## Step 1 – Pairwise searchlight similarity between conditions

**Goal:** For each run, compute **voxelwise similarity maps** between every pair of condition betas (searchlight RSA).

**Main functions**

```python
from rsa_utils import calculate_pairwise_similarity_maps
# or Mahalanobis / crossnobis variant:
from rsa_utils import calculate_mahalanobis_pairwise_maps
```

Internally these use:

```python
from rsa_utils import similarity_searchlight
```

**What it does**

* For each run and each condition pair (e.g. `dog` vs `human`):

  * Extracts beta maps from Step 0.
  * Slides a spherical searchlight (radius `radius`) across the brain.
  * Within each sphere, computes similarity/distance between the two betas.

Supported `method` values (for example):

* `pearson`
* `kendall`
* `correlation` (1 − Pearson)
* `euclidean`
* `mahalanobis` (via separate crossnobis pipeline)

**Input files**

* Beta maps from Step 0, e.g.:

  ```text
  results/GLM/{model}/{specie}-sub-{sub:02d}/
      ses-{session:02d}_task-{task}_run-{run:02d}.feat/stats/pe*.nii.gz
  ```

* A brain mask in the same space:

  ```text
  /path/to/mask.nii.gz
  ```

**Output files**

For each run, condition pair `(stim_i, stim_j)`:

```text
results/RSA/{model}/{specie}-sub-{sub:02d}/
  ses-{session:02d}_task-{task}_run-{run:02d}/
    r-{radius}_{method}_{stim_i}_{stim_j}.nii.gz
```

**Example**

```python
calculate_pairwise_similarity_maps(
    datafolder="/path/to/data",
    dataset="EmoB",
    model="basic",
    specie="D",
    sub_N=1,
    session=1,
    run_N=1,
    task="EmoB",
    stim_types=["dog", "human", "object"],
    mask="/path/to/mask.nii.gz",
    radius=3,
    method="pearson",
    replace_file=False,
    verbose=True,
)
```

---

## Step 2 – Compare empirical similarities with an RSA model

**Goal:** For each run, turn the full set of pairwise searchlight maps into **one voxelwise “model similarity” map**, measuring how well local pattern relations match a theoretical RSA model.

**Main functions**

```python
from rsa_utils import read_model_dict, compare_with_model
```

**What it does**

1. **Load RSA model** (e.g. from Excel) and return:

   * `model_dict['model']`: nested dict of pairwise relationships.
   * `model_dict['categories']`: list of condition labels (must match `stim_types`).
   * `model_dict['pairs']`: canonical list of all unique condition pairs.
2. **Flatten model** to a vector over all unique pairs.
3. Build a **meta-similarity map**:

   * For each voxel, gather its searchlight values for all condition pairs (using the pairwise maps from Step 1).
4. For each voxel, compute similarity between:

   * model vector, and
   * empirical vector for that voxel
     using `rsa_method` (e.g. Kendall, Pearson, Spearman).
5. Save the resulting 3D map.

**Input files**

* All pairwise maps for that run (from Step 1), e.g.:

  ```text
  results/RSA/{model}/{specie}-sub-{sub:02d}/
    ses-{session:02d}_task-{task}_run-{run:02d}/
      r-{radius}_{method}_{stim_i}_{stim_j}.nii.gz
  ```

* RSA model definition:

  ```text
  rsa_models/{rsa_model}.xlsx
  ```

**Output files (per run)**

```text
results/RSA/{model}/{rsa_model}/{specie}-sub-{sub:02d}/
  ses-{session:02d}_task-{task}_run-{run:02d}/
    r-{radius}_{method}_{rsa_method}.nii.gz
```

**Optional subject-mean**

If `create_subject_mean=True`, the function also writes:

```text
results/RSA/{model}/{rsa_model}/{specie}-sub-{sub:02d}/
  r-{radius}_{method}_{rsa_method}_mean.nii.gz
  r-{radius}_{method}_{rsa_method}_std.nii.gz
```

**Example**

```python
compare_with_model(
    ref_img=nib.load("/path/to/mask.nii.gz"),
    mask_affine=nib.load("/path/to/mask.nii.gz").affine,
    datafolder="/path/to/data",
    dataset="EmoB",
    sub_N=1,
    session=1,
    run_N=1,
    specie="D",
    model="basic",
    task="EmoB",
    radius=3,
    rsa_model="emotion-valence-basic",
    method="pearson",
    rsa_method="kendall",
    replace_file=False,
    verbose=True,
    rnd=False,
    reps=1,
    create_subject_mean=False,
)
```

---

## Step 3 – Group-level model similarity maps

**Goal:** Average the per-run RSA model maps (Step 2) across **subjects/sessions/runs** to obtain group-level RSA maps.

**Main function**

```python
from rsa_utils import calculate_group_model_similarity_map
```

**What it does**

* Collects paths to all per-run RSA maps:

  ```text
  results/RSA/{model}/{rsa_model}/{specie}-sub-{sub:02d}/
    ses-{session:02d}_task-{task}_run-{run:02d}/
      r-{radius}_{method}_{rsa_method}.nii.gz
  ```

* Computes voxelwise:

  * Mean across all maps
  * Standard deviation across all maps

* Writes a JSON log with file list and fraction of available data.

**Input**

* `session_and_run_all_dict`: dict like

  ```python
  {
    1: [{'session': 1, 'run': 1}, {'session': 1, 'run': 2}],
    2: [{'session': 1, 'run': 1}],
    ...
  }
  ```

* Other standard arguments: `datafolder`, `dataset`, `specie`, `model`, `task`, `radius`, `rsa_model`, `method`, `rsa_method`.

**Output files**

```text
results/RSA/{model}/{rsa_model}/mean/
  {specie}-r-{radius}_{method}_{rsa_method}_mean.nii.gz
  {specie}-r-{radius}_{method}_{rsa_method}_std.nii.gz
  {specie}-r-{radius}_{method}_{rsa_method}_mean.json   # log
```

---

## Step 4 – Permutation at subject level (label shuffling)

**Goal:** Generate **permuted RSA model maps** to build a null distribution.

This is done by re-running Step 2 with **shuffled model vectors**, not shuffled data.

**Main function**

```python
from rsa_utils import compare_with_model   # with rnd=True
```

**What it does**

* For each subject / session / run:

  * Calls `compare_with_model(..., rnd=True, reps=R)` where `R` is the number of permutations.
  * Each repetition randomizes the mapping between model categories (using `shuffle_vector`) and produces a new map.

**Output files**

Per subject / session / run / permutation:

```text
results/RSA_rnd/{model}/{rsa_model}/{specie}-sub-{sub:02d}/
  ses-{session:02d}_task-{task}_run-{run:02d}/
    r-{radius}_{method}_{rsa_method}_{rnd_N:04d}.nii.gz
```

These are the **building blocks** for group-level permutations in Step 5.

---

## Step 5 – Group-level permutation RSA maps

**Goal:** Build **group-level null RSA maps** by averaging permuted subject maps, mirroring Step 3 but in permutation land.

**Main function**

```python
from rsa_utils import calculate_group_model_similarity_map_rnd
```

**What it does**

For each group permutation index `rnd_N`:

1. For each subject/session/run:

   * Randomly select an individual permutation `rnd_individual_N` from the per-run permuted maps.
   * Collect paths like:

     ```text
     results/RSA_rnd/{model}/{rsa_model}/{specie}-sub-{sub:02d}/
       ses-{session:02d}_task-{task}_run-{run:02d}/
         r-{radius}_{method}_{rsa_method}_{rnd_individual_N:04d}.nii.gz
     ```

2. Average all selected maps voxelwise → **one group null map**.

3. Repeat for `rnd_N = 0 ... reps_group-1`.

**Output files**

```text
results/RSA_rnd/{model}/{rsa_model}/mean/
  {specie}-r-{radius}_{method}_{rsa_method}_mean_{rnd_N:05d}.nii.gz
```

Plus temporary `_tmp.txt` lock-files during the computation.

---

## Step 6 – Voxelwise null mean & std

**Goal:** From the set of group permutation maps from Step 5, compute the **voxelwise null mean and std**.

**Main function**

```python
from rsa_utils import calculate_voxelwise_rnd_distribution
```

**What it does**

* Loops over `rnd_N` group null maps
* Computes voxelwise mean and std across all these images
* Saves them as NIfTI files

**Input files**

```text
results/RSA_rnd/{model}/{rsa_model}/mean/
  {specie}-r-{radius}_{method}_{rsa_method}_mean_{rnd_N:05d}.nii.gz
```

**Output files**

```text
results/RSA_rnd/{model}/
  {specie}-{rsa_model}_mean.nii.gz
  {specie}-{rsa_model}_std.nii.gz
  {specie}-{rsa_model}_mean_log.txt
```

These are your **null distribution** images.

---

## Step 7 – Z-maps (real & permutations)

### 7A – Z-maps for permuted group maps

**Goal:** For each group permutation map (Step 5), compute a z-map:

[
Z_{perm} = \frac{GroupNull - NullMean}{NullStd}
]

**Main function**

```python
from rsa_utils import calculate_z_maps_rnd
```

**Input files**

* Group permutation means (Step 5):

  ```text
  results/RSA_rnd/{model}/{rsa_model}/mean/
    {specie}-r-{radius}_{method}_{rsa_method}_mean_{rnd_N:05d}.nii.gz
  ```

* Null mean/std from Step 6:

  ```text
  results/RSA_rnd/{model}/
    {specie}-{rsa_model}_mean.nii.gz
    {specie}-{rsa_model}_std.nii.gz
  ```

**Output files**

```text
results/RSA_rnd/{model}/{rsa_model}/mean/
  {specie}-r-{radius}_{method}_{rsa_method}_z_{rnd_N:05d}.nii.gz
```

---

### 7B – Z-map for the real group map

**Goal:** Compute a **z-map for the real group RSA** (Step 3) using the null mean/std from Step 6:

[
Z_{real} = \frac{GroupReal - NullMean}{NullStd}
]

**Main function**

```python
from rsa_utils import calculate_z_map_real_data
```

**Input files**

* Real group mean (Step 3):

  ```text
  results/RSA/{model}/{rsa_model}/mean/
    {specie}-r-{radius}_{method}_{rsa_method}_mean.nii.gz
  ```

* Null mean/std (Step 6) as above.

**Output file**

```text
results/RSA/{model}/{rsa_model}/mean/
  {specie}-r-{radius}_{method}_{rsa_method}_z.nii.gz
```

---

## Step 8 – Cluster size distribution from permuted z-maps

**Goal:** Estimate the **null distribution of cluster sizes** at a given z-threshold using permutation z-maps from Step 7.

**Main function**

```python
from rsa_utils import calculate_cluster_size_distribution
```

**What it does**

* For each permutation z-map:

  ```text
  results/RSA_rnd/{model}/{rsa_model}/mean/
    {specie}-r-{radius}_{method}_{rsa_method}_z_{rnd_N:05d}.nii.gz
  ```

* Threshold at `z_threshold` (e.g. 3.1)

* Find clusters (3D connectivity)

* Record their voxel sizes

* Aggregate across permutations into a single dictionary and save

**Output**

```text
results/RSA/{model}/{rsa_model}/dist/
  {specie}-r-{radius}_{method}_{rsa_method}_dist.npy
  {specie}-r-{radius}_{method}_{rsa_method}_dist_log.txt
```

This `.npy` is later used to compute a cluster-extent threshold.

---

## Step 9 – Cluster-extent correction on real z-map

**Goal:** Use the null cluster-size distribution (Step 8) to **cluster-correct** the real group z-map from Step 7.

**Main function**

```python
from rsa_utils import apply_cluster_correction
```

**What it does**

1. Loads:

   * Real group mean RSA map (Step 3)
   * Null mean/std (Step 6)
   * Cluster size distribution `.npy` (Step 8)
2. Recomputes the z-map:

   ```python
   Z_real = (mean_model_img - dist_mean_img) / dist_std_img
   ```
3. Thresholds at `z_threshold` (e.g. 3.1).
4. Uses `get_minimal_cluster_size(...)` and the cluster-size distribution to find the **minimal cluster size** that yields `p < cluster_threshold` (e.g. 0.05 FWER).
5. Removes clusters smaller than that size.
6. Saves the **cluster-corrected** z-map.

**Input parameters**

* `z_threshold`: cluster-forming threshold (e.g. 3.1)
* `cluster_threshold`: cluster-level alpha (e.g. 0.05)
* `forced_minimal_cluster_size`: optional override
* plus the usual: `datafolder`, `dataset`, `specie`, `model`, `rsa_model`, `radius`, `method`, `rsa_method`.

**Output file**

```text
results/RSA/{model}/{rsa_model}/mean/
  {specie}-r-{radius}_{method}_{rsa_method}_z_corrected.nii.gz
```

---

## Step 10 – Export cluster & peak tables (Excel)

**Goal:** Turn the corrected z-map into a **human-readable cluster table** (for manuscripts, slides, etc.).

**Main function**

```python
from rsa_utils import create_tables
```

Internally it calls:

* `extract_clusters_and_peaks(...)`
* `clusters_to_excel(...)`

**What it does**

1. Looks at the corrected z-map:

   ```text
   results/RSA/{model}/{rsa_model}/mean/
     {specie}-r-{radius}_{method}_{rsa_method}_z_corrected.nii.gz
   ```

2. Extracts clusters using 26-connected components.

3. For each cluster:

   * Finds local maxima (peaks)
   * Applies minimum distance between peaks (`min_dist_mm`)
   * Limits to `max_peaks_per_cluster`

4. Writes one Excel sheet listing clusters, sizes, peak coordinates (voxel and mm), and z-values.

**Output file**

```text
results/RSA/{model}/{rsa_model}/mean/
  {specie}-r-{radius}_{method}_{rsa_method}.xlsx
```

**Example**

```python
create_tables(
    datafolder="/path/to/data",
    dataset="EmoB",
    specie="D",
    model="basic",
    rsa_model="emotion-valence-basic",
    radius=3,
    method="mahalanobis",
    rsa_method="kendall",
    min_dist_mm=8.0,
    max_peaks_per_cluster=3,
)
```

---




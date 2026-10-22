# Species-reference audit: steps 16.2–16.4

Source is `specie` and indexes matrix rows. Target is `target_specie` and
indexes matrix columns. This applies equally to D → H and H → D.

## Direction-dependent references removed

| Location | Finding | Change |
| --- | --- | --- |
| `searchlight.py`, step 16.2 and 16.3 dispatch | Passed `dog_participants`, `human_participants`, `dog_mask`, `human_mask` in addition to source/target selections. | Pass only `participants`, `target_participants`, `mask`, `target_mask`. |
| `searchlight.py`, steps 16.2–16.4 | Read `args.dog_batch` and `args.human_batch` even when neutral CLI spellings were supplied. | Read `args.source_batch` and `args.target_batch`. |
| `cross_species_rsa.add_arguments` | Registered species-specific participant/mask flags; neutral batch flags stored their values under species-specific destinations. | Remove those flags; register `--mask`, `--target_mask`, `--source_batch`, `--target_batch`. Source cohort is `--participants` (existing `--participants_forced` remains an alias). |
| `rsa_utils.calculate_voxelwise_cross_similarity` | Translated `mask` into dog/human overrides with a branch on `specie`; converted neutral batches back into species-specific names. | Forward selections and batches by role directly. |
| `rsa_utils.calculate_group_level_cross_similarity` | Selected a cohort using `dog_participants if specie == 'D' else human_participants`. This also left species-named target cohort arguments unused. | Use only explicit `participants` and `target_participants`. Validate target mask/cohort against saved HDF5 metadata. |
| `cross_species_rsa.discover_inputs` | Iterated fixed dog/human override tuples, then reconciled source/target selections. | Iterate `(specie, participants, mask)` and `(target_specie, target_participants, target_mask)`. |
| `cross_species_rsa.kendall_block`, `_run_matrix` | Called source/target arrays `dog`/`human`; used `dog_batch`/`human_batch`, `nd`/`nh` internally. | Use source/target arrays, tile sizes and dimensions. New HDF5 tile attributes are `source_batch`/`target_batch`. |
| `cross_species_rsa.CorrelationReader.iter_blocks` | Exposed species-named tile sizes although rows and columns followed the file's direction. | Expose `source_batch`/`target_batch` and row/column slices. |
| `tools/create_cross_species_package.py` | Exposed species-specific overrides and independently chose source radius by species. | Forward the neutral arguments to shared input discovery. Package the manifest's actual species entries. |
| `tools/colab_gpu/run_colab_cross_species.py` | Used species-named Python arguments and argparse destinations. | Use `source_batch`/`target_batch` throughout. |
| `tools/colab_gpu/colab_rsa_cross_species.ipynb` | Neutral notebook constants were passed as `dog_batch`/`human_batch`. | Pass `source_batch`/`target_batch`. |
| Step 16.4 summary kernel and JSON | Explicitly rejected H → D; always recorded `number_human_voxels`. | Accept either direction, reconstruct source geometry, and record `number_target_voxels` plus both species codes. |
| Steps 16.3 and 16.4 radius defaults | Source default was fixed at 3; CLI could forward `None` into numeric filename formatting. | Resolve each role's radius using shared `default_radius(specie)` before constructing paths. |
| Tests and documentation | Examples and test calls used the old species-named interfaces. | Update calls and exercise asymmetric selections, grids, and masks in both directions. |

The removed argument names are an intentional API/CLI migration. Use:

| Former control | Current control |
| --- | --- |
| `dog_participants` / `human_participants` | `participants` for the selected source, `target_participants` for the selected target |
| `dog_mask` / `human_mask` | `mask` for the selected source, `target_mask` for the selected target |
| `dog_batch` / `human_batch` | `source_batch` / `target_batch` (the old batch names always meant rows/columns, even in reverse runs) |

## Species references retained and why

These references encode physical resources, supported data labels, or historical
file compatibility, rather than assigning a computation direction.

* `searchlight.py` non-cross-species initialization: D/H atlas and label
  selection (Nitzsche/Czeibert versus MNI/AAL3), FSL design/HRF files, coordinate
  transformation policy, default radii, and mask lookup. These branches are
  skipped for runs containing only steps 16.2–16.4. The H branch also contains
  a pre-existing `participants = list(range(1, 41))`; this is unrelated to the
  cross-species cohort selection and remains a candidate for a separate
  config-driven cleanup.
* `searchlight.py` preprocessing steps 0 and 0.5 and later reporting: branches
  on `specie` select different anatomical alignment/analysis methods and atlas
  labels. These are not safe to remove by renaming a variable.
* `searchlight.py` path strings `dog_brain_toolkit`: the repository's name,
  not a species selection. Its former local variable `path_to_dog_brain_toolkit`
  is now `path_to_toolkit` to avoid confusing a repository path with a species.
* `cross_species_rsa.default_radius`: existing D=3/H=4 defaults, applied
  independently to source and target. Explicit radii override them.
* `cross_species_rsa.discover_inputs`: dog Nitzsche atlas fallback if the
  dataset-local source **or target** mask for species D is absent. Explicit
  `mask`/`target_mask` overrides avoid this fallback. Human masks are
  dataset-local. Neither rule fixes the matrix direction.
* Supported species validation and HDF5 group labels `D` and `H`: actual dataset
  identifiers. File attributes select which group is the source and target.
* `CorrelationReader.target_map`: D/H defaults when reading historical files
  that lack direction attributes. The old `human_map` method is retained only
  as a D → H compatibility alias; all new paths use `target_map`.
* `LEGACY_SCHEMA = 'dog-human-rsa-16.2-v1'` and read-only fallbacks for HDF5
  attributes `dog_batch`/`human_batch`: keep old results readable. New files
  write neutral tile attributes. Strict code/input signatures still prevent
  reusing checkpoints made with different runner code.
* Stimulus labels in class CSVs, reference configs, and test fixtures (for
  example DogP/HumH): actual experimental conditions, not axis conventions.

## Using the neutral interface

For H → D, use `--specie H --target_specie D`; `--participants` then selects
humans, `--target_participants` selects dogs, `--mask` is human-space, and
`--target_mask` is dog-space. For D → H the same controls exchange roles.
Keep `--radius`, `--target_radius`, mask names, and participant selections
consistent across steps 16.2 and 16.3. Step 16.4 reads the group file selected
by direction, radii and source mask name.

Existing completed HDF5 results remain inputs to steps 16.3/16.4; no renaming
or recomputation is needed for them. Update uploaded runner code and rebuild
step-16.2 packages when adopting the new arguments.

## Verification

`tests/test_cross_species_rsa.py` checks numerical agreement, forward and reverse
end-to-end CLI runs of 16.2 → 16.3 → 16.4, different source/target cohorts and
mask names, saved geometry and threshold counts, target-selection mismatch
rejection, neutral CLI destinations, portable Colab execution, and checkpoint
recovery including historical batch attribute names. CUDA parity is optional.

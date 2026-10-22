# CPU Colab RSA

## Automatic steps 16.3 and 16.4

Open `rsa_steps16_3_16_4.ipynb` in Colab, choose a CPU runtime, and Run all.
The notebook mounts Drive and automatically discovers complete participant HDF5
files under `My Drive/rsa_colab/results_step16_2`. Dataset, model, class, species
direction, radii, mask names and available participant IDs come from the files.
It prints all detected analyses and runs each one by default. Incomplete files
are reported and excluded. Conflicting target cohorts/masks stop discovery.

Keep these companion modules beside the notebook in `My Drive/rsa_colab`:
`cross_species_rsa.py` and `discover_cross_species_groups.py` (the latter comes
from `tools/colab_gpu/`). The notebook installs NumPy, nibabel, PyYAML and h5py.
No step-1 inputs, config files or GPU are needed. Group matrices and summary
maps are written inside the existing `results_step16_2/<dataset>/results/`
tree, under `<source>-group/` and `mean/` within each class folder.

**Steps 16.3/16.4 support both directions.** Set `specie` to the source and
`target_specie` to the target. Maps always use source geometry and summarize
target voxels. All analysis controls use source/target naming; see
`docs/source_target_audit.md` for the audit and API migration.

**Step 16.4** is available in the final cell of `rsa_step16_3.ipynb`, using
the saved group HDF5. Upload the updated `cross_species_rsa.py` before running.
It saves 20 source-space NIfTI maps counting target voxels strictly above each
threshold, and one source-space mean-correlation NIfTI. The thresholds are evenly
spaced from the finite group minimum to maximum, including both endpoints;
a constant matrix gives 20 identical thresholds. Nonfinite correlations are
excluded; rows with no finite entries have zero counts and NaN means.
The exact thresholds and filenames are recorded in a summary JSON.
Outputs go to `EmoC/results/RSA_cross-specie/basic-block/class_all-mahalanobis/mean`
under the selected data/output root. Locally run `searchlight.py --steps_to_run 16.4`
with the same dataset, model, rsa_class, radius and target_radius as step 16.3.
Filenames retain the mask, radii, distance and correlation method, ending in
`_threshold-01_count.nii.gz` through `_threshold-20_count.nii.gz`, or `_mean.nii.gz`.

For **step 16.3**, open `rsa_step16_3.ipynb` in a CPU Colab runtime and upload
the current `cross_species_rsa.py` to **My Drive/rsa_colab**. It reads existing
HDF5 files directly under `MyDrive/rsa_colab/results_step16_2/EmoC/results/RSA_cross-specie/...`.
The defaults match the supplied radius-4 D-to-H example. Edit the participant
list to require a specific cohort, or leave it `None` to use every matching file.
It writes an equal-participant arithmetic mean at every source/target voxel pair
under `D-group/`, alongside the participant folders. Undefined correlations
propagate as NaN. Original step-1 data, masks, config files and a GPU are unnecessary.
The resulting HDF5 retains both voxel grids and works with
`cross_species_rsa.CorrelationReader.target_map()` to extract a target-space map
for any saved source voxel. Local step 16.3 uses the same kernel through
`rsa_utils.calculate_group_level_cross_similarity`; `--cross_species_output`
selects the root containing the step-16.2 files and receives the group output.

Open `rsa_cpu.ipynb` in Google Colab and choose a CPU runtime. The notebook
mounts Google Drive at `/content/drive`, unpacks `toolkit_cpu.zip`, and writes
the toolbox's standard nested results into
`/content/drive/MyDrive/rsa_colab/results`.

Build the small code archive from the current checkout, then package the input
data from the parent folder of your dataset (on Windows or Linux):

```powershell
& 'C:\ProgramData\anaconda3\python.exe' tools/colab_cpu.py pack --output rsa_colab/toolkit_cpu.zip
& 'C:\ProgramData\anaconda3\python.exe' tools/colab_cpu.py pack-data --datafolder 'P:\userdata\raulh87\data' --radius 4 --allow-missing --output rsa_colab/input_data.zip
```

The Python function is `pack_for_CPU_colab(datafolder, output_zip=None,
dataset='EmoC', model='basic-block', specie='D', participants=None,
mask_type='b_GreyMatter2mmB', radius=3, allow_missing=False)`. It packs the YAML config, BIDS run table, all
step-1 pairwise NIfTI maps for the selected participants and radius, and the dataset-local
mask for human or `cope13` analyses. With `participants=None`, it uses unique
`sub_N` values from `BIDS/{specie}_database-details.csv` and fails if any lacks step-1
maps. It excludes `rsa_models` deliberately.
Pass `--participants` to package a subset when the BIDS table includes a
participant without completed step-1 maps. Or pass `--allow-missing` to package
all available maps and record missing participants. Blank participant fields
in the notebook use only participants with packed step-1 maps.

Upload both zips and the notebook to **My Drive/rsa_colab**. Setup unpacks the
data archive into the flat folder. Add your model CSV separately under
`rsa_models/`. Two ready-to-edit
step-4 and step-5 JSON jobs are in the local `jobs/pending` folder; upload that
folder as well if you want the prebuilt jobs on Drive. The notebook's step-4
and step-5 cells can also create them directly there after Setup.

Step 4 and 5 cells save portable JSON jobs under `jobs/pending` without running
them. To execute one job later in a mounted Colab CPU session, use the command
printed by its cell or call `run_job()` from `toolkit/tools/colab_cpu.py`.

The ignored `_local_data/EmoC` folder holds a small workstation fixture used
to test the notebook's path and result conventions.

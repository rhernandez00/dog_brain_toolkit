# Step 16.2: source participant vs target species mean

Compare each source participant separately with the **equal-participant mean
of target species dissimilarities**, using existing Mahalanobis stim-wise maps
and Kendall tau-b. The target mean is computed once and reused for the sources.
There is no averaging of source participants and no participant-pair output.
`rsa_class` selects the ordered pairs from `rsa_models/by_class/class_<name>.csv`.
No distance estimation, permutations, thresholds or inference are performed.

The pipeline dispatches step 16.2 in its main step loop through
`rsa_utils.calculate_voxelwise_cross_similarity`. This public function accepts
explicit source/target settings, loads the target mean once, and writes each
source participant's outputs. Numerical/GPU and compact I/O helpers remain in
`cross_species_rsa.py` for reuse by the self-contained Colab runner.

## Local execution

```powershell
& 'C:\ProgramData\anaconda3\python.exe' searchlight.py `
  --steps_to_run 16.2 --dataset EmoC --model basic-block `
  --specie D --target_specie H --radius 5 --target_radius 4 `
  --rsa_class all-mahalanobis --cross_species_device auto
```

The only final outputs **per source participant** are one HDF5 and a same-stem
`.log`, for example:

```text
P:\userdata\raulh87\data\EmoC\results\RSA_cross-specie\basic-block\class_all-mahalanobis\D-sub-01\b_GreyMatter2mmB-r-5_mahalanobis_kendall_vs_H-r-4.HDF5
```

`H-r-4` denotes the human group mean dissimilarities at radius 4. The matrix
has exactly as many rows as finite nonzero source mask voxels, and columns as
finite nonzero target mask voxels. Zero-mask air/background voxels never get
rows or columns. In-mask invalid/constant RDMs remain NaN with validity flags,
so coordinate indexing stays fixed. No NIfTI files are written.

`--radius` controls source maps; `--target_radius` controls target maps. Omitted
radii use D=3, H=4 according to each species' role. `--target_specie` defaults to
H; specify `--specie D` for D-to-H. Reverse with `--specie H --target_specie D
--radius 4 --target_radius 5`. Source and target must differ.

`--participants_forced 1 3 4` selects source participants; `--target_participants
1 2 3` selects target participants. Otherwise each species' YAML supplies its
cohort. `--dog_participants`/`--human_participants` remain supported; conflicting
selections are rejected. `--mask_type` controls the source mask and output
prefix, while `--target_mask_type` controls the target mask. Both default to
`b_GreyMatter2mmB`. Dataset ROI masks are preferred, with a dog Nitzsche atlas
fallback. `--dog_mask`/`--human_mask` override species masks explicitly and their
filenames supply mask prefixes. Maps must match their own species mask's grid;
the two species' grids may differ.

`--cross_species_output` now takes an output **root directory**, beneath which
`<dataset>/results/RSA_cross-specie/...` is created. Default: parent data folder.
It no longer takes a single HDF5 filename. Run 16.2 separately from other steps.

## Package and run in Colab

```powershell
& 'C:\ProgramData\anaconda3\python.exe' tools/create_cross_species_package.py `
  --dataset EmoC --model basic-block --rsa_class all-mahalanobis `
  --specie D --target_specie H --radius 5 --target_radius 4 `
  --out tools/colab_gpu/packages/cross_species_EmoC_D5_vs_H4.zip
```

The builder follows the existing create_package.py workflow. It bundles selected
existing source maps, target maps once, both masks/configs, the class CSV and
runner code. It accepts the same participant/mask selection flags and copies
`colab_rsa_cross_species.ipynb` beside the ZIP. Upload both to `MyDrive/rsa_colab`,
select a GPU runtime in Colab, and run all cells. Package selections are printed
before calculation. Rebuild to change selections or radii. **Old v1 packages
and checkpoints must be rebuilt.** The reader still supports completed legacy
HDF5 files.

Set `OUTPUT_ROOT` to the Drive destination root. Colab preserves the same
`<dataset>/results/RSA_cross-specie/.../<source>-sub-XX/` layout as local runs.

## GPU, storage and restart

Install `requirements-cross-species.txt` locally. PyTorch is optional on CPU;
CUDA needs a compatible NVIDIA driver and CUDA-enabled PyTorch. Colab supplies
PyTorch. `DEVICE='auto'` uses CUDA when available and falls back to CPU; `'cuda'`
requires it. Sign-vector matrix multiplication uses float32 with TF32 disabled,
then float64 normalization and float32 storage; sign numerators for 45 pairs
are small exact integers. CPU uses the same formula with NumPy float64.

`--source_batch` (128) and `--target_batch` (4096) bound tile memory. Old aliases
`--dog_batch` and `--human_batch` mean source/target even for H-to-D. Notebook
settings are SOURCE_BATCH/TARGET_BATCH. No full matrix is allocated in RAM/GPU.

Each matrix needs `4 * source_mask_voxels * target_mask_voxels` uncompressed
bytes, plus RDMs and metadata. Compression ratio is data dependent. Local/Drive
staging and snapshot copies require additional storage. Negative distances and
correlations are retained. Storage is float32, with approximately 1e-7 precision.

Local `.partial`/`.initializing` files are temporary and renamed on success.
Colab publishes closed snapshots to the same final HDF5 path with complete=False,
using a temporary `.part` copy. The reader rejects incomplete files. No permanent
checkpoint sidecar is produced. The log is published alongside snapshots.
Snapshots copy the entire partial matrix after source batches, at a default
15-minute interval and on caught interruptions. Hard runtime loss resumes from
the last published snapshot. A fresh runtime restores the HDF5 as a partial or
completed result according to its completion flag. Input/settings/code hashes
reject incompatible reuse. Existing tile sizes are retained on resume; use new
output/staging roots to change them. Use one writer per output/staging directory.

## Read later

```python
from cross_species_rsa import CorrelationReader
with CorrelationReader('b_GreyMatter2mmB-r-5_mahalanobis_kendall_vs_H-r-4.HDF5') as reader:
    source = reader.file.attrs['source_specie']
    coordinate = reader.file[f'{source}/ijk'][0]
    target_image = reader.target_map(coordinate)  # in memory only
    for source_slice, target_slice, values in reader.iter_blocks():
        pass  # later thresholds, weights and summaries
    # reader.image(source, values_per_source_centre) reconstructs a summary map.
```

`correlations` is the compressed float32 matrix. D/ and H/ store group_rdms,
valid, participants, ijk, xyz_mm, shape, affine, and nifti_header. Source
`group_rdms` contains the single participant's RDMs (name retained for reader
compatibility), with reduction='participant'; target reduction is
'equal-participant-mean'. Attributes record direction, source participant, both
radii, matrix orientation, settings, input/code hashes and completion status.
`human_map` is an alias for D-to-H; `target_map` handles either direction.
Logs record participants, settings, progress, resume/reuse and failures.

## Validation

```powershell
& 'C:\ProgramData\anaconda3\python.exe' -B -m unittest discover `
  -s tests -p test_cross_species_rsa.py -v
```

Tests cover SciPy tau-b agreement, ties and invalid RDMs, mask-only dimensions,
coordinate reconstruction, individual sources against equal-weight target means,
independent radii, exact output paths, both directions, only HDF5/log final files,
portable packaging and Colab snapshot recovery. CUDA parity is checked when
available, otherwise explicitly skipped.

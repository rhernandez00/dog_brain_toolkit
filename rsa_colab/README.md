# CPU Colab RSA

Open `rsa_cpu.ipynb` in Google Colab and choose a CPU runtime. The notebook
mounts Google Drive at `/content/drive`, unpacks `toolkit_cpu.zip`, and writes
the toolbox's standard nested results into
`/content/drive/MyDrive/rsa_colab/results`.

Build the small code archive from the current checkout, then package the input
data from the parent folder of your dataset (on Windows or Linux):

```powershell
& 'C:\ProgramData\anaconda3\python.exe' tools/colab_cpu.py pack --output rsa_colab/toolkit_cpu.zip
& 'C:\ProgramData\anaconda3\python.exe' tools/colab_cpu.py pack-data --datafolder 'P:\userdata\raulh87\data' --allow-missing --output rsa_colab/input_data.zip
```

The Python function is `pack_for_CPU_colab(datafolder, output_zip=None,
dataset='EmoC', model='basic-block', specie='D', participants=None,
mask_type='b_GreyMatter2mmB', allow_missing=False)`. It packs the YAML config, BIDS run table, all
step-1 pairwise NIfTI maps for the selected participants, and the dataset-local
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

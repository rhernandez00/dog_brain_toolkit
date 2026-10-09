"""Generate the editable CPU Colab notebook in rsa_colab/."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'rsa_colab' / 'rsa_cpu.ipynb'


def options_cell(step: int, *, job: bool = False) -> str:
    participants = '"1"' if step in (2, 4) else '""'
    text = f'''# @title Step {step} {"— create a job" if job else "— run on CPU"}
# Edit the defaults here, then run only this cell.
dataset = "EmoC" # @param {{type:"string"}}
model = "basic-block" # @param {{type:"string"}}
rsa_model = "action_tendency__all" # @param {{type:"string"}}
specie = "D" # @param ["D", "H"]
participants = {participants} # @param {{type:"string"}}  # comma-separated; blank = packed BIDS selection
dis_method = "mahalanobis" # @param {{type:"string"}}
rsa_method = "kendall" # @param {{type:"string"}}
mah_fold = "stim-wise" # @param ["stim-wise", "stim-wise-multiple-folds", "stim-wise-all-runs", "run-wise"]
mask_type = "b_GreyMatter2mmB" # @param {{type:"string"}}
radius = 3 # @param {{type:"integer"}}
reps = 100 # @param {{type:"integer"}}  # step 4 permutations per participant
reps_group = 1000 # @param {{type:"integer"}}  # step 5 group permutations
z_threshold = 3.1 # @param {{type:"number"}}
cluster_threshold = 0.05 # @param {{type:"number"}}
min_percentage_available = 1.0 # @param {{type:"number"}}
min_dist_mm = 8.0 # @param {{type:"number"}}
replace_file = False # @param {{type:"boolean"}}
replace_rnd_files = False # @param {{type:"boolean"}}
report_title = "" # @param {{type:"string"}}  # used by step 10
model_specie = "" # @param {{type:"string"}}  # optional for steps 3 and 5: D or H

options = dict(step={step}, dataset=dataset, model=model, rsa_model=rsa_model,
               specie=specie, participants=([int(x.strip()) for x in participants.split(",") if x.strip()]
                                             if participants.strip() else None),
               dis_method=dis_method, rsa_method=rsa_method, mah_fold=mah_fold,
               mask_type=mask_type, radius=radius, reps=reps, reps_group=reps_group,
               z_threshold=z_threshold, cluster_threshold=cluster_threshold,
               min_percentage_available=min_percentage_available, min_dist_mm=min_dist_mm,
               replace_file=replace_file, replace_rnd_files=replace_rnd_files,
               report_title=report_title or None, model_specie=model_specie or None)
'''
    text += ('create_job(ROOT, **options)' if job else 'run_step(ROOT, **options)') + '\n'
    return text


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        'colab': {'name': OUTPUT.name, 'provenance': []},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'accelerator': 'NONE',
    }
    nb.cells = [
        nbf.v4.new_markdown_cell('''# RSA pipeline on Colab CPU

Use **Runtime → Change runtime type → CPU**. The free tier is sufficient for the
runtime type; actual steps 2, 4 and 5 may take hours and can exceed a free session.

1. On the workstation run `python tools/colab_cpu.py pack --output rsa_colab/toolkit_cpu.zip`
   and `python tools/colab_cpu.py pack-data --datafolder <parent-of-EmoC> --radius 3 --output rsa_colab/input_data.zip`.
   `pack_for_CPU_colab()` also accepts `dataset`, `model`, `specie`, `participants`
   `mask_type`, `radius` and `allow_missing` options. It takes participants from the BIDS run table's
   `sub_N` column by default, and packs the config, table, step-1 maps and
   dataset-local mask when needed. Use `--allow-missing` to record missing step-1
   participants while still making an archive of the available maps.
2. Upload both zips to **My Drive/rsa_colab**. Add the RSA model CSV yourself at
   `rsa_models/{rsa_model}.csv` (or the `-run-N.csv` series). The data archive
   intentionally excludes model CSVs. To start at later steps, supply their
   prerequisite maps under `results`.
3. Run **Setup** once. Drive authentication is interactive. The code is unpacked
   into `rsa_colab/toolkit`; inputs and outputs use `rsa_colab/results`.
4. Run the desired step cells in pipeline order. Steps 4 and 5 **create JSON jobs**
   in `rsa_colab/jobs/pending`; they do not start permutations. A saved job can be
   run later on a CPU Colab session with
   `python /content/drive/MyDrive/rsa_colab/toolkit/tools/colab_cpu.py run-job /content/drive/MyDrive/rsa_colab/jobs/pending/<job>.json`.

Dependency order: **2 → 3**, **4 → 5 → 6**, **3 + 6 → 7 → 8 → 9 → 10**.
Step 1 must already be available for step 2 and step 4. Keep the same model,
species, distance method, radius, fold, mask and permutation counts across cells.
Leave `participants` blank for the participants recorded in the input archive
(validated against the BIDS run table); use a number such
as `1` for a single participant. A group analysis should include its intended
participants, and step 5 requires their step-4 permutation maps.
'''),
        nbf.v4.new_code_cell('''# @title Setup — mount Drive, unpack functions, install CPU dependencies
from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import subprocess, sys, zipfile

ROOT = Path('/content/drive/MyDrive/rsa_colab')
ROOT.mkdir(parents=True, exist_ok=True)
package = ROOT / 'toolkit_cpu.zip'
if package.is_file():
    with zipfile.ZipFile(package) as archive:
        for name in archive.namelist():
            parts = Path(name).parts
            if not parts or parts[0] != 'toolkit' or '..' in parts or Path(name).is_absolute():
                raise ValueError(f'Unsafe archive member: {name}')
            dest = ROOT / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(name))
elif not (ROOT / 'toolkit' / 'searchlight.py').is_file():
    raise FileNotFoundError(f'Upload {package} before running Setup')

subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', '-r',
                str(ROOT / 'toolkit' / 'requirements-colab.txt')], check=True)
sys.path.insert(0, str(ROOT / 'toolkit' / 'tools'))
from colab_cpu import run_step, create_job, unpack_data
data_package = ROOT / 'input_data.zip'
if data_package.is_file():
    manifest = unpack_data(data_package, ROOT)
    print('Unpacked input data:', manifest['dataset'], manifest['specie'],
          len(manifest.get('ready_participants', manifest['participants'])),
          'participants with step-1 maps')
    if manifest.get('missing_step1_participants'):
        print('Missing step-1 maps:', manifest['missing_step1_participants'])
else:
    print('No input_data.zip found. Supply config, BIDS and step-1 maps directly in rsa_colab.')
print('Ready on CPU. Root:', ROOT)
'''),
    ]
    for step in (2, 3, 4, 5, 6, 7, 8, 9, 10):
        nb.cells.append(nbf.v4.new_code_cell(options_cell(step, job=step in (4, 5))))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUTPUT)
    print(OUTPUT)


if __name__ == '__main__':
    main()

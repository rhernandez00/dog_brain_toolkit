"""Build a portable Colab continuation package for searchlight steps 15.6--15.10."""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
COLAB = REPO / 'tools/colab_gpu'


def build_package(regression_support, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(regression_support) as source:
        old = json.loads(source.read('regression_manifest.json'))
        dataset = old['dataset']
        payload = {n: source.read(n) for n in source.namelist()
                   if n.startswith(f'data/{dataset}/ROI/') and n.endswith('.nii.gz')}
    files = {f'code/{name}': REPO / name for name in ('rsa_utils.py', 'utils.py', 'preprocess_functions.py')}
    files.update({f'code/{name}': COLAB / name for name in (
        'gpu_rsa.py', 'gpu_regression.py', 'run_colab_regression.py', 'run_colab_regression_inference.py')})
    species = {
        'H': dict(labels='Atlas/Hum/AAL3.nii.gz', dictionary='Atlas/Hum/AAL_dictionary.csv',
                  template='Atlas/Hum/MNI152_T1_2mm_brain.nii.gz', apply_coords_transform=True),
        'D': dict(labels='Atlas/Dog/Nitzsche/Czeibert_labels2mm.nii.gz',
                  dictionary='Atlas/Dog/Czeibert_dictionary.csv', template=None, apply_coords_transform=False),
    }
    species = {k: v for k, v in species.items() if k in old['participants_by_species']}
    for spec in species.values():
        for key in ('labels', 'dictionary', 'template'):
            if spec[key]:
                files[spec[key]] = REPO / spec[key]
    for name, path in files.items():
        payload[name] = path.read_bytes()
    manifest = dict(kind='regression_inference', dataset=dataset, species=species,
                    file_sha256={name: hashlib.sha256(data).hexdigest() for name, data in payload.items()})
    output = out_dir / f'regression_inference_support_{dataset}.zip'
    partial = output.with_suffix('.zip.part')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, data in payload.items():
            zf.writestr(name, data)
        zf.writestr('inference_manifest.json', json.dumps(manifest, indent=2))
        zf.write(COLAB / 'colab_rsa_regression_inference.ipynb', 'colab_rsa_regression_inference.ipynb')
    partial.replace(output)
    shutil.copyfile(COLAB / 'colab_rsa_regression_inference.ipynb', out_dir / 'colab_rsa_regression_inference.ipynb')
    print(f'{output}: {output.stat().st_size / 1e6:.2f} MB; species {list(species)}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regression_support', required=True, type=Path)
    parser.add_argument('--out', default=COLAB / 'packages/regression_inference', type=Path)
    args = parser.parse_args()
    build_package(args.regression_support, args.out)

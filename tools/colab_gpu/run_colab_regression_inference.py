"""Colab continuation of regression group ZIPs through searchlight 15.6--15.10.

Runs the packaged rsa_utils implementation unchanged, replacing only its
streaming mean/std reducer with an equivalent float64 Torch reducer. Clustering
and atlas reports use the CPU pipeline itself. One model is staged at a time.
"""
import hashlib
import json
import tempfile
import time
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import rsa_utils as rsa

try:
    from .run_colab_regression import extract_safe, copy_with_progress, log
except ImportError:
    from run_colab_regression import extract_safe, copy_with_progress, log

VERSION = '1.0.0'
STEPS = ('15.6', '15.7', '15.8', '15.9', '15.10')


def discover_groups(results_dir, dataset, model, regression_model, specie,
                    mask_type, radius, dis_method, reps_group, models=None):
    """Validate real/permuted group means, not participant/run checkpoints."""
    if reps_group < 2:
        raise ValueError('At least two group permutations are required.')
    root = Path(results_dir)
    candidates = {}
    suffix = f'_{specie}.zip'
    for path in sorted(root.glob(f'result_regression_group_*{suffix}')):
        target = path.name[len('result_regression_group_'):-len(suffix)]
        if models is not None and target not in models:
            continue
        stem = f'{mask_type}-{specie}-r-{radius}_{dis_method}_beta'
        tail = f'{model}/{regression_model}/{target}/mean/{stem}'
        real = f'{dataset}/results/RSA_regression/{tail}_mean.nii.gz'
        permutations = [f'{dataset}/results/RSA_regression_rnd/{tail}_mean_{i:05d}.nii.gz'
                        for i in range(reps_group)]
        with zipfile.ZipFile(path) as zf:
            names = set(zf.namelist())
        missing = [name for name in [real, *permutations] if name not in names]
        if missing:
            raise ValueError(f'{path.name}: {len(missing)} required group mean maps missing; '
                             f'check analysis settings and finish 15.3/15.5. First: {missing[0]}')
        candidates[target] = path
        log(f'Ready: {target}, real mean + {reps_group} permutation means')
    if models is not None and set(models) - set(candidates):
        raise FileNotFoundError(f'Missing completed group ZIPs for {sorted(set(models) - set(candidates))}. '
                                'Finish steps 15.3/15.5 in the first notebook.')
    if not candidates:
        raise FileNotFoundError(f'No completed result_regression_group_*_{specie}.zip in {root}. '
                                'Participant ZIPs and run_checkpoints are not group results; '
                                'finish steps 15.3/15.5 in the first notebook.')
    return candidates


def torch_mean_stream(paths, result_map_path=None, result_map_path_std=None,
                      verbose=False, mask_img=None, device='cuda'):
    """Same Welford/population-std calculation as rsa.nifti_mean_stream."""
    if not paths:
        raise ValueError('No permutation means to reduce')
    first = nib.load(paths[0])
    mean = torch.zeros(first.shape, dtype=torch.float64, device=device)
    m2 = torch.zeros_like(mean)
    for count, path in enumerate(paths, 1):
        img = nib.load(path)
        rsa.check_same_space(('reference', first), [(str(path), img)])
        data = torch.as_tensor(img.get_fdata(), dtype=torch.float64, device=device)
        delta = data - mean
        mean += delta / count
        m2 += delta * (data - mean)
        if count == 1 or count % 25 == 0 or count == len(paths):
            log(f'15.6: null moments {count}/{len(paths)} on {device}')
    std = torch.sqrt(m2 / len(paths))
    if mask_img is not None:
        mask = torch.as_tensor(mask_img, device=device)
        mean *= mask
        std *= mask
    arrays = mean.cpu().numpy(), std.cpu().numpy()
    for path, values in zip((result_map_path, result_map_path_std), arrays):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(values, first.affine), str(path))
    return arrays


class _ProgressNifti:
    def __init__(self, step):
        self.step, self.count, self.last = step, 0, time.monotonic()

    def __getattr__(self, name):
        return getattr(nib, name)

    def load(self, path, *args, **kwargs):
        self.count += 1
        if self.count == 1 or self.count % 50 == 0 or time.monotonic() - self.last >= 30:
            log(f'{self.step}: reading map {self.count}: {Path(path).name}')
            self.last = time.monotonic()
        return nib.load(path, *args, **kwargs)


def run_inference(results_dir, out_dir, support_root, *, dataset='EmoB', model='basic-block',
                  regression_model='visual_3', specie='H', models=None, radius=None,
                  mask_type='b_GreyMatter2mmB', dis_method='correlation', reps_group=1000,
                  z_threshold=3.1, cluster_threshold=0.05, min_dist_mm=8., device='cuda',
                  work_root='/content/regression_inference_work', force=False,
                  write_permutation_z=True):
    """Run all five steps in order; resume completed model ZIPs by signature."""
    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime, or explicitly set DEVICE="cpu".')
    support_root = Path(support_root)
    config = json.loads((support_root / 'inference_manifest.json').read_text())
    if config['dataset'] != dataset:
        raise ValueError('Inference support package is for a different dataset')
    species = config['species'][specie]
    radius = radius if radius is not None else (4 if specie == 'H' else 3)
    mask_path = support_root / 'data' / dataset / 'ROI' / specie / f'{mask_type}.nii.gz'
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    atlas = nib.load(str(support_root / species['labels']))
    dictionary = pd.read_csv(support_root / species['dictionary'])
    groups = discover_groups(results_dir, dataset, model, regression_model, specie,
                             mask_type, radius, dis_method, reps_group, models)
    settings = dict(dataset=dataset, model=model, regression_model=regression_model,
                    specie=specie, radius=radius, mask_type=mask_type, dis_method=dis_method,
                    reps_group=reps_group, z_threshold=float(z_threshold),
                    cluster_threshold=float(cluster_threshold), min_dist_mm=min_dist_mm)
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for target, source in groups.items():
        stamp = source.stat()
        signature = hashlib.sha256(json.dumps([VERSION, config, settings, target,
                    stamp.st_size, stamp.st_mtime_ns, write_permutation_z], sort_keys=True).encode()).hexdigest()
        output = out_dir / f'result_regression_inference_{target}_{specie}_zt{float(z_threshold)}_p{float(cluster_threshold)}.zip'
        receipt = f'{dataset}/results/RSA_regression/{model}/{regression_model}/{target}/mean/{specie}_inference_receipt.json'
        if output.is_file() and not force:
            try:
                with zipfile.ZipFile(output) as zf:
                    if json.loads(zf.read(receipt))['signature'] == signature:
                        log(f'Skipping completed inference: {target}')
                        continue
            except (KeyError, ValueError, zipfile.BadZipFile):
                pass
        with tempfile.TemporaryDirectory(prefix='inference-', dir=work_root) as temp:
            root = Path(temp)
            local = root / 'input.zip'
            copy_with_progress(source, local)
            extract_safe(local, root / 'data', prefix=f'{dataset}/results/')
            local.unlink()
            input_files = {p for p in (root / 'data').rglob('*') if p.is_file()}
            for step in STEPS:
                log(f'{target}: step {step} starting')
                original_reducer, original_nib = rsa.nifti_mean_stream, rsa.nib
                try:
                    rsa.nifti_mean_stream = lambda *a, **kw: torch_mean_stream(*a, **kw, device=device)
                    rsa.nib = _ProgressNifti(step)
                    complete = rsa.calculate_regression_inference(
                        step=step, datafolder=str(root / 'data'), rsa_models_list=[target],
                        mask=str(mask_path), min_percentage_available=1., **settings,
                        label_dict=dictionary, label_nii_data=atlas.get_fdata(), label_affine=atlas.affine,
                        apply_coords_transform=species['apply_coords_transform'],
                        atlas_file=str(support_root / species['template']) if species.get('template') else None)
                    if not complete:
                        raise RuntimeError(f'{target}: step {step} incomplete')
                finally:
                    rsa.nifti_mean_stream, rsa.nib = original_reducer, original_nib
                log(f'{target}: step {step} complete')
            receipt_path = root / 'data' / receipt
            receipt_path.write_text(json.dumps(dict(signature=signature, steps=list(STEPS),
                source_group_zip=source.name, settings=settings, version=VERSION,
                write_permutation_z=write_permutation_z), indent=2))
            products = [p for p in (root / 'data').rglob('*') if p.is_file() and p not in input_files]
            # receipt can have overwritten a same-named input in a rerun archive.
            if receipt_path not in products:
                products.append(receipt_path)
            local_output = root / 'output.zip'
            log(f'{target}: packaging inference products')
            with zipfile.ZipFile(local_output, 'w', zipfile.ZIP_STORED, allowZip64=True) as zf:
                for p in products:
                    if not write_permutation_z and '/RSA_regression_rnd/' in p.as_posix() and '_z_' in p.name:
                        continue
                    zf.write(p, p.relative_to(root / 'data').as_posix())
            partial = output.with_suffix('.zip.part')
            copy_with_progress(local_output, partial)
            partial.replace(output)
            log(f'Inference results saved: {output.name}')
            written.append(str(output))
    return written

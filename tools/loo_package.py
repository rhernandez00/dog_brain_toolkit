"""Build a self-contained, pooled-crossnobis cohort package for steps 11.2–11.5."""
from pathlib import Path
import json
import zipfile

import nibabel as nib
import yaml

from loo_joint import check_grid, pair_path, pair_table, publish

REPO = Path(__file__).resolve().parents[1]
SUPPORT = ('loo_joint.py', 'stimulus_contribution.py', 'tools/colab_gpu/run_joint_loo.py')


def build(args, datafolder, participants):
    if args.dis_method != 'mahalanobis' or args.mah_fold != 'stim-wise':
        raise ValueError('--loo-group requires pooled stim-wise Mahalanobis pair maps.')
    if args.rsa_method not in ('kendall', 'spearman', 'pearson'):
        raise ValueError('Unsupported leave-one-out correlation method.')
    root = Path(datafolder) / args.dataset
    cfg = root / 'config_files' / f'{args.specie}_{args.model}.yaml'
    config = yaml.safe_load(cfg.read_text())
    if sorted(participants) != sorted(config['participants']):
        raise ValueError('A cohort package requires all configured participants.')
    task = config['task']
    radius = args.radius if args.radius is not None else (3 if args.specie == 'D' else 4)
    mask_name = args.mask_type + '.nii.gz'
    mask = (REPO / 'Atlas/Dog/Nitzsche' / mask_name if args.specie == 'D' and
            args.mask_type != 'cope13' else root / 'ROI' / args.specie / mask_name)
    ref = nib.load(str(mask))
    def member(p):
        return 'data/' + p.relative_to(root.parent).as_posix()
    files = {member(cfg): cfg, f'data/{args.dataset}/ROI/{args.specie}/{mask_name}': mask}
    for cls in ('dog', 'hum'):
        csv = root / 'rsa_models/by_class' / f'class_{cls}.csv'
        files[member(csv)] = csv
        for sub in sorted(participants):
            for pair in pair_table(csv)[1]:
                p = pair_path(root, args.model, args.specie, sub, None, None, task, radius,
                              args.dis_method, pair)
                check_grid(nib.load(str(p)), ref, p)
                files[member(p)] = p
    files.update({'code/' + Path(name).name: REPO / name for name in SUPPORT})
    files['colab_joint_loo.ipynb'] = REPO / 'tools/colab_gpu/colab_joint_loo.ipynb'
    manifest = dict(dataset=args.dataset, specie=args.specie, participants=sorted(participants),
                    model=args.model, task=task, radius=radius, mask_type=args.mask_type,
                    rsa_method=args.rsa_method, classes=['dog', 'hum'])
    out = Path(args.out) / f'loo_group_{args.specie}.zip'
    def write(tmp):
        import hashlib
        hashes = {}
        with zipfile.ZipFile(tmp, 'w', allowZip64=True) as z:
            for name, source in sorted(files.items()):
                h = hashlib.sha256()
                with source.open('rb') as src, z.open(name, 'w', force_zip64=True) as dst:
                    for block in iter(lambda: src.read(1024*1024), b''):
                        h.update(block)
                        dst.write(block)
                hashes[name] = h.hexdigest()
            z.writestr('manifest.json', json.dumps(dict(manifest, hashes=hashes)))
    publish(out, write)
    print(f'Created {out}')
    return out


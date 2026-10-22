"""Build a step-16.2-only Colab package from existing step-1 distance maps."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import cross_species_rsa as rsa


def create_package(manifest, output, root='.'):
    """Package only selected maps and support files, with portable paths."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    portable = copy.deepcopy(manifest)
    pending = []

    def stage(path, name):
        pending.append((Path(root) / path, name))
        return name

    portable['class_csv'] = stage(manifest['class_csv'], 'inputs/class.csv')
    for species in portable['species']:
        entry = portable['species'][species]
        entry['mask'] = stage(entry['mask'], f'inputs/{species}/mask.nii.gz')
        entry['config'] = stage(entry['config'], f'inputs/{species}/config.yaml')
        for participant in entry['participants']:
            participant['maps'] = [stage(path, f"inputs/{species}/sub-{participant['participant']:02d}/pair-{index:03d}.nii.gz")
                                   for index, path in enumerate(participant['maps'])]
    # Also verifies that all inputs are readable before starting a large archive.
    signature, identity = rsa.input_identity(manifest, root)
    portable['source_signature'] = signature
    portable['source_identity'] = identity
    partial = Path(str(output) + '.part')
    if output.exists():
        raise FileExistsError(f'Choose a new package path; already exists: {output}')
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path, name in pending:
            archive.write(path, name)
        archive.writestr('manifest.json', json.dumps(portable, indent=2))
        archive.write(REPO / 'cross_species_rsa.py', 'cross_species_rsa.py')
        archive.write(REPO / 'tools/colab_gpu/run_colab_cross_species.py', 'run_colab_cross_species.py')
    os.replace(partial, output)
    notebook = REPO / 'tools/colab_gpu/colab_rsa_cross_species.ipynb'
    import shutil
    target = output.parent / notebook.name
    if target.resolve() != notebook.resolve():
        shutil.copy2(notebook, target)
    print(f'Created {output}; upload it and {target.name} to Drive.', flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datafolder', default=None)
    parser.add_argument('--dataset', default='EmoC')
    parser.add_argument('--model', default='basic-block')
    parser.add_argument('--rsa_class', default='all-mahalanobis')
    parser.add_argument('--radius', type=float, default=None)
    parser.add_argument('--specie', choices=['D', 'H'], default='D')
    parser.add_argument('--participants', '--participants_forced', dest='participants_forced', nargs='+', type=int, default=[])
    parser.add_argument('--mask_type', default='b_GreyMatter2mmB')
    parser.add_argument('--toolkit_dir', default=str(REPO))
    parser.add_argument('--out', required=True, help='Output ZIP path; notebook is copied alongside it')
    rsa.add_arguments(parser)
    args = parser.parse_args()
    from scheduler.paths import get_paths
    manifest = rsa.discover_inputs(args.datafolder or get_paths()[0], args.dataset, args.model,
                                  args.rsa_class, args.radius,
                                  mask=args.mask, target_mask=args.target_mask, toolkit_dir=args.toolkit_dir,
                                  specie=args.specie, target_specie=args.target_specie,
                                  target_radius=args.target_radius, participants=args.participants_forced or None,
                                  target_participants=args.target_participants,
                                  mask_type=args.mask_type, target_mask_type=args.target_mask_type)
    create_package(manifest, args.out)


if __name__ == '__main__':
    main()

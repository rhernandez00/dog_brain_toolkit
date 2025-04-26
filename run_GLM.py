#!/usr/bin/env python3
"""
Script to run GLM preprocessing for a given subject.
Usage:
    python3 run_GLM.py <sub_N>
"""
import os
import shutil
import argparse

import utils
import preprocess_functions


def main(sub_N):
    # Project configuration
    project_dict = {
        "User": "raulh87",
        "Dataset": "EmoB",
        "Task": "EmoB",
        "Participants": list(range(1, 20)),
        "Runs": [1, 2, 3, 4],
        "Sessions": ["01", "02", "03"],
        "Specie": "D",
        "Atlas_type": "Nitzsche",
    }

    # Determine data directory based on OS
    if os.name == 'nt':  # Windows
        datafolder = os.path.join(
            "P:\\userdata", project_dict['User'], 'data'
        )
    else:
        datafolder = os.path.join(
            '/home', project_dict['User'], 'mnt', 'a471', 'userdata', project_dict['User'], 'data'
        )
    project_dict['Datafolder'] = datafolder

    # GLM parameters
    radius = 55.0
    threshold = 0.5
    smooth = 3
    img_type = 'brain2mm'

    # Paths
    dataset = project_dict['Dataset']
    task = project_dict['Task']
    specie = project_dict['Specie']
    atlas_type = project_dict['Atlas_type']

    # Species label for atlas subfolder
    specie_label = 'Dog' if specie == 'D' else 'Hum'

    # Loop over sessions and runs
    for session in project_dict['Sessions']:
        for run_N in project_dict['Runs']:
            # Check if GLM preprocessing is ready
            if not preprocess_functions.check_file_status(
                project_dict, sub_N, run_N, session, process='GLM'
            ):
                continue

            # Original and target movement file paths
            base_pre = os.path.join(
                datafolder, dataset, 'preprocessing',
                f"{specie}-sub-{sub_N:02d}",
                f"{specie}-sub-{sub_N:02d}_ses-{session}_task-{task}_run-{run_N:02d}.feat",
                'mc', 'prefiltered_func_data_mcf.par'
            )
            target_mov = os.path.join(
                datafolder, dataset, 'movement',
                f"{specie}-sub-{sub_N:02d}_ses-{session}_task-{task}_run-{run_N:02d}.par"
            )
            mov_txt = os.path.join(
                datafolder, dataset, 'movement',
                f"{specie}-sub-{sub_N:02d}_ses-{session}_task-{task}_run-{run_N:02d}_fwd.txt"
            )

            print(f"Copying movement file: {base_pre} -> {target_mov}")
            shutil.copyfile(base_pre, target_mov)

            print("Calculating framewise displacement...")
            preprocess_functions.fwd(
                base_pre, radius, threshold, output_file=mov_txt
            )

            # Input NIfTI file
            input_nifti = os.path.join(
                datafolder, dataset, 'normalized',
                f"{specie}-sub-{sub_N:02d}",
                f"{specie}-sub-{sub_N:02d}_ses-{session}_task-{task}_run-{run_N:02d}.nii.gz"
            )

            # Atlas file
            atlas_file = os.path.join(
                os.getcwd(), 'Atlas', specie_label, atlas_type, f"{img_type}.nii.gz"
            )

            # Extract TR and volumes
            TR, volumes = utils.extract_params(input_nifti)

            # Output directory for FSL
            fsl_out = os.path.join(
                datafolder, dataset, 'results', 'GLM', 'model-01',
                f"{specie}-sub-{sub_N:02d}",
                f"ses-{session}task-{task}run-{run_N:02d}"
            )

            # Condition file
            cond_file = os.path.join(
                datafolder, dataset, 'models', 'model-01',
                f"run-{run_N:02d}", 'cond-01.txt'
            )

            # Prepare FSF template replacement dictionary
            design_in = os.path.join(datafolder, dataset, 'FSL_designs', 'model-01.fsf')
            design_out = os.path.join(datafolder, dataset, 'FSL_designs', 'model-01_modified.fsf')

            labels = {
                'outputdir': (fsl_out,        'set fmri(outputdir)'),
                'TR':        (TR,             'set fmri(tr)'),
                'volumes':   (volumes,        'set fmri(npts)'),
                'BET':       (1 if specie=='H' else 0, 'set fmri(bet_yn)'),
                'smooth':    (smooth,         'set fmri(smooth)'),
                'input':     (input_nifti,    'set feat_files(1)'),
                'atlas':     (atlas_file,     'set fmri(regstandard)'),
                'movement':  (target_mov,     'set confoundev_files(1)'),
                'condition': (cond_file,      'set fmri(custom1)'),
            }

            # Build replacement dict
            to_fill = {}
            for key, (val, find_str) in labels.items():
                rep = f'set {find_str.split()[1]}("{val}"' if isinstance(val, str) else f'set {find_str.split()[1]} {val}'
                # Actually ensure correct format
                if key in ['outputdir', 'input', 'atlas', 'movement', 'condition']:
                    rep = f'{find_str} "{val}"'
                else:
                    rep = f'{find_str} {val}'
                to_fill[key] = {
                    'string_to_find': find_str,
                    'string_to_replace': rep
                }

            # Fill FSF and run FSL
            utils.fill_fsf(to_fill, design_in, design_out)

            # Remove existing .feat dir if present
            if os.path.exists(fsl_out + '.feat'):
                shutil.rmtree(fsl_out + '.feat')

            cmd = f'feat {design_out}'
            print(f"Running: {cmd}")
            if os.name != 'nt':
                os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run GLM for a specified subject.'
    )
    parser.add_argument(
        'sub_N', type=int, help='Subject number (integer)'
    )
    args = parser.parse_args()
    main(args.sub_N)

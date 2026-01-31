# Executable version of RSA pipeline
# 7/Nov/2025

import os
import time
import numpy as np
import nibabel as nib
import pandas as pd
import yaml
import sys
from importlib import reload
import argparse

'''
Input arguments:
--steps_to_run: List of steps to run (default: [1,2,3,4,5,6,7,8,9,10])
--model: GLM model to use (default: 'basic')
--method: Method for pairwise similarity calculation (default: 'correlation')
--rsa_model: RSA model to use
--rsa_method: Method to compare similarity maps with model (default: 'kendall')
--specie: 'D' for Dog, 'H' for Human (default: 'D')
--mask_type: Type of brain mask to use (default: 'b_GreyMatter2mm')
--radius: Radius for searchlight (default: 3)
--z_threshold: Z threshold for z maps (default: 3.1)
--cluster_threshold: Cluster threshold for cluster correction (default: 0.05)

--reps: Number of repetitions for permutations in individual run (default: 100)
--reps_group: Number of repetitions for permutations in group analysis (default: 1000)
--min_percentage_available: Minimum percentage of database available to process (default: 0.8)
--min_dist_mm: Minimum distance between peaks in mm (default: 8.0)
--atlas_type: Type of atlas to use in case of dogs (default: 'Nitzsche')
--replace_file: Overwrite existing output files (default: False)

--participants_forced: List of participants to include (default: [])
--verbose: Verbose output (default: False)
--wait_time: Wait time between steps in seconds (default: 300)
--overwrite_movement: Overwrite existing movement files (default: False)
'''

# parser function
def parse_arguments():
    parser = argparse.ArgumentParser(description='RSA Pipeline Execution')
    parser.add_argument('--steps_to_run', type=int, nargs='+', default=[1,2,3,4,5,6,7,8,9,10],
                        help='List of steps to run')
    parser.add_argument('--model', type=str, default='basic',
                        help='GLM model to use')
    # parser.add_argument('--mask_path', type=str, required=True,
    #                     help='Path to mask file')
    parser.add_argument('--method', type=str, default='correlation',
                        help='Method for pairwise similarity calculation')
    parser.add_argument('--rsa_model', type=str, required=True,
                        help='RSA model to use')
    parser.add_argument('--rsa_method', type=str, default='kendall',
                        help='Method to compare similarity maps with model')
    parser.add_argument('--specie', type=str, default='D',
                        help="'D' for Dog, 'H' for Human")
    parser.add_argument('--mask_type', type=str, default='b_GreyMatter2mm',
                        help='Type of brain mask to use')
    parser.add_argument('--radius', type=int, default=3,
                        help='Radius for searchlight')
    parser.add_argument('--z_threshold', type=float, default=3.1,
                        help='Z threshold for z maps')
    parser.add_argument('--cluster_threshold', type=float, default=0.05,
                        help='Cluster threshold for cluster correction')
    parser.add_argument('--reps', type=int, default=100,
                        help='Number of repetitions for permutations in individual run')
    parser.add_argument('--reps_group', type=int, default=1000,
                        help='Number of repetitions for permutations in group analysis')
    parser.add_argument('--min_percentage_available', type=float, default=0.8,
                        help='Minimum percentage of database available to process')
    parser.add_argument('--min_dist_mm', type=float, default=8.0,
                        help='Minimum distance between peaks in mm')
    parser.add_argument('--atlas_type', type=str, default='Nitzsche',
                        help='Type of atlas to use in case of dogs')
    parser.add_argument('--replace_file', action='store_true',
                        help='Overwrite existing output files')
    parser.add_argument('--participants_forced', type=int, nargs='+', default=[],
                        help='List of participants to include')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--wait_time', type=int, default=300,
                        help='Wait time between steps in seconds')
    parser.add_argument('--overwrite_movement', action='store_true',
                        help='Overwrite existing movement files')
    return parser.parse_args()

# main execution
def main():
    args = parse_arguments()
    steps_to_run = args.steps_to_run
    model = args.model
    # mask_path = args.mask_path
    method = args.method
    
    rsa_model = args.rsa_model
    rsa_method = args.rsa_method
    specie = args.specie
    mask_type = args.mask_type
    radius = args.radius
    z_threshold = args.z_threshold
    cluster_threshold = args.cluster_threshold
    reps = args.reps
    reps_group = args.reps_group
    min_percentage_available = args.min_percentage_available
    min_dist_mm = args.min_dist_mm
    atlas_type = args.atlas_type
    replace_file = args.replace_file
    participants_forced = args.participants_forced
    verbose = args.verbose
    wait_time = args.wait_time
    overwrite_movement = args.overwrite_movement

    dataset = 'EmoB'
    task = 'EmoB'

    # check if rsa_model ends with -basic if it does, model is basic
    if rsa_model.endswith('-basic'):
        model = 'basic'
    else:
        model = 'basic-block'


    if os.name == 'nt':  # Windows
        datafolder = os.path.join(
            "P:\\userdata", 'raulh87', 'data'
        )
        git_folder = r"C:\github"
        mask_path = os.path.join("P:\\userdata", 'raulh87', 'data', 'EmoB', 'ROI', 'Cope13-Emo-Con_Z3.1.nii.gz')
        mask_path = os.path.join("P:\\userdata", 'raulh87', 'data', 'EmoB', 'ROI', 'b_GreyMatter2mm.nii.gz')
    else:
        datafolder = os.path.join(
            '/home', 'raulh87', 'mnt', 'a471', 'userdata', 'raulh87', 'data'
        )
        #'/home/raulh87/mnt/a471/userdata/raulh87/github
        git_folder = os.path.join('/home', 'raulh87', 'mnt', 'a471', 'userdata', 'raulh87', 'github')
        mask_path = os.path.join(
        '/home', 'raulh87', 'mnt', 'a471', 'userdata', 'raulh87', 'data', 'EmoB', 'ROI', 'Cope13-Emo-Con_Z3.1.nii.gz')
        mask_path = os.path.join(
        '/home', 'raulh87', 'mnt', 'a471', 'userdata', 'raulh87', 'data', 'EmoB', 'ROI', 'b_GreyMatter2mm.nii.gz')

    config_path = datafolder + os.sep + dataset + os.sep + 'config_files' + os.sep + model + '.yaml'

    # Load config.yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # stim_types = config['stim_types']
    model_dict = config['model_dict']

    path_to_dog_brain_toolkit = os.path.join(git_folder, 'dog_brain_toolkit')
    rsa_model_path = datafolder + os.sep + dataset + os.sep + 'rsa_models' + os.sep + rsa_model + ".xlsx"
    sys.path.append(path_to_dog_brain_toolkit)
    import utils
    reload(utils)
    import preprocess_functions
    reload(preprocess_functions)
    import rsa_utils
    reload(rsa_utils)
    import utils_EmoB
    reload(utils_EmoB)


    project_dict = {
        "Dataset": config["dataset"],
        "Task": config["task"],
        "Participants": config["participants"],
        "Runs": config["runs"],
        "Sessions": config["sessions"],
        "Specie": config["specie"],
        "Atlas_type": config["atlas_type"],
        "Datafolder": datafolder,
    }

    # GLM parameters
    radius_fwd = config["radius_fwd"]
    threshold_fwd = config["threshold_fwd"]
    smooth = config["smooth"]
    img_type = config["img_type"]
    model = config["model"]
    model_dict = config['model_dict']
    participants = config["participants"]
    stim_types = config['stim_types']
    #atlas_type = config["atlas_type"]


    # list of missing files per subject/session/run
    missing_per_run_list = []

    if specie == 'D':
        specie_label = 'Dog'
        design_template = path_to_dog_brain_toolkit + os.sep + 'FSL_designs' + os.sep + 'basic_DHRF.fsf'
        # print(f"specie_label: {specie_label}, atlas_type: {atlas_type}, img_type: {img_type}")
        atlas_file = os.path.join(path_to_dog_brain_toolkit, 'Atlas', specie_label, atlas_type, f"{img_type}.nii.gz")
        atlas_for_labels = 'Czeibert' # takes for dogs: Czeibert, Johnson. For humans: AAL, Harvard
        atlas_type = 'Nitzsche'  # for dogs only Nitzsche is used for masks
        # label_dict = pd.read_excel(os.path.join(
        # path_to_dog_brain_toolkit, 'Atlas', 'Dog', f"{atlas_for_labels}_dictionary.xlsx"
    # ))
        label_nii_data = nib.load(os.path.join(
        path_to_dog_brain_toolkit, 'Atlas', 'Dog', 'Nitzsche', f"{atlas_for_labels}_labels2mm.nii.gz"
    )).get_fdata()
        
    elif specie == 'H':
        specie_label = 'Hum'
        atlas_type = 'MNI'
        design_template = path_to_dog_brain_toolkit + os.sep + 'FSL_designs' + os.sep + 'basic_H.fsf'
        # "C:\github\dog_brain_toolkit\Atlas\Hum\MNI152_T1_2mm_brain.nii.gz"
        atlas_file = os.path.join(path_to_dog_brain_toolkit, 'Atlas', 'Hum', "  MNI152_T1_2mm_brain.nii.gz")     
    else:
        raise ValueError("Specie must be 'D' for Dog or 'H' for Human")
    
    # This is the mask used for searchlight, determines which voxels are included
    mask = os.path.join(path_to_dog_brain_toolkit, 'Atlas', specie_label, atlas_type, mask_type + '.nii.gz')
    # if participants_forced is not empty, use only those participants
    if len(participants_forced) > 0:
        participants = participants_forced
        
    for step in steps_to_run:
        if step == 0: # compute beta maps by participant/session/run
            print("### Step 0: Computing beta maps ###")
            for sub_N in participants:
                session_and_run_dict = utils_EmoB.get_session_and_run_list(specie, sub_N)
                for entry in session_and_run_dict:
                    session = entry['session']
                    run_N = entry['run']
                    rsa_utils.calculate_beta_maps(datafolder, dataset, model, specie, sub_N, session, run_N, task,
                                stim_types, design_template, atlas_file,
                                smooth,
                                radius_fwd,
                                threshold_fwd,
                                redo_if_exists=False,
                                overwrite_movement=False)
                    print(f"Finished sub-{sub_N:02d} ses-{session:02d} run-{run_N:02d}...")
            print("#### Done computing beta maps ####")
        if step == 1: # compute pairwise similarity maps between beta maps by participant
            print("### Step 1: Computing pairwise similarity maps ###")
            for sub_N in participants:
                session_and_run_dict = utils_EmoB.get_session_and_run_list(specie, sub_N)
                rsa_utils.calculate_pairwise_similarity_maps2(datafolder, dataset, sub_N, session_and_run_dict,
                                    specie, model, stim_types, mask, task, radius, 
                                    method=method, replace_file=replace_file, mah_fold='run-wise', verbose=verbose)
                
                
                print(f"Finished sub-{sub_N:02d}...")
            print("#### Done computing pairwise similarity maps ####")

        if step == 2: # Compute similarity between pairwise similarity maps and a model by participant
            print("### Step 2: Computing similarity between pairwise similarity maps and a model ###")
            for sub_N in participants:
                session_and_run_dict = utils_EmoB.get_session_and_run_list(specie, sub_N)
                rsa_utils.compare_with_model2(datafolder, dataset, sub_N, session_and_run_dict,
                                    specie, model, stim_types,  mask, task, radius, rsa_model=rsa_model,
                                    method=method, rsa_method=rsa_method, replace_file=replace_file, 
                                    verbose=verbose, wait_time=wait_time, rnd=False,
                                    create_subject_mean=False)
                
                print(f"Finished sub-{sub_N:02d}...")
            print(f"#### Done computing similarity between pairwise maps and model ####")
        if step == 3: # Calculate group model similarity map
            print("### Step 3: Computing group model similarity map ###")
            print(f"Using RSA model: {rsa_model}, RSA method: {rsa_method}")
            # build session_and_run_all_dict    
            session_and_run_all_dict = {}
            for sub_N in participants:
                session_and_run_dict = utils_EmoB.get_session_and_run_list(specie, sub_N)
                session_and_run_all_dict[sub_N] = session_and_run_dict
            
            rsa_utils.calculate_group_model_similarity_map(datafolder, dataset, session_and_run_all_dict, specie, model, 
                                                task, radius, rsa_model=rsa_model,
                                                rsa_method=rsa_method,
                                                method=method, replace_file=True, verbose=verbose, 
                                                min_percentage_available=min_percentage_available
                                                )
            print("### Done computing group model similarity map ###")
        if step == 4: # Calculate rnd by repeating step 2 with permuted model
            print("### Step 4: Calculating permutations for model similarity maps ###")
            for sub_N in participants:
                session_and_run_dict = utils_EmoB.get_session_and_run_list(specie, sub_N)
                rsa_utils.compare_with_model2(datafolder, dataset, sub_N, session_and_run_dict,
                                    specie, model, stim_types,  mask, task, radius, rsa_model=rsa_model,
                                    method=method, rsa_method=rsa_method, replace_file=replace_file, 
                                    verbose=verbose, wait_time=wait_time, rnd=True, reps=reps,
                                    create_subject_mean=False, replace_rnd_files=True)
                
                print(f"Finished sub-{sub_N:02d}...")
            print(f"### Done computing rnd similarity between pairwise maps and model ###")
        if step == 5: # Calculate rnd mean model similarity maps by repeating step 3 with permuted models
            print("### Step 5: Calculating permutations of group model similarity maps ###")
            # build session_and_run_all_dict    
            session_and_run_all_dict = {}
            for sub_N in participants:
                session_and_run_dict = utils_EmoB.get_session_and_run_list(specie, sub_N)
                session_and_run_all_dict[sub_N] = session_and_run_dict
            print(f"rsa_model {rsa_model}")
            rsa_utils.calculate_group_model_similarity_map_rnd(datafolder, dataset, session_and_run_all_dict, specie, model, 
                                                task, radius, rsa_model=rsa_model,
                                                rsa_method=rsa_method,
                                                method=method, verbose=verbose, 
                                                min_percentage_available=min_percentage_available,
                                                reps=reps, replace_rnd_files=False, wait_time=300, reps_group=reps_group,
                                                )
            
            print("### Done computing group rnd mean model similarity maps ###")
            #Map computing
        if step == 6: # Calculate per voxel distribution. Load all group model similarity maps. Calculate per voxel mean and std across maps. Save as nifti.
            print("### Step 6: Calculating voxelwise distribution maps from permutations ###")
            verbose = False
            rsa_utils.calculate_voxelwise_rnd_distribution(datafolder, dataset, specie, model, task, radius,
                                        method=method, rsa_method=rsa_method,
                                        rsa_model=rsa_model, reps_group=reps_group,
                                        verbose=verbose)
            print("### Done computing per voxel rnd distribution ###")
        if step == 7: # Calculate z map for each mean rnd model similarity map
            print("### Step 7: Calculating z maps for permutations ###")
            rsa_utils.calculate_z_maps_rnd(datafolder, dataset, specie, model, task, radius,
                                        method=method, rsa_method=rsa_method,
                                        rsa_model=rsa_model,
                                        verbose=verbose, replace_file=True,
                                        reps_group=reps_group)
            ## Z -score mean from real data with mean and std from rnd distribution
            # "P:\userdata\raulh87\data\EmoB\results\RSA\basic\emotion-valence-basic\mean\D-r-3_mahalanobis_kendall_mean.nii.gz"
            rsa_utils.calculate_z_map_real_data(datafolder, dataset, specie, model, radius,
                                        method, rsa_method,
                                        rsa_model,
                                        verbose=verbose)
            
            print("### Done computing z maps for rnd distribution ###")
        if step == 8: # Threshold z maps, calculate cluster size distribution 
            print("### Step 8: Calculating cluster size distribution ###")
            # this function calculates the cluster size distribution
            rsa_utils.calculate_cluster_size_distribution(
            datafolder, dataset, model, rsa_model, radius, specie,
            method, rsa_method, z_threshold=z_threshold, verbose=verbose
            )
            
            print("### Done computing cluster size distribution ###")
        if step == 9: # Threshold z maps, apply cluster correction, save significant maps
            print("### Step 9: Applying cluster correction to z maps ###")
            forced_minimal_cluster_size = None
            # this function applies cluster correction to the real z map
            rsa_utils.apply_cluster_correction(datafolder, dataset, specie, model, rsa_model, radius=radius,
                                method=method, rsa_method=rsa_method, z_threshold=z_threshold, 
                                cluster_threshold=cluster_threshold, forced_minimal_cluster_size=forced_minimal_cluster_size,
                                verbose=verbose)
            
            print("### Done applying cluster correction to z maps ###")
        if step == 10: # Summarize results, create formatted report and save xlsx
            print("### Step 10: Summarizing results and saving to Excel ###")
            rsa_utils.create_tables(datafolder, dataset, specie, model, rsa_model, radius, 
                  method, rsa_method, min_dist_mm=min_dist_mm, max_peaks_per_cluster=3,
                  label_dict=label_dict, label_nii_data=label_nii_data)


if __name__ == "__main__":
    main()



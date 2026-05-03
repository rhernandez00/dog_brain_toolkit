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
Steps possible:
0: Compute beta maps by participant/session/run
1: Compute pairwise similarity maps between beta maps by participant
2: Compute similarity between pairwise similarity maps and a model by participant
3: Calculate group model similarity map
4: Calculate rnd by repeating step 2 with permuted model
5: Calculate rnd mean model similarity maps by repeating step 3 with permuted models
6: Calculate per voxel distribution. Load all group model similarity maps. Calculate per voxel mean and
std across maps. Save as nifti.
7: Calculate z map for each mean rnd model similarity map, will also calculate z map for real data using mean and std from rnd distribution
7.5: Calculate z map for real data using mean and std from rnd distribution (in case you want to do it separately)
8: Threshold z maps, calculate cluster size distribution
9: Threshold z maps, apply cluster correction, save significant maps
10: Summarize results, create formatted report and save xlsx
11: Calculate cross-participant similarity, use one participant as model and calculate similarity with other participants, repeat for all participants
12: DSM extraction: For each significant cluster, extract the similarity values 

# Keep adding numbers for steps, reorganize later for better structure


Input arguments:
--steps_to_run: List of steps to run (default: [1,2,3,4,5,6,7,8,9,10])
--model: GLM model to use (default: 'basic')
--method: Method for pairwise similarity calculation (default: 'mahalanobis')
--rsa_model: RSA model to use
--rsa_method: Method to compare similarity maps with model (default: 'kendall')
--specie: 'D' for Dog, 'H' for Human (default: 'H')
--mask_type: Type of brain mask to use (default: 'b_GreyMatter2mm')
--radius: Radius for searchlight (default: 3)
--z_threshold: Z threshold for z maps (default: 3.1)
--cluster_threshold: Cluster threshold for cluster correction (default: 0.05)
--peak_id: peak_id for running step 12, if not provided, it will calculate based on roi_database.csv

--reps: Number of repetitions for permutations in individual run (default: 100)
--reps_group: Number of repetitions for permutations in group analysis (default: 1000)
--min_percentage_available: Minimum percentage of database available to process (default: 0.8)
--min_dist_mm: Minimum distance between peaks in mm (default: 8.0)
--atlas_type: Type of atlas to use in case of dogs (default: 'Nitzsche')
--replace_file: Overwrite existing output files (default: False)
--shuffle_participants: shuffle participants order in permutations (default: False)

--participants_forced: List of participants to include (default: [])
--verbose: Verbose output (default: False)
--wait_time: Wait time between steps in seconds (default: 300)
--overwrite_movement: Overwrite existing movement files (default: False)
--skip_prefile_check: Skip prefile check and overwrite files if they exist (default: False)
'''

# parser function
def parse_arguments():
    parser = argparse.ArgumentParser(description='RSA Pipeline Execution')
    # parse dataset
    parser.add_argument('--dataset', type=str, default='EmoC',
                        help='Dataset to use')
    # parse task
    parser.add_argument('--task', type=str, default=None,
                        help='Task to use, if not provided, will use the same as dataset')
    parser.add_argument('--steps_to_run', type=int, nargs='+', default=[1,2,3,4,5,6,7,8,9,10],
                        help='List of steps to run')
    parser.add_argument('--model', type=str, default='basic',
                        help='GLM model to use')
    parser.add_argument('--method', type=str, default='mahalanobis',
                        help='Method for pairwise similarity calculation')
    parser.add_argument('--rsa_model', type=str, default=None, required=False,
                        help='RSA model to use')
    parser.add_argument('--rsa_method', type=str, default='kendall',
                        help='Method to compare similarity maps with model')
    parser.add_argument('--specie', type=str, default='H',
                        help="'D' for Dog, 'H' for Human")
    parser.add_argument('--mask_type', type=str, default='b_GreyMatter2mmB',
                        help='Type of brain mask to use')
    parser.add_argument('--radius', type=int, default=None,
                        help='Radius for searchlight')
    parser.add_argument('--z_threshold', type=float, default=3.1,
                        help='Z threshold for z maps')
    parser.add_argument('--cluster_threshold', type=float, default=0.05,
                        help='Cluster threshold for cluster correction')
    parser.add_argument('--peak_id', type=str, default=None,
                        help='peak_id for running step 12, if not provided, it will calculate based on roi_database.csv')
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
    parser.add_argument('--shuffle_participants', action='store_true',
                        help='Shuffle participants order in permutations')
    parser.add_argument('--shuffle_runs', action='store_true',
                        help='Shuffle runs order in permutations (only for step 1)')
    parser.add_argument('--participants_forced', type=int, nargs='+', default=[],
                        help='List of participants to include')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--wait_time', type=int, default=3000,
                        help='Wait time between steps in seconds')
    parser.add_argument('--overwrite_movement', action='store_true',
                        help='Overwrite existing movement files')
    parser.add_argument('--skip_prefile_check', action='store_true',
                        help='Skip prefile check and overwrite files if they exist')
    parser.add_argument('--coords', type=str, default=None,
                        help='Coordinates for similarity files in voxel space, format: x,y,z')
    parser.add_argument('--mah_fold', type=str, default='stim-wise',
                        help='Folding method for Mahalanobis distance, either "stim-wise" or "run-wise"')
    return parser.parse_args()

# main execution
def main():
    args = parse_arguments()
    steps_to_run = args.steps_to_run
    model = args.model
    # mask_path = args.mask_path
    method = args.method
    coords = args.coords
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
    skip_prefile_check = args.skip_prefile_check
    peak_id = args.peak_id
    dataset = args.dataset
    shuffle_runs = args.shuffle_runs
    mah_fold = args.mah_fold # this is the default, but it can be changed to 'run-wise' if needed
    # if task is not provided use the same as dataset
    if args.task is None:
        task = dataset
    else:
        task = args.task

    # # if user provided rsa_model
    # if rsa_model is not None:
    #     # check if rsa_model ends with -basic if it does, model is basic
    #     if rsa_model.endswith('-basic'):
    #         model = 'basic'
    #     else:
    #         model = 'basic-block'
    

    if os.name == 'nt':  # Windows
        datafolder = os.path.join(
            "P:\\userdata", 'raulh87', 'data'
        )
        git_folder = r"C:\github"
    else:
        datafolder = os.path.join(
            '/home', 'raulh87', 'mnt', 'a471', 'userdata', 'raulh87', 'data'
        )
        #'/home/raulh87/mnt/a471/userdata/raulh87/github
        git_folder = os.path.join('/home', 'raulh87', 'mnt', 'a471', 'userdata', 'raulh87', 'github')
        
    config_path = datafolder + os.sep + dataset + os.sep + 'config_files' + os.sep + specie + '_' + model + '.yaml'

    # Load config.yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # stim_types = config['stim_types']
    

    path_to_dog_brain_toolkit = os.path.join(git_folder, 'dog_brain_toolkit')
    
    sys.path.append(path_to_dog_brain_toolkit)
    import utils
    reload(utils)
    import preprocess_functions
    reload(preprocess_functions)
    import rsa_utils
    reload(rsa_utils)
    # import utils_EmoB
    # reload(utils_EmoB)


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
    # if model_dict is in config, get it, otherwise set it to None
    if "model_dict" in config:
        model_dict = config["model_dict"]
    else:
        model_dict = None
    
    participants = config["participants"]
    stim_types = config['stim_types']
    #atlas_type = config["atlas_type"]
    

    # if rsa_model is not None, get rsa_model_path
    if rsa_model is not None:
        rsa_model_path = datafolder + os.sep + dataset + os.sep + 'rsa_models' + os.sep + rsa_model + ".xlsx"
        # get categories from rsa_model definition
        rsa_model_dict = rsa_utils.read_model_dict(rsa_model_path)
        categories = rsa_model_dict['categories']


    # list of missing files per subject/session/run
    missing_per_run_list = []

    if specie == 'D':
        specie_label = 'Dog'
        apply_coords_transform=False #assumes that atlas and everything is tiun the same space
        design_template = path_to_dog_brain_toolkit + os.sep + 'FSL_designs' + os.sep + 'basic_DHRF.fsf'
        # print(f"specie_label: {specie_label}, atlas_type: {atlas_type}, img_type: {img_type}")
        atlas_file = os.path.join(path_to_dog_brain_toolkit, 'Atlas', specie_label, atlas_type, img_type + ".nii.gz")
        atlas_for_labels = 'Czeibert' # takes for dogs: Czeibert, Johnson. For humans: AAL, Harvard
        atlas_type = 'Nitzsche'  # for dogs only Nitzsche is used for masks
        # if radius is None, use 3 as default for dogs
        if radius is None:
            radius = 3
        # get label_dict and label_nii_data for dogs
        label_dict = pd.read_csv(os.path.join(
        path_to_dog_brain_toolkit, 'Atlas', 'Dog', f"{atlas_for_labels}_dictionary.csv"
    ))  

        label_nii_data = nib.load(os.path.join(
        path_to_dog_brain_toolkit, 'Atlas', 'Dog', 'Nitzsche', atlas_for_labels + "_labels2mm.nii.gz"
    )).get_fdata()
        
    elif specie == 'H':
        specie_label = 'Hum'
        apply_coords_transform=True #assumes that atlas and results are in different spaces, applies coordinate transform to get correct labels in excel output
        atlas_type = 'MNI'
        design_template = path_to_dog_brain_toolkit + os.sep + 'FSL_designs' + os.sep + 'basic_H.fsf'
        # "C:\github\dog_brain_toolkit\Atlas\Hum\MNI152_T1_2mm_brain.nii.gz"
        atlas_file = os.path.join(path_to_dog_brain_toolkit, 'Atlas', 'Hum', "MNI152_T1_2mm_brain.nii.gz")     
        
        participants = list(range(1, 41))
        

        # get label_nii_data and label_dict for humans
        label_nii_data = nib.load(os.path.join(
            datafolder, dataset, 'ROI', 'AAL3.nii.gz'
        )).get_fdata()
        # get label_dict
        # "P:\userdata\raulh87\github\dog_brain_toolkit\Atlas\Hum\AAL_dictionary.csv"
        label_dict = pd.read_csv(os.path.join(
            path_to_dog_brain_toolkit, 'Atlas', 'Hum', 'AAL_dictionary.csv'
        ))
        # for humans, if radius is None, use 4 as default
        if radius is None:
            radius = 4
        
    else:
        raise ValueError("Specie must be 'D' for Dog or '`H' for Human")
    
    # This is the mask used for searchlight, determines which voxels are included
    if specie == 'D':
        print(f"specie is D, mask_type: {mask_type}")
        if mask_type == 'cope13':
            ## ----mask should match with the beta maps (GLM space) ---
            mask = os.path.join(datafolder,dataset,'ROI',specie,'cope13.nii.gz')
        else:
            mask = os.path.join(path_to_dog_brain_toolkit, 'Atlas', specie_label, atlas_type, mask_type + '.nii.gz')
        
    elif specie == 'H':
        print(f"specie is H, mask_type: {mask_type}")
        if mask_type == 'cope13':
            print('getting cope 13')
            # get mask from GLM results, cope13
            # mask = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'GLM' + os.sep + 'initial' + os.sep + 'group' + os.sep + 'H-group_cope13.gfeat' + os.sep + 'cope1.feat' + os.sep + 'thresh_zstat1.nii.gz'
            # mask = r"P:\userdata\raulh87\data\EmoB\ROI\H\cope13.nii.gz"
            mask = os.path.join(datafolder,dataset,'ROI',specie,'cope13.nii.gz')
            #b_greyMatter2mmB
        else:
            print('not getting cope 13')
            # get path
            mask = datafolder + os.sep + dataset + os.sep + 'ROI' + os.sep + specie + os.sep + mask_type + '.nii.gz'
    print(f"Using mask: {mask}")
    # if participants_forced is not empty, use only those participants
    if len(participants_forced) > 0:
        participants = participants_forced
    
    # print size of label_nii_data
    print(f"label_nii_data shape: {label_nii_data.shape}")

    for step in steps_to_run:
        if step == 0: # compute beta maps by participant/session/run
            print("### Step 0: Computing beta maps ###")
            # if specie == H warn and skip
            if specie == 'H':
                print("Warning: Step 0 (computing beta maps) is done another way in humans. Skipping...")
                continue

            for sub_N in participants:
                session_and_run_dict = rsa_utils.get_session_and_run_dict(datafolder, dataset, specie, sub_N)
                # session_and_run_dict = utils_EmoB.get_session_and_run_list(specie, sub_N)
                for entry in session_and_run_dict:
                    session = entry['session']
                    run_N = entry['run_N']
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

            # if mask is not b_greyMatter2mm or b_greyMatter2mmB, turn it into b_greyMatter2mmB
            if mask_type not in ['b_GreyMatter2mm', 'b_GreyMatter2mmB']:
                print(f"Mask type is {mask_type}, converting to b_GreyMatter2mmB for searchlight...")
                mask2 = os.path.join(datafolder, dataset, 'ROI', specie, 'b_GreyMatter2mmB.nii.gz')
            else: # if mask is already b_greyMatter2mm or b_greyMatter2mmB, use it directly
                mask2 = mask
                

            if args.shuffle_participants:
                np.random.shuffle(participants)
                # print(f"Shuffled participants order: {participants}")
            for sub_N in participants:
                session_and_run_dict = rsa_utils.get_session_and_run_dict(datafolder, dataset, specie, sub_N)
                if mah_fold == 'run-wise-multiple-runs': # for each run, run once calculate_pairwise_similarity_maps2
                    print("Calculating pairwise similarity maps with run-wise folding for multiple runs...")
                    for entry in session_and_run_dict:
                        session = entry['session']
                        run_N = entry['run_N']
                        print(f"Processing sub-{sub_N:02d} ses-{session:02d} run-{run_N:02d}...")
                        rsa_utils.calculate_pairwise_similarity_maps2(datafolder, dataset, sub_N, [entry],
                                    specie, model, stim_types, mask2, task, radius=radius, 
                                    method=method, replace_file=replace_file, mah_fold=mah_fold, 
                                    verbose=verbose, skip_prefile_check=skip_prefile_check, 
                                    categories=categories, shuffle_runs=shuffle_runs, model_dict=model_dict)

                else:
                    rsa_utils.calculate_pairwise_similarity_maps2(datafolder, dataset, sub_N, session_and_run_dict,
                                    specie, model, stim_types, mask2, task, radius=radius, 
                                    method=method, replace_file=replace_file, mah_fold=mah_fold, 
                                    verbose=verbose, skip_prefile_check=skip_prefile_check, categories=categories, shuffle_runs=shuffle_runs, model_dict=model_dict)
                
                
                print(f"Finished sub-{sub_N:02d}...")
            print("#### Done computing pairwise similarity maps ####")

        if step == 2: # Compute similarity between pairwise similarity maps and a model by participant
            print("### Step 2: Computing similarity between pairwise similarity maps and a model ###")
            if args.shuffle_participants:
                np.random.shuffle(participants)
                print(f"Shuffled participants order: {participants}")
            for sub_N in participants:
                session_and_run_dict = rsa_utils.get_session_and_run_dict(datafolder, dataset, specie, sub_N)
                rsa_utils.compare_with_model2(datafolder, dataset, sub_N, session_and_run_dict,
                                    specie, model, stim_types,  mask, task, radius, rsa_model=rsa_model,
                                    method=method, rsa_method=rsa_method, replace_file=replace_file, 
                                    verbose=verbose, wait_time=wait_time, rnd=False,
                                    create_subject_mean=False, mask_type=mask_type)
                
                print(f"Finished sub-{sub_N:02d}...")
            print(f"#### Done computing similarity between pairwise maps and model ####")
        if step == 3: # Calculate group model similarity map
            print("### Step 3: Computing group model similarity map ###")
            print(f"Using RSA model: {rsa_model}, RSA method: {rsa_method}")
            # build session_and_run_all_dict    
            session_and_run_all_dict = {}
            for sub_N in participants:
                session_and_run_dict = rsa_utils.get_session_and_run_dict(datafolder, dataset, specie, sub_N)
                session_and_run_all_dict[sub_N] = session_and_run_dict
            
            rsa_utils.calculate_group_model_similarity_map(datafolder, dataset, session_and_run_all_dict, specie, model, 
                                                task, radius, rsa_model=rsa_model,
                                                rsa_method=rsa_method,
                                                method=method, replace_file=True, verbose=verbose, 
                                                min_percentage_available=min_percentage_available, mask_type=mask_type
                                                )
            print("### Done computing group model similarity map ###")
        if step == 4: # Calculate rnd by repeating step 2 with permuted model
            print("### Step 4: Calculating permutations for model similarity maps ###")
            # if shuffle_participants is true, shuffle participants order
            if args.shuffle_participants:
                np.random.shuffle(participants)
                print(f"Shuffled participants order: {participants}")
            for sub_N in participants:
                session_and_run_dict = rsa_utils.get_session_and_run_dict(datafolder, dataset, specie, sub_N)
                rsa_utils.compare_with_model2(datafolder, dataset, sub_N, session_and_run_dict,
                                    specie, model, stim_types,  mask, task, radius, rsa_model=rsa_model,
                                    method=method, rsa_method=rsa_method, replace_file=replace_file, 
                                    verbose=verbose, wait_time=wait_time, rnd=True, reps=reps,
                                    create_subject_mean=False, replace_rnd_files=False)
                
                print(f"Finished sub-{sub_N:02d}...")
            print(f"### Done computing rnd similarity between pairwise maps and model ###")
        if step == 5: # Calculate rnd mean model similarity maps by repeating step 3 with permuted models
            print("### Step 5: Calculating permutations of group model similarity maps ###")
            # build session_and_run_all_dict    
            session_and_run_all_dict = {}
            for sub_N in participants:
                session_and_run_dict = rsa_utils.get_session_and_run_dict(datafolder, dataset, specie, sub_N)
                session_and_run_all_dict[sub_N] = session_and_run_dict
            print(f"rsa_model {rsa_model}")
            rsa_utils.calculate_group_model_similarity_map_rnd(datafolder, dataset, session_and_run_all_dict, specie, model, 
                                                task, radius, rsa_model=rsa_model,
                                                rsa_method=rsa_method,
                                                method=method, verbose=verbose, 
                                                min_percentage_available=min_percentage_available,
                                                reps=reps, replace_rnd_files=False, wait_time=300, reps_group=reps_group)
            
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
            rsa_utils.calculate_z_map_real_data(datafolder, dataset, specie, 
                                                model, radius,
                                        method, rsa_method,
                                        rsa_model, verbose=verbose, mask_type=mask_type)
        if step == 75:
            print("### Step 75 (75 actually): Calculating z map for real data ###")
            rsa_utils.calculate_z_map_real_data(datafolder, dataset, specie, model, radius,
                                        method, rsa_method,
                                        rsa_model, verbose=verbose, mask_type=mask_type)
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
                                verbose=verbose, mask_type=mask_type)
            
            print("### Done applying cluster correction to z maps ###")
        if step == 10: # Summarize results, create formatted report and save xlsx
            print("### Step 10: Summarizing results and saving to Excel ###")
            rsa_utils.create_tables(datafolder, dataset, specie, model, rsa_model, radius, 
                  method, rsa_method, min_dist_mm=min_dist_mm, max_peaks_per_cluster=3,
                  label_dict=label_dict, label_nii_data=label_nii_data, 
                  apply_coords_transform=apply_coords_transform, atlas_file=atlas_file, mask=mask, 
                  mask_type=mask_type)
        if step == 11: # Calculate cross-participant similarity, use one participant as model and calculate similarity with other participants, repeat for all participants
            print("### Step 11: Calculating cross-participant similarity ###")
            session_and_run_all_dict = {}
            for sub_N in participants:
                session_and_run_dict = rsa_utils.get_session_and_run_dict(datafolder, dataset, specie, sub_N)
                session_and_run_all_dict[sub_N] = session_and_run_dict
            rsa_utils.calculate_cross_participant_similarity(datafolder, dataset, session_and_run_all_dict, participants, specie,
                                                            mask, model, task, radius, method=method, rsa_method=rsa_method,
                                                            rsa_model=rsa_model, verbose=verbose)
            print("### Done computing cross-participant similarity ###")
        if step == 12: # Calculate similarity across all pairs in a model, save files as txt
            print("### Step 12: DSM extraction: ###")
            # load roi_database
            # "P:\userdata\raulh87\data\EmoB\ROI\roi_database.csv"
            roi_database_path = os.path.join(
                datafolder, dataset, 'ROI', 'roi_database.csv'
            )
            roi_database = pd.read_csv(roi_database_path)
            # filter roi_database to include only rows with matching specie
            roi_database = roi_database[roi_database['specie'] == specie]

            # get session and run list for all participants and save in a dict
            session_and_run_all_dict = {}
            for sub_N in participants:
                session_and_run_dict = rsa_utils.get_session_and_run_dict(datafolder, dataset, specie, sub_N)
                session_and_run_all_dict[sub_N] = session_and_run_dict
            # if peak_id is not provided, iterate over roi_database and calculate for each peak_id
            if peak_id is None:
                for index, row in roi_database.iterrows():
                    print(f"Calculating similarity for peak_id {row['peak_id']} at coordinates {row['voxel_coords']}...")
                    voxel_coords = row['voxel_coords']
                    # voxel_coors is a string in format like this: "(x, y, z)", convert it to a tuple of integers
                    voxel_coords = voxel_coords.strip("()") # remove parentheses
                    voxel_coords = voxel_coords.split(",") # split by comma
                    voxel_coords = tuple(int(coord) for coord in voxel_coords) # convert to tuple of integers

                    print(f"voxel_coords: {voxel_coords}")
                    print(f"type of voxel_coords: {type(voxel_coords)}")
                    rsa_utils.calculate_similarity_across_all_pairs(datafolder, dataset, session_and_run_all_dict=session_and_run_all_dict, participants=participants, specie=specie,
                                                mask=mask, model=model, task=task, radius=radius, method=method, 
                                                rsa_model=rsa_model, voxel_coords=voxel_coords, config_path=config_path, verbose=True, shuffle_participants=True,
                                                shuffle_runs=shuffle_runs, wait_time=wait_time)
            print("### Done computing similarity across all pairs in a model ###")
        if step == 13: # get movement .par files from each run using mcflirt outputs
            print("### Step 13: running mcflirt to get movement parameters (.par) ###")
            # go over each participant
            for sub_N in participants:
                session_and_run_dict = rsa_utils.get_session_and_run_dict(datafolder, dataset, specie, sub_N)
                for entry in session_and_run_dict:
                    session = entry['session']
                    run_N = entry['run_N']
                    rsa_utils.calculate_movement_parameters(datafolder, dataset, task, specie, sub_N, entry, verbose=verbose)
                    # calculate fwd files, for D has movement parameters + fwd, for humans only fwd
                    par_file = os.path.join(datafolder, dataset, 'preprocessing', f"{specie}_mcflirt",
                                             f"{specie}-sub-{sub_N:02d}_ses-{session:02d}_task-{task}_run-{run_N:02d}.par"
                                        )
                    mov_txt = os.path.join(datafolder, dataset, 'movement',
                                            f"{specie}-sub-{sub_N:02d}_ses-{session:02d}_task-{task}_run-{run_N:02d}_fwd.txt")
                    if specie == 'D':
                        preprocess_functions.fwd(par_file, radius_fwd, threshold_fwd, output_file=mov_txt, add_movement_params=True)
                    elif specie == 'H':
                        preprocess_functions.fwd(par_file, radius_fwd, threshold_fwd, output_file=mov_txt, add_movement_params=False)
                    else:
                        raise ValueError("Specie must be 'D' for Dog or '`H' for Human")
    print("### All steps completed! ###")
        


if __name__ == "__main__":
    main()



# This will calculate difference maps between classes
# 07/Feb/2026
# step 0. Compute beta maps by participant/session/run (double check, I haven't updated this)
# step 1. Compute pairwise similarity between two beta maps
# input: two beta maps.
# output: pairwise similarity map (e.g., Pearson correlation between stim A and B) and log with details.
# step 2. Group similarity maps by class (e.g., average similarity map for all happy pairs, all sad pairs, etc.)

import os
import time
import numpy as np
import nibabel as nib
import yaml
import sys
from importlib import reload
import pandas as pd
import argparse
'''
Input parameters:
'''
# parser function
def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate searchlight similarity maps and difference maps for RSA analysis.')
    parser.add_argument('--steps_to_run', nargs='+', type=int, default=[0,1,2,3,4],
                        help='Steps to run. Options: 0 (compute beta maps), 1 (compute pairwise similarity maps), 2 (compute group similarity maps), 3 (compute group difference map), 4 (compute permutations for group similarity map). Default: all steps [0,1,2,3,4]')
    parser.add_argument('--model', type=str, default='basic-block',
                        help='Model name to use for RSA analysis. Default: basic-block')
    parser.add_argument('--dis_method', type=str, default='mahalanobis',
                        help='Method for pairwise similarity calculation. Options: pearson, kendall, euclidean, mahalanobis, correlation. Default: mahalanobis')
    parser.add_argument('--specie', type=str, default='D',
                        help='Specie to analyze. Options: D (Dog), H (Human). Default: D')
    parser.add_argument('--mask_type', type=str, default='b_GreyMatter2mm',
                        help='Type of brain mask to use. Default: b_GreyMatter2mm')
    parser.add_argument('--radius', type=int, default=3,
                        help='Radius for searchlight analysis. Default: 3')
    parser.add_argument('--atlas_type', type=str, default='Nitzsche',
                        help='Type of atlas to use for labeling. Default: Nitzsche for dogs, MNI for humans')
    parser.add_argument('--dataset', type=str, default='EmoB',
                        help='Dataset name. Default: EmoB')
    # assign None as default in comparison_model, if running steps 0 or 1, fine, if running step 2 or 3, need to specify comparison_model    parser.add_argument('--comparison_model', type=str, 
    parser.add_argument('--comparison_model', type=str, default=None,
                        help='Comparison model name to use for group similarity and difference map calculation. Required for steps 2 and 3. Default: None')
    parser.add_argument('--replace_file', action='store_true', default=False,
                        help='Whether to replace existing files. Default: False')
    parser.add_argument('--participants_forced', type=str, default=None,
                        help='Participants to force include in analysis. Default: None')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to print verbose output. Default: False')
    
    # 
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    steps_to_run = args.steps_to_run
    model = args.model
    # mask_path = args.mask_path
    dis_method = args.dis_method
    specie = args.specie
    mask_type = args.mask_type
    radius = args.radius
    dataset = args.dataset
    atlas_type = args.atlas_type
    replace_file = args.replace_file
    participants_forced = args.participants_forced
    verbose = args.verbose
    comparison_model = args.comparison_model
    task = dataset
    
    


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
      
    config_path = datafolder + os.sep + dataset + os.sep + 'config_files' + os.sep + model + '.yaml'

    # Load config.yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # stim_types = config['stim_types']
    model_dict = config['model_dict']

    path_to_dog_brain_toolkit = os.path.join(git_folder, 'dog_brain_toolkit')
    
    sys.path.append(path_to_dog_brain_toolkit)
    import utils
    reload(utils)
    import preprocess_functions
    reload(preprocess_functions)
    import rsa_utils
    reload(rsa_utils)
    import utils_EmoB
    reload(utils_EmoB)

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
        atlas_file = os.path.join(path_to_dog_brain_toolkit, 'Atlas', specie_label, atlas_type, img_type + ".nii.gz")
        atlas_for_labels = 'Czeibert' # takes for dogs: Czeibert, Johnson. For humans: AAL, Harvard
        atlas_type = 'Nitzsche'  # for dogs only Nitzsche is used for masks
        # label_dict = pd.read_excel(os.path.join(
        # path_to_dog_brain_toolkit, 'Atlas', 'Dog', f"{atlas_for_labels}_dictionary.xlsx"
    # ))
        label_nii_data = nib.load(os.path.join(
        path_to_dog_brain_toolkit, 'Atlas', 'Dog', 'Nitzsche', atlas_for_labels + "_labels2mm.nii.gz"
    )).get_fdata()
        
    elif specie == 'H':
        specie_label = 'Hum'
        atlas_type = 'MNI'
        design_template = path_to_dog_brain_toolkit + os.sep + 'FSL_designs' + os.sep + 'basic_H.fsf'
        # "C:\github\dog_brain_toolkit\Atlas\Hum\MNI152_T1_2mm_brain.nii.gz"
        atlas_file = os.path.join(path_to_dog_brain_toolkit, 'Atlas', 'Hum', "  MNI152_T1_2mm_brain.nii.gz")     
        # force to participants to be 1-40 (exclude 36)
        participants = [i for i in range(1, 41) if i != 36]
        # "C:\github\dog_brain_toolkit\Atlas\Hum\AAL3.nii.gz"
        # get  label_nii_data path P:\userdata\raulh87\data\EmoB\ROI\AAL3.nii.gz
        label_nii_data = nib.load(os.path.join(
            datafolder, dataset, 'ROI', 'AAL3.nii.gz'
        )).get_fdata()
        # get label_dict
        # "P:\userdata\raulh87\github\dog_brain_toolkit\Atlas\Hum\AAL_dictionary.csv"
        label_dict = pd.read_csv(os.path.join(
            path_to_dog_brain_toolkit, 'Atlas', 'Hum', 'AAL_dictionary.csv'
        ))
   
    else:
        raise ValueError("Specie must be 'D' for Dog or '`H' for Human")
    
    # This is the mask used for searchlight, determines which voxels are included
    if specie == 'D':
        print(f"specie is D, mask_type: {mask_type}")
        if mask_type == 'cope13':
            # "P:\userdata\raulh87\data\EmoB\ROI\Cope13-Emo-Con_Z3.1.nii.gz"
            # mask = os.path.join(datafolder, dataset, 'ROI', 'Cope13-Emo-Con_Z3.1.nii.gz')
            # "P:\userdata\raulh87\data\EmoB\results\GLM\initial\group\H-group_cope13.gfeat\cope1.feat\thresh_zstat1.nii.gz"
            # "P:\userdata\raulh87\data\EmoB\ROI\H\cope13.nii.gz"
            mask = os.path.join(datafolder,dataset,'ROI',specie,'cope13.nii.gz')
        else:
            mask = os.path.join(path_to_dog_brain_toolkit, 'Atlas', specie_label, atlas_type, mask_type + '.nii.gz')
        
    elif specie == 'H':
        print(f"specie is H, mask_type: {mask_type}")
        if mask_type == 'cope13':
            print('getting cope 13')
            mask = os.path.join(datafolder,dataset,'ROI',specie,'cope13.nii.gz')
        else:
            print('not getting cope 13')
            # get path
            mask = datafolder + os.sep + dataset + os.sep + 'ROI' + os.sep + specie + os.sep + mask_type + '.nii.gz'
    print(f"Using mask: {mask}")
    # if participants_forced is not empty, use only those participants
    if len(participants_forced) > 0:
        participants = participants_forced
    
    # load mask image and binarize
    # mask_img = nib.load(mask_path).get_fdata()
    # mask_img = (mask_img > 0).astype(int)

    for step in steps_to_run:
        if step == 0: # compute beta maps by participant/session/run
            # issue error, not implemented yet, check if beta maps have been computed, if not, compute them.
            raise NotImplementedError("Step 0: Computing beta maps not implemented yet.")

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
                                    dis_method=dis_method, replace_file=replace_file, mah_fold='run-wise', verbose=verbose)
                
                
                print(f"Finished sub-{sub_N:02d}...")
            print("#### Done computing pairwise similarity maps ####")

        if step == 2: # Calculate group model similarity map
            print("### Step 2: Computing group similarity maps ###")
            # build session_and_run_all_dict    
            session_and_run_all_dict = {}
            for sub_N in participants:
                session_and_run_dict = utils_EmoB.get_session_and_run_list(specie, sub_N)
                session_and_run_all_dict[sub_N] = session_and_run_dict
            
            rsa_utils.calculate_similarity_maps_by_class(datafolder=datafolder, dataset=dataset, session_and_run_all_dict=session_and_run_all_dict,
                                        specie=specie, model=model, comparison_model=comparison_model, model_dict=model_dict, task=task, radius=radius,
                                            dis_method=dis_method, mask=mask,
                                            replace_file=False, min_percentage_available=1.0,
                                            verbose=True, avoid_pairs_by_label='actor')
        if step == 3:  # Calculate group model similarity map by stim list
            print("### Step 3: Computing group model difference map###")
            rsa_utils.calculate_difference_map(datafolder, dataset, specie, model,
                                            comparison_model, radius, dis_method,
                                            replace_file=False, verbose=True)


            # rsa_utils.calculate_similarity_maps_by_list(datafolder, dataset, session_and_run_all_dict,
            #                         specie, model, stim_list, category_name, mask_img, task, radius,
            #                         dis_method, replace_file=False, min_percentage_available=1.0,
            #                         verbose=False)
            print("### Done computing group model similarity map ###")
        if step == 4: # Calculate permutations for group model similarity map
            print("### Step 4: Computing group similarity map permutations ###")
            rsa_utils.calculate_similarity_maps_by_group_rnd(datafolder=datafolder, dataset=dataset, session_and_run_all_dict=session_and_run_all_dict,
                                        specie=specie, model=model, comparison_model=comparison_model, model_dict=model_dict, task=task, radius=radius,
                                            stim_types=stim_types, dis_method=dis_method, mask_img=mask_img,
                                            replace_file=False, min_percentage_available=1.0,
                                            verbose=True, avoid_pairs_by_label='actor')
            print("### Done computing group similarity map permutations ###")
    
# create main
if __name__ == "__main__":
    main()
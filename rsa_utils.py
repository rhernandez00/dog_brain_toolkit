import utils
import glob
import os
import sys
import nibabel as nib
import yaml
import pandas as pd
import numpy as np
import os
import numpy as np
from nilearn.image import resample_to_img
from time import time, perf_counter
import shutil
# import random
import random
from scipy.ndimage import label, generate_binary_structure
import warnings
import numpy as np
from scipy import ndimage                                                                                                                                      
import preprocess_functions


def _get_emoc_multiple_fold_stim_files(session_and_run_dict, model_dict):
    """Return EmoC condition IDs and their common cross-validation folds."""
    if not isinstance(model_dict, dict):
        raise ValueError("EmoC stim-wise-multiple-folds requires config['model_dict'].")

    stim_files = []
    seen = set()
    stim_partitions = {}
    for entry in session_and_run_dict:
        run_key = f"run{int(entry['run_N']):02d}"
        run_dict = model_dict.get(run_key)
        if not isinstance(run_dict, dict):
            raise ValueError(f"model_dict is missing metadata for {run_key}.")
        for stim, metadata in run_dict.items():
            if not isinstance(metadata, dict):
                raise ValueError(f"model_dict[{run_key!r}][{stim!r}] must be a dictionary.")
            stim_file = metadata.get('stim_file')
            if not stim_file or 'partition' not in metadata:
                raise ValueError(
                    f"model_dict[{run_key!r}][{stim!r}] requires 'stim_file' and 'partition'."
                )
            if stim_file not in seen:
                stim_files.append(stim_file)
                seen.add(stim_file)
                stim_partitions[stim_file] = set()
            stim_partitions[stim_file].add(metadata['partition'])

    repeated_stim_files = [
        stim_file for stim_file in stim_files if len(stim_partitions[stim_file]) >= 2
    ]
    if len(repeated_stim_files) < 2:
        raise ValueError("EmoC stim-wise-multiple-folds requires at least two repeated stimuli.")

    fold_partitions = set.intersection(
        *(stim_partitions[stim_file] for stim_file in repeated_stim_files)
    )
    if len(fold_partitions) < 2:
        raise ValueError(
            "Repeated EmoC stimuli do not share at least two partitions for cross-validation."
        )
    return repeated_stim_files, fold_partitions


def _rsa_model_output_dir(datafolder, dataset, model, rsa_model, dis_method,
                          mah_fold='stim-wise', rnd=False):
    """Return the fold-isolated root for downstream model-comparison outputs."""
    results_kind = 'RSA_rnd' if rnd else 'RSA'
    output_dir = os.path.join(
        datafolder, dataset, 'results', results_kind, model, rsa_model
    )
    if dis_method == 'mahalanobis' and mah_fold not in (None, 'stim-wise'):
        output_dir = os.path.join(output_dir, mah_fold)
    return output_dir


def _model_similarity_map_file(datafolder, dataset, specie, sub_N, model,
                               rsa_model, task, radius, dis_method,
                               rsa_method, mask_type=None,
                               mah_fold='stim-wise', rnd=False, session=None,
                               run_N=None, rnd_index=None):
    """Build a participant model-similarity path for any supported fold layout."""
    output_dir = os.path.join(
        _rsa_model_output_dir(
            datafolder, dataset, model, rsa_model, dis_method, mah_fold, rnd
        ),
        f"{specie}-sub-{sub_N:02d}",
    )
    per_run = dis_method != 'mahalanobis' or mah_fold == 'stim-wise-all-runs'
    if per_run:
        if session is None or run_N is None:
            raise ValueError("Per-run model-similarity paths require session and run_N.")
        output_dir = os.path.join(
            output_dir,
            f"ses-{int(session):02d}_task-{task}_run-{int(run_N):02d}",
        )
    filename = f"r-{radius}_{dis_method}_{rsa_method}"
    if mask_type:
        filename = f"{mask_type}-{filename}"
    if rnd_index is not None:
        filename = f"{filename}_{rnd_index:04d}"
    return os.path.join(output_dir, f"{filename}.nii.gz")


def _strip_emoc_exemplar(condition):
    """Return an EmoC class key without its trailing exemplar number."""
    suffix_start = len(condition)
    while suffix_start > 0 and condition[suffix_start - 1].isdigit():
        suffix_start -= 1
    if suffix_start == 0 or suffix_start == len(condition):
        raise ValueError(
            f"EmoC condition {condition!r} must end with its exemplar number."
        )
    return condition[:suffix_start]


def _get_emoc_multiple_fold_model_labels(session_and_run_dict, model_dict,
                                         model_categories):
    """Map repeatable exact stimuli to categories understood by an RSA model."""
    stim_files, _ = _get_emoc_multiple_fold_stim_files(
        session_and_run_dict, model_dict
    )
    eligible_categories = set(model_categories)
    model_labels = {}
    for entry in session_and_run_dict:
        run_key = f"run{int(entry['run_N']):02d}"
        run_dict = model_dict[run_key]
        for stim, metadata in run_dict.items():
            stim_file = metadata['stim_file']
            if stim_file not in stim_files:
                continue
            class_label = _strip_emoc_exemplar(stim)
            candidates = (stim_file, stim, class_label)
            model_label = next(
                (candidate for candidate in candidates
                 if candidate in eligible_categories),
                None,
            )
            if model_label is None:
                continue
            existing_label = model_labels.get(stim_file)
            if existing_label is not None and existing_label != model_label:
                raise ValueError(
                    f"Exact stimulus {stim_file!r} maps inconsistently to RSA "
                    f"categories {existing_label!r} and {model_label!r}."
                )
            model_labels[stim_file] = model_label

    selected_stim_files = [
        stim_file for stim_file in stim_files if stim_file in model_labels
    ]
    if len(selected_stim_files) < 2:
        raise ValueError(
            "The RSA model has fewer than two categories represented by the "
            "repeatable exact EmoC stimuli."
        )
    return selected_stim_files, model_labels


def _get_mahalanobis_model_plan(mah_fold, session_and_run_dict, model_dict,
                                stim_types, model_categories, entry=None):
    """Return map labels and their RSA-model labels for a Mahalanobis fold."""
    if mah_fold == 'stim-wise':
        return list(model_categories), {
            category: category for category in model_categories
        }
    if mah_fold == 'stim-wise-multiple-folds':
        return _get_emoc_multiple_fold_model_labels(
            session_and_run_dict, model_dict, model_categories
        )
    if mah_fold == 'stim-wise-all-runs':
        if entry is None:
            raise ValueError("Within-run Mahalanobis model comparison requires a run entry.")
        run_key = f"run{int(entry['run_N']):02d}"
        run_dict = model_dict.get(run_key) if isinstance(model_dict, dict) else None
        class_labels, _, _ = _get_emoc_within_run_class_folds(run_dict, stim_types)
        selected_labels = [
            class_label for class_label in class_labels
            if class_label in set(model_categories)
        ]
        if len(selected_labels) < 2:
            raise ValueError(
                f"RSA model has fewer than two classes available in {run_key}."
            )
        return selected_labels, {
            class_label: class_label for class_label in selected_labels
        }
    raise ValueError(
        "Mahalanobis model comparison supports 'stim-wise', "
        "'stim-wise-multiple-folds', and 'stim-wise-all-runs'."
    )


def _get_rsa_model_value(rsa_model_dict, category_a, category_b):
    """Read a symmetric RSA model value, using zero for a diagonal pair."""
    if category_a == category_b:
        return 0.0
    model_values = rsa_model_dict['model']
    try:
        return model_values[category_a][category_b]
    except KeyError:
        try:
            return model_values[category_b][category_a]
        except KeyError as error:
            raise KeyError(
                f"RSA model has no value for {category_a!r} vs {category_b!r}."
            ) from error


def _build_mahalanobis_model_vector(map_labels, model_labels, rsa_model_dict):
    """Expand the RSA model into the exact pair order used by pairwise maps."""
    pairs = []
    values = []
    for index, label_a in enumerate(map_labels):
        for label_b in map_labels[index + 1:]:
            pairs.append((label_a, label_b))
            values.append(_get_rsa_model_value(
                rsa_model_dict, model_labels[label_a], model_labels[label_b]
            ))
    if not pairs:
        raise ValueError("At least one map pair is required for model comparison.")
    return pairs, np.asarray(values, dtype=float)


def check_existing_similarity_maps(datafolder, dataset, session_and_run_dict, specie, sub_N, model, task, radius, dis_method, stim_types, categories, mah_fold=None, model_dict=None, verbose=False):
    '''
    Checks for existing similarity map files {specie}-sub-{sub_N:02d}
    returns:
    - True if all files exist, False if any file is missing
    '''
    if dis_method != 'mahalanobis':
        ## check if output files already exist
        print(f"Checking for existing similarity map files {specie}-sub-{sub_N:02d}...")
        all_exist = True

        for indx, entry in enumerate(session_and_run_dict):
            session = entry['session']
            run_N = entry['run_N']
            # correct session to 2 digits
            session = f"{session:02d}"
            for i, stim_i in enumerate(stim_types):
                for j, stim_j in enumerate(stim_types):
                    if i >= j:
                        continue  # avoid duplicates and self-comparison
                    output_file = os.path.join(
                        datafolder, dataset, 'results', 'RSA', model,
                        f"{specie}-sub-{sub_N:02d}",
                        f"ses-{session}_task-{task}_run-{run_N:02d}",
                        f"r-{radius}_{dis_method}_{stim_i}_{stim_j}.nii.gz"
                    )
                    # check if output_file exists
                    if not os.path.exists(output_file):
                        if verbose:
                            print(f"Not found {output_file}.")
                        all_exist = False
                    else:
                        if verbose:
                            print(f"Found {output_file} exists.")
    else: # dis_method is mahalanobis
        # dis_method is mahalanobis, check if mah_fold is stim-wise, if so check for output files accordingly
        if mah_fold == 'stim-wise':#
            print("Checking for existing Mahalanobis similarity map files with stim-wise folding...")
            all_exist = True
            # go over all pairs of categories
            for i, cat1 in enumerate(categories):
                for j, cat2 in enumerate(categories):
                    if i >= j:
                        continue  # avoid duplicates and self-comparison
                    output_file = os.path.join(
                        datafolder, dataset, 'results', 'RSA', model,
                        f"{specie}-sub-{sub_N:02d}",
                        f"r-{radius}_{dis_method}_{cat1}_{cat2}.nii.gz"
                    )
                    # check if file exists
                    if not os.path.exists(output_file):
                        if verbose:
                            print(f"Not found {output_file}.")
                        all_exist = False
                    else:
                        if verbose:
                            print(f"Found {output_file} exists.")
        elif mah_fold == 'stim-wise-multiple-folds':
            if dataset != 'EmoC':
                raise ValueError("mah_fold option 'stim-wise-multiple-folds' is only implemented for EmoC.")
            categories, _ = _get_emoc_multiple_fold_stim_files(session_and_run_dict, model_dict)
            print("Checking for existing Mahalanobis similarity map files with stim-wise multiple folds...")
            all_exist = True

            # pairwise file will be associated to a specific stimuli repeated across runs, output is a per-stimuli pair
            # go over all pairs of stims, categories is a list of stims available
            for i, stim1 in enumerate(categories):
                for j, stim2 in enumerate(categories):
                    if i >= j:
                        continue  # avoid duplicates and self-comparison
                    output_file = os.path.join(
                        datafolder, dataset, 'results', 'RSA', model,
                        f"{specie}-sub-{sub_N:02d}",
                        f"r-{radius}_{dis_method}_{stim1}_{stim2}.nii.gz"
                    )
                    # check if file exists
                    if not os.path.exists(output_file):
                        if verbose:
                            print(f"Not found {output_file}.")
                        all_exist = False
                    else:
                        if verbose:
                            print(f"Found {output_file} exists.")
        elif mah_fold == 'stim-wise-all-runs':
            if dataset != 'EmoC':
                raise ValueError(f"mah_fold option 'stim-wise-all-runs' is only implemented for dataset 'EmoC'. For dataset 'EmoB', use 'stim-wise'.")
            print("Checking for existing within-run Mahalanobis class maps...")
            all_exist = True
            for entry in session_and_run_dict:
                session = entry['session']
                run_N = entry['run_N']
                session = f"{session:02d}"
                run_key = f"run{int(run_N):02d}"
                run_dict = model_dict.get(run_key) if isinstance(model_dict, dict) else None
                run_categories, _, _ = _get_emoc_within_run_class_folds(
                    run_dict, stim_types
                )
                for i, cat1 in enumerate(run_categories):
                    for j, cat2 in enumerate(run_categories):
                        if i >= j:
                            continue  # avoid duplicates and self-comparison
                        output_file = os.path.join(
                            datafolder, dataset, 'results', 'RSA', model,
                            f"{specie}-sub-{sub_N:02d}",
                            f"ses-{session}_task-{task}_run-{run_N:02d}",
                            f"r-{radius}_{dis_method}_{cat1}_{cat2}.nii.gz"
                        )
                        # check if file exists
                        if not os.path.exists(output_file):
                            if verbose:
                                print(f"Not found {output_file}.")
                            all_exist = False
                        else:
                            if verbose:
                                print(f"Found {output_file} exists.")
    return all_exist




def calculate_mean_model_cross_participant_similarity_map(datafolder, dataset, session_and_run_all_dict, specie, model, task, radius, rsa_model, rsa_class, rsa_method, dis_method, replace_file=True, verbose=True, min_percentage_available=0, mask_type=None):
    '''
    Calculates average of cross-participant similarity maps for a given rsa_class. Saves the mean and std maps as nifti files.
    #### Note. Assumes run order is relevant #####

    Parameters:
    - datafolder: str, path to the data folder
    - dataset: str, name of the dataset
    - session_and_run_all_dict: dict, dictionary containing session and run information for all participants
    - sub_N: int, participant number
    - specie: str, species name
    - model: str, model name
    - task: str, task name
    - radius: int, radius for the searchlight
    - rsa_model: str, RSA model name
    - rsa_class: str, RSA class name
    - rsa_method: str, RSA method name
    - dis_method: str, dissimilarity method name
    - replace_file: bool, whether to replace existing files
    - verbose: bool, whether to print verbose output
    - min_percentage_available: float, minimum percentage of available data required
    - mask_type: str, type of mask to use

    '''
    # get participants
    # participants = range(1,41)
    participants = list(session_and_run_all_dict.keys())

    print(f"Calculating group-level cross-participant similarity map for {specie} participants: {participants}...")
    log =[]
    log_json = {
        'datafolder': datafolder,
        'dataset': dataset,
        'specie': specie,
        'model': model,
        'task': task,
        'radius': radius,
        'rsa_model': rsa_model,
        'rsa_class': rsa_class,
        'rsa_method': rsa_method,
        'mask_type': mask_type,
        'replace_file': replace_file,
        'verbose': verbose,
        'min_percentage_available': min_percentage_available
    }
    print("Checking for existing output files...")
    if mask_type is None:
        # "P:\userdata\raulh87\data\EmoC\results\RSA_same\basic-block\class_dog\r-3_mahalanobis_kendall_task-EmoC.nii.gz"
        # check if output file already exists
        output = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_same' + os.sep +
                        model + os.sep + rsa_class + os.sep +
                        f"{specie}-r-{radius}_{dis_method}_{rsa_method}_mean.nii.gz")
    else:
        output = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_same' + os.sep +
                        model + os.sep + rsa_class + os.sep + f"{mask_type}-{specie}-r-{radius}_{dis_method}_{rsa_method}_mean.nii.gz")
    
    log_json_output = output.replace('.nii.gz', '.json')
    # check if log_json output file already exists
    if os.path.exists(log_json_output):
        # load existing log_json
        with open(log_json_output, 'r') as f:
            existing_log_json = yaml.safe_load(f)
        # if perc_available in existing log_json is < min_percentage_available, force recalculation
        if existing_log_json['perc_available'] < min_percentage_available:
            print(f"Existing log file {log_json_output} found with perc_available < min_percentage_available. Forcing recalculation.")
            replace_file = True
    if os.path.exists(output) and not replace_file:
        # add log message
        log.append(f"Output file {output} already exists. Skipping group model similarity map calculation.")
        if verbose:
            print(log[-1])
        return

    # gather all cross-participant similarity maps across participants/sessions/runs
    # if they exist, add to files_list
    
    verbose = True
    print('starting loop')
    


    # keep track of the expected files in the database
    files_in_database = 0
    
    
    
    ## make a list of all available files, for each file make an inventory of to which sub_N and run_N it belongs.
    files_list = []
    # determine which possible runs, sessions, and participants are available in session_and_run_all_dict
    for sub_N_1 in session_and_run_all_dict.keys():
        # exclude 

        for entry in session_and_run_all_dict[sub_N_1]:
            run_N_1 = entry['run_N']
            session_1 = entry['session']
            # iterate over sub_N_2
            for sub_N_2 in session_and_run_all_dict.keys():
                # check if sub_N_1 < sub_N_2
                if sub_N_1 >= sub_N_2:
                    continue
                for entry_2 in session_and_run_all_dict[sub_N_2]:
                    run_N_2 = entry_2['run_N']
                    session_2 = entry_2['session']
                    # check if run_N_1 == run_N_2
                    if run_N_1 != run_N_2:
                        continue
                    # build filename for cross-participant similarity map
                    filename1 = os.path.join(
                        datafolder, dataset, 'results', 'RSA_same', model,
                        f"{specie}-sub-{str(sub_N_1).zfill(2)}_ses-{str(session_1).zfill(2)}_task-{task}_run-{str(run_N_1).zfill(2)}", rsa_class,
                        f"{specie}-sub-{str(sub_N_2).zfill(2)}",
                        f"r-{radius}_{dis_method}_{rsa_method}_ses-{str(session_2).zfill(2)}_task-{task}_run-{str(run_N_2).zfill(2)}.nii.gz"
                    )
                    # add a counter to files_in_database
                    files_in_database += 1
                    # check if the file exists, if so add to files_list
                    if os.path.exists(filename1):
                        files_list.append(filename1)
                        if verbose:
                            print(f"added : {filename1}")
                    else:
                        if verbose:
                            print(f"missing: {filename1}")
    # check if enough files are available
    available_percentage = len(files_list) / files_in_database if files_in_database > 0 else 0
    
    print(f"Available files: {len(files_list)} / {files_in_database} ({available_percentage*100:.2f}%)")
    # if there are not enough files available, skip the calculation, min_percentage_available
    if available_percentage < min_percentage_available:
        print(f"Available files ({len(files_list)}) is less than min_percentage_available ({min_percentage_available*100}%). Skipping group model similarity map calculation.")
        return


    # determine output path for mean and std files
    if mask_type is None:
        mean_model_map_path = os.path.join(datafolder, dataset, 'results', 'RSA_same', model, rsa_class, f"{specie}-r-{radius}_{dis_method}_{rsa_method}_mean.nii.gz")
        std_model_map_path = os.path.join(datafolder, dataset, 'results', 'RSA_same', model, rsa_class, f"{specie}-r-{radius}_{dis_method}_{rsa_method}_std.nii.gz")
    else:
        mean_model_map_path = os.path.join(datafolder, dataset, 'results', 'RSA_same', model, rsa_class, f"{mask_type}-{specie}-r-{radius}_{dis_method}_{rsa_method}_mean.nii.gz")
        std_model_map_path = os.path.join(datafolder, dataset, 'results', 'RSA_same', model, rsa_class, f"{mask_type}-{specie}-r-{radius}_{dis_method}_{rsa_method}_std.nii.gz")
    if verbose:
        print(f"Calculating mean and std across {len(files_list)} files for sub_N {sub_N_1}...")
        print(f"Mean output will be saved to {mean_model_map_path}")
        print(f"Std output will be saved to {std_model_map_path}")
            
    nifti_mean(files_list, mean_model_map_path, std_model_map_path, verbose=verbose)
    
    # save log_json
    log_json['perc_available'] = available_percentage
    log_json['output_mean_file'] = mean_model_map_path
    log_json['output_std_file'] = std_model_map_path
    log_json['notes'] = log
    # print saving log
    print(f"Saving log to {log_json_output}...")
    with open(log_json_output, 'w') as f:
        yaml.dump(log_json, f)
    return True 
      


def calculate_movement_parameters(datafolder, dataset, task, specie, sub_N, entry, verbose=False):
    session = entry['session']
    run_N = entry['run_N']
    # path to raw nifti file
    raw_nifti = os.path.join(
            datafolder, dataset, 'BIDS',
            f"{specie}-sub-{sub_N:02d}",
            f"{specie}-sub-{sub_N:02d}_ses-{session:02d}_task-{task}_run-{run_N:02d}_bold.nii.gz"
        )
    mc_path = os.path.join(
            datafolder, dataset, "preprocessing", f"{specie}_mcflirt",
            f"{specie}-sub-{sub_N:02d}_ses-{session:02d}_task-{task}_run-{run_N:02d}"
        )
    

    # run mcflirt with plots
    if verbose:
        print(f"Running mcflirt on {raw_nifti}, output will be saved to {mc_path}...")
        # build command, include verbose flag for mcflirt
        command_mcflirt = f"mcflirt -in {raw_nifti} -out {mc_path} -plots -verbose"
    else:
        command_mcflirt = f"mcflirt -in {raw_nifti} -out {mc_path} -plots"
    
    # run command
    os.system(command_mcflirt)
    print(f"Ran mcflirt on {raw_nifti}, output saved to {mc_path}")
    # get rid of the .nii.gz file that mcflirt creates, we only need the .par file "P:\userdata\raulh87\data\EmoC\preprocessing\D_mcflirt\D-sub-01_ses-01_task-EmoC_run-05.nii.gz"
    mc_nifti = os.path.join(mc_path, f"{specie}-sub-{sub_N:02d}_ses-{session:02d}_task-{task}_run-{run_N:02d}.nii.gz")
    if os.path.exists(mc_nifti):
        os.remove(mc_nifti)
        

def get_session_and_run_dict(datafolder, dataset, specie, sub_N):
    '''
    This function should replace utils_EmoB.get_session_and_run_list
    Issue: session is to complicated to have in the config file, maybe table?
    '''
    # database_table_path = r"P:\userdata\raulh87\data\EmoC\BIDS\D_database-details.csv"
    database_table_path = os.path.join(datafolder, dataset, 'BIDS', f"{specie}_database-details.csv")
    # load database details
    database_df = pd.read_csv(database_table_path)
    # contains sub_N, session, run_N, bold, slice_timing, events, eyetracker
    # filter for sub_N
    subject_df = database_df[database_df['sub_N'] == sub_N]
    # reset index
    subject_df = subject_df.reset_index(drop=True)

    # create a dict and iterate over each available row
    session_and_run_dict = {specie: []}
    for index, row in subject_df.iterrows():
        session_and_run_dict[specie].append({'session': row['session'], 'run_N': row['run_N']})
    return session_and_run_dict[specie]




def calculate_similarity_across_all_pairs(datafolder, dataset, session_and_run_all_dict, participants, specie,
                                        mask, model, task, radius, dis_method, 
                                        rsa_model, voxel_coords, config_path, verbose=False, shuffle_participants=True,
                                        shuffle_runs=False, wait_time=300):
    '''
    Calculates similarity across all pairs in a model, saves files as txt.
    '''
    if shuffle_participants:
        random.shuffle(participants)
    X, Y, Z = voxel_coords
    for sub_N in participants:
        session_and_run_dict = session_and_run_all_dict[sub_N]
        # if shuffle_runs is True, shuffle the order of runs for this participant
        if shuffle_runs:
            random.shuffle(session_and_run_dict)
        for entry in session_and_run_dict:
            session = entry['session']
            session = f"{session:02d}"
            run_N = entry['run_N']
            print(f"Calculating similarity across all pairs for {specie}-sub-{sub_N:02d}, session {session}, run {run_N}...")
            
            # create output path
            output_path = os.path.join(datafolder, dataset, 'results', 'RSA_sphere', model,
                                        f"{specie}-sub-{sub_N:02d}",
                                        f"ses-{session}_task-{task}_run-{run_N:02d}",
                                        f"r-{radius}_{dis_method}_{rsa_model}_voxel-{X}_{Y}_{Z}.txt")
            # check if output file already exists
            if os.path.exists(output_path):
                if verbose:
                    print(f"Skipping: Found existing output file {output_path}.")
                continue
            # create temporary file to flag that this comparison is being processed, to avoid duplicate processing in case of parallel execution
            temp_file = output_path + ".temp"
            if os.path.exists(temp_file):
                # check if temp file is older than wait_time seconds
                temp_file_age = time() - os.path.getmtime(temp_file)
                if temp_file_age < wait_time:
                    if verbose:
                        print(f"Skipping: Found existing temp file {temp_file} that is {temp_file_age:.2f} seconds old, indicating that this comparison is being processed by another instance.")
                    continue
                else:
                    if verbose:
                        print(f"Temp file {temp_file} is older than {wait_time} seconds (age: {temp_file_age:.2f} seconds). Assuming previous process crashed and removing temp file.")
                    os.remove(temp_file)  # remove old temp file
            # make sure output folder exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # create temp file to flag that this comparison is being processed
            open(temp_file, 'a').close()  # create temp file
            try:
                if verbose:
                    print(f"Created temp file {temp_file} to indicate that this comparison is being processed.")
                # get similarity values            
                vals = get_similarity_in_sphere(datafolder, dataset, specie, mask, sub_N, session, run_N, 
                                        model, radius, dis_method,
                                        rsa_model, config_path, verbose=False,
                                        voxel_coords=(X, Y, Z))

                
                # save similarity values to txt file
                np.savetxt(output_path, vals)
            # if any error occurs, print error message and continue with next iteration, removing temp file if it exists
            except Exception as e:
                print(f"Error processing {specie}-sub-{sub_N:02d}, session {session}, run {run_N}: {e}")
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)  # remove temp file after processing is done
                    if verbose:
                        print(f"Removed temp file {temp_file} after processing.")

def get_similarity_in_sphere(datafolder, dataset, specie, mask, sub_N, session, run_N, 
                             model, radius, rsa_method,
                             rsa_model, config_path, verbose=False,
                             voxel_coords=None):
    '''
    Loads pairwise similarity maps for a participant, session, run, and model,
    extracts the similarity values within a sphere of given radius around a 
    given voxel coordinate, saves a txt file with the similarity values for each pair of categories
    in the rsa_model, and returns a vector of similarity values for that voxel.
    '''
    if voxel_coords is None:
        raise ValueError("voxel_coords must be provided as a tuple of (X, Y, Z) voxel coordinates.")
    ref_img = nib.load(mask).get_fdata()
    # ref_affine = nib.load(mask).affine
    rsa_model_path = datafolder + os.sep + dataset + os.sep + 'rsa_models' + os.sep + rsa_model + ".csv"
    config_path = datafolder + os.sep + dataset + os.sep + 'config_files' + os.sep + model + '.yaml'
    # load meta similarity map for participant, session, run, and model
    meta_similarity_map = load_meta_similarity_map(rsa_model_path, ref_img, datafolder, dataset, 
                                                           specie, sub_N, session, run_N, config_path, 
                                                           rsa_method=rsa_method, radius=radius, verbose=verbose)
    X, Y, Z = voxel_coords
    # get similarity [X,Y,Z,:] for voxel with voxel coordinates (X,Y,Z)
    vals = meta_similarity_map[X, Y, Z, :]

    return vals

def kendall_custom(x, y):
        try:
            from scipy.stats import kendalltau
        except Exception as e:
            raise ImportError("SciPy is required for method='kendall'.") from e
        return kendalltau(x, y, nan_policy='omit').correlation

def calculate_cross_participant_similarity_map(meta_similarity_map1, meta_similarity_map2, rsa_method='kendall', verbose=False):
    '''
    Calculate cross-participant similarity by correlating the meta similarity maps (meta_similarity_map1, 
    meta_similarity_map2) with shape (X, Y, Z, n_pairs).
    The resulting cross-participant similarity map (cross_participant_similarity_map) has shape (X, Y, Z) 
    each voxel value in the cross-participant similarity map represents the similarity between all the pairs in that voxel.
    Meaning that cross_participant_similarity_map(Xi,Yi,Zi) in the similarity between meta_similarity_map1(Xi,Yi,Zi,:) 
    and meta_similarity_map2(Xi,Yi,Zi,:).

    inputs:
    meta_similarity_map1: 4D numpy array of shape (X, Y, Z, n_pairs) where n_pairs is the number of unique pairs of categories in the RSA model, containing the pairwise similarity values for each voxel (coordinate [x, y, z]) and each pair of categories for the model participant
    meta_similarity_map2: 4D numpy array of shape (X, Y, Z, n_pairs) where n_pairs is the number of unique pairs of categories in the RSA model, containing the pairwise similarity values for each voxel (coordinate [x, y, z]) and each pair of categories for the target participant
    rsa_method: str, method for RSA calculation (e.g., 'kendall', 'spearman', 'pearson', 'euclidean'), default is 'kendall'
    
    returns:
    cross_participant_similarity_map: 3D numpy array of shape (X, Y, Z) where each voxel value represents the similarity between the meta similarity maps of the model and target participants for that voxel, calculated using the specified rsa_method.

    '''
    # get shape of meta similarity maps
    X, Y, Z, n_pairs = meta_similarity_map1.shape
    # initialize cross-participant similarity map
    cross_participant_similarity_map = np.zeros((X, Y, Z))
    # iterate over all voxels
    for x in range(X):
        if verbose and x % 10 == 0:
            print(f"Processing slice {x+1}/{X}...")
        for y in range(Y):
            if verbose and y % 10 == 0:
                print(f"Processing row {y+1}/{Y}...")
            for z in range(Z):
                if verbose and z % 10 == 0:
                    print(f"Processing column {z+1}/{Z}...")
                # get pairwise similarity values for this voxel for both participants
                vec1 = meta_similarity_map1[x, y, z, :]
                vec2 = meta_similarity_map2[x, y, z, :]
                # calculate similarity between the two vectors using the specified rsa_method
                if rsa_method == 'kendall':
                    val = kendall_custom(vec1, vec2)
                elif rsa_method == 'spearman':
                    # issue not implemented yet
                    raise NotImplementedError("Spearman method is not implemented yet.")
                elif rsa_method == 'pearson':
                    # issue not implemented yet
                    raise NotImplementedError("Pearson method is not implemented yet.")
                else:
                    raise ValueError(f"Invalid rsa_method: {rsa_method}. Must be 'kendall', 'spearman', 'pearson', or 'euclidean'.")
                # assign value to cross-participant similarity map
                cross_participant_similarity_map[x, y, z] = val
    return cross_participant_similarity_map

def calculate_cross_participant_similarity(datafolder, dataset, session_and_run_all_dict, participants, specie, mask, model, task, radius, rsa_class, dis_method='mahalanobis', rsa_method='kendall',  mask_type='b_GreyMatter2mmB', verbose=False, shuffle_participants=True):
    '''
    
    Calculate cross-participant similarity, for all possible participant pairs.
    
    Loads meta similarity maps (4D numpy arrays where [x, y, z, n_pairs],
    contains the pairwise similarity values for each voxel and each pair of categories)
    Calculates cross-participant similarity by correlating the meta similarity maps
    for each coordinate (x, y, z), resulting in a similarity map of shape (X, Y, Z) for 
    each participant-participant pair, to save space only one cross participant map is 
    saved (*sub-01_sub_02.nii.gz, but not *sub-02_sub_01.nii.gz)

    datafolder: str, path to data folder
    dataset: str, name of dataset
    participants: list of int, participant numbers
    specie: str, 'D' for dog, 'H' for human
    mask: str, path to mask file
    model: str, GLM model used
    task: str, task name
    radius: int, radius for searchlight
    dis_method: str, method for pairwise similarity calculation (e.g., 'spearman', 'pearson', 'mahalanobis')
    rsa_method: str, method for RSA calculation (e.g., 'kendall', 'spearman', 'pearson')
    shuffle_participants: bool, whether to shuffle participants order (default True)
    '''
    if shuffle_participants:
        random.shuffle(participants)
    # load reference image for meta similarity map based on mask
    ref_img = nib.load(mask).get_fdata()
    ref_affine = nib.load(mask).affine
    config_path = datafolder + os.sep + dataset + os.sep + 'config_files' + os.sep + specie + '_' + model + '.yaml'


    # go through each participant as model
    for sub_N1 in participants:
        print(f"Using {specie}-sub-{sub_N1:02d} as model...")
        # get sessions and runs for this participant
        session_and_run_dict_model = session_and_run_all_dict[sub_N1]
        # iterate over sessions and runs
        for entry in session_and_run_dict_model:
            session1 = entry['session']
            session1 = f"{session1:02d}"
            run_N1 = entry['run_N']
            print(f"Loading meta similarity map for {specie}-sub-{sub_N1:02d}, session {session1}, run {run_N1}...")  
            # load meta similarity map for participant1
            # meta_similarity_map1: 4D numpy array of shape (X, Y, Z, n_pairs) where n_pairs is the number of unique pairs of categories in the RSA model, containing the pairwise similarity values for each voxel (coordinate [x, y, z]) and each pair of categories for the model participant
            meta_similarity_map1 = load_meta_similarity_map2(ref_img, datafolder, dataset, specie, sub_N1, session1, run_N1, config_path, rsa_class=rsa_class, dis_method=dis_method, radius=radius, verbose=verbose)
            # if meta_similarity_map1 == 0, means that the meta similarity map is missing, skip to next iteration
            if isinstance(meta_similarity_map1, int) and meta_similarity_map1 == 0:
                warnings.warn(f"Meta similarity map for {specie}-sub-{sub_N1:02d}, session {session1}, run {run_N1} is missing. Skipping to next iteration.")
                continue    

            # go through each participant as target
            for sub_N2 in participants:
                # get session and run list for participant2
                session_and_run_dict_target = session_and_run_all_dict[sub_N2]
                for entry_target in session_and_run_dict_target:
                    session2 = entry_target['session']
                    session2 = f"{session2:02d}"
                    run_N2 = entry_target['run_N']
                    # if run_N1 is the same as run_N2, then calculate cross-participant similarity, otherwise skip (invalid comparison)
                    if run_N1 == run_N2:
                        if sub_N2 <= sub_N1:
                            continue  # to save space, only calculate and save one of sub-01_sub-02 and sub-02_sub-01, not both
                        output_path = os.path.join(
                            datafolder, dataset, 'results', 'RSA_same', model,
                            f"{specie}-sub-{sub_N1:02d}_ses-{session1}_task-{task}_run-{run_N1:02d}",
                            f"{rsa_class}", 
                            f"{specie}-sub-{sub_N2:02d}",
                            f"r-{radius}_{dis_method}_{rsa_method}_ses-{session2}_task-{task}_run-{run_N2:02d}.nii.gz"
                        )
                        # make sure output folder exists
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                        # check if output file already exists
                        if os.path.exists(output_path):
                            if verbose:
                                print(f"Skipping: Found existing output file {output_path}.")
                            continue
                        else:
                            # create temporary file to flag that this comparison is being processed, to avoid duplicate processing in case of parallel execution
                            temp_file = output_path + ".temp"
                            if os.path.exists(temp_file):
                                if verbose:
                                    print(f"Skipping: Found existing temp file {temp_file}, indicating that this comparison is being processed by another instance.")
                                continue
                            else:
                                open(temp_file, 'a').close()  # create temp file
                                if verbose:
                                    print(f"Created temp file {temp_file} to indicate that this comparison is being processed.")                        
                        print(f"crossing with {specie}-sub-{sub_N2:02d}...")
                        # load meta similarity map for target participant
                        try:
                            meta_similarity_map2 = load_meta_similarity_map2(ref_img, datafolder, dataset, specie, sub_N2, session2, run_N2, config_path, rsa_class=rsa_class, dis_method=dis_method, radius=radius, verbose=verbose)
                            # if meta_similarity_map2 == 0, means that the meta similarity map is missing, skip to next iteration
                            if isinstance(meta_similarity_map2, int) and meta_similarity_map2 == 0:
                                warnings.warn(f"Meta similarity map for {specie}-sub-{sub_N2:02d}, session {session2}, run {run_N2} is missing. Skipping to next iteration.")
                                continue    

                            # calculate cross-participant similarity by correlating the meta similarity maps for each coordinate (x, y, z), resulting in a similarity map of shape (X, Y, Z) for each participant-participant pair
                            cross_participant_similarity_map = calculate_cross_participant_similarity_map(meta_similarity_map1, meta_similarity_map2, rsa_method=rsa_method, verbose=verbose)
                            # save cross-participant similarity map as nifti file
                            nib.save(nib.Nifti1Image(cross_participant_similarity_map, affine=ref_affine), output_path)
                            print(f"Saved cross-participant similarity map to {output_path}")
                        except Exception as e:
                            print(f"Error processing comparison between {specie}-sub-{sub_N1:02d} and {specie}-sub-{sub_N2:02d} for session {session1} and session {session2}: {e}")
                        finally:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)  # remove temp file after processing is done
                                if verbose:
                                    print(f"Removed temp file {temp_file} after processing.")

        # output: "P:\userdata\raulh87\data\EmoB\results\RSA\basic-block\H-sub-04\anger-strictB\H-sub-02\ses-01_task-EmoB_run-04\b_greyMatter2mmB-r-4_mahalanobis_kendall.nii.gz"



def transform_coords(
    coords_input,
    results_img,
    original_img,
    *,
    rounding="nearest",
    clip=False,
    assume_one_based=False,
):
    """
    Transforms voxel coordinates from the *results space* to the *original atlas space*.

    Parameters
    ----------
    coords_input : array-like
        Iterable of (i, j, k) voxel indices in the results space.
        Can be a single tuple (i, j, k) or an (N, 3) array-like.

    results_img : nibabel.Nifti1Image or str
        The reference image that defines the voxel grid of your results coordinates
        (e.g., your searchlight/stat map like pe1.nii.gz).
        Can be a loaded nibabel image or a filepath.

    original_img : nibabel.Nifti1Image or str
        The original atlas image defining the desired voxel grid (e.g., original_atlas_space.nii.gz).
        Can be a loaded nibabel image or a filepath.

    rounding : {"nearest","floor","ceil","none"}, optional
        How to convert continuous voxel coords into integer voxel indices.
        "nearest" (default) is typically what you want.

    clip : bool, optional
        If True, clip output voxel indices to be within the bounds of original_img shape.

    assume_one_based : bool, optional
        If your input coords are 1-based (some tools do this), set True.
        Then we convert to 0-based internally and convert back to 1-based at the end.

    Returns
    -------
    transformed_coords : list[tuple[int,int,int]]
        List of (i, j, k) voxel indices in the original atlas space.
    """
    # Lazy import so this function can be used in environments where nibabel
    # is only needed when loading paths.
    import nibabel as nib

    def _as_img(x):
        if isinstance(x, str):
            return nib.load(x)
        return x

    res_img = _as_img(results_img)
    orig_img = _as_img(original_img)

    A_res = np.asarray(res_img.affine, dtype=float)
    A_orig = np.asarray(orig_img.affine, dtype=float)

    # Matrix that maps results-voxel -> original-voxel (in homogeneous coords)
    # orig_vox = inv(A_orig) @ A_res @ res_vox
    M = np.linalg.inv(A_orig) @ A_res

    coords = np.asarray(coords_input, dtype=float)
    if coords.shape == (3,):
        coords = coords[None, :]
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords_input must be shape (3,) or (N, 3). Got {coords.shape}")

    if assume_one_based:
        coords = coords - 1.0  # to 0-based

    hom = np.c_[coords, np.ones((coords.shape[0], 1), dtype=float)]
    out = (M @ hom.T).T[:, :3]  # continuous voxel coords in original space

    if rounding == "nearest":
        out = np.rint(out)
    elif rounding == "floor":
        out = np.floor(out)
    elif rounding == "ceil":
        out = np.ceil(out)
    elif rounding == "none":
        # keep as float (rarely useful for ITK-SNAP navigation, but sometimes for debugging)
        pass
    else:
        raise ValueError("rounding must be one of: 'nearest', 'floor', 'ceil', 'none'")

    if rounding != "none":
        out = out.astype(int)

    if clip:
        # Clip using the spatial dims only (first 3)
        shp = np.array(orig_img.shape[:3], dtype=int)
        if rounding == "none":
            # clip floats
            out = np.minimum(np.maximum(out, 0.0), shp - 1.0)
        else:
            out = np.minimum(np.maximum(out, 0), shp - 1)

    if assume_one_based and rounding != "none":
        out = out + 1  # back to 1-based

    # Return list of tuples
    return [tuple(map(int, xyz)) if rounding != "none" else tuple(xyz) for xyz in out]

def get_label_from_stim(cat, items_dict, label_type):
    """Get label from stim using items_dict and label_type."""
    for item in items_dict.values():
        if item['stim_ID'] == cat:
            if label_type in item:
                return item[label_type]
            else:
                raise ValueError(f"Label type {label_type} not found in items_dict.")
    raise ValueError(f"Stim {cat} not found in items_dict.")


## update the file structure here: ------------------------------------------
def calculate_difference_map(datafolder, dataset, specie, model, comparison_model,
                             radius, dis_method, replace_file=False, verbose=False):
    '''Calculate difference map between two similarity maps.
    datafolder: str, path to data folder
    dataset: str, name of dataset
    specie: str, 'D' for dog, 'H' for human
    model: str, GLM model used
    comparison_model: str, name of the comparison model (the model indicates the main folder)
    radius: int, radius for searchlight
    dis_method: str, method for pairwise similarity calculation
    replace_file: bool, whether to replace existing files
    verbose: bool, whether to print verbose output
    '''
    # load comparisons file "P:\userdata\raulh87\data\EmoB\rsa_models\happy\comparisons.csv"
    comparisons_file = (datafolder + os.sep + dataset + os.sep + 'rsa_models' + os.sep +
                        comparison_model + os.sep + 'comparisons.csv')
    # read comparisons file
    comparisons_df = pd.read_csv(comparisons_file)
    for comp_name in comparisons_df['comparisons']:
        # split by _
        parts = comp_name.split('_')
        stim_a = parts[0]
        stim_b = parts[1]
        if verbose:
            print(f"Calculating difference map for {stim_a} - {stim_b}...")
        # determine output mean
        result_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + comparison_model + os.sep + 
                        f"{specie}-r-{radius}_{dis_method}_{stim_a}_minus_{stim_b}.nii.gz")
        # check if output file already exists
        if os.path.exists(result_map_path) and not replace_file:
            if verbose:
                print(f"Skipping: Found existing output file {result_map_path}. Use replace_file=True to overwrite.")
            continue
        # load the two similarity maps
        # "P:\userdata\raulh87\data\EmoB\results\RSA\basic-block\happy\D-r-3_mahalanobis_cat_angry_mean.nii.gz"
        sim_map_a_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + comparison_model + os.sep + 
                        f"{specie}-r-{radius}_{dis_method}_cat_{stim_a}_mean.nii.gz")
        sim_map_b_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + comparison_model + os.sep + 
                        f"{specie}-r-{radius}_{dis_method}_cat_{stim_b}_mean.nii.gz")

        sim_map_a = nib.load(sim_map_a_path).get_fdata()
        sim_map_b = nib.load(sim_map_b_path).get_fdata()
        # calculate difference map
        diff_map = sim_map_a - sim_map_b
        # save difference map
        nib.save(nib.Nifti1Image(diff_map, affine=nib.load(sim_map_a_path).affine), result_map_path)
        if verbose:
            print(f"Saved difference map to {result_map_path}")


# def calculate_similarity_maps_by_group_rnd(datafolder, dataset, session_and_run_all_dict,
#                                        specie, model, comparison_model, model_dict, task, radius,
#                                         stim_types, method, mask_img,
#                                         replace_file=False, min_percentage_available=1.0,
#                                         verbose=False, avoid_pairs_by_label=None):
#     '''Calculate permutations for group average similarity maps for all pairwise combinations in comparisons.
#     datafolder: str, path to data folder
#     dataset: str, name of dataset
#     sub_N: int, subject number
#     session_and_run_dict: list of dicts, each dict contains 'session' and 'run' keys
#     specie: str, 'D' for dog, 'H' for human
#     model: str, GLM model used
#     comparisons: list of tuples, each tuple contains two stimulus types to compare
#     task: str, task name
#     radius: int, radius for searchlight
#     method: str, method for pairwise similarity calculation
#     replace_file: bool, whether to replace existing files
#     verbose: bool, whether to print verbose output
#     comparison_model: str, name of the comparison model (the model indicates the main folder)
#     avoid_pairs_by_label: will check if pairs belong to same label and avoid them if True. Default is None.
#     '''
#     # load comparisons file "P:\userdata\raulh87\data\EmoB\rsa_models\happy\comparisons.csv"
#     comparisons_file = (datafolder + os.sep + dataset + os.sep + 'rsa_models' + os.sep +
#                         comparison_model + os.sep + 'comparisons.csv')
#     # read comparisons file

#     cats_list = []
#     comparisons_df = pd.read_csv(comparisons_file)
#     for comp_name in comparisons_df['comparisons']:
#         # split by _
#         parts = comp_name.split('_')
#         stim_a = parts[0]
#         stim_b = parts[1]
#         cats_list.append(stim_a)
#         cats_list.append(stim_b)
#     # get unique categories
#     unique_cats = list(set(cats_list))
#     # load each category file
#     for cat in unique_cats:
#         # build cat comparisons_file
#         # "P:\userdata\raulh87\data\EmoB\rsa_models\happy\cat_happy.csv"
#         category_file = (datafolder + os.sep + dataset + os.sep + 'rsa_models' + os.sep +
#                         comparison_model + os.sep + f"cat_{cat}.csv")
#         # if verbose
#         if verbose:
#             print(f"Calculating group similarity map for category {cat}...")
#         # determine output mean
#         result_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
#                         model + os.sep + comparison_model + os.sep + 
#                         f"{specie}-r-{radius}_{method}_cat_{cat}_mean.nii.gz")
#         result_map_path_std = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
#                         model + os.sep + comparison_model + os.sep + 
#                         f"{specie}-r-{radius}_{method}_cat_{cat}_std.nii.gz")
#         # check if output file already exists
#         if os.path.exists(result_map_path) and not replace_file:
#             if verbose:
#                 print(f"Skipping: Found existing output file {result_map_path}. Use replace_file=True to overwrite.")
#             continue
        
#         # read category file
#         category_df = pd.read_csv(category_file)
#         files_in_database = 0
#         files_list = [] # list of files to process
#         for sub_N in session_and_run_all_dict.keys():
#             session_and_run_dict = session_and_run_all_dict[sub_N]
#             for entry in session_and_run_dict:
#                 # add a counter
#                 session = entry['session']
#                 session = f"{session:02d}"
#                 run_N = entry['run_N']
#                 items_dict = model_dict[f"run{run_N:02d}"]
#                 # loop over all pairs in category_df
#                 for index, row in category_df.iterrows():
#                     # identify from items_dict the stim_ID matching cat1 and cat2
#                     cat1 = row['cat1']
#                     cat2 = row['cat2']
#                     if avoid_pairs_by_label is not None:
#                         label1 = get_label_from_stim(cat1, items_dict, avoid_pairs_by_label)
#                         label2 = get_label_from_stim(cat2, items_dict, avoid_pairs_by_label)
#                         print(f"Comparing {cat1} ({avoid_pairs_by_label}: {label1}) and {cat2} ({avoid_pairs_by_label}: {label2})")

#                         if label1 == label2:
#                             if verbose:
#                                 print(f"Skipping pair {cat1}-{cat2} for subject {sub_N}, session {session}, run {run_N} due to same {avoid_pairs_by_label} {label1}.")
#                             continue
                    

#                     files_in_database += 1
#                     input_file = os.path.join(
#                                 datafolder, dataset, 'results', 'RSA', model,
#                                 f"{specie}-sub-{sub_N:02d}",
#                                 f"ses-{session}_task-{task}_run-{run_N:02d}",
#                                 f"r-{radius}_{method}_{cat1}_{cat2}.nii.gz"
#                             )
#                     if os.path.exists(input_file):
#                         files_list.append(input_file)
#         # make sure output folder exists
#         output_folder = os.path.dirname(result_map_path)
#         os.makedirs(output_folder, exist_ok=True)
#         # make sure files_in_database is not zero
#         if files_in_database == 0:
#             raise ValueError("No files found in database. Check session_and_run_all_dict.")
        
#         # check if enough files are available
#         percentage_available = len(files_list) / files_in_database
#         if percentage_available < min_percentage_available:
#             if verbose:
#                 print(f"Skipping: Only {percentage_available*100:.2f}% of files available for category {cat}, which is below the threshold of {min_percentage_available*100:.2f}%.")
#             continue
#         else:
#             if verbose:
#                 print(f"Found {len(files_list)} files ({percentage_available*100:.2f}%) to process for group average for category {cat}.")

#         nifti_mean(files_list, result_map_path=result_map_path, result_map_path_std=result_map_path_std, 
#                 mask_img=mask_img, verbose=verbose)
                    
def calculate_similarity_maps_by_class(datafolder, dataset, session_and_run_all_dict,
                                       specie, model, comparison_model, model_dict, task, radius,
                                        dis_method, mask,
                                        replace_file=False, min_percentage_available=1.0,
                                        verbose=False, avoid_pairs_by_label=None):
    '''Calculate class average similarity maps for all pairwise combinations in comparison_model.
    datafolder: str, path to data folder
    dataset: str, name of dataset
    sub_N: int, subject number
    session_and_run_all_dict: list of dicts, each dict contains 'session' and 'run' keys
    specie: str, 'D' for dog, 'H' for human
    model: str, GLM model used
    task: str, task name
    radius: int, radius for searchlight
    dis_method: str, method for pairwise similarity calculation
    replace_file: bool, whether to replace existing files
    verbose: bool, whether to print verbose output
    comparison_model: str, name of the comparison model 
    avoid_pairs_by_label: will check if pairs belong to same label and avoid them if True. Default is None.
    '''
    # load comparisons file, this file should have the two classes to compare in the first two lines, e.g. line 1: happy, line 2: neutral.
    comparisons_file = (datafolder + os.sep + dataset + os.sep + 'rsa_models' + os.sep + 
                        'by_class' + os.sep + 'classes' + os.sep + 'comp_' + comparison_model + '.txt')
                        # comparison_model + os.sep + 'comparisons.csv')
    
    # get mask affine
    mask_affine = nib.load(mask).affine
    mask_img_obj = nib.load(mask)
    mask_img = mask_img_obj.get_fdata().astype(bool)

    # read comparisons file, obtain class_a in line 1 and class_b in line 2
    with open(comparisons_file, 'r') as f:
        lines = f.read().splitlines()
        if len(lines) < 2:
            raise ValueError(f"Comparison model file {comparisons_file} must have at least two lines, one for each class.")
        class_a = lines[0]
        class_b = lines[1]

    unique_classes = [class_a, class_b] # override unique_cats with the classes defined in the comparison model file, to make sure only those are processed

    # load each category file
    for class_name in unique_classes:
        # build class file
        # "P:\userdata\raulh87\data\EmoB\rsa_models\happy\class_happy.csv"
        class_file = (datafolder + os.sep + dataset + os.sep + 'rsa_models' + os.sep +
                        'by_class' + os.sep + 'classes' + os.sep + f"class_{class_name}.csv")
        # if verbose
        if verbose:
            print(f"Calculating group similarity map for category {class_name}...")
        # determine output mean
        result_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + 'by_class' + os.sep + 'classes' + os.sep +
                        f"{specie}-r-{radius}_{method}_class_{class_name}_mean.nii.gz")
        result_map_path_std = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + 'by_class' + os.sep + 'classes' + os.sep +
                        f"{specie}-r-{radius}_{method}_class_{class_name}_std.nii.gz")
        # check if output file already exists
        if os.path.exists(result_map_path) and not replace_file:
            if verbose:
                print(f"Skipping: Found existing output file {result_map_path}. Use replace_file=True to overwrite.")
            continue
        
        # read class file
        class_df = pd.read_csv(class_file)
        files_in_database = 0
        files_list = [] # list of files to process
        for sub_N in session_and_run_all_dict.keys():
            session_and_run_dict = session_and_run_all_dict[sub_N]
            for entry in session_and_run_dict:
                # add a counter
                session = entry['session']
                session = f"{session:02d}"
                run_N = entry['run_N']
                items_dict = model_dict[f"run{run_N:02d}"]
                # loop over all pairs in class_df
                for index, row in class_df.iterrows():
                    # identify from items_dict the stim_ID matching class_a and class_b
                    class_a = row['cat1']
                    class_b = row['cat2']
                    if avoid_pairs_by_label is not None:
                        label1 = get_label_from_stim(class_a, items_dict, avoid_pairs_by_label)
                        label2 = get_label_from_stim(class_b, items_dict, avoid_pairs_by_label)
                        print(f"Comparing {class_a} ({avoid_pairs_by_label}: {label1}) and {class_b} ({avoid_pairs_by_label}: {label2})")

                        if label1 == label2:
                            if verbose:
                                print(f"Skipping pair {class_a}-{class_b} for subject {sub_N}, session {session}, run {run_N} due to same {avoid_pairs_by_label} {label1}.")
                            continue
                    

                    files_in_database += 1
                    input_file = os.path.join(
                                datafolder, dataset, 'results', 'RSA', model,
                                f"{specie}-sub-{sub_N:02d}",
                                f"ses-{session}_task-{task}_run-{run_N:02d}",
                                f"r-{radius}_{method}_{class_a}_{class_b}.nii.gz"
                            )
                    if os.path.exists(input_file):
                        files_list.append(input_file)
        # make sure output folder exists
        output_folder = os.path.dirname(result_map_path)
        os.makedirs(output_folder, exist_ok=True)
        # make sure files_in_database is not zero
        if files_in_database == 0:
            raise ValueError("No files found in database. Check session_and_run_all_dict.")
        
        # check if enough files are available
        percentage_available = len(files_list) / files_in_database
        if percentage_available < min_percentage_available:
            if verbose:
                print(f"Skipping: Only {percentage_available*100:.2f}% of files available for category {class_name}, which is below the threshold of {min_percentage_available*100:.2f}%.")
            continue
        else:
            if verbose:
                print(f"Found {len(files_list)} files ({percentage_available*100:.2f}%) to process for group average for category {class_name}.")

        nifti_mean(files_list, result_map_path=result_map_path, result_map_path_std=result_map_path_std, 
                mask_img=mask_img, verbose=verbose)           
            

        


    
    


def calculate_similarity_maps_by_list(datafolder, dataset, session_and_run_all_dict,
                                specie, model, stim_list, category_name, mask_img, task, radius,
                                dis_method, replace_file=False, min_percentage_available=1.0,
                                verbose=False):
    '''Calculate group average similarity maps for all pairwise combinations in stim_list.
    datafolder: str, path to data folder
    dataset: str, name of dataset
    sub_N: int, subject number
    session_and_run_dict: list of dicts, each dict contains 'session' and 'run' keys
    specie: str, 'D' for dog, 'H' for human
    model: str, GLM model used
    stim_list: list of str, stimulus types
    mask: str, path to mask file
    task: str, task name
    radius: int, radius for searchlight
    dis_method: str, method for pairwise similarity calculation
    replace_file: bool, whether to replace existing files
    verbose: bool, whether to print verbose output
    
    '''
    print(f"Calculating group similarity maps for {category_name}...")
    
    # check if output file already exists
    output_mean = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                    model + os.sep + category_name + os.sep + 'mean' + os.sep +
                    f"{specie}-r-{radius}_{dis_method}_mean.nii.gz")
    output_std = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                    model + os.sep + category_name + os.sep + 'mean' + os.sep +
                    f"{specie}-r-{radius}_{dis_method}_std.nii.gz")
    if os.path.exists(output_mean) and not replace_file:
        print(f"Skipping: Found existing output file {output_mean}. Use replace_file=True to overwrite.")
        return
    
    # get list of participants
    participants = list(session_and_run_all_dict.keys())

    print("Gathering model similarity maps across participants/sessions/runs...")
    files_in_database = 0
    files_list = [] # list of files to process
    for sub_N in participants:
        session_and_run_dict = session_and_run_all_dict[sub_N]
        for entry in session_and_run_dict:
            # add a counter
            files_in_database += 1
            session = entry['session']
            session = f"{session:02d}"
            run_N = entry['run_N']
    
            # combine all pairwise combinations in stim_list
            for i, stim_i in enumerate(stim_list):
                for j, stim_j in enumerate(stim_list):
                    if i >= j:
                        continue  # avoid duplicates and self-comparison
                    input_file = os.path.join(
                        datafolder, dataset, 'results', 'RSA', model,
                        f"{specie}-sub-{sub_N:02d}",
                        f"ses-{session}_task-{task}_run-{run_N:02d}",
                        f"r-{radius}_{dis_method}_{stim_i}_{stim_j}.nii.gz"
                    )
                    if os.path.exists(input_file):
                        files_list.append(input_file)
    # make sure files_in_database is not zero
    if files_in_database == 0:
        raise ValueError("No files found in database. Check session_and_run_all_dict.")
    # check if enough files are available
    percentage_available = len(files_list) / files_in_database
    if percentage_available < min_percentage_available:
        print(f"Skipping: Only {percentage_available*100:.2f}% of files available, which is below the threshold of {min_percentage_available*100:.2f}%.")
        return
    else:
        print(f"Found {len(files_list)} files ({percentage_available*100:.2f}%) to process for group average.")
    # calculate mean and std maps
    nifti_mean(files_list, result_map_path=output_mean, result_map_path_std=output_std, 
               mask_img=mask_img, verbose=verbose)
    



def create_sphere_mask(sample_img, coords_vox, radius):
    '''
    Create a spherical mask in the given image at the specified voxel coordinates and radius.

    Input arguments:
    --sample_img: Nifti image to create the mask in
    --coords_vox: (x,y,z) coordinates of the center in voxels
    --radius: radius of the sphere in voxels

    Output:
    --sphere_mask: Nifti1Image, binary mask with the sphere (uint8, 0/1), same affine/shape as sample_img
    '''
    if radius is None or radius <= 0:
        raise ValueError("radius must be a positive number (in voxels).")

    coords_vox = np.asarray(coords_vox, dtype=float).reshape(3,)
    cx, cy, cz = coords_vox

    shape = sample_img.shape[:3]  # supports sample_img being 3D or 4D
    nx, ny, nz = shape

    # Bounding box (clip to image)
    x0 = max(int(np.floor(cx - radius)), 0)
    x1 = min(int(np.ceil (cx + radius)) + 1, nx)
    y0 = max(int(np.floor(cy - radius)), 0)
    y1 = min(int(np.ceil (cy + radius)) + 1, ny)
    z0 = max(int(np.floor(cz - radius)), 0)
    z1 = min(int(np.ceil (cz + radius)) + 1, nz)

    mask = np.zeros(shape, dtype=np.uint8)

    # Build sphere only within the bounding box (fast + memory friendly)
    xx, yy, zz = np.ogrid[x0:x1, y0:y1, z0:z1]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    sphere_local = dist2 <= (radius ** 2)

    mask[x0:x1, y0:y1, z0:z1] = sphere_local.astype(np.uint8)

    hdr = sample_img.header.copy()
    hdr.set_data_dtype(np.uint8)
    sphere_out = nib.Nifti1Image(mask, affine=sample_img.affine, header=hdr)
    return sphere_out

def calculate_similarity_in_roi(datafolder,
                               dataset,
                               sub_N,
                               session_and_run_dict,
                               stim_types,
                               task,
                               model='basic-block',
                               dis_method='correlation',
                               specie='D',
                               roi_type='sphere',
                               atlas_type='Nitzsche',
                               verbose=False,
                               sample_img=None,
                               **kwargs):
    '''
    Calculate pairwise similarity between stimulus types within a specified ROI.

    Input arguments:
    --datafolder: Path to data folder
    --dataset: Dataset to use 
    --model: GLM model to use (default: 'basic-block')
    --dis_method: Method for pairwise similarity calculation (default: 'correlation')
    --stim_types: List of stimulus types to use
    --specie: 'D' for Dog, 'H' for Human (default: 'D')
    --roi_type: Type of ROI to use (default: 'sphere'). Options: 'sphere', 'segment', 'roi', 'anatomical'
        --sphere: creates a sphere (default: 3). 
            Requires: 
            --coords_vox (x,y,z coordinates of center in voxels)
            --coords_mm (x,y,z coordinates of center in mm)
            --radius (default: 3).
        --segment: uses segment from segmentation map
            Requires: 
                --segmentation_segments (segmentation map to use)
                --number_of_clusters (number of segments to use)
        --roi: uses ROI from ROI file
            Requires: 
                --roi_file (ROI file to use)
        --anatomical: uses anatomical mask from atlas
            Requires:
                --atlas_type (type of atlas to use)
                --region_name (name of region to use)
    --atlas_type: Type of atlas to use in case of dogs (default: 'Nitzsche')
    --verbose: Verbose output (default: False)

    '''
    # Throw exception when this function is called, check use of dis_method or rsa_method
    raise NotImplementedError("This function is should be checked, dis_method or rsa_method seems messed up")


    # in case of the method being mahalanobis and sphere, get it directly from the maps
    if dis_method == 'mahalanobis' and roi_type == 'sphere':
        # load meta_similarity_map
        meta_similarity_map = load_meta_similarity_map(rsa_model_path, ref_img, datafolder, dataset, specie, sub_N, session, run_N, config_path, dis_method=dis_method, radius=radius, verbose=verbose)
        # get the values
        # .....
        # return results_df


    # double check that the atlas matches the specie
    if specie == 'D' and atlas_type not in ['Nitzsche', 'Johnson', 'Datta']:
        raise ValueError(f"Invalid atlas_type {atlas_type} for specie {specie}. Valid options: 'Nitzsche', 'Johnson', 'Datta'.")
    elif specie == 'H' and atlas_type not in ['MNI', 'AAL']:
        raise ValueError(f"Invalid atlas_type {atlas_type} for specie {specie}. Valid options: 'MNI', 'AAL'.")

    if roi_type == 'sphere':
        # check if either coords_vox or coords_mm is provided
        if 'coords_vox' not in kwargs and 'coords_mm' not in kwargs:
            raise ValueError("For roi_type 'sphere', either 'coords_vox' or 'coords_mm' must be provided")
        if 'radius' not in kwargs:
            radius = 3
        else:
            radius = kwargs['radius']
        # if the coordinates are in mm, convert to voxels
        if 'coords_mm' in kwargs:
            if specie == 'D':
                coords_vox = utils.mm_to_vox(kwargs['coords_mm'], atlas_type=atlas_type)
            elif specie == 'H':
                coords_vox = utils.mm_to_vox(kwargs['coords_mm'], sample_img=sample_img)
            else:
                raise ValueError(f"Invalid specie {specie}. Valid options: 'D', 'H'.")
        elif 'coords_vox' in kwargs:
            coords_vox = kwargs['coords_vox']
        # take first entry of session_and_run_dict to get session, run_N
        first_entry = session_and_run_dict[0]
        session = first_entry['session']
        run_N = first_entry['run_N']
        # correct session to 2 digits
        session = f"{session:02d}"
        sample_file_path = os.path.join(
                    datafolder, dataset, 'results', 'GLM', model,
                    f"{specie}-sub-{sub_N:02d}",
                    f"ses-{session}_task-{task}_run-{run_N:02d}.feat",
                    'stats', f'pe1.nii.gz'
                )
        sample_img = nib.load(sample_file_path)
        # print sample_img dimensions
        print(f"Sample image shape: {sample_img.shape}")
        # print coords_vox
        print(f"Coordinates in voxels: {coords_vox}")
        

        mask = create_sphere_mask(sample_img, coords_vox, radius)
        
        if verbose:
            # print dimensions of mask and sample_img
            print(f"Sample image shape: {sample_img.shape}, mask shape: {mask.shape}")
            # print number of voxels > 0 in mask
            print(f"Number of voxels in sphere mask: {np.sum(mask.get_fdata() > 0)}")

    elif roi_type == 'segment':
        if 'segmentation_segments' not in kwargs:
            raise ValueError("For roi_type 'segment', 'segmentation_segments' must be provided")
        if 'number_of_clusters' not in kwargs:
            raise ValueError("For roi_type 'segment', 'number_of_clusters' must be provided")
        segmentation_segments = kwargs['segmentation_segments']
        number_of_clusters = kwargs['number_of_clusters']
        raise NotImplementedError("roi_type 'segment' not implemented yet")
    elif roi_type == 'roi':
        if 'roi_file' not in kwargs:
            raise ValueError("For roi_type 'roi', 'roi_file' must be provided")
        roi_file = kwargs['roi_file']
        raise NotImplementedError("roi_type 'roi' not implemented yet")
    elif roi_type == 'anatomical':
        if 'region_name' not in kwargs:
            raise ValueError("For roi_type 'anatomical', 'region_name' must be provided")
        region_name = kwargs['region_name']
        atlas_type = kwargs.get('atlas_type', 'Nitzsche')
        raise NotImplementedError("roi_type 'anatomical' not implemented yet")
    else:
        raise ValueError("roi_type must be 'sphere', 'segment', 'roi', or 'anatomical'")
    
    # helpers
    def _pearson(x, y):
        x = x - x.mean()
        y = y - y.mean()
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return (x @ y) / denom if denom > 0 else np.nan
    def _correlation(x, y):
        return 1.0 - _pearson(x, y)

    def _kendall(x, y):
        try:
            from scipy.stats import kendalltau
        except Exception as e:
            raise ImportError("SciPy is required for method='kendall'.") from e
        return kendalltau(x, y, nan_policy='omit').correlation

    def _euclidean(x, y):
        return -float(np.linalg.norm(x - y))

    # initialize results dataframe
    results_df = pd.DataFrame()

    for entry in session_and_run_dict:
        session = entry['session']
        run_N = entry['run_N']
        # correct session to 2 digits
        session = f"{session:02d}"
        if verbose:
            print(f"Calculating subject {sub_N}, session {session}, run {run_N}")
        for i, stim_i in enumerate(stim_types):
            for j, stim_j in enumerate(stim_types):
                if i >= j:
                    continue  # avoid duplicates and self-comparison
                input_file_i = os.path.join(
                    datafolder, dataset, 'results', 'GLM', model,
                    f"{specie}-sub-{sub_N:02d}",
                    f"ses-{session}_task-{task}_run-{run_N:02d}.feat",
                    'stats', f'pe{(i+1)*2 - 1}.nii.gz'
                )
                input_file_j = os.path.join(
                        datafolder, dataset, 'results', 'GLM', model,
                        f"{specie}-sub-{sub_N:02d}",
                        f"ses-{session}_task-{task}_run-{run_N:02d}.feat",
                        'stats', f'pe{(j+1)*2 - 1}.nii.gz'
                    )
                # load beta maps
                beta_map_i = nib.load(input_file_i).get_fdata()
                beta_map_j = nib.load(input_file_j).get_fdata()
                # extract data within mask
                data_i = beta_map_i[mask.get_fdata() > 0]
                data_j = beta_map_j[mask.get_fdata() > 0]
                # compute similarity
                if method == 'pearson':
                    similarity = _pearson(data_i, data_j)
                elif method == 'correlation':
                    similarity = _correlation(data_i, data_j)
                elif method == 'kendall':
                    similarity = _kendall(data_i, data_j)
                elif method == 'euclidean':
                    similarity = _euclidean(data_i, data_j)
                # create new_row
                new_row = {
                    'sub_N': sub_N,
                    'session': session,
                    'run_N': run_N,
                    'stim_i': stim_i,
                    'stim_j': stim_j,
                    'similarity': similarity
                }
                # concat to results_df
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    return results_df

def apply_cluster_correction(datafolder, dataset, specie, model, rsa_model, radius,
                             dis_method, rsa_method, z_threshold, cluster_threshold, forced_minimal_cluster_size=None,
                             verbose=False, mask_type=None):
    """
    """
    connectivity = 26  # based on FSL default for 3D images
    # Load cluster sizes dictionary
    cluster_sizes_dict_path = (datafolder + os.sep + dataset + os.sep +
                        'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'dist' + os.sep +
                        f"{specie}-r-{radius}_{dis_method}_{rsa_method}_dist.npy")
    # check if file exists
    if os.path.exists(cluster_sizes_dict_path):
        cluster_sizes_dict = np.load(cluster_sizes_dict_path, allow_pickle=True).item()
        print(f"Loaded cluster sizes from {cluster_sizes_dict_path}")
    else:
        # trigger error
        raise FileNotFoundError(f"Cluster sizes file {cluster_sizes_dict_path} not found")

    
    if mask_type is None:
        # mean model similarity map
        mean_model_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                        f"{specie}-r-{radius}_{dis_method}_{rsa_method}_mean.nii.gz")
    else:
        # mean model similarity map without mask
        mean_model_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                        f"{mask_type}-{specie}-r-{radius}_{dis_method}_{rsa_method}_mean.nii.gz")

    # distribution mean and std maps
    distribution_mean_map_path = (datafolder + os.sep + dataset + os.sep + 
                                'results' + os.sep + 'RSA_rnd' + os.sep +
                                model + os.sep + specie + '-' + rsa_model + '_mean.nii.gz')
    distribution_std_map_path = (datafolder + os.sep + dataset + os.sep +
                                'results' + os.sep + 'RSA_rnd' + os.sep +
                                model + os.sep + specie + '-' + rsa_model + '_std.nii.gz')
    

    # load mean model similarity map
    
    mean_model_img = nib.load(mean_model_map_path)
    img_affine = mean_model_img.affine
    mean_model_img = mean_model_img.get_fdata()
    # load distribution mean and std maps
    dist_mean_img = nib.load(distribution_mean_map_path).get_fdata()
    dist_std_img = nib.load(distribution_std_map_path).get_fdata()
    # apply correction to dist_std_img to avoid division by zero
    eps = 1e-8
    valid_std = dist_std_img > eps

    # calculate z map
    z_map = np.full_like(mean_model_img, np.nan, dtype=np.float32)
    z_map[valid_std] = (
        mean_model_img[valid_std] - dist_mean_img[valid_std]
    ) / dist_std_img[valid_std]
    
    # # calculate z map
    # z_map = (mean_model_img - dist_mean_img) / dist_std_img
    # correct nan
    z_map = np.nan_to_num(z_map, nan=0.0, posinf=0.0, neginf=0.0)

    # threshold z map
    z_map_thresholded = np.where(z_map >= z_threshold, z_map, 0)
    # if forced_minimal_cluster_size is set, use it
    if forced_minimal_cluster_size is not None:
        minimal_cluster_size = forced_minimal_cluster_size
    else:
        # get minimal cluster size
        minimal_cluster_size = get_minimal_cluster_size(cluster_sizes_dict_path, z_threshold, cluster_threshold, verbose=verbose)
    print(f"Minimal cluster size for p<{cluster_threshold}: {minimal_cluster_size} voxels")
    # apply minimal cluster size threshold to z_map_thresholded, remove clusters smaller than minimal_cluster_size
    z_map_thresholded = apply_cluster_size_threshold(z_map_thresholded, minimal_cluster_size, connectivity=connectivity, verbose=verbose)
    # save corrected z map
    # "P:\userdata\raulh87\data\EmoB\results\RSA\basic\emotion-valence-basic\mean\D-r-3_mahalanobis_kendall_z.nii.gz"
    if mask_type is None:
        corrected_z_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                                model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                                f"{specie}-r-{radius}_{dis_method}_{rsa_method}_zt{z_threshold}_corrected.nii.gz")
    else:
        corrected_z_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                                model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                                f"{mask_type}-{specie}-r-{radius}_{dis_method}_{rsa_method}_zt{z_threshold}_corrected.nii.gz")
    nib.save(nib.Nifti1Image(z_map_thresholded, affine=img_affine), corrected_z_map_path)
    print(f"Saved corrected z map to {corrected_z_map_path}")
    return True

import numpy as np
from scipy import ndimage
from time import perf_counter

from typing import Callable, Union, Optional

def apply_cluster_size_threshold(
    z_map_thresholded: np.ndarray,
    minimal_cluster_size: int,
    connectivity: int = 6,
    verbose: Union[bool, Callable[[str], None]] = False
) -> np.ndarray:
    """
    Remove clusters smaller than `minimal_cluster_size` from a *3D* thresholded z-map.

    Parameters
    ----------
    z_map_thresholded : np.ndarray, shape (X, Y, Z)
        3D thresholded z-map (values kept where z >= z_threshold, else 0).
        Voxels <= 0 are treated as background.
    minimal_cluster_size : int
        Clusters with voxel counts < this value are removed (set to 0).
        If 1, nothing is removed (all clusters survive).
    connectivity : int
        Neighborhood connectivity in 3D. One of:
          - ndimage convention 1..3  (1≈6-neigh, 2≈18, 3≈26)
          - Common shorthands: 6, 18, or 26
    verbose : bool or callable
        - False (default): no logs.
        - True: print concise progress/info messages.
        - Callable[[str], None]: custom logger (e.g., `logger.info`).

    Returns
    -------
    np.ndarray
        Copy of the input with small clusters set to 0.

    Raises
    ------
    ValueError
        If the input is not 3D, or if connectivity is invalid, or if minimal_cluster_size < 1.
    """
    t0 = perf_counter()

    # Lightweight logger
    if callable(verbose):
        _log = verbose  # type: ignore[assignment]
    elif verbose:
        def _log(msg: str) -> None:
            print(msg)
    else:
        def _log(msg: str) -> None:  # no-op
            return

    arr = np.asarray(z_map_thresholded)
    if arr.ndim != 3:
        raise ValueError(f"apply_cluster_size_threshold requires a 3D array, got ndim={arr.ndim}.")

    if minimal_cluster_size < 1:
        raise ValueError("minimal_cluster_size must be >= 1.")

    # Normalize connectivity to ndimage's 1..3 scale
    if 1 <= connectivity <= 3:
        conn = connectivity
    elif connectivity in (6, 18, 26):
        conn = {6: 1, 18: 2, 26: 3}[connectivity]
    else:
        raise ValueError("Invalid connectivity for 3D. Use 1..3 or 6/18/26.")

    # Early exit: nothing to remove when threshold is 1
    if minimal_cluster_size == 1:
        _log(f"[cluster] No removal requested (minimal_cluster_size=1). Returning copy.")
        return arr.copy()

    _log(f"[cluster] shape={arr.shape}, voxels={arr.size}")
    _log(f"[cluster] connectivity={connectivity} (ndimage conn={conn}); min_size={minimal_cluster_size}")

    structure = ndimage.generate_binary_structure(rank=3, connectivity=conn)

    # Foreground is strictly > 0 to match your thresholding
    fg = arr > 0
    fg_voxels = int(fg.sum())
    _log(f"[cluster] foreground voxels > 0: {fg_voxels}")

    if fg_voxels == 0:
        _log("[cluster] No foreground voxels; returning copy.")
        return arr.copy()

    labels, nlab = ndimage.label(fg, structure=structure)
    _log(f"[cluster] connected components found (excluding background=0): {nlab}")

    if nlab == 0:
        _log("[cluster] No labeled components; returning copy.")
        return arr.copy()

    # Count sizes per label (label 0 is background)
    label_ids, counts = np.unique(labels, return_counts=True)

    # Map label -> size for easy reporting (excluding background)
    lbl_sizes = dict(zip(label_ids.tolist(), counts.tolist()))
    lbl_sizes.pop(0, None)

    # Labels to drop: exclude background (0)
    small_labels = np.array([lbl for lbl, sz in lbl_sizes.items() if sz < minimal_cluster_size], dtype=labels.dtype)
    n_small = int(small_labels.size)
    _log(f"[cluster] clusters below min size: {n_small}")

    if n_small == 0:
        _log("[cluster] Nothing to remove; returning copy.")
        return arr.copy()

    # Compute voxels to be removed (before modifying)
    to_remove_mask = np.isin(labels, small_labels)
    removed_voxels = int(to_remove_mask.sum())

    out = arr.copy()
    out[to_remove_mask] = 0

    remaining_fg_voxels = int((out > 0).sum())
    pct_removed = (removed_voxels / max(fg_voxels, 1)) * 100.0

    # A few concise stats (don’t spam)
    _log(f"[cluster] removed voxels: {removed_voxels} / {fg_voxels} ({pct_removed:.2f}%)")
    _log(f"[cluster] remaining fg voxels: {remaining_fg_voxels}")
    _log(f"[cluster] runtime: {(perf_counter() - t0)*1000:.1f} ms")

    return out



def get_minimal_cluster_size(cluster_sizes_dict_path, z_threshold, cluster_threshold, verbose=False) -> int:
    """
    Compute the minimal cluster-size threshold for cluster-extent correction.

    Parameters
    ----------
    cluster_sizes_path : str
        Path to a .npy file containing a dict. For each z-threshold key (e.g., "z3.1"),
        the value has a 'cluster_sizes' entry: a list of lists where each inner list
        contains ALL cluster sizes found for one permutation.
    z_threshold : float
        Threshold used to define clusters (e.g., 3.1 for z-maps).
    p_thr : float
        Target tail probability (e.g., 0.05). Returns the smallest integer k such that
        the empirical probability of observing a cluster of size >= k under the null
        (across permutations) is <= p_thr.

    Returns
    -------
    int
        Minimal cluster size threshold (integer).
    """
    if not (0 < cluster_threshold < 1):
        raise ValueError("p_thr must be in (0, 1).")

    cluster_sizes_dict = np.load(cluster_sizes_dict_path, allow_pickle=True).item()
    key = f"z{z_threshold}"
    if key in cluster_sizes_dict:
        sizes_list = cluster_sizes_dict[key]['cluster_sizes']
    else:
        raise KeyError(f"No cluster sizes found for {key} in {cluster_sizes_dict_path}")

    # For FWER control, use the max cluster size per permutation (0 if no clusters)
    max_per_perm = np.array([max(s) if len(s) > 0 else 0 for s in sizes_list], dtype=int)
    if verbose:
        print(f"Cluster size comparison for z>{z_threshold}, p<{cluster_threshold}:")
        for idx, s in enumerate(sizes_list):
            print(f"  Permutation {idx+1}: max cluster size = {max(s) if len(s) > 0 else 0}, all sizes = {s}")
    n_perm = len(max_per_perm)
    if n_perm == 0:
        raise ValueError("No permutations found in 'cluster_sizes'.")

    # Find the smallest k with P(max >= k) <= p_thr
    max_observed = int(max_per_perm.max())
    minimal_cluster_size = None
    for k in range(1, max_observed + 1):
        tail_prob = np.mean(max_per_perm >= k)
        if tail_prob <= cluster_threshold:
            minimal_cluster_size = k
            break

    # If even at k = max_observed the tail prob is > p_thr, bump by 1
    if minimal_cluster_size is None:
        minimal_cluster_size = max_observed + 1

    return int(minimal_cluster_size)



def count_clusters_sizes(
    nifti_path,
    threshold=None,           # e.g., 2.3 for a z-map; if None -> nonzero voxels
    two_sided=False,          # if True, count pos and neg separately
    connectivity=3,           # 1=6-conn, 2=18-conn, 3=26-conn in 3D
    volume_index=None         # for 4D images, pick a single 3D volume (int) or None => all
):
    img = nib.load(nifti_path)
    data = np.asanyarray(img.dataobj)  # lazy and memory-friendly
    data = np.nan_to_num(data)         # treat NaNs as 0
    # make sure connectivity is int and valid
    if connectivity not in (1, 2, 3):
        raise ValueError("connectivity must be 1, 2, or 3 for 3D images.")

    # Select 3D volume if 4D
    if data.ndim == 4:
        if volume_index is None:
            # Apply per-volume
            results = []
            for t in range(data.shape[-1]):
                n, sizes = _count_on_3d(data[..., t], threshold, two_sided, connectivity)
                results.append({'t': t, 'n_clusters': n, 'sizes': sizes})
            return results
        else:
            data = data[..., volume_index]

    # 3D case
    _, sizes = _count_on_3d(data, threshold, two_sided, connectivity)
    return sizes

def _count_on_3d(vol, threshold, two_sided, connectivity=3):
    # Build mask(s)
    if threshold is None:
        pos_mask = vol != 0
        neg_mask = None
    else:
        pos_mask = vol > threshold
        neg_mask = (vol < -threshold) if two_sided else None

    # Connectivity kernel (3D)
    structure = generate_binary_structure(rank=3, connectivity=connectivity)

    # Label positives
    labels_pos, n_pos = label(pos_mask, structure=structure)
    sizes_pos = np.bincount(labels_pos.ravel())[1:]  # drop background

    if neg_mask is not None:
        labels_neg, n_neg = label(neg_mask, structure=structure)
        sizes_neg = np.bincount(labels_neg.ravel())[1:]
        n_total = int(n_pos + n_neg)
        sizes = {
            'positive': np.sort(sizes_pos)[::-1].tolist(),
            'negative': np.sort(sizes_neg)[::-1].tolist()
        }
    else:
        n_total = int(n_pos)
        sizes = np.sort(sizes_pos)[::-1].tolist()

    return n_total, sizes

def nifti_mean(img_list, result_map_path=None, result_map_path_std=None, verbose=False, mask_img=None):
    """
    Compute the voxel-wise mean and std of a list of NIfTI images.

    Parameters
    ----------
    img_list : list of str
        List of file paths to NIfTI images.
    result_map_path : str, optional
        If provided, the mean image will be saved to this path.

    Returns
    -------
    mean_img : np.ndarray
        The voxel-wise mean image data.
    std_img : np.ndarray
        The voxel-wise standard deviation image data.
    """
    if len(img_list) == 0:
        raise ValueError("img_list is empty.")

    # Load the first image to get the shape and affine
    first_img = nib.load(img_list[0])
    img_shape = first_img.shape
    img_affine = first_img.affine
    if verbose:
        print(f"Computing mean for {len(img_list)} images of shape {img_shape}")
    ## Compute mean
    # Initialize an array to hold the sum
    sum_data = np.zeros(img_shape, dtype=np.float64)
    count = 0
    for img_path in img_list:
        img = nib.load(img_path)
        if img.shape != img_shape:
            raise ValueError(f"Image {img_path} has a different shape: {img.shape} != {img_shape}")
        sum_data += img.get_fdata(dtype=np.float64)
        if verbose:
            print(f"Processed {count+1}/{len(img_list)}: {img_path}")
        count += 1
    # calculate mean
    mean_data = sum_data / count
    # print message
    print(f"Computing std deviation for {len(img_list)} images of shape {img_shape}")

    ## Compute std deviation
    sum_sq_diff = np.zeros(img_shape, dtype=np.float64)
    for img_path in img_list:
        img = nib.load(img_path)
        if img.shape != img_shape:
            raise ValueError(f"Image {img_path} has a different shape: {img.shape} != {img_shape}")
        diff = img.get_fdata(dtype=np.float64) - mean_data
        sum_sq_diff += diff ** 2
    # calculate std deviation
    std_data = np.sqrt(sum_sq_diff / count)
    # check if mask_img is provided
    if mask_img is not None:
        if mask_img.shape != img_shape:
            raise ValueError(f"mask_img has a different shape: {mask_img.shape} != {img_shape}")
        if verbose:
            print("Applying mask to mean and std images")
        # apply mask to mean and std data
        mean_data = mean_data * mask_img
        std_data = std_data * mask_img


    if result_map_path:
        # save mean image
        mean_img = nib.Nifti1Image(mean_data, img_affine)
        # make sure the folder exists, if not created it
        result_map_dir = os.path.dirname(result_map_path)
        if not os.path.exists(result_map_dir):
            os.makedirs(result_map_dir)

        nib.save(mean_img, result_map_path)
        if verbose:
            print(f"Saved mean image to {result_map_path}")
        # save std image
    if result_map_path_std:
        std_img = nib.Nifti1Image(std_data, img_affine)
        result_map_dir_std = os.path.dirname(result_map_path_std)
        if not os.path.exists(result_map_dir_std):
            os.makedirs(result_map_dir_std)
        nib.save(std_img, result_map_path_std)
        if verbose:
            print(f"Saved std image to {result_map_path_std}")


    return mean_data, std_data

def kendall_tau_a(a, b):
    """
    Kendall's tau-a correlation coefficient between vectors a and b.
    Matches the behavior of the MATLAB function rankCorr_Kendall_taua:
      - removes entries with NaN in either vector
      - counts concordant/discordant pairs via sign products
      - ties contribute 0 (tau-a; no tie correction)
    Returns np.nan if fewer than 2 valid samples.
    Time complexity: O(n^2) like the MATLAB code.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    n = a.size
    if n < 2:
        return np.nan

    K = 0.0
    for k in range(n - 1):
        pair_a = np.sign(a[k] - a[k + 1:])
        pair_b = np.sign(b[k] - b[k + 1:])
        # ties yield 0, so they neither help nor hurt
        K += np.dot(pair_a, pair_b)

    return K / (n * (n - 1) / 2.0)


def _calculate_model_similarity_map(ref_img, mask_affine, meta_similarity_map,
                                    model_vector, rsa_method):
    """Compare a voxelwise pairwise-map vector with one RSA model vector."""
    result_map = np.zeros(ref_img.shape)
    for x, y, z in np.column_stack(np.where(ref_img > 0)):
        voxel_vector = meta_similarity_map[x, y, z, :]
        if np.all(np.isnan(voxel_vector)):
            continue
        if rsa_method == 'pearson':
            similarity = np.corrcoef(voxel_vector, model_vector)[0, 1]
        elif rsa_method == 'correlation':
            similarity = 1 - np.corrcoef(voxel_vector, model_vector)[0, 1]
        elif rsa_method == 'kendall':
            similarity = kendall_tau_a(voxel_vector, model_vector)
        else:
            raise ValueError(f"Unknown rsa_method: {rsa_method}")
        result_map[x, y, z] = similarity
    return nib.Nifti1Image(result_map, mask_affine)


def _compare_mahalanobis_with_model(datafolder, dataset, sub_N,
                                    session_and_run_dict, specie, model,
                                    stim_types, mask, task, radius, rsa_model,
                                    rsa_method, replace_file, verbose, rnd,
                                    reps, replace_rnd_files, mah_fold,
                                    mask_type, model_dict):
    """Compare fold-specific Mahalanobis pairwise maps with an RSA model."""
    if mah_fold not in {
        'stim-wise', 'stim-wise-multiple-folds', 'stim-wise-all-runs'
    }:
        raise ValueError(
            "Mahalanobis model comparison supports 'stim-wise', "
            "'stim-wise-multiple-folds', and 'stim-wise-all-runs'."
        )

    rsa_model_path = os.path.join(
        datafolder, dataset, 'rsa_models', f"{rsa_model}.csv"
    )
    config_path = os.path.join(
        datafolder, dataset, 'config_files', f"{specie}_{model}.yaml"
    )
    rsa_model_dict = read_model_dict(rsa_model_path)
    ref_img = nib.load(mask).get_fdata()
    mask_affine = nib.load(mask).affine
    model_root = _rsa_model_output_dir(
        datafolder, dataset, model, rsa_model, 'mahalanobis', mah_fold
    )
    rnd_model_root = _rsa_model_output_dir(
        datafolder, dataset, model, rsa_model, 'mahalanobis', mah_fold, rnd=True
    )

    entries = session_and_run_dict if mah_fold == 'stim-wise-all-runs' else [None]
    for entry in entries:
        map_labels, model_labels = _get_mahalanobis_model_plan(
            mah_fold, session_and_run_dict, model_dict, stim_types,
            rsa_model_dict['categories'], entry=entry,
        )
        pairs, model_vector = _build_mahalanobis_model_vector(
            map_labels, model_labels, rsa_model_dict
        )

        if entry is None:
            session = run_N = None
            output_dir = os.path.join(model_root, f"{specie}-sub-{sub_N:02d}")
            rnd_output_dir = os.path.join(
                rnd_model_root, f"{specie}-sub-{sub_N:02d}"
            )
        else:
            session = f"{entry['session']:02d}"
            run_N = entry['run_N']
            run_folder = f"ses-{session}_task-{task}_run-{run_N:02d}"
            output_dir = os.path.join(
                model_root, f"{specie}-sub-{sub_N:02d}", run_folder
            )
            rnd_output_dir = os.path.join(
                rnd_model_root, f"{specie}-sub-{sub_N:02d}", run_folder
            )

        output_stem = f"r-{radius}_mahalanobis_{rsa_method}"
        if mask_type:
            output_stem = f"{mask_type}-{output_stem}"
        real_output_file = os.path.join(output_dir, f"{output_stem}.nii.gz")
        if not rnd and os.path.exists(real_output_file) and not replace_file:
            if verbose:
                print(f"Skipping existing model comparison: {real_output_file}")
            continue

        meta_similarity_map = load_meta_similarity_map(
            rsa_model_path, ref_img, datafolder, dataset, specie, sub_N, session,
            run_N, config_path, dis_method='mahalanobis', radius=radius,
            verbose=verbose, mah_fold=mah_fold, pairs=pairs,
        )

        if rnd:
            os.makedirs(rnd_output_dir, exist_ok=True)
            for rnd_N in range(reps):
                output_file = os.path.join(
                    rnd_output_dir, f"{output_stem}_{rnd_N:04d}.nii.gz"
                )
                if os.path.exists(output_file) and not replace_rnd_files:
                    continue
                shuffled_vector = shuffle_vector(model_vector)
                result_img = _calculate_model_similarity_map(
                    ref_img, mask_affine, meta_similarity_map, shuffled_vector,
                    rsa_method,
                )
                nib.save(result_img, output_file)
                if verbose:
                    print(f"Saved randomized model similarity map: {output_file}")
        else:
            os.makedirs(output_dir, exist_ok=True)
            result_img = _calculate_model_similarity_map(
                ref_img, mask_affine, meta_similarity_map, model_vector,
                rsa_method,
            )
            nib.save(result_img, real_output_file)
            print(f"Saved model similarity map: {real_output_file}")

    return True


def compare_with_model2(datafolder, dataset, sub_N, session_and_run_dict,
                    specie, model, stim_types, mask, task, radius, rsa_model,
                    dis_method='pearson', rsa_method='kendall', replace_file=False, verbose=False, wait_time=300,
                    rnd=False, reps=1000, create_subject_mean=False, replace_rnd_files=False, mah_fold='stim-wise',
                    mask_type=None, categories=None, model_dict=None, skip_prefile_check=False):
    '''
    Compare pairwise similarity maps with a model.
    '''
    if dis_method == 'mahalanobis':
        return _compare_mahalanobis_with_model(
            datafolder=datafolder,
            dataset=dataset,
            sub_N=sub_N,
            session_and_run_dict=session_and_run_dict,
            specie=specie,
            model=model,
            stim_types=stim_types,
            mask=mask,
            task=task,
            radius=radius,
            rsa_model=rsa_model,
            rsa_method=rsa_method,
            replace_file=replace_file,
            verbose=verbose,
            rnd=rnd,
            reps=reps,
            replace_rnd_files=replace_rnd_files,
            mah_fold=mah_fold,
            mask_type=mask_type,
            model_dict=model_dict,
        )

    ###- Pending: Implement logic to check for existing output files, remove manually for now ###

    print(f"Pairwise vs model for: {specie}-sub-{sub_N:02d}, model {model}, rsa_model {rsa_model}...")
    # load the mask to use as reference
    print(f"loading {mask}")
    ref_img = nib.load(mask).get_fdata()
    mask_affine = nib.load(mask).affine
    
    file_list = [] # to keep track of the files available
    all_exist = True # this is a flag to indicate if all output files exist
    

   
    # if method is not mahalabinobis
    if dis_method != 'mahalanobis':
        # Check for existing output files
        for entry in session_and_run_dict:
            session = entry['session']
            run_N = entry['run_N']
            # correct session to 2 digits
            session = f"{session:02d}"
            ## check whether this is real data or if running permutations
            
            if rnd: # permutations, check in RSA_rnd if all permutations exist
                folder_permutations = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep +
                            model + os.sep + rsa_model + os.sep + f"{specie}-sub-{sub_N:02d}" + os.sep + 
                            f"ses-{session}_task-{task}_run-{run_N:02d}")
                # check if folder exists
                if os.path.exists(folder_permutations):
                    # check if there are files matching r-{radius}_{method}_{rsa_method}_rnd-*.nii.gz
                    existing_files = glob.glob(folder_permutations + os.sep + f"r-{radius}_{dis_method}_{rsa_method}_*.nii.gz")
                    if len(existing_files) >= reps:
                        if verbose:
                            print(f"Skipping: Found {len(existing_files)} permutation model comparison files in {folder_permutations}.")
                            # all done, exit function
                        continue
                    else:
                        if verbose:
                            print(f"Running missing. Found only {len(existing_files)} permutation model comparison files in {folder_permutations}, need {reps}.")
                        all_exist = False
                else: # folder does not exist, not necessary to check files
                    if verbose:
                        print(f"Running missing. Folder {folder_permutations} does not exist.")
                    all_exist = False
                    break

            else: # real data, check for the specific file
                if mask is not None:
                    output_file = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                                model + os.sep + rsa_model + os.sep + f"{specie}-sub-{sub_N:02d}" + os.sep + 
                                f"ses-{session}_task-{task}_run-{run_N:02d}" + os.sep +
                                f"{mask_type}-r-{radius}_{dis_method}_{rsa_method}.nii.gz")
                else:
                    output_file = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                                model + os.sep + rsa_model + os.sep + f"{specie}-sub-{sub_N:02d}" + os.sep + 
                                f"ses-{session}_task-{task}_run-{run_N:02d}" + os.sep +
                                f"r-{radius}_{dis_method}_{rsa_method}.nii.gz")
                if os.path.exists(output_file):
                    if verbose:
                        print(f"Skipping: Found existing model comparison file: {output_file}.")
                    continue
                        # all done, exit function
                else: # file does not exist
                    if verbose:
                        print(f"Missing. File {output_file} does not exist.")
                    all_exist = False
                    break

    elif dis_method == 'mahalanobis': # check for mahalanobis files, for which there is only per category pairs
        if mah_fold == 'stim-wise':
            # build output_file path
            output_file = os.path.join(
                        datafolder, dataset, 'results', 'RSA', model, rsa_model,
                        f"{specie}-sub-{sub_N:02d}",
                        f"{mask_type}-r-{radius}_{dis_method}_mahalanobis.nii.gz")
            if os.path.exists(output_file):
                if verbose:
                    print(f"Skipping: Found existing model comparison file: {output_file}.")
                    # all done, exit function
            else:
                if verbose:
                    print(f"Missing. File {output_file} does not exist.")
                all_exist = False
        else: # issue error
            raise NotImplementedError("Only 'stim-wise' fold is implemented for mahalanobis method.")

    
        
        

    # if all files exist, skip computation
    if all_exist:
        print(f"All output files for {specie}-sub-{sub_N:02d} already exist.")        
        if not replace_file:
            print("Skipping computation as replace_file is False.")
            return  # all files exist, skip computation
        else:
            # if not mahalanobis
            if dis_method != 'mahalanobis':
                # delete all existing files to recompute
                print("Removing existing files as replace_file is True.")
                for entry in session_and_run_dict:
                    session = entry['session']  
                    run_N = entry['run_N']
                    # correct session to 2 digits
                    session = f"{session:02d}"
                    # determine filename
                    # if mask is not None:
                    if mask is not None:
                        filename = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                                    model + os.sep + rsa_model + os.sep + f"{specie}-sub-{sub_N:02d}" + os.sep + 
                                    f"ses-{session}_task-{task}_run-{run_N:02d}" + os.sep +
                                    f"{mask_type}-r-{radius}_{dis_method}_{rsa_method}.nii.gz")
                    else:
                        filename = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                                    model + os.sep + rsa_model + os.sep + f"{specie}-sub-{sub_N:02d}" + os.sep + 
                                    f"ses-{session}_task-{task}_run-{run_N:02d}" + os.sep +
                                    f"r-{radius}_{dis_method}_{rsa_method}.nii.gz")
                    if os.path.exists(filename):
                        os.remove(filename)
                        if verbose:
                            print(f"Removed existing file: {filename}.")
                    else:
                        if verbose:
                            print(f"File to remove not found (already missing): {filename}.")
            else: # mahalanobis, only one file
                if os.path.exists(output_file):
                    os.remove(output_file)
                    if verbose:
                        print(f"Removed existing file: {output_file}.")
                else:
                    if verbose:
                        print(f"File to remove not found (already missing): {output_file}.")

    
    ### To this point, at least one output file is missing, proceed to check input files ###
    
    # if skip_prefile_check is True, skip checking for existing files and directly run the computation, otherwise check for existing files first
    if not skip_prefile_check:
        ## check if input pairwise similarity maps necessary are available for each run
        print("Checking for input pairwise similarity maps...")
        # initialize pairs_available_array a boolean array to indicate if all pairwise similarity maps are available for each run
        # initialize it to False
        all_available = True
        if dis_method != 'mahalanobis': # for mahalanobis, there is only one file per subject, no need to check pairwise similarity maps
            pairs_available_array = np.zeros(len(session_and_run_dict), dtype=bool)
            for index, entry in enumerate(session_and_run_dict):
                session = entry['session']
                run_N = entry['run_N']
                session = f"{session:02d}"
                run_folder = os.path.join(
                    datafolder, dataset, 'results', 'RSA', model,
                    f"{specie}-sub-{sub_N:02d}",
                    f"ses-{session}_task-{task}_run-{run_N:02d}"
                )
                existing_files = set(os.listdir(run_folder)) if os.path.isdir(run_folder) else set()
                pairs_available = True
                for i, stim_i in enumerate(stim_types):
                    if not pairs_available:
                        break
                    for j, stim_j in enumerate(stim_types):
                        if i >= j:
                            continue  # only upper triangle
                        fname = f"r-{radius}_{dis_method}_{stim_i}_{stim_j}.nii.gz"
                        if fname not in existing_files:
                            if verbose:
                                print(f"Missing pairwise similarity map: {os.path.join(run_folder, fname)}")
                            pairs_available = False
                            break
                if not pairs_available:
                    pairs_available_array[index] = False
                    all_available = False
            # if any of the runs are missing pairwise similarity maps, skip computation
            if not all_available:
                print("Some input pairwise similarity maps are missing. Skipping computation.")
                return
        elif dis_method == 'mahalanobis':
            mah_folder = os.path.join(
                datafolder, dataset, 'results', 'RSA', model,
                f"{specie}-sub-{sub_N:02d}"
            )
            existing_files = set(os.listdir(mah_folder)) if os.path.isdir(mah_folder) else set()
            for i, cat1 in enumerate(categories):
                if not all_available:
                    break
                for j, cat2 in enumerate(categories):
                    if i >= j:
                        continue  # avoid duplicates and self-comparison
                    fname = f"r-{radius}_mahalanobis_{cat1}_{cat2}.nii.gz"
                    if fname not in existing_files:
                        if verbose:
                            print(f"Missing pairwise similarity map: {os.path.join(mah_folder, fname)}")
                        all_available = False
                        break

        print("All input files are available.")
    else:
        print("Skipping pre-file check as skip_prefile_check is True.")
    
    ### All input files are available, check if there are temporary files to indicate other instance processing it ###
    # initialize file_list to store output files for subject mean
    file_list = []
    tmp_file = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep + 
                        model + os.sep + rsa_model + os.sep +  f"{specie}-sub-{sub_N:02d}_processing.tmp")
    if rnd: # if running permutations, check for existing temporary files
        # temporal file to indicate that this participant is being processed, ignore if skip_prefile_check is True
        if not skip_prefile_check:
            temp_folder = os.path.dirname(tmp_file)
            if not os.path.exists(temp_folder):
                os.makedirs(temp_folder, exist_ok=True)
            # check if tmp_file exists, if it does, check against wait_time
            if os.path.exists(tmp_file):
                # is the file older than wait_time?
                if time() - os.path.getmtime(tmp_file) < wait_time:
                    print(f"Skipping. Existing recent processing temporary file {tmp_file} exists and is less than {wait_time/60} minutes old.")
                    return
                else:
                    print(f"File is old, removing {tmp_file} and repeating calculation. Temporary file is older than {wait_time/60} minutes.")
                    
                    # try to remove tmp_file
                    try:
                        os.remove(tmp_file)
                    except Exception as e:
                        print(f"Error removing temporary file {tmp_file}: {e}")
            # create tmp_file
            with open(tmp_file, 'w') as f:
                f.write('Processing...\n')
            if verbose:
                print(f"Created processing temporary file {tmp_file}.")

    ### Now, all input files are available, and no recent temporary file exists,

    if dis_method != 'mahalanobis':    
        # go through each session and run,
        for entry in session_and_run_dict:
            session = entry['session']
            run_N = entry['run_N']
            # correct session to 2 digits
            session = f"{session:02d}"
            ## check for existing temp files, skip if recent temp files, if they are old, skip calculation, otherwise go on
            print(f"sub-{sub_N:02d}, ses-{session}, run-{run_N:02d}, all checks, computing pairwise similarity maps...")
            # create folder if not exists
            
        
            # try to compute pairwise similarity maps
            # try:  # if error, remove temp file and continue
                # calculate comparison with model
            compare_with_model(
                ref_img=ref_img,
                mask_affine=mask_affine,
                datafolder=datafolder,
                sub_N=sub_N,
                session=session,
                run_N=run_N,
                specie=specie,
                model=model,
                dataset=dataset,
                task=task,
                radius=radius,
                rsa_model=rsa_model,
                dis_method=dis_method,
                rsa_method=rsa_method,
                replace_file=replace_file,
                verbose=verbose,
                rnd=rnd,
                reps=reps,
                replace_rnd_files=replace_rnd_files,
                mask_type=mask_type,
            )
        # except Exception as e:
        #     print(f"Error computing pairwise similarity maps for {specie}-sub-{sub_N:02d}, ses-{session}, run-{run_N:02d}: {e}")
    elif dis_method == 'mahalanobis':
        compare_with_model(
            ref_img=ref_img,
            mask_affine=mask_affine,
            datafolder=datafolder,
            sub_N=sub_N,
            session=None,
            run_N=None,
            specie=specie,
            model=model,
            dataset=dataset,
            task=task,
            radius=radius,
            rsa_model=rsa_model,
            dis_method=dis_method,
            rsa_method=rsa_method,
            replace_file=replace_file,
            verbose=verbose,
            rnd=rnd,
            reps=reps,
            replace_rnd_files=replace_rnd_files,
            mask_type=mask_type,
        )
    # remove temp file
    if os.path.exists(tmp_file):
        # try to remove
        try:
            os.remove(tmp_file)
            print(f"Removed temporary file {tmp_file}.")
        except Exception as e:
            print(f"Error removing temporary file {tmp_file}: {e}")
        
    
    print('Pairwise similarity maps computed.')

def compare_with_model(ref_img, mask_affine, datafolder, sub_N, session, run_N, 
                       specie, model, dataset, task, radius, rsa_model, dis_method='pearson', 
                       rsa_method='pearson', replace_file=False, verbose=False, rnd=False, reps=1000, 
                       replace_rnd_files=False, mask_type=None):
    """
    Compares the meta similarity map with a given RSA model and saves the model similarity map.
    Parameters
    ----------
    ref_img : np.ndarray
        Reference image to define the space and shape of the output similarity map.
    mask_affine : np.ndarray
        Affine transformation matrix for the reference image.
    datafolder : str
        Base directory for data storage.
    sub_N : int
        Subject number.
    session : str
        Session identifier.
    run_N : int
        Run number.
    specie : str
        Species identifier.
    model : str
        Model name.
    rnd: bool
        If True, use a randomized model. For permutation testing.
    """
    print(f"Comparing meta similarity map with model {rsa_model}")
    # if method is mahalanobis, make sure that session and run_N are None, otherwise stop
    if dis_method == 'mahalanobis':
        # check that session and run_N are None
        if session is not None or run_N is not None:
            raise ValueError("For mahalanobis method, session and run_N must be None.")
        else:
            # set both to 0
            session, run_N = 0, 0

    print(f"for {specie}-sub-{sub_N:02d}, ses-{session}, run-{run_N:02d}...")
    rsa_model_path = datafolder + os.sep + dataset + os.sep + 'rsa_models' + os.sep + rsa_model + ".csv"
    config_path = datafolder + os.sep + dataset + os.sep + 'config_files' + os.sep + specie + '_' + model + '.yaml'
 
    # Load config.yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # load rsa model dictionary
    rsa_model_dict = read_model_dict(rsa_model_path)
    # build model_vector
    model_vector = np.zeros(len(rsa_model_dict['pairs']))
    # print('Model vector length: ' + str(len(model_vector)))
    
    for i, pair in enumerate(rsa_model_dict['pairs']):
        model_vector[i] = rsa_model_dict['model'][pair[0]][pair[1]]
    
    print("Loading meta similarity map...")
    meta_similarity_map = load_meta_similarity_map(rsa_model_path, ref_img, datafolder, dataset, specie, sub_N, session, run_N, config_path, dis_method=dis_method, radius=radius, verbose=verbose)
    print("Meta similarity map loaded.")
    # create similarity_table (x, y, z) of all voxels in the mask, results will be added here
    similarity_table = np.column_stack(np.where(ref_img > 0))
    # add 1 to x, y, z to match 1-based indexing in itk-snap
    similarity_table += 1
    # add a column for similarity values, initialized to NaN
    similarity_table = np.hstack((similarity_table, np.full((similarity_table.shape[0], 1), np.nan)))
    # initialize warning table
    warning_table = []

    rnd_N_list = list(range(0, reps))
    # randomize the elements in rnd_N_list
    random.shuffle(rnd_N_list)
    

    for indx, rnd_N in enumerate(rnd_N_list):
        # create an result_map based on the reference image
        result_map = np.zeros(ref_img.shape)
        if rnd:
            # if rnd is True, permute the model values
            # rsa_model_dict = shuffle_model(rsa_model_dict)
            # if method is not mahalanobis, output includes session and run
            if dis_method != 'mahalanobis':
                output_folder = (datafolder + os.sep + dataset + os.sep + 'results' + 
                os.sep + 'RSA_rnd' + os.sep + model + os.sep + rsa_model + os.sep + 
                f"{specie}-sub-{sub_N:02d}" + os.sep +
                f"ses-{session}_task-{task}_run-{run_N:02d}")
            else:
                output_folder = (datafolder + os.sep + dataset + os.sep + 'results' + 
                os.sep + 'RSA_rnd' + os.sep + model + os.sep + rsa_model + os.sep + 
                f"{specie}-sub-{sub_N:02d}")
            # build output filename _[4 digit padded rnd_N]
            output_file = os.path.join(output_folder, f"r-{radius}_{dis_method}_{rsa_method}_{rnd_N:04d}.nii.gz")
            # check if output_file exists
            if os.path.exists(output_file) and not replace_rnd_files:
                print(f"rnd {(indx+1):04d}/{reps:04d} exist, skipping")
                continue
            else:
                print(f"{output_file} does not exist or replace_rnd_files is {replace_rnd_files}, computing rnd {(indx+1):04d}/{reps:04d}")
            model_vector = shuffle_vector(model_vector)
        else:
            if indx > 0:
                if verbose:
                    print("real data, skipping further repetitions")
                break
            # if method is not mahalanobis, output includes session and run
            if dis_method != 'mahalanobis':
                output_folder = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep + 
                model + os.sep + rsa_model + os.sep +  
                f"{specie}-sub-{sub_N:02d}" + os.sep + 
                f"ses-{session}_task-{task}_run-{run_N:02d}")
            else:
                output_folder = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep + 
                model + os.sep + rsa_model + os.sep +  
                f"{specie}-sub-{sub_N:02d}")

            # build output filename
            output_file = os.path.join(output_folder, f"{mask_type}-r-{radius}_{dis_method}_{rsa_method}.nii.gz")
        # check if output_file exists
        # if os.path.exists(output_file) and not replace_file:
        #     print(f"Output file {output_file} already exists. Skipping...")
        #     return output_file, True

        # go through each voxel in similarity_table and calculate similarity between meta_similarity_map and model_vector
        for i, (x, y, z) in enumerate(similarity_table[:, :3]):
            xi, yi, zi = int(x), int(y), int(z)
            if i % 100 == 0 and verbose:
                print(f"Processing voxel {i+1}/{similarity_table.shape[0]} at ({xi},{yi},{zi})")
            voxel_meta_vector = meta_similarity_map[xi-1, yi-1, zi-1, :]  # -1 for 0-based indexing
            if np.all(np.isnan(voxel_meta_vector)):
                # add to warning table
                warning_table.append((xi, yi, zi))
                continue  # skip if all values are NaN
            # calculate similarity
            if rsa_method == 'pearson':
                # calculate pearson correlation
                sim = np.corrcoef(voxel_meta_vector, model_vector)[0, 1]
            elif rsa_method == 'correlation':
                # calculate correlation distance
                sim = 1 - np.corrcoef(voxel_meta_vector, model_vector)[0, 1]
            elif rsa_method == 'kendall':
                sim = kendall_tau_a(voxel_meta_vector, model_vector)

            # elif rsa_method == 'kendall':

            else:
                raise ValueError(f"Unknown rsa_method: {rsa_method}")
            similarity_table[i, 3] = sim
            # assign sim to result_map
            result_map[xi-1, yi-1, zi-1] = sim  # -1 for 0-based indexing
        # save result_map to output_file
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
        nib.save(nib.Nifti1Image(result_map, mask_affine), output_file)
        print(f"Saved similarity map to {output_file}")

def remove_existing_rnd_files(datafolder, dataset, sub_N, session, run_N, specie, model, task, mask_type, 
                              radius, rsa_model, dis_method='pearson', rsa_method='pearson', verbose=False, reps=1000):
    """
    Removes existing randomized RSA result files for a given subject/session/run.
    Parameters
    ----------
    datafolder : str
        Base directory for data storage.
    dataset : str
        Dataset name.
    sub_N : int
        Subject number.
    session : str
        Session identifier.
    run_N : int
        Run number.
    specie : str
        Species identifier.
    model : str
        Model name.
    task : str
        Task name.
    mask_type : str
        Mask type.
    radius : int
        Search radius.
    rsa_model : str
        RSA model name.
    dis_method : str
        Method for similarity calculation.
    rsa_method : str
        RSA method for similarity calculation.
    verbose : bool
        If True, prints additional information.
    """
    output_folder = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep + model + os.sep + rsa_model + os.sep +  f"{specie}-sub-{sub_N:02d}" + os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}"
    if not os.path.exists(output_folder):
        if verbose:
            print(f"No existing folder {output_folder} to remove files from.")
        return
    # go through all the possible files and remove them
    for rep in range(reps):
        file_path = output_folder + os.sep + f"r-{radius}_{dis_method}_{rep:04d}.nii.gz"
        if os.path.exists(file_path):
            if verbose:
                print(f"Removing {file_path}")
            os.remove(file_path)

def load_meta_similarity_map2(ref_img, datafolder, dataset, specie, sub_N, session, run_N, config_path, rsa_class=None, dis_method='pearson', radius=3, verbose=False):
    '''
    Loads the meta similarity map for given parameters.
    
    ref_img: reference image to get the shape of the meta similarity map
    datafolder, dataset, specie, sub_N, session, run_N: parameters to locate the pairwise similarity maps
    config_path: path to the configuration file
    rsa_class: type of RSA comparisons saved in rsa_models/by_class
    dis_method: method used to calculate pairwise similarity maps, used to locate the files
    radius: radius used to calculate pairwise similarity maps, used to locate the files
    verbose: if True, prints additional information

    returns:
    meta_similarity_map: 4D numpy array of shape (X, Y, Z, n_pairs) where n_pairs is the number of unique pairs of categories in the RSA model, containing the pairwise similarity values for each voxel (coordinate [x, y, z]) and each pair of categories

    '''
    # get shape from ref_img
    X, Y, Z = ref_img.shape
    # "P:\userdata\raulh87\data\EmoC\rsa_models\by_class\class_hum.csv"
    # load table with pairs for the class
    pairs_table = pd.read_csv(os.path.join(datafolder, dataset, 'rsa_models', 'by_class', f'class_{rsa_class}.csv'))
    # initialize meta_similarity_map
    n_pairs = len(pairs_table)
    meta_similarity_map = np.empty((X, Y, Z, n_pairs), dtype=np.float32)

    # go over each row
    for index, row in pairs_table.iterrows():
        cat1 = row['cat1']
        cat2 = row['cat2']
        if verbose:
            print(f"Pair {index}: {cat1} vs {cat2}")
        map = load_pairwise_similarity_map(datafolder, dataset, specie, sub_N, session, run_N, cat1, cat2, config_path, dis_method=dis_method, radius=radius, verbose=verbose)
        # if map == 0, means that the pair is missing, return 0
        if isinstance(map, int) and map == 0:
            warnings.warn(f"Pairwise similarity map for {cat1} vs {cat2} is missing. Returning empty meta similarity map.")
            return 0

        meta_similarity_map[..., index] = map
    return meta_similarity_map


def load_meta_similarity_map(rsa_model_path, ref_img, datafolder, dataset, specie, sub_N, session, run_N, config_path, dis_method='pearson', radius=3, verbose=False, mah_fold='stim-wise', pairs=None):
    '''
    Loads the meta similarity map for given parameters.
    
    rsa_model_path: path to the RSA model excel file, used to determine the number of pairs and their names
    ref_img: reference image to get the shape of the meta similarity map
    datafolder, dataset, specie, sub_N, session, run_N: parameters to locate the pairwise similarity maps
    config_path: path to the configuration file
    dis_method: method used to calculate pairwise similarity maps, used to locate the files
    radius: radius used to calculate pairwise similarity maps, used to locate the files
    verbose: if True, prints additional information

    returns:
    meta_similarity_map: 4D numpy array of shape (X, Y, Z, n_pairs) where n_pairs is the number of unique pairs of categories in the RSA model, containing the pairwise similarity values for each voxel (coordinate [x, y, z]) and each pair of categories

    '''
    #

    # rsa_class load the csv with comparisons

    # get shape from ref_img
    X, Y, Z = ref_img.shape
    
    rsa_model_dict = read_model_dict(rsa_model_path)
    if pairs is None:
        pairs = rsa_model_dict['pairs']
    
    n_pairs = len(pairs)
    meta_similarity_map = np.empty((X, Y, Z, n_pairs), dtype=np.float32)
    pair_names = [] # to store pair names

    k = 0 # index for meta_similarity_map
    for cat1, cat2 in pairs:
        map = load_pairwise_similarity_map(
            datafolder, dataset, specie, sub_N, session, run_N, cat1, cat2,
            config_path, dis_method=dis_method, radius=radius, verbose=verbose,
            mah_fold=mah_fold,
        )
        if isinstance(map, int) and map == 0:
            raise FileNotFoundError(
                f"Pairwise similarity map is missing for {cat1!r} vs {cat2!r}."
            )

        if verbose:
            pair_name = f"{cat1}_{cat2}"
            print(f"Loaded pair {pair_name} into index {k}")
            pair_names.append(pair_name) # for sanity check
        meta_similarity_map[..., k] = map
        k += 1
    return meta_similarity_map

def load_pairwise_similarity_map(datafolder, dataset, specie, sub_N, session, run_N, stim_1_name, stim_2_name, config_path, dis_method='pearson', radius=3, verbose=False, mah_fold='stim-wise'):
    # import warnings
    

    # Loads similarity map that compares two stimuli for given parameters

    # Load config.yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = config['model']
    
    task = config['task']
    if dis_method != 'mahalanobis' or mah_fold == 'stim-wise-all-runs':
        if session is None or run_N is None:
            raise ValueError(
                "Per-run pairwise map loading requires both session and run_N."
            )
        # build file path for pairwise similarity map
        file_path = (datafolder + os.sep + dataset + os.sep + 'results' + 
                    os.sep + 'RSA' + os.sep + model + os.sep + 
                    f"{specie}-sub-{sub_N:02d}" + os.sep + 
                    f"ses-{session}_task-{task}_run-{run_N:02d}" + os.sep + 
                    f"r-{radius}_{dis_method}_{stim_1_name}_{stim_2_name}.nii.gz")
        file_path_inverted = (datafolder + os.sep + dataset + os.sep + 'results' +
                os.sep + 'RSA' + os.sep + model + os.sep +
                f"{specie}-sub-{sub_N:02d}" + os.sep +
                f"ses-{session}_task-{task}_run-{run_N:02d}" + os.sep +
                f"r-{radius}_{dis_method}_{stim_2_name}_{stim_1_name}.nii.gz")
    else: # method is mahalanobis, the file name is different, it does not include session and run
        # build file path for Mahalanobis similarity map
        file_path = (datafolder + os.sep + dataset + os.sep + 'results' + 
                    os.sep + 'RSA' + os.sep + model + os.sep + 
                    f"{specie}-sub-{sub_N:02d}" + os.sep + 
                    f"r-{radius}_{dis_method}_{stim_1_name}_{stim_2_name}.nii.gz")
        # build file path for inverted Mahalanobis similarity map
        file_path_inverted = (datafolder + os.sep + dataset + os.sep + 'results' + 
                    os.sep + 'RSA' + os.sep + model + os.sep +
                    f"{specie}-sub-{sub_N:02d}" + os.sep +
                    f"r-{radius}_{dis_method}_{stim_2_name}_{stim_1_name}.nii.gz")
    # check if file exists, if not try with inverted pair name
    if os.path.exists(file_path):
        if verbose:
            print(f"Loading {file_path}")
    elif os.path.exists(file_path_inverted):
        if verbose:
            print(f"Loading {file_path_inverted}")
        file_path = file_path_inverted
    else:
        # issue warning neiter map found
        warnings.warn(f"Neither {file_path} nor {file_path_inverted} exist. Returning empty")
        # return empty or nothing
        return 0
        
        
    map = nib.load(file_path).get_fdata()
    return map

def read_model_dict(model_path, erase_existing_npy=False, return_all_comparisons=False):
    ''' Reads the RSA model from an excel file and returns as a dictionary 
    Input:
        model_path: path to the excel file
    Output:
        rsa_model_dict: dictionary with the RSA model
            - 'model': pairwise dictionary with similarity values
            - 'categories': list of categories
            - 'pairs': list of tuples with category pairs
    '''
    # check if model_path is an excel or csv file
    if model_path.endswith('.xlsx'):
        # rsa_model_dict is saved with the same name as model_path but with .npy extension
        rsa_model_dict_path = model_path.replace('.xlsx', '.npy')
    elif model_path.endswith('.csv'):
        rsa_model_dict_path = model_path.replace('.csv', '.npy')
    else:
        raise ValueError("model_path must be an excel (.xlsx) or csv (.csv) file.")
    

    if not return_all_comparisons:
        if os.path.exists(rsa_model_dict_path):
            if not erase_existing_npy:
                # load and return
                rsa_model_dict = np.load(rsa_model_dict_path, allow_pickle=True).item()
                return rsa_model_dict
            else:
                # delete rsa_model_dict_path
                os.remove(rsa_model_dict_path)

    # if the model_path is xlsx, read with read_excel, if csv, read with read_csv
    if model_path.endswith('.xlsx'):
        # read excel file
        model_table = pd.read_excel(model_path)
    else:
        # read csv file
        model_table = pd.read_csv(model_path)
    # get all columns
    categories = model_table.columns.tolist()
    # drop first element
    categories = categories[1:]
    # print(categories)
    # categories = ['H-1', 'H-2', 'H-3', 'H-4', 'H-5', 'H-6', 'A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'A-6', 'C-1', 'C-2', 'C-3', 'C-4', 'C-5', 'C-6']
    # build pairwise dictionary
    pairs = [] # to store pairs

    rsa_model_dict = {} # initialize dictionary
    
    rsa_model_dict['model'] = {} # initialize model
    for indx1,cat1 in enumerate(categories):
        rsa_model_dict['model'][cat1] = {}
        for indx2,cat2 in enumerate(categories):
            if indx1 >= indx2:
                continue
            rsa_model_dict['model'][cat1][cat2] = model_table[cat1][indx2]
            pairs.append((cat1, cat2))  # add pair to list
    
    # save categories and pairs
    rsa_model_dict['categories'] = categories
    rsa_model_dict['pairs'] = pairs
    # save rsa_model_dict as npy
    np.save(rsa_model_dict_path, rsa_model_dict)
    # if return_all_comparisons, overwrite rsa_model_dict to include all comparisons
    if return_all_comparisons:
        # print("Returning all pairwise comparisons in model dictionary...")
        rsa_model_dict['model'] = {} # initialize model
        
        for indx1,cat1 in enumerate(categories):
            rsa_model_dict['model'][cat1] = {}
            
            for indx2,cat2 in enumerate(categories):
                rsa_model_dict['model'][cat1][cat2] = model_table[cat1][indx2]
                pairs.append((cat1, cat2))  # add pair to list


    # return dictionary
    return rsa_model_dict

def similarity_searchlight(map_1, map_2, mask, radius, dis_method):
    """
    Calculate a voxelwise similarity map between two volumes using a spherical
    searchlight. For each voxel within `mask`, gather the sphere of neighbors
    (Euclidean radius in voxel units) and compute the similarity between the
    vectors of values from map_1 and map_2 within that sphere.

    Parameters
    ----------
    map_1, map_2 : np.ndarray
        Arrays of identical shape (N-dim; typically 3D) with numeric values.
    mask : np.ndarray (bool)
        Boolean array of same shape; computation is performed only where mask==True.
    radius : int or float
        Searchlight radius in voxel units (isotropic).
    dis_method : {'mahalanobis','pearson','euclidean','kendall'}
        Similarity metric:
          - 'pearson'  -> Pearson r
          - 'kendall'  -> Kendall's tau_b (requires SciPy)
          - 'euclidean'-> negative Euclidean distance (−||x−y||_2)
          - 'mahalanobis' -> negative (regularized) Mahalanobis distance

    Returns
    -------
    similarity_map : np.ndarray
        Float array of the same shape as inputs; NaN where not computed.

    Notes
    -----
    * Correlations return coefficients in [-1,1].
    * Distances are negated so that larger = more similar (consistent with
      PyMVPA’s use of distances; we convert to a similarity-like quantity).
    * Mahalanobis uses a regularized covariance estimate over sphere features.
      If scikit-learn is available, Ledoit–Wolf shrinkage is used; otherwise a
      small ridge (λ) is added to the sample covariance for stability.

    """
    # --- validations ---
    if map_1.shape != map_2.shape or map_1.shape != mask.shape:
        raise ValueError("map_1, map_2, and mask must have identical shapes.")
    if mask.dtype != bool:
        mask = mask.astype(bool)
    dis_method = dis_method.lower()
    if dis_method not in {"mahalanobis", "pearson", "euclidean", "kendall", "correlation"}:
        raise ValueError("dis_method must be one of: 'mahalanobis','pearson','euclidean','kendall','correlation'")

    similarity_map = np.full(mask.shape, np.nan, dtype=float)
    ndim = map_1.ndim
    r = float(radius)
    rad = int(np.floor(r))

    # build integer offsets for an N-D sphere
    ranges = [np.arange(-rad, rad + 1) for _ in range(ndim)]
    grid = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1).reshape(-1, ndim)
    keep = (grid.astype(float) ** 2).sum(axis=1) <= r * r + 1e-12
    offsets = grid[keep]

    # helpers
    def _pearson(x, y):
        x = x - x.mean()
        y = y - y.mean()
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return (x @ y) / denom if denom > 0 else np.nan
    def _correlation(x, y):
        return 1.0 - _pearson(x, y)

    def _euclidean(x, y):
        return -float(np.linalg.norm(x - y))

    def _mahalanobis(x, y):
        # regularized Mahalanobis distance
        cov = np.cov(x, rowvar=False)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.pinv(cov)
        diff = x - y
        return -float(np.sqrt(diff @ inv_cov @ diff.T))
    # iterate over all voxels in mask
    it = np.argwhere(mask)
    for center in it:
        neigh = center + offsets
        # in-bounds
        inb = np.all((neigh >= 0) & (neigh < np.array(mask.shape)), axis=1)
        neigh = neigh[inb]
        if neigh.size == 0:
            continue
        # apply mask within sphere
        msub = mask[tuple(neigh.T)]
        if not np.any(msub):
            continue
        neigh = neigh[msub]

        x = map_1[tuple(neigh.T)].astype(float)
        y = map_2[tuple(neigh.T)].astype(float)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]; y = y[valid]

        # need at least 2 points for correlation; 1+ for distances
        if dis_method in {"pearson", "kendall"} and x.size < 2:
            val = np.nan
        elif dis_method in {"euclidean", "mahalanobis"} and x.size < 1:
            val = np.nan
        else:
            if dis_method == "pearson":
                val = _pearson(x, y)
            elif dis_method == "correlation":
                val = _correlation(x, y)
            elif dis_method == "kendall":
                val = kendall_custom(x, y)
            elif dis_method == "euclidean":
                val = _euclidean(x, y)
            elif dis_method == "mahalanobis":
                raise NotImplementedError("This version is deprecated.")
        similarity_map[tuple(center)] = val
                

    return similarity_map


def crossnobis(Y, labels, partitions, sigma=None, shrinkage='ledoitwolf', return_rdm=True):
    """
    Cross-validated Mahalanobis (crossnobis) distances for RSA.

    Parameters
    ----------
    Y : array, shape (n_obs, n_features)
        Pattern estimates (e.g., trial- or run-level betas).
    labels : array, shape (n_obs,)
        Condition ID for each observation (int/str).
    partitions : array, shape (n_obs,)
        Partition/run/session ID for each observation (e.g., run number).
    sigma : None or array (n_features, n_features), optional
        Noise covariance. If None, it is estimated from pooled run-residuals
        with shrinkage (see `shrinkage`).
    shrinkage : {'ledoitwolf','oas','ridge','identity'}, optional
        How to estimate sigma when `sigma is None`.
        - 'ledoitwolf' (default): Ledoit–Wolf shrinkage (sklearn)
        - 'oas': Oracle Approximating Shrinkage (sklearn)
        - 'ridge': diagonal ridge using feature variances
        - 'identity': no whitening (yields cross-validated Euclidean)
    return_rdm : bool, optional
        If True return a (n_conditions x n_conditions) symmetric RDM.
        If False return the condensed upper-triangular vector.

    Returns
    -------
    D : ndarray, shape (n_conditions, n_conditions) or (n_conditions * (n_conditions - 1) / 2,)
        Crossnobis distances (unbiased estimate of squared Mahalanobis distance).
        

    Notes
    -----
    Let Δ_m = u_i,m − u_j,m be the run-m difference (whitened). The crossnobis
    for pair (i,j) is the mean cross-run inner product:
        d(i,j) = (1 / (M*(M-1))) * sum_{m≠n} Δ_m · Δ_n
    which equals the leave-one-run-out formulation. With identity sigma this
    reduces to cross-validated Euclidean distance. :contentReference[oaicite:1]{index=1}
    """
    Y = np.asarray(Y, float)
    labels = np.asarray(labels)
    partitions = np.asarray(partitions)

    if Y.ndim != 2:
        raise ValueError("Y must be 2D (n_obs x n_features).")
    if not (len(Y) == len(labels) == len(partitions)):
        print(f"Lengths: Y={len(Y)}, labels={len(labels)}, partitions={len(partitions)}")
        raise ValueError("Y, labels, and partitions must have the same length.")

    conds = np.unique(labels)
    runs = np.unique(partitions)
    C, M, P = len(conds), len(runs), Y.shape[1]
    if M < 2:
        raise ValueError("Need at least 2 partitions for cross-validation.")

    # --- per-run condition means: U[run, cond, feat] ---
    U = np.zeros((M, C, P), float)
    for mi, m in enumerate(runs):
        for ci, c in enumerate(conds):
            idx = (partitions == m) & (labels == c)
            if not np.any(idx):
                raise ValueError(f"No observations for condition {c} in partition {m}.")
            U[mi, ci] = Y[idx].mean(axis=0)

    # --- estimate noise covariance if not given (pooled run-residuals across conditions) ---
    if sigma is None:
        R = (U - U.mean(axis=0, keepdims=True)).reshape(M*C, P)  # samples x features
        if shrinkage == 'ledoitwolf':
            try:
                from sklearn.covariance import LedoitWolf
                sigma = LedoitWolf().fit(R).covariance_
            except Exception:
                var = R.var(axis=0, ddof=1)
                lam = 1e-3 * (var.mean() + 1e-12)
                sigma = np.diag(var + lam)
        elif shrinkage == 'oas':
            try:
                from sklearn.covariance import OAS
                sigma = OAS().fit(R).covariance_
            except Exception:
                var = R.var(axis=0, ddof=1); lam = 1e-3 * (var.mean() + 1e-12)
                sigma = np.diag(var + lam)
        elif shrinkage == 'ridge':
            var = R.var(axis=0, ddof=1); lam = 1e-3 * (var.mean() + 1e-12)
            sigma = np.diag(var + lam)
        elif shrinkage == 'identity':
            sigma = np.eye(P)
        else:
            raise ValueError("Unknown shrinkage option.")
    else:
        sigma = np.asarray(sigma, float)
        if sigma.shape != (P, P):
            raise ValueError(f"sigma must be ({P},{P}).")

    # --- whiten: apply Σ^{-1/2} so dot-products equal δ^T Σ^{-1} δ ---
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, np.finfo(float).eps, None)
    Winvhalf = (eigvecs / np.sqrt(eigvals)).dot(eigvecs.T)   # Σ^{-1/2}
    Z = np.einsum('mcp,pk->mck', U, Winvhalf)               # whitened patterns

    # --- crossnobis for each condition pair ---
    D = np.zeros((C, C), float)
    denom = M * (M - 1)
    for i in range(C):
        for j in range(i + 1, C):
            Delta = Z[:, i, :] - Z[:, j, :]         # shape: (M, P)
            s = Delta.sum(axis=0)                   # sum over runs
            # sum_{m≠n} Δ_m·Δ_n = ||∑Δ_m||^2 − ∑||Δ_m||^2
            sum_off = float(np.dot(s, s) - np.einsum('mp,mp->', Delta, Delta))
            D[i, j] = D[j, i] = sum_off / denom

    if return_rdm:
        return D
    else:
        iu = np.triu_indices(C, 1)
        return D[iu]


def threshold_z_maps(datafolder, dataset, specie, model, task, radius,
                        dis_method, rsa_method, rsa_model,
                        z_threshold=3.1, connectivity=3,
                        verbose=True, replace_file=True,
                        reps_group=1000):
    """
    Threshold z maps and calculate cluster size distribution
    """
    # Load cluster sizes
    cluster_sizes_dict_path = (datafolder + os.sep + dataset + os.sep +
                        'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'dist' + os.sep +
                        f"{specie}-r-{radius}_{dis_method}_{rsa_method}_dist.npy")
    # check if file exists
    if os.path.exists(cluster_sizes_dict_path):
        cluster_sizes_dict = np.load(cluster_sizes_dict_path, allow_pickle=True)
        print(f"Loaded cluster sizes from {cluster_sizes_dict_path}")
    else: # file does not exist, create empty dictionary
        cluster_sizes_dict = {}  # dictionary to store cluster sizes for each rnd z map
        print(f"Cluster sizes file {cluster_sizes_dict_path} not found. Starting from scratch.")
        
        # initialize log
    # check if there a log (same filename but with _log.txt extension)
    log_file_path = cluster_sizes_dict_path.replace('.npy', '_log.txt')
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            log = f.readlines()
        print(f"Loaded log from {log_file_path}")
    else:
        log = []  # initialize empty log
        print(f"No log found at {log_file_path}. Starting from scratch.")

    # check if cluster_sizes_dict has the current z_threshold and connectivity
    key = f"z{z_threshold}_c{connectivity}"
    if key in cluster_sizes_dict:
        print(f"Found cluster sizes for {key}")
        # stop script
        sys.exit(0)
    else:
        print(f"No cluster sizes found for {key}")
        cluster_sizes_dict[key] = {}  # initialize empty dictionary for this threshold and connectivity

    # rsa_model_path = datafolder + os.sep + dataset + os.sep + 'rsa_models' + os.sep + rsa_model + ".xlsx"
    config_path = datafolder + os.sep + dataset + os.sep + 'config_files' + os.sep + model + '.yaml'

    # Load config.yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from {config_path}")

    # add clear mark to log
    log.append("\n" + "="*50 + "\n")

    # add current date and time to log
    from datetime import datetime
    log.append(f"Log date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # add threshold to log
    log.append(f"Z threshold: {z_threshold}")
    log.append(f"Cluster connectivity: {connectivity}")

    ## Calculate z map for permutations (RSA_rnd folder)
    # check how many rnd mean files are available
    available_files = []
    missing_files = []

    # find all files that match pattern (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep +
                        # model + os.sep + rsa_model + os.sep +
                        # f"r-{radius}_{method}_{rsa_method}_z_*.nii.gz")
    # list all files matching the pattern
    file_list = glob.glob(datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep +
                        model + os.sep + rsa_model + os.sep +
                        f"r-{radius}_{dis_method}_{rsa_method}_z_*.nii.gz")
    # sort file_list
    file_list.sort()
    # initialize sizes_list to store all cluster sizes (same size as file_list)
    sizes_list = np.zeros((len(file_list),), dtype=object)

    for i, file in enumerate(file_list):
        print(f"{i+1} of {len(file_list)}: Processing file {file}...")
        sizes = count_clusters_sizes(file, threshold=z_threshold, connectivity=connectivity) 
        # write sizes to sizes_list
        sizes_list[i] = sizes
        print(f"Found {len(sizes)} clusters")

    # calculate total number of images processed
    number_of_images = len(file_list)
    # add to log
    log.append(f"Processed {number_of_images} z map files for cluster size distribution.")
    # add number of processed files cluster_sizes_dict
    cluster_sizes_dict[key]['number_of_images'] = number_of_images
    # sizes_list is an array of arrays, convert to a single list
    sizes_list = np.concatenate(sizes_list).ravel()
    cluster_sizes_dict[key]['cluster_sizes'] = sizes_list.tolist()  # convert to list for saving as npy
    # log info added to cluster_sizes_dict
    log.append(f"Data added to cluster_sizes_dict under key {key}. Total clusters found: {len(sizes_list)}")

    # create directory of cluster_sizes_dict if it does not exist
    dist_folder = os.path.join(datafolder, dataset, 'results', 'RSA', model, rsa_model, 'dist')
    if not os.path.exists(dist_folder):
        os.makedirs(dist_folder)

    # save cluster_sizes_dict to cluster_sizes_dict_path
    with open(cluster_sizes_dict_path, 'wb') as f:
        np.save(f, cluster_sizes_dict)

    print(f"Saved cluster sizes to {cluster_sizes_dict_path}")
    # add to log
    log.append(f"Saved cluster sizes to {cluster_sizes_dict_path}")
    # print how many images were processed and which were processed
    log.append(f"Processed {len(file_list)} z map files for cluster size distribution.")
    log.append(f"Processed files: {file_list}")
    # save log
    log_path = cluster_sizes_dict_path.replace('.npy', '_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log))
    print(f"Saved log to {log_path}")

def calculate_beta_mapsH(datafolder, dataset, model, specie, sub_N, session, run_N, task,
                        stim_types, design_template, atlas_file,
                        smooth,
                        radius_fwd,
                        threshold_fwd,
                        redo_if_exists,
                        wait_time=300):
    '''
    
    Calculate maps for given human participant, session, run using FSL FEAT
    Parameters
    ----------
    datafolder : str. Path to the data folder
    dataset : str. Dataset name
    model : str. Model name
    specie : str. Species identifier
    sub_N : int. Subject number
    session : str. Session identifier
    run_N : int. Run number
    task : str. Task name
    stim_types : list. List of stimulus types
    design_template : str. Path to the design template file
    atlas_file : str. Path to the atlas file
    smooth : int. Smoothing kernel size.
    radius_fwd : float. Radius for framewise displacement calculation
    threshold_fwd : float. Threshold for framewise displacement calculation
    redo_if_exists : bool. If True, redo the calculation even if maps exist
    wait_time : int. Time to wait between checks for existing maps
    
    '''
    print('### Running calculate_beta_mapsH with the following parameters:')
    # datafolder
    print(f"datafolder: {datafolder}")

    session = str(session).zfill(2)
    print(f"sub-{sub_N:02d}, ses-{session}, run-{run_N:02d}...")
    # "P:\userdata\raulh87\data\EmoB\results\GLM\basic\D-sub-01\ses-01_task-EmoB_run-01_(len(stim_types))"
    design_template_modified = (datafolder + os.sep + dataset + os.sep + 
    'results' + os.sep + 'GLM' + os.sep + model + os.sep + specie +
    '-sub-' + str(sub_N).zfill(2) + os.sep +
    f"ses-{session}_task-{task}_run-{run_N:02d}_{len(stim_types)}.fsf"
    )
    # create a copy of design_template with len(stim_types) in the filename
    # create directories if they do not exist
    os.makedirs(os.path.dirname(design_template_modified), exist_ok=True)
    
    # shutil.copy(design_template, design_template_modified)
    # Create an appropiate design for the number of stim types
    utils.generate_fsf(len(stim_types), design_template, design_template_modified)
    # print(f"Design template modified {design_template_modified}.")
    ## stop program here
    # trigger error
    # raise NotImplementedError("Stopping execution for testing purposes.")

    
    ## Determine if this model/sub/run should be run or skipped
    # Output directory for FSL
    fsl_out = os.path.join(
        datafolder, dataset, 'results', 'GLM', model,
        f"{specie}-sub-{sub_N:02d}",
        f"ses-{session}_task-{task}_run-{run_N:02d}.feat"
    )
    # create a tmp file to indicate processing, replace .feat with .tmp
    tmp_file = fsl_out.replace('.feat', '.tmp')
    # check if tmp file exists
    if os.path.exists(tmp_file):
        # check how old the file is
        mod_time = os.path.getmtime(tmp_file)
        current_time = time()
        # if the file is recent, another process is probably running it, skip
        if current_time - mod_time < wait_time:
            print(f"Temporary file {tmp_file} exists and is recent. Another process may be running it. Skipping GLM...")
            return
        else:
            print(f"Temporary file {tmp_file} exists but is old. Removing stale tmp file...")
            os.remove(tmp_file)
            print(f"Removed stale temporary file {tmp_file}. Proceeding with GLM...")
    print(f"Creating temporary file {tmp_file} to indicate processing...")
    # create tmp file to indicate processing
    with open(tmp_file, 'w') as f:
        f.write('Processing...\n')
        # close file
    f.close() 

    # check if .feat exists, it means that GLM is done, running or failed
    if os.path.exists(fsl_out):
        print(f"FSL output {fsl_out} already exists.")
        # check how old the file is
        mod_time = os.path.getmtime(fsl_out)
        current_time = time()
        if current_time - mod_time > wait_time: # file exist and it's old. No other process is probably running it
            # check if we want to redo
            if redo_if_exists: # we do want to redo. Remove 
                print(f"Existing FSL output {fsl_out} is older than {wait_time} seconds.")
                print(f"Removing existing output to redo GLM...")
                shutil.rmtree(fsl_out)
                print(f"Removed existing FSL output {fsl_out}.")
            else: # file exist, check if finished, if it didn't, remove the folder and redo
                print(f"Checking for .pe1 file to confirm completion...")
                pe1_file = os.path.join(fsl_out, 'stats', 'pe1.nii.gz')
                if os.path.exists(pe1_file):
                    print(f"GLM already completed for sub-{sub_N:02d}, ses-{session}, run-{run_N:02d}. Skipping...")
                    return
                else:
                    print(f"GLM not completed for sub-{sub_N:02d}, ses-{session}, run-{run_N:02d}. Removing incomplete output...")
                    try:
                        shutil.rmtree(fsl_out)
                    except Exception as e:
                        print(f"Error removing incomplete FSL output {fsl_out}: {e}")
                    print(f"Removed incomplete FSL output {fsl_out}. Proceeding with GLM...")
        else: # .feat exist and it's recent, another process is probably running it
            print(f"Existing FSL output {fsl_out} is recent. Skipping GLM...")
            return
    else: 
        print(f"FSL output {fsl_out} does not exist. Proceeding with GLM...")
    # if we reach here, either .feat doesn't exist or we want to redo
    
    ## Original and target movement file paths
    # original par file "P:\userdata\raulh87\data\EmoB\movement\H-sub-01_ses-01_task-EmoB_run-01.par"
    par_file = os.path.join(
        datafolder, dataset, 'preprocessing', 'H_mcflirt',
        f"{specie}-sub-{sub_N:02d}_ses-{session}_task-{task}_run-{run_N:02d}.par"
    )
    mov_txt = os.path.join(
        datafolder, dataset, 'movement',
        f"{specie}-sub-{sub_N:02d}_ses-{session}_task-{task}_run-{run_N:02d}_fwd.txt"
    )
    print("Calculating framewise displacement...")
    preprocess_functions.fwd(
        par_file, radius_fwd, threshold_fwd, output_file=mov_txt, add_movement_params=False
    )

    # Input preprocessed NIfTI file
    # "P:\userdata\raulh87\data\EmoB\BIDS\H-sub-01\H-sub-01_ses-01_task-EmoB_run-01_bold.nii.gz"
    input_nifti = os.path.join(
        datafolder, dataset, 'BIDS',
        f"{specie}-sub-{sub_N:02d}",
        f"{specie}-sub-{sub_N:02d}_ses-{session}_task-{task}_run-{run_N:02d}_bold.nii.gz"
    )
    # Input structural NIfTI file # os.path.join(bids_folder, f"{sub_ID}_T1.nii.gz"))
    structural_nifti = os.path.join(
        datafolder, dataset, 'BIDS',f"{specie}-sub-{sub_N:02d}",f"{specie}-sub-{sub_N:02d}_T1.nii.gz"
    )
    # slice timing file
    # "P:\userdata\raulh87\data\EmoB\BIDS\H-sub-01\H-sub-01_ses-01_task-EmoB_run-01_bold_slicetiming.txt"
    slice_timing_txt = os.path.join(
        datafolder, dataset, 'BIDS',
        f"{specie}-sub-{sub_N:02d}",
        f"{specie}-sub-{sub_N:02d}_ses-{session}_task-{task}_run-{run_N:02d}_bold_slicetiming.txt"
    )
    
    # Extract TR and volumes
    data = nib.load(input_nifti)
    tr = data.header.get_zooms()[3]
    volumes = data.shape[3]

    # TR, volumes = utils.extract_params(input_nifti)
    # Prepare FSF template replacement dictionary
    #design_in = os.path.join(datafolder, dataset, 'FSL_designs', base_design_file)
    design_out = os.path.join(datafolder, dataset, 'FSL_designs', model + f"{specie}-sub-{sub_N:02d}_ses-{session}_run-{run_N:02d}_tmp.fsf")
    labels = {
        'outputdir': (fsl_out,        'set fmri(outputdir)'),
        'TR':        (tr,             'set fmri(tr)'),
        'volumes':   (volumes,        'set fmri(npts)'),
        'BET':       (1 if specie=='H' else 0, 'set fmri(bet_yn)'),
        'smooth':    (smooth,         'set fmri(smooth)'),
        'input':     (input_nifti,    'set feat_files(1)'),
        'atlas':     (atlas_file,     'set fmri(regstandard)'),
        'movement':  (mov_txt,     'set confoundev_files(1)'),
        'structural': (structural_nifti, 'set highres_files(1)'),
        'slice_timing': (slice_timing_txt, 'set fmri(st_file)')
    }

    # add conditions based on stim_types
    for i, stim in enumerate(stim_types, start=1):
        # "C:\data\EmoB\models\all_types\D-sub-01\ses-01_task-EmoB_run-01\A.txt"
        cond_file = os.path.join(
            datafolder, dataset, 'models', model, f"{specie}-sub-{sub_N:02d}",
            f"ses-{session}_task-{task}_run-{run_N:02d}",
            f"{stim}.txt"
        )
        labels[f'condition{i}'] = (cond_file, f'set fmri(custom{i})')


    # Build replacement dict
    to_fill = {}
    for key, (val, find_str) in labels.items():
        rep = f'set {find_str.split()[1]}("{val}"' if isinstance(val, str) else f'set {find_str.split()[1]} {val}'
        # Actually ensure correct format
        if key in labels.keys():
            rep = f'{find_str} "{val}"'
        else:
            rep = f'{find_str} {val}'
        to_fill[key] = {
            'string_to_find': find_str,
            'string_to_replace': rep
        }
    # Fill FSF and run FSL
    utils.fill_fsf(to_fill, design_template_modified, design_out)
    # Remove existing .feat dir if present. 
    if os.path.exists(fsl_out + '.feat'):
        # remove directory, even if not empty
        shutil.rmtree(fsl_out + '.feat')

    cmd = f'feat {design_out}'

    if os.name != 'nt':
        print(f"Running: {cmd}")
        os.system(cmd)
    else:
        print('testing in Windows')
    
    # remove temporary design file
    # os.remove(design_out)
    print(f"GLM completed for sub-{sub_N:02d}, ses-{session}, run-{run_N:02d}.")
    # remove tmp file
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
        print(f"Removed temporary file {tmp_file}.")

def calculate_beta_maps(datafolder, dataset, model, specie, sub_N, session, run_N, task,
                        stim_types, design_template, atlas_file,
                        smooth,
                        radius_fwd,
                        threshold_fwd,
                        redo_if_exists,
                        overwrite_movement, wait_time=1800):
    '''
    
    Calculate beta maps for given subject, session, run using FSL FEAT
    Parameters
    ----------
    datafolder : str. Path to the data folder
    dataset : str. Dataset name
    model : str. Model name
    specie : str. Species identifier
    sub_N : int. Subject number
    session : str. Session identifier
    run_N : int. Run number
    task : str. Task name
    stim_types : list. List of stimulus types
    design_template : str. Path to the design template file
    atlas_file : str. Path to the atlas file
    smooth : int. Smoothing kernel size
    radius_fwd : float. Radius for framewise displacement calculation
    threshold_fwd : float. Threshold for framewise displacement calculation
    redo_if_exists : bool. If True, redo the calculation even if beta maps exist
    overwrite_movement : bool. If True, overwrite movement files
    wait_time : int. Time to wait between checks for existing beta maps
    
    '''
    # if H, run calculate_beta_mapsH, if not continue
    if specie == 'H':
        calculate_beta_mapsH(datafolder, dataset, model, specie, sub_N, session, run_N, task,
                        stim_types, design_template, atlas_file,
                        smooth,
                        radius_fwd,
                        threshold_fwd,
                        redo_if_exists,
                        wait_time)
        return
        
    session = str(session).zfill(2)
    print(f"sub-{sub_N:02d}, ses-{session}, run-{run_N:02d}...")
    # "P:\userdata\raulh87\data\EmoB\results\GLM\basic\D-sub-01\ses-01_task-EmoB_run-01_(len(stim_types))"
    design_template_modified = (datafolder + os.sep + dataset + os.sep + 
    'results' + os.sep + 'GLM' + os.sep + model + os.sep + specie +
    '-sub-' + str(sub_N).zfill(2) + os.sep +
    f"ses-{session}_task-{task}_run-{run_N:02d}_{len(stim_types)}.fsf"
    )

    
    utils.generate_fsf(len(stim_types), design_template, design_template_modified)
    # initialize labels to replace

    
    # check if beta maps have been already calculated
    beta_map_file = (datafolder + os.sep + dataset + os.sep + 'results' + 
                os.sep + 'GLM' + os.sep + model + os.sep + 
                f"{specie}-sub-{sub_N:02d}" + os.sep + 
                f"ses-{session}_task-{task}_run-{run_N:02d}.feat" + os.sep + "stats" + os.sep + "pe1.nii.gz")
    beta_map_file = (datafolder + os.sep + dataset + os.sep + 'results' + 
                os.sep + 'GLM' + os.sep + model + os.sep + 
                f"{specie}-sub-{sub_N:02d}" + os.sep + 
                f"ses-{session}_task-{task}_run-{run_N:02d}.feat")
    if os.path.exists(beta_map_file) and not redo_if_exists:
        print(f"Beta map file {beta_map_file} already exists. Skipping...")
        return
    else:
        print(f"Beta map file {beta_map_file} does not exist. Proceeding with GLM...")
    
    # Check if preprocessing files are ready
    preprocess_file = (datafolder + os.sep + dataset + os.sep + 'normalized' + os.sep + 
                specie + '-sub-' + str(sub_N).zfill(2) + os.sep + 
                specie + '-sub-' + str(sub_N).zfill(2) + 
                '_ses-' + session +
                '_task-' + task +
                '_run-' + str(run_N).zfill(2) + 
                '.nii.gz')
    if not os.path.exists(preprocess_file):
        print(f"Preprocessing file {preprocess_file} not found. Skipping...")
        return
    else:
        print(f"Preprocessing file {preprocess_file} found. Proceeding with GLM...")

    # Output directory for FSL
    fsl_out = os.path.join(
        datafolder, dataset, 'results', 'GLM', model,
        f"{specie}-sub-{sub_N:02d}",
        f"ses-{session}_task-{task}_run-{run_N:02d}"
    )
    # check if fsl_out.feat exists
    if os.path.exists(fsl_out + '.feat'):
        # check if file is older than wait_time seconds. If it is older, remove it, otherwise return
        mod_time = os.path.getmtime(fsl_out + '.feat')
        current_time = time()
        if current_time - mod_time > wait_time:
            print(f"Existing FSL output {fsl_out + '.feat'} is older than {wait_time} seconds. Removing...")
            shutil.rmtree(fsl_out + '.feat')
        else:
            print(f"Existing FSL output {fsl_out + '.feat'} is recent. Skipping GLM...")
            return
        
    
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
    if overwrite_movement or not os.path.exists(mov_txt):
        if os.path.exists(target_mov):
            print(f"Copying movement file: {base_pre} -> {target_mov}")
        shutil.copyfile(base_pre, target_mov)
        
        print("Calculating framewise displacement...")
        preprocess_functions.fwd(
            base_pre, radius_fwd, threshold_fwd, output_file=mov_txt
        )

    # Input preprocessed NIfTI file
    input_nifti = os.path.join(
        datafolder, dataset, 'normalized',
        f"{specie}-sub-{sub_N:02d}",
        f"{specie}-sub-{sub_N:02d}_ses-{session}_task-{task}_run-{run_N:02d}.nii.gz"
    )
    # Input structural NIfTI file # os.path.join(bids_folder, f"{sub_ID}_T1.nii.gz"))
    structural_nifti = os.path.join(
        datafolder, dataset, 'BIDS',f"{specie}-sub-{sub_N:02d}_T1.nii.gz"
    )
    
    # Extract TR and volumes
    data = nib.load(input_nifti)
    tr = data.header.get_zooms()[3]
    volumes = data.shape[3]

    # TR, volumes = utils.extract_params(input_nifti)
    # Prepare FSF template replacement dictionary
    #design_in = os.path.join(datafolder, dataset, 'FSL_designs', base_design_file)
    design_out = os.path.join(datafolder, dataset, 'FSL_designs', model + f"{specie}-sub-{sub_N:02d}_tmp.fsf")
    labels = {
        'outputdir': (fsl_out,        'set fmri(outputdir)'),
        'TR':        (tr,             'set fmri(tr)'),
        'volumes':   (volumes,        'set fmri(npts)'),
        'BET':       (1 if specie=='H' else 0, 'set fmri(bet_yn)'),
        'smooth':    (smooth,         'set fmri(smooth)'),
        'input':     (input_nifti,    'set feat_files(1)'),
        'atlas':     (atlas_file,     'set fmri(regstandard)'),
        'movement':  (mov_txt,     'set confoundev_files(1)'),
        'structural': (structural_nifti, 'set highres_files(1)'),
        #'condition': (cond_file,      'set fmri(custom1)'),
    }


    # add conditions based on stim_types
    for i, stim in enumerate(stim_types, start=1):
        # "C:\data\EmoB\models\all_types\D-sub-01\ses-01_task-EmoB_run-01\A.txt"
        cond_file = os.path.join(
            datafolder, dataset, 'models', model, f"{specie}-sub-{sub_N:02d}",
            f"ses-{session}_task-{task}_run-{run_N:02d}",
            f"{stim}.txt"
        )
        labels[f'condition{i}'] = (cond_file, f'set fmri(custom{i})')


    # Build replacement dict
    to_fill = {}
    for key, (val, find_str) in labels.items():
        rep = f'set {find_str.split()[1]}("{val}"' if isinstance(val, str) else f'set {find_str.split()[1]} {val}'
        # Actually ensure correct format
        if key in labels.keys():
            rep = f'{find_str} "{val}"'
        else:
            rep = f'{find_str} {val}'
        to_fill[key] = {
            'string_to_find': find_str,
            'string_to_replace': rep
        }
    # Fill FSF and run FSL
    utils.fill_fsf(to_fill, design_template_modified, design_out)
    # Remove existing .feat dir if present. 
    if os.path.exists(fsl_out + '.feat'):
        # remove directory, even if not empty
        shutil.rmtree(fsl_out + '.feat')

    cmd = f'feat {design_out}'
    print(f"Running: {cmd}")
    if os.name != 'nt':
        os.system(cmd)
    # remove temporary design file
    os.remove(design_out)



def calculate_pairwise_similarity_maps2(datafolder, dataset, sub_N, session_and_run_dict,
                          specie, model, stim_types, mask, task, radius, 
                          dis_method, replace_file, mah_fold='stim-wise',
                          sigma=None, shrinkage='ledoitwolf', return_rdm=True,
                          verbose=False, skip_prefile_check=False, categories=None, shuffle_runs=False,
                          model_dict=None):
    '''
    Calculate pairwise similarity maps using searchlight approach.
    - Checks if input files are available.
    - Checks if output files already exist.
    - Calculates pairwise similarity maps using specified method.
    

    Dissimilarity methods:
        - Pearson correlation
        - Kendall correlation
        - Euclidean distance
        - Mahalanobis distance
        - Correlation distance
    Mahalanobis:
        mah_fold: folding strategy for Mahalanobis distance with cross-validation
            Single run (---untested maybe unimplemented---):
                When a single run is available, calculate distance within the same run 
                (creates folds within run and cross-validates).
                Folding strategies:
                    - max-fold: max partitions possible
                    - odd-even: splits data into odd and even trials
            Multiple runs:
                When multiple runs are available, calculate distance across runs 
                (cross-validated across runs).
                Folding strategies:
                    - stim-wise-all-runs: EmoC-only. Computes a separate class-level                                            crossnobis analysis inside each run, using final exemplar IDs
                                            (for example, DogA1--DogA4) as the cross-validation folds.
                    - stim-wise: collapse all stimuli of the same type across runs, use runs as folds. Output is a dsm that compares each possible stimuli type vs each possible stimuli type
                    
    Inputs:
        - datafolder: str. Path to data folder
        - dataset: str. Dataset name
        - sub_N: int. Subject number
        - session_and_run_dict: list of dicts. Each dict contains 'session' and 'run' keys.
            -- eventually it will contain maps where session_and_run_dict[indx]['maps'] will be a dict with keys as stim_types and values as the corresponding beta maps. For now, it is only used to loop over sessions and runs.
            -- session_and_run_dict[2]['maps'][stim] will give the beta map for session_and_run_dict[2] and stimulus stim.
        - specie: str. Species identifier ('H' or 'D')
        - model: str. Model name for beta maps
        - stim_types: list of str. List of stimulus types
        - mask: np.ndarray. Mask for searchlight. This marks the voxels to include in the analysis.
        - task: str. Task name
        - radius: int. Searchlight radius in voxels
        - dis_method: str. Dissimilarity method ('pearson', 'kendall', 'euclidean', 'mahalanobis', 'correlation')
        - replace_file: bool. If True, replace existing output files. If False, skip calculation if output files exist.
        - mah_fold: str. Folding strategy for Mahalanobis distance with cross-validation. Options:
            - 'max-fold': max partitions possible within run (for single run)
            - 'odd-even': splits data into odd and even trials (for single run)
            - 'run-wise': each run is a fold (deprecated, do not use)
            - 'stim-wise': folds based on stimulus categories (for multiple runs)
            - run-wise-multiple-runs: checking
                        - 'stim-wise-multiple-folds': EmoC-only exact-stimulus folding. Uses
                            config ``stim_file`` labels and their repeated ``partition`` values.
        - sigma: None or array (n_features, n_features), optional. Noise covariance for Mahalanobis. If None, it is estimated from pooled run-residuals with shrinkage.
        - shrinkage: {'ledoitwolf','oas','ridge','identity'}, optional.
        - return_rdm: bool, optional. If True return a (n_conditions x n_conditions) symmetric RDM. If False return the condensed upper-triangular vector.
        - verbose: bool. If True, print detailed logs of the process.
        - skip_prefile_check: bool. If True, skip the check for input beta files and proceed to calculation.

        
    '''
    # mah_fold == 'run-wise-multiple-runs':
    # if mah_fold is stim-wise, get categories
    # stim-wise or 'run-wise-multiple-runs'
    if mah_fold == 'stim-wise':
        if categories is None:
            categories = set()
            if dataset == 'EmoB':
                for stim in stim_types:
                    if '-' in stim:
                        category = stim.split('-')[0]
                        categories.add(category)
            elif dataset == 'EmoC':
                for stim in stim_types:
                    # remove the last character
                    category = stim[:-1]
                    categories.add(category)
    elif mah_fold == 'stim-wise-multiple-folds':
        if dataset == 'EmoC':
            categories, _ = _get_emoc_multiple_fold_stim_files(session_and_run_dict, model_dict)
        else:
            raise ValueError(f"mah_fold option 'stim-wise-multiple-folds' is only implemented for dataset 'EmoC'. For dataset 'EmoB', use 'stim-wise'.")
    elif mah_fold == 'stim-wise-all-runs':
        if dataset != 'EmoC':
            raise ValueError(
                "mah_fold option 'stim-wise-all-runs' is only implemented for dataset 'EmoC'."
            )
        categories = None

    # if none, ignore
    elif mah_fold is None:
        pass
    
    else:
        raise ValueError(f"Invalid mah_fold option: {mah_fold}. Tested on 'stim-wise'.")

    print(f"Calculating pairwise similarity maps for {specie}-sub-{sub_N:02d} using dissimilarity method: {dis_method} ")
    
    if skip_prefile_check:
        print("Skipping check for existing output files as skip_prefile_check is True.")
        all_exist = False  # force calculation
    else: # check if output files already exist (only for non-Mahalanobis methods, as Mahalanobis with stim-wise folding has a different file structure)
        # if method is not mahalanobis
        all_exist = check_existing_similarity_maps(datafolder, dataset, session_and_run_dict, specie, sub_N, model, task, radius, dis_method, stim_types, categories=categories, mah_fold=mah_fold, model_dict=model_dict, verbose=verbose)


    
    ## check if there is any output file missing
    if all_exist:
        print(f"All output files exist for sub-{sub_N:02d}, ({len(session_and_run_dict)}).")
        if not replace_file:
            print("Skipping calculation as replace_file is False.")
            return None  # no missing files
        else:
            print("Will replace existing files as replace_file is True.")
    else:
        print("Some output files are missing. Proceeding to calculation.")
    
    print("Checking for input beta maps...")
    missing_files_to_process = 0 # keep track of the output files that cannot be calculated due to missing input files
    missing_run_dict = {}  # keep track of missing files for session/run
    ## check if input beta files are available
    missing = {}
    if skip_prefile_check:
        print("Skipping check for input beta files as skip_prefile_check is True.")
    else:
        missing_files_to_process = 0 # keep track of the output files that cannot be calculated due to missing input files
        missing_run_dict = {}  # keep track of missing files for session/run
        ## check if input beta files are available
        missing = {}
        # iterate over session_and_run_dict
        for entry in session_and_run_dict:
            session = entry['session']
            run_N = entry['run_N']
            # correct session to 2 digits
            session = f"{session:02d}"
            pair_check = 0 # keep track of pairs that can be calculated
            # add an entry for this session/run in missing_run_dict
            missing_run_dict[(session, run_N)] = []

            missing_flag = False  # flag to indicate if any input file is missing for this session/run
            # iterate over stim_types
            for i, stim_i in enumerate(stim_types):
                for j, stim_j in enumerate(stim_types):
                    if i >= j:
                        continue  # avoid duplicates and self-comparison
                    input_file_i = os.path.join(
                        datafolder, dataset, 'results', 'GLM', model,
                        f"{specie}-sub-{sub_N:02d}",
                        f"ses-{session}_task-{task}_run-{run_N:02d}.feat",
                        'stats', f'pe{(i+1)*2 - 1}.nii.gz'
                    )
                    input_file_j = os.path.join(
                        datafolder, dataset, 'results', 'GLM', model,
                        f"{specie}-sub-{sub_N:02d}",
                        f"ses-{session}_task-{task}_run-{run_N:02d}.feat",
                        'stats', f'pe{(j+1)*2 - 1}.nii.gz'
                    )
                    if not os.path.exists(input_file_i):
                        if verbose:
                            print(f"Missing input file: {input_file_i}")
                        missing_key = (session, run_N, stim_i)
                        missing[missing_key] = input_file_i
                        # flag that this session/run has missing files
                        missing_flag = True
                    else:
                        if verbose:
                            print(f"Found input file: {input_file_i}")
                        pair_check += 1
                    if not os.path.exists(input_file_j):
                        if verbose:
                            print(f"Missing input file: {input_file_j}")
                        missing_key = (session, run_N, stim_j)
                        missing[missing_key] = input_file_j
                        # flag that this session/run has missing files
                        missing_flag = True
                    else:
                        if verbose:
                            print(f"Found input file: {input_file_j}")
                        pair_check += 1
                # if missing_flag is True, add this session/run to missing_run_dict
            if missing_flag:
                missing_run_dict[(session, run_N)].append('missing input files')
                missing_files_to_process += 1
    
    ## All input files are available, proceed to calculate pairwise similarity maps
    # if dis_method is 'mahalanobis', run mahalanobis for all runs, otherwise run 
    # calculate_pairwise_similarity_maps for each session/run
    if dis_method == 'mahalanobis':
        if len(missing) > 0:
            print(f"Missing input files for Mahalanobis for sub-{sub_N:02d}:")
            for key, file in missing.items():
                session, run_N, stim = key
                print(f"Session: {session}, Run: {run_N}, Stimulus: {stim} -> {file}")
            # return missing files
            return missing
            
        print("All input files are available.")
        print("Calculating Mahalanobis pairwise similarity maps...")
        calculate_mahalanobis_pairwise_maps(datafolder, dataset, sub_N, session_and_run_dict,
                          specie, model, stim_types, mask, task, radius=radius, replace_file=False,
                          mah_fold=mah_fold, sigma=sigma,
                          shrinkage=shrinkage, return_rdm=return_rdm, verbose=verbose, save_inverted=False,
                          model_dict=model_dict)
    else:
        if len(missing) > 0:
            print(f"Missing input files for {dis_method} for sub-{sub_N:02d}:")
            for key, file in missing.items():
                session, run_N, stim = key
                if verbose:
                    print(f"Session: {session}, Run: {run_N}, Stimulus: {stim} -> {file}")
        # check if there are at least one session/run with all input files available
        # if len(missing_run_dict) == len(session_and_run_dict):
        #     # if all runs are missing, we can't calculate anything
        #     print(f"All runs are missing for sub-{sub_N:02d}, skipping...")
        #     return missing
        print(f"Calculating {dis_method} pairwise similarity maps...")
        if shuffle_runs:
            print("Shuffling runs...")
            np.random.shuffle(session_and_run_dict)

        for entry in session_and_run_dict:
            session = entry['session']
            run_N = entry['run_N']
            # correct session to 2 digits
            session = f"{session:02d}"
            # check if this session/run has missing files
            # if (session, run_N) in missing_run_dict:
            #     print(f"Skipping session {session}, run {run_N} due to missing input files.")
            #     continue
            # If we reach this point, it means all input files are available
            calculate_pairwise_similarity_maps(datafolder, dataset, sub_N, session, 
                                       run_N, specie, model, stim_types, mask, 
                                       task, radius=radius, dis_method=dis_method, 
                                       replace_file=replace_file, verbose=verbose)
    return None  # no missing files


def _get_emoc_within_run_class_folds(run_dict, stim_types):
    """Return class labels and exemplar folds for one EmoC run."""
    if not isinstance(run_dict, dict):
        raise ValueError("EmoC within-run class folding requires a run metadata dictionary.")

    stim_indices = {stim: index for index, stim in enumerate(stim_types)}
    class_members = {}
    records = []
    for stim in run_dict:
        if stim not in stim_indices:
            raise ValueError(
                f"Run metadata condition {stim!r} is not present in the GLM stimulus list."
            )

        suffix_start = len(stim)
        while suffix_start > 0 and stim[suffix_start - 1].isdigit():
            suffix_start -= 1
        if suffix_start == 0 or suffix_start == len(stim):
            raise ValueError(
                f"EmoC condition {stim!r} must end with its exemplar number "
                "for within-run class folding."
            )

        class_label = stim[:suffix_start]
        partition = int(stim[suffix_start:])
        class_members.setdefault(class_label, {})
        if partition in class_members[class_label]:
            raise ValueError(
                f"Class {class_label!r} has multiple conditions for exemplar fold {partition}."
            )
        class_members[class_label][partition] = stim
        records.append((stim, class_label, partition))

    class_labels = list(class_members)
    if len(class_labels) < 2:
        raise ValueError("Within-run class folding requires at least two stimulus classes.")

    common_partitions = set.intersection(
        *(set(members) for members in class_members.values())
    )
    if len(common_partitions) < 2:
        raise ValueError(
            "Within-run class folding requires at least two shared exemplar folds "
            "for every stimulus class."
        )

    return class_labels, [
        record for record in records if record[2] in common_partitions
    ], stim_indices


def _calculate_emoc_within_run_class_maps(datafolder, dataset, sub_N,
                                           session_and_run_dict, specie, model,
                                           stim_types, mask, task, radius,
                                           sigma=None, shrinkage='ledoitwolf',
                                           return_rdm=True, verbose=False,
                                           save_inverted=False, model_dict=None):
    """Calculate per-run EmoC class maps from within-run exemplar folds."""
    if dataset != 'EmoC':
        raise ValueError("mah_fold option 'stim-wise-all-runs' is only implemented for EmoC.")
    if not isinstance(model_dict, dict):
        raise ValueError("EmoC within-run class folding requires config['model_dict'].")

    mask_img_obj = nib.load(mask)

    for entry in session_and_run_dict:
        session = f"{entry['session']:02d}"
        run_N = entry['run_N']
        run_key = f"run{int(run_N):02d}"
        run_dict = model_dict.get(run_key)
        if not isinstance(run_dict, dict):
            raise ValueError(f"model_dict is missing metadata for {run_key}.")

        categories, condition_folds, stim_indices = _get_emoc_within_run_class_folds(
            run_dict, stim_types
        )
        if verbose:
            print(
                f"Loading within-run class folds for sub-{sub_N:02d}, "
                f"ses-{session}, run-{run_N:02d}: {categories}"
            )

        run_mask_img_obj = mask_img_obj
        run_mask_img = run_mask_img_obj.get_fdata().astype(bool)
        maps = {}
        affine = None
        for stim, _, _ in condition_folds:
            input_file = os.path.join(
                datafolder, dataset, 'results', 'GLM', model,
                f"{specie}-sub-{sub_N:02d}",
                f"ses-{session}_task-{task}_run-{run_N:02d}.feat",
                'stats', f"pe{stim_indices[stim] * 2 + 1}.nii.gz"
            )
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Beta map file not found: {input_file}")

            map_img_obj = nib.load(input_file)
            map_data = map_img_obj.get_fdata()
            if map_data.shape != run_mask_img.shape:
                run_mask_img_obj = resample_to_img(
                    source_img=run_mask_img_obj,
                    target_img=map_img_obj,
                    interpolation="nearest"
                )
                run_mask_img = run_mask_img_obj.get_fdata().astype(bool)
                print(
                    f"Warning: Beta map shape {map_data.shape} does not match mask shape "
                    f"{run_mask_img.shape}. Resampled mask to match beta map."
                )
            maps[stim] = map_data
            affine = map_img_obj.affine

        ndim = run_mask_img.ndim
        searchlight_radius = float(radius)
        radius_floor = int(np.floor(searchlight_radius))
        ranges = [np.arange(-radius_floor, radius_floor + 1) for _ in range(ndim)]
        grid = np.stack(
            np.meshgrid(*ranges, indexing='ij'), axis=-1
        ).reshape(-1, ndim)
        offsets = grid[(grid.astype(float) ** 2).sum(axis=1) <= searchlight_radius ** 2 + 1e-12]

        pairs = [
            (cat1, cat2)
            for index, cat1 in enumerate(categories)
            for cat2 in categories[index + 1:]
        ]
        similarity_maps = {
            pair: np.full(run_mask_img.shape, np.nan, dtype=float)
            for pair in pairs
        }
        labels = np.array([class_label for _, class_label, _ in condition_folds])
        partitions = np.array([partition for _, _, partition in condition_folds])
        condition_indices = {
            condition: index for index, condition in enumerate(np.unique(labels))
        }

        print(
            f"Calculating within-run class crossnobis for sub-{sub_N:02d}, "
            f"ses-{session}, run-{run_N:02d}..."
        )
        for center in np.argwhere(run_mask_img):
            neighbors = center + offsets
            in_bounds = np.all(
                (neighbors >= 0) & (neighbors < np.array(run_mask_img.shape)), axis=1
            )
            neighbors = neighbors[in_bounds]
            if neighbors.size == 0:
                continue
            neighbors = neighbors[run_mask_img[tuple(neighbors.T)]]
            if neighbors.size == 0:
                continue

            indices = tuple(neighbors.T)
            Y = np.vstack([maps[stim][indices] for stim, _, _ in condition_folds])
            D = crossnobis(
                Y, labels, partitions, sigma=sigma, shrinkage=shrinkage,
                return_rdm=return_rdm
            )
            for cat1, cat2 in pairs:
                similarity_maps[(cat1, cat2)][tuple(center)] = D[
                    condition_indices[cat1], condition_indices[cat2]
                ]

        output_dir = os.path.join(
            datafolder, dataset, 'results', 'RSA', model,
            f"{specie}-sub-{sub_N:02d}",
            f"ses-{session}_task-{task}_run-{run_N:02d}"
        )
        os.makedirs(output_dir, exist_ok=True)
        for cat1, cat2 in pairs:
            output_file = os.path.join(
                output_dir, f"r-{radius}_mahalanobis_{cat1}_{cat2}.nii.gz"
            )
            sim_map_nifti = nib.Nifti1Image(similarity_maps[(cat1, cat2)], affine)
            nib.save(sim_map_nifti, output_file)
            print(f"Saved similarity map: {output_file}")
            if save_inverted:
                inverted_file = os.path.join(
                    output_dir, f"r-{radius}_mahalanobis_{cat2}_{cat1}.nii.gz"
                )
                nib.save(sim_map_nifti, inverted_file)

    return True


def calculate_mahalanobis_pairwise_maps(datafolder, dataset, sub_N, session_and_run_dict,
                          specie, model, stim_types, mask, task, radius, replace_file=False,
                          mah_fold='stim-wise', sigma=None,
                          shrinkage='ledoitwolf', return_rdm=True, verbose=False, save_inverted=False, model_dict=None):
    '''
    Calculate Mahalanobis distance maps using searchlight approach.
    - Uses cross-validation based on specified folding strategy.
    - Supports single-run and multi-run scenarios.
    
    mask : np.ndarray (bool)
        Boolean array of same shape; computation is performed only where mask==True.
    radius : int or float
        Searchlight radius in voxel units (isotropic).

    Folding strategies:
        Single run:
            - max-fold: max partitions possible
            - odd-even: splits data into odd and even trials
        Multiple runs:
            Each run is a fold:
            - run-wise: labels are all stimuli types on one run, across runs. 
                Output: single map for each pair of stimuli types listed in stim_types 
            - stim-wise: labels are stimulus types (category instead of specific stimulus)
                        - stim-wise-multiple-folds: EmoC-only. Uses exact ``stim_file`` IDs
                            as labels and config ``partition`` values as cross-validation folds.
                            One direct-subject map is created for each repeatable stimulus pair.
            - stim-wise-all-runs: EmoC-only. Computes each run independently,
                collapsing exemplar keys such as ``DogA1``--``DogA4`` to ``DogA``
                and using their exemplar IDs as cross-validation folds. Maps are
                written in the corresponding run folder.
            
                
    
    Inputs:
        - datafolder: str. Path to data folder
        - dataset: str. Dataset name
        - sub_N: int. Subject number
        - session_and_run_dict: list of dicts. Each dict contains 'session' and 'run' keys.
        - specie: str. Species identifier ('H' or 'D')
        - model: str. Model name for beta maps
        - stim_types: list of str. List of stimulus types
        - mask: path. Mask for searchlight. This is the search space, the computation is performed only within the mask.
        - task: str. Task name
        - radius: float. Searchlight radius in voxel units
        - replace_file: bool. If True, replace existing output files
        - mah_fold: str. Folding strategy for Mahalanobis distance
        - sigma: np.ndarray or None. Noise covariance matrix
        - shrinkage: str. Shrinkage method for covariance estimation
        - return_rdm: bool. If True, return full RDM matrix
        - save_inverted: bool. If True, save inverted Mahalanobis maps
    Output:
        - similarity_map for each pairwise condition comparison
    '''
    if mah_fold == 'stim-wise-all-runs':
        return _calculate_emoc_within_run_class_maps(
            datafolder=datafolder,
            dataset=dataset,
            sub_N=sub_N,
            session_and_run_dict=session_and_run_dict,
            specie=specie,
            model=model,
            stim_types=stim_types,
            mask=mask,
            task=task,
            radius=radius,
            sigma=sigma,
            shrinkage=shrinkage,
            return_rdm=return_rdm,
            verbose=verbose,
            save_inverted=save_inverted,
            model_dict=model_dict,
        )

    # verbose = True
    # print(verbose)
    # load mask
    mask_img_obj = nib.load(mask)
    mask_img = mask_img_obj.get_fdata().astype(bool)
    # For stim-wise folding, collapse individual stimulus IDs to categories.
    if mah_fold == 'stim-wise':
        # check if each stim in stim_types can be stripped to determine category (e.g. face-1 -> face, body-2 -> body)
        categories = set()
        for stim in stim_types:
            if dataset == 'EmoB':
                if '-' in stim:
                    category = stim.split('-')[0]        
                else:
                    raise ValueError(f"Stimulus type {stim} does not contain a '-' to determine category for stim-wise folding.")
            elif dataset == 'EmoC':
                # remove the last number
                category = stim[:-1]
                categories.add(category)
            else:
                raise ValueError(f"Dataset {dataset} not recognized for determining categories from stimulus types.")
        # if dataset == 'EmoC'
    elif mah_fold == 'stim-wise-multiple-folds':
        if dataset != 'EmoC':
            raise ValueError("mah_fold option 'stim-wise-multiple-folds' is only implemented for EmoC.")
        categories, fold_partitions = _get_emoc_multiple_fold_stim_files(
            session_and_run_dict, model_dict
        )
    else: # if not stim-wise folding, categories are just stim_types
        categories = stim_types
    category_set = set(categories)
    # print categories found
    print(f"Categories for Mahalanobis folding: {categories}")
    #### --- loading beta maps --- ####
    print("Loading beta maps...")

    # Go over every run for the participant and load beta maps to session_and_run_dict
    for indx, entry in enumerate(session_and_run_dict):
        session = entry['session']
        run_N = entry['run_N']
        # correct session to 2 digits
        session = f"{session:02d}"
        if verbose:
            print(f"Loading beta maps for session {session}, run {run_N} for sub-{sub_N:02d}...")

        # initialize maps dict
        session_and_run_dict[indx]['maps'] = {}
        for i, stim in enumerate(stim_types):
            input_file = os.path.join(
                datafolder, dataset, 'results', 'GLM', model,
                f"{specie}-sub-{sub_N:02d}",
                f"ses-{session}_task-{task}_run-{run_N:02d}.feat",
                'stats', f'pe{(i+1)*2 - 1}.nii.gz'
            )
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Beta map file not found: {input_file}")
            # load map
            map_img_obj = nib.load(input_file)
            map_data = map_img_obj.get_fdata()
            # make sure map_data has same shape as mask
            if map_data.shape != mask_img.shape:
                # resample mask to match map (nearest neighbor = IMPORTANT for masks)
                mask_img_obj = resample_to_img(
                    source_img=mask_img_obj,
                    target_img=map_img_obj,
                    interpolation="nearest"
                )
                mask_img = mask_img_obj.get_fdata().astype(bool)
                # issue a warning
                print(f"Warning: Beta map shape {map_data.shape} does not match mask shape {mask_img.shape}. Resampled mask to match beta map.")

                # if map_data.shape != mask_img.shape:
                    # raise ValueError(f"Beta map shape {map_data.shape} does not match mask shape {mask_img.shape}.")

            # store in session_and_run_dict
            session_and_run_dict[indx]['maps'][stim] = map_data
    print("All beta maps loaded successfully.")
    ##### --- end of loading beta maps --- #####

    #### ---- initialize similarity maps ####
    print("Preparing for crossnobis (cross-validated Mahalanobis)...")

    # prepare for searchlight
    ndim = mask_img.ndim # dimensions based on mask
    r = float(radius)
    rad = int(np.floor(r))

    # build integer offsets for an N-D sphere
    ranges = [np.arange(-rad, rad + 1) for _ in range(ndim)]
    grid = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1).reshape(-1, ndim)
    keep = (grid.astype(float) ** 2).sum(axis=1) <= r * r + 1e-12
    offsets = grid[keep]
    # initialize similarity_maps
    similarity_maps = {}
    if mah_fold == 'run-wise':
        # initialize similarity_maps based on stim_types (one map for each stim pair )
        for i, stim_i in enumerate(stim_types):
            for j, stim_j in enumerate(stim_types):
                if i >= j:
                    continue  # avoid duplicates and self-comparison
                key = (stim_i, stim_j)
                similarity_maps[key] = np.full(mask_img.shape, np.nan, dtype=float)
    # elif mah_fold == stim-wise or 'stim-wise-multiple-folds'
    elif mah_fold == 'stim-wise':
        for indx1, cat1 in enumerate(categories):
            for indx2, cat2 in enumerate(categories):
                if indx1 >= indx2:
                    continue  # avoid duplicates and self-comparison
                key = (cat1, cat2)
                similarity_maps[key] = np.full(mask_img.shape, np.nan, dtype=float)
    elif mah_fold == 'stim-wise-multiple-folds': 
        # build similarity maps
        for indx1, cat1 in enumerate(categories):
            for indx2, cat2 in enumerate(categories):
                if indx1 >= indx2:
                    continue  # avoid duplicates and self-comparison
                key = (cat1, cat2)
                similarity_maps[key] = np.full(mask_img.shape, np.nan, dtype=float)

    # iterate over all voxels in mask
    it = np.argwhere(mask_img)
    for center in it: # iterate over each sphere center
        # get neighboring voxels within radius
        neigh = center + offsets
        # in-bounds
        inb = np.all((neigh >= 0) & (neigh < np.array(mask_img.shape)), axis=1)
        neigh = neigh[inb]
        if neigh.size == 0:
            continue
        # apply mask within sphere
        msub = mask_img[tuple(neigh.T)]
        if not np.any(msub):
            continue
        neigh = neigh[msub]
        # get indices of valid voxels
        indices = tuple(neigh.T)
        
        # assign labels and partitions based on mah_fold
        # if len(session_and_run_dict) > 1: # multiple runs available, runs are partitions
        if mah_fold == 'run-wise': # labels are all stimuli types on one run, across runs. Output: single map for each pair of stimuli types, where distance is calculated across runs (cross-validated across runs)
            # prepare data matrix Y (stim_types x voxels)
            Y_list, labels, partitions = [], [], []
            for run_idx, entry in enumerate(session_and_run_dict):
                run_N = entry['run_N']
                for stim_idx, stim in enumerate(stim_types):
                    map_data = entry['maps'][stim]
                    voxel_values = map_data[indices]
                    Y_list.append(voxel_values)
                    labels.append(stim)
                    partitions.append(run_N)
            Y = np.vstack(Y_list)
        # elif stim-wise or stim-wise-multiple-folds
        elif mah_fold == 'stim-wise': 
            # prepare data matrix Y (stim_types x voxels)
            Y_list, labels, partitions = [], [], []
            for run_idx, entry in enumerate(session_and_run_dict):
                run_N = entry['run_N']
                for stim_idx, stim in enumerate(stim_types):
                    map_data = entry['maps'][stim]
                    voxel_values = map_data[indices]
                    Y_list.append(voxel_values)
                    # determine category by stripping stim string
                    if dataset == 'EmoB':
                        if '-' in stim:
                            category = stim.split('-')[0]
                        else:
                            raise ValueError(f"Stimulus type {stim} does not contain a '-' to determine category for stim-wise folding.")                            
                    elif dataset == 'EmoC':
                        # remove the last number
                        category = stim[:-1]
                    else:
                        raise ValueError(f"Dataset {dataset} not recognized for determining categories from stimulus types.")
                    labels.append(category)
                    partitions.append(run_N)  # still use runs as partitions for cross-validation
            Y = np.vstack(Y_list)
        elif mah_fold == 'stim-wise-multiple-folds':
            # prepare data matrix Y (stim_types x voxels)
            Y_list, labels, partitions = [], [], []
            # print contents of session_and_run_dict
            # print(f"session_and_run_dict={session_and_run_dict}")
            for run_idx, entry in enumerate(session_and_run_dict):
                run_dict = model_dict[f"run{entry['run_N']:02d}"]
                # stim_file identifies the repeated stimulus; partition identifies
                # its independent presentation fold across runs.
                for stim, metadata in run_dict.items():
                    stim_file = metadata['stim_file']
                    partition = metadata['partition']
                    if stim_file not in category_set or partition not in fold_partitions:
                        continue
                    map_data = entry['maps'][stim]
                    voxel_values = map_data[indices]
                    Y_list.append(voxel_values)
                    labels.append(stim_file)
                    partitions.append(partition)  # use stimulus repetition as partition for cross-validation
                    # print(f"partition: {partition}, label: {stim_file}")
                Y = np.vstack(Y_list)
    
        else:
            raise ValueError(f"Unknown mah_fold strategy for multiple runs: {mah_fold}")
        # else:
        #     # single run available
        #     raise NotImplementedError("Single run Mahalanobis calculation not implemented yet.")
        if verbose:
            # do anything
            a = 1
            #print(f"Calculating crossnobis at voxel {tuple(center)} with {Y.shape[0]} observations and {Y.shape[1]} features.")
        D = crossnobis(Y, labels, partitions, sigma=sigma, shrinkage=shrinkage, return_rdm=return_rdm)
        if mah_fold == 'run-wise':
            # store distances in similarity_maps
            for i, stim_i in enumerate(stim_types):
                for j, stim_j in enumerate(stim_types):
                    if i >= j:
                        continue  # avoid duplicates and self-comparison
                    key = (stim_i, stim_j)
                    similarity_maps[key][tuple(neigh.T)] = D[i, j]
        elif mah_fold == 'stim-wise' or mah_fold == 'stim-wise-multiple-folds':
            # output for both is a similarity map for each pair of categories, in the case of stim-wise, a single map represents all runs, in the case of stim-wise-multiple-folds, a single map represents a single run, but the distance is calculated based on the stimulus repetitions within that run
            # store distances in similarity_maps category-wise
            condition_indices = {condition: index for index, condition in enumerate(np.unique(labels))}
            for indx1, cat1 in enumerate(categories):
                for indx2, cat2 in enumerate(categories):
                    if indx1 >= indx2:
                        continue  # avoid duplicates and self-comparison
                    key = (cat1, cat2)
                    similarity_maps[key][tuple(neigh.T)] = D[
                        condition_indices[cat1], condition_indices[cat2]
                    ]
    print("Searchlight calculation completed.")
    print("Saving similarity maps...")
    
    ## Save similarity maps
    
    
    
    # save similarity maps for each pair of stimulus types
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i >= j:
                continue  # avoid duplicates and self-comparison
            if verbose:
                print(f"Saving similarity map for {cat1} vs {cat2}...")
            key = (cat1, cat2)
            # build output path
            # if mah_fold is stim-wise, save one map for each pair of categories
            if mah_fold == 'stim-wise':
                output_file = os.path.join(
                    datafolder, dataset, 'results', 'RSA', model,
                    f"{specie}-sub-{sub_N:02d}",
                    f"r-{radius}_mahalanobis_{cat1}_{cat2}.nii.gz"
                )
            elif mah_fold == 'stim-wise-multiple-folds':
                output_file = os.path.join(
                    datafolder, dataset, 'results', 'RSA', model,
                    f"{specie}-sub-{sub_N:02d}",
                    f"r-{radius}_mahalanobis_{cat1}_{cat2}.nii.gz"
                )
            
            
            # create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            # save NIfTI
            affine = nib.load(input_file).affine  # use affine from last loaded map
            sim_map_nifti = nib.Nifti1Image(similarity_maps[key], affine)
            nib.save(sim_map_nifti, output_file)
            print(f"Saved similarity map: {output_file}")
            if save_inverted:
                # get same folder but change file name to reflect inverted map
                output_fileb = output_file.replace(f"mahalanobis_{cat1}_{cat2}", f"mahalanobis_{cat2}_{cat1}")

                nib.save(sim_map_nifti, output_fileb)
      

def calculate_pairwise_similarity_maps(datafolder, dataset, sub_N, session, 
                                       run_N, specie, model, stim_types, mask, 
                                       task, radius=3, dis_method='correlation', 
                                       replace_file=False, verbose=False):
    '''
    Calculate pairwise similarity maps between all stimulus types for a given subject/session/run.
    Parameters
    ----------
    datafolder : str. Base directory for data storage.
    dataset : str. Dataset name.
    sub_N : int. Subject number.
    session : str. Session identifier.
    run_N : int. Run number.
    specie : str. Species identifier.
    model : str. Model name.
    stim_types : list. List of stimulus type names.
    mask : str. Path to the brain mask NIfTI file.
    task : str. Task name.
    radius : int. Radius for searchlight (default: 3).
    dis_method : str. Method for pairwise similarity calculation (default: 'correlation').
    replace_file : bool. Overwrite existing output files (default: False).
    verbose : bool. Verbose output (default: False).
    

    Returns

    '''

    log = []
    log.append("calculate_pairwise_similarity_maps called with parameters:")
    log.append(f"datafolder: {datafolder}")
    log.append(f"dataset: {dataset}")
    log.append(f"sub_N: {sub_N}")
    log.append(f"session: {session}")
    log.append(f"run_N: {run_N}")
    log.append(f"specie: {specie}")
    log.append(f"model: {model}")
    log.append(f"mask: {mask}")
    for stim_N1, stim_1_name in enumerate(stim_types):
        blank = False  # flag to indicate if either map is blank
        # stim_1_name = stim_types[stim_N1]
        file_1_path = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'GLM' + os.sep + model + os.sep + f"{specie}-sub-{sub_N:02d}" + os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}.feat" + os.sep + 'stats' + os.sep + f"pe{(stim_N1+1)*2 - 1}.nii.gz"
        file_1_map = nib.load(file_1_path).get_fdata()
        # check if file_1_map is empty (all zeros)
        if np.all(file_1_map == 0):
            blank = True # all next maps will be blank
        
        
        # add to log
        log.append(f"Processing {dis_method} similarity for stim {stim_1_name}, file: {file_1_path}")
        
        for stim_N2, stim_2_name in enumerate(stim_types):
            # if stim_N1 == stim_N2, skip
            if stim_N1 >= stim_N2:
                # print("Skipping identical stimuli...")
                continue                
            file_2_path = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'GLM' + os.sep + model + os.sep + f"{specie}-sub-{sub_N:02d}" + os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}.feat" + os.sep + 'stats' + os.sep + f"pe{(stim_N2+1)*2 - 1}.nii.gz"
            log.append(f"Comparing with stim {stim_N2}, file: {file_2_path}")
            # add to log
            # build output path
            output_path = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep + model + os.sep + f"{specie}-sub-{sub_N:02d}" + os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}" + os.sep + f"r-{radius}_{dis_method}_{stim_1_name}_{stim_2_name}.nii.gz"
            # same but the order of the stims is inverted
            output_pathb = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep + model + os.sep + f"{specie}-sub-{sub_N:02d}" + os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}" + os.sep + f"r-{radius}_{dis_method}_{stim_2_name}_{stim_1_name}.nii.gz"
            # check if file_2_path exists
            file_available = os.path.exists(output_path)

            # check if output_path exists
            if file_available and not replace_file:
                if verbose:
                    print(f"Output exists: {output_path}. Skipping...")
                continue
            # check that file 2 exists
            if not os.path.exists(file_2_path):
                if verbose:
                    print(f"File {file_2_path} does not exist. Skipping...")
                    log.append(f"File {file_2_path} does not exist. Skipping...")
                continue
            
            file_2_map = nib.load(file_2_path).get_fdata()
            # check if either map is blank
            if blank or np.all(file_2_map == 0):
                if verbose:
                    print(f"One of the maps is blank (all zeros). Creating blank similarity map for {output_path}...")
                log.append(f"One of the maps is blank (all zeros). Creating blank similarity map for {output_path}...")
                # create blank similarity map
                similarity_map = np.zeros(nib.load(mask).get_fdata().shape)
            else:
                similarity_map = similarity_searchlight(file_1_map, file_2_map, nib.load(mask).get_fdata().astype(bool), radius=radius, dis_method=dis_method)
            # add to log
            log.append(f"Saved similarity map to {output_path}")
            # check if output directory exists
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            nib.save(nib.Nifti1Image(similarity_map, nib.load(file_1_path).affine), output_path)
            # save the version with the stims inverted
            nib.save(nib.Nifti1Image(similarity_map, nib.load(file_1_path).affine), output_pathb)
            print(f"Saved {dis_method} similarity map to {output_path}")
    if verbose:
        # print that we are done
        print(f"sub {sub_N:02d} session {session} run {run_N:02d} pairwise similarity computation done.")

    # save log
    output_folder = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 
                     'RSA' + os.sep + model + os.sep + f"{specie}-sub-{sub_N:02d}" + 
                     os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}")
    log_path = output_folder + os.sep + f"r-{radius}_{dis_method}_log.txt"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    with open(log_path, 'w') as f:
        for line in log:
            f.write(line + '\n')


def calculate_group_model_similarity_map(datafolder, dataset, session_and_run_all_dict, specie, model,
                                          task, radius, rsa_model, rsa_method,
                                          dis_method, replace_file, min_percentage_available=1.0, verbose=False,
                                          mask_type=None, mah_fold='stim-wise'):
    '''Calculate the group model similarity map.
    Inputs:
        datafolder: str. Path to data folder.
        dataset: str. Dataset name.
        specie: str. 'D' or 'H'.
        model: str. GLM model name.
        mask_img: numpy array. Mask image data. Area to compute similarity.
        session_and_run_all_dict: dict. Dictionary with session and run information for all participants.
        stim_types: list of str. Stimulus types.
        task: str. Task name.
        radius: int. Searchlight radius.
        rsa_model: str. Path to RSA model file.
        dis_method: str. Similarity method.
        replace_file: bool. Whether to replace existing files.
        verbose: bool. Whether to print verbose output.
        min_percentage_available: float. Minimum percentage of available data to process.
        mask_type: str. Type of brain mask to use.
    outputs:
        mean_model_map_path: str. Path to mean model similarity map.
        std_model_map_path: str. Path to standard deviation model similarity map.
        log: log file.
        log_json: json file with details about the processing, same name as output file but with .json extension.
    
        Creates a 
    '''
    participants = list(session_and_run_all_dict.keys())

    print("Calculating group model similarity map...")
    log = [] # log messages
    # initialize log_json
    log_json = {
        'datafolder': datafolder,
        'dataset': dataset,
        'specie': specie,
        'model': model,
        'mask_type': mask_type,
        'task': task,
        'radius': radius,
        'rsa_model': rsa_model,
        'dis_method': dis_method,
        'mah_fold': mah_fold,
        'replace_file': replace_file,
        'min_percentage_available': min_percentage_available,
        'participants': participants,
        'file_list': [],
        'perc_available': 0.0,
        'output_mean_file': '',
        'output_std_file': '',
        'notes': []
    }
    
    print("Checking for existing output files...")
    if mask_type is None:
        # check if output file already exists
        output = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                        f"{specie}-r-{radius}_{dis_method}_{rsa_method}_mean.nii.gz")
    else:
        output = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                        f"{mask_type}-{specie}-r-{radius}_{dis_method}_{rsa_method}_mean.nii.gz")
    # same name but with .json extension
    log_json_output = output.replace('.nii.gz', '.json')
    
    

    # check if log_json output file already exists
    if os.path.exists(log_json_output):
        # load existing log_json
        with open(log_json_output, 'r') as f:
            existing_log_json = yaml.safe_load(f)
        # if perc_available in existing log_json is < min_percentage_available, force recalculation
        if existing_log_json['perc_available'] < min_percentage_available:
            print(f"Existing log file {log_json_output} found with perc_available < min_percentage_available. Forcing recalculation.")
            replace_file = True
    if os.path.exists(output) and not replace_file:
        # add log message
        log.append(f"Output file {output} already exists. Skipping group model similarity map calculation.")
        if verbose:
            print(log[-1])
        return
    
    # gather all model similarity maps across participants/sessions/runs
    # if they exist, add to files_list
    print("Gathering model similarity maps across participants/sessions/runs...")
    files_in_database = 0
    files_list = [] # list of files to process
    per_run = dis_method != 'mahalanobis' or mah_fold == 'stim-wise-all-runs'
    for sub_N in participants:
        entries = session_and_run_all_dict[sub_N] if per_run else [None]
        for entry in entries:
            files_in_database += 1
            session = entry['session'] if entry is not None else None
            run_N = entry['run_N'] if entry is not None else None
            model_sim_map_file = _model_similarity_map_file(
                datafolder, dataset, specie, sub_N, model, rsa_model, task,
                radius, dis_method, rsa_method, mask_type=mask_type,
                mah_fold=mah_fold, session=session, run_N=run_N,
            )
            if not os.path.exists(model_sim_map_file):
                log.append(f"Model similarity map {model_sim_map_file} not found. Skipping.")
                if verbose:
                    print(log[-1])
                continue
            else:
                log.append(f"Adding model similarity map {model_sim_map_file} to processing list.")
                if verbose:
                    print(log[-1])
            files_list.append(model_sim_map_file)

    # make sure files_in_database is not zero
    if files_in_database == 0:
        # no files expected? there is an error
        raise ValueError("No files found for group model similarity map calculation.")
    
    # check if enough files are available
    available_percentage = len(files_list) / files_in_database
    if available_percentage < min_percentage_available:
        log.append(f"Only {available_percentage*100:.2f}% of data available. Minimum required is {min_percentage_available*100:.2f}%. Skipping group model similarity map calculation.")
        # if verbose:
        print(log[-1])
        return
    else:
        log.append(f"{available_percentage*100:.2f}% of data available. Proceeding with group model similarity map calculation.")
        if verbose:
            print(log[-1])
    
    # determine mean_model_map_path
    if mask_type is None:
        mean_model_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                        f"{specie}-r-{radius}_{dis_method}_{rsa_method}_mean.nii.gz")
        std_model_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                        f"{specie}-r-{radius}_{dis_method}_{rsa_method}_std.nii.gz")
    else:
        mean_model_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                        f"{mask_type}-{specie}-r-{radius}_{dis_method}_{rsa_method}_mean.nii.gz")
        std_model_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                        f"{mask_type}-{specie}-r-{radius}_{dis_method}_{rsa_method}_std.nii.gz")
    # create output directory if it doesn't exist
    output_dir = os.path.dirname(mean_model_map_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # update log_json
    log_json['file_list'] = files_list
    log_json['perc_available'] = available_percentage
    log_json['output_mean_file'] = mean_model_map_path
    log_json['output_std_file'] = std_model_map_path
    # calculate mean model similarity map by averaging all files in files_list
    # nifti_mean(files_list, mean_model_map_path, std_model_map_path, mask_img=mask_img)
    nifti_mean(files_list, mean_model_map_path, std_model_map_path)
    #mask_img

    print("Done computing mean and std model similarity maps.")
    ## update log_json with output files
    # if log_json already exists, delete and recreate
    if os.path.exists(log_json_output):
        os.remove(log_json_output)

    print(f"Saving log: {log_json_output}")
    with open(log_json_output, 'w') as f:
        yaml.dump(log_json, f)
    return True

def calculate_group_model_similarity_map_rnd(datafolder, dataset, session_and_run_all_dict, specie, model, 
                                            task, radius, rsa_model,
                                            rsa_method='pearson',
                                            dis_method='pearson', verbose=False, 
                                            min_percentage_available=1.0,
                                            reps=1000, replace_rnd_files=False, wait_time=300,reps_group=1000,
                                            mah_fold='stim-wise', mask_type=None
                                            ):
    # print the variable rsa_model
    print(f"rsa_model: {rsa_model}")
    # 
    participants = list(session_and_run_all_dict.keys())
    print("Checking for existing output files...")
    # check that outptut folder exists
    output_folder = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean')
    # create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    rnd_N_list = list(range(0, reps_group))
    # randomly shuffle rnd_N_list
    random.shuffle(rnd_N_list)

    # Pre-scan available permutation files once per run so the main loop can
    # sample from known-good indices without calling os.path.exists each iteration.
    available_rnd_indices = {}  # key: (sub_N, session, run_N) or (sub_N, None, None)
    file_counter_total = 0
    prefix_mask = f"{mask_type}-" if mask_type else ""
    rnd_file_prefix = f"{prefix_mask}r-{radius}_{dis_method}_{rsa_method}_"
    per_run = dis_method != 'mahalanobis' or mah_fold == 'stim-wise-all-runs'
    for sub_N in participants:
        entries = session_and_run_all_dict[sub_N] if per_run else [None]
        for entry in entries:
            file_counter_total += 1
            session = entry['session'] if entry is not None else None
            run_N = entry['run_N'] if entry is not None else None
            key = (sub_N, session, run_N)
            run_rnd_folder = os.path.dirname(_model_similarity_map_file(
                datafolder, dataset, specie, sub_N, model, rsa_model, task,
                radius, dis_method, rsa_method, mask_type=mask_type,
                mah_fold=mah_fold, rnd=True, session=session, run_N=run_N,
                rnd_index=0,
            ))
            if os.path.isdir(run_rnd_folder):
                indices = []
                for fname in os.listdir(run_rnd_folder):
                    if fname.startswith(rnd_file_prefix) and fname.endswith('.nii.gz'):
                        try:
                            idx = int(fname[len(rnd_file_prefix):-7])
                            if 0 <= idx < reps:
                                indices.append(idx)
                        except ValueError:
                            pass
                available_rnd_indices[key] = indices
            else:
                available_rnd_indices[key] = []
    print(f"Pre-scanned permutation files across {file_counter_total} runs.")

    # check if output file already exists
    for indx, rnd_N in enumerate(rnd_N_list):
        # output file path
        mean_model_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                        f"{specie}-r-{radius}_{dis_method}_{rsa_method}_mean_{rnd_N:05d}.nii.gz")
        # temp file path (same but _tmp.txt)
        mean_model_map_path_tmp = mean_model_map_path.replace('.nii.gz', '_tmp.txt')
        if os.path.exists(mean_model_map_path):
            if not replace_rnd_files:
                if verbose: 
                    print(f"rsa_model {rsa_model} sub_N {sub_N} rnd {(indx+1):05d}/{reps_group:05d} exist, skipping...")
                continue
            else:
                if verbose:
                    print(f"rsa_model {rsa_model} sub_N {sub_N} rnd {(indx+1):05d}/{reps_group:05d} exist, replacing...")
        else:
            if verbose:
                print(f"rsa_model {rsa_model} sub_N {sub_N} rnd {(indx+1):05d}/{reps_group:05d}, file {mean_model_map_path} does not exist, calculating...")
        
        # check if temp file exists and how long ago it was modified
        if os.path.exists(mean_model_map_path_tmp):
            mod_time = os.path.getmtime(mean_model_map_path_tmp)
            elapsed_time = time() - mod_time
            if elapsed_time < wait_time:
                if verbose:
                    print(f"rsa_model {rsa_model} sub_N {sub_N} rnd {(indx+1):05d}/{reps_group:05d} skipping as it is being processed by another instance. Skipping...")
                continue
            else:
                if verbose:
                    print(f"rsa_model {rsa_model} sub_N {sub_N} rnd {(indx+1):05d}/{reps_group:05d} old temp found, calculating model similarity map...")
        else:
            if verbose:
                print(f"rsa_model {rsa_model} sub_N {sub_N} rnd {(indx+1):05d}/{reps_group:05d} temporal file created {mean_model_map_path_tmp}...")
            # create temp file
            with open(mean_model_map_path_tmp, 'w') as f:
                f.write(f"Temporary file for rnd {rnd_N:05d}\n")

        
        # build list of available model similarity maps using pre-scanned index cache
        files_list = []
        for sub_N in participants:
            entries = session_and_run_all_dict[sub_N] if per_run else [None]
            for entry in entries:
                session = entry['session'] if entry is not None else None
                run_N = entry['run_N'] if entry is not None else None
                indices = available_rnd_indices.get((sub_N, session, run_N), [])
                if not indices:
                    if verbose:
                        print(
                            f"rsa_model {rsa_model} sub_N {sub_N} rnd "
                            f"{(indx+1):05d}/{reps_group:05d} has no permutation "
                            f"files for session={session}, run={run_N}; skipping."
                        )
                    continue
                rnd_individual_N = random.choice(indices)
                files_list.append(_model_similarity_map_file(
                    datafolder, dataset, specie, sub_N, model, rsa_model, task,
                    radius, dis_method, rsa_method, mask_type=mask_type,
                    mah_fold=mah_fold, rnd=True, session=session, run_N=run_N,
                    rnd_index=rnd_individual_N,
                ))
        # check if enough files are available
        available_percentage = len(files_list) / file_counter_total
        if available_percentage < min_percentage_available:
            print(f"rsa_model {rsa_model} sub_N {sub_N} rnd {(indx+1):05d}/{reps_group:05d} not enough files available ({available_percentage*100:.2f}%). Needed {min_percentage_available*100:.2f}%. Skipping...")
            # remove temp file
            if os.path.exists(mean_model_map_path_tmp):
                os.remove(mean_model_map_path_tmp)
            continue
        else:
            print(f"rsa_model {rsa_model} sub_N {sub_N} rnd {(indx+1):05d}/{reps_group:05d} processing {len(files_list)} files ({available_percentage*100:.2f}% available)...")
        try:
            # calculate group model similarity map
            # nifti_mean(files_list, result_map_path=mean_model_map_path, verbose=False, mask_img=mask_img)
            nifti_mean(files_list, result_map_path=mean_model_map_path, verbose=False)
        except Exception as e:
            print(f"Error {e} calculating group model similarity map for rsa_model {rsa_model} sub_N {sub_N} rnd {rnd_N:05d}. Skipping...")
        # remove temp file
        if os.path.exists(mean_model_map_path_tmp):
            try:
                os.remove(mean_model_map_path_tmp)
            except Exception as e:
                print(f"Error {e} removing temporary file {mean_model_map_path_tmp}.")
            continue
    return True

def calculate_voxelwise_rnd_distribution(datafolder, dataset, specie, model, task, radius,
                                    dis_method='pearson', rsa_method='pearson',
                                    rsa_model='emotion-valence-basic', reps_group=1000,
                                    verbose=False):
    """
    Calculate per voxel distribution. Load all group model similarity maps. Calculate per voxel mean and std across maps. Save as nifti.
    Inputs:
    - datafolder: path to data folder
    - dataset: dataset name
    - specie: 'D' or 'H'
    - model: GLM model name
    - task: task name
    - radius: searchlight radius
    - dis_method: method for pairwise similarity calculation
    - rsa_method: method to compare similarity maps with model
    - rsa_model: model to compare with
    - reps_group: number of repetitions for permutations in group analysis
    - replace_file: whether to overwrite existing output file
    - verbose: whether to print messages
    Outputs:
    - writes mean distribution map
    - writes std distribution map
    """

    # initialize log
    log = [] # log messages

    # check how many rnd mean files are available
    available_files = []
    missing_files = []
    for rnd_N in range(0, reps_group):
        group_mean_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                        f"{specie}-r-{radius}_{dis_method}_{rsa_method}_mean_{rnd_N:05d}.nii.gz")
        if os.path.exists(group_mean_map_path):
            if verbose:
                print(f"added: {group_mean_map_path}")
            available_files.append(group_mean_map_path)
        else:
            if verbose:
                print(f"missing: {group_mean_map_path}")
            missing_files.append(group_mean_map_path)
    print(f"Found {len(available_files)} available rnd mean files.")
    print(f"Missing {len(missing_files)} rnd mean files.")
    # add to log
    log.append(f"Found {len(available_files)} available rnd mean files.")
    log.append(f"Missing {len(missing_files)} rnd mean files.")
    # "P:\userdata\raulh87\data\EmoB\results\RSA_rnd\basic\emotion-valence-basic_mean.nii.gz"
    # calculate mean and std across available files
    distribution_mean_map_path = (datafolder + os.sep + dataset + os.sep + 
                                'results' + os.sep + 'RSA_rnd' + os.sep +
                                model + os.sep + specie + '-' + rsa_model + '_mean.nii.gz')
    distribution_std_map_path = (datafolder + os.sep + dataset + os.sep +
                                'results' + os.sep + 'RSA_rnd' + os.sep +
                                model + os.sep + specie + '-' + rsa_model + '_std.nii.gz')
    # add to log
    log.append(f"Calculating distribution mean map: {distribution_mean_map_path}")
    log.append(f"Calculating distribution std map: {distribution_std_map_path}")
    # print message
    print(f"Calculating distribution mean map: {distribution_mean_map_path}")
    print(f"Calculating distribution std map: {distribution_std_map_path}")

    # calculate distribution mean and std maps
    mean_data, std_data = nifti_mean(available_files, distribution_mean_map_path, distribution_std_map_path, verbose=verbose)
    # save log
    log_path = distribution_mean_map_path.replace('.nii.gz', '_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log))
    return True

def calculate_z_map_real_data(datafolder, dataset, specie, model, radius,
                              dis_method, rsa_method, rsa_model, verbose=False, mask_type=None):

    # paths
    distribution_mean_map_path = os.path.join(datafolder, dataset, 'results', 'RSA_rnd',
                                              model, f'{specie}-{rsa_model}_mean.nii.gz')
    distribution_std_map_path  = os.path.join(datafolder, dataset, 'results', 'RSA_rnd',
                                              model, f'{specie}-{rsa_model}_std.nii.gz')

    if mask_type is None:
        group_mean_map_path = os.path.join(datafolder, dataset, 'results', 'RSA',
                                        model, rsa_model, 'mean',
                                        f'{specie}-r-{radius}_{dis_method}_{rsa_method}_mean.nii.gz')

        group_z_map_path = os.path.join(datafolder, dataset, 'results', 'RSA',
                                        model, rsa_model, 'mean',
                                        f'{specie}-r-{radius}_{dis_method}_{rsa_method}_z.nii.gz')
    else:
        group_mean_map_path = os.path.join(datafolder, dataset, 'results', 'RSA',
                                        model, rsa_model, 'mean',
                                        f'{mask_type}-{specie}-r-{radius}_{dis_method}_{rsa_method}_mean.nii.gz')

        group_z_map_path = os.path.join(datafolder, dataset, 'results', 'RSA',
                                        model, rsa_model, 'mean',
                                        f'{mask_type}-{specie}-r-{radius}_{dis_method}_{rsa_method}_z.nii.gz')

    # load images
    ref_img   = nib.load(group_mean_map_path)         # reference space
    mean_arr  = ref_img.get_fdata()
    dmean_arr = nib.load(distribution_mean_map_path).get_fdata()
    dstd_arr  = nib.load(distribution_std_map_path).get_fdata()

    # sanity checks: same shape + affine
    if mean_arr.shape != dmean_arr.shape or mean_arr.shape != dstd_arr.shape:
        raise ValueError("Array shapes differ; resampling needed before z calculation.")
    if not np.allclose(ref_img.affine, nib.load(distribution_mean_map_path).affine) \
       or not np.allclose(ref_img.affine, nib.load(distribution_std_map_path).affine):
        raise ValueError("Affines differ; resample distribution maps to ref_img before z.")

    # robust z
    with np.errstate(divide='ignore', invalid='ignore'):
        z = (mean_arr - dmean_arr) / dstd_arr
        z[~np.isfinite(z)] = 0  # handles std=0, NaN, inf

    # build output using ref header+affines (and keep codes)
    ref_hdr = ref_img.header.copy()
    z_img = nib.Nifti1Image(z.astype(np.float32), ref_img.affine, header=ref_hdr)

    # explicitly set qform/sform and their codes (belt & suspenders)
    qform, qcode = ref_img.get_qform(), int(ref_img.header['qform_code'])
    sform, scode = ref_img.get_sform(), int(ref_img.header['sform_code'])
    if qform is not None:
        z_img.set_qform(qform, code=qcode if qcode > 0 else 1)
    if sform is not None:
        z_img.set_sform(sform, code=scode if scode > 0 else 1)

    nib.save(z_img, group_z_map_path)
    if verbose:
        print(f"Saved z map to {group_z_map_path}")
    return True


def calculate_z_maps_rnd(datafolder, dataset, specie, model, task, radius,
                                    dis_method='pearson', rsa_method='pearson',
                                    rsa_model='emotion-valence-basic',
                                    verbose=False, reps_group=1000, replace_file=False):

    """
    Calculate z map by comparing group model similarity map with the distribution mean and std maps.
    """
    # initialize log
    log= []
    # load distribution mean and std maps
    distribution_mean_map_path = (datafolder + os.sep + dataset + os.sep + 
                                'results' + os.sep + 'RSA_rnd' + os.sep +
                                model + os.sep + specie + '-' + rsa_model + '_mean.nii.gz')
    distribution_std_map_path = (datafolder + os.sep + dataset + os.sep +
                                'results' + os.sep + 'RSA_rnd' + os.sep +
                                model + os.sep + specie + '-' + rsa_model + '_std.nii.gz')
    # load distribution mean and std maps
    dist_mean_img = nib.load(distribution_mean_map_path).get_fdata()
    dist_std_img = nib.load(distribution_std_map_path).get_fdata()
    # add to log
    log.append(f"Loaded distribution mean map: {distribution_mean_map_path}")
    log.append(f"Loaded distribution std map: {distribution_std_map_path}")

    ## Calculate z map for permutations (RSA_rnd folder)
    # check how many rnd mean files are available
    available_files = []
    missing_files = []
    for rnd_N in range(0, reps_group):
        # clean output and print status
        if verbose:
            print(f"Processing rnd {rnd_N+1}/{reps_group}...")

        group_mean_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                        f"{specie}-r-{radius}_{dis_method}_{rsa_method}_mean_{rnd_N:05d}.nii.gz")
        group_z_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                        f"{specie}-r-{radius}_{dis_method}_{rsa_method}_z_{rnd_N:05d}.nii.gz")
        # check if z map already exists
        if os.path.exists(group_z_map_path) and not replace_file:
            print(f"Z map {group_z_map_path} already exists. Skipping...")
            available_files.append(group_z_map_path)
            continue
        # check if file exists
        if os.path.exists(group_mean_map_path):
            # print
            if verbose:
                print(f"Calculating z map for {group_mean_map_path}...")
            # load file
            group_mean_img = nib.load(group_mean_map_path).get_fdata()
            # calculate z map
            z_map = (group_mean_img - dist_mean_img) / dist_std_img
            # save z map
            nib.save(nib.Nifti1Image(z_map, nib.load(group_mean_map_path).affine), group_z_map_path)
            # add to available files
            available_files.append(group_z_map_path)
        else:
            # add to missing files
            missing_files.append(group_mean_map_path)
            if verbose:
                print(f"Not found, skipping {group_mean_map_path}")

    # add to log
    log.append(f"Calculated z maps for {len(available_files)} available rnd mean files.")
    log.append(f"Missing {len(missing_files)} rnd mean files.")
    # print
    print(f"Calculated z maps for {len(available_files)} available rnd mean files.")
    print(f"Missing {len(missing_files)} rnd mean files.")

    log_path = group_z_map_path.replace(f"_{rnd_N:05d}.nii.gz", '_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log))
    print(f"Saved log to {log_path}")
    return True


import itertools
import random
import math

def shuffle_vector(vector, verbose=False):
    L = len(vector)
    if verbose:
        print("Original vector:")
        print(vector)
    # Solve quadratic n(n-1)/2 = L for n
    n = (1 + math.isqrt(1 + 8*L)) // 2
    if n * (n - 1) // 2 != L:
        raise ValueError("Vector length is not a valid number of pairs.")

    # Original pairs in canonical order (i < j)
    pairs = list(itertools.combinations(range(n), 2))

    # Map old pairs → values
    pair_to_val = dict(zip(pairs, vector))

    # Random permutation of category labels
    perm = list(range(n))
    random.shuffle(perm)

    # Rebuild new pairs with permuted labels
    shuffled_pairs = [(perm[i], perm[j]) for (i, j) in pairs]

    # Ensure canonical ordering (smaller index first)
    shuffled_pairs = [(min(i, j), max(i, j)) for (i, j) in shuffled_pairs]

    # Build new vector in canonical pair order
    vector_rnd = [pair_to_val[p] for p in shuffled_pairs]

    return vector_rnd

def calculate_cluster_size_distribution(
    datafolder, dataset, model, rsa_model, radius, specie,
    dis_method, rsa_method, z_threshold=3.1,
    verbose=False
):
    """
    Calculate cluster size distribution for rnd z maps.
    Inputs:
    - datafolder: path to data folder
    - dataset: dataset name
    - model: beta map model name
    - rsa_model: RSA model name
    - radius: searchlight radius
    - dis_method: similarity method for pairwise similarity calculation
    - rsa_method: RSA comparison method (e.g., 'kendall','pearson', 'spearman')
    - z_threshold: z threshold for cluster definition
    - connectivity: cluster connectivity (1, 2, or 3). Default is 3 (faces).
    - verbose: whether to print messages
    Outputs:
    - writes cluster sizes dictionary to npy file
    """
    connectivity=3 # based on FSL default for 3D images
    # Load cluster sizes
    cluster_sizes_dict_path = (datafolder + os.sep + dataset + os.sep +
                        'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'dist' + os.sep +
                        f"{specie}-r-{radius}_{dis_method}_{rsa_method}_dist.npy")
    # check if file exists
    if os.path.exists(cluster_sizes_dict_path):
        cluster_sizes_dict = np.load(cluster_sizes_dict_path, allow_pickle=True).item()
        print(f"Loaded cluster sizes from {cluster_sizes_dict_path}")
    else: # file does not exist, create empty dictionary
        cluster_sizes_dict = {}  # dictionary to store cluster sizes for each rnd z map
        print(f"Cluster sizes file {cluster_sizes_dict_path} not found. Starting from scratch.")
        
        # initialize log
    # check if there a log (same filename but with _log.txt extension)
    log_file_path = cluster_sizes_dict_path.replace('.npy', '_log.txt')
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            log = f.readlines()
        print(f"Loaded log from {log_file_path}")
    else:
        log = []  # initialize empty log
        print(f"No log found at {log_file_path}. Starting from scratch.")

    # check if cluster_sizes_dict has the current z_threshold and connectivity
    key = f"z{z_threshold}"
    if key in cluster_sizes_dict:
        print(f"Found cluster sizes for {key}")
        # stop script
        return 10
        
    else:
        print(f"No cluster sizes found for {key}")
        cluster_sizes_dict[key] = {}  # initialize empty dictionary for this threshold and connectivity

    rsa_model_path = datafolder + os.sep + dataset + os.sep + 'rsa_models' + os.sep + rsa_model + ".csv"
    config_path = datafolder + os.sep + dataset + os.sep + 'config_files' + os.sep + specie + '_' + model + '.yaml'

    # Load config.yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from {config_path}")

    # add clear mark to log
    log.append("\n" + "="*50 + "\n")

    # add current date and time to log
    from datetime import datetime
    log.append(f"Log date and time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # add threshold to log
    log.append(f"Z threshold: {z_threshold}")


    ## Calculate z map for permutations (RSA_rnd folder)
    # check how many rnd mean files are available
    available_files = []
    missing_files = []

    # find all files that match pattern (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep +
                        # model + os.sep + rsa_model + os.sep +
                        # f"r-{radius}_{method}_{rsa_method}_z_*.nii.gz")
    search_query = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep +
                    model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                    f"{specie}-r-{radius}_{dis_method}_{rsa_method}_z_*.nii.gz")

    # print message
    if verbose:
        print("Searching for rnd z map files using:")
        print(search_query)
    
    # list all files matching the pattern on search_query
    file_list = glob.glob(search_query)
    if verbose:
        print(f"Found {len(file_list)} rnd z map files.")
    
    # sort file_list
    file_list.sort()
    # initialize sizes_list to store all cluster sizes (same size as file_list)
    sizes_list = np.zeros((len(file_list),), dtype=object)

    for i, file in enumerate(file_list):
        print(f"{i+1} of {len(file_list)}: Processing file {file}...")
        sizes = count_clusters_sizes(file, threshold=z_threshold, connectivity=connectivity) 
        # write sizes to sizes_list
        sizes_list[i] = sizes
        print(f"Found {len(sizes)} clusters")

    # calculate total number of images processed
    number_of_images = len(file_list)
    # add to log
    log.append(f"Processed {number_of_images} z map files for cluster size distribution.")
    # add number of processed files cluster_sizes_dict
    cluster_sizes_dict[key]['number_of_images'] = number_of_images
    # sizes_list is an array of arrays, convert to a single list
    # sizes_list = np.concatenate(sizes_list).ravel()
    cluster_sizes_dict[key]['cluster_sizes'] = sizes_list#.tolist()  # convert to list for saving as npy
    # log info added to cluster_sizes_dict
    log.append(f"Data added to cluster_sizes_dict under key {key}. Total clusters found: {len(sizes_list)}")

    # create directory of cluster_sizes_dict if it does not exist
    dist_folder = os.path.join(datafolder, dataset, 'results', 'RSA', model, rsa_model, 'dist')
    if not os.path.exists(dist_folder):
        os.makedirs(dist_folder)

    # save cluster_sizes_dict to cluster_sizes_dict_path
    with open(cluster_sizes_dict_path, 'wb') as f:
        np.save(f, cluster_sizes_dict)

    print(f"Saved cluster sizes to {cluster_sizes_dict_path}")
    # add to log
    log.append(f"Saved cluster sizes to {cluster_sizes_dict_path}")
    # print how many images were processed and which were processed
    log.append(f"Processed {len(file_list)} z map files for cluster size distribution.")
    log.append(f"Processed files: {file_list}")
    # save log
    log_path = cluster_sizes_dict_path.replace('.npy', '_log.txt')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log))
    print(f"Saved log to {log_path}")
    return True

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import Normalize, TwoSlopeNorm
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Tuple, Callable, Union


def _extract_categories(model_dict: Dict[str, Dict[str, float]], categories: Optional[List[str]] = None) -> List[str]:
    """Return an ordered list of categories. If `categories` is None, use keys order as loaded."""
    if categories is None:
        # Preserve insertion order (Py3.7+ dicts preserve insertion order)
        categories = list(model_dict.keys())
    else:
        # Sanity-check: ensure all provided categories exist
        missing = [c for c in categories if c not in model_dict]
        if missing:
            raise KeyError(f"Categories not found in model: {missing}")
    return categories


def _build_matrix(rsa_model_dict: Dict, categories: List[str],
                  value_fn: Optional[Callable[[float], float]] = None,
                  default_nan: float = np.nan) -> np.ndarray:
    """Build a numeric matrix V[i, j] from nested rsa_model_dict['model'][row][col]."""
    model = rsa_model_dict['model']
    n = len(categories)
    V = np.full((n, n), default_nan, dtype=float)
    for i, r in enumerate(categories):
        # Each row must exist
        if r not in model:
            continue
        for j, c in enumerate(categories):
            try:
                val = model[r][c]
            except Exception:
                val = default_nan
            if value_fn is not None and val is not np.nan and val is not None:
                try:
                    val = value_fn(val)
                except Exception:
                    # If transform fails, fall back to original
                    pass
            V[i, j] = val
    return V


def plot_rsa_circle_matrix(
    rsa_model_dict: Dict,
    categories: Optional[List[str]] = None,
    value_fn: Optional[Callable[[float], float]] = None,
    cmap: Union[str, mcolors.Colormap] = 'Greys',
    diagonal_color: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    center: Optional[float] = None,
    circle_radius: float = 0.42,
    size_by_value: bool = False,
    size_range: Tuple[float, float] = (0.15, 0.48),
    edgecolor: str = 'white',
    linewidth: float = 0.5,
    show_grid: bool = False,
    annotate: bool = False,
    annot_fmt: str = '{:.2f}',
    nan_color: str = '#d9d9d9',
    figsize: Tuple[float, float] = (9, 9),
    title: Optional[str] = None,
    savepath: Optional[str] = None,
    background: str = 'white',
    invert_yaxis: bool = True,
    tight_layout: bool = True,
    dpi: int = 150,
    visible_border: bool = False,
    draw_colorbar: bool = False,
):
    """
    Draw a matrix of pairwise comparisons using independently drawn circles.

    Parameters
    ----------
    rsa_model_dict : dict
        Dictionary as loaded from npy, with nested structure rsa_model_dict['model'][row][col].
    categories : list[str], optional
        Order of rows/cols. Defaults to insertion order of keys in rsa_model_dict['model'].
    value_fn : callable, optional
        Function applied to each scalar value before plotting (e.g., np.abs, lambda v: v*100).
    cmap : str or Colormap
        Matplotlib colormap name or object.
    diagonal_color : str, optional
        If provided, override the fill color of diagonal cells with this color.
    vmin, vmax : float, optional
        Color scale limits. If None, computed from data (ignoring NaNs).
    center : float, optional
        If provided, use a TwoSlopeNorm centered at this value (handy for diverging data).
    circle_radius : float
        Base radius of circles in axis units (cells are 1x1). Ignored if size_by_value=True.
    size_by_value : bool
        If True, scale circle size based on normalized value between `size_range`.
    size_range : (float, float)
        Min and max circle radius if size_by_value=True.
    edgecolor : str
        Circle outline color.
    linewidth : float
        Circle outline width.
    show_grid : bool
        If True, draws light grid lines for the cell boundaries.
    annotate : bool
        If True, writes the numeric value on top of the circles.
    annot_fmt : str
        Format string for annotations.
    nan_color : str
        Fill color for NaN or missing values.
    figsize : (float, float)
        Figure size in inches.
    title : str, optional
        Title of the plot.
    savepath : str, optional
        If provided, the figure is saved to this path.
    background : str
        Figure/axes background color.
    invert_yaxis : bool
        If True, puts the first row at the top (matrix-like view).
    tight_layout : bool
        If True, calls plt.tight_layout() before returning.
    dpi : int
        Figure DPI.

    Returns
    -------
    fig, ax : Matplotlib figure and axes.
    """
    model = rsa_model_dict['model']
    categories = _extract_categories(model, categories)
    V = _build_matrix(rsa_model_dict, categories, value_fn=value_fn)

    # Set up normalization for colors
    finite_vals = V[np.isfinite(V)]
    if finite_vals.size == 0:
        raise ValueError("No finite values found in RSA model to plot.")

    if vmin is None:
        vmin = float(np.nanmin(finite_vals))
    if vmax is None:
        vmax = float(np.nanmax(finite_vals))

    if center is not None:
        # Ensure vmin < center < vmax as required by TwoSlopeNorm. If not, expand
        # symmetrically around the center using the data spread.
        if not (vmin < center < vmax):
            spread = float(np.nanmax(np.abs(finite_vals - center))) if finite_vals.size else 1.0
            if not np.isfinite(spread) or spread == 0:
                spread = 1.0
            vmin = center - spread
            vmax = center + spread
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    # Prepare figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor(background)
    fig.patch.set_facecolor(background)

    n = len(categories)
    # Draw cell boundaries (optional)
    if show_grid:
        for i in range(n + 1):
            ax.plot([-0.5, n - 0.5], [i - 0.5, i - 0.5], lw=0.5, alpha=0.3)
            ax.plot([i - 0.5, i - 0.5], [-0.5, n - 0.5], lw=0.5, alpha=0.3)

    # Draw circles cell by cell
    for i in range(n):
        for j in range(n):
            val = V[i, j]
            x, y = j, i
            if np.isnan(val):
                face = nan_color
                r = circle_radius if not size_by_value else np.mean(size_range)
            else:
                face = plt.get_cmap(cmap)(norm(val))
                if size_by_value:
                    # Scale radius linearly between size_range based on normalized value
                    nv = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
                    r = size_range[0] + nv * (size_range[1] - size_range[0])
                else:
                    r = circle_radius

            # Override diagonal color if requested
            if diagonal_color is not None and i == j:
                face = diagonal_color

            circ = patches.Circle((x, y), radius=r, facecolor=face,
                                  edgecolor=edgecolor, linewidth=linewidth)
            ax.add_patch(circ)

            if annotate and np.isfinite(val):
                ax.text(x, y, annot_fmt.format(val), ha='center', va='center', fontsize=9)

    # Axes cosmetics
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, n - 0.5)
    if invert_yaxis:
        ax.invert_yaxis()
    ax.set_aspect('equal')

    if visible_border:
        # Ticks & labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_yticklabels(categories)
    else:
        # remove border and ticks
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

    if draw_colorbar:
        # Colorbar
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Comparison value', rotation=90, va='center')

    if title:
        ax.set_title(title)

    if tight_layout:
        plt.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches='tight', facecolor=background)
        print(f"Figure saved to: {savepath}")

    return fig, ax


from scipy.ndimage import label, generate_binary_structure

def cluster_masks_3d(vol, threshold=None, two_sided=False, connectivity=3, dtype=np.uint8):
    """
    Return binary masks for each connected cluster in a 3D volume.

    Parameters
    ----------
    vol : np.ndarray
        3D array (e.g., statistical map) to cluster.
    threshold : float or None
        If None, any nonzero voxel is considered for clustering.
        If a float, uses vol > threshold for positive clusters,
        and (if two_sided) vol < -threshold for negative clusters.
    two_sided : bool
        If True, find clusters separately for positive and negative sides.
    connectivity : {1, 2, 3}
        Neighborhood connectivity in 3D (1=6-connectivity, 2=18, 3=26).
    dtype : np.dtype
        Output dtype for binary masks (default uint8).

    Returns
    -------
    masks : list[np.ndarray] or dict[str, list[np.ndarray]]
        - If two_sided=False: a list of binary masks (one per cluster), sorted by size desc.
        - If two_sided=True: dict with keys 'positive' and 'negative', each a list of masks.
    """

    # Build mask(s)
    if threshold is None:
        pos_mask = vol != 0
        neg_mask = None
    else:
        pos_mask = vol > threshold
        neg_mask = (vol < -threshold) if two_sided else None

    # 3D connectivity kernel
    structure = generate_binary_structure(rank=3, connectivity=connectivity)

    def _masks_from_binary(binary_mask):
        if binary_mask is None:
            return []
        labels, n = label(binary_mask, structure=structure)
        if n == 0:
            return []

        # Count voxels in each label (drop background at index 0)
        counts = np.bincount(labels.ravel())
        if counts.size <= 1:
            return []
        sizes = counts[1:]  # ignore background

        # Sort labels by size (desc)
        order = np.argsort(sizes)[::-1]
        lbl_ids_sorted = (order + 1)  # labels start at 1

        # Generate binary masks (1s and 0s) per cluster
        masks = [(labels == lbl_id).astype(dtype, copy=False) for lbl_id in lbl_ids_sorted]
        return masks

    clusters = _masks_from_binary(pos_mask)

    if two_sided:
        neg_masks = _masks_from_binary(neg_mask)
        return {'positive': clusters, 'negative': neg_masks}

    return 

import numpy as np
import nibabel as nib
from scipy.ndimage import maximum_filter, label, generate_binary_structure
import shutil

def world_coords(ijk, affine):
    ijk = np.asarray(ijk)
    ijk_h = np.c_[ijk, np.ones(len(ijk))]
    xyz = ijk_h @ affine.T
    return xyz[:, :3]

def local_maxima(mask, stat, footprint=None):
    """Return indices of voxels that are local maxima inside 'mask'."""
    if footprint is None:
        # 26-neighborhood in 3D
        footprint = generate_binary_structure(3, 3)
    # Find voxels that equal the local max in neighborhood
    neighborhood_max = maximum_filter(stat, footprint=footprint, mode='nearest')
    is_local_max = (stat == neighborhood_max) & mask
    # Optional: break plateaus by preferring higher smoothed value or center-of-mass
    return np.argwhere(is_local_max)

def pick_subpeaks(affine, stat, cluster_mask, min_dist_mm=8.0, max_peaks=None):
    """
    cluster_mask: boolean array for a single cluster
    stat: same-shape array of Z/T values
    Returns list of (z, (i,j,k), (x,y,z))
    """
    # candidate local maxima within this cluster
    cand_ijk = local_maxima(cluster_mask, stat)
    cand_vals = stat[tuple(cand_ijk.T)]
    order = np.argsort(-cand_vals)
    cand_ijk = cand_ijk[order]
    cand_vals = cand_vals[order]
    cand_xyz = world_coords(cand_ijk, affine)

    kept = []
    kept_xyz = []

    for v, ijk, xyz in zip(cand_vals, cand_ijk, cand_xyz):
        if not kept:
            kept.append((float(v), tuple(ijk), tuple(xyz)))
            kept_xyz.append(xyz)
        else:   
            dists = np.linalg.norm(np.vstack(kept_xyz) - xyz, axis=1)
            if np.all(dists >= min_dist_mm):
                kept.append((float(v), tuple(ijk), tuple(xyz)))
                kept_xyz.append(xyz)
        if max_peaks is not None and len(kept) >= max_peaks:
            break
    return kept

def extract_clusters_and_peaks(
    nifti_path,
    stat_thresh=None,
    min_dist_mm=8.0,
    max_peaks_per_cluster=3,
    label_dict=None,
    label_nii_data=None,
    label_affine=None,
):
    img = nib.load(nifti_path)
    stat = img.get_fdata()
    affine = img.affine
    # print dimensions of image in terms of number of voxels in each dimension
    print(f"Image shape (voxels): {stat.shape}")

    # Normalize label inputs (allow passing a NIfTI image or a path)
    if label_nii_data is not None and label_dict is not None:
        if isinstance(label_nii_data, (str, os.PathLike)):
            label_img = nib.load(str(label_nii_data))
            label_nii_data = label_img.get_fdata()
            if label_affine is None:
                label_affine = label_img.affine
        elif hasattr(label_nii_data, "get_fdata") and hasattr(label_nii_data, "affine"):
            # nibabel image-like
            label_img = label_nii_data
            label_nii_data = label_img.get_fdata()
            if label_affine is None:
                label_affine = label_img.affine
        else:
            # numpy array
            if label_affine is None:
                # Backward-compatible fallback only if shapes match exactly
                if hasattr(label_nii_data, "shape") and label_nii_data.shape == stat.shape:
                    label_affine = affine
                else:
                    raise ValueError(
                        "label_nii_data was provided as a numpy array, but label_affine is missing and "
                        "label_nii_data.shape != stat.shape. Pass a nibabel image (nib.load(...)) or "
                        "provide label_affine so peak mm coordinates can be mapped into atlas voxel space."
                    )

    # threshold (if not already cluster-thresholded)
    if stat_thresh is not None:
        mask = stat >= stat_thresh
    else:
        mask = np.isfinite(stat) & (stat > 0)  # or however your map is defined

    # connected components (26-connectivity)
    struct = generate_binary_structure(3, 3)
    labeled, n_clu = label(mask, structure=struct)

    results = []
    for c in range(1, n_clu + 1):
        cluster_mask = labeled == c
        # cluster peak set
        peaks = pick_subpeaks(
            affine=affine,
            stat=stat,
            cluster_mask=cluster_mask,
            min_dist_mm=min_dist_mm,
            max_peaks=max_peaks_per_cluster
        )
        # peaks must be a list of (z, (i,j,k), (x,y,z))
        print("this the print for peaks")
        print(peaks)
        print("end of print for peaks")
        # go through every peak and label region
        if label_dict is not None and label_nii_data is not None:
            print(f"Labelling peaks for cluster {c}...")
            for idx in range(len(peaks)):
                stat_ijk = peaks[idx][1]
                peak_xyz_mm = np.asarray(peaks[idx][2], dtype=float)

                # Map peak mm coords -> label voxel coords (handles differing grids)
                label_ijk_f = nib.affines.apply_affine(np.linalg.inv(label_affine), peak_xyz_mm)
                label_ijk = tuple(np.rint(label_ijk_f).astype(int))

                print(f"stat ijk: {stat_ijk} -> label ijk: {label_ijk}")
                print(f"label_nii_data shape: {label_nii_data.shape}")

                in_bounds = all(0 <= label_ijk[d] < label_nii_data.shape[d] for d in range(3))
                if in_bounds:
                    label_val = int(label_nii_data[label_ijk])
                else:
                    label_val = 0
                # find the row matching with label_val in label_dict
                
                # create a version of label_dict['Number'] with int values (in case they are strings in the original dataframe)
                label_dict_list = label_dict['Number'].tolist()
                # replace NaN values with 0 in label_dict_list
                label_dict_list = [0 if pd.isna(x) else x for x in label_dict_list]

                # print(label_dict_list)
                label_dict_list = [int(x) for x in label_dict_list]
                print(f"label_dict_list: {label_dict_list}")
                print(f"label_val: {label_val}")

                label_row = label_dict[label_dict['Number'] == label_val]
                if not in_bounds:
                    region_name = "OutOfAtlas"
                else:
                    region_name = label_row['Region'].values[0] if not label_row.empty else "Unknown"
                # add region name to peaks
                peaks[idx] = (peaks[idx][0], peaks[idx][1], peaks[idx][2], region_name)


        # gather cluster-level descriptors
        cluster_size = int(cluster_mask.sum())
        cluster_max = max(peaks, key=lambda t: t[0]) if peaks else None
        results.append({
            "cluster_id": c,
            "size_vox": cluster_size,
            "peaks": [{"Z": z, "ijk": ijk, "xyz_mm": xyz, "region": region} for (z, ijk, xyz, region) in peaks],
            "peak_Z": cluster_max[0] if cluster_max else None,
            "peak_xyz_mm": cluster_max[2] if cluster_max else None
        })
    return results

def clusters_to_table(results, out_path, apply_coords_transform=False, atlas_file=None, mask=None):
    """
    Build a hierarchical (cluster -> subpeak) table from `results`
    and save it as a CSV file (no Excel/openpyxl dependency, which is missing on
    the Linux remote and used to break step 10).
    """
    if apply_coords_transform:
        # get the original image to apply coordinate transform
        original_img = nib.load(atlas_file)
        # get the results image to apply coordinate transform
        results_img = nib.load(mask)



    rows = []
    # print('hit new 2')
    for cluster in results:
        cid = cluster['cluster_id']
        size = cluster['size_vox']
        peak_Z_cluster = cluster['peak_Z']
        peak_xyz_cluster = cluster['peak_xyz_mm']

        for sub_idx, peak in enumerate(cluster['peaks'], start=1):
            i, j, k = peak['ijk']
            
            
            # apply coordinate transform if requested
            if apply_coords_transform:
            # get inputs for transform_coords
                coords_input = (i, j, k)
                # get new_coordinates
                new_coordinates = transform_coords(coords_input,results_img,original_img)[0]
                print(f"new coordinates are {new_coordinates}")
                print(f"type of new coordinates: {type(new_coordinates)}")
                # update i, j, k with new_coordinates
                i, j, k = new_coordinates


            x_mm, y_mm, z_mm = peak['xyz_mm']
            region = peak['region']

            rows.append({
                'cluster_id': cid,
                'subpeak_id': sub_idx,
                'cluster_size_vox': size,
                'subpeak_Z': peak['Z'],
                'subpeak_x_vox': i,
                'subpeak_y_vox': j,
                'subpeak_z_vox': k,
                'subpeak_x_mm': x_mm,
                'subpeak_y_mm': y_mm,
                'subpeak_z_mm': z_mm,
                'region': region

            })
            # print each element in rows
            # for key, value in rows[-1].items():
            #     print(f"{key}: {value}")

    df = pd.DataFrame(rows)
    print(f"DataFrame created with {len(df)} rows.")
    print(df.head())
    # Make it hierarchical: first cluster, then subpeak
    df = df.set_index(['cluster_id', 'subpeak_id']).sort_index()

    df.to_csv(out_path)
    return df

def create_tables(datafolder, dataset, specie, model, rsa_model, radius,
                  dis_method, rsa_method, z_threshold=3.1, min_dist_mm=8.0, max_peaks_per_cluster=3,
                  label_dict=None, label_nii_data=None, apply_coords_transform=True,
                  atlas_file=None, mask=None, mask_type=None):
    res_folder = r"G:\My Drive\Results" + os.sep + dataset + os.sep + "current-results"
    if mask_type is None:
        res_image = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                    model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                    f"{specie}-r-{radius}_{dis_method}_{rsa_method}_zt{z_threshold}_corrected.nii.gz"
                    )
        # create copy of res_image using the name of the rsa_model
        # G:\My Drive\Results\EmoB\current-results

        res_image_copy = (res_folder + os.sep + 'RSA' + os.sep +
                    f"{specie}_{rsa_model}_zt{z_threshold}_corrected.nii.gz"
                    )

        out_path =  (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                    model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                    f"{specie}-r-{radius}_{dis_method}_{rsa_method}_zt{z_threshold}.csv")

        out_path_copy = (res_folder + os.sep + 'RSA' + os.sep +
                    f"{specie}_{rsa_model}_zt{z_threshold}.csv")
    else:
        res_image = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                    model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                    f"{mask_type}-{specie}-r-{radius}_{dis_method}_{rsa_method}_zt{z_threshold}_corrected.nii.gz"
                    )
        # create copy of res_image using the name of the rsa_model
        res_image_copy = (res_folder + os.sep + 'RSA' + os.sep + specie + os.sep +
                          mask_type + os.sep +
                    f"{specie}_{rsa_model}_zt{z_threshold}_corrected.nii.gz"
                    )

        out_path =  (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                    model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                    f"{mask_type}-{specie}-r-{radius}_{dis_method}_{rsa_method}_zt{z_threshold}.csv")

        out_path_copy = (res_folder + os.sep + 'RSA' + os.sep + specie + os.sep +
                         mask_type + os.sep +
                    f"{specie}_{rsa_model}_zt{z_threshold}.csv")
    # res_image = r"P:\userdata\raulh87\data\EmoB\results\RSA\basic-block\old_emotion-valence\mean\D-r-3_mahalanobis_kendall_z_corrected.nii.gz"
    results = extract_clusters_and_peaks(res_image, stat_thresh=None, min_dist_mm=min_dist_mm, 
                                         max_peaks_per_cluster=max_peaks_per_cluster, label_dict=label_dict,
                                         label_nii_data=label_nii_data)
    # if results is empty
    if not results:
        print(f"No clusters found in {res_image}. No table will be created.")
        return False
        
        
    
    clusters_to_table(results, out_path, apply_coords_transform=apply_coords_transform, atlas_file=atlas_file, mask=mask)
    print(f"Files written in: {out_path}")

    # Mirror the result image + table to the Google-Drive results tree. This is a
    # convenience for the Windows dashboard and must never fail the step: the Drive
    # root (res_folder) is a hardcoded G:\ path that does not exist on the Linux
    # remote, so guard the whole block and continue on any error.
    try:
        os.makedirs(os.path.dirname(out_path_copy), exist_ok=True)
        # Copy the result image and table to the short Drive filenames.
        shutil.copyfile(res_image, res_image_copy)
        shutil.copyfile(out_path, out_path_copy)
        print(f"Copied result image to: {res_image_copy}")
        print(f"Copied table to: {out_path_copy}")

        # Also mirror the UNTHRESHOLDED z-map (the corrected map's name minus
        # '_corrected') so the dashboard viewer's threshold slider can explore the
        # data freely; clusters/tables remain threshold-specific (this corrected
        # copy + csv). See viz/viewer_app.py.
        res_image_unthr = res_image.replace('_corrected', '')
        res_unthr_copy = res_image_copy.replace('_corrected', '')
        if os.path.exists(res_image_unthr):
            shutil.copyfile(res_image_unthr, res_unthr_copy)
            print(f"Copied unthresholded z-map to: {res_unthr_copy}")
        else:
            print(f"Unthresholded z-map not found, skipped: {res_image_unthr}")
    except Exception as e:
        print(f"WARNING: skipped Google-Drive mirror copy "
              f"({e.__class__.__name__}: {e}). Primary outputs are in: "
              f"{os.path.dirname(out_path)}")
    return True

import os
import nibabel as nib
import yaml
import pandas as pd
import numpy as np

def load_meta_similarity_map(rsa_model_path, ref_img, datafolder, dataset, sub_N, session, run_N, config_path, method='pearson', radius=3, verbose=False):
    # loads the meta similarity map for given parameters
    # Load config.yaml

    # get shape from ref_img
    X, Y, Z = ref_img.shape
    
    rsa_model_dict = read_model_dict(rsa_model_path)
    categories = rsa_model_dict['categories']
    n_pairs = int(len(categories) * (len(categories) - 1) / 2)
    meta_similarity_map = np.empty((X, Y, Z, n_pairs), dtype=np.float32)
    pair_names = [] # to store pair names

    k = 0 # index for meta_similarity_map
    for indx1,cat1 in enumerate(categories):
        for indx2,cat2 in enumerate(categories):
            if indx1 >= indx2:
                continue
            # load similarity comparison
            map = load_pairwise_similarity_map(datafolder, dataset, sub_N, session, run_N, cat1, cat2, config_path, method=method, radius=radius, verbose=verbose)
            if verbose:
                pair_name = f"{cat1}_{cat2}"
                print(f"Loaded pair {pair_name} into index {k}")
                pair_names.append(pair_name) # for sanity check
            # store in meta_similarity_map
            meta_similarity_map[..., k] = map
            k += 1
    return meta_similarity_map

def load_pairwise_similarity_map(datafolder, dataset, sub_N, session, run_N, stim_1_name, stim_2_name, config_path, method='pearson', radius=3, verbose=False):
    # Loads similarity map that compares two stimuli for given parameters

    # Load config.yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = config['model']
    specie = config['specie']
    task = config['task']

    file_path = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep + model + os.sep + f"{specie}-sub-{sub_N:02d}" + os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}" + os.sep + f"r-{radius}_{method}_{stim_1_name}_{stim_2_name}.nii.gz"
    if verbose:
        print(f"Loading {file_path}")
    map = nib.load(file_path).get_fdata()
    return map

def read_model_dict(model_path):
    ''' Reads the RSA model from an excel file and returns as a dictionary 
    Input:
        model_path: path to the excel file
    Output:
        rsa_model_dict: dictionary with the RSA model
            - 'model': pairwise dictionary with similarity values
            - 'categories': list of categories
            - 'pairs': list of tuples with category pairs
    '''
    model_table = pd.read_excel(model_path)
    # get all columns
    categories = model_table.columns.tolist()
    # drop first element
    categories = categories[1:]
    # build pairwise dictionary
    pairs = [] # to store pairs

    rsa_model_dict = {}
    rsa_model_dict['model'] = {}
    for indx1,cat1 in enumerate(categories):
        rsa_model_dict['model'][cat1] = {}
        for indx2,cat2 in enumerate(categories):
            if indx1 == indx2:
                continue
            rsa_model_dict['model'][cat1][cat2] = model_table[cat1][indx2]
            pairs.append((cat1, cat2))  # add pair to list
    rsa_model_dict['categories'] = categories
    rsa_model_dict['pairs'] = pairs
    return rsa_model_dict

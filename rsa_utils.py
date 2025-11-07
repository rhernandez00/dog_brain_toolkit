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
from time import time, perf_counter
import shutil
# import random
import random
from scipy.ndimage import label, generate_binary_structure

import numpy as np
from scipy import ndimage

import preprocess_functions

def apply_cluster_correction(datafolder, dataset, specie, model, rsa_model, radius,
                             method, rsa_method, z_threshold, cluster_threshold, forced_minimal_cluster_size=None,
                             verbose=False):
    """
    """
    connectivity = 26  # based on FSL default for 3D images
    # Load cluster sizes dictionary
    cluster_sizes_dict_path = (datafolder + os.sep + dataset + os.sep +
                        'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'dist' + os.sep +
                        f"{specie}-r-{radius}_{method}_{rsa_method}_dist.npy")
    # check if file exists
    if os.path.exists(cluster_sizes_dict_path):
        cluster_sizes_dict = np.load(cluster_sizes_dict_path, allow_pickle=True).item()
        print(f"Loaded cluster sizes from {cluster_sizes_dict_path}")
    else:
        # trigger error
        raise FileNotFoundError(f"Cluster sizes file {cluster_sizes_dict_path} not found")

    

    # mean model similarity map
    mean_model_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                    model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                    f"{specie}-r-{radius}_{method}_{rsa_method}_mean.nii.gz")

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
    # calculate z map
    z_map = (mean_model_img - dist_mean_img) / dist_std_img
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
    corrected_z_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                             model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                             f"{specie}-r-{radius}_{method}_{rsa_method}_z_corrected.nii.gz")
    nib.save(nib.Nifti1Image(z_map_thresholded, affine=img_affine), corrected_z_map_path)
    print(f"Saved corrected z map to {corrected_z_map_path}")

import numpy as np
from scipy import ndimage

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

def nifti_mean(img_list, result_map_path=None, result_map_path_std=None, verbose=False):
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

    std_data = np.sqrt(sum_sq_diff / count)

    if result_map_path:
        # save mean image
        mean_img = nib.Nifti1Image(mean_data, img_affine)
        nib.save(mean_img, result_map_path)
        if verbose:
            print(f"Saved mean image to {result_map_path}")
        # save std image
    if result_map_path_std:
        std_img = nib.Nifti1Image(std_data, img_affine)
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


def compare_with_model2(datafolder, dataset, sub_N, session_and_run_dict,
                    specie, model, stim_types, mask, task, radius, rsa_model,
                    method='pearson', rsa_method='kendall', replace_file=False, verbose=False, wait_time=300,
                    rnd=False, reps=1000, create_subject_mean=False, replace_rnd_files=False):
    '''
    Compare pairwise similarity maps with a model.
    '''
    ###- Pending: Implement logic to check for existing output files, remove manually for now ###

    # if calculating permutations, create_subject_mean should be False
    if rnd:
        create_subject_mean = False

    print(f"Pairwise similarity vs model for: {specie}-sub-{sub_N:02d}, model {model}...")
    # load the mask to use as reference
    ref_img = nib.load(mask).get_fdata()
    mask_affine = nib.load(mask).affine
    
    all_exist = True # this is a flag to indicate if all output files exist
    # Check for existing output files
    for entry in session_and_run_dict:
        session = entry['session']
        run_N = entry['run']
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
                existing_files = glob.glob(folder_permutations + os.sep + f"r-{radius}_{method}_{rsa_method}_*.nii.gz")
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
            filename = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                        model + os.sep + rsa_model + os.sep + f"{specie}-sub-{sub_N:02d}" + os.sep + 
                        f"ses-{session}_task-{task}_run-{run_N:02d}" + os.sep +
                        f"r-{radius}_{method}_{rsa_method}.nii.gz")
            if os.path.exists(filename):
                if verbose:
                    print(f"Skipping: Found existing model comparison file: {filename}.")
                    # all done, exit function
                continue
            else: # file does not exist
                if verbose:
                    print(f"Missing. File {filename} does not exist.")
                all_exist = False
                break

    # if all files exist, skip computation
    if all_exist:
        print(f"All output files for {specie}-sub-{sub_N:02d} already exist.")        
        if not replace_file:
            print("Skipping computation as replace_file is False.")
            return  # all files exist, skip computation
        else:
            # issue error, not implemented
            raise NotImplementedError("Replacing existing files is not implemented.")
    
    ### To this point, at least one output file is missing, proceed to check input files ###
    
    ## check if input pairwise similarity maps necessary are available for each run
    print("Checking for input pairwise similarity maps...")
    # initialize pairs_available_array a boolean array to indicate if all pairwise similarity maps are available for each run
    # initialize it to False
    all_available = True
    pairs_available_array = np.zeros(len(session_and_run_dict), dtype=bool)
    for index, entry in enumerate(session_and_run_dict):
        session = entry['session']
        run_N = entry['run']
        # correct session to 2 digits
        session = f"{session:02d}"
        # check if all beta maps exist
        pairs_available = True
        for i, stim_i in enumerate(stim_types):
            for j, stim_j in enumerate(stim_types):
                if i >= j:
                    continue  # only upper triangle
                filename  = os.path.join(
                    datafolder, dataset, 'results', 'RSA', model,
                    f"{specie}-sub-{sub_N:02d}",
                    f"ses-{session}_task-{task}_run-{run_N:02d}",
                    f"r-{radius}_{method}_{stim_i}_{stim_j}.nii.gz"
                )
                # check if file exists
                if not os.path.exists(filename):
                    if verbose:
                        print(f"Missing pairwise similarity map: {filename}")
                    pairs_available = False
                    break
        if not pairs_available:
            pairs_available_array[index] = False
            all_available = False
    # if any of the runs are missing pairwise similarity maps, skip computation
    if not all_available:
        print("Some input pairwise similarity maps are missing. Skipping computation.")
        return
    print("All input files are available.")
    
    ### All input files are available, check if there are temporary files to indicate other instance processing it ###
    # initialize file_list to store output files for subject mean
    file_list = []
    tmp_file = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep + 
                        model + os.sep + rsa_model + os.sep +  f"{specie}-sub-{sub_N:02d}_processing.tmp")
    if rnd: # if running permutations, check for existing temporary files
        # temporal file to indicate that this participant is being processed
        
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
                # remove tmp_file
                os.remove(tmp_file)
        # create tmp_file
        with open(tmp_file, 'w') as f:
            f.write('Processing...\n')
        if verbose:
            print(f"Created processing temporary file {tmp_file}.")

    ### Now, all input files are available, and no recent temporary file exists,

    
    
    # go through each session and run,
    for entry in session_and_run_dict:
        session = entry['session']
        run_N = entry['run']
        # correct session to 2 digits
        session = f"{session:02d}"
        ## check for existing temp files, skip if recent temp files, if they are old, skip calculation, otherwise go on
        print(f"sub-{sub_N:02d}, ses-{session}, run-{run_N:02d}, all checks, computing pairwise similarity maps...")
        # create folder if not exists
        
    
        # try to compute pairwise similarity maps
        try:  # if error, remove temp file and continue
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
                method=method,
                rsa_method=rsa_method,
                replace_file=replace_file,
                verbose=verbose,
                rnd=rnd,
                reps=reps,
                replace_rnd_files=replace_rnd_files
            )
        except Exception as e:
            print(f"Error computing pairwise similarity maps for {specie}-sub-{sub_N:02d}, ses-{session}, run-{run_N:02d}: {e}")
                    
    # remove temp file
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
        print(f"Removed temporary file {tmp_file}.")
        
    

    # if create_subject_mean is True, compute mean similarity map across runs for the subject
    if create_subject_mean:
        print(f"{specie}-sub-{sub_N:02d}, computing subject mean similarity map...")
        # gather all output files
        for entry in session_and_run_dict:
            session = entry['session']
            run_N = entry['run']
            # correct session to 2 digits
            session = f"{session:02d}"
            # build output filename
            res_file = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep +
                    'RSA' + os.sep + model + os.sep + rsa_model + os.sep +
                    f"{specie}-sub-{sub_N:02d}" + os.sep +
                    f"ses-{session}_task-{task}_run-{run_N:02d}" + os.sep +
                    f"r-{radius}_{method}_{rsa_method}.nii.gz")
            # check if res_file exists
            if os.path.exists(res_file):
                file_list.append(res_file)
            else:
                # issue warning
                print(f"Warning: Missing file for subject mean: {res_file}")
                
        # build result_map_path
        result_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep +
                            'RSA' + os.sep + model + os.sep + rsa_model + os.sep +
                            f"{specie}-sub-{sub_N:02d}" + os.sep +
                            f"r-{radius}_{method}_{rsa_method}_mean.nii.gz")
        # same but std
        result_map_path_std = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep +
                           'RSA' + os.sep + model + os.sep + rsa_model + os.sep +
                           f"{specie}-sub-{sub_N:02d}" + os.sep +
                           f"r-{radius}_{method}_{rsa_method}_std.nii.gz")
        # compute mean across file_list
        nifti_mean(file_list, result_map_path=result_map_path, result_map_path_std=result_map_path_std, verbose=False)
        print(f"Saved to {result_map_path}, subject mean similarity map")

    print('Pairwise similarity maps computed.')

def compare_with_model(ref_img, mask_affine, datafolder, sub_N, session, run_N, 
                       specie, model, dataset, task, radius, rsa_model, method='pearson', 
                       rsa_method='pearson', replace_file=False, verbose=False, rnd=False, reps=1000, 
                       replace_rnd_files=False):
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
    print(f"for {specie}-sub-{sub_N:02d}, ses-{session}, run-{run_N:02d}...")
    rsa_model_path = datafolder + os.sep + dataset + os.sep + 'rsa_models' + os.sep + rsa_model + ".xlsx"
    config_path = datafolder + os.sep + dataset + os.sep + 'config_files' + os.sep + model + '.yaml'
 
    # Load config.yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # stim_types = config['stim_types']
    model_dict = config['model_dict']
    run_list = config["runs"]
    session_list = config["sessions"]

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
    # load rsa model dictionary
    rsa_model_dict = read_model_dict(rsa_model_path)
    # build model_vector
    model_vector = np.zeros(len(rsa_model_dict['pairs']))
    # print('Model vector length: ' + str(len(model_vector)))
    
    for i, pair in enumerate(rsa_model_dict['pairs']):
        model_vector[i] = rsa_model_dict['model'][pair[0]][pair[1]]

    
    
    
    meta_similarity_map = load_meta_similarity_map(rsa_model_path, ref_img, datafolder, dataset, sub_N, session, run_N, config_path, method=method, radius=radius, verbose=verbose)
    # create similarity_table (x, y, z) of all voxels in the mask, results will be added here
    similarity_table = np.column_stack(np.where(ref_img > 0))
    # add 1 to x, y, z to match 1-based indexing in itk-snap
    similarity_table += 1
    # add a column for similarity values, initialized to NaN
    similarity_table = np.hstack((similarity_table, np.full((similarity_table.shape[0], 1), np.nan)))
    # initialize warning table
    warning_table = []
    
    # create temporal file to indicate that this participant is being processed
    output_folder = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep + 
                     model + os.sep + rsa_model + os.sep +  f"{specie}-sub-{sub_N:02d}" + os.sep + 
                     f"ses-{session}_task-{task}_run-{run_N:02d}")
    # build output filename _[4 digit padded rnd_N]
    # output_file = os.path.join(output_folder, f"r-{radius}_{method}_{rsa_method}_{rnd_N:04d}.nii.gz")


    for rnd_N in range(0, reps):
        # create an result_map based on the reference image
        result_map = np.zeros(ref_img.shape)
        if rnd:
            # if rnd is True, permute the model values
            # rsa_model_dict = shuffle_model(rsa_model_dict)
            output_folder = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep + model + os.sep + rsa_model + os.sep +  f"{specie}-sub-{sub_N:02d}" + os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}"
            # build output filename _[4 digit padded rnd_N]
            output_file = os.path.join(output_folder, f"r-{radius}_{method}_{rsa_method}_{rnd_N:04d}.nii.gz")
            # check if output_file exists
            if os.path.exists(output_file) and not replace_rnd_files:
                print(f"rnd {rnd_N:04d}/{reps:04d} exist, skipping")
                continue

            model_vector = shuffle_vector(model_vector)
        else:
            if rnd_N > 0:
                print("real data, skipping further repetitions")
                break
            output_folder = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep + model + os.sep + rsa_model + os.sep +  f"{specie}-sub-{sub_N:02d}" + os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}"
            # build output filename
            output_file = os.path.join(output_folder, f"r-{radius}_{method}_{rsa_method}.nii.gz")
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
                              radius, rsa_model, method='pearson', rsa_method='pearson', verbose=False, reps=1000):
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
    method : str
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
    # "P:\userdata\raulh87\data\EmoB\results\RSA_rnd\basic\emotion_valence\D-sub-01\ses-01_task-EmoB_run-01\r-3_pearson_kendall_0000.nii.gz"
    for rep in range(reps):
        file_path = output_folder + os.sep + f"r-{radius}_{method}_{rep:04d}.nii.gz"
        if os.path.exists(file_path):
            if verbose:
                print(f"Removing {file_path}")
            os.remove(file_path)

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
    
    # rsa_model_dict is saved with the same name as model_path but with .npy extension
    rsa_model_dict_path = model_path.replace('.xlsx', '.npy')
    

    if not return_all_comparisons:
        if os.path.exists(rsa_model_dict_path):
            if not erase_existing_npy:
                # load and return
                rsa_model_dict = np.load(rsa_model_dict_path, allow_pickle=True).item()
                return rsa_model_dict
            else:
                # delete rsa_model_dict_path
                os.remove(rsa_model_dict_path)

    # read excel file
    model_table = pd.read_excel(model_path)
    # get all columns
    categories = model_table.columns.tolist()
    # drop first element
    categories = categories[1:]
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

def similarity_searchlight(map_1, map_2, mask, radius, method):
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
    method : {'mahalanobis','pearson','euclidean','kendall'}
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
    method = method.lower()
    if method not in {"mahalanobis", "pearson", "euclidean", "kendall", "correlation"}:
        raise ValueError("method must be one of: 'mahalanobis','pearson','euclidean','kendall','correlation'")

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

    def _kendall(x, y):
        try:
            from scipy.stats import kendalltau
        except Exception as e:
            raise ImportError("SciPy is required for method='kendall'.") from e
        return kendalltau(x, y, nan_policy='omit').correlation

    def _euclidean(x, y):
        return -float(np.linalg.norm(x - y))

    
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
        if method in {"pearson", "kendall"} and x.size < 2:
            val = np.nan
        elif method in {"euclidean", "mahalanobis"} and x.size < 1:
            val = np.nan
        else:
            if method == "pearson":
                val = _pearson(x, y)
            elif method == "correlation":
                val = _correlation(x, y)
            elif method == "kendall":
                val = _kendall(x, y)
            elif method == "euclidean":
                val = _euclidean(x, y)
            elif method == "mahalanobis":
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
    D : ndarray
        Crossnobis distances (unbiased estimate of squared Mahalanobis distance).
        Can be negative around zero.

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
                        method, rsa_method, rsa_model,
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
                        f"{specie}-r-{radius}_{method}_{rsa_method}_dist.npy")
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
                        f"r-{radius}_{method}_{rsa_method}_z_*.nii.gz")
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


def calculate_beta_maps(datafolder, dataset, model, specie, sub_N, session, run_N, task,
                        stim_types, design_template, atlas_file,
                        smooth,
                        radius_fwd,
                        threshold_fwd,
                        redo_if_exists,
                        overwrite_movement, wait_time=300):
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
        current_time = time.time()
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
                          method, replace_file, mah_fold='run-wise',
                          sigma=None, shrinkage='ledoitwolf', return_rdm=True,
                          verbose=False):
    '''
    Calculate pairwise similarity maps using searchlight approach.
    - Checks if input files are available.
    - Checks if output files already exist.
    - Calculates pairwise similarity maps using specified method.
    

    Methods:
        - Pearson correlation
        - Kendall correlation
        - Euclidean distance
        - Mahalanobis distance
        - Correlation distance
    Mahalanobis:
        mah_fold: folding strategy for Mahalanobis distance with cross-validation
            Single run:
                When a single run is available, calculate distance within the same run 
                (creates folds within run and cross-validates).
                Folding strategies:
                    - max-fold: max partitions possible
                    - odd-even: splits data into odd and even trials
            Multiple runs:
                When multiple runs are available, calculate distance across runs 
                (cross-validated across runs).
                Folding strategies:
                    - run-wise: each run is a fold
    Input:
        - beta maps per condition and run
        - mask
        - folding strategy (in case of Mahalanobis)
    '''
    print(f"Calculating pairwise similarity maps for sub-{sub_N:02d} using method: {method} ")
    ## check if output files already exist
    print("Checking for existing similarity map files...")
    all_exist = True
    for indx, entry in enumerate(session_and_run_dict):
        session = entry['session']
        run_N = entry['run']
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
                    f"r-{radius}_{method}_{stim_i}_{stim_j}.nii.gz"
                )
                # check if output_file exists
                if not os.path.exists(output_file):
                    if verbose:
                        print(f"Not found {output_file}.")
                    all_exist = False
                else:
                    if verbose:
                        print(f"Found {output_file} exists.")
    
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
    
    ## check if input beta files are available
    missing = {}
    # iterate over session_and_run_dict
    for entry in session_and_run_dict:
        session = entry['session']
        run_N = entry['run']
        # correct session to 2 digits
        session = f"{session:02d}"
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
                    missing_key = (session, run_N, stim_i)
                    missing[missing_key] = input_file_i
                if not os.path.exists(input_file_j):
                    missing_key = (session, run_N, stim_j)
                    missing[missing_key] = input_file_j
    if len(missing) > 0:
        print(f"Missing input files for sub-{sub_N:02d}:")
        for key, file in missing.items():
            session, run_N, stim = key
            print(f"Session: {session}, Run: {run_N}, Stimulus: {stim} -> {file}")
        return missing  # return missing files info
    print("All input files are available.")
    ## All input files are available, proceed to calculate pairwise similarity maps
    # if method is 'mahalanobis', run mahalanobis for all runs, otherwise run 
    # calculate_pairwise_similarity_maps for each session/run
    if method == 'mahalanobis':
        print("Calculating Mahalanobis pairwise similarity maps...")
        calculate_mahalanobis_pairwise_maps(datafolder, dataset, sub_N, session_and_run_dict,
                          specie, model, stim_types, mask, task, radius, replace_file=False,
                          mah_fold=mah_fold, sigma=sigma,
                          shrinkage=shrinkage, return_rdm=return_rdm, verbose=verbose)
    else:
        print(f"Calculating {method} pairwise similarity maps...")
        for entry in session_and_run_dict:
            session = entry['session']
            run_N = entry['run']
            # correct session to 2 digits
            session = f"{session:02d}"
            calculate_pairwise_similarity_maps(datafolder, dataset, sub_N, session, 
                                       run_N, specie, model, stim_types, mask, 
                                       task, radius=3, method=method, 
                                       replace_file=replace_file, verbose=verbose)
    return None  # no missing files

def calculate_mahalanobis_pairwise_maps(datafolder, dataset, sub_N, session_and_run_dict,
                          specie, model, stim_types, mask, task, radius, replace_file=False,
                          mah_fold='run-wise', sigma=None,
                          shrinkage='ledoitwolf', return_rdm=True, verbose=False):
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
            - run-wise: each run is a fold
    
    Input:
        - beta maps per condition and run
        - mask
        - folding strategy
    Output:
        - similarity_map for each pairwise condition comparison
    '''
    # verbose = True
    # print(verbose)
    # load mask
    mask = nib.load(mask).get_fdata().astype(bool)
    

    print("Loading beta maps...")

    # Go over every run for the participant and load beta maps to session_and_run_dict
    for indx, entry in enumerate(session_and_run_dict):
        session = entry['session']
        run_N = entry['run']
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
            map_data = nib.load(input_file).get_fdata()
            # make sure map_data has same shape as mask
            if map_data.shape != mask.shape:
                raise ValueError(f"Beta map shape {map_data.shape} does not match mask shape {mask.shape}.")

            # store in session_and_run_dict
            session_and_run_dict[indx]['maps'][stim] = map_data
    print("All beta maps loaded successfully.")


    print("Preparing for crossnobis (cross-validated Mahalanobis)...")

    # prepare for searchlight
    ndim = mask.ndim # dimensions based on mask
    r = float(radius)
    rad = int(np.floor(r))

    # build integer offsets for an N-D sphere
    ranges = [np.arange(-rad, rad + 1) for _ in range(ndim)]
    grid = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1).reshape(-1, ndim)
    keep = (grid.astype(float) ** 2).sum(axis=1) <= r * r + 1e-12
    offsets = grid[keep]
    # initialize similarity_maps
    similarity_maps = {}
    for i, stim_i in enumerate(stim_types):
        for j, stim_j in enumerate(stim_types):
            if i >= j:
                continue  # avoid duplicates and self-comparison
            key = (stim_i, stim_j)
            similarity_maps[key] = np.full(mask.shape, np.nan, dtype=float)
    # iterate over all voxels in mask
    it = np.argwhere(mask)
    for center in it: # iterate over each sphere center
        # get neighboring voxels within radius
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
        # get indices of valid voxels
        indices = tuple(neigh.T)
        
        # prepare partitions based on mah_fold
        if len(session_and_run_dict) > 1:
            # multiple runs available
            if mah_fold == 'run-wise':
                # prepare data matrix Y (stim_types x voxels)
                Y_list, labels, partitions = [], [], []
                for run_idx, entry in enumerate(session_and_run_dict):
                    run_N = entry['run']
                    for stim_idx, stim in enumerate(stim_types):
                        map_data = entry['maps'][stim]
                        voxel_values = map_data[indices]
                        Y_list.append(voxel_values)
                        labels.append(stim)
                        partitions.append(run_N)
                Y = np.vstack(Y_list)
            else:
                raise ValueError(f"Unknown mah_fold strategy for multiple runs: {mah_fold}")
        else:
            # single run available
            raise NotImplementedError("Single run Mahalanobis calculation not implemented yet.")
        if verbose:
            # do anything
            a = 1
            #print(f"Calculating crossnobis at voxel {tuple(center)} with {Y.shape[0]} observations and {Y.shape[1]} features.")
        D = crossnobis(Y, labels, partitions, sigma=sigma, shrinkage=shrinkage, return_rdm=return_rdm)
        # store distances in similarity_maps
        for i, stim_i in enumerate(stim_types):
            for j, stim_j in enumerate(stim_types):
                if i >= j:
                    continue  # avoid duplicates and self-comparison
                key = (stim_i, stim_j)
                similarity_maps[key][tuple(neigh.T)] = D[i, j]
    print("Searchlight calculation completed.")
    print("Saving similarity maps...")
    ## Save similarity maps
    # for each session/run
    for entry in session_and_run_dict:
        session = entry['session']
        run_N = entry['run']
        # correct session to 2 digits
        session = f"{session:02d}"
        if verbose:
            print(f"session {session}, run {run_N} for sub-{sub_N:02d}...")
        # save similarity maps
        for i, stim_i in enumerate(stim_types):
            for j, stim_j in enumerate(stim_types):
                if i >= j:
                    continue  # avoid duplicates and self-comparison
                if verbose:
                    print(f"Saving similarity map for {stim_i} vs {stim_j}...")
                key = (stim_i, stim_j)
                # build output path
                output_file = os.path.join(
                    datafolder, dataset, 'results', 'RSA', model,
                    f"{specie}-sub-{sub_N:02d}",
                    f"ses-{session}_task-{task}_run-{run_N:02d}",
                    f"r-{radius}_mahalanobis_{stim_i}_{stim_j}.nii.gz"
                )
                # same but with stimuli inverted
                output_fileb = os.path.join(
                    datafolder, dataset, 'results', 'RSA', model,
                    f"{specie}-sub-{sub_N:02d}",
                    f"ses-{session}_task-{task}_run-{run_N:02d}",
                    f"r-{radius}_mahalanobis_{stim_j}_{stim_i}.nii.gz"
                )
                # create output directory if it doesn't exist
                output_dir = os.path.dirname(output_file)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                # save NIfTI
                affine = nib.load(input_file).affine  # use affine from last loaded map
                sim_map_nifti = nib.Nifti1Image(similarity_maps[key], affine)
                nib.save(sim_map_nifti, output_file)
                nib.save(sim_map_nifti, output_fileb)
                print(f"Saved similarity map: {output_file}")

def calculate_pairwise_similarity_maps(datafolder, dataset, sub_N, session, 
                                       run_N, specie, model, stim_types, mask, 
                                       task, radius=3, method='correlation', 
                                       replace_file=False, verbose=False):
    '''
    Calculate pairwise similarity maps between all stimulus types for a given subject/session/run.
    Parameters
    ----------
    datafolder : str
        Base directory for data storage.
    dataset : str
        Dataset name.
    sub_N : int


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
        # stim_1_name = stim_types[stim_N1]
        file_1_path = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'GLM' + os.sep + model + os.sep + f"{specie}-sub-{sub_N:02d}" + os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}.feat" + os.sep + 'stats' + os.sep + f"pe{(stim_N1+1)*2 - 1}.nii.gz"
        file_1_map = nib.load(file_1_path).get_fdata()
        
        # add to log
        log.append(f"Processing {method} similarity for stim {stim_1_name}, file: {file_1_path}")
        
        for stim_N2, stim_2_name in enumerate(stim_types):
            # if stim_N1 == stim_N2, skip
            if stim_N1 >= stim_N2:
                # print("Skipping identical stimuli...")
                continue                
            file_2_path = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'GLM' + os.sep + model + os.sep + f"{specie}-sub-{sub_N:02d}" + os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}.feat" + os.sep + 'stats' + os.sep + f"pe{(stim_N2+1)*2 - 1}.nii.gz"
            log.append(f"Comparing with stim {stim_N2}, file: {file_2_path}")
            # add to log
            # build output path
            output_path = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep + model + os.sep + f"{specie}-sub-{sub_N:02d}" + os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}" + os.sep + f"r-{radius}_{method}_{stim_1_name}_{stim_2_name}.nii.gz"
            # same but the order of the stims is inverted
            output_pathb = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep + model + os.sep + f"{specie}-sub-{sub_N:02d}" + os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}" + os.sep + f"r-{radius}_{method}_{stim_2_name}_{stim_1_name}.nii.gz"
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
                    log.append(f"    File {file_2_path} does not exist. Skipping...")
                continue
            
            file_2_map = nib.load(file_2_path).get_fdata()

            similarity_map = similarity_searchlight(file_1_map, file_2_map, nib.load(mask).get_fdata().astype(bool), radius=radius, method=method)
            # add to log
            log.append(f"    Saved similarity map to {output_path}")
            # check if output directory exists
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            nib.save(nib.Nifti1Image(similarity_map, nib.load(file_1_path).affine), output_path)
            # save the version with the stims inverted
            nib.save(nib.Nifti1Image(similarity_map, nib.load(file_1_path).affine), output_pathb)
            print(f"Saved {method} similarity map to {output_path}")
    if verbose:
        # print that we are done
        print(f"sub {sub_N:02d} session {session} run {run_N:02d} pairwise similarity computation done.")

    # save log
    output_folder = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 
                     'RSA' + os.sep + model + os.sep + f"{specie}-sub-{sub_N:02d}" + 
                     os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}")
    log_path = output_folder + os.sep + f"r-{radius}_{method}_log.txt"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    with open(log_path, 'w') as f:
        for line in log:
            f.write(line + '\n')


def calculate_group_model_similarity_map(datafolder, dataset, session_and_run_all_dict, specie, model,
                                          task, radius, rsa_model, rsa_method,
                                          method, replace_file, min_percentage_available=1.0, verbose=False):
    '''Calculate the group model similarity map.
    Inputs:
        datafolder: str. Path to data folder.
        dataset: str. Dataset name.
        specie: str. 'D' or 'H'.
        model: str. GLM model name.
        stim_types: list of str. Stimulus types.
        mask: str. Path to mask file.
        task: str. Task name.
        radius: int. Searchlight radius.
        rsa_model: str. Path to RSA model file.
        method: str. Similarity method.
        replace_file: bool. Whether to replace existing files.
        verbose: bool. Whether to print verbose output.
        min_percentage_available: float. Minimum percentage of available data to process.
    
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
        'task': task,
        'radius': radius,
        'rsa_model': rsa_model,
        'method': method,
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
    # check if output file already exists
    output = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                    model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                    f"{specie}-r-{radius}_{method}_{rsa_method}_mean.nii.gz")
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
    files_list = [] # list of files to process
    for sub_N in participants:
        session_and_run_dict = session_and_run_all_dict[sub_N]
        for entry in session_and_run_dict:
            session = entry['session']
            session = f"{session:02d}"
            run_N = entry['run']
            # check if model similarity map exists
            model_sim_map_file = os.path.join(
                datafolder, dataset, 'results', 'RSA', model, rsa_model,
                f"{specie}-sub-{sub_N:02d}", 
                f"ses-{session}_task-{task}_run-{run_N:02d}",
                f"r-{radius}_{method}_{rsa_method}.nii.gz"
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
    # check if enough files are available
    available_percentage = len(files_list) / (len(participants) * len(session_and_run_dict))
    if available_percentage < min_percentage_available:
        log.append(f"Only {available_percentage*100:.2f}% of data available. Minimum required is {min_percentage_available*100:.2f}%. Skipping group model similarity map calculation.")
        if verbose:
            print(log[-1])
        return
    else:
        log.append(f"{available_percentage*100:.2f}% of data available. Proceeding with group model similarity map calculation.")
        if verbose:
            print(log[-1])
    
    # determine mean_model_map_path
    mean_model_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                    model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                    f"{specie}-r-{radius}_{method}_{rsa_method}_mean.nii.gz")
    std_model_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                    model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                    f"{specie}-r-{radius}_{method}_{rsa_method}_std.nii.gz")
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
    nifti_mean(files_list, mean_model_map_path, std_model_map_path)
    print("Done computing mean and std model similarity maps.")
    ## update log_json with output files
    # if log_json already exists, delete and recreate
    if os.path.exists(log_json_output):
        os.remove(log_json_output)

    print(f"Saving log: {log_json_output}")
    with open(log_json_output, 'w') as f:
        yaml.dump(log_json, f)
    
def calculate_group_model_similarity_map_rnd(datafolder, dataset, session_and_run_all_dict, specie, model, 
                                            task, radius, rsa_model,
                                            rsa_method='pearson',
                                            method='pearson', verbose=False, 
                                            min_percentage_available=1.0,
                                            reps=1000, replace_rnd_files=False, wait_time=300,reps_group=1000
                                            ):
    participants = list(session_and_run_all_dict.keys())
    print("Checking for existing output files...")
    # check that outptut folder exists
    output_folder = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean')
    # create output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)
    
    # check if output file already exists
    for rnd_N in range(0, reps_group):
        # output file path
        mean_model_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                        f"{specie}-r-{radius}_{method}_{rsa_method}_mean_{rnd_N:05d}.nii.gz")
        # temp file path (same but _tmp.txt)
        mean_model_map_path_tmp = mean_model_map_path.replace('.nii.gz', '_tmp.txt')
        if os.path.exists(mean_model_map_path):
            if not replace_rnd_files:
                if verbose:
                    print(f"rnd {rnd_N:05d}/{reps_group:05d} exist, skipping...")
                continue
            else:
                if verbose:
                    print(f"rnd {rnd_N:05d}/{reps_group:05d} exist, replacing...")
        
        # check if temp file exists and how long ago it was modified
        if os.path.exists(mean_model_map_path_tmp):
            mod_time = os.path.getmtime(mean_model_map_path_tmp)
            elapsed_time = time() - mod_time
            if elapsed_time < wait_time:
                if verbose:
                    print(f"rnd {rnd_N:05d}/{reps_group:05d} skipping as it is being processed by another instance. Skipping...")
                continue
            else:
                if verbose:
                    print(f"rnd {rnd_N:05d}/{reps_group:05d} old temp found, calculating model similarity map...")
        else:
            if verbose:
                print(f"rnd {rnd_N:05d}/{reps_group:05d} temporal file created {mean_model_map_path_tmp}...")
            # create temp file
            with open(mean_model_map_path_tmp, 'w') as f:
                f.write(f"Temporary file for rnd {rnd_N:05d}\n")

        
        # build list of available model similarity maps
        files_list = [] # list of files to process
        for sub_N in participants:
            session_and_run_dict = session_and_run_all_dict[sub_N]
            for entry in session_and_run_dict:
                session = entry['session']
                session = f"{session:02d}"
                run_N = entry['run']
                # determine rnd_individual_N (rand sample from reps) for this subject/session/run
                rnd_individual_N = np.random.randint(0, reps)

                # build model similarity map file path
                model_sim_map_file = os.path.join(
                    datafolder, dataset, 'results', 'RSA_rnd', model, rsa_model,
                    f"{specie}-sub-{sub_N:02d}", 
                    f"ses-{session}_task-{task}_run-{run_N:02d}",
                    f"r-{radius}_{method}_{rsa_method}_{rnd_individual_N:04d}.nii.gz"
                )
                # make sure file exists
                if not os.path.exists(model_sim_map_file):
                    if verbose:
                        print(f"Not found skipping {model_sim_map_file}.")
                    continue
                
                files_list.append(model_sim_map_file)
        # check if enough files are available
        available_percentage = len(files_list) / (len(participants) * len(session_and_run_dict))
        if available_percentage < min_percentage_available:
            print(f"rnd {rnd_N:05d}/{reps_group:05d} not enough files available ({available_percentage*100:.2f}%). Needed {min_percentage_available*100:.2f}%. Skipping...")
            # remove temp file
            if os.path.exists(mean_model_map_path_tmp):
                os.remove(mean_model_map_path_tmp)
            continue
        else:
            print(f"rnd {rnd_N:05d}/{reps_group:05d} processing {len(files_list)} files ({available_percentage*100:.2f}% available)...")
        try:
            # calculate group model similarity map
            nifti_mean(files_list, result_map_path=mean_model_map_path, verbose=False)
        except Exception as e:
            print(f"Error {e} calculating group model similarity map for rnd {rnd_N:05d}. Skipping...")
        # remove temp file
        if os.path.exists(mean_model_map_path_tmp):
            os.remove(mean_model_map_path_tmp)
            
def calculate_voxelwise_rnd_distribution(datafolder, dataset, specie, model, task, radius,
                                    method='pearson', rsa_method='pearson',
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
    - method: method for pairwise similarity calculation
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
                        f"{specie}-r-{radius}_{method}_{rsa_method}_mean_{rnd_N:05d}.nii.gz")
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

def calculate_z_map_real_data(datafolder, dataset, specie, model, radius,
                              method, rsa_method, rsa_model, verbose=False):

    # paths
    distribution_mean_map_path = os.path.join(datafolder, dataset, 'results', 'RSA_rnd',
                                              model, f'{specie}-{rsa_model}_mean.nii.gz')
    distribution_std_map_path  = os.path.join(datafolder, dataset, 'results', 'RSA_rnd',
                                              model, f'{specie}-{rsa_model}_std.nii.gz')

    group_mean_map_path = os.path.join(datafolder, dataset, 'results', 'RSA',
                                       model, rsa_model, 'mean',
                                       f'{specie}-r-{radius}_{method}_{rsa_method}_mean.nii.gz')

    group_z_map_path = os.path.join(datafolder, dataset, 'results', 'RSA',
                                    model, rsa_model, 'mean',
                                    f'{specie}-r-{radius}_{method}_{rsa_method}_z.nii.gz')

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


def calculate_z_maps_rnd(datafolder, dataset, specie, model, task, radius,
                                    method='pearson', rsa_method='pearson',
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
                        f"{specie}-r-{radius}_{method}_{rsa_method}_mean_{rnd_N:05d}.nii.gz")
        group_z_map_path = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep +
                        model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                        f"{specie}-r-{radius}_{method}_{rsa_method}_z_{rnd_N:05d}.nii.gz")
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
    method, rsa_method, z_threshold=3.1,
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
    - method: similarity method for pairwise similarity calculation
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
                        f"{specie}-r-{radius}_{method}_{rsa_method}_dist.npy")
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

    rsa_model_path = datafolder + os.sep + dataset + os.sep + 'rsa_models' + os.sep + rsa_model + ".xlsx"
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


    ## Calculate z map for permutations (RSA_rnd folder)
    # check how many rnd mean files are available
    available_files = []
    missing_files = []

    # find all files that match pattern (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep +
                        # model + os.sep + rsa_model + os.sep +
                        # f"r-{radius}_{method}_{rsa_method}_z_*.nii.gz")
    search_query = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep +
                    model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                    f"{specie}-r-{radius}_{method}_{rsa_method}_z_*.nii.gz")

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

            circ = patches.Circle((x, y), radius=r, facecolor=face, edgecolor=edgecolor, linewidth=linewidth)
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
        # indicate where it was saved
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

def extract_clusters_and_peaks(nifti_path, stat_thresh=None, min_dist_mm=8.0, max_peaks_per_cluster=3):
    img = nib.load(nifti_path)
    stat = img.get_fdata()
    affine = img.affine

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
        # gather cluster-level descriptors
        cluster_size = int(cluster_mask.sum())
        cluster_max = max(peaks, key=lambda t: t[0]) if peaks else None
        results.append({
            "cluster_id": c,
            "size_vox": cluster_size,
            "peaks": [{"Z": z, "ijk": ijk, "xyz_mm": xyz} for (z, ijk, xyz) in peaks],
            "peak_Z": cluster_max[0] if cluster_max else None,
            "peak_xyz_mm": cluster_max[2] if cluster_max else None
        })
    return results

def clusters_to_excel(results, out_path):
    """
    Build a hierarchical (cluster -> subpeak) table from `results`
    and save it as an Excel file.
    """
    rows = []
    # print('hit new 2')
    for cluster in results:
        cid = cluster['cluster_id']
        size = cluster['size_vox']
        peak_Z_cluster = cluster['peak_Z']
        peak_xyz_cluster = cluster['peak_xyz_mm']

        for sub_idx, peak in enumerate(cluster['peaks'], start=1):
            i, j, k = peak['ijk']
            x_mm, y_mm, z_mm = peak['xyz_mm']

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
            })
            # print each element in rows
            # for key, value in rows[-1].items():
            #     print(f"{key}: {value}")

    df = pd.DataFrame(rows)
    # Make it hierarchical: first cluster, then subpeak
    df = df.set_index(['cluster_id', 'subpeak_id']).sort_index()

    df.to_excel(out_path)
    return df

def create_tables(datafolder, dataset, specie, model, rsa_model, radius, 
                  method, rsa_method, min_dist_mm=8.0, max_peaks_per_cluster=3):
    res_image = (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                f"{specie}-r-{radius}_{method}_{rsa_method}_z_corrected.nii.gz"
                )
    
    # res_image = r"P:\userdata\raulh87\data\EmoB\results\RSA\basic-block\old_emotion-valence\mean\D-r-3_mahalanobis_kendall_z_corrected.nii.gz"
    results = extract_clusters_and_peaks(res_image, stat_thresh=None, min_dist_mm=min_dist_mm, max_peaks_per_cluster=max_peaks_per_cluster)
    out_path =  (datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep +
                model + os.sep + rsa_model + os.sep + 'mean' + os.sep +
                f"{specie}-r-{radius}_{method}_{rsa_method}.xlsx")
    clusters_to_excel(results, out_path)
    print(f"Files written in: {out_path}")

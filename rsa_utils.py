import os
import nibabel as nib
import yaml
import pandas as pd
import numpy as np
# import random
import random
from scipy.ndimage import label, generate_binary_structure

import numpy as np
from scipy import ndimage

def apply_cluster_size_threshold(
    z_map_thresholded: np.ndarray,
    minimal_cluster_size: int,
    connectivity: int = 6
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
    connectivity : int
        Neighborhood connectivity in 3D. One of:
          - ndimage convention 1..3  (1≈6-neigh, 2≈18, 3≈26)
          - Common shorthands: 6, 18, or 26

    Returns
    -------
    np.ndarray
        Copy of the input with small clusters set to 0.

    Raises
    ------
    ValueError
        If the input is not 3D, or if connectivity is invalid, or if p is out of range.
    """
    arr = np.asarray(z_map_thresholded)
    if arr.ndim != 3:
        raise ValueError(f"apply_cluster_size_threshold requires a 3D array, got ndim={arr.ndim}.")

    if minimal_cluster_size <= 1:
        raise ValueError("minimal_cluster_size must be > 1.")

    # Normalize connectivity to ndimage's 1..3 scale
    if 1 <= connectivity <= 3:
        conn = connectivity
    elif connectivity in (6, 18, 26):
        conn = {6: 1, 18: 2, 26: 3}[connectivity]
    else:
        raise ValueError("Invalid connectivity for 3D. Use 1..3 or 6/18/26.")

    if minimal_cluster_size == 1:
        # Nothing to remove—every 1-voxel cluster survives
        return arr.copy()

    structure = ndimage.generate_binary_structure(rank=3, connectivity=conn)

    # Foreground is strictly > 0 to match your thresholding
    fg = arr > 0

    labels, nlab = ndimage.label(fg, structure=structure)
    if nlab == 0:
        return arr.copy()

    # Count sizes per label (label 0 is background)
    label_ids, counts = np.unique(labels, return_counts=True)

    # Labels to drop: exclude background (0)
    small_labels = label_ids[(label_ids != 0) & (counts < minimal_cluster_size)]
    if small_labels.size == 0:
        return arr.copy()

    out = arr.copy()
    out[np.isin(labels, small_labels)] = 0
    return out


def get_minimal_cluster_size(cluster_sizes_path: str, p_thr: float) -> int:
    """
    Compute the minimal cluster-size threshold for cluster-extent correction.

    Parameters
    ----------
    cluster_sizes_path : str
        Path to a .npy file containing an array of arrays (or lists), where each inner
        array holds ALL cluster sizes found for a single permutation.
    p_thr : float
        Target tail probability (e.g., 0.05). Returns the smallest integer k such that
        the empirical probability of observing a cluster of size >= k under the null
        (across permutations) is <= p_thr.

    Returns
    -------
    int
        Minimal cluster size threshold (integer).
    """
    if not (0 < p_thr < 1):
        raise ValueError("p_thr must be in (0, 1).")

    sizes_list = np.load(cluster_sizes_path, allow_pickle=True)

    # Normalize to list-of-arrays
    if isinstance(sizes_list, np.ndarray) and sizes_list.dtype == object:
        permutations = [np.asarray(x).astype(float).ravel() if x is not None else np.array([])
                        for x in sizes_list]
    elif isinstance(sizes_list, (list, tuple)):
        permutations = [np.asarray(x).astype(float).ravel() if x is not None else np.array([])
                        for x in sizes_list]
    else:
        permutations = [np.asarray(x).astype(float).ravel() for x in sizes_list]

    if len(permutations) == 0:
        raise ValueError("No permutations found in the provided file.")

    # Per-permutation maximum cluster size (0 if no clusters in that permutation)
    max_sizes = np.array([float(np.max(p)) if p.size > 0 else 0.0 for p in permutations], dtype=float)
    N = max_sizes.size
    if N == 0:
        raise ValueError("No valid permutations after parsing.")

    # Sort ascending; find the first index with tail prob <= p_thr
    m_sorted = np.sort(max_sizes)
    i0 = int(np.ceil((1.0 - p_thr) * N))

    # If p is extremely small, pick > max to get tail 0
    if i0 >= N:
        return int(np.floor(m_sorted[-1])) + 1

    k = m_sorted[i0]
    # Return smallest integer threshold meeting the criterion (conservative ceil)
    minimal_cluster_size = int(np.ceil(k))

    # Guard in case ceil bumped us into a slightly stricter tail
    while np.mean(max_sizes >= minimal_cluster_size) > p_thr:
        minimal_cluster_size += 1

    return minimal_cluster_size


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


def compare_with_model(ref_img, mask_affine, datafolder, sub_N, session, run_N, specie, model, dataset, task, mask_type, radius, rsa_model, method='pearson', rsa_method='pearson', replace_file=False, verbose=False, rnd=False, reps=1000):
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

    
    # create an result_map based on the reference image
    result_map = np.zeros(ref_img.shape)
    
    meta_similarity_map = load_meta_similarity_map(rsa_model_path, ref_img, datafolder, dataset, sub_N, session, run_N, config_path, method=method, radius=radius, verbose=verbose)
    # create similarity_table (x, y, z) of all voxels in the mask, results will be added here
    similarity_table = np.column_stack(np.where(ref_img > 0))
    # add 1 to x, y, z to match 1-based indexing in itk-snap
    similarity_table += 1
    # add a column for similarity values, initialized to NaN
    similarity_table = np.hstack((similarity_table, np.full((similarity_table.shape[0], 1), np.nan)))
    # initialize warning table
    warning_table = []
    # create blank 

    for rnd_N in range(0, reps):
        if rnd:
            # if rnd is True, permute the model values
            # rsa_model_dict = shuffle_model(rsa_model_dict)
            output_folder = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA_rnd' + os.sep + model + os.sep + rsa_model + os.sep +  f"{specie}-sub-{sub_N:02d}" + os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}"
            # build output filename _[4 digit padded rnd_N]
            output_file = os.path.join(output_folder, f"r-{radius}_{method}_{rsa_method}_{rnd_N:04d}.nii.gz")
            model_vector = shuffle_vector(model_vector)
        else:
            if rnd_N > 0:
                print("real data, skipping further repetitions")
                break
            output_folder = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep + model + os.sep + rsa_model + os.sep +  f"{specie}-sub-{sub_N:02d}" + os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}"
            # build output filename
            output_file = os.path.join(output_folder, f"r-{radius}_{method}_{rsa_method}.nii.gz")
        # check if output_file exists
        if os.path.exists(output_file) and not replace_file:
            print(f"Output file {output_file} already exists. Skipping...")
            return output_file, True

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

    def _mahalanobis(x, y):
        # X has 2 samples (the two maps), features=len(sphere)
        X = np.vstack([x, y])  # shape: (2, K)
        diff = X[0] - X[1]
        K = diff.size
        # try Ledoit–Wolf if available
        VI = None
        try:
            from sklearn.covariance import LedoitWolf
            # LedoitWolf returns covariance over features
            lw = LedoitWolf().fit(X)
            cov = lw.covariance_
            # safety net: if cov has inf/nan, fall back to ridge
            if not np.all(np.isfinite(cov)):
                raise ValueError("non-finite covariance")
            VI = np.linalg.pinv(cov, hermitian=True)
        except Exception:
            # fallback: sample covariance with a small ridge
            # sample cov over features with rowvar=False
            cov = np.cov(X, rowvar=False)  # shape (K,K), rank-deficient for 2 samples
            if np.ndim(cov) == 0:  # K==1 edge case -> scalar
                cov = np.array([[float(cov)]])
            # ridge λ scaled to average variance (trace/K)
            tr = float(np.trace(cov)) if np.isfinite(np.trace(cov)) else 0.0
            lam = 1e-3 * (tr / max(K, 1) + 1.0)  # small positive
            cov = cov + lam * np.eye(K)
            try:
                VI = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                VI = np.linalg.pinv(cov, hermitian=True)
        d2 = float(diff @ VI @ diff)   # squared Mahalanobis distance
        return -np.sqrt(d2)            # return negative distance (higher = more similar)

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
                val = _mahalanobis(x, y)
        similarity_map[tuple(center)] = val

    return similarity_map

import itertools
import random
import math

def shuffle_vector(vector):
    L = len(vector)
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
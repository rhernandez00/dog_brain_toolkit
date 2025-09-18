import os
import nibabel as nib
import yaml
import pandas as pd
import numpy as np

import numpy as np

# def check_file_status
#"P:\userdata\raulh87\data\EmoB\results\RSA\basic\emotion_valence\D-sub-01\ses-01_task-EmoB_run-01\r-3_pearson_kendall.nii.gz"

def nifti_mean(img_list, result_map_path=None):
    """
    Compute the voxel-wise mean of a list of NIfTI images.

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
    """
    if len(img_list) == 0:
        raise ValueError("img_list is empty.")

    # Load the first image to get the shape and affine
    first_img = nib.load(img_list[0])
    img_shape = first_img.shape
    img_affine = first_img.affine

    # Initialize an array to hold the sum
    sum_data = np.zeros(img_shape, dtype=np.float64)
    count = 0

    for img_path in img_list:
        img = nib.load(img_path)
        if img.shape != img_shape:
            raise ValueError(f"Image {img_path} has a different shape: {img.shape} != {img_shape}")
        sum_data += img.get_fdata(dtype=np.float64)
        count += 1

    mean_data = sum_data / count

    if result_map_path:
        mean_img = nib.Nifti1Image(mean_data, img_affine)
        nib.save(mean_img, result_map_path)

    return mean_data

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


def compare_with_model(ref_img, mask_affine, datafolder, sub_N, session, run_N, specie, model, dataset, task, mask_type, radius, rsa_model, method='pearson', rsa_method='pearson', replace_file=False, verbose=False):


    rsa_model_path = datafolder + os.sep + dataset + os.sep + 'rsa_models' + os.sep + rsa_model + ".xlsx"
    config_path = datafolder + os.sep + dataset + os.sep + 'config_files' + os.sep + model + '.yaml'
    
    # "P:\userdata\raulh87\data\EmoB\results\RSA\basic"
    output_folder = datafolder + os.sep + dataset + os.sep + 'results' + os.sep + 'RSA' + os.sep + model + os.sep + rsa_model + os.sep +  f"{specie}-sub-{sub_N:02d}" + os.sep + f"ses-{session}_task-{task}_run-{run_N:02d}"
    output_file = os.path.join(output_folder, f"r-{radius}_{method}_{rsa_method}.nii.gz")

    # check if output_file exists
    if os.path.exists(output_file) and not replace_file:
        print(f"Output file {output_file} already exists. Skipping...")
        return output_file, True

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
    if method not in {"mahalanobis", "pearson", "euclidean", "kendall"}:
        raise ValueError("method must be one of: 'mahalanobis','pearson','euclidean','kendall'")

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
            elif method == "kendall":
                val = _kendall(x, y)
            elif method == "euclidean":
                val = _euclidean(x, y)
            elif method == "mahalanobis":
                val = _mahalanobis(x, y)
        similarity_map[tuple(center)] = val

    return similarity_map
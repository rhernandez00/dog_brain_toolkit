# Some functions used across the project
# Author: Raul Hernandez

import os
import shutil
import subprocess
import pandas as pd

import json
import numpy as np

import json
import numpy as np
from pathlib import Path
import re


# def get_project_dict(config):
#     #
#     fields_to_add = [
#         'Project', 'User', 'Dataset', 'Task', 'Participants',
#         'Runs', 'Sessions', 'Specie', 'Atlas_type', 'Datafolder'
#     ]
#     # initialize project_dict
#     project_dict = {}
#     # add fields if they are in config
#     for field in fields_to_add:
#         if field not in config:
#             print(f"Field {field} not in config, please add it.")
#             continue
#         else:
#             project_dict[field] = config[field]
#     return project_dict

def run_visual_model(datafolder, database, sub_ID, session, run_N, task, smooth, 
                     slice_timming_path, input_nifti, anat_file, cond_file, 
                     design_template, design_out, replace_existing=False):
    
    TR, volumes = extract_params(input_nifti)
    #fsl_out = "C:\data\EmoB\results\GLM\visual\D-sub-01.gfeat"
    # "P:\userdata\raulh87\data\EmoB\results\GLM\visual\H-01\ses-01_task-EmoB_run-01.feat"
    fsl_out = (datafolder + os.sep + database + os.sep + "results" + os.sep + "GLM" + os.sep + 
               "visual" + os.sep + sub_ID + os.sep + f"ses-01_task-{database}_run-{str(run_N).zfill(2)}" + '.feat')
    # check if fsl_out exists
    if os.path.exists(fsl_out) and not replace_existing:
        print(f"Output folder {fsl_out} already exists, skipping...")
        return
    else:
        print(f"Output folder {fsl_out} does not exist, proceeding...")

    labels = {
                'outputdir': (fsl_out,        'set fmri(outputdir)'),
                'TR':        (TR,             'set fmri(tr)'),
                'volumes':   (volumes,        'set fmri(npts)'),
                'smooth':    (smooth,         'set fmri(smooth)'),
                'slice_timing': (slice_timming_path, 'set fmri(st_file)'),
                'input':     (input_nifti,    'set feat_files(1)'),
                'anatomical': (anat_file,      'set highres_files(1)'),
                'condition': (cond_file,      'set fmri(custom1)'),
            }
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
    print(f"Template: {design_template}")
    print(f"Output design: {design_out}")
    # Fill FSF and run FSL
    fill_fsf(to_fill, design_template, design_out)
    # Remove existing .feat dir if present
    if os.path.exists(fsl_out + '.feat'):
        shutil.rmtree(fsl_out + '.feat')
    cmd = f'feat {design_out}'
    print(f"Running: {cmd}")
    if os.name != 'nt':
        os.system(cmd)


def get_path(path_label, project_dict, local_data=True, rnd=False, figure_letter='A'):
    
    # if project_dict has no experiment, set it to 'Segmentation'
    if 'Project' not in project_dict:
        project = 'Segmentation'
    else:
        project = project_dict['Project']
    # if project_dict has no User, dont add it
    if 'User' in project_dict:
        username = project_dict['User']
    # same for dataset
    if 'Dataset' in project_dict:
        dataset = project_dict['Dataset']
    # if dataset == 'CAPS_K9':
    #     dataset = 'Segmentation'
    
    if path_label == 'results' or path_label == 'Results':
        if os.name == 'nt':
            if local_data:
                path = r"G:\My Drive\Results\\" + project
            else:
                # username
                path = r"P:\userdata\\" + username + r"\data" + os.sep + dataset + os.sep + 'results'
        else:
            path =  '/home' + os.sep + username + '/mnt/a471/userdata/' + username + '/data' + os.sep + dataset + os.sep + 'results'
        
        if rnd:
            path = os.path.join(path, 'rnd')
    # figures or Figures
    elif path_label == 'figures' or path_label == 'Figures':
        # G:\My Drive\[project]\Figures\Figure_[figure_letter]
        # path = 
        path = r"C:\github" + os.sep + project + os.sep + "Figures" + os.sep + f"Figure_{figure_letter}"


        # path = r"G:\My Drive\\" + project + r"\Figures" + os.sep + f"Figure_{figure_letter}"
        
    # project or Project
    elif path_label == 'project' or path_label == 'Project':
        if os.name == 'nt':
            if local_data:
                path = r"G:\My Drive\\" + project
            else:
                path = r"P:\userdata\\" + username + r"\data" + os.sep + project
        else:
            path = '/home' + os.sep + username + '/mnt/a471/userdata/' + username + os.sep + project
    elif path_label == 'datafolder' or path_label == 'Datafolder':
        if os.name == 'nt':
            if local_data:
                path = r"C:\data"
            else:
                path = r"P:\userdata\raulh87\data"
        else:
            path = '/home' + os.sep + username + '/mnt/a471/userdata/' + username + os.sep + 'data'
    elif path_label == 'git_folder':
        if os.name == 'nt':
            path = r"C:\github"
        else:
            path = '/home' + os.sep + username + '/mnt/a471/userdata/' + username + os.sep + 'github'
    else:
        print('Path label not recognized')
    return path

def generate_fsf(n: int, template_path: str, outfile_path: str) -> None:
    from pathlib import Path
    import re

    template_lines = Path(template_path).read_text(encoding="utf-8").splitlines()

    out_parts = []

    # Header: lines 1..276 (1-based) — includes the single confound file line at 275
    head = "\n".join(template_lines[:276]) + "\n"
    head = re.sub(r"\bNUM\b", str(n), head).replace("NUM2", str(n * 2))
    out_parts.append(head)

    # Per-EV block: lines 277..316 (1-based)
    block_tpl = "\n".join(template_lines[276:316]) + "\n"
    for i in range(1, n + 1):
        con = f"cond{i:03d}"
        block = block_tpl.replace("NUM", str(i)).replace("CONDITION", con)
        out_parts.append(block)

        # Orthogonalization (unchanged)
        for ii in range(0, n + 1):
            out_parts.append(f"# Orthogonalise EV {i} wrt EV {ii}\n")
            out_parts.append(f"set fmri(ortho{i}.{ii}) 0\n\n")

    # Contrast sections + masking (unchanged)
    out_parts.append("# Contrast & F-tests mode\n# real : control real EVs\n# orig : control original EVs\n")
    out_parts.append("set fmri(con_mode_old) orig\nset fmri(con_mode) orig\n\n")

    contrastnum = n * 2
    c = 1
    for i in range(1, n + 1):
        con = f"cond{i:03d}"
        out_parts.append(f"# Display images for contrast_real {i}\nset fmri(conpic_real.{i}) 1\n\n")
        out_parts.append(f"# Title for contrast_real {i}\nset fmri(conname_real.{i}) \"{con}\"\n\n")
        for ii in range(1, contrastnum + 1):
            out_parts.append(f"# Real contrast_real vector {i} element {ii}\nset fmri(con_real{i}.{ii}) {'1.0' if c == ii else '0'}\n\n")
        c += 2

    for i in range(1, n + 1):
        con = f"cond{i:03d}"
        out_parts.append(f"# Display images for contrast_orig {i}\nset fmri(conpic_orig.{i}) 1\n\n")
        out_parts.append(f"# Title for contrast_orig {i}\nset fmri(conname_orig.{i}) \"{con}\"\n\n")
        for ii in range(1, n + 1):
            out_parts.append(f"# Real contrast_orig vector {i} element {ii}\nset fmri(con_orig{i}.{ii}) {'1.0' if i == ii else '0'}\n\n")

    out_parts.append("# Contrast masking - use >0 instead of thresholding?\nset fmri(conmask_zerothresh_yn) 0\n\n")
    for i in range(1, n + 1):
        for ii in range(1, n + 1):
            if i != ii:
                out_parts.append(f"# Mask real contrast/F-test {i} with real contrast/F-test {ii}?\nset fmri(conmask{i}_{ii}) 0\n\n")

    # Tail: lines 310..end (1-based)
    out_parts.append("\n".join(template_lines[309:]) + "\n")

    Path(outfile_path).write_text("".join(out_parts), encoding="utf-8")


def run_individual_GLM(project_dict, model, sub_N, session_and_run_list):
    """
    Run individual GLM analysis for the specified project and parameters.
    
    Args:
        project_dict (dict): Project configuration dictionary.
        model (str): Model name.
        sub_N (int): Subject number.
        session_and_run_list (list): List of dictionaries with session and run information.
    """
    # Ensure the data folder exists
    if not os.path.exists(project_dict['Datafolder']):
        os.makedirs(project_dict['Datafolder'])

    # determine N_runs based on the session and run list
    N_runs = len(session_and_run_list)

    # Determine data directory based on OS
    if os.name == 'nt':  # Windows
        datafolder = os.path.join(
            "P:\\userdata", project_dict['User'], 'data'
        )
    else:
        datafolder = os.path.join(
            '/home', project_dict['User'], 'mnt', 'a471', 'userdata', project_dict['User'], 'data'
        )
    # project_dict['Datafolder'] = datafolder
    print(f"Data folder: {datafolder}")

    specie = project_dict['Specie']
    dataset = project_dict['Dataset']
    atlas_type = project_dict['Atlas_type']
    task = project_dict['Task']
    # Species label for atlas subfolder
    specie_label = 'Dog' if specie == 'D' else 'Hum'
    img_type = 'brain2mm'
    # Atlas file
    atlas_file = os.path.join(
        os.getcwd(), 'Atlas', specie_label, atlas_type, f"{img_type}.nii.gz"
    )

    # Output directory for FSL
    fsl_out = os.path.join(
        datafolder, dataset, 'results', 'GLM', model,
        f"{specie}-sub-{sub_N:02d}"
    )

    # Prepare FSF template replacement dictionary
    design_in = os.path.join(datafolder, dataset, 'FSL_designs', 'individual_' + str(N_runs) + '.fsf')
    design_out = os.path.join(datafolder, dataset, 'FSL_designs', 'individual_' + str(N_runs) +  '_sub-' + str(sub_N).zfill(2) + '_modified.fsf')


    labels_to_replace = ['outputdir', 'atlas']
    # add 'input#' based on the number of runs
    for i in range(1, N_runs + 1):
        labels_to_replace.append(f'input{i}')


    input_feat_list = []
    for session_and_run in session_and_run_list:
        session = session_and_run['session']
        run = session_and_run['run']
        # input = "P:\userdata\raulh87\data\EmoB\results\GLM\visual\D-sub-01\ses-01_task-EmoB_run-01.feat"
        input = os.path.join(
            datafolder, dataset, 'results', 'GLM', model,
            f"{specie}-sub-{sub_N:02d}",
            f"ses-{session:02d}_task-{task}_run-{run:02d}.feat"
        )
        # Append to list
        input_feat_list.append(input)
        
    labels = {
        'outputdir': (fsl_out,        'set fmri(outputdir)'),
        'atlas':     (atlas_file,     'set fmri(regstandard)'),
    }
    # Add inputs to labels
    for i, input_feat in enumerate(input_feat_list, start=1):
        labels[f'input{i}'] = (input_feat, f'set feat_files({i})')
        #set feat_files(1)

    print(labels)
    # Build replacement dict
    to_fill = {}
    for key, (val, find_str) in labels.items():
        rep = f'set {find_str.split()[1]}("{val}"' if isinstance(val, str) else f'set {find_str.split()[1]} {val}'
        # Actually ensure correct format
        if key in labels_to_replace:
            rep = f'{find_str} "{val}"'
        else:
            rep = f'{find_str} {val}'
        to_fill[key] = {
            'string_to_find': find_str,
            'string_to_replace': rep
        }

    # Fill FSF and run FSL
    fill_fsf(to_fill, design_in, design_out)

    # Remove existing .feat dir if present
    if os.path.exists(fsl_out + '.feat'):
        shutil.rmtree(fsl_out + '.feat')

    cmd = f'feat {design_out}'
    print(f"Running: {cmd}")
    if os.name != 'nt':
        os.system(cmd)
    return labels_to_replace

def get_slice_timing(json_path: str, txt_out_path: str):
    """
    Load a BIDS JSON sidecar, extract the 'SliceTiming' field, write each
    timing value to a text file (one per line), and return as a NumPy array.

    Parameters
    ----------
    json_path : str
        Path to the input JSON file.
    txt_out_path : str
        Path to the output text file where timings will be written.

    Returns
    -------
    np.ndarray
        1D array of slice timing values.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    KeyError
        If the 'SliceTiming' key is missing.
    ValueError
        If the 'SliceTiming' value is not a list of numbers.
    """
    # Load JSON sidecar
    with open(json_path, 'r') as f:
        info = json.load(f)

    # Extract and validate SliceTiming
    try:
        timings = info['SliceTiming']
    except KeyError:
        raise KeyError(f"'SliceTiming' not found in {json_path}")

    if not isinstance(timings, list) or not all(isinstance(t, (int, float)) for t in timings):
        raise ValueError(f"'SliceTiming' in {json_path} must be a list of numbers")

    # Convert to NumPy array
    arr = np.array(timings, dtype=float)

    # Write one value per line to the output text file
    np.savetxt(txt_out_path, arr, fmt='%g')

    return arr



def apply_flirt(input_file, output_file, reference_file, parameters):
    """
    This function applies a transformation matrix to an input file using flirt
    
    Parameters:
    - input_file (str): The path to the input file.
    - output_file (str): The path to the output file.
    - reference_file (str): The path to the reference file.
    - parameters (dict): A dictionary containing the parameters for flirt x_min, x_max, y_min, y_max, z_min, z_max
    
    Returns:
    - None
    """
    x_min = parameters['x_min']
    x_max = parameters['x_max']
    y_min = parameters['y_min']
    y_max = parameters['y_max']
    z_min = parameters['z_min']
    z_max = parameters['z_max']
    # Construct the command to run
    command = f"flirt -in {input_file} -ref {reference_file} -out {output_file} -omat {output_file.replace('.nii.gz', '.mat')} -bins 256 -cost corratio -searchrx {x_min} {x_max} -searchry {y_min} {y_max} -searchrz {z_min} {z_max} -dof 12 -interp trilinear"
    print(command)
    if os.name != 'nt':
        os.system(command)

def job_list_to_table(job_list):
    # Define column_id dictionary with keys for columns and column labels
    column_id = {'sub_N': {'label': 'Participant'},
                 'run_N': {'label': 'Run'},
                 'Process': {'label': 'Process'},
                 'Status': {'label': 'Status'},
                 'Atlas_type': {'label': 'Atlas_type'},
                 'Dataset': {'label': 'Dataset'},
                 'Full_prepro': {'label': 'Full_prepro'},
                 'first_time': {'label': 'First time'},
                 'use_anatomic': {'label': 'Use anatomical for registration'},
                 'Combination': {'label': 'Combination'},
                 'session_and_run': {'label': 'session_and_run'},
                 }

    # Initialize schedule_table
    schedule_table = pd.DataFrame(columns=[column_id[key]['label'] for key in column_id.keys()])

    # List to hold new rows
    rows = []

    # Populate schedule_table with job_list
    for job in job_list:
        new_row = {column_id[key]['label']: job[key] for key in column_id.keys()}
        rows.append(new_row)

    # Convert list of rows to DataFrame and concatenate
    schedule_table = pd.concat([schedule_table, pd.DataFrame(rows)], ignore_index=True)

    return schedule_table

def fill_fsf(to_fill_dict, design_path, design_modified_path):
    """"
    This function fills in the design.fsf file with new values and saves a copy
    # Arguments
    to_fill_dict: dictionary with labels and their new values
    design_path: path to the original design.fsf file
    design_modified_path: path to the new design.fsf file
    """
    # make sure the design_modified_path directory exists
    os.makedirs(os.path.dirname(design_modified_path), exist_ok=True)

    # create a copy of the design file and save it as design_copy.fsf
    shutil.copy(design_path, design_modified_path)
    

    # Open design.fsf file and read lines
    with open(design_modified_path, 'r') as file:
        lines = file.readlines()

    # Prepare a list to hold modified lines
    modified_lines = []

    # Replace the entire line if the target string is found
    for line in lines:
        found = False
        for label in to_fill_dict:
            if to_fill_dict[label]['string_to_find'] in line:
                # Replace the entire line
                modified_lines.append(to_fill_dict[label]['string_to_replace'] + '\n')
                found = True
                break  # Assume each line only needs one replacement and break for efficiency
        if not found:
            # If no replacement was made, keep the original line
            modified_lines.append(line)

    # Write the modified lines back to the design.fsf file or a new file
    with open(design_modified_path, 'w') as file:
        file.writelines(modified_lines)
    # print name of output file
    print('Output design file: ' + design_modified_path)
    
def extract_params(input_file):
    """
    Gets the TR and number of volumes of a file using fslinfo
    
    Parameters:
    - input_file (str): The path to the input file for the fslinfo command.
    
    Returns:
    TR - TR
    volumes - number of volumes
    
    """
    # Check if the os is windows, if it is means that this is a test without actually running
    if os.name == 'nt':
        print('This is a test, no actual fslinfo command will be run, giving back random values')
        return 2.0, 100

    # Construct the command to run
    command = f"fslinfo {input_file}"
    
    # Run the command and capture the output
    try:
        # subprocess.check_output returns the output of the command
        output = subprocess.check_output(command, shell=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run '{command}': {e}")
        return None
    
    # Split the output into lines
    lines = output.split('\n')
    
    # Loop through each line
    for line in lines:
        # Check if the line contains 'dim4'
        if 'dim4' in line:
            # Split the line by spaces and extract the value after 'dim4'
            parts = line.split()
            # Assuming the value is always after 'dim4', which is at index 1
            volumes = int(parts[1])  # Convert the value to float
            break
    for line in lines:
        if 'pixdim4' in line:
            # Split the line by spaces and extract the value after 'pixdim4'
            parts2 = line.split()
            # Assuming the value is always after 'dim4', which is at index 1
            TR = float(parts2[1])  # Convert the value to float
            break
    
    return TR,volumes

def reorient_file(input_file, output_file, combination):
    # This function reorients the input file using fsl and the combination of rotations
    """"
    input_file: str, path to the input file
    output_file: str, path to the output file
    combination: list of strings, combination of rotations to apply (e.g. ['-x', 'z', '-y'])
    """
    # check if the os is windows
    if os.name == 'nt':
        print('The system is windows, this is a test, no actual fsl commands will be run')

    print('Working with ' + input_file)
    # create copy of input file to output file
    print('Creating ' + output_file + '...')
    if os.name != 'nt': # the system is not windows, create the copy
        shutil.copyfile(input_file, output_file)
    
    # delete orientation
    print('Deleting orientation...')
    command = f"fslorient -deleteorient {output_file}"
    # if the system is windows, print the command, if not, run it
    print(command)
    if os.name != 'nt':
        os.system(command)
    # swap axes
    print('Swapping axes...')
    command = f"fslswapdim {output_file} {combination[0]} {combination[1]} {combination[2]} {output_file}"
    print(command)
    if os.name != 'nt':
        os.system(command)
    # adding labels
    print('Adding labels...')
    command = f"fslorient -setqformcode 1 -setqformcode 1 {output_file}"
    print(command)
    if os.name != 'nt':
        os.system(command)

    print('Reorientation done!')

def write_params_file(params_file, params_dict):
    '''
    the function will write a txt file with the parameters to be loaded in a bash script
    params_dict: dictionary with the parameters to be written
    '''

    with open(params_file, 'w') as f:
        for key in params_dict:
            f.write(f'{key}={params_dict[key]}\n')

    print('Parameters file saved as ' + params_file)

def read_params_file(params_file):
    '''
    the function will read a txt file with the parameters to be loaded in a bash script
    params_file: path to the file with the parameters
    '''
    params_dict = {}
    with open(params_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            key, value = line.split('=')
            params_dict[key] = value.strip()

    return params_dict

def fill_design_fsf(func_file, atlas_file, movement_path, smooth, cond_txt, design_path, design_modified_path):
    label_list = ['atlas', 'Smooth', 'Input', 'movement', 'cond']
    to_fill_dict = dict()
    for label in label_list:
        to_fill_dict[label] = dict()
        if label == 'atlas':
            to_fill_dict[label]['string_to_find'] = 'set fmri(regstandard))'
            to_fill_dict[label]['string_to_replace'] = ('set fmri(regstandard) "' + atlas_file + '"')
        elif label == 'Smooth':
            to_fill_dict[label]['string_to_find'] = 'set fmri(smooth)'
            to_fill_dict[label]['string_to_replace'] = ('set fmri(smooth) ' + str(smooth))
        elif label == 'Input':
            to_fill_dict[label]['string_to_find'] = 'set feat_files(1)'
            to_fill_dict[label]['string_to_replace'] = ('set feat_files(1) "' + func_file + '"')
        elif label == 'movement':
            to_fill_dict[label]['string_to_find'] = 'set confoundev_files(1)'
            to_fill_dict[label]['string_to_replace'] = ('set confoundev_files(1) "' + movement_path + '"')
        elif label == 'cond':
            to_fill_dict[label]['string_to_find'] = 'set fmri(custom1) '
            to_fill_dict[label]['string_to_replace'] = ('set fmri(custom1) "' + cond_txt + '"')
            
    # fill in the design.fsf file
    fill_fsf(to_fill_dict, design_path, design_modified_path)
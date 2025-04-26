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
# Some functions used across the project
# Author: Raul Hernandez

import os
import shutil
import subprocess

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
    print('Output file: ' + design_modified_path)

def extract_params(input_file):
    """
    Gets the TR and number of volumes of a file using fslinfo
    
    Parameters:
    - input_file (str): The path to the input file for the fslinfo command.
    
    Returns:
    TR - TR
    volumes - number of volumes
    
    """
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
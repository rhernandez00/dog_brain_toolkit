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
    print('Working with ' + input_file)
    # create copy of input file to output file
    print('Creating ' + output_file + '...')
    shutil.copyfile(input_file, output_file)
    
    # delete orientation
    print('Deleting orientation...')
    command = f"fslorient -deleteorient {output_file}"
    os.system(command)
    # swap axes
    print('Swapping axes...')
    command = f"fslswapdim {output_file} {combination[0]} {combination[1]} {combination[2]} {output_file}"
    os.system(command)
    # adding labels
    print('Adding labels...')
    command = f"fslorient -setqformcode 1 -setqformcode 1 {output_file}"
    os.system(command)

    print('Reorientation done!')

def write_params(params_file, x_lim1, y_lim1, z_lim1, x_lim2, y_lim2, z_lim2):
    '''
    the function will write a txt file with the limits to be loaded in a bash script
    initialx=x_lim1
    finalx=x_lim2
    initialy=y_lim1
    finaly=y_lim2
    initialz=z_lim1
    finalz=z_lim2
    '''

    with open(params_file, 'w') as f:
        f.write(f'initialx={x_lim1}\n')
        f.write(f'finalx={x_lim2}\n')
        f.write(f'initialy={y_lim1}\n')
        f.write(f'finaly={y_lim2}\n')
        f.write(f'initialz={z_lim1}\n')
        f.write(f'finalz={z_lim2}\n')

    print('cutting parameters file saved as ' + filename)

def write_params_bet(params_file, betx_1, bety_1, betz_1, thr_1, betx_2, bety_2, betz_2, thr_2, betx_3, bety_3, betz_3, thr_3):
    '''
    the function will write a txt file with BET parameters to be loaded in a bash script
    betx_1, bety_1, betz_1, thr_1, betx_2, bety_2, betz_2, thr_2, betx_3, bety_3, betz_3, thr_3
    '''

    with open(params_file, 'w') as f:
        f.write(f'betx_1={betx_1}\n')
        f.write(f'bety_1={bety_1}\n')
        f.write(f'betz_1={betz_1}\n')
        f.write(f'thr_1={thr_1}\n')
        f.write(f'betx_2={betx_2}\n')
        f.write(f'bety_2={bety_2}\n')
        f.write(f'betz_2={betz_2}\n')
        f.write(f'thr_2={thr_2}\n')
        f.write(f'betx_3={betx_3}\n')
        f.write(f'bety_3={bety_3}\n')
        f.write(f'betz_3={betz_3}\n')
        f.write(f'thr_3={thr_3}\n')

    print('BET parameters file saved as ' + params_file)
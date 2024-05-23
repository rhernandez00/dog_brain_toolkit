import os
import utils

# Description: Functions to preprocess fMRI data using FSL
# Author: Raul Hernandez

def preprocess_run(sub_N, run_N, dataset, task, specie, datafolder, session='', smooth=0, combination=['-x','z','-y']):
    """
    Preprocesses a single run of a single subject.
    Reorients file
    """

    # path to design.fsf, current directory followed by design.fsf
    design_path = os.path.join(os.getcwd(), 'FSL_designs' + os.sep + 'preprocess.fsf')

    ## determine input file and output directory ## 

    # input directory in BIDS format
    filename = datafolder + os.sep + dataset + os.sep + 'BIDS' + os.sep + specie + '-sub-' + str(sub_N).zfill(3) + os.sep + 'func' + os.sep
    # check if session is not empty
    if session != '':
        filename += 'ses-' + session + os.sep
    filename += specie + '-sub-' + str(sub_N).zfill(3)
    if session != '':
        filename += '_ses-' + session
    filename += '_task-' + task + '_run-' + str(run_N).zfill(2) + '_bold.nii.gz'

    # get TR and number of volumes
    TR,volumes = utils.extract_params(filename)

    # create output directory, where the fsl output will be saved (preprocessed data)
    outputdir = datafolder + os.sep + dataset + os.sep + 'preprocessing' + os.sep + specie + '-sub-' + str(sub_N).zfill(3)
    if session != '':
        outputdir += '_ses-' + session
    
    fsl_outputdir = datafolder + os.sep + dataset + os.sep + 'preprocessing' + os.sep + specie + '-sub-' + str(sub_N).zfill(3)
    # check if session is not empty
    if session != '':
        fsl_outputdir += '_ses-' + session
    fsl_outputdir += '_task-' + task + '_run-' + str(run_N).zfill(2)

    ## Filling out the design.fsf file ##
    # create list of labels to fill in the design.fsf file
    label_list = ['Outputdir', 'TR', 'Volumes', 'BET', 'Smooth', 'Input']

    # create dictionary to fill in the design.fsf file
    to_fill_dict = dict()
    for label in label_list:
        to_fill_dict[label] = dict()
        if label == 'Outputdir':
            to_fill_dict[label]['string_to_find'] = 'set fmri(outputdir)'
            to_fill_dict[label]['string_to_replace'] = ('set fmri(outputdir) "' + fsl_outputdir + '"')
        elif label == 'TR':
            to_fill_dict[label]['string_to_find'] = 'set fmri(tr)'
            to_fill_dict[label]['string_to_replace'] = ('set fmri(tr) ' + str(TR))
        elif label == 'Volumes':
            to_fill_dict[label]['string_to_find'] = 'set fmri(npts)'
            to_fill_dict[label]['string_to_replace'] = ('set fmri(npts) ' + str(volumes))
        elif label == 'BET':
            to_fill_dict[label]['string_to_find'] = 'set fmri(bet_yn)'
            if specie == 'H':
                to_fill_dict[label]['string_to_replace'] = ('set fmri(bet_yn) 1')
            elif specie == 'D':
                to_fill_dict[label]['string_to_replace'] = ('set fmri(bet_yn) 0')
        elif label == 'Smooth':
            to_fill_dict[label]['string_to_find'] = 'set fmri(smooth)'
            to_fill_dict[label]['string_to_replace'] = ('set fmri(smooth) ' + str(smooth))
        elif label == 'Input':
            to_fill_dict[label]['string_to_find'] = 'set feat_files(1)'
            to_fill_dict[label]['string_to_replace'] = ('set feat_files(1) "' + filename + '"')

    # fill in the design.fsf file
    design_path = os.path.join(os.getcwd(), 'FSL_designs'  + os.sep + 'preprocess.fsf')
    design_modified_path = os.path.join(os.getcwd(), 'FSL_designs'  + os.sep + 'preprocess_modified.fsf')

    utils.fill_fsf(to_fill_dict, design_path, design_modified_path)

    # run feat
    command = 'feat ' + design_modified_path

    # check if system is windows, if so, do not execute command
    print(command)
    if os.name != 'nt':
        os.system(command)
    else:
        print("The system is windows, command not executed.")

    ## reorient run ##

    base_filename = specie + '-sub-' + str(sub_N).zfill(3)

    if session != '':
        base_filename += 'ses-' + session + os.sep
    base_filename += '_task-' + task + '_run-' + str(run_N).zfill(2)

    # non-oriented file
    non_oriented_file = base_filename + '_not-oriented.nii.gz'
    # oriented file
    reoriented_file = base_filename + '_reoriented.nii.gz'

    #copy preprocessed_file to non_oriented_file
    #shutil.copyfile(outputdir + os.sep + preprocessed_file, outputdir + os.sep + non_oriented_file)
    #print(non_oriented_file + ' created')
    print('FSL output directory: ' + fsl_outputdir)

    # check if system is windows, if so, do not execute command
    if os.name == 'nt': # Windows
        print("system is windows, command not executed.")
        print("orientation to use: ",combination)
        
        print("non-oriented file: " + outputdir + os.sep + non_oriented_file)
        print("oriented file: " + outputdir + os.sep + reoriented_file)
        
    utils.reorient_file(outputdir + os.sep + non_oriented_file, outputdir + os.sep + reoriented_file, combination)


def get_mean_fct(sub_N, runs_to_use, base_run, dataset, task, specie, datafolder, session='', first_time=True):
    """
    Calculates the mean functional image for a subject and a task.
    The mean functional image is calculated by averaging the mean images of each run.
    The mean image of each run is calculated by averaging all volumes of the run.
    The mean image of each run is calculated by averaging all volumes of the run.
    The motion is corrected for each run using the first volume of the first run as reference.
    The motion parameters are saved in a .par file.
    The mean image of each run
    """
    # Check if the system is windows
    if os.name == 'nt':
        print('The system is Windows, this is a test, no actual system or FSL commands will be run')

    # output directory where the fsl output will be saved (preprocessed data)
    outputdir = datafolder + os.sep + dataset + os.sep + 'preprocessing' + os.sep + specie + '-sub-' + str(sub_N).zfill(3)
    # check if session is not empty
    if session != '':
        outputdir += '_ses-' + session
    #run_N = 1
    #outputdir += '_task-' + task + '_run-' + str(run_N).zfill(2)


    # movement directory
    movementdir = datafolder + os.sep + dataset + os.sep + 'movement'
    # create movement directory if it does not exist
    if not os.path.exists(movementdir):
        os.makedirs(movementdir)

    ## obtain volume to be used as base to correct all others ##
    filename = specie + '-sub-' + str(sub_N).zfill(3)
    # adding session if there is one
    if session != '':
        filename += '_ses-' + session
    filename += '_task-' + task + '_run-' + str(base_run).zfill(2) + '_reoriented.nii.gz'

    if first_time: # get the volume
        # get the first volume of the first run to use as base volume
        command = f"fslroi {outputdir + os.sep + filename} {outputdir + os.sep + 'base_vol.nii.gz'} 0 1"
        os.system(command)
        first_time = False
    else:
        # check if base_vol exists
        if not os.path.exists(outputdir + os.sep + 'base_vol.nii.gz'):
            print('base_vol.nii.gz does not exist, run the code with first_time = True')
            print('path: ' + outputdir + os.sep + 'base_vol.nii.gz')
            raise ValueError('base_vol.nii.gz does not exist, run the code with first_time = True')
    ## ----- ##

    # This string will be used to generate the mean image
    mean_images = ''

    ## calculate motion for each run and generate par file ##
    for n,run_N in enumerate(runs_to_use):
        print('processing ' + str(n+1) + ' of ' + str(len(runs_to_use)))
        filename = specie + '-sub-' + str(sub_N).zfill(3)
        # adding session if there is one
        if session != '':
            filename += '_ses-' + session
        filename += '_task-' + task + '_run-' + str(run_N).zfill(2) + '_reoriented.nii.gz'
        # calculate motion and generate par file
        print('calculating motion...')
        
        command = f"mcflirt -in {outputdir + os.sep + filename} -out {outputdir + os.sep + filename[:-7] + '_mc_tmp.nii.gz'} -plots"
        
        # print commmand
        print(command)
        # if the system is not windows, run the command
        if os.name != 'nt':
            os.system(command)
        # rename par file
        tmp_par_file = outputdir + os.sep + filename[:-7] + '_mc_tmp.nii.gz.par'
        par_file = movementdir + os.sep + filename[:-18] + '.par'
        if os.name != 'nt':
            os.rename(tmp_par_file, par_file)
        # print file saved
        
        print('par file saved as: ' + par_file)
        
        print('correcting motion...')
        # motion correct the file to the base_vol
        command = f"mcflirt -in {outputdir + os.sep + filename} -out {outputdir + os.sep + filename[:-7] + '_mc.nii.gz'} -reffile {outputdir + os.sep + 'base_vol.nii.gz'}"
        # print commmand
        print(command)
        # if the system is not windows, run the command
        if os.name != 'nt':
            os.system(command)
        # print file saved
        print('motion corrected file saved as ' + filename[:-7] + '_mc.nii.gz')

        # remove temporary file
        tmp_file = outputdir + os.sep + filename[:-7] + '_mc_tmp.nii.gz'
        print('removing temporal file: ' + tmp_file)
        # if the system is not windows, run the command
        if os.name != 'nt':
            os.remove(outputdir + os.sep + filename[:-7] + '_mc_tmp.nii.gz')
        # calculate mean image
        print('calculating mean image...')
        command = f"fslmaths {outputdir + os.sep + filename[:-7] + '_mc.nii.gz'} -Tmean {outputdir + os.sep + filename[:-18] + '_mean.nii.gz'}"
        # print commmand
        print(command)
        # if the system is not windows, run the command
        if os.name != 'nt':
            os.system(command)
        # add filename to mean_images
        mean_images += outputdir + os.sep + filename[:-18] + '_mean.nii.gz' + ' '

    if first_time: # if yes, calculate mean image
        mean_fct_file = outputdir + os.sep + specie + '-sub-' + str(sub_N).zfill(3)
        # adding session if there is one
        if session != '':
            mean_fct_file += '_ses-' + session
        mean_fct_file += '_task-' + task + '_mean_fct_uncut.nii.gz'

        # append mean images to a single 4D image
        command = f"fslmerge -t {mean_fct_file} {mean_images}"
        # print commmand
        print(command)
        # if the system is not windows, run the command
        if os.name != 'nt':
            os.system(command)
        print('mean fct file saved as ' + mean_fct_file)

        # calculate mean image
        command = f"fslmaths {mean_fct_file} -Tmean {mean_fct_file}"
        # print commmand
        print(command)
        # if the system is not windows, run the command
        if os.name != 'nt':
            os.system(command)

    print('done')

def mean_to_STD(sub_N, dataset, task, specie, datafolder, atlas_type, session='', img_type='brain2mm'):
    """
    This function will take the mean functional image of a subject and transform it to the space of the atlas.
    sub_N: subject number
    dataset: dataset name
    task: task name
    specie: specie name
    datafolder: path to the data folder
    atlas_type: type of atlas to use
    session: session number
    img_type: type of image to use, default is 'brain2mm'

    """
    atlas_type='Czeibert'
    

    # working directory
    workingdir = datafolder + os.sep + dataset + os.sep + 'preprocessing' + os.sep + specie + '-sub-' + str(sub_N).zfill(3)

    mean_fct_file = workingdir + os.sep + specie + '-sub-' + str(sub_N).zfill(3)
    # adding session if there is one
    if session != '':
        mean_fct_file += '_ses-' + session
    cut_mean_fct_file = mean_fct_file + '_task-' + task + '_mean_fct.nii.gz'
    mean_fct_file += '_task-' + task + '_mean_fct_uncut.nii.gz'
    masked_mean_fct_file = mean_fct_file[:-7] + '_brain.nii.gz'
    mean_fct_file_STD = mean_fct_file[:-7] + '_STD.nii.gz'
    mean_fct2STD_mat = mean_fct_file[:-7] + '_2STD.mat'


    if specie == 'D':
        specieS = 'Dog'
    elif specie == 'H':
        specieS = 'Hum'

    

    # generate path to atlas. The atlas is in the same folder as the script
    atlas_file = os.getcwd() + os.sep + "Atlas" + os.sep + specieS + os.sep + atlas_type + os.sep + img_type + ".nii.gz"

    command = f"flirt -in {masked_mean_fct_file} -ref {atlas_file} -out {mean_fct_file_STD} -omat {mean_fct2STD_mat} -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp trilinear"
    # if the system is windows, don't run the command, just write it down
    print(command)
    if os.name == 'nt': # Windows
        print("system is windows, command not executed. Command is:")
    else:
        os.system(command)
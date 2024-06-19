import os
import utils
import shutil

from nilearn.plotting import plot_anat, show
import ipywidgets as widgets
import utils
import os
from importlib import reload
from nilearn import plotting
from nilearn import image as nli
import nibabel as nib
import numpy as np
from ipywidgets import HBox, VBox
import numpy as np
from nilearn.plotting import plot_anat, show
import ipywidgets as widgets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from IPython.display import display, Markdown, clear_output

# Description: Functions to preprocess fMRI data using FSL
# Author: Raul Hernandez

def bet_app(project_dict, sub_N):
    # Create a custom colormaps
    # black
    black = np.array([0, 0, 0, 0])   # RGBA for black (last value is alpha)
    # red
    red = np.array([1, 0, 0, 1])     # RGBA for red
    # blue
    blue = np.array([0, 0, 1, 1])    # RGBA for blue
    # yellow
    yellow = np.array([1, 1, 0, 1])  # RGBA for yellow (last value is alpha)

    # create a red colormap
    red_cmap = ListedColormap([black, red])
    # create a blue colormap
    blue_cmap = ListedColormap([black, blue])
    # create a yellow colormap
    yellow_cmap = ListedColormap([black, yellow])
    def plot_bet(x,y,z, betx_1, bety_1, betz_1, betx_2, bety_2, betz_2, betx_3, bety_3, betz_3, plot_mask):
    
        """
        Plot slices from the sagittal, coronal, and axial views side by side.
        """
        fig, axes = plt.subplots(1, 3, figsize=(8, 5))
        
        # Sagittal
        sagittal_slice = data[x, :, :]
        axes[0].imshow(sagittal_slice.T, cmap='gray', origin='lower')
        axes[0].scatter(bety_1,betz_1,s=200, c='red')
        # plot blue only if betx_2 matches the current slice
        if betx_2 == x:
            axes[0].scatter(bety_2,betz_2,s=200, c='blue')
        axes[0].scatter(bety_3,betz_3,s=200, c='yellow')
        axes[0].axis('off')
        # display mask if plot_mask is True
        if plot_mask:
            # get mask1 for sagital slice
            sagital_mask1 = data_mask1[x, :, :]
            # plot mask1 in red
            axes[0].imshow(sagital_mask1.T, cmap=red_cmap, origin='lower', alpha=0.5)
            # get mask2 for sagital slice
            sagital_mask2 = data_mask2[x, :, :]
            # plot mask2 in blue
            axes[0].imshow(sagital_mask2.T, cmap=blue_cmap, origin='lower', alpha=0.5)
            # get mask3 for sagital slice
            sagital_mask3 = data_mask3[x, :, :]
            # plot mask3 in yellow
            axes[0].imshow(sagital_mask3.T, cmap=yellow_cmap, origin='lower', alpha=0.5)

        # Coronal
        coronal_slice = data[:, y, :]
        axes[1].imshow(coronal_slice.T, cmap='gray', origin='lower')
        # plot red only if bety_1 matches the current slice
        if bety_1 == y:
            axes[1].scatter(betx_1,betz_1,s=200, c='red')
        axes[1].scatter(betx_2,betz_2,s=200, c='blue')
        axes[1].scatter(betx_3,betz_3,s=200, c='yellow')
        axes[1].axis('off')
        # display mask if plot_mask is True
        if plot_mask:
            # get mask1 for coronal slice
            coronal_mask1 = data_mask1[:, y, :]
            # plot mask1 in red
            axes[1].imshow(coronal_mask1.T, cmap=red_cmap, origin='lower', alpha=0.5)
            # get mask2 for coronal slice
            coronal_mask2 = data_mask2[:, y, :]
            # plot mask2 in blue
            axes[1].imshow(coronal_mask2.T, cmap=blue_cmap, origin='lower', alpha=0.5)
            # get mask3 for coronal slice
            coronal_mask3 = data_mask3[:, y, :]
            # plot mask3 in yellow
            axes[1].imshow(coronal_mask3.T, cmap=yellow_cmap, origin='lower', alpha=0.5)

        # Axial
        axial_slice = data[:, :, z]
        axes[2].imshow(axial_slice.T, cmap='gray', origin='lower')
        axes[2].scatter(betx_1,bety_1,s=200, c='red')
        axes[2].scatter(betx_2,bety_2,s=200, c='blue')
        # plot yellow only if betz_3 matches the current slice
        if betz_3 == z:
            axes[2].scatter(betx_3,bety_3,s=200, c='yellow')
        axes[2].axis('off')
        # display mask if plot_mask is True
        if plot_mask:
            # get mask1 for axial slice
            axial_mask1 = data_mask1[:, :, z]
            # plot mask1 in red
            axes[2].imshow(axial_mask1.T, cmap=red_cmap, origin='lower', alpha=0.5)
            # get mask2 for axial slice
            axial_mask2 = data_mask2[:, :, z]
            # plot mask2 in blue
            axes[2].imshow(axial_mask2.T, cmap=blue_cmap, origin='lower', alpha=0.5)
            # get mask3 for axial slice
            axial_mask3 = data_mask3[:, :, z]
            # plot mask3 in yellow
            axes[2].imshow(axial_mask3.T, cmap=yellow_cmap, origin='lower', alpha=0.5)
        
        # make a tight layout
        plt.tight_layout()

        # make the background black
        fig.patch.set_facecolor('black')

        # display the plot
        plt.show()

    dataset = project_dict['Dataset']
    session = project_dict['Session']
    task = project_dict['Task']
    specie = project_dict['Specie']
    datafolder = project_dict['Datafolder']

    # working directory
    workingdir = datafolder + os.sep + dataset + os.sep + 'preprocessing' + os.sep + specie + '-sub-' + str(sub_N).zfill(3)

    mean_fct_file = workingdir + os.sep + specie + '-sub-' + str(sub_N).zfill(3)
    # adding session if there is one
    if session != '':
        mean_fct_file += '_ses-' + session
    cut_mean_fct_file = mean_fct_file + '_task-' + task + '_mean_fct.nii.gz'
    mask_file = mean_fct_file + '_task-' + task + '_mean_fct_mask.nii.gz'
    mask_file1 = mean_fct_file + '_task-' + task + '_mean_fct_mask1.nii.gz'
    mask_file2 = mean_fct_file + '_task-' + task + '_mean_fct_mask2.nii.gz'
    mask_file3 = mean_fct_file + '_task-' + task + '_mean_fct_mask3.nii.gz'
    mean_fct_file += '_task-' + task + '_mean_fct.nii.gz'
    masked_mean_fct_file = mean_fct_file[:-7] + '_brain.nii.gz'

    # load cut_mean_fct_file
    img = nib.load(cut_mean_fct_file)

    # Get the data from the image
    data = img.get_fdata()

    # if the mask files exist, load them
    if os.path.exists(mask_file1):
        mask_img1 = nib.load(mask_file1)
        data_mask1 = mask_img1.get_fdata()
        mask_img2 = nib.load(mask_file2)
        data_mask2 = mask_img2.get_fdata()
        mask_img3 = nib.load(mask_file3)
        data_mask3 = mask_img3.get_fdata()


    # determine min and max values for each axis
    maxX,maxY,maxZ = img.shape

    # define function to apply BET
    def apply_bet_button(betx_1, bety_1, betz_1, thr_1, betx_2, bety_2, betz_2, thr_2, betx_3, bety_3, betz_3, thr_3):
        global data_mask1, data_mask2, data_mask3
        # determine name of params_file based on mean_fct_file
        params_file = mean_fct_file[:-7] + '_bet_params.txt'
        # create a dict with the parameters
        param_dict = {'betx_1':betx_1, 'bety_1':bety_1, 'betz_1':betz_1, 'thr_1':thr_1,
                        'betx_2':betx_2, 'bety_2':bety_2, 'betz_2':betz_2, 'thr_2':thr_2,
                        'betx_3':betx_3, 'bety_3':bety_3, 'betz_3':betz_3, 'thr_3':thr_3,
                        'output_file':mask_file, 'masked_file':masked_mean_fct_file,
                        'mask_file1':mask_file1, 'mask_file2':mask_file2, 'mask_file3':mask_file3,
                        }

        # save the cutting parameters
        utils.write_params_file(params_file, param_dict)
        print('button pressed')
        command = f"./run_bet.sh {cut_mean_fct_file} {params_file}"
        os.system(command)

        # update button description and status
        col1.children[3].description = 'Display mask'
        col1.children[3].disabled = False
        
        # Load each of the three masks
        
        mask_img1 = nib.load(mask_file1)
        data_mask1 = mask_img1.get_fdata()
        mask_img2 = nib.load(mask_file2)
        data_mask2 = mask_img2.get_fdata()
        mask_img3 = nib.load(mask_file3)
        data_mask3 = mask_img3.get_fdata()


    # create 4 sets of sliders, one for the slice, and 3 for the spheres marking the initial place of the BET sphere

    # check if mask file 1 already exist
    if os.path.exists(mask_file1):
        button_description = 'Display mask'
        button_disabled = False
    else:
        button_description = 'Mask not available'
        button_disabled = True

    # sliders for the slice
    col1 = widgets.VBox([widgets.IntSlider(min=0, max=maxX, step=1, value=np.round(maxX/2), description='X'),
                        widgets.IntSlider(min=0, max=maxY, step=1, value=np.round(maxY/2), description='Y'),
                        widgets.IntSlider(min=0, max=maxZ, step=1, value=np.round(maxZ/2), description='Z'),
                        widgets.ToggleButton(value=False, description=button_description, button_style='info', disabled=button_disabled),
                        ])

    # sliders for the spheres

    # first sphere, has 4 values, x,y,z and threshold
    col2 = widgets.VBox([widgets.IntSlider(min=0, max=maxX, step=1, value=np.round(maxX/2), description='betx_1'),
                        widgets.IntSlider(min=0, max=maxY, step=1, value=14, description='bety_1'),
                        widgets.IntSlider(min=0, max=maxZ, step=1, value=11, description='betz_1'),
                        widgets.FloatSlider(min=0, max=1, step=0.05, value=0.8, description='thr_1'),
                        ])
    # second sphere, has 4 values, x,y,z and threshold
    col3 = widgets.VBox([widgets.IntSlider(min=0, max=maxX, step=1, value=np.round(maxX/2), description='betx_2'),
                        widgets.IntSlider(min=0, max=maxY, step=1, value=21, description='bety_2'),
                        widgets.IntSlider(min=0, max=maxZ, step=1, value=18, description='betz_2'),
                        widgets.FloatSlider(min=0, max=1, step=0.05, value=0.8, description='thr_2'),
                        ])
    # third sphere, has 4 values, x,y,z and threshold
    col4 = widgets.VBox([widgets.IntSlider(min=0, max=maxX, step=1, value=np.round(maxX/2), description='betx_3'),
                        widgets.IntSlider(min=0, max=maxY, step=1, value=30, description='bety_3'),
                        widgets.IntSlider(min=0, max=maxZ, step=1, value=6, description='betz_3'),
                        widgets.FloatSlider(min=0, max=1, step=0.05, value=0.8, description='thr_3'),
                        ])

    # linkin the sliders to the function
    bet_app_out = widgets.interactive_output(plot_bet, {'x':col1.children[0], 'y':col1.children[1], 'z':col1.children[2],
                                                    'betx_1':col2.children[0], 'bety_1':col2.children[1], 'betz_1':col2.children[2],
                                                    'betx_2':col3.children[0], 'bety_2':col3.children[1], 'betz_2':col3.children[2],
                                                    'betx_3':col4.children[0], 'bety_3':col4.children[1], 'betz_3':col4.children[2],
                                                    'plot_mask':col1.children[3],
                                                    })
    # button to apply the BET
    col5 = widgets.Button(description='Apply BET')

    # setting the button to call the function
    col5.on_click(lambda b: apply_bet_button(
        col2.children[0].value, col2.children[1].value, col2.children[2].value, col2.children[3].value,
        col3.children[0].value, col3.children[1].value, col3.children[2].value, col3.children[3].value,
        col4.children[0].value, col4.children[1].value, col4.children[2].value, col4.children[3].value,
    ))


    row1 = HBox([col1,col2,col3,col4])
    row2 = HBox([col5])

    bet_app_tab = VBox([row1,row2])
    return bet_app_tab, bet_app_out

def crop_appB(project_dict): ##### Doesn't work#####
    # create a dropdown menu to select the subject
    dropdown_sub = widgets.Dropdown(
        options=project_dict['Participants'],
        description='Participants',
        disabled=False,
    )
    tab_crop_appB,out = widgets.interactive(crop_app, {'project_dict':project_dict, 'sub_N':dropdown_sub})
    tab_crop_app = VBox([dropdown_sub, tab_crop_appB])
    return tab_crop_app,out

def crop_app(project_dict, sub_N):
    
    def plot_slices(x, y, z, x_lim1, y_lim1, z_lim1, x_lim2, y_lim2, z_lim2):
        """
        Plot slices from the sagittal, coronal, and axial views side by side.
        """
        fig, axes = plt.subplots(1, 3, figsize=(8, 5))

        # Sagittal
        sagittal_slice = data[x, :, :]
        axes[0].imshow(sagittal_slice.T, cmap='gray', origin='lower')
        axes[0].axis('off')

        # Coronal
        coronal_slice = data[:, y, :]
        axes[1].imshow(coronal_slice.T, cmap='gray', origin='lower')
        axes[1].axis('off')

        # Axial
        axial_slice = data[:, :, z]
        axes[2].imshow(axial_slice.T, cmap='gray', origin='lower')
        axes[2].axis('off')

        # make a tight layout
        plt.tight_layout()

        # make the background black
        fig.patch.set_facecolor('black')

        
        # Plotting red lines
        # plotting lines in sagital slice
        axes[0].axhline(y=z_lim1, color='red', lw=2)
        axes[0].axvline(x=y_lim1, color='red', lw=2)

        # plotting lines in coronal slice
        axes[1].axvline(x=x_lim1, color='red', lw=2)
        axes[1].axhline(y=z_lim1, color='red', lw=2)
        
        # plotting lines in axial slice
        axes[2].axvline(x=x_lim1, color='red', lw=2)
        axes[2].axhline(y=y_lim1, color='red', lw=2)

        # Plotting blue lines
        # plotting lines in sagital slice
        axes[0].axhline(y=z_lim2, color='blue', lw=2)
        axes[0].axvline(x=y_lim2, color='blue', lw=2)

        # plotting lines in coronal slice
        axes[1].axvline(x=x_lim2, color='blue', lw=2)
        axes[1].axhline(y=z_lim2, color='blue', lw=2)

        # plotting lines in axial slice
        axes[2].axvline(x=x_lim2, color='blue', lw=2)
        axes[2].axhline(y=y_lim2, color='blue', lw=2)

        plt.show()


    def apply_cut_button(x_lim1, y_lim1, z_lim1, x_lim2, y_lim2, z_lim2):
        # determine name of params_file based on mean_fct_file
        params_file = mean_fct_file[:-7] + '_cut_params.txt'

        # create a dict with the cutting parameters
        param_dict = {'x_lim1':x_lim1, 'y_lim1':y_lim1,
                        'z_lim1':z_lim1, 'x_lim2':x_lim2,
                        'y_lim2':y_lim2, 'z_lim2':z_lim2, 
                        'output_file':cut_mean_fct_file,
                        }

        # save the cutting parameters
        utils.write_params_file(params_file, param_dict)
        print('button pressed')
        command = f"./remove_slices.sh {mean_fct_file} {params_file}"
        os.system(command)

    dataset = project_dict['Dataset']
    session = project_dict['Session']
    task = project_dict['Task']
    specie = project_dict['Specie']
    datafolder = project_dict['Datafolder']


    # working directory
    workingdir = datafolder + os.sep + dataset + os.sep + 'preprocessing' + os.sep + specie + '-sub-' + str(sub_N).zfill(3)

    mean_fct_file = workingdir + os.sep + specie + '-sub-' + str(sub_N).zfill(3)
    
    # adding session if there is one
    if session != '':
        mean_fct_file += '_ses-' + session
    cut_mean_fct_file = mean_fct_file + '_task-' + task + '_mean_fct.nii.gz'
    mean_fct_file += '_task-' + task + '_mean_fct_uncut.nii.gz'


    # determine min and max values for each axis
    img = nib.load(mean_fct_file)

    # Get the data from the image
    data = img.get_fdata()

    slider_style = {'description_width': 'initial', 'width': '2px'}

    maxX,maxY,maxZ = img.shape
    col1 = widgets.VBox([widgets.IntSlider(min=0, max=maxX, step=1, value=np.round(maxX/2), description='X'),
                        widgets.IntSlider(min=0, max=maxY, step=1, value=np.round(maxY/2), description='Y'),
                        widgets.IntSlider(min=0, max=maxZ, step=1, value=np.round(maxZ/2), description='Z')])

    col2 = widgets.VBox([widgets.IntSlider(min=0, max=maxX, step=1, value=0, description='lim X'),
                        widgets.IntSlider(min=0, max=maxY, step=1, value=0, description='lim Y'),
                        widgets.IntSlider(min=0, max=maxZ, step=1, value=0, description='lim Z')])

    col3 = widgets.VBox([widgets.IntSlider(min=0, max=maxX, step=1, value=maxX, description='lim X'),
                        widgets.IntSlider(min=0, max=maxY, step=1, value=maxY, description='lim Y'),
                        widgets.IntSlider(min=0, max=maxZ, step=1, value=maxZ, description='lim Z')])

    col4 = widgets.Button(description='Apply cut')

    out = widgets.interactive_output(plot_slices, {'x':col1.children[0], 'y':col1.children[1], 'z':col1.children[2],
                                                    'x_lim1':col2.children[0], 'y_lim1':col2.children[1], 'z_lim1':col2.children[2],
                                                    'x_lim2':col3.children[0], 'y_lim2':col3.children[1], 'z_lim2':col3.children[2],
                                                    })

    # setting the button to call the function
    col4.on_click(lambda b: apply_cut_button(
        col2.children[0].value, col2.children[1].value, col2.children[2].value,
        col3.children[0].value, col3.children[1].value, col3.children[2].value,
    ))

    tab_crop_app = VBox([HBox([col1,col2,col3]),col4])

    return tab_crop_app,out

def check_job_status(job):
    # This function will check if the job finished, failed or is still running
    # Right now it will return 'Finished'
    job_status = 'Finished'
    return job_status

def run_process(job):
    '''
    Will select the variables from the schedule_table and the project_dict to run the process
    Can run: 
    preprocess_run
    get_mean_fct
    crop_interface
    bet_interface
    mean_to_STD
    run_to_STD
    '''
    # get the process to run
    
    # get the other variables of the job
    # user = job['User']
    dataset = job['Dataset']
    session = job['Session']
    task = job['Task']
    sub_N = job['sub_N']
    specie = job['Specie']
    process = job['Process']
    datafolder = job['Datafolder']
    job_status = job['Status']
    combination = job['Combination']
    smooth = job['Smooth']
    atlas_type = job['Atlas_type']
    base_run = 1
    img_type = 'brain2mm'

    print(f"Running process: {process}")
    # run the adecuate process
    if process == 'Preprocess':
        print('Running preprocess')
        for run_N in job['run_N']:
            preprocess_run(
                sub_N, run_N, dataset, task, specie,
                datafolder, session, smooth,
                    combination)
    elif process == 'Mean fct':
        print('Running get_mean_fct')
        runs_to_use =  job['run_N']
        get_mean_fct(
            sub_N, runs_to_use, base_run, dataset, 
            task, specie, datafolder, session, first_time=True)
    elif process == 'Crop':
        print('Running crop_interface')
        #crop_interface(job)
    elif process == 'BET':
        print('Running bet_interface')
        #bet_interface(job)
    elif process == 'Mean to atlas':
        print('Running mean_to_STD')
        mean_to_STD(
            sub_N, dataset, task, specie, datafolder, 
            atlas_type, session, img_type)
    elif process == 'Runs to atlas':
    
        print('Running run_to_STD')
        for run_N in job['run_N']:
            run_to_STD(
                sub_N, run_N, dataset, task, 
                specie, datafolder, atlas_type, 
                img_type, session=session)
    elif process == 'Motion':
        print('Running process_motion')
        #process_motion(job)
    else:
        print('Process not found')


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
    
    fsl_outputdir = datafolder + os.sep + dataset + os.sep + 'preprocessing' + os.sep + specie + '-sub-' + str(sub_N).zfill(3) + os.sep + specie + '-sub-' + str(sub_N).zfill(3)
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
    
    # check if previous feat preprocessing directory exists, if so, delete it
    if os.path.exists(fsl_outputdir + '.feat'):
        shutil.rmtree(fsl_outputdir + '.feat')


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

    preprocessed_file = fsl_outputdir + '.feat' + os.sep + 'filtered_func_data.nii.gz'
    # check if the system is windows
    if os.name == 'nt':
        print('A copy should have been created, but this is Windows')
        print(non_oriented_file + ' a copy of this file here:')
        print('FSL output directory: ' + fsl_outputdir)
    else:
        #copy preprocessed_file to non_oriented_file
        shutil.copyfile(preprocessed_file, outputdir + os.sep + non_oriented_file)
        print(non_oriented_file + ' created')
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
        # print commmand
        print(command)
        # if the system is not windows, run the command
        if os.name != 'nt':
            os.system(command)
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
    mean_fct_file += '_task-' + task + '_mean_fct.nii.gz'
    masked_mean_fct_file = mean_fct_file[:-7] + '_brain.nii.gz'
    mean_fct_file_STD = mean_fct_file[:-7] + '_STD.nii.gz'
    mean_fct2STD_mat = mean_fct_file[:-7] + '2STD.mat'


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
        print("System is Windows, command not executed")
    else:
        os.system(command)

def run_to_STD(sub_N, run_N, dataset, task, specie, datafolder, atlas_type, img_type='brain2mm', session=''):
    """
    This function will take a semi-processed run of a participant 
    cut it, apply BET and transform it to the space of the atlas.
    sub_N: subject number
    run_N: run number
    dataset: dataset name
    task: task name
    specie: specie name
    datafolder: path to the data folder
    atlas_type: type of atlas to use
    img_type: type of image to use, default is 'brain2mm'
    session: session number (in case there is one)
    """

    # if the system is windows
    if os.name == 'nt':
        print('The system is Windows, FSL functions ans bash scripts will not be executed')

    # working directories
    std_dir = datafolder + os.sep + dataset + os.sep + 'normalized' + os.sep + specie + '-sub-' + str(sub_N).zfill(3)
    preprocess_dir = datafolder + os.sep + dataset + os.sep + 'preprocessing' + os.sep + specie + '-sub-' + str(sub_N).zfill(3)

    # cutting parameters file
    params_file = preprocess_dir + os.sep + specie + '-sub-' + str(sub_N).zfill(3) 
    # adding session if there is one
    if session != '':
        params_file += '_ses-' + session

    params_file += '_task-' + task + '_mean_fct_uncut_cut_params.txt'

    filename = specie + '-sub-' + str(sub_N).zfill(3)
    # adding session if there is one
    if session != '':
        filename += '_ses-' + session
    # name for STD file
    std_file = filename + '_task-' + task + '_run-' + str(run_N).zfill(2) + '.nii.gz'
    # name for reoriented and motion corrected file
    preprocessed_file = filename + '_task-' + task + '_run-' + str(run_N).zfill(2) + '_reoriented_mc.nii.gz'
    cut_file = filename + '_task-' + task + '_run-' + str(run_N).zfill(2) + '_reoriented_mc_cut.nii.gz'
    
    #print('STD file saved as ' + std_dir + os.sep + std_file)
    #print('preprocessed file is ' + preprocess_dir + os.sep + preprocessed_file)
    
    # updating output params_file
    params_dict = utils.read_params_file(params_file)
    params_dict['output_file'] = preprocess_dir + os.sep + cut_file
    params_file_current = params_file[:-30] + '_run-' + str(run_N).zfill(2) + '_cut_params.txt'
    # add mask file to parameters
    params_dict['mask_file'] = preprocess_dir + os.sep + filename + '_task-' + task + '_mean_fct_mask.nii.gz'

    # save the cutting parameters
    utils.write_params_file(params_file_current, params_dict)

    # cut preprocessed file and apply BET
    command = f"./remove_slices.sh {preprocess_dir + os.sep + preprocessed_file} {params_file_current}"
    
    print(command)
    if os.name != 'nt': # Windows
        os.system(command)

    # apply transformation to STD
    mean_fct2STD_mat = preprocess_dir + os.sep + specie + '-sub-' + str(sub_N).zfill(3)
    # adding session if there is one
    if session != '':
        mean_fct2STD_mat += '_ses-' + session
    mean_fct2STD_mat += '_task-' + task + '_mean_fct2STD.mat'
    
    # determine folder for the atlas    
    if specie == 'D':
        specieS = 'Dog'
    elif specie == 'H':
        specieS = 'Hum'
    atlas_file = os.getcwd() + os.sep + "Atlas" + os.sep + specieS + os.sep + atlas_type + os.sep + img_type + ".nii.gz"
    
    # if normalized directory does not exist, create it
    if not os.path.exists(std_dir):
        os.makedirs(std_dir)
        print('Directory ' + std_dir + ' created')
    else:
        print('Directory ' + std_dir + ' already exists')

    command = f"flirt -in {preprocess_dir + os.sep + cut_file} -ref {atlas_file} -applyxfm -init {mean_fct2STD_mat} -out {std_dir + os.sep + std_file}"
    print(command)
    if os.name != 'nt': # Windows
        os.system(command)
    print('done')
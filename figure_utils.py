
import os
import numpy as np
import nibabel as nib
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import copy
import re
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib import rcParams
import shutil
from matplotlib.patches import Arc
from matplotlib.patches import Wedge
from matplotlib.patches import Rectangle
from scipy.optimize import linear_sum_assignment

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


def write_text_on_png(
    input_path,
    text,
    x=None,
    y=None,
    output_path=None,
    font_family="DejaVu Sans",
    font_size=20,
    color="white",
    dpi=100,
    ha="left",
    va="top",
):
    """
    Open a PNG file, write text at pixel coordinates (x, y), and save the result.

    Parameters
    ----------
    input_path : str or Path
        Path to the input PNG file.
    text : str
        Text to write on the image.
    x : int, float, or None, optional
        X coordinate in pixels. If None, text is centered horizontally.
    y : int, float, or None, optional
        Y coordinate in pixels. If None, text is centered vertically.
    output_path : str or Path or None, optional
        Path to save the modified image.
        If None, saves as '<original_name>_text.png'.
    font_family : str, optional
        Matplotlib font family name, e.g. "DejaVu Sans", "Arial".
    font_size : int or float, optional
        Font size.
    color : str or tuple, optional
        Text color, e.g. "white", "red", (1, 0, 0).
    dpi : int, optional
        Output DPI for saving.
    ha : str, optional
        Horizontal alignment: "left", "center", or "right".
        If x is None, defaults to "center" unless explicitly changed after editing code.
    va : str, optional
        Vertical alignment: "top", "center", "bottom", or "baseline".
        If y is None, defaults to "center" unless explicitly changed after editing code.

    Returns
    -------
    Path
        Path to the saved output image.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() != ".png":
        raise ValueError(f"Input file must be a PNG: {input_path}")

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_text.png")
    else:
        output_path = Path(output_path)

    img = mpimg.imread(str(input_path))
    height, width = img.shape[:2]

    # Auto-center coordinates when x or y is None
    if x is None:
        x = width / 2
        if ha == "left":
            ha = "center"

    if y is None:
        y = height / 2
        if va == "top":
            va = "center"

    # Create a figure sized to match the image
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # image-style coordinates: origin at top-left
    ax.axis("off")

    ax.text(
        x,
        y,
        text,
        fontsize=font_size,
        color=color,
        fontfamily=font_family,
        ha=ha,
        va=va,
    )

    fig.savefig(str(output_path), dpi=dpi, transparent=True)
    plt.close(fig)

    return output_path

def homogeneity_in_surrogate(sub_N, datafolder, database_for_surrogates, surrogate_N, vox_list, number_of_clusters, cluster_ID, segmentation_type='anatomical'):
    # clusters,clusters_ID_list,_ = get_clusters(project_dict_segmentation, number_of_clusters, datafolder)
    # vox_list = clusters[cluster_ID]
    # define the path for the functional file based on surrogate_N
    func_path = os.path.join(datafolder, database_for_surrogates, 'surrogates', 'surrogate_' + str(surrogate_N).zfill(4),
                                    'D-sub-' + str(sub_N).zfill(2), 'D-sub-' + str(sub_N).zfill(2) + '_ses-01_task-resting_state_run-01.nii.gz')
    print(f"Loading: {func_path}")
    # load the nii file
    func_nii = nib.load(func_path).get_fdata()
    # flatten func_nii keeping the time dimension (4th dimension)
    func_nii_flat = np.reshape(func_nii, (-1, func_nii.shape[-1]))
    # if number_of_clusters is str, it means it is a clustering segmentation
    if segmentation_type == 'anatomical':
        res_path = os.path.join(datafolder, database_for_surrogates, 'surrogates_homogeinity', 
                                f'surrogate_{str(surrogate_N).zfill(4)}',
                        f'D-sub-{str(sub_N).zfill(2)}',
                        f'D-sub-{str(sub_N).zfill(2)}_ses-01_task-resting_state_run-01_homogeneity_cluster-{number_of_clusters}_clusterID-{str(int(cluster_ID)).zfill(3)}.txt'
                    )
    else:
        res_path = os.path.join(datafolder, database_for_surrogates, 'surrogates_homogeinity', 
                                f'surrogate_{str(surrogate_N).zfill(4)}',
                        f'D-sub-{str(sub_N).zfill(2)}',
                        f'D-sub-{str(sub_N).zfill(2)}_ses-01_task-resting_state_run-01_homogeneity_cluster-{str(number_of_clusters).zfill(3)}_clusterID-{str(int(cluster_ID)).zfill(3)}.txt'
                    )
    
    # create the directory if it does not exist
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    
    # initialize the corr_list
    corr_list = []

    for n,nVox1 in enumerate(vox_list):
        for nVox2 in vox_list[n+1:]:
            #getting the time series for both voxels
            time_series1 = func_nii_flat[nVox1]
            time_series2 = func_nii_flat[nVox2]
                
            # Calculate the Pearson correlation between the two time series
            corr = np.corrcoef(time_series1, time_series2)[0, 1]
            # if corr is nan, continue to the next iteration
            if np.isnan(corr):
                continue
            
            corr_list.append(corr)
            # Store the results
            # results.append([nVox1, nVox2, corr])

    # calculate average correlation
    average_corr = np.mean(corr_list)
    # calculate number of correlations
    n_corr = len(corr_list)
    # save the average correlation and number of correlations to res_path
    np.savetxt(res_path, np.array([average_corr, n_corr]))
    print(f"Saved file: {res_path}")

def fill_missing_pairs(data_table):
    # Get unique IDs from both columns
    unique_epi_ids = data_table['ID_epi'].unique()
    unique_k9_ids = data_table['ID_K9'].unique()

    # create dictionary with N_voxels
    ID_dict = {}
    id_list = ['ID_epi', 'ID_K9']
    for id_type in id_list:
        # get unique IDs
        unique_ids = data_table[id_type].unique()
        for unique_id in unique_ids:
            # get the number of voxels for each ID
            n_voxels = data_table[data_table[id_type] == unique_id]['N_voxels_' + id_type.split('_')[1]].values[0]
            ID_dict[unique_id] = n_voxels
        #ID_dict[id_type] = {}

    # Create a complete set of pairs
    complete_pairs = pd.MultiIndex.from_product([unique_epi_ids, unique_k9_ids], names=['ID_epi', 'ID_K9'])

    # Reindex the data_table to include all pairs
    data_table_complete = data_table.set_index(['ID_epi', 'ID_K9']).reindex(complete_pairs).reset_index()

    # Fill all missing values with 0 for numeric columns
    numeric_columns = ['N_voxels_epi', 'N_voxels_K9', 'Union', 'Dice_coeff', 'Overlap', 'Index', 'Expected_index', 'Max_index', 'ARI', 'rnd_N']
    for col in numeric_columns:
        # Fill NaN values with 0
        data_table_complete[col] = data_table_complete[col].fillna(0)
    
    # check every row
    for index, row in data_table_complete.iterrows():
        # if N_voxels_epi is 0, fill it with the ID_dict value
        if row['N_voxels_epi'] == 0:
            data_table_complete.at[index, 'N_voxels_epi'] = ID_dict[row['ID_epi']]
        # if N_voxels_K9 is 0, fill it with the ID_dict value
        if row['N_voxels_K9'] == 0:
            data_table_complete.at[index, 'N_voxels_K9'] = ID_dict[row['ID_K9']]

    # sort N_voxels_epi
    data_table_complete = data_table_complete.sort_values(by=['ID_epi', 'ID_K9'], ascending=[True, True])
    return data_table_complete



def plot_single_slice2(background_map,data_map,slice_type,slice_num,axes,cmap_background,cmap_overlay,alpha,clear_background,nan_zero=True,overlay_norm=None,flip_x=True):
    # Plot a single slice of the data_map on top of the background_map
    # slice_type: 'x', 'y', or 'z'
    # slice_num: slice number
    # nan_zero: if True, zeros in data_map are set to np.nan
    
    # create a copy of the background_map
    data_map_copy = copy.deepcopy(background_map)
    # paste data in data_map over the background_map_copy
    data_map_copy[:] = data_map[:]
    data_map = data_map_copy

    if nan_zero:
        # Ensure it's float to support NaN
        data_map = np.array(data_map, dtype=float)
        # Set zeros to NaN
        data_map[data_map == 0] = np.nan
    x_change = 0
    # displace data_map in x if positive
    if x_change > 0:
        data_map[x_change:,:,:] = data_map[:-x_change,:,:]
    elif x_change < 0:
        data_map[:x_change,:,:] = data_map[-x_change:,:,:]

    # Dog overlays require the X flip; human overlays retain their orientation.
    if flip_x:
        data_map = np.flip(data_map, axis=0)

    # print background_map class
    # print(type(background_map))
    # print data_map class
    # print(type(data_map))
    # print('Data map shape: ' + str(data_map.shape))
    slice_num_back = slice_num
    axes.clear()
    if slice_type == 'x':
        # initialize slice_background with zeros
        slice_background = np.zeros_like(background_map[slice_num_back, :, :])
        # initialize slice_overlay with zeros
        slice_overlay = np.zeros_like(data_map[slice_num, :, :])

        slice_background = background_map[slice_num_back, :, :]
        slice_overlay = data_map[slice_num, :, :]
    elif slice_type == 'y':
        # initialize slice_background with zeros
        slice_background = np.zeros_like(background_map[:, slice_num_back, :])
        # initialize slice_overlay with zeros
        slice_overlay = np.zeros_like(data_map[:, slice_num, :])

        slice_background = background_map[:, slice_num_back, :]
        slice_overlay = data_map[:, slice_num, :]
    elif slice_type == 'z':
        # initialize slice_background with zeros
        slice_background = np.zeros_like(background_map[:, :, slice_num_back])
        # initialize slice_overlay with zeros
        slice_overlay = np.zeros_like(data_map[:, :, slice_num])

        slice_background = background_map[:, :, slice_num_back]
        slice_overlay = data_map[:, :, slice_num]    
    
    


    # Apply clear_background option to slice_background
    if clear_background:
        slice_background = np.array(slice_background, dtype=float)  # Ensure it's float to support NaN
        slice_background[slice_background == 0] = np.nan
    else:
        # Plot the base slice with modified gray colormap
        axes.imshow(slice_background.T, cmap=cmap_background, origin='lower', alpha=alpha)
    # Plot the overlay slice with modified overlay colormap
    axes.imshow(slice_overlay.T, cmap=cmap_overlay, origin='lower', norm=overlay_norm)
    axes.axis('off')
    return slice_background, slice_overlay

def plot_slice2(project_dict, base_filename, figure_letter, slice_dict, data_map, alpha=1.0, 
                alphabetic_slides=False, write_file=True, clear_background=False, figure_output_folder=None,
                background_file=None, overlay_norm=None):
    # Plot a slice of the data_map on top of the atlas
    # project_dict: dictionary with the project information
    # base_filename: prefix for the output figure
    # figure_letter: letter of the figure
    # slice_dict: dictionary containing the following keys:
    #   'atlas_type': type of atlas to use
    #   'x': list of x coordinates to plot
    #   'y': list of y coordinates to plot
    #   'z': list of z coordinates to plot
    #   'cmap': colormap to use for the data_map
    #   'specie': specie of the atlas
    # data_map: 3D numpy array with the data to plot
    # alpha: transparency of the plot
    # alphabetic_slides: if True, the slices are labeled with a letter. If False the slices are labeled with the corresponding letter and number
    # write_file: if True, the figure is saved. If False, the figure is not saved
    # clear_background: True = invisible background, False = background will be visible

    project_path = get_path('project', project_dict)
    figure_folder = get_path('figures', project_dict, local_data=False, figure_letter=figure_letter)
    # if figure_output_folder is not None, use it instead of figure_folder
    if figure_output_folder is not None:
        figure_folder = figure_output_folder
        
    # An explicit atlas also supports human maps, whose atlas has no Nitzsche folder.
    # overlay_norm keeps indexed colours stable when a slice contains only a
    # subset of the labels in data_map; None preserves the original behaviour.
    if background_file is not None:
        background_map = nib.load(background_file).get_fdata()
    elif slice_dict['specie'] == 'D': # Base case for dog
        background_map = get_atlas(slice_dict['atlas_type'], 'D', 'Czeibert_brain2mm')
    elif slice_dict['specie'] == 'H':
        atlas_file = Path(__file__).resolve().parent / 'Atlas' / 'Hum' / 'MNI152_T1_2mm_brain.nii.gz'
        background_map = nib.load(atlas_file).get_fdata()
    else:
        raise ValueError(f"Unsupported species: {slice_dict['specie']}")
    #get_atlas(atlas_type='Nitzsche', specie='D', img_type='b_GreyMatter2mm')
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    # Get the colormap for overlay
    cmap_overlay = slice_dict['cmap']
    # get all unique values in data_map
    # unique_values = np.unique(data_map)
    # check that there is the same number of unique elements in the colormap and in the data_map
    # if len(unique_values) != len(cmap_overlay.colors):
    #     print('Number of unique values in data_map and colormap do not match')
    #     print('Unique values in data_map: ' + str(len(unique_values)))
    #     # print(unique_values)
    #     print('Unique values in colormap: ' + str(len(cmap_overlay.colors)))
    #     # trigger an exception
    #     raise ValueError('Number of unique values in data_map and colormap do not match')
    # else:
    #     print('Unique values in data_map: ' + str(len(unique_values)))
    #     print(unique_values)
    #     print('Unique values in colormap: ' + str(len(cmap_overlay.colors)))
        
    cmap_background = plt.get_cmap('gray')  # Get the gray colormap for the background
        
    # Figure counter for alphabetic_slides
    fig_counter = 0
    # Determine slice_type and extract the corresponding slice
    if 'x' in slice_dict:
        slice_type = 'x'
        for slice_num in slice_dict['x']:
            plot_single_slice2(background_map,data_map,slice_type,slice_num,axes,cmap_background,cmap_overlay,alpha,clear_background,overlay_norm=overlay_norm,flip_x=slice_dict['specie'] != 'H')
            if alphabetic_slides:
                # Figure path
                figure_fname = f"{base_filename}_{chr(65+fig_counter)}.png"
                fig_counter += 1
            else:
                # Figure path
                figure_fname = f"{base_filename}_{slice_type}_{slice_num}.png"
            save_fig(fig,figure_folder,figure_fname, write_file)
    if 'y' in slice_dict:
        slice_type = 'y'
        for slice_num in slice_dict['y']:
            plot_single_slice2(background_map,data_map,slice_type,slice_num,axes,cmap_background,cmap_overlay,alpha,clear_background,overlay_norm=overlay_norm,flip_x=slice_dict['specie'] != 'H')
            if alphabetic_slides:
                # Figure path
                figure_fname = f"{base_filename}_{chr(65+fig_counter)}.png"
                fig_counter += 1
            else:
                figure_fname = f"{base_filename}_{slice_type}_{slice_num}.png"
            save_fig(fig,figure_folder,figure_fname, write_file)
    if 'z' in slice_dict:
        slice_type = 'z'
        for slice_num in slice_dict['z']:
            plot_single_slice2(background_map,data_map,slice_type,slice_num,axes,cmap_background,cmap_overlay,alpha,clear_background,overlay_norm=overlay_norm,flip_x=slice_dict['specie'] != 'H')
            if alphabetic_slides:
                # Figure path
                figure_fname = f"{base_filename}_{chr(65+fig_counter)}.png"
                fig_counter += 1
            else:
                # Figure path
                figure_fname = f"{base_filename}_{slice_type}_{slice_num}.png"            
            save_fig(fig,figure_folder,figure_fname, write_file)


def plot_single_slice(atlas,data_map,slice_type,slice_num,axes,zero_transparent,cmap_background,cmap_overlay,alpha,vmin,vmax):
    '''
    Plot a single slice of the data_map on top of the atlas
    atlas: 3D numpy array with the atlas
    data_map: 3D numpy array with the data to plot
    slice_type: 'x', 'y', or 'z'
    slice_num: slice number
    zero_transparent: if True, zeros in data_map and slice_base are set to np.nan
    cmap_background: colormap to use for the atlas
    cmap_overlay: colormap to use for the data_map
    alpha: transparency of the overlay
    vmin: minimum value for the colormap
    vmax: maximum value for the colormap

    '''
    # This function is deprecated, use plot_single_slice2 instead
    ################################################################################
    # issue error and stop the function
    # raise NotImplementedError("plot_single_slice is deprecated, use plot_single_slice2 instead.")

    axes.clear()
    if slice_type == 'x':
        slice_base = atlas[slice_num, :, :]
        slice_overlay = data_map[slice_num, :, :]
    elif slice_type == 'y':
        slice_base = atlas[:, slice_num, :]
        slice_overlay = data_map[:, slice_num, :]
    elif slice_type == 'z':
        slice_base = atlas[:, :, slice_num]
        slice_overlay = data_map[:, :, slice_num]    
    
    # Apply zero_transparent option to slice_base
    if zero_transparent:
        slice_base = np.array(slice_base, dtype=float)  # Ensure it's float to support NaN
        slice_base[slice_base == 0] = np.nan

    # Plot the base slice with modified gray colormap
    axes.imshow(slice_base.T, cmap=cmap_background, origin='lower')
    axes.axis('off')

    # Plot the overlay slice with modified overlay colormap
    axes.imshow(slice_overlay.T, cmap=cmap_overlay, origin='lower', alpha=alpha, vmin=vmin, vmax=vmax)
    # return slice_base, slice_overlay

def plot_slice(project_dict, base_filename, figure_letter, slice_dict, data_map, zero_transparent, nan_transparent, alpha, alphabetic_slides=False, write_file=True, project_path=None):
    """
    Plot a slice of the data_map on top of the atlas
    project_dict: dictionary with the project information
    base_filename: prefix in the name for the output figure
    figure_letter: letter of the figure

    slice_dict: dictionary containing the following keys:
      'atlas_type': type of atlas to use
      'x': list of x coordinates to plot
      'y': list of y coordinates to plot
      'z': list of z coordinates to plot
      'cmap': colormap to use for the data_map
      'specie': specie of the atlas
      'vmin': minimum value for the colormap
      'vmax': maximum value for the colormap
    data_map: 3D numpy array with the data to plot
    zero_transparent: if True, zeros in data_map and slice_base are set to np.nan
    nan_transparent: if True, NaNs in data_map are plotted as transparent
    alpha: transparency of the overlay
    alphabetic_slides: if True, the slices are labeled with a letter. If False the slices are labeled with the corresponding letter and number
    """
    # This function is deprecated, use plot_slice2 instead
    ################################################################################
    # issue error and stop the function
    # raise NotImplementedError("plot_slice is deprecated, use plot_slice2 instead.")

    # if project_path is None, get it from project_dict
    if project_path is None:
        project_path = get_path('project', project_dict)
    figure_folder = os.path.join(project_path, 'Figures', f'Figure_{figure_letter}')
    # Apply zero_transparent option to data_map
    if zero_transparent:
        data_map[data_map == 0] = np.nan
    
    # if slice_dict does not have 'img_type', set it to 'Czeibert_brain2mm'
    if 'img_type' not in slice_dict:
        # if specie is 'D', set img_type to 'Czeibert_brain2mm'
        if slice_dict['specie'] == 'D':
            img_type = 'Czeibert_brain2mm'
        else:
            # stop the function and print an error
            raise ValueError(f"img_type not specified and specie is {slice_dict['specie']}")
    else:
        img_type = slice_dict['img_type']

    

    atlas = get_atlas(slice_dict['atlas_type'], slice_dict['specie'], img_type)
    #get_atlas(atlas_type='Nitzsche', specie='D', img_type='b_GreyMatter2mm')
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))

    # Determine vmin and vmax
    if 'vmin' in slice_dict:
        vmin = slice_dict['vmin']
    else:
        vmin = np.nanmin(data_map)
    if 'vmax' in slice_dict:
        vmax = slice_dict['vmax']
    else:
        vmax = np.nanmax(data_map)

    # Modify overlay colormap to handle NaNs if nan_transparent is True
    cmap_overlay = plt.get_cmap(slice_dict['cmap'])  # Get the colormap for overlay
    if nan_transparent:
        cmap_overlay = copy.copy(cmap_overlay)               # Copy the colormap
        cmap_overlay.set_bad(color=(1, 1, 1, 0))            # Set NaNs to be fully transparent
    else:
        print("decision pending")

    # Modify 'gray' colormap to handle NaNs if zero_transparent is True
    cmap_background = plt.get_cmap('gray')  # Get the gray colormap for the background
    if zero_transparent:
        cmap_background = copy.copy(cmap_background)       # Copy the gray colormap
        cmap_background.set_bad(color=(1, 1, 1, 0))  # Set NaNs to be fully transparent
    
    # Figure counter for alphabetic_slides
    fig_counter = 0
    # Determine slice_type and extract the corresponding slice
    if 'x' in slice_dict:
        slice_type = 'x'
        for slice_num in slice_dict['x']:
            plot_single_slice(atlas,data_map,slice_type,slice_num,axes,zero_transparent,cmap_background,cmap_overlay,alpha,vmin,vmax)
            if alphabetic_slides:
                # Figure path
                figure_fname = f"{base_filename}_{chr(65+fig_counter)}.png"
                fig_counter += 1
            else:
                # Figure path
                figure_fname = f"{base_filename}_{slice_type}_{slice_num}.png"
            save_fig(fig,figure_folder,figure_fname, write_file)
    if 'y' in slice_dict:
        slice_type = 'y'
        for slice_num in slice_dict['y']:
            plot_single_slice(atlas,data_map,slice_type,slice_num,axes,zero_transparent,cmap_background,cmap_overlay,alpha,vmin,vmax)
            if alphabetic_slides:
                # Figure path
                figure_fname = f"{base_filename}_{chr(65+fig_counter)}.png"
                fig_counter += 1
            save_fig(fig,figure_folder,figure_fname, write_file)
    if 'z' in slice_dict:
        slice_type = 'z'
        for slice_num in slice_dict['z']:
            plot_single_slice(atlas,data_map,slice_type,slice_num,axes,zero_transparent,cmap_background,cmap_overlay,alpha,vmin,vmax)
            if alphabetic_slides:
                # Figure path
                figure_fname = f"{base_filename}_{chr(65+fig_counter)}.png"
                fig_counter += 1
            else:
                # Figure path
                figure_fname = f"{base_filename}_{slice_type}_{slice_num}.png"            
            save_fig(fig,figure_folder,figure_fname, write_file)


def save_fig(fig,figure_folder,figure_fname, write_file=True):
    # Save the figure with transparency
    # fig: figure to save
    # figure_folder: folder to save the figure
    # figure_fname: name of the figure
    # write_file: if True, the figure is saved. If False, the figure is not saved
    # Check if folder exists, if not create it
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
        print(f"Folder not found, created folder {figure_folder}")
    else:
        print("Files will be saved in " + figure_folder)
    # Figure path
    figure_path = os.path.join(figure_folder, figure_fname)
    # if write_file:
    if write_file:
        # Save the figure with transparency
        plt.savefig(figure_path, bbox_inches='tight', transparent=True);
        # Indicate that the figure was saved
        print(f"Figure saved as {figure_path}")
    else:
        # Indicate that the figure was not saved
        print(f"Figure not saved as {figure_path}")


    # Close the figure to free memory
    # plt.close(fig);

def get_color(color_range, N_color):
    """
    Generate HEX color codes within a limited color range.
    
    Args:
        color_range (int): The number of possible colors to sample from.
        N_color (int): The specific index of the color within the range.
    
    Returns:
        str: HEX value of the selected color.
    """
    if N_color < 0 or N_color >= color_range:
        raise ValueError("N_color must be within the range of 0 and color_range-1")
    
    # Generate a colormap (we'll use 'viridis' as an example)
    cmap = plt.cm.get_cmap('viridis', color_range)
    
    # Get the RGB color at the given index
    rgb_color = cmap(N_color)
    
    # Convert the RGB color to HEX
    hex_color = mcolors.rgb2hex(rgb_color[:3])
    
    return hex_color


def get_atlas(atlas_type='Nitzsche', specie='D', img_type='b_GreyMatter2mm'):
    if specie == 'D':
        specieS = 'Dog'
    elif specie == 'H':
        specieS = 'Hum'
    else:
        print('Wrong specie')
        return None
    atlas_file = r"C:\github\dog_brain_toolkit\Atlas" + os.sep + specieS + os.sep + atlas_type + os.sep + img_type + ".nii.gz"
    img = nib.load(atlas_file)
    atlas = img.get_fdata()
    return atlas


def filter_max(results_path,comp_type,number_of_clusters, n_reps, stat_to_max, allow_repeat,rnd=True):
    # This function filters the data_table to get the row with the highest stat_to_max for each ID_K9
    # results_path: path to the results folder
    # comp_type: type of comparison ['beagle','others','all']
    # number_of_clusters: number of clusters
    # n_reps: number of repetitions
    # stat_to_max: statistic to maximize
    # allow_repeat: allow repeated ID_epi
    # rnd: if True, it will load the rnd files
    results_path = results_path + os.sep + 'cluster_stats'
    final_table = pd.DataFrame()
    missing_log = []
    for rnd_N in range(n_reps+1):
        if rnd:
            data_table_path = os.path.join(results_path, 'rnd', 'data_table_' + str(number_of_clusters) + '_' + comp_type + '_' + str(rnd_N).zfill(4) + '.csv')
            # check if the file exists
            # if not os.path.exists(data_table_path):
            #     # print the missing file
            #     print('File not found: ' + data_table_path)
            #     # print skipping the current number of clusters
            #     print('Skipping number of clusters: ' + str(number_of_clusters))
            #     # add to missing_log
            #     missing_log.append(data_table_path)
            #     break
        else:
            # get the data_table_path
            data_table_path = os.path.join(results_path, 'data_table_' + str(number_of_clusters) + '_' + comp_type + '.csv')

        # check if the file exists
        if not os.path.exists(data_table_path):
            # print the missing file
            print('File not found: ' + data_table_path)
            # print skipping the current number of clusters
            print('Skipping number of clusters: ' + str(number_of_clusters))
            # add to missing_log
            missing_log.append(data_table_path)
            # skip to the next number of clusters
            continue
            #break
    
        # print table to load
        print('Loading table: ' + data_table_path)
        # load the data_table
        data_table = pd.read_csv(data_table_path)
        # Add number of clusters to the data_table
        data_table['N_clusters'] = number_of_clusters
        
        # get sorted list of unique ID_K9
        ID_K9_list = data_table['ID_K9'].unique()
        # sort the list
        ID_K9_list.sort()

        # list to keep track of the matching ID_epi
        matching_ID_epi = []


        # for every ID_K9, get the highest matching ID_epi
        for ID_K9 in ID_K9_list:
            # print the current ID
            # print('ID: ' + str(ID_K9))
            data_table_sel = filter_table(data_table, ID_K9, stat_to_max, allow_repeat)
            # add rnd_N to the loaded row
            data_table_sel['rnd_N'] = rnd_N
            # add the ID_epi to the matching_ID_epi list
            matching_ID_epi.append(data_table_sel['ID_epi'].values[0])

            # add the row to the final_table
            final_table = pd.concat([final_table, data_table_sel], ignore_index=True)
    
    return final_table

def get_path(path_label, project_dict, local_data=True, rnd=False, figure_letter='A'):
    # if project_dict has no experiment, set it to 'Segmentation'
    if 'Project' not in project_dict:
        project = 'Segmentation'
    else:
        project = project_dict['Project']
    
    #### remove username#####
    # if Dataset is present, take dataset, otherwise take database
    if 'Dataset' in project_dict:
        dataset = project_dict['Dataset']
    else:
        dataset = project_dict['database']
    # if dataset == 'CAPS_K9':
    #     dataset = 'Segmentation'
    
    if path_label == 'results' or path_label == 'Results':
        if os.name == 'nt':
            if local_data:
                path = r"G:\My Drive\Results\\" + project
            else:
                path = r"P:\userdata\raulh87\data" + os.sep + dataset + os.sep + 'results'
        else:
            path = os.path.join('/home', 'raulh87', 'mnt', 'a471', 'userdata', 'raulh87', 'data', dataset, 'results')
        
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
                path = r"P:\userdata\raulh87" + os.sep + project
        else:
            path = '/home' + os.sep + username + '/mnt/a471/userdata/' + username + os.sep + project
        
    else:
        print('Path label not recognized')
    return path

# def find_best_match(data_table, stat_to_max='Dice_coeff'):


def filter_table(data_table,ID,stat_to_max='Dice_coeff',allow_repeat=False, max_stat=False):
    stats_to_use = ['Dice_coeff', 'ARI', 'Overlap']
    # list to keep track of the matching ID_epi
    matching_ID_epi = []    
    # if not allow_repeat:
    #     # print the current ID
    #     print('ID: ' + str(ID))
    # get the rows with the current ID_K9
    data_table_sel = data_table[data_table['ID_K9'] == ID]
    
    if max_stat:
        # get the row with the highest stat_to_max
        data_table_sel = data_table_sel[data_table_sel[stat_to_max] == data_table_sel[stat_to_max].max()]
        # save the ID_epi to a new variable
        ID_epi = data_table_sel['ID_epi'].values[0]
        # check allow_repeat
        if not allow_repeat:
            # check if the ID_epi is already in the matching_ID_epi list
            while ID_epi in matching_ID_epi:
                # if it is, filter again the data_table_sel
                data_table_sel = data_table[data_table['ID'] == ID]
                # remove row with the current ID_epi
                data_table_sel = data_table_sel[data_table_sel['ID_epi'] != ID_epi]
                # get again the row with the highest stat_to_max
                data_table_sel = data_table_sel[data_table_sel[stat_to_max] == data_table_sel[stat_to_max].max()]
                # save the ID_epi to a new variable
                ID_epi = data_table_sel['ID_epi'].values[0]

        # add the ID_epi to the matching_ID_epi list
        matching_ID_epi.append(data_table_sel['ID_epi'].values[0])
    else:
        # generate new dataframe with same columns
        data_table_tmp = pd.DataFrame(columns=data_table.columns)
        # for every stat in stats_to_use
        for stat in stats_to_use:
            # get the row with the highest stat
            tmp_vals = data_table_sel[stat].values
            # add average to data_table_tmp
            data_table_tmp[stat] = [tmp_vals.mean()]

        # add the ID_epi to the matching_ID_epi list
        data_table_tmp['ID_K9'] = ID
        # add N_clusters to the data_table_tmp
        data_table_tmp['N_clusters'] = data_table_sel['N_clusters'].values[0]
        # copy data_table_tmp over data_table_sel
        data_table_sel = data_table_tmp

    return data_table_sel

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment   # SciPy ≥ 1.4
except ImportError as e:
    raise ImportError("SciPy is required for this implementation: "
                      "pip install scipy") from e


import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment   # SciPy ≥ 1.4

def best_assignment_Kuhn(
    df: pd.DataFrame,
    row_id: str = "ID_K9",
    col_id: str = "ID_epi",
    value:  str = "Dice_coeff",
    fill_value: float = 0.0,
    return_matrix: bool = False,
):
    """
    Solve the maximum-weight assignment problem for a long-format DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain the columns [row_id, col_id, value].
    row_id, col_id, value : str
        Names of the row labels, column labels, and score column.
    fill_value : float
        Value to use for missing pairs when densifying.
    return_matrix : bool
        If True, also return the dense matrix and label lists.

    Returns
    -------
    assignment_df : pandas.DataFrame
        Columns [row_id, col_id, value] of the optimal matching.
    total_score : float
        Sum of the chosen `value`s.
    (optionally) score_matrix, rows, cols
    """
    # 1) Get unique labels and remember original counts
    unique_rows = list(df[row_id].unique())
    unique_cols = list(df[col_id].unique())
    n_rows0, n_cols0 = len(unique_rows), len(unique_cols)

    # 2) Pad to square
    rows = unique_rows.copy()
    cols = unique_cols.copy()
    if n_rows0 > n_cols0:
        # pad cols
        for k in range(n_rows0 - n_cols0):
            cols.append(f"__DUMMY_col_{k}")
    elif n_cols0 > n_rows0:
        # pad rows
        for k in range(n_cols0 - n_rows0):
            rows.append(f"__DUMMY_row_{k}")
    n = len(rows)

    # 3) Build dense score matrix
    score_matrix = np.full((n, n), fill_value, dtype=float)
    row_idx = {r: i for i, r in enumerate(rows)}
    col_idx = {c: j for j, c in enumerate(cols)}
    for r, c, v in df[[row_id, col_id, value]].itertuples(index=False):
        score_matrix[row_idx[r], col_idx[c]] = v

    # 4) Convert to cost and solve
    cost_matrix = -score_matrix
    r_ind, c_ind = linear_sum_assignment(cost_matrix)

    # 5) Build assignment list, skipping dummy–dummy
    assignments = []
    for i, j in zip(r_ind, c_ind):
        # skip if both are padding
        if i >= n_rows0 and j >= n_cols0:
            continue
        assignments.append((rows[i], cols[j], score_matrix[i, j]))

    assignment_df = pd.DataFrame(assignments, columns=[row_id, col_id, value])
    total_score = assignment_df[value].sum()

    if return_matrix:
        return assignment_df, total_score, score_matrix, rows, cols
    return assignment_df



def filter_table_Dice_coeff(data_table, rnd_N=0):
    '''
    Filter the data_table to get the best matches for the Dice_coeff
    returns data_table_sel
    '''
    stat_to_max= 'Dice_coeff'
    # get unique values from ID_K9
    ID_K9_list = data_table['ID_K9'].unique()
    # get unique values from ID_epi
    ID_epi_list = data_table['ID_epi'].unique()

    # initialize numpy array to contain all pairs from data_table
    M = np.zeros((len(ID_K9_list), len(ID_epi_list)))
    M_stat = np.zeros((len(ID_K9_list), len(ID_epi_list)))
    # print(ID_epi_list)
    # print(ID_K9_list)
    # iterate over each row
    for index,row in data_table.iterrows():
        # get the ID_K9
        ID_K9 = row['ID_K9']
        # get the ID_epi
        ID_epi = row['ID_epi']
        # get the Dice_coeff
        stat = row[stat_to_max]
        # assign the Dice_coeff to the position in M_stat[ID_K9, ID_epi]
        M_stat[ID_K9_list == ID_K9, ID_epi_list == ID_epi] = stat
        # M_stat[ID_K9, ID_epi] = stat

    # print(M_stat)
    # flip M to get the cost matrix
    cost_matrix = 1 - M_stat
    # Apply selection algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Extract the best correspondence
    best_matches = list(zip(row_ind, col_ind))
    # Extract the values from M
    # best_values = [M_stat[i, j] for i, j in best_matches]
    # print('Best values: ' + str(best_values))

    # initialize rows
    rows = []
    # for each match
    for match in best_matches:
        # get the row from data_table
        ID_K9 = ID_K9_list[match[0]]
        ID_epi = ID_epi_list[match[1]]
        # get stat_val
        stat_val = M_stat[match[0], match[1]]
        # initialize row
        row = {'ID_K9': ID_K9, 'ID_epi': ID_epi, 'Dice_coeff': stat_val, 'rnd_N': rnd_N}
        # append row to rows
        rows.append(row)
    # create data_table_sel
    data_table_sel = pd.DataFrame(rows)    
    
    # return data_table_sel
    return data_table_sel

def get_datafolder(username, local_data=True):
    # This function returns the path to the data folder
    if os.name == 'nt':
        if local_data:
            datafolder = r"C:\data"
        else:
            datafolder = r"P:\userdata" + os.sep + username + r"\data"
    else:
        datafolder = '/home' + os.sep + username + '/mnt/a471/userdata/' + username + '/data'
    return datafolder

def make_rnd_map(project_dict, number_of_clusters, mask="b_GreyMatter2mm", corr_type="t", rnd_N=0):
    # 
    username = project_dict['User']
      # takes t or s
    datafolder = get_datafolder(username, local_data=True)

    cluster_map_folder = os.path.join(datafolder, project_dict['Dataset'], 'hierarchical_clustering', 'rnd')
    
    rnd_N_S = str(rnd_N).zfill(4)
    cluster_map_path_rnd = os.path.join(
        cluster_map_folder,
        f"{project_dict['Specie']}-group-{corr_type}corr-{mask}_{number_of_clusters}_{rnd_N_S}.nii.gz"
    )
    # Check if the folder exists, if not create it
    if not os.path.exists(cluster_map_folder):
        os.makedirs(cluster_map_folder)

    # Loads an existing map to be used as base -------------------
    
    cluster_map_path = os.path.join(
        datafolder,
        project_dict['Dataset'],
        'hierarchical_clustering',
        f"{project_dict['Specie']}-group-{corr_type}corr-{mask}_{number_of_clusters}.nii.gz"
    )

    # Load the cluster map and get the original shape
    original_img = nib.load(cluster_map_path)
    cluster_map = original_img.get_fdata()
    original_shape = cluster_map.shape

    # Flatten the cluster map
    cluster_map = cluster_map.flatten()

    # Generate list of clusters
    clusters_ID_list = list(set(cluster_map.flatten()))
    # Remove 0 from the list
    clusters_ID_list.remove(0)

    clusters = {}
    # Find the position of each element in the cluster_ID_list
    for cluster_ID in clusters_ID_list:
        clusters[cluster_ID] = [i for i, x in enumerate(cluster_map) if x == cluster_ID]

    # Finished the loading part --------------------

    all_voxels = []
    # Get the voxels for each cluster
    for key in clusters.keys():
        all_voxels.extend(clusters[key])

    # Clusters randomized
    clusters_rnd = dict()

    # Get a random sample based on the number of voxels of the key, remove the sampled voxels from the list
    for key in clusters.keys():
        clusters_rnd[key] = np.random.choice(all_voxels, len(clusters[key]), replace=False)
        all_voxels = list(set(all_voxels) - set(clusters_rnd[key]))

    # Initialize a new cluster map with zeros
    new_cluster_map = np.zeros_like(cluster_map)

    # Assign new cluster IDs to the randomized voxel indices
    for cluster_ID in clusters_rnd:
        new_cluster_map[clusters_rnd[cluster_ID]] = cluster_ID

    # Reshape the new cluster map to the original shape
    new_cluster_map = new_cluster_map.reshape(original_shape)

    # Create a new NIfTI image using the new cluster map and the original affine
    new_img = nib.Nifti1Image(new_cluster_map, affine=original_img.affine)

    # Save the new NIfTI file using the specified path
    nib.save(new_img, cluster_map_path_rnd)

def which_cluster_files(project_dict, number_of_clusters):
    mask = "b_GreyMatter2mm"
    corr_type = "t"  # Can be 't' or 's'
    username = project_dict.get('User')
    dataset = project_dict.get('Dataset')
    specie = project_dict.get('Specie')
    datafolder = get_datafolder(username, local_data=False)
    base_dir = os.path.join(
        datafolder,
        dataset,
        'hierarchical_clustering',
        'rnd'
    )

    # Compile regex pattern to match filenames
    pattern = re.compile(
        rf"{re.escape(specie)}-group-{re.escape(corr_type)}corr-{re.escape(mask)}_(\d{{4}})_{number_of_clusters}\.nii\.gz$"
    )

    perms_numbers = []
    perms_files = []

    # indicate the directory to search
    # print('Searching in: ' + base_dir)
    # indicate the pattern to search
    # print('Pattern: ' + pattern.pattern)

    # Use os.scandir for efficient directory traversal
    with os.scandir(base_dir) as entries:
        for entry in entries:
            if entry.is_file():
                match = pattern.match(entry.name)
                if match:
                    perms_numbers.append(int(match.group(1)))
                    perms_files.append(entry.path)

    # Sort the numbers and files based on perms_numbers
    order = np.argsort(perms_numbers)
    perms_numbers = [perms_numbers[i] for i in order]
    perms_files = [perms_files[i] for i in order]

    return perms_numbers                                                                                 





import os
import nibabel as nib
import random

def get_clusters(project_dict, number_of_clusters, datafolder, rnd=False, rnd_N=0, shuffle=False, filter_with_mask=False, local_data=False):
    """
    This function returns a dictionary with the clusters and a list with the cluster IDs.
    
    Parameters
    ----------
    project_dict: dict 
        Dictionary with the project information.
    number_of_clusters: int 
        Number of clusters selected for the parcellation.
    rnd: bool 
        Whether to load the random cluster map (default: False).
    rnd_N: int 
        Index for the random cluster map (default: 0).
    shuffle: bool 
        If True, reassign all voxel indices among clusters randomly 
        (preserving cluster sizes, but randomizing their voxel positions).
    filter_with_mask: bool
        If True, apply a mask to the cluster map, removing voxels outside the mask.
    
    Returns
    -------
    clusters: dict 
        Dictionary of clusters, where each key is a cluster ID and
        each value is a list of voxel indices belonging to that cluster.
    clusters_ID_list: list 
        List of cluster IDs.
    """

    mask = "b_GreyMatter2mm"
    corr_type = "t"  # Takes 't' or 's'
    # username = project_dict['User']
    # datafolder = get_datafolder(username, local_data=local_data)
    if project_dict['Dataset'] == 'Czeibert': # this is a special case for Czeibert
        if os.name == 'nt':
            cluster_map_path = r"C:\github\dog_brain_toolkit\Atlas\Dog\Nitzsche\Czeibert2mm\b_GreyMatter_labels.nii.gz"
        else:
            cluster_map_path = '/home/raulh87/mnt/a471/userdata/raulh87/github/dog_brain_toolkit/Atlas/Dog/Nitzsche/Czeibert2mm/b_GreyMatter_labels.nii.gz'
    elif project_dict['Dataset'] == 'Johnson': # this is a special case for Johnson
        if os.name == 'nt':
            cluster_map_path = r"C:\github\dog_brain_toolkit\Atlas\Dog\Nitzsche\Johnson_cortical-wins_labels2mm.nii.gz"
        else:
            cluster_map_path = '/home/raulh87/mnt/a471/userdata/raulh87/github/dog_brain_toolkit/Atlas/Dog/Nitzsche/Johnson_cortical-wins_labels2mm.nii.gz'
    elif project_dict['Dataset'] == 'Johnson_gyri': # this is a special case for Johnson
        if os.name == 'nt':
            cluster_map_path = r"C:\github\dog_brain_toolkit\Atlas\Dog\Nitzsche\Johnson_gyri_cortical-wins_labelsGreyMatter2mm.nii.gz"
        else:
            cluster_map_path = '/home/raulh87/mnt/a471/userdata/raulh87/github/dog_brain_toolkit/Atlas/Dog/Nitzsche/Johnson_gyri_cortical-wins_labelsGreyMatter2mm.nii.gz'
    elif project_dict['Dataset'] == 'Johnson_lobe': # this is a special case for Johnson
        if os.name == 'nt':
            cluster_map_path = r"C:\github\dog_brain_toolkit\Atlas\Dog\Nitzsche\Johnson_lobe_labelsGreyMatter2mm.nii.gz"
        else:
            cluster_map_path = '/home/raulh87/mnt/a471/userdata/raulh87/github/dog_brain_toolkit/Atlas/Dog/Nitzsche/Johnson_lobe_labelsGreyMatter2mm.nii.gz'
    else:
        if rnd:
            cluster_map_path = os.path.join(
                datafolder, project_dict['Dataset'], 'hierarchical_clustering', 'rnd',
                f"{project_dict['Specie']}-group-{corr_type}corr-{mask}_"
                f"{str(rnd_N).zfill(4)}_{number_of_clusters}.nii.gz"
            )
            # Check if file exists
            if not os.path.exists(cluster_map_path):
                print(f"File {cluster_map_path} does not exist, skipping...")
                return 0, 0
        else:
            cluster_map_path = os.path.join(
                datafolder, project_dict['Dataset'], 'hierarchical_clustering',
                f"{project_dict['Specie']}-group-{corr_type}corr-{mask}_"
                f"{number_of_clusters}.nii.gz"
            )
    # print (cluster_map_path)
    print('Loading cluster map from: ' + cluster_map_path)
    # Load the cluster map
    cluster_map = nib.load(cluster_map_path).get_fdata()
    
    # if filter_with_mask:
    if filter_with_mask:
        # Load the mask
        mask_path = os.path.join(datafolder, 'CAPS_K9', 'masks', f'{mask}.nii.gz')
        # Load the mask
        mask = nib.load(mask_path).get_fdata()
        # Apply the mask to the cluster map
        cluster_map[mask == 0] = 0
        print('Applied mask from: ' + mask_path)

    # Generate list of cluster IDs
    clusters_ID_list = list(set(cluster_map.flatten()))
    # Remove 0 (background)
    if 0 in clusters_ID_list:
        clusters_ID_list.remove(0)

    # Flatten to a 1D vector
    cluster_map_flatten = cluster_map.flatten()

    # Build the dictionary: cluster_id -> list of voxel indices
    clusters = {}
    for cluster_ID in clusters_ID_list:
        clusters[cluster_ID] = [i for i, x in enumerate(cluster_map_flatten) if x == cluster_ID]

    # If shuffle is requested, gather all voxel indices, shuffle them,
    # and then reassign to each cluster, preserving the cluster sizes.
    if shuffle:
        # apply shuffle
        clusters = shuffle_clusters(clusters, clusters_ID_list)

    return clusters, clusters_ID_list, cluster_map_path

def get_clusters2(project_dict, number_of_clusters, rnd=False, rnd_N=0, shuffle=False, local_data=False):
    # should be an exact copy of the above without the mask
    """
    This function returns a dictionary with the clusters and a list with the cluster IDs.
    
    Parameters
    ----------
    project_dict: dict 
        Dictionary with the project information.
    number_of_clusters: int 
        Number of clusters selected for the parcellation.
    rnd: bool 
        Whether to load the random cluster map (default: False).
    rnd_N: int 
        Index for the random cluster map (default: 0).
    shuffle: bool 
        If True, reassign all voxel indices among clusters randomly 
        (preserving cluster sizes, but randomizing their voxel positions).
    
    Returns
    -------
    clusters: dict 
        Dictionary of clusters, where each key is a cluster ID and
        each value is a list of voxel indices belonging to that cluster.
    clusters_ID_list: list 
        List of cluster IDs.
    """

    mask = "b_GreyMatter2mm"
    corr_type = "t"  # Takes 't' or 's'
    # if username is available in project_dict['User'], use it
    if 'User' in project_dict:
        username = project_dict['User']
    else:
        username = 'raulh87'
    datafolder = get_datafolder(username, local_data=local_data)
    if project_dict['Dataset'] == 'Czeibert': # this is a special case for Czeibert
        if os.name == 'nt':
            cluster_map_path = r"C:\github\dog_brain_toolkit\Atlas\Dog\Nitzsche\Czeibert2mm\b_GreyMatter_labels.nii.gz"
        else:
            cluster_map_path = '/home/raulh87/mnt/a471/userdata/raulh87/github/dog_brain_toolkit/Atlas/Dog/Nitzsche/Czeibert2mm/b_GreyMatter_labels.nii.gz'
    elif project_dict['Dataset'] == 'Johnson': # this is a special case for Johnson
        if os.name == 'nt':
            cluster_map_path = r"C:\github\dog_brain_toolkit\Atlas\Dog\Nitzsche\Johnson_cortical-wins2mm.nii.gz"
        else:
            cluster_map_path = '/home/raulh87/mnt/a471/userdata/raulh87/github/dog_brain_toolkit/Atlas/Dog/Nitzsche/Johnson_cortical-wins2mm.nii.gz'                   
    else:
        if rnd:
            cluster_map_path = os.path.join(
                datafolder, project_dict['Dataset'], 'hierarchical_clustering', 'rnd',
                f"{project_dict['Specie']}-group-{corr_type}corr-{mask}_"
                f"{str(rnd_N).zfill(4)}_{number_of_clusters}.nii.gz"
            )
            # Check if file exists
            if not os.path.exists(cluster_map_path):
                print(f"File {cluster_map_path} does not exist, skipping...")
                return 0, 0
        else:
            cluster_map_path = os.path.join(
                datafolder, project_dict['Dataset'], 'hierarchical_clustering',
                f"{project_dict['Specie']}-group-{corr_type}corr-{mask}_"
                f"{number_of_clusters}.nii.gz"
            )

    # Load the cluster map
    cluster_map = nib.load(cluster_map_path).get_fdata()

    # Generate list of cluster IDs
    clusters_ID_list = list(set(cluster_map.flatten()))
    # Remove 0 (background)
    if 0 in clusters_ID_list:
        clusters_ID_list.remove(0)

    # Flatten to a 1D vector
    cluster_map_flatten = cluster_map.flatten()

    # Build the dictionary: cluster_id -> list of voxel indices
    clusters = {}
    for cluster_ID in clusters_ID_list:
        clusters[cluster_ID] = [i for i, x in enumerate(cluster_map_flatten) if x == cluster_ID]

    # If shuffle is requested, gather all voxel indices, shuffle them,
    # and then reassign to each cluster, preserving the cluster sizes.
    if shuffle:
        # apply shuffle
        clusters = shuffle_clusters(clusters, clusters_ID_list)

    return clusters, clusters_ID_list, cluster_map

def shuffle_clusters(clusters, clusters_ID_list):
     # Gather all voxel indices from all clusters
    all_voxels = []
    for cluster_ID in clusters_ID_list:
        all_voxels.extend(clusters[cluster_ID])

    # Shuffle them in place
    random.shuffle(all_voxels)

    # Re-distribute to each cluster preserving original count
    shuffled_clusters = {}
    start_idx = 0
    for cluster_ID in clusters_ID_list:
        n_vox = len(clusters[cluster_ID])
        shuffled_clusters[cluster_ID] = all_voxels[start_idx : start_idx + n_vox]
        start_idx += n_vox

    clusters = shuffled_clusters
    return clusters

def get_clusters_deprecated(project_dict, number_of_clusters, rnd=False, rnd_N=0): # this was used for tne invididual segment analysis
    # This function returns a dictionary with the clusters and a list with the cluster IDs
    # project_dict: dictionary with the project information
    # number_of_clusters: number of clusters selected for the parcellation
    # returns: clusters, clusters_ID_list
    # clusters: dictionary with the clusters
    # clusters_ID_list: list with the cluster IDs

    mask = "b_GreyMatter2mm"
    corr_type = "t" #takes t or s
    username = project_dict['User']
    datafolder = get_datafolder(username, local_data=True)

    if rnd:
        cluster_map_path = os.path.join(datafolder, project_dict['Dataset'], 'hierarchical_clustering',
                                        'rnd',
                            project_dict['Specie'] + '-group-' + corr_type + 'corr-' + mask + '_' + str(number_of_clusters) + '_' + str(rnd_N).zfill(4) + '.nii.gz')
        # check if the file exist
        if not os.path.exists(cluster_map_path):
            make_rnd_map(project_dict, number_of_clusters, mask, corr_type, rnd_N)
    
    else:
        cluster_map_path = os.path.join(datafolder, project_dict['Dataset'], 'hierarchical_clustering',
                            project_dict['Specie'] + '-group-' + corr_type + 'corr-' + mask + '_' + str(number_of_clusters) + '.nii.gz')

    # Load the cluster map
    cluster_map = nib.load(cluster_map_path).get_fdata()

    # generate list of clusters
    clusters_ID_list = list(set(cluster_map.flatten()))
    # remove 0 from the list
    clusters_ID_list.remove(0)

    # transform the cluster map into a vector
    cluster_map = cluster_map.flatten()

    clusters = {}
    # find the position of each element in the cluster_ID_list
    for cluster_ID in clusters_ID_list:
        clusters[cluster_ID] = [i for i, x in enumerate(cluster_map) if x == cluster_ID]
    
    return clusters,clusters_ID_list


def get_homogeinity(project_dict, datafolder, sub_N, vox_list):
    # This function calculates the homogeneity of a parcellation
    # project_dict: dictionary with the project information
    # sub_N: subject number
    # vox_list: list of voxels in the cluster

    # username = project_dict['User']
    # if os.name == 'nt':
    #     datafolder = r"P:\userdata" + os.sep + username + r"\data"
    #     #datafolder = r"C:\data"
    # else:
    #     datafolder = '/home' + os.sep + username + '/mnt/a471/userdata/' + username + '/data'

    # path to the functional nii file
    func_path = os.path.join(datafolder, project_dict['Dataset'], 'normalized', project_dict['Specie'] + '-sub-' + str(sub_N).zfill(2),
                            project_dict['Specie'] + '-sub-' + str(sub_N).zfill(2) + '_ses-01'+ '_task-' + project_dict['Task'] + '_run-' + str(project_dict['Runs'][0]).zfill(2) + '.nii.gz')

    # load the nii file
    func_nii = nib.load(func_path).get_fdata()
    # flatten func_nii keeping the time dimension (4th dimension)
    func_nii_flat = np.reshape(func_nii, (-1, func_nii.shape[-1]))

    # initialize the corr_list
    # results = []
    corr_list = []

    for n,nVox1 in enumerate(vox_list):
        # if len of vox_list is bigger than 100, then
        # print every 50th iteration
        if len(vox_list) > 100:
            if n % 50 == 0:
                print(f"Processing voxel {n+1} of {len(vox_list)}")
        for nVox2 in vox_list[n+1:]:
            #getting the time series for both voxels
            time_series1 = func_nii_flat[nVox1]
            time_series2 = func_nii_flat[nVox2]
                
            # Calculate the Pearson correlation between the two time series
            corr = np.corrcoef(time_series1, time_series2)[0, 1]
            # if corr is nan, continue to the next iteration
            if np.isnan(corr):
                continue
            
            corr_list.append(corr)
            # Store the results
            # results.append([nVox1, nVox2, corr])

    # calculate average correlation
    average_corr = np.mean(corr_list)
    # calculate number of correlations
    n_corr = len(corr_list)
    return average_corr, n_corr, datafolder


def get_project_dict(dict_type):
    dicts = {
        "beagle": {
                "Project": "Segmentation",
                "Dataset": "CAPS_epi1",
                "Session": "",
                "Task": "rest",
                "Participants": [1,2,3,4,5,6,7,8,9,10],
                "Runs": [1],
                "Specie": "D",
                "Atlas_type": "Nitzsche",
                "color": "#4BACC6", # blue
            },
        "others": { 
                "Project": "Segmentation",
                "Dataset": "CAPS_epi2",
                "Session": "",
                "Task": "rest",
                "Participants": [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37],
                "Runs": [1],
                "Specie": "D",
                "Atlas_type": "Nitzsche",
                "color": "#CF005B", # red
            },
        "healthy": { 
                "Project": "Segmentation",
                "Dataset": "CAPS_epi",
                "Session": "",
                "Task": "rest",
                "Participants": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                "Runs": [1],
                "Specie": "D",
                "Atlas_type": "Nitzsche",
                "color": "#CF005B", # red
            },
        "all" : {
                "Project": "Segmentation",
                "Dataset": "CAPS_epi",
                "Session": "",
                "Task": "rest",
                "Participants": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37],
                "Runs": [1],
                "Specie": "D",
                "Atlas_type": "Nitzsche",
                "color": "9bbb59", # green
            },
        "K9" : {
                "Project": "Segmentation",
                "Dataset": "CAPS_K9",
                "Session": "",
                "Task": "resting_state",
                "Participants": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,22,23,27,28,30,31,34,36,37],
                "Runs": [1],
                "Specie": "D",
                "Atlas_type": "Nitzsche",
                "color": "f9b500", # yellow
            },
        "CAPS_K9" : {
                "Project": "Segmentation",
                "Dataset": "CAPS_K9",
                "Session": "01",
                "Task": "resting_state",
                "Participants": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,22,23,27,28,30,31,34,36,37],
                "Runs": [1],
                "Specie": "D",
                "Atlas_type": "Nitzsche",
                "color": "f9b500", # yellow
            },
        "CAPS_Knee" : {
                "Project": "Segmentation",
                "Dataset": "CAPS_Knee",
                "Session": "01",
                "Task": "resting_state",
                "Participants": [2,5,6,7,9,11,12,14,18,27],
                "Runs": [1],
                "Specie": "D",
                "Atlas_type": "Nitzsche",
                "color": "f9b500", # yellow
            },
        "Knee" : {
                "Project": "Segmentation",
                "Dataset": "CAPS_Knee",
                "Session": "01",
                "Task": "resting_state",
                "Participants": [2,5,6,7,9,11,12,14,18,27],
                "Runs": [1],
                "Specie": "D",
                "Atlas_type": "Nitzsche",
                "color": "f9b500", # yellow
            },
        "K9p" : {
                "Project": "Segmentation",
                "Dataset": "CAPS_K9",
                "Session": "",
                "Task": "resting_state",
                "Participants": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,22,23,27,28,30,31,34,36,37],
                "Runs": [1],
                "Specie": "D",
                "Atlas_type": "Nitzsche",
                "color": "f9b500", # yellow
            },
        "Czeibert" : {
                "Project": "Segmentation",
                "Dataset": "Czeibert",
                "Session": "",
                "Task": "resting_state",
                "Participants": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,22,23,27,28,30,31,34,36,37],
                "Runs": [1],
                "Specie": "D",
                "Atlas_type": "Nitzsche",
                "color": "f9b500", # yellow
            },
        "Johnson" : {
                "Project": "Segmentation",
                "Dataset": "Johnson",
                "Session": "",
                "Task": "resting_state",
                "Participants": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,22,23,27,28,30,31,34,36,37],
                "Runs": [1],
                "Specie": "D",
                "Atlas_type": "Nitzsche",
                "color": "#1B3B6F", # blue
            },
        "Johnson_gyri" : {
                "Project": "Segmentation",
                "Dataset": "Johnson_gyri",
                "Session": "",
                "Task": "resting_state",
                "Participants": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,22,23,27,28,30,31,34,36,37],
                "Runs": [1],
                "Specie": "D",
                "Atlas_type": "Nitzsche",
                "color": "#6DA34D", # green
            },
        "Johnson_lobe" : {
                "Project": "Segmentation",
                "Dataset": "Johnson_lobe",
                "Session": "",
                "Task": "resting_state",
                "Participants": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,22,23,27,28,30,31,34,36,37],
                "Runs": [1],
                "Specie": "D",
                "Atlas_type": "Nitzsche",
                "color": "#E07A5F", # orange
            },
        }
    # get list of keys
    keys = list(dicts.keys())

    return dicts[dict_type],keys


import nibabel as nib

def load_BOLD(datafolder,dataset,specie,sub_N,task,run_N):
    '''
    This function will load the BOLD file of a sub_N, task and run
    datafolder: path to the data folder
    dataset: name of the dataset
    specie: specie of the participant
    sub_N: subject number
    task: task name
    run_N: run number
    '''

    project_folder = datafolder + os.sep + dataset


    participant = specie + '-sub-' + str(sub_N).zfill(3)

    nii_path = project_folder + os.sep + 'normalized' + os.sep + participant + os.sep + participant + '_task-' + task + '_run-' + str(run_N).zfill(2) + '.nii.gz'
    
    # load nii file
    img = nib.load(nii_path)

    # Get the data from the image
    data = img.get_fdata()

    return data


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

def plot_and_save_colorbar(
    output_file,
    cmap="hot",
    font_family="Arial",
    font_size=12,
    font_color="black",
    range_values=(0, 1),
    label="",
    save_file=True,
    tick_position="bottom",
    tick_format=None,
    size=(0.03, 0.4),
    ):
    """
    Creates and saves a colorbar with customizable parameters, including tick position.

    Parameters:
    - output_file (str): Name of the output file (default: 'colorbar.png').
    - cmap (str): The colormap to use (default: 'viridis').
    - font_family (str): Font family for the labels (default: 'sans-serif').
    - font_size (int): Font size for the labels (default: 12).
    - font_color (str): Font color for the labels (default: 'black').
    - range_values (tuple): Range of values to display (default: (0, 1)).
    - label (str): Label for the colorbar (default: '').
    - save_file (bool): Whether to save the colorbar to a file (default: True).
    - tick_position (str): Position of the ticks. 
                          Accepts: "bottom", "top", "left", "right".
                          Automatically sets bar orientation:
                            - "bottom" or "top" => horizontal
                            - "left" or "right" => vertical
    - size (tuple): (width, height) of the drawn bar as a fraction of the file, e.g.
                    (0.08, 0.7) for a vertical bar, (0.7, 0.08) for a horizontal one.
                    The file itself has a fixed size per orientation (2.5x6 in vertical,
                    6x2.5 in horizontal), so every call gives identical files and bars.
    """

    # Determine orientation based on tick_position
    if tick_position in ["bottom", "top"]:
        orientation = "horizontal"
    else:
        orientation = "vertical"

    # Fixed canvas per orientation: every file has the same pixel size, and the bar sits
    # at the same place and size in it (nothing is auto-cropped or auto-laid-out)
    if orientation == "horizontal":
        figsize = (6, 2.5)
        w, h = size
        rect = [(1 - w) / 2, 0.6, w, h]            # centred, ticks/labels below
    else:
        figsize = (2.5, 6)
        w, h = size
        rect = [0.15, (1 - h) / 2, w, h]           # ticks/labels to the right

    # Normalize the range of values
    norm = Normalize(vmin=range_values[0], vmax=range_values[1])

    # Create the figure and the axis at the exact position and size requested
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)

    # Create the colorbar
    cb = ColorbarBase(ax, cmap=plt.get_cmap(cmap), norm=norm, orientation=orientation)

    # Set the colorbar label with specified font properties
    if label:
        cb.set_label(label, fontfamily=font_family, fontsize=font_size, color=font_color)

    # Remove the colorbar border
    cb.outline.set_visible(False)

    # Show only min and max ticks
    cb.set_ticks([range_values[0], range_values[1]])

    # Tick label text: printf-style tick_format (e.g. '%.2f'), or the raw value if None
    tick_labels = [(tick_format % v) if tick_format else f"{v}" for v in range_values]

    # Set the tick labels with specified font properties
    if orientation == "horizontal":
        # Tick labels for horizontal orientation
        cb.ax.set_xticklabels(
            tick_labels,
            fontfamily=font_family,
            fontsize=font_size,
            color=font_color,
        )
        # Move tick positions
        cb.ax.xaxis.set_ticks_position(tick_position)
        cb.ax.xaxis.set_label_position(tick_position)
    else:
        # Tick labels for vertical orientation
        cb.ax.set_yticklabels(
            tick_labels,
            fontfamily=font_family,
            fontsize=font_size,
            color=font_color,
        )
        # Move tick positions
        cb.ax.yaxis.set_ticks_position(tick_position)
        cb.ax.yaxis.set_label_position(tick_position)

    # Remove the tick lines
    if orientation == "horizontal":
        cb.ax.xaxis.set_tick_params(size=0)
    else:
        cb.ax.yaxis.set_tick_params(size=0)

    # Save or not
    if save_file:
        plt.savefig(output_file, dpi=300, transparent=True)

        print("Colorbar saved as " + output_file)
    else:
        print("Colorbar not saved in: " + output_file)

    # Close the figure
    plt.close(fig)




def draw_bucket(n_voxels,
                proportion=1.0,
                matrix_in=None,
                rows=1,
                initial_row=1,
                initial_col=1,
                color=(1, 0, 1)):
    """
    Draws a horizontal 'bucket' representing n_voxels. The number of columns
    in the bucket is scaled by 'proportion' (pixels per voxel).

    Parameters
    ----------
    n_voxels : int
        The number of voxels to represent.
    proportion : float, optional
        The scaling factor: pixels per voxel. Default is 1.0.
    matrix_in : np.ndarray or None, optional
        If provided, must be an image array of shape (R, C, 3) into which we draw.
        If None, the function returns just the small 'bucket' array.
    rows : int, optional
        How tall (in pixels) the bucket will be. Default is 1.
    initial_row : int, optional
        The top row in matrix_in where the bucket should be drawn (1-based). Default is 1.
    initial_col : int, optional
        The left column in matrix_in where the bucket should be drawn (1-based). Default is 1.
    color : tuple of float, optional
        RGB color values for the bucket. Default is white (1,1,1).

    Returns
    -------
    matrix_out : np.ndarray
        The updated image array (if matrix_in was given),
        or a newly created 2D array (if matrix_in was None).
    """
    
    if n_voxels == 0:
        # Nothing to draw for zero voxels
        return matrix_in
    
    init_col_0 = int(np.floor(initial_col * proportion - proportion + 1))
    cols = int(np.floor(n_voxels * proportion))

    if cols == 0:
        # Skip if the number of columns to draw is zero
        return matrix_in

    # Clip to matrix_in bounds if provided
    if matrix_in is not None:
        max_cols = matrix_in.shape[1]
        init_col_0 = max(0, min(init_col_0, max_cols))
        cols = min(cols, max_cols - init_col_0)

    # Create the voxel image (rows x cols) filled with 1's
    voxel_img = np.ones((rows, cols), dtype=float)

    # print color used
    # print('The color used is: ' + str(color))
    # print('The initial row is: ' + str(initial_row))
    if matrix_in is None:
        # If no matrix_in is provided, return just the voxel_img
        return voxel_img
    else:
        # Convert initial_row to 0-based indexing for Python
        init_row_0 = initial_row - 1

        # Broadcast the voxel_img into an RGB shape
        colored_img = voxel_img[..., None] * color  # shape: (rows, cols, 3)

        # Ensure the row range doesn't exceed matrix_in bounds
        max_rows = matrix_in.shape[0]
        rows = min(rows, max_rows - init_row_0)

        # Place the colored_img into matrix_in
        matrix_in[init_row_0:init_row_0+rows,
                  init_col_0:init_col_0+cols, :] = colored_img

        return matrix_in



def plot_branch(e, matrix_in, options):
    """
    Recursively draw 'buckets' for each branch (and its inside/outside branches)
    onto a color image array.

    Parameters
    ----------
    e : dict
        A node in the hierarchy. Expects at least:
            e['level']: int, the "row level" or depth
            e['n_voxels']: int, how many voxels are in e['map']
            e['location']: int, the "column offset" for this node
            Optional subfields: e['inside'], e['outside'], each also a dict with same structure
    matrix_in : np.ndarray
        The image array into which we draw, shaped (H, W, 3).
    options : dict
        Plotting options with keys:
            - 'proportion': float (pixels per voxel)
            - 'rows': int (how tall each bucket is)
            - 'initial_row': list or array of row starts for each level
            - 'colors': list of color triples, e.g. [(1,0,0), (0,1,0), ...]

    Returns
    -------
    matrix_out : np.ndarray
        The updated matrix_in with new buckets drawn.
    """

    level = e['level']
    n_voxels = e['n_voxels']
    proportion = options['proportion']
    rows = options['rows']
    # Initial row and column for this level
    initial_row = options['initial_row'][level - 1]
    initial_col = e['location']

    # Color to use at this level
    color = options['colors'][level - 1]

    # Draw the current node as a horizontal 'bucket'.
    matrix_in = draw_bucket(n_voxels,
                            proportion=proportion,
                            matrix_in=matrix_in,
                            rows=rows,
                            initial_row=initial_row,
                            initial_col=initial_col,
                            color=color)

    # Recursively plot inside branch if it exists
    if 'inside' in e:
        matrix_in = plot_branch(e['inside'], matrix_in, options)

    # Recursively plot outside branch if it exists
    if 'outside' in e:
        matrix_in = plot_branch(e['outside'], matrix_in, options)

    return matrix_in


def plot_arc(initial_location, extension, level, color, level_distance, arch_width=0.8, same_level=False):
    """
    Plots a thick arc on a radial layout.

    Parameters
    ----------
    initial_location : float
        Starting angle (in degrees) of the arc.
    extension : float
        Angular extent (in degrees) of the arc.
    level : int
        Which level (ring) this arc should go on.
    color : tuple or str
        The color of the arc (e.g., (1,0,0) for red).
    level_distance : float
        The radial distance between levels.
    arch_width : float, optional
        The thickness of the arc, by default 0.8
    """
    # if same_level is True, level 2 and above will be at the same level
    # if same_level:
    #     if level > 0:
    #         level = 1
    # Calculate the outer radius based on the level
    outer_radius = (level + 1) * level_distance
    
    # Calculate the inner radius based on the arch_width
    inner_radius = outer_radius - arch_width
    
    # Create a Wedge patch to represent the thick arc
    arc_patch = Wedge(
        center=(0, 0),                   # Center of the circle
        r=outer_radius,                  # Outer radius
        theta1=initial_location,         # Starting angle
        theta2=initial_location + extension,  # Ending angle
        width=arch_width,                # Thickness of the arc
        facecolor=color,                 # Fill color of the arc
        edgecolor='none'                 # No edge color for smoothness
    )
    
    # Add the arc to the current axes
    ax = plt.gca()
    ax.add_patch(arc_patch)
    
    # (Optional) Draw a small inner circle to represent the boundary if it's the first level
    # if level == 1:
    #     center_circle = plt.Circle(
    #         (0, 0), 
    #         inner_radius, 
    #         edgecolor='black', 
    #         fill=False, 
    #         lw=1
    #     )
    #     ax.add_patch(center_circle)
        # print('Inner radius: ' + str(inner_radius))
    
    # Ensure the aspect ratio is equal to maintain circular arcs
    plt.axis('equal')
    # remove the axis
    plt.axis('off')


import matplotlib.patches as mpatches

def plot_profile(e, options):
    """
    This function generates a radial plot that represents the number of voxels for each category on each level of e.
    It also adds a legend based on options['legend_list'] and options['colors'] once at the top level (level == 1).
    """
    total_voxels = options['total_voxels']
    level = e['level']
    n_voxels = e['n_voxels']

    # Calculate the initial location in degrees
    initial_location = e['location'] / total_voxels * 360

    level_distance = options['level_distance']
    arch_width = options['arch_width']

    # Calculate the angular extension of this arc (in degrees)
    extension = n_voxels / total_voxels * 360

    # Pick the color for this region
    color = options['color_dict'][e['name']]
    # print('The color is: ' + str(color))
    # Plot the arc
    plot_arc(initial_location, extension, level, color, level_distance, arch_width)

    # If there are children, plot them
    if 'inside' in e:
        plot_profile(e['inside'], options)
    if 'outside' in e:
        plot_profile(e['outside'], options)

    # If we're back to the top level (level 1), add the legend.
    # (We do this only once so the legend is not duplicated on every recursive call.)
    if level == 1:
        # Build legend handles: one handle per entry in legend_list/colors
        handles = []
        legend_list = options.get('legend_list', [])
        # colors = options.get('colors', [])

        for idx, label in enumerate(legend_list):
            # Create a rectangle patch (color swatch) for the legend
            patch = mpatches.Patch(color=options['color_dict'][label], label=label)
            handles.append(patch)

        plt.legend(handles=handles, loc='best')

        # Ensure the aspect ratio is equal and no axis lines are shown
        plt.axis('equal')
        plt.axis('off')
         # if there is a title in the options, add it
    if 'title' in options:
        plt.title(options['title'])
        

        
def add_branch(e, new_map, new_name, input_level=[]):
    """
    The function checks which voxels of 'new_map' fall inside 'e["map"]'
    or outside 'e["map"]'. It then creates or updates 'inside' and/or 'outside'
    branches of e accordingly.
    
    Parameters
    ----------
    e : dict
        Dictionary representing a node in the hierarchy. Expects:
            - e['map']: 3D numpy array (binary) 
            - e['n_voxels']: number of voxels in e['map']
            - e['location']: integer, used to keep track of where to plot
            - e['name']: string name of this node
            - e['child']: bool indicating if this node has no sub-branches
            - Optionally 'inside' and/or 'outside' sub-dictionaries if e['child'] is False.
    new_map : np.ndarray
        3D binary numpy array to be added as a branch of e
    new_name : str
        Name for the new branch
    keep_level : bool, optional
        If True, the new branch will have the same level as e. Default is False.
    
    Returns
    -------
    e : dict
        The updated dictionary with possibly new 'inside' or 'outside' branches.
    """
    
    # level = e['level'] + 1
    # if level is empty
    if not input_level:
        level = e['level'] + 1
    else:
        level = input_level
    
    # Calculate which voxels go inside (overlap) and outside (non-overlap)
    # Overlap is where both e['map'] and new_map are 1
    map_in = (e['map'] & new_map).astype(int)
    
    # Outside is all of new_map that isn't inside
    map_out = new_map.copy()
    map_out[map_in == 1] = 0
    
    # Count voxels for inside and outside
    in_sum = np.sum(map_in)
    out_sum = np.sum(map_out)
    
    # If 'child' is True, this node currently has no inside/outside branches.
    if e['child']:
        if in_sum > 0:
            # Create the 'inside' branch
            e['inside'] = {
                'name': new_name,
                'map': map_in,
                'n_voxels': in_sum,
                'location': e['location'], # location is the same as the parent
                'child': True,
                'level': level,
            }
            # Since we're adding a new branch, 'child' becomes False
            e['child'] = False
        
        if out_sum > 0:
            # Create the 'outside' branch
            e['outside'] = {
                'name': new_name,
                'map': map_out,
                'n_voxels': out_sum,
                'location': e['location'] + e['n_voxels'], # location is the current location plus the number of voxels
                'child': True,
                'level': level,
            }
            # Since we're adding a new branch, 'child' becomes False
            e['child'] = False
    
    else:
        # e already has inside/outside branches
        if in_sum > 0:
            if 'inside' not in e:
                # If there's no 'inside' yet, create it
                e['inside'] = {
                    'name': new_name,
                    'map': map_in,
                    'n_voxels': in_sum,
                    'location': e['location'],
                    'child': True,
                    'level': level,
                }
            else:
                # Otherwise, recurse
                e['inside'] = add_branch(e['inside'], map_in, new_name, input_level=input_level)
        
        if out_sum > 0:
            if 'outside' not in e:
                # If there's no 'outside' yet, create it
                e['outside'] = {
                    'name': new_name,
                    'map': map_out,
                    'n_voxels': out_sum,
                    'location': e['location'] + e['n_voxels'],
                    'child': True,
                    'level': level,
                }
            else:
                # Otherwise, recurse
                e['outside'] = add_branch(e['outside'], map_out, new_name, input_level=input_level)

    return e

def generate_colors(n):
    """
    Generate `n` distinct colors using a matplotlib colormap.
    """
    cmap = plt.cm.hsv  # or 'rainbow', 'viridis', 'tab20', etc.
    colors = [cmap(i / float(n)) for i in range(n)]
    return colors

import numpy as np
import copy

def plot_single_slice_glass(
    background_map,
    data_map,
    slice_type,
    slice_num,
    axes,
    cmap_background,
    cmap_overlay,
    alpha,
    clear_background,
    nan_zero=True,
    mip_half_width=None,  # None = full-depth MIP (classic glass); 0 = single slice; k = +/-k slab
):
    """
    Glass-style projection for a single view (x/y/z) using MIPs and a thin outline from the background.
    Handles different resolutions (e.g., hi-res background vs low-res data) by resizing the 2D overlay MIP
    to match the background MIP.

    Parameters mirror plot_single_slice2; 'slice_num' is only used if mip_half_width is not None.
    Returns: (background_2d, overlay_2d) arrays in the plotted orientation (before .T)
    """

    # ---------- helpers (no external deps) ----------
    def _resize2d_bilinear(arr, new_shape):
        """Resize a 2D float array to new_shape using separable 1D np.interp (bilinear)."""
        a = np.asarray(arr, dtype=float)
        h, w = a.shape
        new_h, new_w = new_shape
        if h == 0 or w == 0 or new_h == 0 or new_w == 0:
            return np.zeros((new_h, new_w), dtype=float)

        # Coordinates in source space corresponding to target pixel centers
        y_src = np.linspace(0, h - 1, new_h)
        x_src = np.linspace(0, w - 1, new_w)

        # Interp rows, then columns
        # Step 1: along x for each row
        row_interp = np.empty((h, new_w), dtype=float)
        x = np.arange(w)
        for r in range(h):
            row_interp[r, :] = np.interp(x_src, x, a[r, :], left=a[r, 0], right=a[r, -1])

        # Step 2: along y for each column
        final = np.empty((new_h, new_w), dtype=float)
        y = np.arange(h)
        for c in range(new_w):
            final[:, c] = np.interp(y_src, y, row_interp[:, c], left=row_interp[0, c], right=row_interp[-1, c])

        return final

    def _binary_outline(mask2d):
        """One-pixel outline from a 2D binary mask via simple erosion (no scipy)."""
        m = mask2d.astype(bool)
        if m.size == 0:
            return np.zeros_like(mask2d, dtype=float)
        er = np.zeros_like(m)
        # crude 4-neighbor erosion (ignore border)
        if m.shape[0] >= 3 and m.shape[1] >= 3:
            core = m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:]
            er[1:-1, 1:-1] = core
        outline = (m & ~er).astype(float)
        return outline

    def _mip(volume, axis, slab=None):
        """Max-intensity projection; slab=(i0,i1) inclusive indices on 'axis' or None for full depth."""
        v = np.asarray(volume)
        if slab is None:
            return np.nanmax(v, axis=axis)
        i0, i1 = slab
        i0 = max(0, i0); i1 = min(v.shape[axis]-1, i1)
        if i1 < i0:
            i0, i1 = i1, i0
        slicer = [slice(None)] * v.ndim
        slicer[axis] = slice(i0, i1+1)
        slab_v = v[tuple(slicer)]
        return np.nanmax(slab_v, axis=axis)

    # ---------- prep volumes ----------
    # Work on copies; match your original flipping convention on data_map
    data_vol = copy.deepcopy(data_map)
    back_vol = background_map  # no need to copy; we never mutate

    # Convert zeros to NaN for overlay if requested (so they don't contribute to MIP)
    if nan_zero:
        data_vol = np.array(data_vol, dtype=float)
        data_vol[data_vol == 0] = np.nan

    # Match your original: flip overlay in x (axis=0)
    data_vol = np.flip(np.asarray(data_vol), axis=0)

    # Background mask for outline (nonzero = brain)
    back_mask = np.asarray(back_vol) != 0

    # ---------- choose axis + slab based on slice_type / slice_num ----------
    if slice_type == 'x':
        axis = 0  # sagittal → yz
        bg_plane_shape = (back_vol.shape[1], back_vol.shape[2])
        # slab bounds on the chosen axis
        slab = None if mip_half_width is None else (slice_num - mip_half_width, slice_num + mip_half_width)
    elif slice_type == 'y':
        axis = 1  # coronal → xz
        bg_plane_shape = (back_vol.shape[0], back_vol.shape[2])
        slab = None if mip_half_width is None else (slice_num - mip_half_width, slice_num + mip_half_width)
    elif slice_type == 'z':
        axis = 2  # axial → xy
        bg_plane_shape = (back_vol.shape[0], back_vol.shape[1])
        slab = None if mip_half_width is None else (slice_num - mip_half_width, slice_num + mip_half_width)
    else:
        raise ValueError("slice_type must be one of {'x','y','z'}")

    # ---------- compute background projections ----------
    # Binary MIP for outline
    bg_mask_mip = _mip(back_mask.astype(float), axis=axis, slab=slab) > 0
    bg_outline = _binary_outline(bg_mask_mip.astype(bool))

    # Optional grayscale background MIP (nice when clear_background=False)
    bg_gray_mip = _mip(np.asarray(back_vol, dtype=float), axis=axis, slab=slab)

    # ---------- compute overlay projection ----------
    overlay_mip = _mip(np.asarray(data_vol, dtype=float), axis=axis, slab=slab)

    # ---------- resize overlay MIP to match background MIP if shapes differ ----------
    target_h, target_w = bg_plane_shape
    ov_h, ov_w = overlay_mip.shape
    if (ov_h, ov_w) != (target_h, target_w):
        overlay_mip_resized = _resize2d_bilinear(overlay_mip, (target_h, target_w))
    else:
        overlay_mip_resized = overlay_mip

    # ---------- plot ----------
    axes.clear()

    # 1) optional grayscale background
    if not clear_background:
        # normalize background for display (robust scaling)
        bg = np.array(bg_gray_mip, dtype=float)
        finite = np.isfinite(bg)
        if np.any(finite):
            p1, p99 = np.nanpercentile(bg[finite], [1, 99])
            if p99 > p1:
                bg = np.clip((bg - p1) / (p99 - p1), 0, 1)
            else:
                bg = np.zeros_like(bg)
        else:
            bg = np.zeros_like(bg)
        axes.imshow(bg.T, cmap=cmap_background, origin='lower', alpha=alpha)

    # 2) draw a thin outline ("glass")
    # use a slightly higher alpha so it’s visible even if clear_background=True
    if np.any(bg_outline):
        axes.imshow(bg_outline.T, cmap='gray', origin='lower', alpha=min(1.0, alpha*0.8))

    # 3) overlay data (respect NaNs to keep holes transparent)
    # We'll not normalize here—leave that to your cmap limits if you set them externally.
    axes.imshow(overlay_mip_resized.T, cmap=cmap_overlay, origin='lower')

    axes.axis('off')

    # Return the raw 2D arrays (before .T), in case you want to reuse them
    return bg_gray_mip, overlay_mip_resized

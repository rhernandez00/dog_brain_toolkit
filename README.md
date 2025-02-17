# Dog brain toolkit

Repository to store dog fMRI preprocessing and analysis app
![Odin](Odin.jpg)

## Overview

The dog_brain_toolkit is a comprehensive toolkit for preprocessing and analyzing dog fMRI data. The toolkit includes various scripts and tools for data preprocessing, quality check, and normalization using different atlases.

## Project Structure

The project is organized into the following directories:

- `Atlas/`: Contains files related to the atlases used for normalization and other processes.
  - `Dog/`: Contains atlases specific to dogs.
    - `Czeibert/`: Contains files related to the Czeibert atlas.
    - `Johnson/`: Contains files related to the Johnson atlas.
    - `Nitzsche/`: Contains files related to the Nitzsche atlas.
      - `README.md`: Provides detailed information about the binary masks and the original readme from Nitzsche.
    - `transform_mask.sh`: Script for transforming masks.
  - `README.md`: Gives an overview of the atlases used in the project.

- `FSL_designs/`: Contains FSL design files for preprocessing.
  - `preprocess.fsf`: FSL design file for preprocessing.
  - `preprocess_modified.fsf`: Modified FSL design file for preprocessing.
  - `preprocess_no-slice-timing.fsf`: FSL design file for preprocessing without slice timing correction.
  - `preprocess_regular-up.fsf`: FSL design file for regular preprocessing.

- `Quality_check/`: Contains scripts and tools for quality checking fMRI data.
  - `functions/`: Contains various MATLAB functions for quality checking.
  - `NIfTI_toolbox/`: Contains the NIfTI toolbox for handling NIfTI files.
  - `README.md`: Provides an overview of the quality check tools.

- `utils.py`: Contains utility functions used across the project.

- `crop_and_bet.ipynb`: Jupyter notebook for cropping and brain extraction.

- `interface.ipynb`: Jupyter notebook for the user interface.

- `preprocess_functions.py`: Contains functions for preprocessing fMRI data using FSL.

- `preprocess.ipynb`: Jupyter notebook for preprocessing fMRI data.

## Usage

### Preprocessing

To preprocess the fMRI data, use the `preprocess_functions.py` script. The script includes functions for reorienting files, calculating motion, and generating mean functional images.

### Quality Check

The `Quality_check/` directory contains scripts and tools for quality checking fMRI data. The `run_quality_check.m` script calculates framewise displacement and DVARS using the `bramila_fwd` and `bramila_dvars` functions.

### Normalization

The `Atlas/` directory contains files related to the atlases used for normalization. The `resample_atlas.sh` script can be used to resample the atlases to a different resolution.

## Contribution Guidelines

Contributions to the dog_brain_toolkit are welcome. Please follow these guidelines when contributing:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes.
4. Submit a pull request with a clear description of your changes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

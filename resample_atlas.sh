#!/bin/bash
# This script will resample all the masks from a given atlas and reduce its resolution
# Usage: ./resample_atlas.sh <atlas_folder> <output_resolution>
# Example: ./resample_atlas.sh /path/to/atlas 2
# Author: Raul Hernandez

atlas_folder=$1
output_resolution=$2

# Optional: create an output folder in the same directory as atlas_folder
output_folder="${atlas_folder}${output_resolution}mm"
mkdir -p "${output_folder}"

# get a list of all .nii.gz files in the folder
files=$(ls "${atlas_folder}"/*.nii.gz)

# for every file in the folder
for file in $files
do
    # get the filename (basename) without the .nii.gz extension
    filename=$(basename "$file" .nii.gz)

    echo "processing: $filename"

    flirt -in "$file" -ref "$file" -out "${output_folder}/${filename}.nii.gz" -applyisoxfm "${output_resolution}" -interp nearestneighbour
done

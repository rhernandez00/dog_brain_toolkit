#!/bin/bash
# This script will remove slices from a 3D image using fslroi.
# Usage: ./remSlices.sh <input_file> <output_file> <params_file>
# The script will obtain the cutting parameters from params_file
# Example: ./remSlices.sh 3Dimage.nii.gz 3Dimage_cut.nii.gz 3Dimage.txt
# Author: Raul Hernandez

input_file=$1
params_file=$2

# read params_file and load variables in the file, will read:
# x_lim1, x_lim2, y_lim1, x_lim2, z_lim1, z_lim2
source ${params_file}

# print parameters
echo "Initial x: $x_lim1"
echo "Final x: $x_lim2"
echo "Initial y: $y_lim1"
echo "Final y: $y_lim2"
echo "Initial z: $z_lim1"
echo "Final z: $z_lim2"
echo "output file: $output_file"

# determine current folder
currentFolder=$(pwd)

# determine working folder based on output file
workingFolder=$(dirname ${output_file})
# go to working folder
cd ${workingFolder}

echo ${workingFolder}

echo cutting down x
fslsplit ${input_file} slice -x
names=""
for fileNum in $(seq -f "%04g" ${x_lim1} ${x_lim2})
do
	file=slice${fileNum}
	names+=${file}" "
done

fslmerge -x mergedFile.nii.gz ${names}

filesToRemove=($(ls slice*))
for fileR in "${filesToRemove[@]}"
do
	rm ${fileR}
done

echo cutting down y
fslsplit mergedFile.nii.gz slice -y
names=""
for fileNum in $(seq -f "%04g" ${y_lim1} ${y_lim2})
do
	file=slice${fileNum}
	names+=${file}" "
done

fslmerge -y mergedFile.nii.gz ${names}

filesToRemove=($(ls slice*))
for fileR in "${filesToRemove[@]}"
do
	rm ${fileR}
done

echo cutting down z

fslsplit mergedFile.nii.gz slice -z

names=""
for fileNum in $(seq -f "%04g" ${z_lim1} ${z_lim2})
do
	file=slice${fileNum}
	names+=${file}" "
done

fslmerge -z ${output_file} ${names}

filesToRemove=($(ls slice*))
for fileR in "${filesToRemove[@]}"
do
#	echo ${fileR}
	rm ${fileR}
done

# remove merged file
rm mergedFile.nii.gz
# go back to original folder
cd ${currentFolder}

echo "Done! file saved as ${output_file}" 

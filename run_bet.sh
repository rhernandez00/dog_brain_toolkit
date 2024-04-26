#!/bin/bash
# This script runs 1 to 3 BET based on input coordinates for the spheres
# Usage: ./run_bet.sh <input_file> <output_file> <params_file>

# Author: Raul Hernandez

input_file=$1
params_file=$2

# read params_file and load variables in the file, will read:
# betx_1, bety_1, betz_1, thr_1, betx_2, bety_2, betz_2, thr_2, 
# betx_3, bety_3, betz_3, thr_3, 
# output_file, mask_file1, mask_file2, mask_file3
source ${params_file}

# get working folder
workingFolder=$(dirname ${output_file})


# run first bet
bet ${input_file} ${mask_file1} -f ${thr_1} -g 0 -c ${betx_1} ${bety_1} ${betz_1} -m
# get filename mask created without extension
filename=$(basename ${mask_file1} .nii.gz)
# erase mask_file1
rm ${mask_file1}
# copy mask_file1 to output_file
cp ${filename}_mask.nii.gz ${output_file}
# copy mask file created to mask_file1
cp ${filename}_mask.nii.gz ${mask_file1}
# remove mask file created
rm ${filename}_mask.nii.gz

# if thr2 > 0, create mask2 and add to output_file
if [ $(echo "$thr_2 > 0" | bc) -eq 1 ]; then
    # run second bet
    bet ${input_file} ${mask_file2} -f ${thr_2} -g 0 -c ${betx_2} ${bety_2} ${betz_2} -m
    # get filename mask created without extension
    filename=$(basename ${mask_file2} .nii.gz)
    # erase mask_file2
    rm ${mask_file2}
    # copy mask file created to mask_file2
    cp ${filename}_mask.nii.gz ${mask_file2}
    # add mask_file2 to output_file
    fslmaths ${output_file} -add ${mask_file2} ${output_file}
    # remove mask file created
    rm ${filename}_mask.nii.gz

fi

# if thr3 > 0, create mask3 and add to output_file
if [ $(echo "$thr_3 > 0" | bc) -eq 1 ]; then
    # run third bet
    bet ${input_file} ${mask_file3} -f ${thr_3} -g 0 -c ${betx_3} ${bety_3} ${betz_3} -m
    # get filename mask created without extension
    filename=$(basename ${mask_file3} .nii.gz)
    # erase mask_file3
    rm ${mask_file3}
    # copy mask file created to mask_file3
    cp ${filename}_mask.nii.gz ${mask_file3}
    # add mask_file3 to output_file
    fslmaths ${output_file} -add ${mask_file3} ${output_file}
    # remove mask file created
    rm ${filename}_mask.nii.gz
fi

# binarize output_file
fslmaths ${output_file} -bin ${output_file}
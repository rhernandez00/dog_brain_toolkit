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

bet ${input_file} ${mask_file1} -f ${thr_1} -g 0 -c ${betx_1} ${bety_1} ${betz_1} -m
cp ${mask_file1} ${output_file}
rm ${mask_file1}_mask.nii.gz

# if thr2 > 0, create mask2 and add to output_file
if [ $(echo "$thr_2 > 0" | bc) -eq 1 ]; then
    bet ${input_file} ${mask_file2} -f ${thr_2} -g 0 -c ${betx_2} ${bety_2} ${betz_2} -m
    rm ${mask_file2}_mask.nii.gz
    fslmaths ${output_file} -add ${mask_file2} ${output_file}
fi

# if thr3 > 0, create mask3 and add to output_file
if [ $(echo "$thr_3 > 0" | bc) -eq 1 ]; then
    bet ${input_file} ${mask_file3} -f ${thr_3} -g 0 -c ${betx_3} ${bety_3} ${betz_3} -m
    rm ${mask_file3}_mask.nii.gz
    fslmaths ${output_file} -add ${mask_file3} ${output_file}
fi

# binarize output_file
#fslmaths ${output_file} -bin ${output_file}
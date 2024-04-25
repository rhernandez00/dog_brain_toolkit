#!/bin/bash
# This script runs 1 to 3 BET based on input coordinates for the spheres
# Usage: ./run_bet.sh <inputfile> <outputfile> <paramsfile>

# Author: Raul Hernandez

inputfile=$1
outputfile=$2
paramsfile=$3

# read paramsfile and load variables in the file, will read:
# betx_1, bety_1, betz_1, thr_1, betx_2, bety_2, betz_2, thr_2, betx_3, bety_3, betz_3, thr_3
source ${paramsfile}

# get working folder
workingFolder=$(dirname ${outputfile})

bet ${inputfile} ${workingFolder}/mask1.nii.gz -f ${thr_1} -g 0 -c ${betx_1} ${bety_1} ${betz_1} -m
cp ${workingFolder}/mask1.nii.gz ${outputfile}

# if thr2 > 0, create mask2 and add to outputfile
if [ $(echo "$thr_2 > 0" | bc) -eq 1 ]; then
    bet ${inputfile} ${workingFolder}/mask2.nii.gz -f ${thr_2} -g 0 -c ${betx_2} ${bety_2} ${betz_2} -m
    fslmaths ${outputfile} -add ${workingFolder}/mask2.nii.gz ${outputfile}
fi

# if thr3 > 0, create mask3 and add to outputfile
if [ $(echo "$thr_3 > 0" | bc) -eq 1 ]; then
    bet ${inputfile} ${workingFolder}/mask3.nii.gz -f ${thr_3} -g 0 -c ${betx_3} ${bety_3} ${betz_3} -m
    fslmaths ${outputfile} -add ${workingFolder}/mask3.nii.gz ${outputfile}
fi

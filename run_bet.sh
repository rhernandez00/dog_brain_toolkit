#!/bin/bash
# This script runs 1 to 3 BET based on input coordinates for the spheres
# files go without extension
# Usage: ./run_bet.sh <inputfile> <outputfile> <thr1> <x1> <y1> <z1> <thr2> <x2> <y2> <z2> <thr3> <x3> <y3> <z3> 

# Author: Raul Hernandez

inputfile=$1
outputfile=$2

thr1=$3
x1=$4
y1=$5
z1=$6
thr2=$7
x2=$8
y2=$9
z2=$10
thr3=$11
x3=$12
y3=$13
z3=$14

# get working folder
workingFolder=$(dirname ${outputfile})

bet ${inputfile} ${workingFolder}\mask1.nii.gz -f ${thr1} -g 0 -c ${x1} ${y1} ${z1} -m
cp ${workingFolder}\mask1.nii.gz ${outputfile}

# if thr2 > 0, create mask2 and add to outputfile
if [ $thr2 -gt 0 ]; then
    bet ${inputfile} ${workingFolder}/mask2.nii.gz -f ${thr2} -g 0 -c ${x2} ${y2} ${z2} -m
    fslmaths ${outputfile} -add ${workingFolder}/mask2.nii.gz ${outputfile}
fi

# if thr3 > 0, create mask3 and add to outputfile
if [ $thr3 -gt 0 ]; then
    bet ${inputfile} ${workingFolder}/mask3.nii.gz -f ${thr3} -g 0 -c ${x3} ${y3} ${z3} -m
    fslmaths ${outputfile} -add ${workingFolder}/mask3.nii.gz ${outputfile}
fi

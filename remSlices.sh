#!/bin/bash
# Author: Raul Hernandez
# This script will remove slices from a 3D image using fslroi.
# Usage: ./remSlices.sh <inputfile> <outputfile> <initialx> <finalx> <initialy> <finaly> <initialz> <finalz>
# Assumes .nii.gz files
# Example: ./remSlices.sh 3Dimage 0 100 0 100 0 100

inputfile=$1
outputfile=$2

initialx=$3
finalx=$4
initialy=$5
finaly=$6
initialz=$7
finalz=$8

echo cutting down x
fslsplit ${inputfile} slice -x
names=""
for fileNum in $(seq -f "%04g" ${initialx} ${finalx})
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
for fileNum in $(seq -f "%04g" ${initialy} ${finaly})
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
for fileNum in $(seq -f "%04g" ${initialz} ${finalz})
do
	file=slice${fileNum}
	names+=${file}" "
done

fslmerge -z cut_${inputfile}.nii.gz ${names}

filesToRemove=($(ls slice*))
for fileR in "${filesToRemove[@]}"
do
#	echo ${fileR}
	rm ${fileR}
done

# move the final file to the output file
cp cut_${inputfile}.nii.gz ${outputfile}.nii.gz
rm cut_${inputfile}.nii.gz

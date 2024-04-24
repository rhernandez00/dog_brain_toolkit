#!/bin/bash
# This script will remove slices from a 3D image using fslroi.
# Usage: ./remSlices.sh <inputfile> <outputfile> <paramsfile>
# The script will obtain the cutting parameters from paramsfile
# Example: ./remSlices.sh 3Dimage.nii.gz 3Dimage_cut.nii.gz 3Dimage.txt
# Author: Raul Hernandez

inputfile=$1
outputfile=$2
paramsfile=$3

# read paramsfile and load variables in the file, will read:
# initialx, finalx, initialy, finaly, initialz, finalz
source ${paramsfile}

echo "Initial x: $initialx"
echo "Final x: $finalx"
echo "Initial y: $initialy"
echo "Final y: $finaly"
echo "Initial z: $initialz"
echo "Final z: $finalz"

# determine current folder
currentFolder=$(pwd)

# determine working folder based on output file
workingFolder=$(dirname ${outputfile})
# go to working folder
cd ${workingFolder}

echo ${workingFolder}

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

fslmerge -z ${outputfile} ${names}

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

echo "Done! file saved as ${outputfile}" 

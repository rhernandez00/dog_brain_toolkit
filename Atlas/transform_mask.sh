# ! /bin/bash

# inputs
inputatlas=Czeibert
outputatlas=Nitzsche
mask=LAmygdala

echo input atlas: ${inputatlas}
echo output atlas: ${outputatlas}
echo input mask: ${mask}

inputfolder=/media/sf_github/dog_brain_toolkit/Atlas/Dog/${inputatlas}/${inputatlas}
outputfolder=/media/sf_github/dog_brain_toolkit/Atlas/Dog/${outputatlas}/${inputatlas}

# check if output folder exists
if [ ! -d $outputfolder ]; then
   echo Creating output folder: ${outputfolder}
   mkdir -p $outputfolder
fi

flirt -in ${inputfolder}/${mask}.nii.gz -applyxfm -init /media/sf_github/dog_brain_toolkit/Atlas/Dog/${outputatlas}/transformations/${inputatlas}2${outputatlas}.mat -out ${outputfolder}/${mask}.nii.gz -paddingsize 0.0 -interp trilinear -ref /media/sf_github/dog_brain_toolkit/Atlas/Dog/${outputatlas}/brain.nii.gz



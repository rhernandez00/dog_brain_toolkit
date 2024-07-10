# Binary masks (Nitzsche)
Binary masks where generated using:
fslmaths [mask] -thr 0.05 -bin [mask]_bin

b_fullBrain:
fslmaths b_GreyMatter_bin.nii.gz -add b_WhiteMatter.nii.gz b_fullBrain.nii.gz
fslmaths b_fullBrain.nii.gz -bin b_fullBrain.nii.gz

# Original readme.txt from Nitzsche. 
## Note, this repository doesn't follow the file names as described below

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   			
%	canine brain atlas, templates and TPM 		
%	by Bjoern Nitzsche, PhD, DVM		
%	Version: 1.0		
%	date: 2018/01/31		
%			
%	Files are zipped:		
%	canine_template.zip
%	---> canine_template.nii
%	canine_TPM.zip
%	---> canine_TPM.nii
%	labels.zip
%	---> canine_labels_unilateral.nii
%	---> labels_binaries.xls
%	mask.zip
%	---> gm_mask.nii
%	---> wm_mask.nii
%	--> ventricle.nii
%	surface.zip
%	---> brain.surf.gii
%	---> gm.surf.nii
%			
%	Data format:		
%	NIFTI-1, float32 (mask:uint8)
%
%	TPM are in 4-d and ordered as follows
%		TPM.nii,1 - gm
%		TPM.nii,2 - wm
%		TPM.nii,3 - ventricles
%		TPM.nii,4 - subdural csf
%		TPM.nii,5 - outer brain tissue
%
%	-------------------------------------------------------------------------------------------------------
%
%	CITATION		
%	The work based on the ff. publication. Please don't forget to cite when using the template, tpm and/or atlas_labels:		
%			
%	Nitzsche, Bjoern, Boltze, Johannes, Ludewig, Eberhard, Flegel, Thomas, Schmidt, Martin J., Seeger, Johannes, 
%	Barthel, Henryk, Brooks, Olivia W., Gounis, Matthew J., Stoffel, Michael H., and Schulze, Sabine. 
%	A  stereotaxic breed-averaged, symmetric T2w canine brain atlas including detailed morphological and 
%	volumetrical data sets. NeuroImage.2018. doi:10.1016/j.neuroimage.2018.01.066	
%			
%	CONTACT	
%	bjoern.nitzsche@medizin.uni-leipzig.de	
%			
%	-------------------------------------------------------------------------------------------------------
%
%	BRIEF DESCRIPTION 		
%	16 MR images (3T MRI) from different canine breed were aligned and		
%	semiautomated segmented into grey and white matter and cerebrospinal fluid (csf+ventricle).		
%	Finaly, non-linear transformation matrices were applied to each mask (isovoxel 0.5 mm) and tissue classes were merged.		
%			
%	-------------------------------------------------------------------------------------------------------		
%			
%	Before using SPM segmentation routine:		
%	1. Go to installation root of SPM		
%	2. Open the folder ../spm/tpm and rename the exsiting tpm  (human, TPM.nii) (into e.g. TPM.human.nii)		
%	3. Copie the canine tpm (canine_TPM) into the folder ../spm/tpm		
%	--> Please note: 'old' SPM8 segmentation routine requires 3 seperated .nii TPM (gm, wm, csf) while we here provide 5 TPM in a 4-D.nii
%	--> Therefore, SPM12 or the 'New Segmentation' routine is recommended 	
%			
%	Recommendation for 'New segmentation' using SPM12:		
%	1. Open the PET/MR Module in SPM		
%	2. Aligne your targets to the template using a) Display b) realign and c) coregistering		
%	"3. Then press ""Segment"""		
%	4. Change the following parameters into (depend on the quality of your raw data):		
%			--> Replace the last entry (6th entry) for 'Tissue probality map' 
%				(eg. [root]\Toolbox\spm12\tpm\TPM.nii,6) 
%				with 
%				[root]\Toolbox\spm12\tpm\TPM.nii,5 (= TPM for outer brain tissue)
%				to avoid interference with the routine
%			--> Gaussians per class: [2 2 4 4]
%			--> Affine Regularisation: [Averaged sized template] or [No Affine Registration]
%	5. Save and run job		
%	"6. Check results by """"Check Reg"""""
%					
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%			



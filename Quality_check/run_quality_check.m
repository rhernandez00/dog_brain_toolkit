function tableOut = run_quality_check(project_folder, sub_N, run_N, specie, task, session, thr_fwd, thr_dvars, radius, whichType)
addpath([pwd,filesep,'Quality_check', filesep, 'functions']);
addpath([pwd,filesep,'Quality_check', filesep,'NIfTI_toolbox']);
% radius FSL uses 55 as default


% This script calculates framewise displacement and DVARS using bramila_fwd and bramila_dvars. 
% These quality measurements come from the paper: Power et al. (2012) doi:10.1016/j.neuroimage.2011.10.018
% Author: Raul Hernandez June/2024

% This script is intented to be used by the tools in the toolkit. The
% original script can be found the repository fMRI_QA

% which type to use for generating the txt? 
% Note. I only tested this for 'or', others might not work
% whichType = 
% Takes:
% 'fwd': for generating using only volumes marked by fwd
% 'dvars': for generating using only volumnes marked by dvars
% 'or': will mark the volumes who excede the threshold in any of the two measurements
% 'and': will mark the volumes who excede the threshold in both measurements


movement_file = [specie, '-sub-',sprintf('%03d',sub_N)];
if ~strcmp(session, '')
    movement_file = [movement_file, '_ses-', session];
end
movement_file = [movement_file, '_task-', task, '_run-', sprintf('%02d',run_N), '.par'];
cfg.radius = radius; 
cfg.prepro_suite =  'fsl-fs';
cfg.motionparam = [project_folder,filesep,'movement',filesep,movement_file];

fwd = bramila_framewiseDisplacement(cfg);
clear cfg
indx = fwd > thr_fwd;
indxNums = find(indx);
totalVolumes = numel(indx);
lostVolumes = numel(indxNums);
indxfwd = indx;

nKey = 1;
tableOut.fwd(nKey) = mean(fwd);
tableOut.fwd_clean(nKey) = mean(fwd(fwd < thr_fwd));
tableOut.volumesLostAfter_fwd(nKey) = lostVolumes;
tableOut.totalVolumes(nKey) = totalVolumes;


% This is for dvars
vol_file = [specie, '-sub-',sprintf('%03d',sub_N)];
if ~strcmp(session, '')
    vol_file = [vol_file, '_ses-', session];
end
vol_file = [vol_file, '_task-', task, '_run-', sprintf('%02d',run_N), '_bold.nii.gz'];

nii_path = [project_folder,filesep,'BIDS',filesep, specie, '-sub-',sprintf('%03d',sub_N)];
if ~strcmp(session,'')
    nii_path = [nii_path, filesep, session];
end
nii_path = [nii_path,filesep,'func',filesep,vol_file];

vol = load_untouch_nii(nii_path);

cfg.vol = vol.img; %assigns the nii file
[~,img]=bramila_dvars(cfg); %runs dvars as % of change in the whole image
dvarsPer = abs(mean(img,2)); %calculates the absolute of the average % of change across the image
indx = dvarsPer > thr_dvars; %thresholds the %
indxNums = find(indx);
totalVolumes = numel(indx);
lostVolumes = length(indxNums);
indxDvars = indx;

if tableOut.totalVolumes(nKey) ~= totalVolumes
    error(['The volumes in the movement file and the nifti file',...
        'do not match for: ', vol_file, ' volumes in nifti (.nii.gz): ',...
        num2str(totalVolumes), ' volumes in movement file (.par): ',...
        num2str(tableOut.totalVolumes(nKey))]);
end

tableOut.dvars(nKey) = mean(dvarsPer);
tableOut.dvars_clean(nKey) = mean(dvarsPer(dvarsPer < thr_dvars));
tableOut.volumesLostAfter_dvars(nKey) = lostVolumes;

indx1 = indxfwd;
indx2 = indxDvars;

indx = sum([indx1,indx2],2)>0; %or
tableOut.lostVolumes_fw_or_dvars(nKey) = sum(indx);

indxAnd = sum([indx1,indx2],2)>1; %and
tableOut.lostVolumes_fw_and_dvars(nKey) = sum(indxAnd);

switch whichType
    case 'dvars'
        indxOut = logical(indx2);
    case 'fwd'
        indxOut = logical(indx1);
    case 'or'
        indxOut = logical(indx);
    case 'and'
        indxOut = logical(indxAnd);
end


txt_file = [specie, '-sub-',sprintf('%03d',sub_N)];
if ~strcmp(session, '')
    txt_file = [txt_file, '_ses-', session];
end
txt_file = [txt_file, '_task-', task, '_run-', sprintf('%02d',run_N)];
writeTxt([project_folder,filesep,txt_file,'.txt'],indxOut); %

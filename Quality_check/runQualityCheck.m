function run_quality_check(project_folder, thr_fwd, thr_dvars, radius)

% radius some option needed for fwd. Attila shared a script that used 28. FSL uses 55 as default


% This script calculates framewise displacement and DVARS using bramila_fwd and bramila_dvars. 
% These quality measurements come from the paper: Power et al. (2012) doi:10.1016/j.neuroimage.2011.10.018
% Author: Raul Hernandez January/2023

% This script is intented to be used by the tools in the toolkit. The
% original script can be found the repository fMRI_QA

% which type to use for generating the txt? 
whichType = 'or'; %Note. I only tested this for 'or', others might not work
%Takes:
% 'fwd': for generating using only volumes marked by fwd
% 'dvars': for generating using only volumnes marked by dvars
% 'or': will mark the volumes who excede the threshold in any of the two measurements
% 'and': will mark the volumes who excede the threshold in both measurements



tableName = [project_folder,'quality_check',filesep,'table.xlsx']; %path of the output table 
filesPath = [project_folder,'quality_check',filesep,'/input']; %folder of files to analyze
savePath = [project_folder,'quality_check',filesep,'/txt']; %output folder

if ~exist(savePath,'dir')
    mkdir(savePath);
end

cfg.radius = radius; 

fileList = dir([filesPath,'/*.*']);
fileList = {fileList.name};
fileList(1:2) = [];

primaryKeyList = cell(1,size(fileList,2));
for nFile = 1:size(fileList,2)
    fileName = fileList{nFile};
    dot = strfind(fileName,'.');
    dot = dot(end);
    extention = fileName(dot+1:end);
    switch extention
        case 'par' %This is a movement file. Running fwd
            primaryKey = fileName(1:end-4);
        case 'txt' %This is a movement file. Running fwd
            primaryKey = fileName(1:end-4);
        case 'nii' %This is a nii file. Running dvars
            primaryKey = fileName(1:end-4);
        case 'gz' %This is a nii file. Running dvars
            primaryKey = fileName(1:end-7);
        otherwise
            error(['Unrecognized extention: ', extention, ' for file: ',fileName]);
    end
    primaryKeyList{nFile} = primaryKey;
end

keyList = unique(primaryKeyList)';
totalKeys = length(keyList);
preproSuiteList = cell(numel(keyList),1);
preproSuiteList(:) ={'unkown'};
tableOut = table(keyList,preproSuiteList,zeros(size(keyList)),...
    zeros(size(keyList)),zeros(size(keyList)),zeros(size(keyList)),...
    'VariableNames',{'primaryKey','PreproSuite','fwd','fwd_clean',...
    'volumesLostAfter_fwd','totalVolumes'});

%This runs fwd for each primaryKey if movement files are found
indxfwd = cell(1,numel(keyList));
for nKey = 1:numel(keyList) 
    primaryKey = keyList{nKey};
    if exist([filesPath,'/',primaryKey,'.txt'],'file') && exist([filesPath,'/',primaryKey,'.par'],'file')
        error(['There are txt and par files for: ',primaryKey,' there should only be one']);
    elseif exist([filesPath,'/',primaryKey,'.txt'],'file')
        PreproSuite = 'spm';
        cfg.prepro_suite =  'spm';
        cfg.motionparam = [filesPath,'/',primaryKey,'.txt'];
    elseif exist([filesPath,'/',primaryKey,'.par'],'file')
        PreproSuite = 'fsl';
        cfg.prepro_suite =  'fsl-fs';
        cfg.motionparam = [filesPath,'/',primaryKey,'.par'];
    else %no movement files for that primaryKey
        disp(['no movement files for: ', primaryKey]);
        continue
    end
    
    fwd = bramila_framewiseDisplacement(cfg);
    



    indx = fwd > thr_fwd;
    indxNums = find(indx);
    totalVolumes = numel(indx);
    lostVolumes = numel(indxNums);
    tableOut.fwd(nKey) = mean(fwd);
    tableOut.fwd_clean(nKey) = mean(fwd(fwd < thr_fwd));
    tableOut.volumesLostAfter_fwd(nKey) = lostVolumes;
    tableOut.totalVolumes(nKey) = totalVolumes;
    

    tableOut.PreproSuite{nKey} = PreproSuite;
    indxfwd{nKey} = indx;
end
clear cfg

%This runs dvars
indxDvars = cell(1,numel(keyList));

for nKey = 1:numel(keyList)
    primaryKey = keyList{nKey};
    if exist([filesPath,'/',primaryKey,'.nii.gz'],'file') && exist([filesPath,'/',primaryKey,'.nii'],'file')
        error(['There are two nifti files for: ',primaryKey,' there should only be one']);
    elseif exist([filesPath,'/',primaryKey,'.nii'],'file')
        vol = load_untouch_nii([filesPath,'/',primaryKey,'.nii']);
    elseif exist([filesPath,'/',primaryKey,'.nii.gz'],'file')
        vol = load_untouch_nii([filesPath,'/',primaryKey,'.nii.gz']);
    else
        continue
    end
    cfg.vol = vol.img; %assigns the nii file
    
    [~,img]=bramila_dvars(cfg); %runs dvars as % of change in the whole image
    dvarsPer = abs(mean(img,2)); %calculates the absolute of the average % of change across the image
    indx = dvarsPer > thr_dvars; %thresholds the %
    indxNums = find(indx);
    totalVolumes = numel(indx);
    lostVolumes = length(indxNums);
    if tableOut.totalVolumes(nKey) == 0
        tableOut.totalVolumes(nKey) = totalVolumes;
    else
        if tableOut.totalVolumes(nKey) ~= totalVolumes
            error(['The volumes in the movement file and the nifti file',...
                'do not match for: ', primaryKey, ' volumes in nii: ',...
                num2str(totalVolumes), ' volumes in movement file: ',...
                num2str(tableOut.totalVolumes(nKey))]);
        end
    end
    tableOut.dvars(nKey) = mean(dvarsPer);
    tableOut.dvars_clean(nKey) = mean(dvarsPer(dvarsPer < thr_dvars));
    tableOut.lostVolumesAfterdvars(nKey) = lostVolumes;
    indxDvars{nKey} = indx;
end

for nKey = 1:numel(keyList)
    primaryKey = keyList{nKey};
    if isempty(indxDvars{nKey})
        indxDvars{nKey} = zeros(tableOut.totalVolumes(nKey),1);
    end
    if isempty(indxfwd{nKey})
        indxfwd{nKey} = zeros(tableOut.totalVolumes(nKey),1);
    end
    indx1 = indxfwd{nKey};
    indx2 = indxDvars{nKey};
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
    writeTxt([savePath,'/',primaryKey,'.txt'],indxOut); %
end
if exist(tableName,'file')
    disp(['Table found: ', tableName, ' deleting it...']);
    delete(tableName);
end
writetable(tableOut,tableName);
disp([tableName, ' written'])

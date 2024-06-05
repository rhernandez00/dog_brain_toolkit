function [lostVolumes,totalVolumes,matTxt,vals,fileName] = fuctionToTxt(project,specie,thr,participant,run,varargin)

filesPath = getArgumentValue('pathIn',['G:/My Drive','/Results/',project,'/movement'],varargin{:});
savePath = getArgumentValue('pathOut',['D:/Raul/results/',project,'/movement/',specie],varargin{:});
saveTxt = getArgumentValue('saveTxt',true,varargin{:});
program = getArgumentValue('program','FSL',varargin{:});
functionToUse = getArgumentValue('functionToUse','fwd',varargin{:});

switch functionToUse
    case 'fwd'
        switch program
            case 'FSL'
                cfg.prepro_suite =  'fsl-fs';
                cfg.motionparam =[filesPath,'/',project,specie,sprintf('%02d',participant),'run',sprintf('%02d',run),'.par'];
            case 'spm'
                cfg.prepro_suite =  'spm';
                cfg.motionparam =[filesPath,'/',project,specie,sprintf('%02d',participant),'run',sprintf('%02d',run),'.txt'];
            otherwise
                error([program, ' not available. Programs accepted are FSL and spm']);
        end

        cfg.radius = 28;

        fileName  = [filesPath,'/',project,specie,sprintf('%02d',participant),'run',sprintf('%02d',run),'.par'];
        disp(['Running fwd for: ',cfg.motionparam]);
        [fwd,~]=bramila_framewiseDisplacement(cfg);
        vals = fwd;
    case 'dvars'
        fileName = [filesPath,'/',project,specie,sprintf('%02d',participant),'run',sprintf('%02d',run),'.nii'];
        folderName = [filesPath,'/',project,specie,sprintf('%02d',participant),'run',sprintf('%02d',run)];
        
        if ~exist(fileName,'file') %checking if the file or folder are available. Then load
            disp(['File ,' fileName,' not found, checking for folder']);
            if ~exist(folderName,'dir')
                error(['Folder ,' folderName,' not found']);
            else
                disp(['Folder ,' folderName,' found, merging into a 4D file']);
                vol.img = load_niiFolder(folderName);
                
            end
        else
            disp(['File ,' fileName,' found, loading...']);
            vol = load_untouch_nii(fileName);
        end
        
        cfg.vol = vol.img;
        [dvars]=bramila_dvars(cfg);
        dvarsZ = [0;zscore(dvars(2:end))];
        vals = dvarsZ;
    otherwise
        error('Wrong function, accepted ade fwd and dvars');
        
        
end


indx = vals>thr;            
indxNums = find(indx);
totalVolumes = numel(indx);
lostVolumes = length(indxNums);
if ~isempty(indxNums)
    matTxt = zeros(length(vals),length(indxNums));
    for nCol = 1:length(indxNums)
        matTxt(indxNums(nCol),nCol) = 1;
    end
else
    matTxt = zeros(length(vals),1);
end
% if appendMov
%     matTxt
% end
fileOut = [savePath,'/sub',sprintf('%03d',participant),'_run',sprintf('%03d',run),'.txt'];
if saveTxt
    disp(['Writting file: ',fileOut]);
    %writematrix(matTxt,fileOut);
    writeTxt(fileOut,matTxt);
    %xlswrite(fileOut,logical(matTxt));
    %dlmwrite(fileOut,matTxt);
else
    disp('No txt created');
end


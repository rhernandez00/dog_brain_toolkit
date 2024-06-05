function vol = load_niiFolder(folderName)
%loads nii files from a folder into a single 4D file
saveNii = false;
%folderName = 'C:\Users\Hallgato\Dropbox\MVPA\QA\inputFiles\PhonD01run01';
fileList = {dir([folderName,'\*.nii']).name};

filesNumber = zeros(1,numel(fileList));
for nFile = 1:numel(filesNumber)
    filesNumber(nFile) = str2double(fileList{nFile}(end-8:end-4));
end
baseName = fileList{1}(1:end-9);

data = load_untouch_nii([folderName,'\',baseName,sprintf('%05d',1),'.nii']);

vol = zeros(size(data.img,1),size(data.img,2),size(data.img,3),max(filesNumber));

for nFile = 1:max(filesNumber)
    data = load_untouch_nii([folderName,'\',baseName,sprintf('%05d',nFile),'.nii']);
    vol(:,:,:,nFile) = data.img;
end

if saveNii
    data.img = vol;
    data.hdr.dime.dim(1) = 4;
    data.hdr.dime.dim(5) = max(filesNumber);
    save_untouch_nii(data,'tmp.nii.gz');
end
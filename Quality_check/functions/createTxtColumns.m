function createTxtColumns(indx,fileOut)

% indx = vals>thr;            
indxNums = find(indx);
%totalVolumes = numel(indx);
%lostVolumes = length(indxNums);
if ~isempty(indxNums)
    matTxt = zeros(length(indx),length(indxNums));
    for nCol = 1:length(indxNums)
        matTxt(indxNums(nCol),nCol) = 1;
    end
else
    matTxt = zeros(length(vals),1);
end



disp(['Writting file: ',fileOut]);
writematrix(matTxt,fileOut);
%xlswrite(fileOut,logical(matTxt));
%dlmwrite(fileOut,matTxt);


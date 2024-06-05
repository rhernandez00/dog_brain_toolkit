function writeTxt(outputFile,matTxt)
%writes a txt file (outputFile) from the matrix matTxt, if matTxt is a
%table, it will write a csv file with the variable names in the first row

if exist(outputFile,'file') %checks if the file exist, if it does, it erases it
    delete(outputFile);
end

fileID = fopen(outputFile,'w'); 

if istable(matTxt)
    fNames = matTxt.Properties.VariableNames;
    lineY = fNames{1};
    for nField = 2:numel(fNames)
        lineY = [lineY,' ',fNames{nField}];
    end
    fprintf(fileID,lineY,'%s');
    fprintf(fileID,'\n');
    
    matTxt = matTxt{:,:};
end    

    
for nRow = 1:size(matTxt,1) %writes every line from matTxt
    lineX = num2str(matTxt(nRow,:));
    for n = 1:20 %removes extra white spaces
        lineX = strrep(lineX,'  ',' ');
    end
    lineY = strrep(lineX,' ',' ');
    fprintf(fileID,lineY,'%s');
    if nRow ~= size(matTxt,1)
        fprintf(fileID,'\n');
    end
end
fclose(fileID);

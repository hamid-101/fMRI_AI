function [labels,tag,chunk] = readlable(filename)
delimiter = ' ';
startRow = 2;
formatSpec = '%s%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'MultipleDelimsAsOne', true, 'HeaderLines' ,startRow-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
fclose(fileID);
tmp = dataArray{:, 1};
chunk = dataArray{:, 2};
tag = unique(tmp);
tmp1 = tag{end} ;
tag{end} = tag{6};
tag{6} = tmp1;
labels = zeros(size(tmp));

for j=1:length(tmp)
    for i=1:length(tag)
        if strcmp(tmp(j),tag(i))
            labels(j) = i;
        end
    end
end
end

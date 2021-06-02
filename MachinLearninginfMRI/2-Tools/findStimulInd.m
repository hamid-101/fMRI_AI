function [Ind]=findStimulInd(X,tag,name)
tmp = [];
for i=1:length(tag)
   if strfind(tag{i},name)
      tmp = i;
      break;
   end
end
Ind = find(X==tmp);
end

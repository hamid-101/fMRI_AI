%% Copyright (C) ©2021,Hamid Hakim;‘All rights reserved’ 
clear
warning('On')
addpath('..\\2-Tools\niftitools');
addpath('..\\2-Tools\classifier');
addpath('..\\2-Tools\classifier\kernel');
addpath('..\\2-Tools\classifier\optimisation');
addpath('..\\2-Tools\utils');
addpath('..\\2-Tools\external');
addpath('..\\2-Tools\');
DataFolder = '..\\1-Data\haxby2001\subj2\';
hos = 'mask8b_house_vt.nii';
fac = 'mask8_face_vt.nii';
mas = 'mask.nii';
bol = 'bold.nii';
lab = 'labels.txt';
ClassifyMethod = {'svm','lda','logreg','ensemble'};

AllClassMask = 'mask4_vt.nii';
[label,tag,chunk] = readlable([DataFolder,lab]);
inputdata = load_nii([DataFolder,bol]);
AllClassMaskData = load_nii([DataFolder,AllClassMask]);
X = double(inputdata.img);
mask = double(AllClassMaskData.img);

%% Mask Importan Voxels
xdata = mask4d(X,mask);
RawData = zscore(xdata,0,2);
%% Exctract Time Course
group = label;

NPCAfeature = [0,1,5,10,100,463];
for index=1:length(NPCAfeature)
    NFeat = NPCAfeature(index);
    if NFeat~=0
        xdata = pca(RawData)';
        xdata = xdata(1:NFeat,:);
    else
        xdata = RawData;
    end
    %% Classification with chunk leaveOneGroupOut
    category=6;
    Y = ones(size(group));
    Y(group==category) = 2;

    CVO.chunk = unique(chunk);
    CVO.NumTestSets = length(unique(chunk));

    for i = 1:CVO.NumTestSets
        % Create Train & Test fold
        trIdx = chunk~=i-1;
        teIdx = chunk==i-1;
        CVO.TestSize(i) = sum(teIdx);

        % Modeling
        ytest = MyClassify('svm',xdata(:,trIdx)',Y(trIdx),xdata(:,teIdx)');
% pca();
        % Accuracy
        err(index,i) = ClassifyScore(ytest,Y(teIdx));
        fprintf('%d,',(i));
    end
end
boxplot(err)
title('Category-specific classification accuracy for Rest classifiers');
ylabel('Classification accurancy (f1 score)');   
colormap('jet')
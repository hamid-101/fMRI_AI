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
[label,tag,chunk] = Myreadlable([lab]);
inputdata = load_nii([DataFolder,bol]);
AllClassMaskData = load_nii([DataFolder,AllClassMask]);
X = double(inputdata.img);
mask = double(AllClassMaskData.img);

%%
%% Voxel Selection
%% Mask Importan Voxels
xdata = mask4d(X,mask);
xdata = zscore(xdata,0,2);
%% Exctract Time Course
group = label;

%% Classification with chunk leaveOneGroupOut
category=9; %% 9=Rest

Y = ones(size(group));
Y(group==category) = 2;

CVO.chunk = unique(chunk);
CVO.NumTestSets = length(unique(chunk));

err = zeros(CVO.NumTestSets,1);
for i = 1:CVO.NumTestSets
    % Create Train & Test fold
    trIdx = chunk~=i-1 ;%CVO.training(i);
    teIdx = chunk==i-1;%CVO.test(i);
    CVO.TestSize(i) = sum(teIdx);

    % Modeling
    ytest = MyClassify('lda',xdata(:,trIdx)',Y(trIdx),xdata(:,teIdx)');

    % Accuracy
    err(i) = ClassifyScore(ytest,Y(teIdx));
    fprintf('%d,',(i));
end
boxplot(err)
title('Category-specific classification accuracy for Rest classifiers');
ylabel('Classification accurancy (f1 score)');   
colormap('jet')
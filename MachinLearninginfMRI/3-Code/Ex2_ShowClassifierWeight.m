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
T1 = 'anat.nii';
lab = 'labels.txt';
ClassifyMethod = {'lda','svm','logreg','ensemble'};

AllClassMask = 'mask4_vt.nii';
[label,tag,chunk] = Myreadlable([lab]);
inputdata = load_nii([DataFolder,bol]);
AllClassMaskData = load_nii([DataFolder,AllClassMask]);
AnatomData = load_nii([DataFolder,AllClassMask]);
X = double(inputdata.img);
mask = double(AllClassMaskData.img);



% HouseClassMaskData = load_nii([DataFolder,hos]);
% FaceClassMaskData = load_nii([DataFolder,fac]);
% houseMask = double(HouseClassMaskData.img);
% FaceMask = double(HouseClassMaskData.img);
faceInd=findStimulInd(label,tag,'face');
houseInd=findStimulInd(label,tag,'house');
[xdata,inds] = mask4d(X,mask);
xdata = zscore(xdata,0,2);


%% Exctract Weighted of Voxels
xdata = xdata(:,([faceInd;houseInd]));
Y = label(([faceInd;houseInd]));
[~,model] = MyClassify('svm',xdata',Y,[]);
WhiteBrain = zeros(size(mask));%mean(X,4)./10;
WhiteBrain(inds) = (model.w);

AnatomData.img = WhiteBrain;
nii= AnatomData;%make_nii(WhiteBrain);
save_nii(nii,'ClassifierWeight.nii');
view_nii(nii);
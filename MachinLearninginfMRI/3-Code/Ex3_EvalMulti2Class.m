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
ClassifyMethod = {'lda','svm','logreg','ensemble'};

AllClassMask = 'mask4_vt.nii';
[label,tag,chunk] = Myreadlable([DataFolder,lab]);
inputdata = load_nii([DataFolder,bol]);
AllClassMaskData = load_nii([DataFolder,AllClassMask]);
X = double(inputdata.img);
mask = double(AllClassMaskData.img);

%%
%% Voxel Selection






%% Find nonRest Time
RestInd=findStimulInd(label,tag,'rest');
%% Mask Importan Voxels
xdata = mask4d(X,mask);
xdata = zscore(xdata,0,2);
%% Exctract Time Course
xdata(:,RestInd)=[];
group = label;
group(RestInd)=[];
chunk(RestInd)=[];

tag = tag(1:end-1);% remove rest tag
cvErr = zeros(length(ClassifyMethod),length(tag),2);    
for method = 2:length(ClassifyMethod)
    %% Classification with chunk leaveOneGroupOut
    fprintf('\n\n\n%s:',ClassifyMethod{method});
    for category=1:length(tag)
        fprintf('\n\t%s:\t\t',tag{category});
        Y = ones(size(group));
        Y(group==category) = 2;

        CVO.chunk = unique(chunk);
        CVO.NumTestSets = length(unique(chunk));

        err = zeros(CVO.NumTestSets,1);
        for i = 1:CVO.NumTestSets
            % Create Train & Test fold
            trIdx = chunk~=i-1;%CVO.training(i);
            teIdx = chunk==i-1;%CVO.test(i);
            CVO.TestSize(i) = sum(teIdx);

            % Modeling
            ytest = MyClassify(ClassifyMethod{method},xdata(:,trIdx)',Y(trIdx),xdata(:,teIdx)');

            % Accuracy
            err(i) = ClassifyScore(ytest,Y(teIdx));
            fprintf('%d,',(i));
        end
        cvErr(method, category,:) = [mean(err),std((err))];
    end
end
bar(cvErr(:,:,1)')
title('Category-specific classification accuracy for different classifiers');
ylabel('Classification accurancy (f1 score)');
set(gca, 'XTickLabel', tag);
legend(ClassifyMethod);    
colormap('jet')






% %% Classification with CV
% cvErr = zeros(length(tag),2);
% for category=1:length(tag)
%     fprintf('\n%s:\t',tag{category});
%     Y = zeros(size(group));
%     Y(group==category) = 1;
%     CVO = cvpartition(Y,'k',20);
%     err = zeros(CVO.NumTestSets,1);
%     for i = 1:CVO.NumTestSets
%         trIdx = CVO.training(i);
%         teIdx = CVO.test(i);
%         
%         param = mv_get_classifier_param('lda');
%         model = train_lda(param,xdata(:,trIdx)',Y(trIdx));
%         [ytest,dval,~]  = test_lda(model,xdata(:,teIdx)');
% 
%         
% %         cf = train_svm([],xdata(:,trIdx)',group(trIdx));
% %         svmStruct = svmtrain(xdata(:,trIdx)',group(trIdx));
% %         ytest = svmclassify(svmStruct,xdata(:,trIdx)');
% 
%         err(i) = sum(ytest==Y(teIdx));
%         fprintf('%d,',(i));
%     end
%     cvErr(category,:) = [sum(err)/sum(CVO.TestSize),std((err)./(CVO.TestSize)')];
% end
% disp(1);
%% Weigh of Each Voxel






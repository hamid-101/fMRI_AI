%% Copyright (C) ©2021,Hamid Hakim;‘All rights reserved’ 
clear
warning('Off')
addpath('..\\2-Tools\niftitools');
addpath('..\\2-Tools\classifier');
DataFolder = '..\\1-Data\haxby2001\subj2\';
hos = 'mask8b_house_vt.nii';
fac = 'mask8_face_vt.nii';
mas = 'mask.nii';
bol = 'bold.nii';
lab = 'labels.txt';
ClassifyMethod = {'multiclass_lda'};

AllClassMask = 'mask4_vt.nii';
[label,tag,chunk] = readlable([DataFolder,lab]);
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

cvErr = zeros(length(ClassifyMethod),length(tag),2);    
method= 1;
%% Classification with chunk leaveOneGroupOut
category=1;
    Y = group;
    CVO.chunk = unique(chunk);
    CVO.NumTestSets = length(unique(chunk));

    err = zeros(CVO.NumTestSets,1);
    for i = 1:CVO.NumTestSets
        % Create Train & Test fold
        trIdx = chunk~=i-1;%CVO.training(i);
        teIdx = chunk==i-1;%CVO.test(i);
        CVO.TestSize(i) = sum(teIdx);
        
        % Modeling
        a = pca(xdata)';
        ab = a(1:15,:);
        Model = fitcecoc(ab(:,trIdx)',Y(trIdx),'Learners','svm');
        [ytest,dval] = predict(Model,ab(:,teIdx)');
        
        for k=1:9
           output(:,k) =  double(ytest==k);
           trueVal(:,k) = double(Y(teIdx)==k);
        end
        
        [c,cm(:,:,i),ind,per] = confusion(trueVal',output');
        % Accuracy
        fprintf('%d,',(i));
    end
   


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






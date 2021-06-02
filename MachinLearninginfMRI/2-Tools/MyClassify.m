function [ytest,model] = MyClassify(Method,TrainData,TrainLabel,TestData)
ytest = [];
param = mv_get_classifier_param(Method);
train_model = str2func(['train_',Method]);
test_model = str2func(['test_',Method]);

model = train_model(param,TrainData,TrainLabel);
if size(TestData,1)>1
[ytest,dval]  = test_model(model,TestData);
end
end
function [F1] = ClassifyScore(yHaT,yval)
tp = sum((yHaT == 2) & (yval == 2));
fp = sum((yHaT == 2) & (yval == 1));
fn = sum((yHaT == 1) & (yval == 2));

precision = (tp+0.00000000001) / (tp + fp+0.00000000001);
recall = tp / (tp + fn);
F1 = (2 * precision * recall) / (precision + recall);
end
% precision = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);
% 
% recall = @(confusionMat) diag(confusionMat)./sum(confusionMat,1)';
% 
% f1Scores = @(confusionMat) 2*(precision(confusionMat).*recall(confusionMat))./(precision(confusionMat)+recall(confusionMat))
% 
% meanF1 = @(confusionMat) mean(f1Scores(confusionMat))
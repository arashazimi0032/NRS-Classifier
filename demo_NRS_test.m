clear all; close all; clc

load U_PaviaDataset_tuning

% Normalize
Normalize = max(DataTrain(:));
DataTrain = DataTrain./Normalize;
DataTest = DataTest./Normalize;

lambda = [0.5]; % need to find the optimal

class = NRS_Classification(DataTrain, CTrain, DataTest, lambda);
[confusion, accur_NRS, TPR, FPR] = confusion_matrix_wei(class, CTest);


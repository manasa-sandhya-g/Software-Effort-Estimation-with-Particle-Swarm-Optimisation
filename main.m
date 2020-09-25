%%%%%% MATLAB initializations and some parameters and variables:
clc
clear
clear all
close all
addpath('./../data/')
addpath('./M5PrimeLab/')
warning('off', 'all')
global report_progress_1; report_progress_1 = 1;  % report dataset
global report_progress_2; report_progress_2 = 1;  % report index of fold
global report_progress_3; report_progress_3 = 1;  % report regression method
global report_progress_4; report_progress_4 = 1;  % report index of simulation
global report_progress_5; report_progress_5 = 1;  % report index of generation in GA
global report_progress_6; report_progress_6 = 0;  % report index of chromosome in GA
global MMRE_overall;
global PRED_25_overall;
global Sum_Ab_Res_overall;
global Med_Ab_Res_overall;
global SD_Ab_Res_overall;
global number_of_removed_features_overall;
population_size = 500;
number_of_generations = 25;
number_of_PSO_simulations = 10;


%%%%%% Reading datasets:
number_of_datasets = 3;
data = cell(number_of_datasets, 1);
folder = "D:/UWaterloo/WINTER2020/ECE602/pro/data/";
for dataset_index = 1:number_of_datasets
    if dataset_index == 1
        filename = 'D.txt';
        number_of_features = 13;
    elseif dataset_index == 2
        filename = 'Albrecht.txt';
        number_of_features = 9;
    elseif dataset_index == 3
        filename = 'Kemerer.txt';
        number_of_features = 9;
    end
    [fileID, msg] = fopen(fullfile(folder,filename),'rt');
    disp(fileID);
    if fileID < 0
        error('Failed to open file "%s" because: "%s"', filename, msg);
    end
    format = '%f';
    for feature_index = 1:number_of_features-1
        format = [format, ' %f'];
    end
    disp(format);
    C = textscan(fileID,format,...
        'TreatAsEmpty',{'?'},'EmptyValue',nan);
    disp(C{1});
    fclose(fileID);

    %%%%%% fixing missing data:
    for feature_index = 1:length(C)
        for sample_index = 1:length(C{feature_index})
            if isnan(C{feature_index}(sample_index))
                average_of_feature = nanmean(C{feature_index}); % taking mean ignoring nan values
                C{feature_index}(sample_index) = average_of_feature;
            end
        end
    end

    %%%%%% preparing X and Y of datasets:
    if dataset_index == 1   %--> Desharnais dataset
        data{dataset_index}.features = [];
        for feature_index = 1:length(C)
            if feature_index ~= 1 && feature_index ~= 2 && feature_index ~= 6 && feature_index ~= 7 
                %%%%%% feature_index == 1 --> ID
                %%%%%% feature_index == 2 --> project ID
                %%%%%% feature_index == 6 --> project length
                %%%%%% feature_index == 7 --> effort
                data{dataset_index}.features = [data{dataset_index}.features, C{feature_index}];
            end
        end
        data{dataset_index}.efforts = C{7};
    elseif dataset_index == 2   %--> Albrecht dataset
        data{dataset_index}.features = [];
        for feature_index = 1:length(C)
            if feature_index ~= 1 && feature_index ~= 6 && feature_index ~= 9 
                %%%%%% feature_index == 1 --> ID
                %%%%%% feature_index == 6 --> FPAdj --> papers don't mention it
                %%%%%% feature_index == 9 --> effort
                data{dataset_index}.features = [data{dataset_index}.features, C{feature_index}];
            end
        end
        data{dataset_index}.efforts = C{9};
    elseif dataset_index == 3   %--> Kemerer dataset
        data{dataset_index}.features = [];
        for feature_index = 1:length(C)
            if feature_index ~= 1 && feature_index ~= 2 && feature_index ~= 9
                %%%%%% feature_index == 1 --> ID
                %%%%%% feature_index == 2 --> project ID
                %%%%%% feature_index == 9 --> effort
                data{dataset_index}.features = [data{dataset_index}.features, C{feature_index}];
            end
        end
        data{dataset_index}.efforts = C{9};
    end

end


%%%%%% experiments on datasets:
dataset_index = 0;
for dataset = {'Desharnais', 'Albrecht', 'Kemerer'}
%for dataset = {'Kemerer'}
    dataset_index = dataset_index + 1;
    dataset_name = cell2mat(dataset);
    if report_progress_1 == true; str = sprintf('=========================> Processing Dataset: %s', dataset_name); disp(str); end;
    if strcmp(dataset_name, 'Desharnais')
        X = data{1}.features;
        Y = data{1}.efforts;
        number_of_samples = size(X, 1);
        number_of_folds = 1;
        rand_index = randperm(number_of_samples);
        X_train_folds{1} = X(rand_index(1:63), :);
        Y_train_folds{1} = Y(rand_index(1:63), :);
        X_test_folds{1} = X(rand_index(64:end), :);
        Y_test_folds{1} = Y(rand_index(64:end), :);
    elseif strcmp(dataset_name, 'Albrecht')
        X = data{2}.features;
        Y = data{2}.efforts;
        number_of_samples = size(X, 1);
        number_of_folds = number_of_samples;
        for fold_index = 1:number_of_folds
            mask_train = true(number_of_samples, 1);
            mask_train(fold_index) = false;
            X_train_folds{fold_index} = X(mask_train, :);
            Y_train_folds{fold_index} = Y(mask_train, :);
            X_test_folds{fold_index} = X(~mask_train, :);
            Y_test_folds{fold_index} = Y(~mask_train, :);
        end
    elseif strcmp(dataset_name, 'Kemerer')
        X = data{3}.features;
        Y = data{3}.efforts;
        number_of_samples = size(X, 1);
        number_of_folds = number_of_samples;
        for fold_index = 1:number_of_folds
            mask_train = true(number_of_samples, 1);
            mask_train(fold_index) = false;
            X_train_folds{fold_index} = X(mask_train, :);
            Y_train_folds{fold_index} = Y(mask_train, :);
            X_test_folds{fold_index} = X(~mask_train, :);
            Y_test_folds{fold_index} = Y(~mask_train, :);
        end
    end

    for fold_index = 1:number_of_folds
        if report_progress_2 == true; str = sprintf('=============> Index of fold: %d', fold_index); disp(str); end;
        X_train = X_train_folds{fold_index};
        Y_train = Y_train_folds{fold_index};
        X_test = X_test_folds{fold_index};
        Y_test = Y_test_folds{fold_index};
        method_index = 0;
        %for regression_method = {'SVR with RBF kernel', 'SVR with linear kernel', 'MLP', 'M5P'}
        for regression_method = {'SVR with RBF kernel'}
            regression_method_ = cell2mat(regression_method);
            if report_progress_3 == true; str = sprintf('*********** Regression method: %s', regression_method_); disp(str); end;
            method_index = method_index + 1;
            best_MMRE = zeros(number_of_PSO_simulations, 1); best_PRED_25 = zeros(number_of_PSO_simulations, 1);
            best_Sum_Ab_Res = zeros(number_of_PSO_simulations, 1); best_Med_Ab_Res = zeros(number_of_PSO_simulations, 1);
            best_SD_Ab_Res = zeros(number_of_PSO_simulations, 1); best_number_of_removed_features = zeros(number_of_PSO_simulations, 1);
            for simulation_index = 1:number_of_PSO_simulations
                if report_progress_4 == true; str = sprintf('****** Index of simulation: %d', simulation_index); disp(str); end;
                [~, best_MMRE(simulation_index), best_PRED_25(simulation_index), best_Sum_Ab_Res(simulation_index), ...
                    best_Med_Ab_Res(simulation_index), best_SD_Ab_Res(simulation_index), ...
                    best_number_of_removed_features(simulation_index)] = PSO(population_size, ...
                    number_of_generations, ...
                    regression_method_, X_train, Y_train, X_test, Y_test, dataset_name);
            end
            MMRE(fold_index,method_index).average = mean(best_MMRE);
            MMRE(fold_index,method_index).std = std(best_MMRE);
            PRED_25(fold_index,method_index).average = mean(best_PRED_25);
            PRED_25(fold_index,method_index).std = std(best_PRED_25);
            Sum_Ab_Res(fold_index,method_index).average = mean(best_Sum_Ab_Res);
            Sum_Ab_Res(fold_index,method_index).std = std(best_Sum_Ab_Res);
            Med_Ab_Res(fold_index,method_index).average = mean(best_Med_Ab_Res);
            Med_Ab_Res(fold_index,method_index).std = std(best_Med_Ab_Res);
            SD_Ab_Res(fold_index,method_index).average = mean(best_SD_Ab_Res);
            SD_Ab_Res(fold_index,method_index).std = std(best_SD_Ab_Res);
            number_of_removed_features(fold_index,method_index).average = mean(best_number_of_removed_features);
            number_of_removed_features(fold_index,method_index).std = std(best_number_of_removed_features);
        end
    end
    method_index = 0;
    %for regression_method = {'SVR with RBF kernel', 'SVR with linear kernel', 'MLP', 'M5P'}
    for regression_method = {'SVR with RBF kernel'}
        method_index = method_index + 1;
        sum1 = 0; sum2 = 0; sum3 = 0; sum4 = 0; sum5 = 0; sum6 = 0;
        sum7 = 0; sum8 = 0; sum9 = 0; sum10 = 0; sum11 = 0; sum12 = 0;
        for fold_index = 1:number_of_folds
            sum1 = sum1 + MMRE(fold_index,method_index).average;
            sum2 = sum2 + PRED_25(fold_index,method_index).average;
            sum3 = sum3 + Sum_Ab_Res(fold_index,method_index).average;
            sum4 = sum4 + Med_Ab_Res(fold_index,method_index).average;
            sum5 = sum5 + SD_Ab_Res(fold_index,method_index).average;
            sum6 = sum6 + number_of_removed_features(fold_index,method_index).average;
            sum7 = sum7 + MMRE(fold_index,method_index).std;
            sum8 = sum8 + PRED_25(fold_index,method_index).std;
            sum9 = sum9 + Sum_Ab_Res(fold_index,method_index).std;
            sum10 = sum10 + Med_Ab_Res(fold_index,method_index).std;
            sum11 = sum11 + SD_Ab_Res(fold_index,method_index).std;
            sum12 = sum12 + number_of_removed_features(fold_index,method_index).std;
        end
        MMRE_overall(dataset_index, method_index).average = sum1 / number_of_folds;
        PRED_25_overall(dataset_index, method_index).average = sum2 / number_of_folds;
        Sum_Ab_Res_overall(dataset_index, method_index).average = sum3 / number_of_folds;
        Med_Ab_Res_overall(dataset_index, method_index).average = sum4 / number_of_folds;
        SD_Ab_Res_overall(dataset_index, method_index).average = sum5 / number_of_folds;
        number_of_removed_features_overall(dataset_index, method_index).average = sum6 / number_of_folds;
        MMRE_overall(dataset_index, method_index).std = sum7 / number_of_folds;
        PRED_25_overall(dataset_index, method_index).std = sum8 / number_of_folds;
        Sum_Ab_Res_overall(dataset_index, method_index).std = sum9 / number_of_folds;
        Med_Ab_Res_overall(dataset_index, method_index).std = sum10 / number_of_folds;
        SD_Ab_Res_overall(dataset_index, method_index).std = sum11 / number_of_folds;
        number_of_removed_features_overall(dataset_index, method_index).std = sum12 / number_of_folds;
    end
end



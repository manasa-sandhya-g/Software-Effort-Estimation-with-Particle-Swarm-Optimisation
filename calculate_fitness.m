function [fitness, MMRE, PRED_25, Sum_Ab_Res, Med_Ab_Res, SD_Ab_Res] = calculate_fitness(X_train, Y_train, X_test, Y_test, regression_method, parameters, selected_features, dataset_name)

%%%%% selecting features:
number_of_features = size(X_train, 2);
X_train_featureSelected = [];
X_test_featureSelected = [];
for feature_index = 1:number_of_features
    if selected_features(feature_index) == 1
        X_train_featureSelected = [X_train_featureSelected, X_train(:, feature_index)];
        X_test_featureSelected = [X_test_featureSelected, X_test(:, feature_index)];
    end
end
X_train = X_train_featureSelected;
X_test = X_test_featureSelected;

%%%%% regression:
if strcmp(regression_method, 'SVR with RBF kernel')
    C = round(parameters(1));
    epsilon = parameters(2);
    gamma = parameters(3);
    SVMModel = fitrsvm(X_train,Y_train,'KernelFunction','rbf','BoxConstraint',C,'Epsilon',epsilon,'KernelScale',1/sqrt(gamma));
    Y_test_hat = predict(SVMModel, X_test);
elseif strcmp(regression_method, 'SVR with linear kernel')
    C = round(parameters(1));
    epsilon = parameters(2);
    SVMModel = fitrsvm(X_train,Y_train,'KernelFunction','linear','BoxConstraint',C,'Epsilon',epsilon);
    Y_test_hat = predict(SVMModel, X_test);
elseif strcmp(regression_method, 'MLP')
    number_of_nodes_in_hidden_layer = round(parameters(1));
    hidden_layers = [number_of_nodes_in_hidden_layer];
    net = fitnet(hidden_layers,'trainbr');
    net.trainParam.showWindow = false;   % hide the window of training
    net.trainParam.showCommandLine = false;  % hide the window of training
    nntraintool('close')   % hide the window of training
    net = train(net, X_train', Y_train');
    net.trainParam.epochs = round(parameters(2));
    net.trainParam.lr = parameters(3);
    net.trainParam.mc = parameters(4);
    %view(net)
    Y_test_hat = net(X_test');
    Y_test_hat = Y_test_hat';
elseif strcmp(regression_method, 'M5P')
    modelTree = true;
    minLeafSize = round(parameters(1));
    if round(parameters(2)) == 1
        prune = true;
    else
        prune = false;
    end
    if round(parameters(3)) == 1
        smoothingK = 15;
    else
        smoothingK = 0;
    end
%     trainParams = m5pparams(modelTree, minLeafSize, [], prune, smoothingK);
%     [model, time, ensembleResults] = m5pbuild(X_train, Y_train, trainParams);
%     Y_test_hat = m5ppredict(model, X_test);
    % hiding the displays of "tree" functions: --> https://stackoverflow.com/questions/9518146/suppress-output
    [T,trainParams] = evalc('m5pparams(modelTree, minLeafSize, [], prune, smoothingK);');
    [T,model, time, ensembleResults] = evalc('m5pbuild(X_train, Y_train, trainParams);');
    [T,Y_test_hat] = evalc('m5ppredict(model, X_test);');
end

%%%%% calculate error/accuracy metrics:
number_of_test_samples = length(Y_test);
MMRE = (1/number_of_test_samples) * sum(abs(Y_test - Y_test_hat) ./ Y_test);
k = 0;
for test_sample_index = 1:number_of_test_samples
    MRE_of_sample = abs(Y_test(test_sample_index) - Y_test_hat(test_sample_index)) / Y_test(test_sample_index);
    if MRE_of_sample <= 0.25
        k = k + 1;
    end
end
PRED_25 = 100 * (k / number_of_test_samples);
Ab_Res = abs(Y_test - Y_test_hat);
Sum_Ab_Res = sum(Ab_Res);
Med_Ab_Res = median(Ab_Res);
SD_Ab_Res = std(Ab_Res);

%%%%% calculate fitness:
if ~strcmp(dataset_name, 'COCOMO')
    fitness = (100 - PRED_25) + MMRE;
else  % if dataset is COCOMO
    fitness = MMRE;
end

end
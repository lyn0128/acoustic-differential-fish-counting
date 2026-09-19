function results = train_bp_model(inputFile, cageId)
%TRAIN_BP_MODEL Bayesian-regularised BP fitting with an independent test set.

T = readtable(inputFile, VariableNamingRule="preserve");
T = T(logical(T.valid) & string(T.cage_id) == string(cageId), :);
ringNames = compose("ring_%02d", 1:59);
X = table2array(T(:, ringNames))';
y = T.reference_count';

if height(T) ~= 27
    warning("Expected 27 valid rows for a cage; found %d.", height(T));
end

% The study design uses 18 datasets for model development and 9 for
% independent validation. The model-development subset is split 80/20.
development = 1:min(18, height(T));
independent = 19:min(27, height(T));
rng(2026, "twister");
order = development(randperm(numel(development)));
nTrain = floor(0.80 * numel(development));
trainIdx = order(1:nTrain);
internalTestIdx = order(nTrain+1:end);

net = fitnet(30, "trainbr");
net.divideFcn = "divideind";
net.divideParam.trainInd = trainIdx;
net.divideParam.valInd = [];
net.divideParam.testInd = internalTestIdx;
net.performFcn = "mse";
[net, trainingRecord] = train(net, X(:, development), y(development));

prediction = net(X);
error = prediction - y;
absoluteError = abs(error);
errorRate = 100 * absoluteError ./ max(abs(y), eps);
accuracy = 100 - errorRate;

results = table(string(T.dataset_id), y', prediction', error', errorRate', accuracy', ...
    VariableNames=["dataset_id","reference_count","prediction","error", ...
                   "error_rate_percent","accuracy_percent"]);
results.Properties.UserData.net = net;
results.Properties.UserData.trainingRecord = trainingRecord;
results.Properties.UserData.independentRows = independent;
writetable(results, "bp_predictions_" + string(cageId) + ".csv");
end


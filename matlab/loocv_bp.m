function results = loocv_bp(inputFile)
%LOOCV_BP Leave one cage-level dataset out with fixed hyperparameters.

T = readtable(inputFile, VariableNamingRule="preserve");
T = T(logical(T.valid), :);
if ismember("direct_estimate", string(T.Properties.VariableNames))
    X = T.direct_estimate';
else
    ringNames = compose("ring_%02d", 1:59);
    X = sum(table2array(T(:, ringNames)), 2)';
end
y = T.reference_count';
n = height(T);
prediction = nan(1, n);

for heldOut = 1:n
    trainIdx = setdiff(1:n, heldOut);
    rng(2026 + heldOut, "twister");
    net = fitnet(30, "trainbr");
    net.divideFcn = "dividetrain";
    net.performFcn = "mse";
    net = train(net, X(:, trainIdx), y(trainIdx));
    prediction(heldOut) = net(X(:, heldOut));
end

error = prediction - y;
absoluteError = abs(error);
errorRate = 100 * absoluteError ./ max(abs(y), eps);
accuracy = 100 - errorRate;
results = table(string(T.dataset_id), y', prediction', error', errorRate', accuracy', ...
    VariableNames=["dataset_id","reference_count","prediction","error", ...
                   "error_rate_percent","accuracy_percent"]);
writetable(results, "loocv_predictions.csv");
fprintf("LOOCV MAE: %.4f\n", mean(absoluteError));
fprintf("LOOCV RMSE: %.4f\n", sqrt(mean(error.^2)));
fprintf("LOOCV mean accuracy: %.4f%%\n", mean(accuracy));
fprintf("LOOCV error-rate SD: %.4f%%\n", std(errorRate));
end

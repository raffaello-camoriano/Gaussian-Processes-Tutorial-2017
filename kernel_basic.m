clearAllButBP
close all

%% Make dataset

% Dataset settings
n = 20;
nTe = 100;
w = 1;
sigma_n = 0.2;
xMin = 0;
xMax = 2*pi;

[ X , Y ] = generate_sinusoidal_DS(n, w, sigma_n, xMin, xMax);
[ Xte , Yte ] = generate_sinusoidal_DS(nTe, w, sigma_n, xMin, xMax);


%% Compute posterior distribution

% Prior settings
kernel = 'gaussian';
kerPar = 2; % prior over the parameters (since N = 1, Sigma_p is a scalar)
sigma_f = 1;

K = KernelMatrix(X', X', kernel, kerPar, sigma_f);
alpha = (K + sigma_n^2 * eye(n)) \ Y;


%% Plot resulting distribution and test predictions

% test grid
Xgrid = linspace(xMin, xMax, 200);
K_gridTrain = KernelMatrix(Xgrid', X', kernel, kerPar, sigma_f);
K_gridGrid = KernelMatrix(Xgrid', Xgrid', kernel, kerPar, sigma_f);

Ygrid_pred_mean = K_gridTrain * alpha;
Ygrid_pred_cov = K_gridGrid + sigma_n^2 * eye(size(K_gridGrid)) - K_gridTrain *  ((K + sigma_n^2 * eye(n)) \ K_gridTrain');
Ygrid_pred_std = sqrt(diag(Ygrid_pred_cov))';

figure
hold on
title('Training set and predictive distribution')
scatter(X, Y)
% plot mean
plot(Xgrid', Ygrid_pred_mean)
% plot +2 std
plot(Xgrid', Ygrid_pred_mean + 2 * Ygrid_pred_std', '--b')
% plot -2 std
plot(Xgrid', Ygrid_pred_mean - 2 * Ygrid_pred_std', '--b')
hold off

% test set predictions
K_testTrain = KernelMatrix(Xte', X', kernel, kerPar, sigma_f);
K_testTest = KernelMatrix(Xte', Xte', kernel, kerPar, sigma_f);

Yte_pred_mean = K_testTrain * alpha;
Yte_pred_cov = K_testTest + sigma_n^2 * eye(size(K_testTest)) - K_testTrain *  ((K + sigma_n^2 * eye(n)) \ K_testTrain');
Yte_pred_std = sqrt(diag(Yte_pred_cov))';

figure
hold on
title('Test predictions and predictive distribution')
% plot mean
plot(Xgrid', Ygrid_pred_mean)
% plot +2 std
plot(Xgrid', Ygrid_pred_mean + 2 * Ygrid_pred_std', '--b')
% plot -2 std
plot(Xgrid', Ygrid_pred_mean - 2 * Ygrid_pred_std', '--b')
% plot test points
scatter(Xte, Yte, 'xr')
% plot predictions
scatter(Xte, Yte_pred_mean, 'or')
% plot connecting lines
plot([Xte;Xte], [Yte'; Yte_pred_mean'], ':r')

%% Log-Marginal likelihood

LML = -0.5 * Y' * alpha - 0.5 * real(log(det(K+ sigma_n^2 * eye(n)))) - 0.5  * n * log(2*pi)
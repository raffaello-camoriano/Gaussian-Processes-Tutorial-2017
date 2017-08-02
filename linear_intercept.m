clearAllButBP
close all

%% Make dataset

% Dataset settings
n = 100;
nTe = 1000;
m = 0.5;
q = 1;
sigma_n = 0.2;
xMin = 100;
xMax = 101;

[ X , Y ] = generate_linear_DS(n, m, q, sigma_n, xMin, xMax); % training
X = [X; ones(size(X))];

[ Xte , Yte ] = generate_linear_DS(nTe, m, q, sigma_n, xMin, xMax); % test
Xte = [Xte; ones(size(Xte))];


%% Compute posterior distribution

% Prior settings
Sigma_p = eye(2) * 1000; % prior over the parameters (since N = 1, Sigma_p is a scalar)
% Sigma_p = [10; 0.000000001] .* eye(2) * 1000; % prior over the parameters (since N = 1, Sigma_p is a scalar)

A = 1/(sigma_n^2) * (X * X') + inv(Sigma_p);
invA = inv(A);
w_mean = 1/(sigma_n^2) * invA * X * Y;
w_cov = invA;

%% Plot resulting distribution and test predictions

% test grid
Xgrid = linspace(xMin, xMax, 10000);
Xgrid = [Xgrid; ones(size(Xgrid))];

Ygrid_pred_mean =  Xgrid' * w_mean;
Ygrid_pred_cov = Xgrid' * w_cov * Xgrid;
Ygrid_pred_std = sqrt(diag(Ygrid_pred_cov));
min(Ygrid_pred_std)
max(Ygrid_pred_std)
% Ygrid_pred_std = diag(Ygrid_pred_cov)';

figure
hold on
title('Predictive distribution & training set')
% plot mean
plot(Xgrid(1,:), Ygrid_pred_mean)
% plot +2 std
plot(Xgrid(1,:), Ygrid_pred_mean + 2 * Ygrid_pred_std, '--b')
% plot -2 std
plot(Xgrid(1,:), Ygrid_pred_mean - 2 * Ygrid_pred_std, '--b')
% plot training
scatter(X(1,:), Y, 'g')

% test set predictions
Yte_pred_mean =  Xte' * w_mean;
Yte_pred_cov = Xte' * w_cov * Xte;
Yte_pred_std = sqrt(diag(Yte_pred_cov))';

figure
hold on
title('Test predictions and predictive distribution')
% plot mean
plot(Xgrid(1,:), Ygrid_pred_mean)
% plot +2 std
plot(Xgrid(1,:), Ygrid_pred_mean + 2 * Ygrid_pred_std, '--b')
% plot -2 std
plot(Xgrid(1,:), Ygrid_pred_mean - 2 * Ygrid_pred_std, '--b')
% plot test points
scatter(Xte(1,:), Yte, 'xr')
% plot predictions
scatter(Xte(1,:), Yte_pred_mean, 'or')
% plot connecting lines
plot([Xte(1,:);Xte(1,:)], [Yte'; Yte_pred_mean'], ':r')

mean(X(1,:))
[m, idx] = min(Ygrid_pred_std);
Xgrid(1,idx)


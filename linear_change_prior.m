clearAllButBP
close all

%% Make dataset

% Dataset settings
n = 15;
m = 0.5;
sigma_n = 0.1;
xMin = 0;
xMax = 1;

[ X , Y ] = generate_linear_DS(n, m, sigma_n, xMin, xMax); % training
[ Xte , Yte ] = generate_linear_DS(n, m, sigma_n, xMin, xMax); % test

figure
scatter(X, Y)
title('Training set')

% Prior settings0
% prior over the parameters (since N = 1, Sigma_p is a scalar)
Sigma_p_vec = [0.000001,0.1,0.2,0.5,1,2,5,10,20,50, 100, 1000];

for Sigma_p = Sigma_p_vec
    
    %% Compute posterior distribution

    A = 1/(sigma_n^2) * X * X' + pinv(Sigma_p);
    invA = pinv(A);
    w_mean = 1/(sigma_n^2) * invA * X * Y;
    w_cov = invA;

    %% Plot resulting distribution and test predictions

    % test grid
    Xgrid = linspace(xMin, xMax, 100);
    Ygrid_pred_mean = w_mean * Xgrid;
    Ygrid_pred_cov = Xgrid' * w_cov * Xgrid;
    Ygrid_pred_std = sqrt(diag(Ygrid_pred_cov))';
    % Ygrid_pred_std = diag(Ygrid_pred_cov)';

    % figure
    % hold on
    % title('Predictive distribution')
    % % plot mean
    % plot(Xgrid, Ygrid_pred_mean)
    % % plot +2 std
    % plot(Xgrid, Ygrid_pred_mean + 2 * Ygrid_pred_std, '--b')
    % % plot -2 std
    % plot(Xgrid, Ygrid_pred_mean - 2 * Ygrid_pred_std, '--b')

    % test set predictions
    Yte_pred_mean = w_mean * Xte;
    Yte_pred_cov = Xte' * w_cov * Xte;
    Yte_pred_std = sqrt(diag(Yte_pred_cov))';

    figure
    hold on
    title('Test predictions and predictive distribution')
    % plot mean
    plot(Xgrid, Ygrid_pred_mean)
    % plot +2 std
    plot(Xgrid, Ygrid_pred_mean + 2 * Ygrid_pred_std, '--b')
    % plot -2 std
    plot(Xgrid, Ygrid_pred_mean - 2 * Ygrid_pred_std, '--b')
    % plot test points
    scatter(Xte, Yte, 'xr')
    % plot predictions
    scatter(Xte, Yte_pred_mean, 'or')
    % plot connecting lines
    M = [Yte'; Yte_pred_mean];
    plot([Xte;Xte], M, ':r')
end
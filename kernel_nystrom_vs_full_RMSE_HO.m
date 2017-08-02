addpath(genpath('./utils/'))
addpath('./dataset_scripts/')
clearAllButBP
close all

%% Make dataset

% Dataset settings
n = 2000;
nVal = 1000;
nTe = 300;
w = 1;
sigma_n = 0.2;
xMin = 0;
xMax = 2*pi;

[ X , Y ] = generate_sinusoidal_DS(n, w, sigma_n, xMin, xMax);
[ Xval , Yval ] = generate_sinusoidal_DS(nVal, w, sigma_n, xMin, xMax);
[ Xte , Yte ] = generate_sinusoidal_DS(nTe, w, sigma_n, xMin, xMax);

%% Model settings

% Prior settings
kernel = 'gaussian';
kerParVec = logspace(-3,1,30); % prior over the parameters (since N = 1, Sigma_p is a scalar)
sigma_f = 1;
M = 300;    % Nystrom subsampling level

t_HO_NGPR = [];
bestRMSE_HO_NGPR = Inf;
bestIter_HO_NGPR = [];
bestKerPar_HO_NGPR = [];
RMSE_vec_NGPR = [];
bestAlpha_HO_NGPR = [];
bestKnm_HO_NGPR = [];
bestKmm_HO_NGPR = [];

t_HO_FGPR = [];
bestRMSE_HO_FGPR = Inf;
bestIter_HO_FGPR = [];
bestKerPar_HO_FGPR = [];
RMSE_vec_FGPR = [];
bestAlpha_HO_FGPR = [];
bestKnm_HO_FGPR = [];
bestKmm_HO_FGPR = [];

% Grid search: Hold-out Cross Validation

for i = 1:numel(kerParVec)
    
    clc
    progressBar( i , numel(kerParVec) )
    
    kerPar = kerParVec(i);
    

    
    %% Full GPR HO RMSE
    
    tic
    
    K = KernelMatrix(X', X', kernel, kerPar, sigma_f);
    alpha = (K + sigma_n^2 * eye(n)) \ Y;
    
    % Compute validation error
    K_valTrain = KernelMatrix(Xval', X', kernel, kerPar, sigma_f);
%     K_valVal = KernelMatrix(Xval', Xval', kernel, kerPar, sigma_f);

    Yval_pred_mean = K_valTrain * alpha;

    RMSE = sqrt( sum( (Yval_pred_mean - Yval).^2 ) );
    t_HO_FGPR = [t_HO_FGPR; toc];
    
    RMSE_vec_FGPR = [RMSE_vec_FGPR ; RMSE];
    
    if bestRMSE_HO_FGPR > RMSE
        
        bestRMSE_HO_FGPR = RMSE;
        bestIter_HO_FGPR = i;
        bestKerPar_HO_FGPR = kerPar;   
        bestAlpha_HO_FGPR = alpha;
        bestK_HO_FGPR = K;
    end
    
    
    %% Nystrom GPR HO RMSE
    
    tic    
    
    % Randomly subsample training points and form subsampled Gram matrix

    Knm = KernelMatrix(X', X(:,1:M)', kernel, kerPar, sigma_f);
    Kmm = Knm(1:M,:);
    alpha = (Knm' * Knm + sigma_n^2 * Kmm) \ Knm' * Y;
    
    % Compute validation error
    K_valTrain = KernelMatrix(Xval', X(:,1:M)', kernel, kerPar, sigma_f);
    K_valVal = KernelMatrix(Xval', Xval', kernel, kerPar, sigma_f);

    Yval_pred_mean = K_valTrain * alpha;

    RMSE = sqrt( sum( (Yval_pred_mean - Yval).^2 ) );
    t_HO_NGPR = [t_HO_NGPR; toc];
    
    RMSE_vec_NGPR = [RMSE_vec_NGPR ; RMSE];
    
    if bestRMSE_HO_NGPR > RMSE
        
        bestRMSE_HO_NGPR = RMSE;
        bestIter_HO_NGPR = i;
        bestKerPar_HO_NGPR = kerPar;   
        bestAlpha_HO_NGPR = alpha;
        bestKnm_HO_NGPR = Knm;
        bestKmm_HO_NGPR = Kmm;
    end
end

t_HO_FGPR = cumsum(t_HO_FGPR);
t_HO_NGPR = cumsum(t_HO_NGPR);

%% Plot timings
figure
hold on
plot(t_HO_FGPR)
plot(t_HO_NGPR)
title('Grid search times (HO)')
xlabel('Parameter guess')
ylabel('Cumulative time')
legend('Full GPR','Nys GPR')
hold off

%% Plot validation RMSE for Nystrom GPR
figure
hold on
plot(RMSE_vec_NGPR)
title('Hold-out Validation Error - -Nystrom GPR')
xlabel('Parameter guess')
ylabel('RMSE')
hold off

%% Plot validation RMSE for full GPR
figure
hold on
plot(RMSE_vec_FGPR)
title('Hold-out Validation Error - Full GPR')
xlabel('Parameter guess')
ylabel('RMSE')
hold off

%% Compute resulting test distribution and point predictions

% test set predictions
K_testTrain = KernelMatrix(Xte', X(:,1:M)', kernel, bestKerPar_HO_NGPR, sigma_f);
K_testTest = KernelMatrix(Xte', Xte', kernel, bestKerPar_HO_NGPR, sigma_f);

Yte_pred_mean = K_testTrain * bestAlpha_HO_NGPR;
% Yte_pred_cov = K_testTest + sigma_n^2 * eye(size(K_testTest)) - K_testTrain *  ((K + sigma_n^2 * eye(n)) \ K_testTrain');
% Yte_pred_std = sqrt(diag(Yte_pred_cov))';

%% Nystrom GPR - Compute grid distribution and plot it with test predictions

% test grid
Xgrid = linspace(xMin, xMax, 200);
K_gridTrain = KernelMatrix(Xgrid', X(:,1:M)', kernel, bestKerPar_HO_NGPR, sigma_f);
K_gridGrid = KernelMatrix(Xgrid', Xgrid', kernel, bestKerPar_HO_NGPR, sigma_f);

Ygrid_pred_mean = K_gridTrain * bestAlpha_HO_NGPR;
Ygrid_pred_cov = sigma_n^2 * eye(size(K_gridGrid)) + sigma_n^2 * K_gridTrain *  ((bestKnm_HO_NGPR' * bestKnm_HO_NGPR + sigma_n^2 * bestKmm_HO_NGPR) \ K_gridTrain');
Ygrid_pred_std = sqrt(diag(Ygrid_pred_cov))';


figure
hold on
title('Nystrom GPR test predictions and predictive distribution')
% % plot mean
plot(Xgrid', Ygrid_pred_mean)
% % plot +2 std
plot(Xgrid', Ygrid_pred_mean + 2 * Ygrid_pred_std', '--b')
% % plot -2 std
plot(Xgrid', Ygrid_pred_mean - 2 * Ygrid_pred_std', '--b')
% plot test points
scatter(Xte, Yte, 'xr')
% plot predictions
scatter(Xte, Yte_pred_mean, 'or')
% % plot connecting lines
plot([Xte;Xte], [Yte'; Yte_pred_mean'], ':r')


%% Full GPR - Compute grid distribution and plot it with test predictions

% test grid
Xgrid = linspace(xMin, xMax, 200);
K_gridTrain = KernelMatrix(Xgrid', X', kernel, bestKerPar_HO_NGPR, sigma_f);
K_gridGrid = KernelMatrix(Xgrid', Xgrid', kernel, bestKerPar_HO_NGPR, sigma_f);

Ygrid_pred_mean = K_gridTrain * bestAlpha_HO_FGPR;
Ygrid_pred_cov = K_gridGrid + sigma_n^2 * eye(size(K_gridGrid)) - K_gridTrain *  ((K + sigma_n^2 * eye(n)) \ K_gridTrain');
Ygrid_pred_std = real(sqrt(diag(Ygrid_pred_cov)))';

figure
hold on
title('Full GPR test predictions and predictive distribution')
% % plot mean
plot(Xgrid', Ygrid_pred_mean)
% % plot +2 std
plot(Xgrid', Ygrid_pred_mean + 2 * Ygrid_pred_std', '--b')
% % plot -2 std
plot(Xgrid', Ygrid_pred_mean - 2 * Ygrid_pred_std', '--b')
% plot test points
scatter(Xte, Yte, 'xr')
% plot predictions
scatter(Xte, Yte_pred_mean, 'or')
% % plot connecting lines
plot([Xte;Xte], [Yte'; Yte_pred_mean'], ':r')


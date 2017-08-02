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
M = 300;    % Subsampling level

t_NLML = [];
bestNLML = Inf;
bestIter_NLML = [];
bestKerPar_NLML = [];
NLML_vec = [];

t_HO = [];
bestRMSE_HO = Inf;
bestIter_HO = [];
bestKerPar_HO = [];
RMSE_vec = [];
bestAlpha_HO = [];
bestKnm_HO = [];
bestKmm_HO = [];

% Grid search: Negative LML (NLML) vs Hold-out Cross Validation

for i = 1:numel(kerParVec)
    
    clc
    progressBar( i , numel(kerParVec) )
    
    kerPar = kerParVec(i);
    
    % Randomly subsample training points

    Knm = KernelMatrix(X', X(:,1:M)', kernel, kerPar, sigma_f);
    Kmm = Knm(1:M,:);
    alpha = (Knm' * Knm + sigma_n^2 * Kmm) \ Knm' * Y;

    %% Negative Log-Marginal Likelihood (NLML)
% 
%     tic
%     NLML = 0.5 * Y' * alpha + 0.5 * real(log(det(K + sigma_n^2 * eye(n)))) + 0.5  * n * log(2*pi);
%     t_NLML = [t_NLML; toc];
%     
%     NLML_vec = [NLML_vec ; NLML];
%     
%     if bestNLML > NLML
%         
%         bestNLML = NLML;
%         bestIter_NLML = i;
%         bestKerPar_NLML = kerPar;        
%     end
    
    %% HO RMSE
    
    tic
    % Compute validation error
    K_valTrain = KernelMatrix(Xval', X(:,1:M)', kernel, kerPar, sigma_f);
    K_valVal = KernelMatrix(Xval', Xval', kernel, kerPar, sigma_f);

    Yval_pred_mean = K_valTrain * alpha;

    RMSE = sqrt( sum( (Yval_pred_mean - Yval).^2 ) );
    t_HO = [t_HO; toc];
    
    RMSE_vec = [RMSE_vec ; RMSE];
    
    if bestRMSE_HO > RMSE
        
        bestRMSE_HO = RMSE;
        bestIter_HO = i;
        bestKerPar_HO = kerPar;   
        bestAlpha_HO = alpha;
        bestKnm_HO = Knm;
        bestKmm_HO = Kmm;
    end
end

t_NLML = cumsum(t_NLML);
t_HO = cumsum(t_HO);

%% Plot timings
figure
hold on
% plot(t_NLML)
plot(t_HO)
title('Grid search times (HO)')
xlabel('Parameter guess')
ylabel('Cumulative time')
% legend('LML','HO')
hold off

%% Plot validation RMSE
figure
hold on
plot(RMSE_vec)
title('Hold-out Validation Error')
xlabel('Parameter guess')
ylabel('RMSE')
hold off

% %% Plot NLML
% figure
% hold on
% plot(NLML_vec)
% title('Negative Log-Marginal Likelihood')
% xlabel('Parameter guess')
% ylabel('NLML')
% hold off


%% Compute resulting test distribution and point predictions

% test set predictions
K_testTrain = KernelMatrix(Xte', X(:,1:M)', kernel, bestKerPar_HO, sigma_f);
K_testTest = KernelMatrix(Xte', Xte', kernel, bestKerPar_HO, sigma_f);

Yte_pred_mean = K_testTrain * bestAlpha_HO;
% Yte_pred_cov = K_testTest + sigma_n^2 * eye(size(K_testTest)) - K_testTrain *  ((K + sigma_n^2 * eye(n)) \ K_testTrain');
% Yte_pred_std = sqrt(diag(Yte_pred_cov))';

%% Compute grid distribution and plot it with test predictions

% test grid
Xgrid = linspace(xMin, xMax, 200);
K_gridTrain = KernelMatrix(Xgrid', X(:,1:M)', kernel, bestKerPar_HO, sigma_f);
K_gridGrid = KernelMatrix(Xgrid', Xgrid', kernel, bestKerPar_HO, sigma_f);

Ygrid_pred_mean = K_gridTrain * bestAlpha_HO;
Ygrid_pred_cov = sigma_n^2 * eye(size(K_gridGrid)) + sigma_n^2 * K_gridTrain *  ((bestKnm_HO' * bestKnm_HO + sigma_n^2 * bestKmm_HO) \ K_gridTrain');
Ygrid_pred_std = sqrt(diag(Ygrid_pred_cov))';


figure
hold on
title('Test predictions and predictive distribution')
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

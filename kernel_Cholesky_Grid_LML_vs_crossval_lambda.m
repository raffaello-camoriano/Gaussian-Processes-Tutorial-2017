addpath(genpath('./utils/'))
addpath('./dataset_scripts/')
clearAllButBP
close all

%% Make dataset

% Dataset settings
n = 3000;
nVal = 1000;
nTe = 1000;
w = 1;
sigma_n_real = 0.2;
xMin = 0;
xMax = 2*pi;

[ X , Y ] = generate_sinusoidal_DS(n, w, sigma_n_real, xMin, xMax);
[ Xval , Yval ] = generate_sinusoidal_DS(nVal, w, sigma_n_real, xMin, xMax);
[ Xte , Yte ] = generate_sinusoidal_DS(nTe, w, sigma_n_real, xMin, xMax);

%% Compute Kernel and alpha coefficients

% Prior settings
kernel = 'gaussian';
kerPar = 2.65; % prior over the parameters (since N = 1, Sigma_p is a scalar)
sigma_f = 1;

sigma_n_vec = logspace(-2,0,20);

t_NLML = [];
bestNLML = Inf;
bestIter_NLML = [];
bestKerPar_NLML = [];
NLML_vec = [];

t_HO = [];
bestRMSE_HO = Inf;
bestIter_HO = [];
bestSigma_n_HO = [];
RMSE_vec = [];

% Grid search: Negative LML (NLML) vs Hold-out Cross Validation

for i = 1:numel(sigma_n_vec)
    
    clc
    progressBar( i , numel(sigma_n_vec) )
    
    sigma_n = sigma_n_vec(i);

    K = KernelMatrix(X', X', kernel, kerPar, sigma_f);
    L = chol(K + sigma_n^2 * eye(n),'lower');
    alpha = L'\(L \ Y);

    %% Negative Log-Marginal Likelihood (NLML)

    tic
    NLML = 0.5 * Y' * alpha ... % Data-fit term (only one depending on outputs Y)
        + sum(log(diag(L))) ... % Complexity penalty
        + 0.5  * n * log(2*pi); % Normalization constant
    t_NLML = [t_NLML; toc];
    
    NLML_vec = [NLML_vec ; NLML];
    
    if bestNLML > NLML
        
        bestNLML = NLML;
        bestIter_NLML = i;
        bestSigma_n_NLML = sigma_n;        
    end
    
    %% HO RMSE
    
    tic
    % Compute validation error
    K_valTrain = KernelMatrix(Xval', X', kernel, kerPar, sigma_f);

    Yval_pred_mean = K_valTrain * alpha;

    RMSE = sqrt( sum( (Yval_pred_mean - Yval).^2 ) );
    t_HO = [t_HO; toc];
    
    RMSE_vec = [RMSE_vec ; RMSE];
    
    if bestRMSE_HO > RMSE
        
        bestRMSE_HO = RMSE;
        bestIter_HO = i;
        bestSigma_n_HO = sigma_n;        
    end
end

t_NLML = cumsum(t_NLML);
t_HO = cumsum(t_HO);

%% Plot timings
figure
hold on
plot(t_NLML)
plot(t_HO)
title('Grid search times (NLML vs HO)')
xlabel('Parameter guess')
ylabel('Cumulative time')
legend('LML','HO')
set(gca,'YScale','log');
hold off

%% Plot validation RMSE
figure
hold on
plot(RMSE_vec)
plot([bestIter_HO, bestIter_HO],ylim, 'r--');
title('Hold-out Validation Error')
xlabel('Parameter guess')
ylabel('RMSE')
hold off

%% Plot NLML
figure
hold on
plot(NLML_vec)
title('Negative Log-Marginal Likelihood')
plot([bestIter_NLML, bestIter_NLML],ylim, 'r--');
xlabel('Parameter guess')
ylabel('NLML')
hold off

%% Compute resulting test distribution and point predictions

% test set predictions
K_testTrain = KernelMatrix(Xte', X', kernel, kerPar, sigma_f);
K_testTest = KernelMatrix(Xte', Xte', kernel, kerPar, sigma_f);

Yte_pred_mean = K_testTrain * alpha;
Yte_pred_cov = K_testTest + sigma_n^2 * eye(size(K_testTest)) - K_testTrain *  ((K + sigma_n^2 * eye(n)) \ K_testTrain');
Yte_pred_std = sqrt(diag(Yte_pred_cov))';


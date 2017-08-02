clearAllButBP
close all

%% Make dataset

% Dataset settings
n = 2000;
nVal = 1000;
nTe = 1000;
w = 1;
sigma_n = 0.2;
xMin = 0;
xMax = 2*pi;

[ X , Y ] = generate_sinusoidal_DS(n, w, sigma_n, xMin, xMax);
[ Xval , Yval ] = generate_sinusoidal_DS(nVal, w, sigma_n, xMin, xMax);
[ Xte , Yte ] = generate_sinusoidal_DS(nTe, w, sigma_n, xMin, xMax);


%% Compute Kernel and alpha coefficients

% Prior settings
kernel = 'gaussian';
kerParVec = 0.1:0.4:10; % prior over the parameters (since N = 1, Sigma_p is a scalar)
sigma_f = 1;

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

% Grid search: Negative LML (NLML) vs Hold-out Cross Validation

for i = 1:numel(kerParVec)
    
    clc
    progressBar( i , numel(kerParVec) )
    
    kerPar = kerParVec(i);

    K = KernelMatrix(X', X', kernel, kerPar, sigma_f);
    alpha = (K + sigma_n^2 * eye(n)) \ Y;

    %% Negative Log-Marginal Likelihood (NLML)

    tic
    NLML = 0.5 * Y' * alpha + 0.5 * real(log(det(K + sigma_n^2 * eye(n)))) + 0.5  * n * log(2*pi);
    t_NLML = [t_NLML; toc];
    
    NLML_vec = [NLML_vec ; NLML];
    
    if bestNLML > NLML
        
        bestNLML = NLML;
        bestIter_NLML = i;
        bestKerPar_NLML = kerPar;        
    end
    
    %% HO RMSE
    
    tic
    % Compute validation error
    K_valTrain = KernelMatrix(Xval', X', kernel, kerPar, sigma_f);
    K_valVal = KernelMatrix(Xval', Xval', kernel, kerPar, sigma_f);

    Yval_pred_mean = K_valTrain * alpha;

    RMSE = sqrt( sum( (Yval_pred_mean - Yval).^2 ) );
    t_HO = [t_HO; toc];
    
    RMSE_vec = [RMSE_vec ; RMSE];
    
    if bestRMSE_HO > RMSE
        
        bestRMSE_HO = RMSE;
        bestIter_HO = i;
        bestKerPar_HO = kerPar;        
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
hold off

%% Plot validation RMSE
figure
hold on
plot(RMSE_vec)
title('Hold-out Validation Error')
xlabel('Parameter guess')
ylabel('RMSE')
hold off

%% Plot NLML
figure
hold on
plot(NLML_vec)
title('Negative Log-Marginal Likelihood')
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

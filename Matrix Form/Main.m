%% MAMP 
% ==================================================================================
% L. Liu, S. Huang and B. M. Kurkoski, "Memory AMP," 
% in IEEE Transactions on Information Theory, 2022, doi: 10.1109/TIT.2022.3186166.
% ==================================================================================
% Author: Shunqi Huang
% ==================================================================================
%% Parameter Initialization
clc; clear; 
%close all;
rng('shuffle')

iter_O = 15;                        
iter_M = 35;                        
sim_times = 100;
kappa = 50;                         
N = 2048;                           
delta = 0.5;                         
M = round(delta * N);                
L = 3;                               
SNR_dB = 30;                        
% Prior distribution
% Supported type: 'BPSK', 'QPSK', '16QAM', 'BG', 'BCG', 'RD'
% 'BG': Bernoulli-Gaussian (required fileds: 'p_1', 'u_g', 'v_g')
% 'BCG': Bernoulli-Complex Gaussian (required fileds: 'p_1', 'u_g', 'v_g')
% x = b * g, b ~ Bern(p_1), g ~ N(u_g, v_g) or CN(u_g, v_g)
% ------------------------------------------------------------------------
% 'RD': Real discrete distribution (required fileds: 'X', 'P')
% X = [x_1, ..., x_n], Pr(x = x_i) = p_i, P = [p_1, ..., p_n]
% ------------------------------------------------------------------------
p_1 = 0.1;
u_g = 0;                    
v_g = 1 / p_1;    
E_x = 0;
v_x = 1;
info = struct('type', 'BCG', 'mean', E_x, 'var', v_x);
info.p_1 = p_1;                 % only for BG or BCG
info.u_g = u_g;                 % only for BG or BCG
info.v_g = v_g;                 % only for BG or BCG
% noise variance
v_n = v_x ./ (10.^(0.1.*SNR_dB));   
% singular values of A
T = min(M, N);
dia = kappa.^(-(0:T-1)' / T);
dia = sqrt(N) * dia / norm(dia);
% MSE
MSE_O = zeros(1, iter_O);
MSE_M = zeros(1, iter_M);
% generate A
A = randn(M, N) + randn(M, N)*1i;
[U, ~, V] = svd(A);
if M < N
    S = [diag(dia), zeros(M, N-M)];
else
    S = [diag(dia); zeros(M-N, N)];
end
A = U * S * V';

%% Simulations
for r = 1 : sim_times
    r
    % source
    b = binornd(1, p_1, N, 1);
    g_re = normrnd(u_g , sqrt(v_g), [N, 1]);
    g_im = normrnd(u_g , sqrt(v_g), [N, 1]);
    g = (g_re + g_im * 1i) / sqrt(2);
    x = b .* g;                          
    % noise
    n_re = normrnd(0, sqrt(v_n), [M, 1]); 
    n_im = normrnd(0, sqrt(v_n), [M, 1]);
    n = (n_re + n_im*1i) / sqrt(2);         
    % y
    y = A * x + n;
    % OAMP(SVD)
    [MSE, ~] = OAMP_SVD(A, V, x, y, dia, v_n, iter_O, info);
    MSE_O = MSE_O + MSE;
    % MAMP
    [MSE, ~] = MAMP(A, x, y, dia, v_n, L, iter_M, info);
    MSE_M = MSE_M + MSE;
end
MSE_O = MSE_O / sim_times;
MSE_M = MSE_M / sim_times;

%% plot figures
plot_len = max([iter_O, iter_M]);
% OAMP
semilogy(0:plot_len, [v_x MSE_O MSE_O(end)*ones(1,plot_len-iter_O)], 'b-', 'LineWidth', 1.5);
hold on;
% MAMP
semilogy(0:plot_len, [v_x MSE_M MSE_M(end)*ones(1,plot_len-iter_M)], 'r-', 'LineWidth', 1.5);
hold on;
title(['\kappa=', num2str(kappa), ';M=', num2str(M), ';N=', num2str(N), ';SNR=', num2str(SNR_dB)]);
legend('OAMP/VAMP', 'MAMP');
xlabel('Number of iterations', 'FontSize', 11);
ylabel('MSE', 'FontSize', 11);
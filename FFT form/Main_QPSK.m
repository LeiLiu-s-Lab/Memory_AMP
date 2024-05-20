% ========================================================================
% Model: y = Ax + n, n~N(0, v_n)
% x is IID with x_i ~ Px(x), {A, y, v_n, Px(x)} are known.
% Here, A = D * P * F, D is a diagonal matrix
% P is a random permutation matrix, F is the DFT matrix
% ========================================================================
% In this form, OAMP is also low-complexity. 
% This code is only for comparing the MSE performace of OAMP and MAMP with less time.

%% Parameter Initialization
clc; clear; 
%close all;
rng('shuffle')

N = 8192;
delta = 1;
M = round(delta * N);
% ========================================================================
% Demodulator Information
% Supported type: 'BPSK', 'QPSK', '16QAM', 'BG', 'BCG', 'RD'
% 'BG': Bernoulli-Gaussian (required fileds: 'p_1', 'u_g', 'v_g')
% 'BCG': Bernoulli-Complex Gaussian (required fileds: 'p_1', 'u_g', 'v_g')
% x = b * g, b ~ Bern(p_1), g ~ N(u_g, v_g) or CN(u_g, v_g)
% ------------------------------------------------------------------------
% 'RD': Real discrete distribution (required fileds: 'X', 'P')
% X = [x_1, ..., x_n], Pr(x = x_i) = p_i, P = [p_1, ..., p_n]
% ------------------------------------------------------------------------
% ========================================================================
% "E_x" and "v_x" are the mean and variance of P_x, respectively.
E_x = 0;                                
v_x = 1;
info = struct('type', 'QPSK', 'mean', E_x, 'var', v_x);
SNR_dB = 11;                        
kappa = 10;
iter = 10;
iter_M = 16;
v_n = v_x ./ (10.^(0.1.*SNR_dB));
L = 3;
% 
T = min(M, N);
dia = kappa.^(-(0:T-1)' / T);
dia = sqrt(N) * dia / norm(dia);
%
sim_times = 100;
MSE_O = zeros(1, iter);
MSE_M = zeros(1, iter_M);

%% Simulations
for r = 1 : sim_times
    disp(r)
    % QPSK
    x_re = 2 * randi([0, 1], [N, 1]) - 1;
    x_im = 2 * randi([0, 1], [N, 1]) - 1;
    x = (x_re + x_im * 1i) / sqrt(2);                  
    % noise
    n_re = normrnd(0, sqrt(v_n), [M, 1]); 
    n_im = normrnd(0, sqrt(v_n), [M, 1]);
    n = (n_re + n_im * 1i) / sqrt(2);         
    % y = D*P*F*x + n;
    % D is a diagonal matrix, dia = diag(D)
    % P is a random permutation matrix, F is the DFT matrix
    index_ev = randperm(N);
    index_ev = index_ev(1:T);
    index_ev = index_ev';
    x_f = fft(x) / sqrt(N);
    y = [dia .* x_f(index_ev); zeros(M-N, 1)] + n;
    % OAMP
    [MSE, ~] = OAMP(x, y, dia, index_ev, v_n, iter, info);
    MSE_O = MSE_O + MSE;
    % MAMP
    [MSE, ~, ~] = MAMP(index_ev, x, y, dia, v_n, L, iter_M, info);
    MSE_M = MSE_M + MSE;
end
MSE_O = MSE_O / sim_times;
MSE_M = MSE_M / sim_times;

%% plot figures
semilogy(0:iter, [v_x MSE_O], 'b-', 'LineWidth', 1.5);
hold on
semilogy(0:iter_M, [v_x MSE_M], 'r-', 'LineWidth', 1.5);
title(['\kappa=', num2str(kappa), ';M=', num2str(M), ';N=', num2str(N), ';SNR(dB)=', num2str(SNR_dB)]);
legend('OAMP', 'MAMP');
xlabel('Number of iterations', 'FontSize', 11);
ylabel('MSE', 'FontSize', 11);
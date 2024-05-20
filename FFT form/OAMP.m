%% OAMP
% y = DPFx + n;
% D is a diagonal matrix, dia = diag(D)
% P is a random permutation matrix, F is the DFT matrix
function [MSE, Var] = OAMP(x, y, dia, index_ev, v_n, it, info)
    MSE = zeros(1, it);
    Var = zeros(1, it);
    M = length(y);
    N = length(x);
    u_nle = zeros(N, 1);                    
    v_nle = 1;                              
    thres_0 = 1e-6;

    % iterations
    for t = 1 : it
        % LE
        [u_le_p, v_le_p] = LE_OAMP(u_nle, v_nle, dia, index_ev, y, v_n, M, N);
        [u_le, v_le] = Orth(u_le_p, v_le_p, u_nle, v_nle);
        % NLE
        [u_nle_p, v_nle_p] = Demodulator(u_le, v_le, info);
        MSE(t) = (u_nle_p - x)' * (u_nle_p - x) / N;                
        Var(t) = v_nle_p;
        if MSE(t) <= thres_0
            MSE(t:end) = thres_0;
            Var(t:end) = thres_0;
            break
        end
        [u_nle, v_nle] = Orth(u_nle_p, v_nle_p, u_le, v_le);
    end
end

%% Orthogonalization
function [u_orth, v_orth] = Orth(u_post, v_post, u_pri, v_pri)
    v_orth = 1 / (1 / v_post - 1 / v_pri);
    u_orth = v_orth * (u_post / v_post - u_pri / v_pri);  
end

%% LE for OAMP
function [u_post, v_post] = LE_OAMP(u, v, dia, index_ev, y, v_n, M, N)
    rho = v_n / v;
    Dia = [abs(dia).^2; zeros(M-N, 1)];         % abs(dia).^2 as dia may be complex-valued
    D = 1 ./ (Dia + rho);                           
    tmp = y - A_times_x(u, index_ev, dia, M, N);
    tmp2 = D .* tmp;
    tmp3 = AH_times_x(tmp2, index_ev, dia, M, N);
    u_post = u + tmp3;
    v_post = v - v * sum(Dia.*D) / N;
end

%% Ax
function Ax = A_times_x(x, index_ev, dia, M, N)
    x_f = fft(x) / sqrt(N);
    Ax = [dia .* x_f(index_ev); zeros(M-N, 1)];
end

%% AHx
function AHx = AH_times_x(x, index_ev, dia, M, N)
    tmp = zeros(N, 1);
    T = min(M, N);
    tmp(index_ev) = conj(dia) .* x(1:T);        % conj(dia) as dia may be complex-valued
    AHx = ifft(tmp*sqrt(N));
end

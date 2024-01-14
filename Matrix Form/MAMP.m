%% MAMP (overflow-avoiding)
% ====================================================================
% Input: (1) dia: singular values of A, (2) v_n: noise variance
% (3) L: damping length, (4) info: prior information of x
% Output: (1) MSE/Var: MSE performance via t iterations
% (2) x_post: soft decision of x
% !! when dia is unknown, please use the estimated MAMP version !!  
% ====================================================================
function [MSE, Var, x_post] = MAMP(A, x, y, dia, v_n, L, it, info)
    M = length(y);
    N = length(x);
    delta = M / N;
    lam = [dia.^2; zeros(M-N, 1)];                  % eigenvalue of AAH
    lam_s = 0.5 * (max(lam) + min(lam));   
    lam_B = lam_s * ones(M, 1) - lam;               % eigenvalue of B
    sgn_B = zeros(M, 1);
    sgn_B(lam_B > 0) = 1;
    sgn_B(lam_B < 0) = -1;
    w_0 = (lam_s * M - sum(lam_B)) / N;          
    w_1 = (lam_s * sum(lam_B) - sum(lam_B.^2)) / N; 
    wb_00 = lam_s * w_0 - w_1 - w_0 * w_0;
    x_phi = info.mean .* ones(N, 1);
    v_phi = zeros(it, it);
    x_phi(:, 1) = zeros(N, 1);                      % E(x) = 0
    log_vth = zeros(1, it);                         % log(|vartheta_{t,i}|)
    sgn_xi = zeros(1, it);                          % sgn(xi)               
    z = zeros(M, it);
    z(:, 1) = y - A * x_phi(:, 1);
    v_phi(1, 1) = real(z(:, 1)' * z(:, 1) / N - delta * v_n) / w_0;
    chi = zeros(1, 2*it-1);
    theta_1 = 1 / (lam_s + v_n / v_phi(1, 1));
    for k = 1 : 2*it-1
        chi(k) = sum((lam_s-lam_B) .* (sgn_B).^k .* exp(k*log(abs(lam_B)*theta_1))) / N;
    end
    log_th1 = log(theta_1);
    z_hat = zeros(M, 1);
    r_hat = zeros(N, 1);
    MSE = zeros(1, it);
    Var = zeros(1, it);
    res_list = zeros(N, it);
    thres_0 = 1e-6;
    index = [];
    
    % iterations
    for t = 1 : it
        % MLE
        [log_vth, sgn_xi, z_hat, r_hat, r, v_gam] = MLE_MAMP(A, x_phi, v_phi, chi, sgn_xi, log_vth, ...
            y, z_hat, r_hat, w_0, wb_00, lam_s, v_n, log_th1, t);
        v_gam = real(v_gam);
        % NLE
        [x_post, v_post] = Denoiser(r, v_gam, info);
        res_list(:, t) = x_post;
        MSE(t) = (x_post - x)' * (x_post - x) / N;
        Var(t) = v_post;
        if Var(t) <= thres_0
            tmp = (x_post - x)' * (x_post - x) / N;
            MSE(t:end) = max(tmp, thres_0);
            Var(t:end) = thres_0;
            break
        elseif t > 2
            % only for robustness, not necessary 
            tt = Is_stop(Var, thres_0, t);      
            if tt > 0
                MSE(tt+1:end) = MSE(tt);
                Var(tt+1:end) = Var(tt);
                x_post = res_list(:, tt);
                break;
            end
        end
        x_phi(:, t+1) = (x_post / v_post - r / v_gam) / (1 / v_post - 1 / v_gam);
        z(:, t+1) = y - A * x_phi(:, t+1);
        v_phi(t+1, t+1) = (z(:, t+1)' * z(:, t+1) / N - delta * v_n) / w_0;
        for k = 1 : t
            v_phi(t+1, k) = (z(:, t+1)' * z(:, k) / N - delta * v_n) / w_0;
            v_phi(k, t+1) = v_phi(t+1, k)';
        end
        if v_phi(t+1, t+1) < 0
            index = [index, t+1];
            v_phi(t+1, t+1) = 1 / (1 / v_post - 1 / v_gam);
        else
            % damping
            [x_phi, z, index, v_phi] = Damping(x_phi, v_phi, z, index, L, t+1);
        end
    end
end

%% Stop
function tt = Is_stop(Var, thres_0, t)
    tt = 0;
    thres = thres_0 / 10;
    if Var(t-1) > Var(t) 
        if Var(t-1) - Var(t) < thres
            tt = t;
        end
    else
        if Var(t-2) - Var(t) < thres
            if Var(t-2) < Var(t-1)
                tt = t - 2;
            else
                tt = t - 1;
            end
        end
    end
end

%% Damping
% if damping is at NLE, set tt = t + 1
function [X, Z, index, V] = Damping(X, V, Z, index, L, tt)
    % find out damping index
    l = min(L, tt);
    d = 0;
    dam_ind = [];
    for k = flip(1:tt)
        if ~ismember(k, index)
            d = d + 1;
            dam_ind = [k, dam_ind];
            if d == l
                break
            end
        end
    end
    l = length(dam_ind);
    % delete rows and columns
    del_ind = setdiff(1:tt, dam_ind);
    V_da = V(1:tt, 1:tt);
    V_da(del_ind, :) = [];
    V_da(:, del_ind) = [];
    % obtain zeta
    if rcond(V_da) < 1e-15
        zeta = [zeros(1, l-2), 1, 0];
        index = [index, tt];
        V(tt, tt) = V(tt-1, tt-1);
    else
        o = ones(l, 1);
        tmp = V_da \ o;
        v_s = real(o' * tmp);
        zeta = tmp / v_s;
        zeta = zeta.';
        V(tt, tt) = 1 / v_s;
    end
    % update
    X(:, tt) = sum(zeta.*X(:, dam_ind), 2);
    Z(:, tt) = sum(zeta.*Z(:, dam_ind), 2);
    for k = 1 : tt-1
        V(k, tt) = sum(zeta.*V(k, dam_ind));
        V(tt, k) = V(k, tt)';
    end
end
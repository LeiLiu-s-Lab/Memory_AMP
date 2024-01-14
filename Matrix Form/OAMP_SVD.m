%% OAMP (SVD)
function [MSE, Var, x_p] = OAMP_SVD(A, V, x, y, dia, v_n, it, info)
    MSE = zeros(1, it);
    Var = zeros(1, it);
    M = length(y);
    N = length(x);
    u_nle = info.mean .* ones(N, 1);
    v_nle = info.var;
    AHy = A' * y;
    thres_0 = 1e-6;

    % iterations
    for t = 1 : it
        % LE
        [u_le_p, v_le_p] = LE_OAMP(V, AHy, u_nle, v_nle, dia, v_n, M, N);
        [u_le, v_le] = Orth(u_le_p, v_le_p, u_nle, v_nle);
        % NLE
        [u_nle_p, v_nle_p] = Denoiser(u_le, v_le, info);
        MSE(t) = (u_nle_p - x)' * (u_nle_p - x) / N;
        Var(t) = v_nle_p;
        if Var(t) <= thres_0
            x_p = u_nle_p;
            tmp = (x_p - x)' * (x_p - x) / N;
            MSE(t:end) = max(tmp, thres_0);
            Var(t:end) = thres_0;
            break
        end
        x_p = u_nle_p;
        [u_nle, v_nle] = Orth(u_nle_p, v_nle_p, u_le, v_le);
    end
end

%% Orthogonalization
function [u_orth, v_orth] = Orth(u_post, v_post, u_pri, v_pri)
    v_orth = 1 / (1 / v_post - 1 / v_pri);
    u_orth = v_orth * (u_post / v_post - u_pri / v_pri);  
end

%% LE
function [u_post, v_post] = LE_OAMP(V, AHy, u, v, dia, v_n, M, N)
    rho = v_n / v;
    Dia = [dia.^2; zeros(N-M, 1)];
    D = 1 ./ (rho + Dia);
    u_post = V * (D .* (V' * (AHy + rho * u)));
    v_post = v_n / N * sum(D);
end
%% Demodulator for prior distributions of x
% ------------------------------------------------------------------
% Model: r = x + n, n ~ N(0, v) or CN(0, v);
% r is an N*1 vector, v is a positive number;
% info is a struct that has a filed 'type';
% (*) if type is 'BG' or 'BCG', info needs fileds 'p_1', 'u_g', 'v_g';
% (*) if type is 'RD', info has fileds 'X', 'P'.
% ------------------------------------------------------------------
function [u_p, v_p] = Demodulator(r, v, info)
    if ~isreal(v)
        warning('The variance v is not real.')
    elseif v < 0
        warning('The variance v is not positive.')
    end
    type = info.type;
    if strcmpi(type, 'BPSK')
        if ~isreal(r)
            r = real(r);
            v = v / 2;
        end
        [u_p, v_p] = Denoiser_BPSK(r, v);
    elseif strcmpi(type, 'QPSK')
        [u_1, v_1] = Denoiser_BPSK(sqrt(2)*real(r), v);    % real part
        [u_2, v_2] = Denoiser_BPSK(sqrt(2)*imag(r), v);    % imaginary part
        u_p = (u_1 + u_2 * 1i) / sqrt(2);
        v_p = (v_1 + v_2) / 2;
    elseif strcmpi(type, '16QAM')
        X = [-3, -1, 1, 3] / sqrt(10);
        P = [0.25, 0.25, 0.25, 0.25];
        [u_1, v_1] = Denoiser_RD(real(r), v/2, X, P);    % real part
        [u_2, v_2] = Denoiser_RD(imag(r), v/2, X, P);    % imaginary part
        u_p = u_1 + u_2 * 1i;
        v_p = v_1 + v_2;
    elseif strcmpi(type, 'BG')
        if ~isreal(r)
            r = real(r);
            v = v / 2;
        end
        p_1 = info.p_1;
        u_g = info.u_g;
        v_g = info.v_g;
        [u_p, v_p] = Denoiser_BG(r, v, p_1, u_g, v_g);
    elseif strcmpi(type, 'BCG')
        p_1 = info.p_1;
        u_g = info.u_g;
        v_g = info.v_g;
        [u_p, v_p] = Denoiser_BCG(r, v, p_1, u_g, v_g);
    elseif strcmpi(type, 'RD')
        if ~isreal(r)
            r = real(r);
            v = v / 2;
        end
        X = info.X;
        P = info.P;
        [u_p, v_p] = Denoiser_RD(r, v, X, P);
    else
        error('This type of prior distribution is not supported currently.')
    end
end

%% BPSK
function [u_p, v_p] = Denoiser_BPSK(r, v)
    EXP_B = 50;
    d = -2 * r / v;
    d(d > EXP_B) = EXP_B;
    d(d < -EXP_B) = -EXP_B;
    p_1 = 1 ./ (1 + exp(d));
    u_p = 2 .* p_1 - 1;
    v_p = mean(1 - u_p.^2);
end

%% Real discrete distribution
% X = [x_1, ..., x_n], Pr(x = x_i) = p_i, P = [p_1, ..., p_n]
function [u_p, v_p] = Denoiser_RD(r, v, X, P)
    N = length(r);
    n = length(X);
    EXP_B = 50;
    u_p = zeros(N, 1);
    v_p = zeros(N, 1);
    for ii = 1 : N
        p_p = zeros(1, n);
        for jj = 1 : n
            d = (X(jj)^2 - X.^2 + 2 * r(ii) * (X-X(jj))) / (2*v);
            d(d > EXP_B) = EXP_B;
            d(d < -EXP_B) = -EXP_B;
            s = sum(P.*exp(d));
            p_p(jj) = P(jj) / s;
        end
        u_p(ii) = sum(p_p.*X);
        v_p(ii) = sum(p_p.*X.^2) - u_p(ii)^2;
    end
    v_p = mean(v_p);
end

%% Complex discrete distribution
% X = [x_1, ..., x_n], Pr(x = x_i) = p_i, P = [p_1, ..., p_n]
function [u_p, v_p] = Denoiser_CD(r, v, X, P)
    N = length(r);
    n = length(X);
    EXP_B = 50;
    u_p = zeros(N, 1);
    v_p = zeros(N, 1);
    for ii = 1 : N
        p_p = zeros(1, n);
        for jj = 1 : n
            d = (abs(r-X(jj))^2 - abs(r-X).^2) / v;
            d(d > EXP_B) = EXP_B;
            d(d < -EXP_B) = -EXP_B;
            p_p(jj) = P(jj) / sum(P.*exp(d));
        end
        u_p(ii) = sum(p_p.*X);
        v_p(ii) = sum(p_p.*abs(X).^2) - u_p(ii)^2;
    end
    v_p = mean(v_p);
end

%% Bernoulli-Gaussian
function [u_p, v_p] = Denoiser_BG(r, v, P, u_g, v_g)
    N = length(r);
    EXP_B = 50;
    u_g = u_g * ones(N, 1);
    % post Bernoull
    c = sqrt((v + v_g) / v);
    d = 0.5 * ((r - u_g).^2 / (v + v_g) - (r.^2) / v);
    d(d > EXP_B) = EXP_B;
    d(d < -EXP_B) = -EXP_B;
    p1 = P ./ (P + (1-P) * c * exp(d));
    % post Gaussian
    v_pg = 1 / (1 / v + 1 / v_g);
    u_pg = v_pg * (r / v + u_g / v_g);
    % post u and v
    u_p = p1 .* u_pg;
    v_p = (p1 - p1.^2) .* (u_pg.^2) + p1 * v_pg;
    v_p = mean(v_p);
end

%% Bernoulli-Complex Gaussian
function [u_p, v_p] = Denoiser_BCG(r, v, P, u_g, v_g)
    N = length(r);
    EXP_B = 50;
    u_g = u_g * ones(N, 1);
    % post Bernoull
    c = (v + v_g) / v;
    d = abs(r - u_g).^2 / (v + v_g) - abs(r).^2 / v;
    d(d > EXP_B) = EXP_B;
    d(d < -EXP_B) = -EXP_B;
    p1 = P ./ (P + (1-P) * c * exp(d));
    % post Gaussian
    v_pg = 1 / (1 / v + 1 / v_g);
    u_pg = v_pg * (r / v + u_g / v_g);
    % post u and v
    u_p = p1 .* u_pg;
    v_p = (p1 - p1.^2) .* (abs(u_pg).^2) + p1 * v_pg;
    v_p = mean(v_p);
end
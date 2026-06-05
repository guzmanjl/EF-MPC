function [Su, Pu, g] = DMC_pred(G, Np, Nc, N_past)
%==========================================================================
% DMC step response matrices for a SISO discrete-time model
%
%   y(k) = G(z^-1) u(k)
%
% The step response g is computed by:
%   1. Extracting the delay-free numerator and denominator (trimming any
%      leading zeros that c2d may have embedded to represent the delay).
%   2. Reading the delay d from G.IODelay.
%   3. Computing the step response via filter() on the delay-free model.
%   4. Shifting by d samples (prepend d zeros).
%
% This mirrors the GPC_pred.m approach and avoids the double-delay issue
% that occurs when calling step(G,...) directly on a TF that has both
% IODelay set and leading zeros in the numerator from c2d.
%
% Prediction:
%   y(k+j|k) = Su(j,:) * DeltaU_future + Pu(j,:) * dup_past + eps
%
% Inputs:
%   G       : discrete-time TF (IODelay must carry the true delay)
%   Np      : prediction horizon
%   Nc      : control horizon
%   N_past  : past Delta u values for free response (N_sr = Np + N_past)
%
% Outputs:
%   Su      : forced response [Np x Nc], lower-triangular Toeplitz
%             Su(i,j) = g(i-j+1) for i>=j, else 0
%   Pu      : free response  [Np x N_past]
%             Pu(i,c) = g(i+c): contribution of Delta u(k-c) to y(k+i|k)
%   g       : step response vector [N_sr x 1]
%==========================================================================

    if G.Ts <= 0
        error('DMC_pred: G must be a discrete-time transfer function.');
    end

    Ts   = G.Ts;
    N_sr = Np + N_past;

    %% Extract delay and delay-free numerator/denominator
    % (mirrors local_get_tf_data in GPC_pred.m)
    num = G.Numerator{1,1};   num = num(:).';
    den = G.Denominator{1,1}; den = den(:).';

    % Trim leading zeros from numerator (handles delay embedded by c2d)
    idx = find(abs(num) > 1e-14, 1, 'first');
    if isempty(idx), num = 0; else, num = num(idx:end); end

    % Normalize to monic denominator
    if abs(den(1)) < 1e-14
        error('DMC_pred: zero leading denominator coefficient.');
    end
    num = num / den(1);
    den = den / den(1);

    % Delay from IODelay property (not from numerator zeros)
    d = round(G.IODelay);

    % Trim trailing zeros from numerator
    idx = find(abs(num) > 1e-14, 1, 'last');
    if isempty(idx), num = 0; else, num = num(1:idx); end

    %% Step response of the delay-free system via filter
    % filter(num, den, ones(N,1)) gives y(k) = g(k) directly for k=1..N.
    % For a ZOH-discretized strictly proper system the numerator coefficient
    % is non-zero, so y(1) = num(1) = g(1) ≠ 0.  No offset needed.
    y    = filter(num, den, ones(N_sr, 1));
    g_nd = y(:);   % [g(1), g(2), ..., g(N_sr)]

    %% Include delay by shifting (prepend d zeros)
    g = zeros(N_sr, 1);
    if d < N_sr
        g(d+1:end) = g_nd(1:N_sr-d);
    end

    %% Remove numerical noise
    g(abs(g) < 1e-12) = 0;

    %% Forced response matrix Su [Np x Nc]
    % Su(i,j) = g(i-j+1) for i >= j, else 0 (lower-triangular Toeplitz)
    Su = zeros(Np, Nc);
    for i = 1:Np
        for j = 1:min(i, Nc)
            Su(i,j) = g(i - j + 1);
        end
    end

    %% Free response matrix Pu [Np x N_past]
    % Pu(i,c) = g(i+c): contribution of Delta u(k-c) to y(k+i|k)
    % Settled value g(N_sr) used when i+c exceeds the computed window
    Pu = zeros(Np, N_past);
    for i = 1:Np
        for c = 1:N_past
            idx = i + c;
            if idx <= N_sr
                Pu(i,c) = g(idx);
            else
                Pu(i,c) = g(N_sr);
            end
        end
    end

end

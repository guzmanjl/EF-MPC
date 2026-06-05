function [A_sim,Bu_sim,Bd_sim] = Model_sim(G1,G2)
%==========================================================================
% Generic GPC prediction matrices for a TISO model:
%
%   y(k) = G1(z^-1) u(k) + G2(z^-1) d(k)
%
% where
%   G1(z^-1) = num1(z^-1) / den1(z^-1)
%   G2(z^-1) = num2(z^-1) / den2(z^-1)
%
% Inputs:
%   G1  -> discrete-time tf of manipulated input channel
%   G2  -> discrete-time tf of measurable disturbance channel
%
%   A_sim  -> common denominator for plant simulation: den1*den2
%   Bu_sim -> equivalent numerator for input simulation
%   Bd_sim -> equivalent numerator for disturbance simulation
%
% Notes:
%   1) This function assumes G1 and G2 are already in discrete time.
%   2) It supports arbitrary-order proper models.
%   3) Delays are included by padding zeros in the numerators.
%==========================================================================

 %% Check discrete-time models
    if G1.Ts <= 0 || G2.Ts <= 0
        error('GPC_pred_TISO: G1 and G2 must be discrete-time transfer functions.');
    end

    if abs(G1.Ts - G2.Ts) > 1e-12
        error('GPC_pred_TISO: G1 and G2 must have the same sampling time.');
    end

    %% Extract numerator, denominator and delays
    [num1, den1, d1] = local_get_tf_data(G1);
    [num2, den2, d2] = local_get_tf_data(G2);

    %% Common plant model for simulation
    % y = (num1/den1)u + (num2/den2)d
    % => A_sim y = Bu_sim u + Bd_sim d
    A_sim  = conv(den1, den2);
    Bu_sim = conv(num1, den2);
    Bd_sim = conv(num2, den1);

    % Include explicit delays for simulation
    Bu_sim = [zeros(1,d1), Bu_sim];
    Bd_sim = [zeros(1,d2), Bd_sim];
    
end


%==========================================================================
% Local function: extract normalized discrete tf coefficients and delay
%==========================================================================
function [num, den, delay] = local_get_tf_data(G)

    num = G.Numerator{1,1};
    den = G.Denominator{1,1};

    % Remove leading zeros if any
    num = local_trim_leading_zeros(num);
    den = local_trim_leading_zeros(den);

    % Normalize so that den(1) = 1
    if abs(den(1)) < 1e-14
        error('local_get_tf_data: denominator leading coefficient is zero.');
    end

    num = num / den(1);
    den = den / den(1);

    % Delay
    %delay = round(G.OutputDelay);
    delay = round(G.IODelay);

    % Remove trailing zeros in numerator
    num = local_trim_trailing_zeros(num);

end


%==========================================================================
% Utility: remove leading zeros
%==========================================================================
function x = local_trim_leading_zeros(x)
    idx = find(abs(x) > 1e-14, 1, 'first');
    if isempty(idx)
        x = 0;
    else
        x = x(idx:end);
    end
end


%==========================================================================
% Utility: remove trailing zeros
%==========================================================================
function x = local_trim_trailing_zeros(x)
    idx = find(abs(x) > 1e-14, 1, 'last');
    if isempty(idx)
        x = 0;
    else
        x = x(1:idx);
    end
end
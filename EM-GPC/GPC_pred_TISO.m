function [H1,P1,H2,P2,Qy] = GPC_pred_TISO(G1,G2,Np,Nc)
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
%   Np  -> prediction horizon
%   Nc  -> control horizon
%
% Outputs:
%   H1     -> forced response matrix for future Delta u
%   P1     -> free response matrix for past Delta u
%   H2     -> forced response matrix for future Delta d
%   P2     -> free response matrix for past Delta d
%   Qy     -> free response matrix for past outputs
%
%   A_pred -> common augmented denominator for predictor: den1*den2*Delta
%   Bu_pred-> equivalent numerator for input predictor
%   Bd_pred-> equivalent numerator for disturbance predictor
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

    %% Common CARIMA predictor model
    % A_pred = A_sim * Delta
    A_pred  = conv(A_sim, [1 -1]);
    Bu_pred = Bu_sim;
    Bd_pred = Bd_sim;

    %% Build prediction matrices
    [H1,P1,Qy] = local_build_gpc_matrices(A_pred,Bu_pred,Np,Nc,d1);
    [H2,P2,~ ] = local_build_gpc_matrices(A_pred,Bd_pred,Np,Nc,d2);

    %% Remove tiny numerical noise
    tol = 1e-10;

    H1(abs(H1) < tol) = 0;
    P1(abs(P1) < tol) = 0;
    H2(abs(H2) < tol) = 0;
    P2(abs(P2) < tol) = 0;
    Qy(abs(Qy) < tol) = 0;

    A_pred(abs(A_pred) < tol) = 0;
    Bu_pred(abs(Bu_pred) < tol) = 0;
    Bd_pred(abs(Bd_pred) < tol) = 0;

    A_sim(abs(A_sim) < tol) = 0;
    Bu_sim(abs(Bu_sim) < tol) = 0;
    Bd_sim(abs(Bd_sim) < tol) = 0;

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
% Local function: build GPC matrices from A and B
%==========================================================================
function [H,P,Q] = local_build_gpc_matrices(A,B,Np,Nc,delay)
% A already includes Delta = [1 -1]
%
% Predictor form:
%   A(z^-1) y(k) = B(z^-1) Delta u(k-1)
%
% H -> future forced response due to future increments
% P -> free response due to past input increments
% Q -> free response due to past outputs

    nA = length(A);
    nB = length(B);

    %% Build Ca
    if Np >= nA
        Ca = [A, zeros(1,Np-nA)].';
    else
        Ca = A(1:Np).';
    end

    for i = 1:Np-1
        Cacol = [zeros(i,1); Ca(1:Np-i,1)];
        Ca = [Ca, Cacol];
    end

    %% Build Ha
    Ha = [];
    if Np > nA
        for i = 1:nA-1
            Hacol = [A(i+1:end), zeros(1, Np - length(A(i+1:end)))].';
            Ha = [Ha, Hacol];
        end
    else
        for i = 1:Np
            Halin = [A(i+1:end), zeros(1,i-1)];
            Ha = [Ha; Halin];
        end
    end

    %% Build Cb
    if Np > nB
        Cb = [B, zeros(1,Np-nB)].';
    else
        Cb = B(1:Np).';
    end

    for i = 1:Nc-1
        Cbcol = [zeros(i,1); Cb(1:Np-i,1)];
        Cb = [Cb, Cbcol];
    end

    %% Build Hb
    Hb = [];
    if Np > nB
        if nB == 1
            Hb = zeros(Np,1);
        else
            for i = 1:nB-1
                Hbcol = [B(i+1:end), zeros(1, Np - length(B(i+1:end)))].';
                Hb = [Hb, Hbcol];
            end
        end
    else
        for i = 1:Np
            Hblin = [B(i+1:end), zeros(1,i-1)];
            Hb = [Hb; Hblin];
        end
    end

    %% Solve predictor matrices
    H = Ca \ Cb;
    P = Ca \ Hb;
    Q = Ca \ Ha;

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
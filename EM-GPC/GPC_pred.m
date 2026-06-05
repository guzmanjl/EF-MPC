function [H,P,Q,A,B] = GPC_pred(G,Np,Nc)
%==========================================================================
% Generic GPC prediction matrices for a discrete-time SISO model
%
%   y(k) = G(z^-1) u(k)
%
% where
%   G(z^-1) = B(z^-1) / A0(z^-1)
%
% The CARIMA predictor is built using:
%   A(z^-1) = A0(z^-1) * Delta(z^-1)
%   Delta(z^-1) = 1 - z^-1
%
% Inputs:
%   G   -> discrete-time transfer function
%   Np  -> prediction horizon
%   Nc  -> control horizon
%
% Outputs:
%   H   -> forced response matrix for future Delta u
%   P   -> free response matrix for past Delta u
%   Q   -> free response matrix for past outputs
%   A   -> augmented denominator polynomial A0*Delta
%   B   -> numerator polynomial including delay
%
% Notes:
%   1) G must already be a discrete-time transfer function.
%   2) Delay is included by padding zeros in the numerator.
%   3) This function supports arbitrary-order proper models.
%==========================================================================

    %% Check model
    if G.Ts <= 0
        error('GPC_pred: G must be a discrete-time transfer function.');
    end

    %% Extract numerator, denominator and delay
    [num, den, d] = local_get_tf_data(G);

    %% Build CARIMA augmented denominator
    A = conv(den, [1 -1]);

    %% Include delay in numerator
    B = [zeros(1,d), num];

    %% Build prediction matrices
    [H,P,Q] = local_build_gpc_matrices(A,B,Np,Nc);

    %% Remove numerical noise
    tol = 1e-10;
    H(abs(H) < tol) = 0;
    P(abs(P) < tol) = 0;
    Q(abs(Q) < tol) = 0;

end


%==========================================================================
% Local function: extract normalized discrete tf coefficients and delay
%==========================================================================
function [num, den, delay] = local_get_tf_data(G)

    num = G.Numerator{1,1};
    den = G.Denominator{1,1};

    % Remove leading zeros
    num = local_trim_leading_zeros(num);
    den = local_trim_leading_zeros(den);

    % Normalize so den(1)=1
    if abs(den(1)) < 1e-14
        error('local_get_tf_data: denominator leading coefficient is zero.');
    end

    num = num / den(1);
    den = den / den(1);

    % Delay stored separately in tf object
    delay = round(G.OutputDelay);

    % Remove trailing zeros in numerator
    num = local_trim_trailing_zeros(num);

end


%==========================================================================
% Local function: build GPC matrices from A and B
%==========================================================================
function [H,P,Q] = local_build_gpc_matrices(A,B,Np,Nc)
% A already includes Delta = [1 -1]
%
% Predictor form:
%   A(z^-1) y(k) = B(z^-1) Delta u(k-1)

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

    %% Keep only Np rows
    H = H(1:Np,:);
    P = P(1:Np,:);
    Q = Q(1:Np,:);

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
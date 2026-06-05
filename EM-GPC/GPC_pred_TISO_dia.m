function [H1,P1,H2,P2,Qy,A_sim,Bu_sim,Bd_sim] = GPC_pred_TISO_dia(G1,G2,Np,Nc)
%==========================================================================
% GPC prediction matrices for a TISO system using the Diophantine method
%
%   y(k) = G1(z^-1) u(k) + G2(z^-1) d(k)
%
% with:
%   G1(z^-1) = num1(z^-1) / den1(z^-1)
%   G2(z^-1) = num2(z^-1) / den2(z^-1)
%
% The predictor is built in CARIMA form:
%
%   A(z^-1) y(k) = Bu(z^-1) Delta u(k-1) + Bd(z^-1) Delta d(k-1)
%
% where:
%   A(z^-1) = A_sim(z^-1) * Delta(z^-1)
%   Delta(z^-1) = 1 - z^-1
%
% Inputs:
%   G1  -> discrete-time TF of manipulated-input channel
%   G2  -> discrete-time TF of measurable-disturbance channel
%   Np  -> prediction horizon
%   Nc  -> control horizon
%
% Outputs:
%   H1     -> forced response matrix for future Delta u
%   P1     -> free response matrix for past Delta u
%   H2     -> forced response matrix for future Delta d
%   P2     -> free response matrix for past Delta d
%   Qy     -> free response matrix for past outputs
%             use in prediction as:  f = -Qy*yp + P1*dup_p + P2*ddp_p
%
%   A_sim  -> common denominator for plant simulation
%   Bu_sim -> common numerator for input channel simulation
%   Bd_sim -> common numerator for disturbance channel simulation
%
% Notes:
%   1) G1 and G2 must already be discrete-time transfer functions
%   2) Different orders and delays are supported
%   3) Delays can be stored either as explicit leading zeros in numerator
%      and/or as TF delay properties
%==========================================================================

    %% Check models
    if G1.Ts <= 0 || G2.Ts <= 0
        error('GPC_pred_TISO: G1 and G2 must be discrete-time transfer functions.');
    end

    if abs(G1.Ts - G2.Ts) > 1e-12
        error('GPC_pred_TISO: G1 and G2 must have the same sample time.');
    end

    %% Extract normalized numerator/denominator and total delay
    [num1, den1, d1] = local_get_tf_data(G1);
    [num2, den2, d2] = local_get_tf_data(G2);

    %% Common plant model for simulation
    % y = (num1/den1)u + (num2/den2)d
    % => A_sim y = Bu_sim u + Bd_sim d
    A_sim  = conv(den1, den2);
    Bu_sim = conv(num1, den2);
    Bd_sim = conv(num2, den1);

    % Include total delays explicitly in the numerators
    Bu_sim = [zeros(1,d1), Bu_sim];
    Bd_sim = [zeros(1,d2), Bd_sim];

    %% CARIMA predictor denominator
    % A_pred y = Bu_sim Delta u(k-1) + Bd_sim Delta d(k-1)
    A_pred = conv(A_sim, [1 -1]);

    %% Build GPC matrices via Diophantine method
    [H1,P1,Qy] = local_build_diophantine_predictor(A_pred, Bu_sim, Np, Nc);
    [H2,P2,~ ] = local_build_diophantine_predictor(A_pred, Bd_sim, Np, Nc);

    %% Clean numerical noise
    tol = 1e-10;

    H1(abs(H1) < tol) = 0;
    P1(abs(P1) < tol) = 0;
    H2(abs(H2) < tol) = 0;
    P2(abs(P2) < tol) = 0;
    Qy(abs(Qy) < tol) = 0;

    A_sim(abs(A_sim) < tol) = 0;
    Bu_sim(abs(Bu_sim) < tol) = 0;
    Bd_sim(abs(Bd_sim) < tol) = 0;

end


%==========================================================================
% Extract discrete TF data robustly:
% - normalizes denominator to monic form
% - handles delay stored as numerator leading zeros and/or tf properties
%==========================================================================
function [num, den, delay] = local_get_tf_data(G)

    tol = 1e-14;

    num = G.Numerator{1,1};
    den = G.Denominator{1,1};

    num = num(:).';
    den = den(:).';

    % Trim denominator leading zeros
    den = local_trim_leading_zeros(den);

    if isempty(den) || abs(den(1)) < tol
        error('local_get_tf_data: invalid denominator.');
    end

    % Normalize denominator to den(1)=1
    num = num / den(1);
    den = den / den(1);

    % Delay embedded as leading zeros in numerator
    idx = find(abs(num) > tol, 1, 'first');
    if isempty(idx)
        explicit_delay = 0;
        num = 0;
    else
        explicit_delay = idx - 1;
        num = num(idx:end);
    end

    % Remove trailing zeros in numerator
    num = local_trim_trailing_zeros(num);

    % Delay stored in TF properties
    stored_delay = 0;

    if isprop(G,'IODelay')
        try
            stored_delay = stored_delay + round(double(G.IODelay));
        catch
        end
    end

    if isprop(G,'InputDelay')
        try
            stored_delay = stored_delay + round(double(G.InputDelay));
        catch
        end
    end

    if isprop(G,'OutputDelay')
        try
            stored_delay = stored_delay + round(double(G.OutputDelay));
        catch
        end
    end

    delay = explicit_delay*0 + stored_delay;

    % Force row vectors
    num = num(:).';
    den = den(:).';

end


%==========================================================================
% Build predictor matrices using the Diophantine equation
%
% For each prediction step j:
%   1 = E_j(z^-1) A(z^-1) + z^(-j) F_j(z^-1)
%
% Then:
%   y(k+j|k) = sum future terms + sum past-input terms + F_j * yp
%
% To preserve the user's convention:
%   f = -Qy*yp + P*dup_p
%
% we return:
%   Qy = -Fmat
% so that:
%   -Qy*yp = Fmat*yp
%==========================================================================
function [H,P,Qy] = local_build_diophantine_predictor(A,B,Np,Nc)

    A = A(:).';
    B = B(:).';

    nA = length(A);
    nB = length(B);

    % Number of past outputs and past input increments
    ny_past = nA - 1;
    nu_past = nB - 1;

    H   = zeros(Np,Nc);
    P   = zeros(Np,nu_past);
    Fm  = zeros(Np,ny_past);   % F matrix, later Qy = -Fm

    %--------------------------------------------------------------
    % For each prediction step j, solve:
    %   1 = E_j A + z^(-j) F_j
    %--------------------------------------------------------------
    for j = 1:Np

        % Solve for E_j = e0 + e1 z^-1 + ... + e_{j-1} z^{-(j-1)}
        E = local_solve_diophantine_E(A,j);

        % Tail of E*A gives F_j
        EA = conv(E,A);   % length j+nA-1

        % Since 1 - E*A = z^{-j} F_j, and first j coefficients of EA
        % are [1 0 ... 0], then:
        %   F_j = -(EA(j+1:end))
        Fj = -EA(j+1:end);

        % Pad Fj to fixed length nA-1
        Fj_pad = zeros(1,ny_past);
        Fj_pad(1:min(length(Fj),ny_past)) = Fj(1:min(length(Fj),ny_past));

        % Convolution with input/disturbance numerator
        C = conv(E,B);    % coefficients multiplying Delta u / Delta d

        %----------------------------------------------------------
        % Build row j of forced matrix H
        %
        % y(k+j|k) includes:
        %   C(1) * Delta u(k+j-1)
        %   C(2) * Delta u(k+j-2)
        %   ...
        %
        % For decision vector:
        %   [Delta u(k), Delta u(k+1), ..., Delta u(k+Nc-1)]'
        %
        % row j, col c gets coefficient:
        %   C(j-c+1), provided j-c+1 >= 1
        %----------------------------------------------------------
        for c = 1:Nc
            idx = j - c + 1;
            if idx >= 1 && idx <= length(C)
                H(j,c) = C(idx);
            end
        end

        %----------------------------------------------------------
        % Build row j of past-input free-response matrix P
        %
        % Past increment vector is assumed:
        %   dup_p = [Delta u(k-1); Delta u(k-2); ...]'
        %
        % Thus row j uses:
        %   C(j+1), C(j+2), ..., C(j+nB-1)
        %----------------------------------------------------------
        for r = 1:nu_past
            idx = j + r;
            if idx <= length(C)
                P(j,r) = C(idx);
            end
        end

        % Past-output contribution
        Fm(j,:) = Fj_pad;
    end

    % Return Qy with user's convention:
    % free response = -Qy*yp + P*dup_p
    Qy = -Fm;

end


%==========================================================================
% Solve the Diophantine equation for E_j:
%
%   1 = E_j(z^-1) A(z^-1) + z^(-j) F_j(z^-1)
%
% This means the first j coefficients of conv(E_j,A) must be:
%   [1 0 0 ... 0]
%
% E_j has degree j-1.
%==========================================================================
function E = local_solve_diophantine_E(A,j)

    A = A(:).';
    nA = length(A);

    % Lower-triangular Toeplitz matrix built from A
    T = zeros(j,j);
    for row = 1:j
        for col = 1:row
            idx = row - col + 1;
            if idx <= nA
                T(row,col) = A(idx);
            end
        end
    end

    rhs = zeros(j,1);
    rhs(1) = 1;

    E = (T \ rhs).';
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
function [Phi, Theta_u, Theta_d] = MPC_pred_TISO(model_ss,parameter)
%==========================================================================
% Build prediction matrices for a TISO/MIMO augmented state-space model
%
% Prediction model:
%   Y = Phi*x(k) + Theta_u*DeltaU + Theta_d*DeltaD
%
% where:
%   x(k)      : augmented state at current time
%   DeltaU    : future manipulated-input increments over control horizon Nc
%   DeltaD    : future disturbance increments over control horizon Nc
%   Y         : stacked predicted outputs over prediction horizon Np
%
% Required fields in model_ss:
%   Atil   : augmented state matrix
%   Butil  : augmented input matrix for Delta u
%   Ctil   : augmented output matrix
%   Bdtil  : augmented input matrix for Delta d
%   Np     : prediction horizon
%   Nc     : control horizon
%
% Outputs:
%   Phi     : free-response matrix multiplying x(k)
%   Theta_u : forced-response matrix for future DeltaU
%   Theta_d : forced-response matrix for future DeltaD
%==========================================================================

    %% Extract matrices
    A  = model_ss.Atil;
    Bu = model_ss.Butil;
    C  = model_ss.Ctil;
    Bd = model_ss.Bdtil;

    Np = parameter.Np;
    Nc = parameter.Nc;

    %% Dimensions
    nx = size(A,1);
    nu = size(Bu,2);
    nd = size(Bd,2);
    ny = size(C,1);

    %% Preallocate
    Phi      = zeros(ny*Np, nx);
    Theta_u  = zeros(ny*Np, nu*Nc);
    Theta_d  = zeros(ny*Np, nd*Nc);

    %% Build Phi
    % Phi = [ C A
    %         C A^2
    %         ...
    %         C A^Np ]
    for i = 1:Np
        rows = (i-1)*ny + (1:ny);
        Phi(rows,:) = C*(A^i);
    end

    %% Build Theta_u and Theta_d
    % For predicted output y(k+i|k), the contribution of Deltau(k+j-1)
    % is:
    %   C * A^(i-j) * Bu,   if i >= j
    % and zero otherwise.
    %
    % Same for disturbance increments with Bd.
    for i = 1:Np
        rows = (i-1)*ny + (1:ny);

        for j = 1:Nc
            if i >= j
                cols_u = (j-1)*nu + (1:nu);
                cols_d = (j-1)*nd + (1:nd);

                Theta_u(rows, cols_u) = C*(A^(i-j))*Bu;
                Theta_d(rows, cols_d) = C*(A^(i-j))*Bd;
            end
        end
    end

end
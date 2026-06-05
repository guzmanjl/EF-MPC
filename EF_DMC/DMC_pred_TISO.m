function [Su_u, Pu_u, Su_d, Pu_d, g_u, g_d] = DMC_pred_TISO(G1, G2, Np, Nc, N_past)
%==========================================================================
% DMC step response matrices for a TISO model:
%
%   y(k) = G1(z^-1) u(k) + G2(z^-1) d(k)
%
% The prediction is built via superposition of two independent step
% response models — one per channel.  For linear systems this is exact:
%
%   y(k+j|k) = Su_u(j,:)*DeltaU_future   (forced, input)
%             + Su_d(j,:)*DeltaD_future   (forced, disturbance preview)
%             + Pu_u(j,:)*dup_past        (free, past Delta u)
%             + Pu_d(j,:)*ddp_past        (free, past Delta d)
%             + eps                       (output correction)
%
% This is equivalent to GPC_pred_TISO's combined-denominator approach:
%   Su_u  <=>  H1 = (num1*den2)/(den1*den2*Delta) = G1/(1-z^-1)
%   Su_d  <=>  H2 = (num2*den1)/(den1*den2*Delta) = G2/(1-z^-1)
% The cross-denominator terms cancel, leaving independent step responses.
%
% Inputs:
%   G1     : discrete-time TF of manipulated input channel
%   G2     : discrete-time TF of measurable disturbance channel
%   Np     : prediction horizon
%   Nc     : control horizon
%   N_past : number of past Delta u / Delta d values for free response
%
% Outputs:
%   Su_u   : forced response matrix for future Delta u [Np x Nc]
%   Pu_u   : free response matrix for past Delta u     [Np x N_past]
%   Su_d   : forced response matrix for future Delta d [Np x Nc]
%   Pu_d   : free response matrix for past Delta d     [Np x N_past]
%   g_u    : step response of G1 [N_sr x 1], N_sr = Np + N_past
%   g_d    : step response of G2 [N_sr x 1]
%==========================================================================

    if G1.Ts <= 0 || G2.Ts <= 0
        error('DMC_pred_TISO: G1 and G2 must be discrete-time transfer functions.');
    end

    if abs(G1.Ts - G2.Ts) > 1e-12
        error('DMC_pred_TISO: G1 and G2 must have the same sampling time.');
    end

    [Su_u, Pu_u, g_u] = DMC_pred(G1, Np, Nc, N_past);
    [Su_d, Pu_d, g_d] = DMC_pred(G2, Np, Nc, N_past);

end

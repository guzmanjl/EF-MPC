function [Delta_u, Delta_u_fb, Delta_u_ff, y_fb, y_ff, y_tot, flag] = ...
    DMC_Controller_EF(DMC_ctrl, f_fb, f_ff, uk_1, yref)
%==========================================================================
% EF-DMC: Online evaluation of the precompiled YALMIP optimizer
%
% Inputs:
%   DMC_ctrl  : struct from DMC_buildOptimizer_EF
%   f_fb      : FB free response vector   [Np x 1]  (pre-computed in main)
%   f_ff      : FF free response vector   [Np x 1]  (includes H_ff*v_f)
%   uk_1      : previous absolute u(k-1)  [nu x 1]
%   yref      : future reference          [Np x 1]
%
% Outputs:
%   Delta_u    : total control increments  [Nc x 1]
%   Delta_u_fb : FB increments             [Nc x 1]
%   Delta_u_ff : FF increments             [Nc x 1]
%   y_fb       : FB predicted output       [Np x 1]
%   y_ff       : FF predicted output       [Np x 1]
%   y_tot      : total predicted output    [Np x 1]
%   flag       : 1 if solved correctly, 0 otherwise
%==========================================================================

    flag = 0;

    try
        [sol, diag] = DMC_ctrl.opt{f_fb, f_ff, uk_1, yref};

        if diag == 0
            Delta_u    = sol{1}; Delta_u(abs(Delta_u) < 1e-8)       = 0;
            Delta_u_fb = sol{2}; Delta_u_fb(abs(Delta_u_fb) < 1e-8) = 0;
            Delta_u_ff = sol{3}; Delta_u_ff(abs(Delta_u_ff) < 1e-8) = 0;
            y_fb       = sol{4};
            y_ff       = sol{5};
            y_tot      = y_fb + y_ff;
            flag       = 1;
        else
            Delta_u    = zeros(DMC_ctrl.nu * DMC_ctrl.Nc, 1);
            Delta_u_fb = zeros(DMC_ctrl.nu * DMC_ctrl.Nc, 1);
            Delta_u_ff = zeros(DMC_ctrl.nu * DMC_ctrl.Nc, 1);
            y_fb       = zeros(DMC_ctrl.ny * DMC_ctrl.Np, 1);
            y_ff       = zeros(DMC_ctrl.ny * DMC_ctrl.Np, 1);
            y_tot      = zeros(DMC_ctrl.ny * DMC_ctrl.Np, 1);
        end

    catch
        Delta_u    = zeros(DMC_ctrl.nu * DMC_ctrl.Nc, 1);
        Delta_u_fb = zeros(DMC_ctrl.nu * DMC_ctrl.Nc, 1);
        Delta_u_ff = zeros(DMC_ctrl.nu * DMC_ctrl.Nc, 1);
        y_fb       = zeros(DMC_ctrl.ny * DMC_ctrl.Np, 1);
        y_ff       = zeros(DMC_ctrl.ny * DMC_ctrl.Np, 1);
        y_tot      = zeros(DMC_ctrl.ny * DMC_ctrl.Np, 1);
    end
end

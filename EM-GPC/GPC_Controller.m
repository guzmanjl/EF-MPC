function [du, du_fb, du_ff, y_fb_pred, y_ff_pred, y_tot_pred, flag] = ...
    GPC_Controller(GPC_ctrl, yp_fb, du_fb_p, yp_ff, du_ff_p, dd_ff_p, ddf_future, uk_1, W_future, parameter)

%==========================================================================
% Online evaluation of the precompiled YALMIP optimizer
%
% Inputs:
%   yp_fb       : past outputs for FB predictor
%   dup_fb_p    : past FB control increments
%   yp_ff       : past outputs for FF predictor
%   dup_ff_p    : past FF control increments
%   ddp_ff_p    : past disturbance increments
%   ddf_future  : future disturbance increments preview
%   uk_1        : previous absolute input
%   W_future    : future reference over horizon
%
% Outputs:
%   du          : total future control increment sequence
%   du_fb       : FB increment sequence
%   du_ff       : FF increment sequence
%   y_fb_pred   : FB predicted output
%   y_ff_pred   : FF predicted output
%   y_tot_pred  : total predicted output
%   flag        : 1 if optimizer solved correctly, 0 otherwise
%==========================================================================

% ------- Condition for 1st order and 1 Np
if isempty(du_fb_p)
    du_fb_p = 0;
end

%% Solve optimization problem 
    flag = 0;
    info.problem = -999;
    info.message = 'Unknown error';
    
    try
        [sol, diag] = GPC_ctrl.opt{yp_fb, du_fb_p, yp_ff, du_ff_p, dd_ff_p, ddf_future, uk_1, W_future};
        
        info.problem = diag;
        info.message = yalmiperror(diag);
        
        if info.problem == 0
            du    = sol{1}; du(abs(du) < 1e-8) = 0;
            du_fb = sol{2}; du_fb(abs(du_fb) < 1e-8) = 0;
            du_ff = sol{3}; du_ff(abs(du_ff) < 1e-8) = 0;
            y_fb_pred  = sol{4};
            y_ff_pred  = sol{5};
            y_tot_pred = sol{6};
            flag = 1;
        else
            du         = zeros(GPC_ctrl.nu*GPC_ctrl.Nc,1);
            du_fb      = zeros(GPC_ctrl.nu*GPC_ctrl.Nc,1);
            du_ff      = zeros(GPC_ctrl.nu*GPC_ctrl.Nc,1);
            y_fb_pred  = zeros(GPC_ctrl.ny*GPC_ctrl.Np,1);
            y_ff_pred  = zeros(GPC_ctrl.ny*GPC_ctrl.Np,1);
            y_tot_pred = zeros(GPC_ctrl.ny*GPC_ctrl.Np,1);
        end
        
    catch ME
        info.problem = -1;
        info.message = ME.message
        flag = 0;
    end
end
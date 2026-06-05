function [lambda_opt, info_opt] = tuneLambdaFBbyArea(parameter, A, Bu, Bd)
%==========================================================================
% Tune parameter.lambda so that the closed-loop REFERENCE TRACKING AREA
% of the FEEDBACK channel is as close as possible to the desired FOPDT area
% associated with tau_bc.
%
% This function assumes that the FB prediction matrices are already stored in
% parameter:
%   parameter.Gu_fb
%   parameter.Pup_fb
%   parameter.Qyp_fb
%
% Only the feedback channel is used for tuning:
%   - no disturbance
%   - no feedforward action
%
% The cost minimized is:
%   J(lambda) = (A_obt(lambda) - A_des)^2
%
% OUTPUTS:
%   lambda_opt : optimal lambda
%   info_opt   : structure with additional results
%==========================================================================
% Search lambda on a logarithmic scale

%% Desired discrete FOPDT area (consistent with evaluation)

tau_bc   = parameter.tau_bc;     % in samples
delay_u  = parameter.delay_u;    % in samples
Ts       = parameter.Ts;

Ntail = round(5*tau_bc);         % tail horizon (5 time constants)

% Use unit step (scales linearly with Delta_ref)
A_des = computeDiscreteFOPDTArea(1, tau_bc, delay_u, Ts, Ntail);

%% Optimize lambda

log10_lambda_min = -4;
log10_lambda_max =  4;

cost_fun = @(x) local_cost_FB(10.^x, parameter, A, Bu, Bd, A_des);

opts = optimset('TolX',1e-2, 'Display','iter');

[x_opt, J_opt] = fminbnd(cost_fun, log10_lambda_min, log10_lambda_max, opts);

lambda_opt = 10.^x_opt;

[J_check, info_opt] = local_cost_FB(lambda_opt, parameter, A, Bu, Bd, A_des);

info_opt.J_opt   = J_opt;
info_opt.J_check = J_check;
end


%==========================================================================
function [J, info] = local_cost_FB(tuning_val, parameter_base, A, Bu, Bd, A_des)
% Simulate the FEEDBACK-ONLY closed loop for a given lambda and compute the
% tracking area error with respect to the desired FOPDT area.

    % Copy parameter and update lambda
    parameter = parameter_base;
    parameter.lambda = tuning_val(1);

    % If FF matrices are missing, provide harmless zero-size-compatible terms
    if ~isfield(parameter,'Gu_ff') || isempty(parameter.Gu_ff)
        parameter.Gu_ff  = zeros(parameter.ny*parameter.Np, parameter.nu*parameter.Nc);
    end
    if ~isfield(parameter,'Pup_ff') || isempty(parameter.Pup_ff)
        parameter.Pup_ff = zeros(parameter.ny*parameter.Np, 0);
    end
    if ~isfield(parameter,'Gd_ff') || isempty(parameter.Gd_ff)
        parameter.Gd_ff  = zeros(parameter.ny*parameter.Np, parameter.nd*parameter.Nc);
    end
    if ~isfield(parameter,'Pdp_ff') || isempty(parameter.Pdp_ff)
        parameter.Pdp_ff = zeros(parameter.ny*parameter.Np, 0);
    end
    if ~isfield(parameter,'Qyp_ff') || isempty(parameter.Qyp_ff)
        parameter.Qyp_ff = zeros(parameter.ny*parameter.Np, 1);
    end

    % Build controller
    GPC_ctrl = GPC_buildOptimizer(parameter);
    
    %--------------------------------------------------------------
    % Simulation settings for tuning
    %--------------------------------------------------------------
    k0      = 20;   % reference step instant
    ref_amp = 1;    % unit step for linear tuning

    TN = max(150, k0 + round(parameter.delay_u + 6*parameter.tau_bc + parameter.Np + 20));

    ref = zeros(TN + parameter.Np + 2,1);
    ref(k0:end) = ref_amp;

    %--------------------------------------------------------------
    % Preallocation
    %--------------------------------------------------------------
    y           = zeros(TN+1,1);
    u_opt       = zeros(TN+1,1);
    du_total    = zeros(TN+1,1);
    du_fb       = zeros(TN+1,1);
    du_ff       = zeros(TN+1,1);
    flag        = zeros(TN+1,1);

    yfb_pred    = zeros(parameter.Np, TN+1);

    na  = length(A);
    nbu = length(Bu);

    y_hist = zeros(na-1,1);
    u_hist = zeros(nbu,1);

    uk_1 = 0;

    % Predictor histories
    yp_fb    = zeros(size(parameter.Qyp_fb,2),1);
    du_fb_p  = zeros(size(parameter.Pup_fb,2),1);

    % FF histories kept zero
    yp_ff    = zeros(max(size(parameter.Qyp_ff,2),1),1);
    du_ff_p  = zeros(size(parameter.Pup_ff,2),1);
    dd_ff_p  = zeros(size(parameter.Pdp_ff,2),1);

    %--------------------------------------------------------------
    % Closed-loop simulation
    %--------------------------------------------------------------
    for k = 1:TN

        % Update measured output histories
        yp_fb = [y(k); yp_fb(1:end-1)];

        if ~isempty(yp_ff)
            yp_ff = [0; yp_ff(1:end-1)];
            yp_ff(1) = y(k);
        end

        % Future reference
        ref_future = ref(k+1:k+parameter.Np);

        % No disturbance preview
        ddf_future = zeros(parameter.Nc,1);

        % Controller call
        [du_k, dufb_k, duff_k, yfbk_pred, ~, ~, flag(k+1)] = ...
            GPC_Controller( ...
                GPC_ctrl, ...
                yp_fb, du_fb_p, ...
                yp_ff, du_ff_p, dd_ff_p, ddf_future, ...
                uk_1, ref_future, ...
                parameter);

        % Store increments
        du_total(k+1) = du_k(1);
        du_fb(k+1)    = dufb_k(1);
        du_ff(k+1)    = duff_k(1);

        % Store prediction
        yfb_pred(:,k+1) = yfbk_pred;

        % Update histories
        if ~isempty(du_fb_p)
            du_fb_p = [du_fb(k+1); du_fb_p(1:end-1)];
        end
        if ~isempty(du_ff_p)
            du_ff_p = [0; du_ff_p(1:end-1)];
        end
        if ~isempty(dd_ff_p)
            dd_ff_p = [0; dd_ff_p(1:end-1)];
        end

        % Apply first increment
        u_opt(k+1) = uk_1 + du_fb(k+1);
        uk_1       = u_opt(k+1);

        % Plant simulation
        u_hist = [u_opt(k+1); u_hist(1:end-1)];
        y_hist = [y(k); y_hist(1:end-1)];

        y(k+1) = -A(2:end)*y_hist + Bu*u_hist;
    end

    %--------------------------------------------------------------
    % Tracking area evaluation
    % Include anticipation due to preview
    %--------------------------------------------------------------
    kstart = 1;
    kf     = TN;

    e_abs = abs(ref(kstart:kf) - y(kstart:kf));
    A_obt = parameter.Ts * sum(e_abs);

    % Cost
    J = (A_obt - A_des)^2;

    % Penalize failed optimization
    if any(flag(2:end) < 0)
        J = J + 1e8;
    end

    % Return info
    info.lambda = tuning_val;
    info.A_obt  = A_obt;
    info.A_des  = A_des;
    info.ratio  = A_obt / max(A_des,1e-12);
    info.k0     = k0;
    info.kstart = kstart;
    info.kf     = kf;
    info.y      = y;
    info.ref    = ref;
    info.u_opt  = u_opt;
    info.du_fb  = du_fb;
    info.flag   = flag;
end

function A_des = computeDiscreteFOPDTArea(Delta_ref, tau_bc, delay_u, Ts, Ntail)

    e = zeros(delay_u + Ntail + 1,1);

    for k = 0:length(e)-1
        if k < delay_u
            e(k+1) = abs(Delta_ref);
        else
            e(k+1) = abs(Delta_ref) * exp(-(k-delay_u+1)/tau_bc);
        end
    end

    A_des = Ts * sum(e);
end
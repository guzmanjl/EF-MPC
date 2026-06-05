function GPC_ctrl = GPC_buildOptimizer(parameter)
%==========================================================================
% Build YALMIP optimizer for GPC with feedback/feedforward decomposition
%
% Decision variables:
%   u_fb : future feedback increments
%   u_ff : future feedforward increments
%   u    : total future increments = u_fb + u_ff
%   y_fb : predicted output due to feedback channel
%   y_ff : predicted output due to feedforward/disturbance channel
%
% Objective:
%   J = (y_fb - W)'*R*(y_fb - W) + u_fb'*Q*u_fb + y_ff'*y_ff
%
% Constraints:
%   y_fb = Gu_fb*u_fb + F_fb
%   y_ff = Gu_ff*u_ff + F_ff
%   u    = u_fb + u_ff
%   umin <= cumulative(u) + u(k-1) <= umax
%
% Notes:
%   - The total future control applied to the plant is u = u_fb + u_ff
%   - Output constraints can be added either on y_fb or on y_fb+y_ff
%     depending on the formulation you want.
%==========================================================================

    %% -----------------------------
    % Parameters
    % ------------------------------
    nu        = parameter.nu;
    nd        = parameter.nd;
    ny        = parameter.ny;
    Np        = parameter.Np;
    Nc        = parameter.Nc;
    
    delay_u   = parameter.delay_u;
    delay_d   = parameter.delay_d;

    lambda    = parameter.lambda(:);
    gamma_fb  = parameter.gamma_fb(:);
    gamma_ff  = parameter.gamma_ff(:);

    umax      = parameter.umax(:);
    umin      = parameter.umin(:);
    dumax     = parameter.dumax(:);
    dumin     = parameter.dumin(:);
    ymax      = parameter.ymax(:);
    ymin      = parameter.ymin(:);

    Gu_fb     = parameter.Gu_fb;
    Pup_fb    = parameter.Pup_fb;
    Qyp_fb    = parameter.Qyp_fb;

    Gu_ff     = parameter.Gu_ff;
    Pup_ff    = parameter.Pup_ff;
    Gd_ff     = parameter.Gd_ff;
    Pdp_ff    = parameter.Pdp_ff;
    Qyp_ff    = parameter.Qyp_ff;

    
    % ------- Condition for 1st order and 1 Np
    if isempty(Pup_fb)
        Pup_fb = 0;
    end
    
    %% -----------------------------
    % Sizes inferred from matrices
    % ------------------------------
    n_yp_fb   = size(Qyp_fb,2);      % past outputs for FB predictor
    n_dup_fb  = size(Pup_fb,2);      % past input increments for FB predictor

    n_yp_ff   = size(Qyp_ff,2);      % past outputs for FF predictor
    n_dup_ff  = size(Pup_ff,2);      % past input increments for FF predictor
    n_ddp_ff  = size(Pdp_ff,2);      % past disturbance increments
    n_ddf_ff  = size(Gd_ff,2);       % future disturbance increments

    % Normally for SISO:
    % size(Gu_fb) = [Np Nc]
    % size(Gu_ff) = [Np Nc]

    %% -----------------------------
    % Decision variables
    % ------------------------------
    u_fb  = sdpvar(nu*Nc,1);         % future feedback increments
    u_ff  = sdpvar(nu*Nc,1);         % future feedforward increments
    u     = sdpvar(nu*Nc,1);         % total future increments

    y_fb  = sdpvar(ny*Np,1);         % feedback predicted output
    y_ff  = sdpvar(ny*Np,1);         % feedforward predicted output
        
    e_abs = sdpvar(ny*Np,1);         % Error variable for closed loop performance
    
    %% -----------------------------
    % Online parameters
    % ------------------------------
    y_fb_p   = sdpvar(n_yp_fb,1);   % past outputs for FB
    du_fb_p  = sdpvar(max(n_dup_fb,1),1);  % past input increments for FB

    y_ff_p      = sdpvar(n_yp_ff,1);   % past outputs for FF
    du_ff_p     = sdpvar(n_dup_ff,1);  % past input increments for FF
    dd_ff_p     = sdpvar(n_ddp_ff,1);  % past disturbance increments
    dd_ff_pred  = sdpvar(n_ddf_ff,1);  % future disturbance increments

    uk1_p     = sdpvar(nu,1);        % previous absolute input u(k-1)
    yref_p    = sdpvar(ny*Np,1);     % output reference
        
    %% -----------------------------
    % Free responses
    % ------------------------------
    F_fb = -Qyp_fb*y_fb_p + Pup_fb*du_fb_p;

    F_ff = -Qyp_ff*y_ff_p + Pup_ff*du_ff_p + Pdp_ff*dd_ff_p + Gd_ff*dd_ff_pred;

    %% -----------------------------
    % Weight matrices
    % ------------------------------
    % Output tracking weight
    if numel(lambda) == 1
        R = lambda*eye(ny*Np);
    else
        R = diag(kron(lambda,ones(Np,1)));
    end

    % FB increment penalty
    if numel(gamma_fb) == 1
        Q_fb = gamma_fb*eye(nu*Nc);
    else
        Q_fb = diag(kron(gamma_fb,ones(Nc,1)));
    end
    
    % FF increment penalty
    if numel(gamma_ff) == 1
        Q_ff = gamma_ff*eye(nu*Nc);
    else
        Q_ff = diag(kron(gamma_ff,ones(Nc,1)));
    end

    %% -----------------------------
    % Reference stacking
    % ------------------------------
    %W = yref_p(1)*ones(Np,1);
    W = yref_p;

    %% -----------------------------
    % Objective
    % ------------------------------
    objective = (y_fb - W)'*R*(y_fb - W) + u_fb'*Q_fb*u_fb + (y_ff)'*R*(y_ff) + u_ff'*Q_ff*u_ff;

    %% -----------------------------
    % Constraints
    % ------------------------------
    %u_ff = -((Gu_ff'*Gu_ff) + Q_ff)\(Gu_ff'*F_ff);
    
    constraints = [];

    % Prediction models
    constraints = [constraints, y_fb == Gu_fb*u_fb + F_fb];
    constraints = [constraints, y_ff == Gu_ff*u_ff + F_ff];

    % Total control increment decomposition
    constraints = [constraints, u == u_fb + u_ff];

    % Bounds on delta-u (total applied increment)
    for i = 1:nu
        idx = (i-1)*Nc + (1:Nc);
        constraints = [constraints, dumin(i) <= u(idx) <= dumax(i)];
    end

    % Bounds on absolute u using cumulative sum of total increments
    T_int = tril(ones(Nc,Nc));
    for i = 1:nu
        idx = (i-1)*Nc + (1:Nc);
        u_abs = T_int*u(idx) + uk1_p(i)*ones(Nc,1);
        constraints = [constraints, umin(i) <= u_abs <= umax(i)];
    end
    
    % Optional output constraints
    % If you want to constrain total predicted output, use y_fb + y_ff
    y_total = y_fb + y_ff;
    for i = 1:ny
        idy = (i-1)*Np + (1:Np);
        %constraints = [constraints, ymin(i) <= y_total(idy) <= ymax(i)];
    end
    
    %% -----------------------------
    % Solver options
    % ------------------------------
    ops = sdpsettings('solver','quadprog','warning',1,'verbose',0);

    %% -----------------------------
    % Build optimizer
    % ------------------------------
    GPC_ctrl.opt = optimizer(constraints, objective, ops, ...
        {y_fb_p, du_fb_p, y_ff_p, du_ff_p, dd_ff_p, dd_ff_pred, uk1_p, yref_p}, ...
        {u, u_fb, u_ff, y_fb, y_ff, y_total});

    %% -----------------------------
    % Store useful data
    % ------------------------------
    GPC_ctrl.nu      = nu;
    GPC_ctrl.nd      = nd;
    GPC_ctrl.ny      = ny;
    GPC_ctrl.Nc      = Nc;
    GPC_ctrl.Np      = Np;

    GPC_ctrl.Gu_fb   = Gu_fb;
    GPC_ctrl.Pup_fb  = Pup_fb;
    GPC_ctrl.Qyp_fb  = Qyp_fb;

    GPC_ctrl.Gu_ff   = Gu_ff;
    GPC_ctrl.Pup_ff  = Pup_ff;
    GPC_ctrl.Gd_ff   = Gd_ff;
    GPC_ctrl.Pdp_ff  = Pdp_ff;
    GPC_ctrl.Qyp_ff  = Qyp_ff;

    GPC_ctrl.R       = R;
    GPC_ctrl.Q_fb    = Q_fb;
    GPC_ctrl.Q_ff    = Q_ff;
end
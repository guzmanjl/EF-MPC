function MPC_ctrl = MPC_buildOptimizer(parameter)
%==========================================================================
% Build YALMIP optimizer for SS-MPC with FB/FF decomposition
%
% Physical prediction:
%   y_pred = Phi*x_tilde + Theta_u*u + Theta_d*dd_pred
%
% Auxiliary decompositions used in the cost:
%   y_fb = Phi_fb*x_fb + Theta_u*u_fb
%   y_ff = Phi_ff*x_ff + Theta_u*u_ff + Theta_d*dd_pred
%
% where:
%   u = u_fb + u_ff
%
% FB free response uses only selected FB-related states.
% FF free response uses only disturbance-related states, excluding the
% actual FB input-related states.
%==========================================================================

    %% -----------------------------
    % Parameters
    % ------------------------------
    nu        = parameter.nu;
    nd        = parameter.nd;
    ny        = parameter.ny;
    Np        = parameter.Np;
    Nc        = parameter.Nc;
    nx_ext    = parameter.nx_ext;

    lambda    = parameter.lambda(:);
    gamma_fb  = parameter.gamma_fb(:);
    gamma_ff  = parameter.gamma_ff(:);

    umax      = parameter.umax(:);
    umin      = parameter.umin(:);
    dumax     = parameter.dumax(:);
    dumin     = parameter.dumin(:);
    ymax      = parameter.ymax(:);
    ymin      = parameter.ymin(:);

    Phi       = parameter.Phi;
    Theta_u   = parameter.Theta_u;
    Theta_d   = parameter.Theta_d;

    idx_fb    = parameter.idx_fb(:).';
    idx_ff    = parameter.idx_ff(:).';

    %% -----------------------------
    % Reduced prediction matrices
    % ------------------------------
    Phi_fb = Phi(:,idx_fb);
    Phi_ff = Phi(:,idx_ff);

    nxfb   = length(idx_fb);
    nxff   = length(idx_ff);

    %% -----------------------------
    % Decision variables
    % ------------------------------
    u_fb   = sdpvar(nu*Nc,1);      % future feedback increments
    u_ff   = sdpvar(nu*Nc,1);      % future feedforward increments
    u      = sdpvar(nu*Nc,1);      % total future increments

    y_fb   = sdpvar(ny*Np,1);      % auxiliary FB prediction
    y_ff   = sdpvar(ny*Np,1);      % auxiliary FF prediction
    y_pred = sdpvar(ny*Np,1);      % true physical prediction

    %% -----------------------------
    % Online parameters
    % ------------------------------
    x_tilde_p  = sdpvar(nx_ext,1);     % full augmented state
    x_fb_p     = sdpvar(nxfb,1);       % selected FB substate
    x_ff_p     = sdpvar(nxff,1);       % selected FF substate
    dd_pred_p  = sdpvar(nd*Nc,1);      % future disturbance increments
    uk1_p      = sdpvar(nu,1);         % previous absolute input
    yref_p     = sdpvar(ny*Np,1);      % future reference trajectory

    %% -----------------------------
    % Free responses
    % ------------------------------
    F_fb = Phi_fb*x_fb_p;
    F_ff = Phi_ff*x_ff_p + Theta_d*dd_pred_p;
    

    %% -----------------------------
    % Weight matrices
    % ------------------------------
    if numel(lambda) == 1
        R = lambda*eye(ny*Np);
    else
        R = diag(kron(lambda,ones(Np,1)));
    end

    if numel(gamma_fb) == 1
        Q_fb = gamma_fb*eye(nu*Nc);
    else
        Q_fb = diag(kron(gamma_fb,ones(Nc,1)));
    end

    if numel(gamma_ff) == 1
        Q_ff = gamma_ff*eye(nu*Nc);
    else
        Q_ff = diag(kron(gamma_ff,ones(Nc,1)));
    end

    %% -----------------------------
    % Reference
    % ------------------------------
    W = yref_p;

    %% -----------------------------
    % Objective
    % ------------------------------
    objective = (y_fb - W)'*R*(y_fb - W) + u_fb'*Q_fb*u_fb + (y_ff)'*R*(y_ff) + u_ff'*Q_ff*u_ff;

    %% -----------------------------
    % Constraints
    % ------------------------------
    constraints = [];

    % Auxiliary decomposed predictions
    constraints = [constraints, y_fb == Theta_u*u_fb + F_fb];
    constraints = [constraints, y_ff == Theta_u*u_ff + F_ff];

    % Total increment decomposition
    constraints = [constraints, u == u_fb + u_ff];

    % True physical prediction
    constraints = [constraints, y_pred == Phi*x_tilde_p + Theta_u*u + Theta_d*dd_pred_p];

    % Bounds on Delta u
    for i = 1:nu
        idx = (i-1)*Nc + (1:Nc);
        constraints = [constraints, dumin(i) <= u(idx) <= dumax(i)];
    end

    % Bounds on absolute u
    T_int = tril(ones(Nc,Nc));
    for i = 1:nu
        idx = (i-1)*Nc + (1:Nc);
        u_abs = T_int*u(idx) + uk1_p(i)*ones(Nc,1);
        constraints = [constraints, umin(i) <= u_abs <= umax(i)];
    end

    % Optional output constraints on true physical prediction
    for i = 1:ny
        idy = (i-1)*Np + (1:Np);
        % constraints = [constraints, ymin(i) <= y_pred(idy) <= ymax(i)];
    end

    %% -----------------------------
    % Solver options
    % ------------------------------
    ops = sdpsettings('solver','quadprog','warning',1,'verbose',0);

    %% -----------------------------
    % Build optimizer
    % ------------------------------
    MPC_ctrl.opt = optimizer(constraints, objective, ops, ...
        {x_tilde_p, x_fb_p, x_ff_p, dd_pred_p, uk1_p, yref_p}, ...
        {u, u_fb, u_ff, y_fb, y_ff, y_pred});

    %% -----------------------------
    % Store useful data
    % ------------------------------
    MPC_ctrl.nu       = nu;
    MPC_ctrl.nd       = nd;
    MPC_ctrl.ny       = ny;
    MPC_ctrl.Nc       = Nc;
    MPC_ctrl.Np       = Np;
    MPC_ctrl.nx_ext   = nx_ext;

    MPC_ctrl.Phi      = Phi;
    MPC_ctrl.Phi_fb   = Phi_fb;
    MPC_ctrl.Phi_ff   = Phi_ff;
    MPC_ctrl.Theta_u  = Theta_u;
    MPC_ctrl.Theta_d  = Theta_d;

    MPC_ctrl.idx_fb   = idx_fb;
    MPC_ctrl.idx_ff   = idx_ff;

    MPC_ctrl.R        = R;
    MPC_ctrl.Q_fb     = Q_fb;
    MPC_ctrl.Q_ff     = Q_ff;
end
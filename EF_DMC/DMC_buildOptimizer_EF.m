function DMC_ctrl = DMC_buildOptimizer_EF(parameter)
%==========================================================================
% EF-DMC: Build YALMIP optimizer with Explicit Feedback/Feedforward
%
% Prediction:
%   y_fb = G_fb * Delta_u_fb + f_fb
%   y_ff = G_ff * Delta_u_ff + f_ff       (f_ff includes H_ff * v_f)
%   Delta_u = Delta_u_fb + Delta_u_ff
%
% f_fb and f_ff are pre-computed in main and passed as online parameters:
%
%   f_fb = G_p_fb * u_fb_p  +  eps_fb * ones
%   f_ff = G_p_ff * u_ff_p  +  H_p_ff * v_p  +  H_ff * v_f  +  eps_ff * ones
%
% Objective:
%   J = (y_fb - W)'*R*(y_fb - W)  +  Delta_u_fb'*Q_fb*Delta_u_fb
%     +  y_ff'*R*y_ff              +  Delta_u_ff'*Q_ff*Delta_u_ff
%
% Constraints:
%   y_fb   = G_fb * Delta_u_fb + f_fb
%   y_ff   = G_ff * Delta_u_ff + f_ff
%   Delta_u = Delta_u_fb + Delta_u_ff
%   dumin <= Delta_u                    <= dumax
%   umin  <= T * Delta_u + u_{k-1}      <= umax
%==========================================================================

    nu       = parameter.nu;
    ny       = parameter.ny;
    Np       = parameter.Np;
    Nc       = parameter.Nc;

    lambda   = parameter.lambda(:);
    gamma_fb = parameter.gamma_fb(:);
    gamma_ff = parameter.gamma_ff(:);

    umax  = parameter.umax(:);
    umin  = parameter.umin(:);
    dumax = parameter.dumax(:);
    dumin = parameter.dumin(:);

    G_fb = parameter.G_fb;   % [Np x Nc]  FB forced response
    G_ff = parameter.G_ff;   % [Np x Nc]  FF forced response

    %% Decision variables
    Delta_u_fb = sdpvar(nu*Nc, 1);
    Delta_u_ff = sdpvar(nu*Nc, 1);
    Delta_u    = sdpvar(nu*Nc, 1);

    y_fb = sdpvar(ny*Np, 1);
    y_ff = sdpvar(ny*Np, 1);

    %% Online parameters
    f_fb_p  = sdpvar(ny*Np, 1);   % FB free response  [Np x 1]
    f_ff_p  = sdpvar(ny*Np, 1);   % FF free response  [Np x 1]  (includes H_ff*v_f)
    uk1_p   = sdpvar(nu, 1);       % previous absolute u(k-1)
    yref_p  = sdpvar(ny*Np, 1);    % future reference  [Np x 1]

    %% Weight matrices
    if isscalar(lambda)
        R = lambda * eye(ny*Np);
    else
        R = diag(kron(lambda, ones(Np,1)));
    end

    if isscalar(gamma_fb)
        Q_fb = gamma_fb * eye(nu*Nc);
    else
        Q_fb = diag(kron(gamma_fb, ones(Nc,1)));
    end

    if isscalar(gamma_ff)
        Q_ff = gamma_ff * eye(nu*Nc);
    else
        Q_ff = diag(kron(gamma_ff, ones(Nc,1)));
    end

    %% Objective
    W = yref_p;
    objective = (y_fb - W)'*R*(y_fb - W) + Delta_u_fb'*Q_fb*Delta_u_fb ...
              + y_ff'*R*y_ff             + Delta_u_ff'*Q_ff*Delta_u_ff;

    %% Constraints
    constraints = [y_fb == G_fb * Delta_u_fb + f_fb_p];
    constraints = [constraints, y_ff == G_ff * Delta_u_ff + f_ff_p];
    constraints = [constraints, Delta_u == Delta_u_fb + Delta_u_ff];

    T_int = tril(ones(Nc, Nc));
    for i = 1:nu
        idx   = (i-1)*Nc + (1:Nc);
        u_abs = T_int * Delta_u(idx) + uk1_p(i) * ones(Nc,1);
        constraints = [constraints, dumin(i) <= Delta_u(idx) <= dumax(i)];
        constraints = [constraints, umin(i)  <= u_abs        <= umax(i)];
    end

    %% Solver options
    ops = sdpsettings('solver','quadprog','warning',1,'verbose',0);

    %% Build optimizer
    DMC_ctrl.opt = optimizer(constraints, objective, ops, ...
        {f_fb_p, f_ff_p, uk1_p, yref_p}, ...
        {Delta_u, Delta_u_fb, Delta_u_ff, y_fb, y_ff});

    %% Metadata
    DMC_ctrl.nu   = nu;
    DMC_ctrl.ny   = ny;
    DMC_ctrl.Nc   = Nc;
    DMC_ctrl.Np   = Np;
    DMC_ctrl.G_fb = G_fb;
    DMC_ctrl.G_ff = G_ff;
    DMC_ctrl.R    = R;
    DMC_ctrl.Q_fb = Q_fb;
    DMC_ctrl.Q_ff = Q_ff;
end

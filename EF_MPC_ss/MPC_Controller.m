function [u, u_fb, u_ff, y_fb, y_ff, y_pred, flag] = ...
    MPC_Controller(MPC_ctrl, x_tilde, x_fb, x_ff, dd_future, uk_1, ref_future)

    try
        [sol, problem] = MPC_ctrl.opt{...
            x_tilde, ...
            x_fb, ...
            x_ff, ...
            dd_future, ...
            uk_1, ...
            ref_future};

        u      = sol{1};
        u_fb   = sol{2};
        u_ff   = sol{3};
        y_fb   = sol{4};
        y_ff   = sol{5};
        y_pred = sol{6};

        tol = 1e-6;

        u(abs(u) < tol)         = 0;
        u_fb(abs(u_fb) < tol)   = 0;
        u_ff(abs(u_ff) < tol)   = 0;
        y_fb(abs(y_fb) < tol)   = 0;
        y_ff(abs(y_ff) < tol)   = 0;
        y_pred(abs(y_pred) < tol) = 0;

        flag = (problem == 0);

    catch
        u      = zeros(MPC_ctrl.nu*MPC_ctrl.Nc,1);
        u_fb   = zeros(MPC_ctrl.nu*MPC_ctrl.Nc,1);
        u_ff   = zeros(MPC_ctrl.nu*MPC_ctrl.Nc,1);
        y_fb   = zeros(MPC_ctrl.ny*MPC_ctrl.Np,1);
        y_ff   = zeros(MPC_ctrl.ny*MPC_ctrl.Np,1);
        y_pred = zeros(MPC_ctrl.ny*MPC_ctrl.Np,1);
        flag   = 0;
    end
end
%% ========================================================================
%  EF-DMC: Explicit Feedback / Feedforward DMC
%  ========================================================================
%--------------------------------------------------------------------------
% REQUIREMENTS: YALMIP, DMC_pred, DMC_pred_TISO, Model_sim
%               DMC_buildOptimizer_EF, DMC_Controller_EF
%==========================================================================
clc
clear
close all

addpath(genpath('/Users/JuanDiego/Desktop/YALMIP-master'))

%% -----------------------------------------------------------------------
%  System models
%% -----------------------------------------------------------------------
s   = tf('s');
Ts  = 1;

delay_u = 15;
delay_d = 10;

Gu      = 1 * exp(-delay_u*s) / (10*s + 1);
Gu_d         = c2d(Gu, Ts, 'zoh');
Gu_d.IODelay = delay_u;

Gd      = 0.8 * exp(-delay_d*s) / (5*s + 1);
Gd_d         = c2d(Gd, Ts, 'zoh');
Gd_d.IODelay = delay_d;


%% Plant simulation matrices
[A, Bu, Bd] = Model_sim(Gu_d, Gd_d);

%% -----------------------------------------------------------------------
%  Controller parameters
%% -----------------------------------------------------------------------
Np     = 60;
Nc     = Np;
N_past = 120;

lambda   = 1;
gamma_fb = 1;
gamma_ff = 0;

%% -----------------------------------------------------------------------
%  FB matrices  (full delay model)
%% -----------------------------------------------------------------------
[G_fb, G_p_fb, g_u_fb] = DMC_pred(Gu_d, Np, Nc, N_past);

%% -----------------------------------------------------------------------
%  FF matrices
%% -----------------------------------------------------------------------
Gd_ff_d          = Gd_d;
Gd_ff_d.IODelay  = max(delay_d, delay_u);

[G_ff, G_p_ff, H_ff, H_p_ff, g_u_ff, g_d_ff] = ...
    DMC_pred_TISO(Gu_d, Gd_ff_d, Np, Nc, N_past);

%% Pre-compute differential free-response matrices  (paper notation)
G_p_fb_diff = G_p_fb - ones(Np,1) * g_u_fb(1:N_past)';
G_p_ff_diff = G_p_ff - ones(Np,1) * g_u_ff(1:N_past)';
H_p_ff_diff = H_p_ff - ones(Np,1) * g_d_ff(1:N_past)';

%% -----------------------------------------------------------------------
%  Build optimizer
%% -----------------------------------------------------------------------
parameter.nu       = 1;
parameter.ny       = 1; 
parameter.Np       = Np;
parameter.Nc       = Nc;
parameter.lambda   = lambda;
parameter.gamma_fb = gamma_fb;
parameter.gamma_ff = gamma_ff;
parameter.umax     =  2;    parameter.umin  = -2;
parameter.dumax    =  100;   parameter.dumin = -100;
parameter.G_fb     = G_fb;
parameter.G_ff     = G_ff;

DMC_ctrl = DMC_buildOptimizer_EF(parameter);

%% -----------------------------------------------------------------------
%  Simulation
%% -----------------------------------------------------------------------
TN = 180;

ref = zeros(TN + parameter.Np + 2, 1);
ref(30:end)  = 0.2;
ref(130:end) = 0;

dis = zeros(TN + parameter.Np + 2, 1);
dis(60:end)  = -0.8;
dis(130:end) = 0;

%% Disturbance increments (non-causal shift if delay_d < delay_u)
dv = [0; diff(dis)];

if delay_d < delay_u
    shift = delay_u - delay_d;
    dv = [dv(1+shift:end); zeros(shift,1)];
end

%% Preallocate
y         = zeros(TN+1, 1);
u_log     = zeros(TN+1, 1);
du_log    = zeros(TN+1, 1);
du_fb_log = zeros(TN+1, 1);
du_ff_log = zeros(TN+1, 1);
flag      = zeros(TN+1, 1);

y_fb_all = zeros(Np, TN+1);
y_ff_all = zeros(Np, TN+1);

na  = length(A);   nbu = length(Bu);   nbd = length(Bd);
y_hist    = zeros(na-1, 1);
u_hist    = zeros(nbu,  1);
d_hist    = zeros(nbd,  1);
u_hist_ff = zeros(nbu,  1);   % for yff_aux simulation
y_hist_ff = zeros(na-1, 1);

uk_1    = 0;
uk_1_ff = 0;
u_fb_p  = zeros(N_past, 1);   % past Delta_u_fb
u_ff_p  = zeros(N_past, 1);   % past Delta_u_ff
v_p     = zeros(N_past, 1);   % past Delta_v

yff_aux   = zeros(TN+1, 1);     % plant output driven by u_ff + v only
yff_pred1 = 0;                  % first element of previous step y_ff prediction (GPC-like anchor)

%% Simulation loop
wait_bar = waitbar(0, 'Simulating EF-DMC...');

for k = 1:TN

    %% FB free response — anchor at (y - yff_aux): FB-only component
    f_fb = G_p_fb_diff * u_fb_p + (y(k) - yff_aux(k)) * ones(Np, 1);

    %% FF free response — GPC-style: anchor at previous step's y_ff prediction
    %  yff_pred1 = yff_pred(1,k) in GPC notation: the one-step-ahead FF
    %  prediction from the previous QP.  This creates the same absorption
    %  loop as -Qyp_ff*yp_ff in GPC: the anchor carries the disturbance
    %  preview information forward, driving Du_ff → 0 during tracking.
    v_f  = dv(k+1 : k+Nc);
    f_ff = G_p_ff_diff * u_ff_p + H_p_ff_diff * v_p + yff_pred1 * ones(Np,1) + H_ff * v_f;

    %% Future reference
    yref = ref(k+1 : k+Np);

    %% Solve QP
    [Du_k, Du_fb_k, Du_ff_k, y_fb_k, y_ff_k, ~, flag(k+1)] = ...
        DMC_Controller_EF(DMC_ctrl, f_fb, f_ff, uk_1, yref);

    %% Store
    du_log(k+1)    = Du_k(1);
    du_fb_log(k+1) = Du_fb_k(1);
    du_ff_log(k+1) = Du_ff_k(1);
    y_fb_all(:,k+1) = y_fb_k;
    y_ff_all(:,k+1) = y_ff_k;
    yff_pred1       = y_ff_k(1);   % GPC-like: carry first predicted y_ff to next step

    %% Apply total increment
    u_log(k+1) = uk_1 + du_log(k+1);
    uk_1       = u_log(k+1);

    %% Update past-increment buffers
    u_fb_p = [du_fb_log(k+1); u_fb_p(1:end-1)];
    u_ff_p = [du_ff_log(k+1); u_ff_p(1:end-1)];
    v_p    = [dv(k+1);        v_p(1:end-1)];

    %% Simulate plant (total input u_fb + u_ff)
    u_hist = [u_log(k+1); u_hist(1:end-1)];
    d_hist = [dis(k+1);   d_hist(1:end-1)];
    y_hist = [y(k);       y_hist(1:end-1)];

    y(k+1) = -A(2:end)*y_hist + Bu*u_hist + Bd*d_hist;

    %% Simulate auxiliary FF model (u_ff + v only) — for FB channel correction
    u_ff_log_k      = uk_1_ff + du_ff_log(k+1);
    uk_1_ff         = u_ff_log_k;
    u_hist_ff       = [u_ff_log_k; u_hist_ff(1:end-1)];
    y_hist_ff       = [yff_aux(k); y_hist_ff(1:end-1)];

    yff_aux(k+1) = -A(2:end)*y_hist_ff + Bu*u_hist_ff + Bd*d_hist;

    waitbar(k/TN, wait_bar)
end

close(wait_bar)

%% -----------------------------------------------------------------------
%  Plots
%% -----------------------------------------------------------------------
t = 0:TN;

figure(1); clf

subplot(3,1,1)
plot(t, y,            'LineWidth', 1.5); hold on
plot(t, ref(1:TN+1), 'r--', 'LineWidth', 1.5)
ylabel('y')
legend('y', 'ref', 'Location', 'best')
title('EF-DMC: closed-loop response')
grid on

subplot(3,1,2)
plot(t, u_log,     'LineWidth', 1.5); hold on
plot(t, du_fb_log, 'b:', 'LineWidth', 1.5)
plot(t, du_ff_log, 'm:', 'LineWidth', 1.5)
plot([0 TN], [ parameter.umax  parameter.umax], 'r--', 'LineWidth', 1)
plot([0 TN], [ parameter.umin  parameter.umin], 'r--', 'LineWidth', 1)
legend('u_{total}', '\Deltau_{fb}', '\Deltau_{ff}', 'u_{lim}', 'Location', 'best')
ylabel('Input')
grid on

subplot(3,1,3)
plot(t, dis(1:TN+1), 'LineWidth', 1.5)
ylabel('v  (disturbance)')
xlabel('Time [samples]')
grid on

%% ========================================================================
%  EXTENDED FEEDFORWARD GENERALIZED PREDICTIVE CONTROL (EF-GPC)
%  ------------------------------------------------------------------------
%  This script implements and simulates a GPC controller for a SISO system
%  with:
%     - Input (manipulated variable) dynamics including input delay
%     - Measurable disturbance dynamics with independent delay
%     - Combined feedback (FB) and feedforward (FF) control structure
%
%  The controller is formulated using:
%     - CARIMA-based prediction models
%     - Diophantine/Toeplitz prediction matrices (H, P, Q)
%     - Separate FB and FF prediction models
%     - YALMIP-based quadratic programming optimization
%
%  Control objective:
%     - Track a future reference trajectory over a prediction horizon (Np)
%     - Compensate measurable disturbances using preview information
%     - Minimize control effort (FB) and optionally FF contribution
%
%  The optimization problem is defined as:
%
%     min J = ||y_fb - w||^2 + ||Δu_fb||^2 + ||y_ff||^2
%
%  subject to:
%     y_fb = G_fb Δu_fb + F_fb
%     y_ff = G_ff Δu_ff + F_ff
%     Δu   = Δu_fb + Δu_ff
%     input and output constraints
%
%  Key features:
%     - Explicit handling of different delays in input and disturbance
%     - Disturbance preview (future increments Δd)
%     - Separate free-response terms for FB and FF
%     - Direct discrete-time simulation via difference equations
%
%  The process is simulated using:
%
%     y(k) = -A(2:end)*y_past + Bu*u_past + Bd*d_past
%
%  where (A, Bu, Bd) are obtained from the discretized plant model.
%
%  Simulation includes:
%     - Reference step changes
%     - Disturbance step (or ramp) profiles
%     - Online optimization using precompiled YALMIP optimizer
%
%--------------------------------------------------------------------------
% REQUIREMENTS
%--------------------------------------------------------------------------
% - YALMIP toolbox
% - Functions:
%     GPC_buildOptimizer
%     GPC_Controller
%     GPC_pred_TISO
%     GPC_pred
%     GPC_Controller
%
%--------------------------------------------------------------------------
% Author: Igor M.L. Pataro
%==========================================================================

clc
clear
close all

%% Add path for Yalmip master folder
addpath(genpath('/Users/igorpataro/Documents/MATLAB/YALMIP-master'))

%% Define System Model
s           = tf('s');
Ts          = 1;

num_u       = (1);
den_u       = (10*s + 1);
delay_u     = 10;
Gu          = num_u*exp(-delay_u*s)/den_u;

% Discretize
Gu_d         = c2d(Gu,Ts,'zoh');
Gu_d.IODelay = delay_u;

%% Define Disturbance Model
nun_d       = (0.8);
den_d       = (5*s + 1);
delay_d     = 15;
Gd          = nun_d*exp(-delay_d*s)/den_d;

% Discretize
Gd_d        = c2d(Gd,Ts,'zoh');
Gd_d.IODelay = delay_d;

%% Model Simulation parameters in discrete time
[A,Bu,Bd] = Model_sim(Gu_d,Gd_d);

%% Define Controller Parameters
parameter.Ts         = Ts;
parameter.nu         = 1;
parameter.nd         = 1;
parameter.ny         = 1;
parameter.delay_u    = delay_u;
parameter.delay_d    = delay_d;
parameter.Np         = 60;
parameter.Nc         = parameter.Np;
parameter.lambda     = 1;
parameter.gamma_fb   = 1;
parameter.gamma_ff   = 0;
parameter.umax       = 200;
parameter.umin       = -200;
parameter.dumax      = 100;
parameter.dumin      = -100;
parameter.ymax       = 100;
parameter.ymin       = -100;

%% Obtain GPC matrices for input + disturbance model
if delay_d >= delay_u
    Gu_d.IODelay = 0;
    Gd_d.IODelay = delay_d-delay_u;
else
    Gu_d.IODelay = 0;
    Gd_d.IODelay = 0;
end

[Gu_ff,Pup_ff,Gd_ff,Pdp_ff,Qyp_ff] = GPC_pred_TISO(Gu_d,Gd_d,parameter.Np,parameter.Nc);

parameter.Gu_ff   = Gu_ff;
parameter.Pup_ff  = Pup_ff;
parameter.Gd_ff   = Gd_ff;
parameter.Pdp_ff  = Pdp_ff;
parameter.Qyp_ff  = Qyp_ff;

%% Obtain GPC matrices for feedback model
Gu_d.IODelay = delay_u;
[Gu_fb,Pup_fb,Qyp_fb] = GPC_pred(Gu_d,parameter.Np,parameter.Nc);

parameter.Gu_fb   = Gu_fb;
parameter.Pup_fb  = Pup_fb;
parameter.Qyp_fb  = Qyp_fb;

%% Create optimization problem
GPC_ctrl = GPC_buildOptimizer(parameter);

%% Simulation horizon
TN = 180;

%% Build future reference and disturbance profiles
% Reference changes
ref           = zeros(TN + parameter.Np + 2,1);
ref(30:end)   = 0.2;
ref(130:end)  = 0;

% Disturbance changes
dis           = zeros(TN + parameter.Np + 2,1);

% Step disturbance
dis(60:end)   = -0.8;   % downward step
dis(130:end)  = 0.0;    % upward step

%% Disturbance increments profile
dd_profile = [0; diff(dis)];

% Non-causal scenario
if delay_d < delay_u
    delay_diff = delay_u-delay_d;
    dd_profile = [dd_profile(1+delay_diff:end); zeros(delay_diff,1)];
end

%% Preallocate arrays
y           = zeros(TN+1,1);
yff_aux     = zeros(TN+1,1);
yu          = zeros(TN+1,1);
yd          = zeros(TN+1,1);

u_opt       = zeros(TN+1,1);
u_opt_ff    = zeros(TN+1,1);
du_total    = zeros(TN+1,1);
du_fb       = zeros(TN+1,1);
du_ff       = zeros(TN+1,1);

flag        = zeros(TN+1,1);

yfb_pred    = zeros(parameter.Np,TN+1);
yff_pred    = zeros(parameter.Np,TN+1);
ytot_pred   = zeros(parameter.Np,TN+1);

na  = length(A);
nbu = length(Bu);
nbd = length(Bd);

% Histories for plant simulation
y_hist = zeros(na-1,1);   % [y(k), y(k-1), ...] after update convention
u_hist = zeros(nbu,1);    % [u(k), u(k-1), ...]
d_hist = zeros(nbd,1);    % [d(k), d(k-1), ...]

% Histories for ff model simulation
u_hist_ff = zeros(nbu,1);    % [u(k), u(k-1), ...]
y_hist_ff = zeros(na-1,1);    % [u(k), u(k-1), ...]

%% Initial conditions
uk_1        = 0;
uk_1_ff     = 0;

% Predictor histories
yp_fb       = zeros(size(parameter.Qyp_fb,2),1);
yp_ff       = zeros(size(parameter.Qyp_ff,2),1);
du_fb_p     = zeros(size(parameter.Pup_fb,2),1);

du_ff_p    = zeros(size(parameter.Pup_ff,2),1);
dd_ff_p    = zeros(size(parameter.Pdp_ff,2),1);

% Optional prediction error filter
ef          = 0;
yk_pred     = 0;
error       = zeros(TN+1,1);

%% Waitbar
wait_bar = waitbar(0,'Simulating EF-GPC...');
%load test_u_MPC

for k = 1:TN
    %% Current measured output enters histories
    yp_fb = [y(k) - yff_aux(k); yp_fb(1:end-1)];
    yp_ff = [yff_pred(1,k); yp_ff(1:end-1)];
    
    %% Prediction error
    error(k) = y(k) -  yfb_pred(1,k);
    ef     = ef + 0.15*(error(k) - ef); 
    
    %% Build future reference trajectory W_future (Np x 1 for SISO)
    ref_future = ref(k+1:k+parameter.Np);

    %% Build future disturbance increments preview
    % Length must match number of columns of Gd_ff (Np)
    ddf_future = dd_profile(k+1:k+parameter.Nc);
    
    %% Run GPC controller
    [du_k, dufb_k, duff_k, yfbk_pred, yffk_pred, ytotk_pred, flag(k+1)] = ...
        GPC_Controller( ...
            GPC_ctrl, ...
            yp_fb, du_fb_p, ...
            yp_ff, du_ff_p, dd_ff_p, ddf_future, ...
            uk_1, ref_future, ...
            parameter);

    %% Save increments
    du_total(k+1)   = du_k(1);
    du_fb(k+1)      = dufb_k(1);
    du_ff(k+1)      = duff_k(1);

    %% Save predictions
    yfb_pred(:,k+1)   = yfbk_pred;
    yff_pred(:,k+1)   = yffk_pred;
    ytot_pred(:,k+1)  = ytotk_pred;

    %% Update input histories
    du_fb_p = [du_fb(k+1); du_fb_p(1:end-1)];
    du_ff_p = [du_ff(k+1); du_ff_p(1:end-1)];
    dd_ff_p = [dd_profile(k+1); dd_ff_p(1:end-1)];

    %% Apply first increment
    u_opt(k+1) = uk_1 + du_total(k+1);
    uk_1       = u_opt(k+1);
    
    %% Simulate process
    % Update histories with current applied values
    u_hist = [u_opt(k+1); u_hist(1:end-1)];
    d_hist = [dis(k+1);   d_hist(1:end-1)];
    y_hist = [y(k); y_hist(1:end-1)];
    
    % Process output
    y(k+1) =  - A(2:end)*y_hist + Bu*u_hist + Bd*d_hist;
    
    %% Simulate FF component on the output (Model)
    u_opt_ff(k+1) = uk_1_ff + du_ff(k+1);
    uk_1_ff       = u_opt_ff(k+1);
    
    u_hist_ff = [u_opt_ff(k+1); u_hist_ff(1:end-1)];
    y_hist_ff = [yff_aux(k); y_hist_ff(1:end-1)];
    
    yff_aux(k+1) =  - A(2:end)*y_hist_ff + Bu*u_hist_ff + Bd*d_hist;

    %% Waitbar update
    waitbar(k/TN,wait_bar)

end

close(wait_bar)

%% Final plots of predictions at last instant
figure(1); clf
subplot(3,1,1)
plot(y(1:TN),'LineWidth',1.5)
hold on
plot(ref(1:TN),'r--','LineWidth',1.5)
ylabel('Output')
grid on

subplot(3,1,2)
plot(u_opt(1:TN),'LineWidth',1.5)
hold on
plot(du_fb(1:TN),'b:','LineWidth',1.5)
plot(du_ff(1:TN),'m:','LineWidth',1.5)
%plot(ones(TN,1)*parameter.umax,'r--','LineWidth',1.5)
%plot(ones(TN,1)*parameter.umin,'r--','LineWidth',1.5)
%legend('u_{total}','\Delta u_{fb}','\Delta u_{ff}','u_{lim}');
legend('u_{total}','\Delta u_{fb}','\Delta u_{ff}');
ylabel('Input')
grid on

subplot(3,1,3)
stairs(dis(1:TN),'LineWidth',1.5)
ylabel('Disturbance')
xlabel('Time')
grid on


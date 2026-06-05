%==========================================================================
% EXTENDED FF STATE-SPACE MPC 
%--------------------------------------------------------------------------
% This script implements a Model Predictive Control (MPC) strategy in
% state-space form for a SISO system with measurable disturbance.
%
% The controller is formulated using an augmented incremental model that
% explicitly includes:
%   - Input increments (Δu)
%   - Disturbance increments (Δd)
%   - Input and disturbance delays
%
% The prediction model is obtained from a discrete-time state-space
% representation derived from transfer functions, and extended to include
% delay states and previous inputs/disturbances.
%
%--------------------------------------------------------------------------
% CONTROL STRUCTURE
%--------------------------------------------------------------------------
% The control action is decomposed into:
%
%   Δu = Δu_fb + Δu_ff
%
% where:
%   - Δu_fb : feedback component (tracking)
%   - Δu_ff : feedforward component (disturbance compensation)
%
% Two auxiliary internal models are used:
%   - x_fb_aux : state driven only by Δu_fb
%   - x_ff_aux : state driven by Δu_ff and Δd
%
% The physical plant state (x_proc) is updated using the total input Δu.
%
% A consistency condition is enforced:
%
%   x_fb_aux(idx_fb) = x_proc(idx_fb) - x_ff_aux(idx_fb)
%
% ensuring coherence between the decomposed models and the real system.
%
%--------------------------------------------------------------------------
% MPC FORMULATION
%--------------------------------------------------------------------------
% The predicted output is decomposed as:
%
%   y_fb = Phi_fb * x_fb + Theta_u * Δu_fb
%   y_ff = Phi_ff * x_ff + Theta_u * Δu_ff + Theta_d * Δd
%
% while the true plant prediction is:
%
%   y_pred = Phi * x_proc + Theta_u * Δu + Theta_d * Δd
%
% The optimization problem minimizes:
%
%   J = (y_fb - r)' Q (y_fb - r)
%     + Δu_fb' R_fb Δu_fb
%     + y_ff' Q y_ff
%     + Δu_ff' R_ff Δu_ff
%
% subject to:
%   - input increment constraints
%   - input magnitude constraints
%
%--------------------------------------------------------------------------
% FEATURES
%--------------------------------------------------------------------------
% - Generic handling of arbitrary-order systems
% - Explicit treatment of input and disturbance delays
% - Separation of feedback and feedforward contributions
% - Disturbance preview over the prediction horizon
% - Consistency correction using plant state feedback
%
%--------------------------------------------------------------------------
% SIMULATION
%--------------------------------------------------------------------------
% The system is simulated in closed-loop with:
%   - Step changes in reference
%   - Step disturbance profile
%   - Optional plant mismatch (state scaling at k = 200)
%
% Outputs include:
%   - System output tracking
%   - Control input and its FB/FF components
%   - Disturbance profile
%
%--------------------------------------------------------------------------
% REQUIREMENTS
%--------------------------------------------------------------------------
% - YALMIP toolbox
% - Functions:
%     augment_ss_incremental_delays
%     MPC_pred_TISO
%     MPC_buildOptimizer
%     MPC_Controller
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

%% Define Disturbance Model
nun_d       = (0.8);
den_d       = (5*s + 1);
delay_d     = 15;
Gd          = nun_d*exp(-delay_d*s)/den_d;

%% Transfer Function Matrix
G_tf        = [Gu Gd];
G_tf_d      = c2d(G_tf,Ts,'zoh');

delays = [delay_u delay_d];

%% Obtain extended incremental model with delays
[Atil,Butil,Ctil,Bdtil,model_ss] = augment_ss_incremental_delays(G_tf_d,delays,size(Gd,2),size(Gd,2));

parameter.nx        = model_ss.nx;        % original state dimension
parameter.nx_ext    = model_ss.nx_ext;    % augmented state dimension
parameter.nzu       = model_ss.nzu;       % input-delay states
parameter.nzd       = model_ss.nzd;       % disturbance-delay states
parameter.A         = model_ss.A;
parameter.Bu        = model_ss.Bu;
parameter.Bd        = model_ss.Bd;
parameter.C         = model_ss.C;
parameter.D         = model_ss.D;
parameter.delay_u   = delay_u;
parameter.delay_d   = delay_d;
parameter.Atil      = Atil;
parameter.Butil     = Butil;
parameter.Bdtil     = Bdtil;
parameter.Ctil      = Ctil;
parameter.Ts        = Ts;
parameter.Np        = 60;
parameter.Nc        = parameter.Np;
parameter.nd        = 1;
parameter.nu        = size(G_tf,2) - parameter.nd;
parameter.ny        = size(G_tf,1);
parameter.lambda    = 1;
parameter.gamma_fb  = 1;
parameter.gamma_ff  = 0;
parameter.umax      = 200;
parameter.umin      = -200;
parameter.dumax     = 100;
parameter.dumin     = -100;
parameter.ymax      = 100;
parameter.ymin      = -100;

% FB indices
[A_fb, ~,~,~,] = ssdata(c2d(Gu,Ts,'zoh'));

parameter.nx_fb     = size(A_fb,1);
idx_x_fb  = 1:parameter.nx_fb;
idx_zu    = parameter.nx + (1:parameter.nzu);
idx_uprev = parameter.nx + parameter.nzu + parameter.nzd + (1:parameter.nu);

parameter.idx_fb = [idx_x_fb idx_zu idx_uprev];

% FF indices
idx_x_ff  = 1:parameter.nx;
idx_zd    = parameter.nx + parameter.nzu + (1:parameter.nzd);
idx_dprev = parameter.nx + parameter.nzu + parameter.nzd + parameter.nu + (1:parameter.nd);

% Include input delay states and previous input in FF too
parameter.idx_ff = [idx_x_ff idx_zu idx_uprev idx_zd idx_dprev];

%% Construct model prediction matrices with disturbance

[Phi, Theta_u, Theta_d] = MPC_pred_TISO(model_ss, parameter);

parameter.Phi       = Phi;
parameter.Theta_u   = Theta_u;
parameter.Theta_d   = Theta_d;

%% Create optimization problem

MPC_ctrl = MPC_buildOptimizer(parameter);

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

% if delay_d < delay_u
%    delay_diff = delay_u-delay_d;
%    dd_profile = [dd_profile(1+delay_diff:end); zeros(delay_diff,1)];
% end

%% Preallocate arrays
y           = zeros(TN+1,1);
u_opt       = zeros(TN+1,1);
du_total    = zeros(TN+1,1);
du_fb       = zeros(TN+1,1);
du_ff       = zeros(TN+1,1);

flag        = zeros(TN+1,1);

yfb_pred    = zeros(parameter.Np,TN+1);
yff_pred    = zeros(parameter.Np,TN+1);
ytot_pred   = zeros(parameter.Np,TN+1);

%% Initial conditions
uk_1 = 0;

% Physical augmented state
x_proc  = zeros(parameter.nx_ext,1);

% Auxiliary states for decomposition
x_fb_aux = zeros(parameter.nx_ext,1);
x_ff_aux = zeros(parameter.nx_ext,1);

%% Optional prediction error filter
ef    = 0;
error = zeros(TN+1,1);

%% Waitbar
wait_bar = waitbar(0,'Simulating EF-SSMPC...');

for k = 1:TN

    %% Prediction error (using physical prediction)
    error(k) = y(k) - ytot_pred(1,k);
    ef       = ef + 0.15*(error(k) - ef); 

    %% Build reduced states for FB and FF
    x_fb = x_fb_aux(parameter.idx_fb);
    x_ff = x_ff_aux(parameter.idx_ff);

    %% Build future reference trajectory
    ref_future = ref(k+1:k+parameter.Np);

    %% Build future disturbance increments preview
    ddf_future = dd_profile(k+1:k+parameter.Nc);

    %% Run MPC controller
    [du_k, dufb_k, duff_k, yfbk_pred, yffk_pred, ypred_k, flag(k+1)] = ...
        MPC_Controller( ...
            MPC_ctrl, ...
            x_proc, ...
            x_fb, ...
            x_ff, ...
            ddf_future, ...
            uk_1, ...
            ref_future);

    %% Save first increments
    du_total(k+1) = du_k(1);
    du_fb(k+1)    = dufb_k(1);
    du_ff(k+1)    = duff_k(1);

    %% Save predictions
    yfb_pred(:,k+1)  = yfbk_pred;
    yff_pred(:,k+1)  = yffk_pred;
    ytot_pred(:,k+1) = ypred_k;

    %% Apply first increment
    u_opt(k+1) = uk_1 + du_total(k+1);
    uk_1       = u_opt(k+1);

    %% Apply input and disturbance in the system model
    % Full augmented model uses Delta u and Delta d
    x_proc  = parameter.Atil*x_proc + ...
              parameter.Butil*du_total(k+1) + ...
              parameter.Bdtil*dd_profile(k+1);
          
    if k == 200
        x_proc(1:parameter.nx) = x_proc(1:parameter.nx)*0.8;
    end
    
    y(k+1)  = parameter.Ctil*x_proc;
    
    %% Apply the separated control actions in each model for FB and FF
    %x_fb_aux = parameter.Atil*x_fb_aux + ...
    %           parameter.Butil*du_fb(k+1);
       
    x_ff_aux = parameter.Atil*x_ff_aux + ...
               parameter.Butil*du_ff(k+1) + ...
               parameter.Bdtil*dd_profile(k+1);   
    
    % Consistency correction using plant state
    x_fb_aux(parameter.idx_fb) = x_proc(parameter.idx_fb) - x_ff_aux(parameter.idx_fb);
    
    %% Waitbar update
    waitbar(k/TN,wait_bar)

end
close(wait_bar)

%% Final plots
figure(1); clf
subplot(3,1,1)
plot(y(1:TN),'LineWidth',1.5)
hold on
plot(ref(1:TN),'r--','LineWidth',1.5)
ylabel('Output')
legend('y','ref','Location','best')
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

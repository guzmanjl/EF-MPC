function [Atil,Butil,Ctil,Bdtil,model_ss] = augment_ss_incremental_delays(Gd_tf,delays,nu,nd)
%==========================================================================
% Build an augmented discrete-time state-space model for MPC/GPC with:
%   - incremental manipulated inputs Delta u
%   - incremental measurable disturbances Delta d
%   - explicit delay chains as augmented states
%
% INPUTS:
%   Gd_tf   : discrete-time transfer matrix [ny x (nu+nd)]
%   delays  : vector of integer sample delays for each input channel
%             Example: [delay_u1 ... delay_unu delay_d1 ... delay_dnd]
%   nu      : number of manipulated inputs
%   nd      : number of measurable disturbances
%
% OUTPUTS:
%   Atil    : augmented state matrix
%   Butil   : augmented input matrix for Delta u
%   Ctil    : augmented output matrix
%   Bdtil   : augmented input matrix for Delta d
%   model_ss: structure with useful original and augmented model data
%
% ORIGINAL DELAY-FREE DISCRETE MODEL:
%   x(k+1) = A x(k) + Bu u_del(k) + Bd d_del(k)
%   y(k)   = C x(k) + Du u_del(k) + Dd d_del(k)
%
% AUGMENTED STATE:
%   x_tilde = [ x ; z_u ; z_d ; u_prev ; d_prev ]
%
% where:
%   z_u = delay-chain states for manipulated inputs
%   z_d = delay-chain states for disturbances
%   u_prev = absolute manipulated inputs at time k-1
%   d_prev = absolute disturbance values at time k-1
%
% Inputs to the augmented model:
%   Delta u(k), Delta d(k)
%
%   u(k) = u_prev(k) + Delta u(k)
%   d(k) = d_prev(k) + Delta d(k)
%
% Notes:
%   - Gd_tf must already be discrete-time
%   - delays must be integer sample delays
%==========================================================================

    %% Checks
    if Gd_tf.Ts <= 0
        error('augment_ss_incremental_delays: Gd_tf must be discrete-time.');
    end

    if nargin < 3 || isempty(nu)
        error('augment_ss_incremental_delays: nu must be provided.');
    end

    if nargin < 4 || isempty(nd)
        error('augment_ss_incremental_delays: nd must be provided.');
    end

    [ny, nin] = size(Gd_tf);

    if nu + nd ~= nin
        error('augment_ss_incremental_delays: nu + nd must equal number of inputs of Gd_tf.');
    end

    if length(delays) ~= nin
        error('augment_ss_incremental_delays: length(delays) must equal number of inputs of Gd_tf.');
    end

    delays = round(delays(:).');

    %% Get delay-free state-space realization
    % The delays are handled explicitly in the augmented model,
    % so remove them from the realization first
    Gaux = Gd_tf;
    if isprop(Gaux,'IODelay')
        try
            Gaux.IODelay = zeros(size(Gaux.IODelay));
        catch
        end
    end
    if isprop(Gaux,'InputDelay')
        try
            Gaux.InputDelay = zeros(size(Gaux.InputDelay));
        catch
        end
    end
    if isprop(Gaux,'OutputDelay')
        try
            Gaux.OutputDelay = zeros(size(Gaux.OutputDelay));
        catch
        end
    end

    [A,B,C,D] = ssdata(ss(Gaux));

    %% Split manipulated and disturbance channels
    Bu = B(:,1:nu);
    Bd = B(:,nu+1:nu+nd);

    Du = D(:,1:nu);
    Dd = D(:,nu+1:nu+nd);

    du = delays(1:nu);
    dd = delays(nu+1:nu+nd);

    %% Dimensions of original model
    nx = size(A,1);

    %% Build delay-chain matrices for manipulated inputs
    [Su, Eu, Ju, Nu0, nzu] = local_build_delay_chain(du);

    %% Build delay-chain matrices for disturbances
    [Sd, Ed, Jd, Nd0, nzd] = local_build_delay_chain(dd);

    %% Augmented dimensions
    nx_ext = nx + nzu + nzd + nu + nd;

    %% Indexing
    ix  = 1:nx;
    izu = nx + (1:nzu);
    izd = nx + nzu + (1:nzd);
    iu  = nx + nzu + nzd + (1:nu);
    id  = nx + nzu + nzd + nu + (1:nd);

    %% Allocate matrices
    Atil  = zeros(nx_ext,nx_ext);
    Butil = zeros(nx_ext,nu);
    Bdtil = zeros(nx_ext,nd);
    Ctil  = zeros(ny,nx_ext);

    %% x(k+1)
    Atil(ix,ix) = A;

    if nzu > 0
        Atil(ix,izu) = Bu*Ju;
    end
    if nzd > 0
        Atil(ix,izd) = Bd*Jd;
    end

    Atil(ix,iu) = Bu*Nu0;
    Atil(ix,id) = Bd*Nd0;

    Butil(ix,:) = Bu*Nu0;
    Bdtil(ix,:) = Bd*Nd0;

    %% z_u(k+1) = Su z_u(k) + Eu u(k)
    if nzu > 0
        Atil(izu,izu) = Su;
        Atil(izu,iu)  = Eu;
        Butil(izu,:)  = Eu;
    end

    %% z_d(k+1) = Sd z_d(k) + Ed d(k)
    if nzd > 0
        Atil(izd,izd) = Sd;
        Atil(izd,id)  = Ed;
        Bdtil(izd,:)  = Ed;
    end

    %% u_prev(k+1) = u_prev(k) + Delta u(k)
    Atil(iu,iu) = eye(nu);
    Butil(iu,:) = eye(nu);

    %% d_prev(k+1) = d_prev(k) + Delta d(k)
    Atil(id,id) = eye(nd);
    Bdtil(id,:) = eye(nd);

    %% Output matrix
    Ctil(:,ix) = C;

    if any(Du(:) ~= 0)
        if nzu > 0
            Ctil(:,izu) = Ctil(:,izu) + Du*Ju;
        end
        Ctil(:,iu) = Ctil(:,iu) + Du*Nu0;
    end

    if any(Dd(:) ~= 0)
        if nzd > 0
            Ctil(:,izd) = Ctil(:,izd) + Dd*Jd;
        end
        Ctil(:,id) = Ctil(:,id) + Dd*Nd0;
    end

    %% Useful output structure
    model_ss.A   = A;
    model_ss.B   = B;
    model_ss.C   = C;
    model_ss.D   = D;
    model_ss.Bu  = Bu;
    model_ss.Bd  = Bd;
    model_ss.Du  = Du;
    model_ss.Dd  = Dd;
    
    model_ss.Atil  = Atil;
    model_ss.Butil = Butil;
    model_ss.Ctil  = Ctil;
    model_ss.Bdtil = Bdtil;
    
    model_ss.du  = du;
    model_ss.dd  = dd;

    model_ss.nx      = nx;
    model_ss.nzu     = nzu;
    model_ss.nzd     = nzd;
    model_ss.nx_ext  = nx_ext;
    model_ss.nu      = nu;
    model_ss.nd      = nd;
    model_ss.ny      = ny;
end


%==========================================================================
% Helper: build delay chain matrices for a vector of delays
%==========================================================================
function [S,E,J,N0,nz] = local_build_delay_chain(dvec)

    nin = length(dvec);
    nz  = sum(dvec);

    if nz == 0
        S  = zeros(0,0);
        E  = zeros(0,nin);
        J  = zeros(nin,0);
        N0 = eye(nin);
        return
    end

    S  = zeros(nz,nz);
    E  = zeros(nz,nin);
    J  = zeros(nin,nz);
    N0 = zeros(nin,nin);

    offset = 0;
    for i = 1:nin
        di = dvec(i);

        if di == 0
            N0(i,i) = 1;
        else
            idx = offset + (1:di);

            Si = [zeros(1,di); eye(di-1), zeros(di-1,1)];

            Ei = zeros(di,1);
            Ei(1) = 1;

            Ji = zeros(1,di);
            Ji(end) = 1;

            S(idx,idx) = Si;
            E(idx,i)   = Ei;
            J(i,idx)   = Ji;

            offset = offset + di;
        end
    end
end
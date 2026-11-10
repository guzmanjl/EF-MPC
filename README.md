# Unlocking Feedforward Capabilities in MPC

MATLAB implementations of three Model Predictive Control algorithms with embedded feedforward capabilities for measurable disturbance rejection, as proposed in:

> J. L. Guzmán, I. Pataro, J. D. Gil, M. Berenguel, *"Unlocking feedforward capabilities in Model Predictive Control algorithms to deal with measurable disturbances"*, 2026.

---

## Overview

Standard MPC formulations penalise control effort, which prevents them from fully reproducing the behaviour of classical feedforward compensators — even when the disturbance model is perfectly known. This repository provides a unified framework that embeds **true feedforward capabilities** into MPC without removing the control-effort penalty.

The core idea is to decompose the control action into two independent components:

```
Δu(k) = Δu_fb(k) + Δu_ff(k)
```

| Component | Objective | Control effort penalty |
|-----------|-----------|----------------------|
| `Δu_fb` | Set-point tracking and robustness | `λ_c > 0` (tunable) |
| `Δu_ff` | Measurable disturbance rejection | `λ_v = 0` (pure feedforward) |

Both actions are computed within a **single QP optimisation**, so actuator constraints are enforced on the combined signal `u_fb + u_ff`. This allows the optimiser to coordinate both objectives when, for example, feedforward inversion is non-trivial or the actuator is near saturation.

---

## Repository Structure

```
.
├── EF_GPC/                     # Extended Feedforward GPC (CARIMA / transfer function)
│   ├── main_EF_GPC.m           # Main simulation script
│   ├── GPC_pred.m              # GPC prediction matrices (SISO, feedback channel)
│   ├── GPC_pred_TISO.m         # GPC prediction matrices (TISO, FB + FF channels)
│   ├── GPC_pred_TISO_dia.m     # Variant with Diophantine-equation formulation
│   ├── GPC_buildOptimizer.m    # YALMIP optimizer construction (FB/FF decomposition)
│   ├── GPC_Controller.m        # Online optimizer call
│   ├── Model_sim.m             # Discrete-time plant simulation helper
│   └── tuneLambdaFBbyArea.m    # Utility: auto-tune λ_c by area criterion
│
├── EF_DMC/                     # Extended Feedforward DMC (step-response / convolution)
│   ├── main_EF_DMC.m           # Main simulation script
│   ├── DMC_pred.m              # DMC step-response matrices (SISO)
│   ├── DMC_pred_TISO.m         # DMC step-response matrices (TISO, FB + FF)
│   ├── DMC_buildOptimizer_EF.m # YALMIP optimizer construction (FB/FF decomposition)
│   ├── DMC_Controller_EF.m     # Online optimizer call
│   └── Model_sim.m             # Discrete-time plant simulation helper
│
└── EF_MPC_ss/                  # Extended Feedforward State-Space MPC
    ├── main_EF_MPC_ss.m        # Main simulation script
    ├── augment_ss_incremental_delays.m  # Build augmented incremental SS model with delay chains
    ├── MPC_pred_TISO.m         # State-space prediction matrices (Φ, Θ_u, Θ_d)
    ├── MPC_pred.m              # State-space prediction matrices (no disturbance)
    ├── MPC_buildOptimizer.m    # YALMIP optimizer construction (FB/FF decomposition)
    └── MPC_Controller.m        # Online optimizer call
```

---

## Algorithms

### EF-GPC — Extended Feedforward Generalised Predictive Control

Prediction is derived from the CARIMA model using Toeplitz/Diophantine matrices. The free responses are split as:

- **FB free response**: `f_c = G_p u_{p,c} + S y_p`  ← driven by plant measurements
- **FF free response**: `f_v = G_p u_{p,v} + S ŷ_p + H v + H_p v_p`  ← open-loop, driven by disturbance model only

Key files: `GPC_pred.m` (feedback channel), `GPC_pred_TISO.m` (joint TISO model for the FF channel).

### EF-DMC — Extended Feedforward Dynamic Matrix Control

Prediction is built from the process step-response coefficients `{g_i}` and disturbance step-response coefficients `{h_i}`. Free responses follow the same FB/FF split as EF-GPC, using Toeplitz matrices `G`, `G_p`, `H`, `H_p`.

Key files: `DMC_pred.m` (SISO step-response), `DMC_pred_TISO.m` (joint TISO model).

### EF-MPC-SS — Extended Feedforward State-Space MPC

An augmented incremental state-space model is constructed that explicitly embeds input and disturbance delay chains:

```
x̃(k+1) = Ã x̃(k) + B̃_u Δu(k) + B̃_d Δv(k)
y(k)    = C̃ x̃(k)
```

where `x̃ = [x; z_u; z_d; u_prev; d_prev]`. Two parallel internal states track the FB and FF contributions separately, with a consistency correction applied each step to enforce `x̃_fb + x̃_ff = x̃_proc`.

Key files: `augment_ss_incremental_delays.m` (augmented model builder), `MPC_pred_TISO.m` (stacked prediction matrices Φ, Θ_u, Θ_d).

---

## Optimisation Problem

All three algorithms solve the same structured QP at each sampling instant:

**Minimise:**

$$J = \underbrace{(\hat{y}_c - w_c)^T(\hat{y}_c - w_c) + u_c^T Q_{\lambda_c} u_c}_{\text{feedback: tracking}} + \underbrace{(\hat{y}_v - w_v)^T(\hat{y}_v - w_v) + u_v^T Q_{\lambda_v} u_v}_{\text{feedforward: disturbance rejection}}$$

**Subject to:**
- `ŷ_c = G u_c + f_c`
- `ŷ_v = G u_v + f_v`
- `u = u_c + u_v`
- `u_min ≤ u(k-1) + cumsum(Δu) ≤ u_max`
- `Δu_min ≤ Δu_c + Δu_ff ≤ Δu_max`

Setting `Q_λv = 0` and `w_v = 0` recovers **pure feedforward** behaviour (classical process inversion) for the disturbance channel, while `Q_λc > 0` keeps the feedback channel well-conditioned.

---

## Requirements

- **MATLAB** R2019b or later (Control System Toolbox required for `c2d`, `tf`, `ss`)
- **YALMIP** (free, available at [yalmip.github.io](https://yalmip.github.io))
- A QP solver compatible with YALMIP (e.g., `quadprog` from MATLAB Optimization Toolbox, or OSQP)

---

## Quick Start

1. Download and add YALMIP to your MATLAB path:
   ```matlab
   addpath(genpath('/path/to/YALMIP-master'))
   ```

2. Open one of the main scripts, update the YALMIP path at the top, and run:
   ```matlab
   run('EF_GPC/main_EF_GPC.m')
   run('EF_DMC/main_EF_DMC.m')
   run('EF_MPC_ss/main_EF_MPC_ss.m')
   ```

Each script is self-contained and produces a three-panel figure showing:
- **Top**: plant output `y` vs. reference
- **Middle**: total input `u` and its decomposition into `Δu_fb` and `Δu_ff`
- **Bottom**: disturbance profile `v`

---

## Simulation Scenarios

Each main script reproduces the simulation sequence from the paper over 180 sampling instants:

| Time | Event |
|------|-------|
| k = 30 | Set-point step change |
| k = 60 | Measurable disturbance step |
| k = 130 | Simultaneous set-point + disturbance change |

The default process and disturbance models are:

$$P_u(s) = \frac{1}{10s+1}e^{-10s}, \qquad P_v(s) = \frac{0.8}{5s+1}e^{-15s}$$

The `main_EF_DMC.m` script uses the **non-invertible** (non-causal) scenario with swapped delays (`delay_u = 15`, `delay_d = 10`), which is one of the cases where classical external feedforward fails and the proposed embedded formulation excels.

---

## Key Tuning Parameters

| Parameter | Description | Typical value |
|-----------|-------------|---------------|
| `Np` | Prediction horizon | 60 |
| `Nc` | Control horizon | 60 |
| `lambda` (= `gamma_fb`) | FB control-effort weight | 1 |
| `gamma_ff` | FF control-effort weight | **0** (pure FF) |
| `umax / umin` | Absolute input bounds | ±1.3 (constrained scenario) |
| `dumax / dumin` | Input increment bounds | ±100 (unconstrained) |

---

## Reference

If you use this code, please cite:

```bibtex
@article{guzman2026unlocking,
  title   = {Unlocking feedforward capabilities in Model Predictive Control algorithms
             to deal with measurable disturbances},
  author  = {Guzm\'{a}n, Jos\'{e} Luis and Pataro, Igor and Gil, Juan D. and Berenguel, Manuel},
  journal = {Journal of Process Control},
  DOI = {https://doi.org/10.1016/j.jprocont.2026.103887),
  year    = {2026}
}
```

[https://doi.org/10.1016/j.jprocont.2026.103887](https://doi.org/10.1016/j.jprocont.2026.103887) 

**Acknowledgements:** This work was financed by projects PID2023-150739OB-I00 and PDC2025-165379-I00 funded by the Spanish Ministry of Science.

---

## Authors

- José Luis Guzmán — University of Almería (joseluis.guzman@ual.es)
- Igor Pataro — University of Almería (igorpataro@ual.es)
- Juan Diego Gil — University of Almería (juandiego.gil@ual.es)
- Manuel Berenguel — University of Almería (beren@ual.es)

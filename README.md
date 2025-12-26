# Ageing–Rejuvenation Dynamics (illustrative mechanistic model)

This repository provides an illustrative, mechanistic implementation of the dynamical model described in the Supplementary Information of the accompanying manuscript on an integrative-first framework of ageing and rejuvenation.

The model is intended to make the conceptual architecture executable and transparent: slow lesion-like inputs interact with organism-level signalling fields and reinforcement at the integrative interface, generating regime-dependent ageing trajectories and intervention signatures. The implementation is deliberately minimal and modular so that additional state variables, alternative functional forms, and application-specific parameterisations can be introduced as needed.

## Repository structure

- `model/parameters.py`  
  Default parameters (single source of truth) and helper utilities.
- `model/dynamics.py`  
  ODE right-hand-side and simulation wrapper.
- `model/interventions.py`  
  Intervention schedules (systemic-field modulation; interface modulation; cell-intrinsic modulation).
- `model/hazard.py`  
  Emergent hazard readout from system state (for illustrative Gompertz-like acceleration).
- `run_simulation.py`  
  One-command run to reproduce illustrative trajectories and figures.

## Quick start

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

"""
Minimal dynamical model used in Supplementary Information (illustrative / mechanistic).

State vector order (consistent with your notebook):
  y = [D, S, A, Y]

Parameters (names aligned to Supplementary File 1, Table 1):
  μ_D    : damage accumulation rate
  λ_Y    : decline rate of systemic capacity
  Y_min  : residual regenerative capacity
  α_DS   : damage–senescence coupling
  β_SY   : suppression of senescence by Y
  α_AS   : reinforcement of senescence by A
  α_SA   : production of A by senescence
  γ_A    : feedback amplification coefficient
  β_AY   : suppression of A by Y

Notes:
- Time is in arbitrary units (a.u.), as in the Supplement.
- This code is intended to reproduce the manuscript Supplement figures from the provided equations/parameters.
"""

from __future__ import annotations

from typing import Dict, Tuple, Callable
import numpy as np
from scipy.integrate import solve_ivp


Params = Dict[str, float]


def minimal_model_v2(t: float, state: np.ndarray, params: Params) -> np.ndarray:
    """
    ODE right-hand side for the minimal model.

    State order:
      D = cumulative molecular damage
      S = senescence-associated burden
      A = systemic pro-ageing field
      Y = regenerative / clearance capacity
    """

    D, S, A, Y = state

    # Parameters (aligned to Supplementary File 1, Table 1)
    mu_D = params["μ_D"]        # damage accumulation rate
    lam_Y = params["λ_Y"]       # decline rate of systemic capacity
    Y_min = params["Y_min"]     # residual regenerative capacity

    alpha_DS = params["α_DS"]   # damage–senescence coupling
    beta_SY = params["β_SY"]    # suppression of senescence by Y
    alpha_AS = params["α_AS"]   # reinforcement of senescence by A

    alpha_SA = params["α_SA"]   # production of A by senescence
    gamma_A = params["γ_A"]     # feedback amplification coefficient
    beta_AY = params["β_AY"]    # suppression of A by Y

    # ODEs (same structure as your notebook)
    dDdt = mu_D
    dYdt = -lam_Y * (Y - Y_min)
    dSdt = alpha_DS * dDdt - beta_SY * Y + alpha_AS * A
    dAdt = alpha_SA * S + gamma_A * S * A - beta_AY * Y * A

    return np.array([dDdt, dSdt, dAdt, dYdt], dtype=float)


def run(
    params: Params,
    t_end: float,
    y0: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    max_step: float = 0.1,
) -> "solve_ivp":
    """
    Integrate the model from t=0 to t=t_end.
    Returns the SciPy solve_ivp solution object (with sol.t and sol.y).
    """
    sol = solve_ivp(
        fun=lambda t, y: minimal_model_v2(t, y, params),
        t_span=(0.0, float(t_end)),
        y0=np.array(y0, dtype=float),
        max_step=float(max_step),
        dense_output=True,
    )
    return sol


def compute_X_h(
    sol,
    h_min: float = 0.0015,
    kappa: float = 0.0023,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hazard proxy used for internal plotting in the notebook.

    X = A - Y
    h = h_min + kappa * max(0, X)

    Returns: t, X, h
    """
    t = sol.t
    D, S, A, Y = sol.y
    X = A - Y
    h = h_min + kappa * np.maximum(0.0, X)
    return t, X, h

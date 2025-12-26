"""
Ageing & Rejuvenation Dynamics — illustrative mechanistic model

This script implements a compact, mechanistic dynamical system capturing:
- near-linear accumulation of primary lesion burden (D)
- a nonlinear interface/propagation burden (S) representing integrative amplification
- systemic pro-ageing field (A) and pro-youth buffering/clearance capacity (Y)
- an emergent mortality-hazard readout (h) as a function of systemic state

It is designed to generate regime-dependent intervention signatures and
illustrate how supralinear systemic reinforcement can produce Gompertz-like
late-life acceleration while D remains near-linear.

Usage (examples):
    python ageing_rejuvenation_dynamics.py --outdir figures
    python ageing_rejuvenation_dynamics.py --scenario late_reset --outdir figures
    python ageing_rejuvenation_dynamics.py --scenario early_prevention --outdir figures

Dependencies:
    numpy, scipy, matplotlib
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# -------------------------
# Model parameters
# -------------------------

@dataclass(frozen=True)
class Params:
    # Primary lesion accumulation
    d0: float = 0.02            # baseline lesion input rate
    d_decay: float = 0.002      # slow clearance/repair term for D

    # Interface burden / amplification (senescence-associated signalling proxy)
    s_prod_from_D: float = 0.04 # conversion of D into S (stress→interface)
    s_self_amp: float = 0.12    # self-amplification strength of S (nonlinear)
    s_half: float = 0.55        # half-saturation for self-amplification
    s_decay: float = 0.03       # baseline attenuation of S

    # Systemic fields
    a_prod_from_S: float = 0.20 # S drives pro-ageing field A
    a_decay: float = 0.18       # clearance of A

    y_recovery: float = 0.10    # baseline recovery of Y towards y_max
    y_max: float = 1.00         # ceiling for Y
    y_supp_by_A: float = 0.12   # A suppresses Y (immune/endocrine drift etc.)
    y_decay: float = 0.02       # baseline decay of Y

    # Coupling: Y constrains S/A (buffering)
    y_suppresses_S: float = 0.25  # Y reduces effective S amplification
    y_clears_A: float = 0.22      # Y increases effective A clearance

    # Hazard readout (emergent)
    h0: float = 1e-4            # baseline hazard scale
    h_S: float = 3.2            # hazard sensitivity to S
    h_A: float = 2.1            # hazard sensitivity to A
    h_Y: float = 1.8            # hazard protection from Y


@dataclass(frozen=True)
class Intervention:
    """
    Time-dependent intervention knobs.
    Provide callables that return multiplicative (>=0) or additive terms.

    - d_mult(t): multiplies d0 (cell-autonomous "damage-input modulation")
    - a_remove(t): additive removal flux on A (e.g., dilution/TPE-like)
    - y_boost(t): additive boost flux on Y (e.g., EV/CM-like buffering restoration)
    - s_clear(t): additive clearance flux on S (e.g., senolytic/immune recalibration-like)
    """
    d_mult: Callable[[float], float] = lambda t: 1.0
    a_remove: Callable[[float], float] = lambda t: 0.0
    y_boost: Callable[[float], float] = lambda t: 0.0
    s_clear: Callable[[float], float] = lambda t: 0.0


# -------------------------
# Core dynamics
# -------------------------

def rhs(t: float, x: np.ndarray, p: Params, u: Intervention) -> np.ndarray:
    """
    State x = [D, S, A, Y]
    """
    D, S, A, Y = x

    # Keep variables within sane numerical bounds without hard clipping the solver state.
    # (We use soft constraints in the equations rather than post-step clipping.)
    Y_eff = max(0.0, Y)

    # 1) Primary lesion burden D: near-linear accumulation with weak repair
    d_in = p.d0 * max(0.0, u.d_mult(t))
    dD = d_in - p.d_decay * D

    # 2) Interface burden S: produced from D and self-amplifies, restrained by Y
    # Self-amplification uses a saturating nonlinearity and is damped by Y.
    amp = p.s_self_amp * (S / (p.s_half + S + 1e-12)) * S
    amp *= (1.0 / (1.0 + p.y_suppresses_S * Y_eff))
    dS = p.s_prod_from_D * D + amp - p.s_decay * S - max(0.0, u.s_clear(t))

    # 3) Pro-ageing field A: driven by S and cleared, with additional clearance from Y
    a_clear = (p.a_decay + p.y_clears_A * Y_eff) * A
    dA = p.a_prod_from_S * S - a_clear - max(0.0, u.a_remove(t))

    # 4) Pro-youth / buffering capacity Y: recovers towards y_max, suppressed by A
    # Recovery term: y_recovery * (y_max - Y); suppression: y_supp_by_A * A * Y
    dY = p.y_recovery * (p.y_max - Y_eff) - p.y_supp_by_A * A * Y_eff - p.y_decay * Y_eff + max(0.0, u.y_boost(t))

    return np.array([dD, dS, dA, dY], dtype=float)


def hazard(D: np.ndarray, S: np.ndarray, A: np.ndarray, Y: np.ndarray, p: Params) -> np.ndarray:
    """
    Emergent hazard readout h(t) (no feedback into state).
    """
    # Keep Y from creating negative exponent blow-ups if solver slightly crosses below 0.
    Y_eff = np.maximum(Y, 0.0)
    z = p.h_S * S + p.h_A * A - p.h_Y * Y_eff
    return p.h0 * np.exp(z)


# -------------------------
# Scenarios (regime-dependent signatures)
# -------------------------

def scenario_baseline() -> Intervention:
    return Intervention()


def scenario_early_prevention(t_on: float = 20.0, strength: float = 0.55) -> Intervention:
    # Reduce effective lesion input rate early (cell-autonomous prevention-dominant).
    def d_mult(t: float) -> float:
        return (1.0 - strength) if t >= t_on else 1.0
    return Intervention(d_mult=d_mult)


def scenario_late_reset(
    t_on: float = 70.0,
    a_remove_rate: float = 0.35,
    y_boost_rate: float = 0.12,
    duration: float = 6.0,
) -> Intervention:
    # Late-life systemic signalling reset: remove A and boost Y for a limited window.
    def window(t: float) -> bool:
        return (t_on <= t <= (t_on + duration))

    def a_remove(t: float) -> float:
        return a_remove_rate if window(t) else 0.0

    def y_boost(t: float) -> float:
        return y_boost_rate if window(t) else 0.0

    return Intervention(a_remove=a_remove, y_boost=y_boost)


def scenario_combo(
    t_early: float = 25.0,
    d_strength: float = 0.45,
    t_late: float = 70.0,
    a_remove_rate: float = 0.30,
    y_boost_rate: float = 0.10,
    duration: float = 6.0,
) -> Intervention:
    # Combined: early prevention + late reset
    u1 = scenario_early_prevention(t_on=t_early, strength=d_strength)
    u2 = scenario_late_reset(t_on=t_late, a_remove_rate=a_remove_rate, y_boost_rate=y_boost_rate, duration=duration)

    def d_mult(t: float) -> float:
        return u1.d_mult(t)

    def a_remove(t: float) -> float:
        return u2.a_remove(t)

    def y_boost(t: float) -> float:
        return u2.y_boost(t)

    return Intervention(d_mult=d_mult, a_remove=a_remove, y_boost=y_boost)


SCENARIOS: Dict[str, Callable[[], Intervention]] = {
    "baseline": scenario_baseline,
    "early_prevention": scenario_early_prevention,
    "late_reset": scenario_late_reset,
    "combo": scenario_combo,
}


# -------------------------
# Simulation + plotting
# -------------------------

def simulate(
    u: Intervention,
    p: Params,
    t_span: Tuple[float, float] = (0.0, 90.0),
    dt: float = 0.1,
    x0: Tuple[float, float, float, float] = (0.0, 0.03, 0.04, 1.0),
) -> Dict[str, np.ndarray]:
    t_eval = np.arange(t_span[0], t_span[1] + dt, dt)

    sol = solve_ivp(
        fun=lambda t, x: rhs(t, x, p, u),
        t_span=t_span,
        y0=np.array(x0, dtype=float),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-7,
        atol=1e-9,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    D, S, A, Y = sol.y
    h = hazard(D, S, A, Y, p)

    return {"t": sol.t, "D": D, "S": S, "A": A, "Y": Y, "h": h}


def save_plots(outdir: str, name: str, res: Dict[str, np.ndarray]) -> None:
    os.makedirs(outdir, exist_ok=True)
    t = res["t"]

    # Plot 1: State trajectories
    plt.figure()
    plt.plot(t, res["D"], label="D (primary lesions)")
    plt.plot(t, res["S"], label="S (interface / amplification)")
    plt.plot(t, res["A"], label="A (pro-ageing field)")
    plt.plot(t, res["Y"], label="Y (buffering capacity)")
    plt.xlabel("Time (arbitrary units)")
    plt.ylabel("State (a.u.)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_states.png"), dpi=300)
    plt.close()

    # Plot 2: Hazard (log scale) to visualise Gompertz-like acceleration
    plt.figure()
    plt.semilogy(t, res["h"], label="h (emergent hazard)")
    plt.xlabel("Time (arbitrary units)")
    plt.ylabel("Hazard (log scale, a.u.)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_hazard.png"), dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="baseline", choices=sorted(SCENARIOS.keys()))
    parser.add_argument("--outdir", type=str, default="figures")
    parser.add_argument("--tmax", type=float, default=90.0)
    args = parser.parse_args()

    p = Params()
    u = SCENARIOS[args.scenario]()  # default parameters for that scenario

    res = simulate(u=u, p=p, t_span=(0.0, float(args.tmax)))
    save_plots(args.outdir, args.scenario, res)

    # Minimal terminal output (kept small on purpose)
    print(f"Saved: {os.path.join(args.outdir, args.scenario + '_states.png')}")
    print(f"Saved: {os.path.join(args.outdir, args.scenario + '_hazard.png')}")


if __name__ == "__main__":
    main()

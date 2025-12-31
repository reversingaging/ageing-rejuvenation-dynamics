"""
Plotting utilities for Supplementary figures.

Assumptions:
- solve_ivp solution uses state order: [D, S, A, Y]
- time is arbitrary units (a.u.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from model import compute_X_h


# ----------------------------
# Global plot style (simple, journal-friendly)
# ----------------------------
def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


# ----------------------------
# Helper: extract states from solve_ivp solution
# Assumes sol.y = [D, S, A, Y]
# ----------------------------
def extract_states(sol):
    t = sol.t
    D, S, A, Y = sol.y
    return t, D, S, A, Y


# ----------------------------
# Supplement Fig. S2: baseline dynamics (4 panels: D, Y, S, A)
# Optional inset on A panel (normalised A and Y in early regime)
# ----------------------------
def plot_baseline_S2(
    sol,
    xlim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    add_inset: bool = True,
    inset_xlim: Optional[Tuple[float, float]] = None,
    outdir: str = "figures_supp",
    filename_prefix: str = "FigS2_baseline",
    save: bool = True,
):
    """
    Produces a single 2x2 multi-panel figure.

    Panels:
      (a) D(t): cumulative molecular damage
      (b) Y(t): regenerative / clearance capacity
      (c) S(t): senescence-associated burden
      (d) A(t): systemic pro-ageing field (+ optional inset of A and Y, normalised)

    No numeric age claims; time axis is arbitrary units (a.u.).
    """
    set_plot_style()
    t, D, S, A, Y = extract_states(sol)

    if xlim is None:
        xlim = (float(t.min()), float(t.max()))

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(9.5, 6.5), constrained_layout=True
    )

    # Panel 1: D
    ax = axes[0, 0]
    ax.plot(t, D)
    ax.set_xlim(*xlim)
    ax.set_xlabel("time, t (a.u.)")
    ax.set_ylabel("D(t)")
    ax.set_title("Cumulative molecular damage")

    # Panel 2: Y
    ax = axes[0, 1]
    ax.plot(t, Y)
    ax.set_xlim(*xlim)
    ax.set_xlabel("time, t (a.u.)")
    ax.set_ylabel("Y(t)")
    ax.set_title("Regenerative / clearance capacity")

    # Panel 3: S
    ax = axes[1, 0]
    ax.plot(t, S)
    ax.set_xlim(*xlim)
    ax.set_xlabel("time, t (a.u.)")
    ax.set_ylabel("S(t)")
    ax.set_title("Senescence-associated burden")

    # Panel 4: A (+ inset)
    axA = axes[1, 1]
    axA.plot(t, A)
    axA.set_xlim(*xlim)
    axA.set_xlabel("time, t (a.u.)")
    axA.set_ylabel("A(t)")
    axA.set_title("Systemic pro-ageing field")

    if title:
        fig.suptitle(title, y=1.02)

    if add_inset:
        if inset_xlim is None:
            t0 = xlim[0]
            t1 = xlim[0] + (xlim[1] - xlim[0]) * 0.55
            inset_xlim = (t0, t1)

        mask = (t >= inset_xlim[0]) & (t <= inset_xlim[1])
        t_in = t[mask]
        A_in = A[mask]
        Y_in = Y[mask]

        # Normalise for visibility; avoid division by zero
        A_den = np.max(A_in) if np.max(A_in) > 0 else 1.0
        Y_den = np.max(Y_in) if np.max(Y_in) > 0 else 1.0
        A_norm = A_in / A_den
        Y_norm = Y_in / Y_den

        inset = axA.inset_axes([0.08, 0.25, 0.6, 0.52])
        inset.plot(t_in, A_norm, label="A(t)")
        inset.plot(t_in, Y_norm, label="Y(t)")
        inset.set_xlim(inset_xlim)
        inset.set_ylim(0, 1.2)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title("early regime (normalised)", fontsize=9)
        inset.legend(loc="lower left", fontsize=8)

    if save:
        pdf_path = outdir_path / f"{filename_prefix}.pdf"
        svg_path = outdir_path / f"{filename_prefix}.svg"
        png_path = outdir_path / f"{filename_prefix}.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        fig.savefig(png_path, bbox_inches="tight")

    return fig


# ----------------------------
# Optional: single-panel overlay (A and Y)
# ----------------------------
def plot_A_Y_overlay(
    sol,
    xlim: Optional[Tuple[float, float]] = None,
    title: str = "Systemic field and clearance capacity",
    outdir: str = "figures_supp",
    filename_prefix: str = "FigS2A_A_Y_overlay",
    save: bool = True,
):
    set_plot_style()
    t, D, S, A, Y = extract_states(sol)

    if xlim is None:
        xlim = (float(t.min()), float(t.max()))

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 4.5))
    plt.plot(t, A, label="A(t): systemic pro-ageing field")
    plt.plot(t, Y, label="Y(t): regenerative/clearance capacity")
    plt.xlim(*xlim)
    plt.xlabel("time, t (a.u.)")
    plt.ylabel("state (a.u.)")
    plt.title(title)
    plt.legend()

    if save:
        pdf_path = outdir_path / f"{filename_prefix}.pdf"
        svg_path = outdir_path / f"{filename_prefix}.svg"
        png_path = outdir_path / f"{filename_prefix}.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(svg_path, bbox_inches="tight")
        fig.savefig(png_path, bbox_inches="tight")

    return fig


# ----------------------------
# Supplement Fig. S1: baseline dynamics + hazard panels (3x2 layout)
# ----------------------------
def plot_baseline_S1_with_hazard(
    sol,
    xlim: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    add_inset: bool = True,
    inset_xlim: Optional[Tuple[float, float]] = None,
    outdir: str = "figures_supp",
    filename_prefix: str = "FigS1_baseline_with_hazard",
    save: bool = True,
    h_min: float = 0.0015,
    excess_xlim_frac: float = 0.4,
    hazard_ylim: Optional[Tuple[float, float]] = None,
    excess_ylim: Tuple[float, float] = (1e-3, 0.3),
):
    """
    Produces a single 3x2 multi-panel figure.
    Layout:
      Row 1: D(t) | Y(t)
      Row 2: S(t) | A(t) (+ inset of normalised A and Y)
      Row 3: h(t) | excess hazard (semi-log)

    - x-axis: time t (a.u.)
    - avoids numeric age claims; no Gompertz fitting
    - uses compute_X_h() from model.py
    """
    set_plot_style()
    t, D, S, A, Y = extract_states(sol)

    # Hazard computation
    t_h, X, h = compute_X_h(sol, h_min=h_min)

    # Ensure consistent time base (usually identical)
    if len(t_h) != len(t) or np.any(np.abs(t_h - t) > 1e-12):
        h = np.interp(t, t_h, h)
        t_h = t

    if xlim is None:
        xlim = (float(t.min()), float(t.max()))

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(10.5, 9.5), constrained_layout=True
    )

    # Row 1: D
    ax = axes[0, 0]
    ax.plot(t, D)
    ax.set_xlim(*xlim)
    ax.set_xlabel("time, t (a.u.)")
    ax.set_ylabel("D(t)")
    ax.set_title("Cumulative molecular damage")

    # Row 1: Y
    ax = axes[0, 1]
    ax.plot(t, Y)
    ax.set_xlim(*xlim)
    ax.set_xlabel("time, t (a.u.)")
    ax.set_ylabel("Y(t)")
    ax.set_title("Regenerative / clearance capacity")

    # Row 2: S
    ax = axes[1, 0]
    ax.plot(t, S)
    ax.set_xlim(*xlim)
    ax.set_xlabel("time, t (a.u.)")
    ax.set_ylabel("S(t)")
    ax.set_title("Senescence-associated burden")

    # Row 2: A (+ Y) + inset
    axA = axes[1, 1]
    axA.plot(t, A, label="A(t)")
    axA.plot(t, Y, label="Y(t)")
    axA.set_xlim(*xlim)
    axA.set_xlabel("time, t (a.u.)")
    axA.set_ylabel("A(t), Y(t)")
    axA.set_title("Systemic field and capacity")
    axA.legend(loc="best", fontsize=9)

    if add_inset:
        if inset_xlim is None:
            t0 = xlim[0]
            t1 = xlim[0] + (xlim[1] - xlim[0]) * 0.55
            inset_xlim = (t0, t1)

        mask = (t >= inset_xlim[0]) & (t <= inset_xlim[1])
        t_in = t[mask]
        A_in = A[mask]
        Y_in = Y[mask]

        A_den = np.max(A_in) if np.max(A_in) > 0 else 1.0
        Y_den = np.max(Y_in) if np.max(Y_in) > 0 else 1.0
        A_norm = A_in / A_den
        Y_norm = Y_in / Y_den

        inset = axA.inset_axes([0.10, 0.28, 0.6, 0.55])
        inset.plot(t_in, A_norm, label="A(t)")
        inset.plot(t_in, Y_norm, label="Y(t)")
        inset.set_xlim(inset_xlim)
        inset.set_ylim(0, 1.2)
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title("early regime (normalised)", fontsize=9)
        inset.legend(loc="lower left", fontsize=8)

    # Row 3: hazard
    ax = axes[2, 0]
    ax.plot(t_h, h)
    ax.set_xlim(*xlim)
    ax.set_xlabel("time, t (a.u.)")
    ax.set_ylabel("h(t)")
    ax.set_title("Hazard")

    if hazard_ylim is not None:
        ax.set_ylim(*hazard_ylim)

    # Row 3: excess hazard (semi-log)
    ax = axes[2, 1]
    excess = np.maximum(h - h_min, 1e-12)

    t_start = xlim[0] + (xlim[1] - xlim[0]) * float(excess_xlim_frac)
    mask_ex = (t_h >= t_start) & (t_h <= xlim[1])

    ax.semilogy(t_h[mask_ex], excess[mask_ex])
    ax.set_xlim(t_start, xlim[1])
    ax.set_ylim(*excess_ylim)
    ax.set_xlabel("time, t (a.u.)")
    ax.set_ylabel(r"$h(t)-h_{\min}$")
    ax.set_title("Excess hazard (semi-log)")

    if title:
        fig.suptitle(title, y=1.02)

    if save:
        for ext in ["pdf", "svg", "png"]:
            path = outdir_path / f"{filename_prefix}.{ext}"
            fig.savefig(path, bbox_inches="tight")

    return fig

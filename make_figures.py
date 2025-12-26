"""
Generate Supplementary figures from the illustrative ageing–rejuvenation dynamical model.

This script is intended to be run from the repository root:
  python make_figures.py

It produces figure files under:
  figures_supp/
"""

from __future__ import annotations

from pathlib import Path

from model import run
from plotting import plot_baseline_S2, plot_baseline_S1_with_hazard


def main() -> None:
    # Output directory
    outdir = Path("figures_supp")
    outdir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Initial conditions and time horizon
    # State order assumed throughout: [D, S, A, Y]
    # ----------------------------
    y0 = (0.0, 0.0, 0.0, 1.0)
    t_end = 90.0
    max_step = 0.1

    # ----------------------------
    # Parameters (must match Supplement naming exactly)
    # Paste your exact params dict here.
    # ----------------------------
    params = {
        # TODO: paste exact parameter names + values from your Supplement
        # Example (REMOVE): "a": 1.0,
    }

    if not params:
        raise RuntimeError(
            "Parameters not set. Paste the exact params dict (names + values) from the Supplement."
        )

    # ----------------------------
    # Run baseline simulation
    # ----------------------------
    sol0 = run(params=params, t_end=t_end, y0=y0, max_step=max_step)

    # ----------------------------
    # Fig S2: baseline 2x2 panels (D, Y, S, A) + inset
    # ----------------------------
    plot_baseline_S2(
        sol=sol0,
        xlim=(0, t_end),
        title=None,
        add_inset=True,
        inset_xlim=(0, 40),
        outdir=str(outdir),
        filename_prefix="FigS2_baseline",
        save=True,
    )

    # ----------------------------
    # Fig S1: baseline 3x2 panels incl hazard + excess hazard
    # ----------------------------
    plot_baseline_S1_with_hazard(
        sol=sol0,
        xlim=(0, t_end),
        title=None,
        add_inset=True,
        h_min=0.0015,
        outdir=str(outdir),
        filename_prefix="FigS1_baseline",
        save=True,
    )

    print("Done. Figures saved under figures_supp/.")


if __name__ == "__main__":
    main()

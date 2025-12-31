Contributing

Thanks for your interest.

This repository accompanies a Perspective manuscript and provides a small illustrative / mechanistic dynamical model and plotting scripts to reproduce the Supplementary figures.

Scope
- Bug fixes (typos, broken imports, plotting issues, reproducibility issues)
- Documentation improvements (README clarity, comments)
- Small refactors that do not change the model equations or reported baseline outputs

Out of scope (for now)
- Changing equations, parameterisation, or figure definitions
- Adding new intervention scenarios or fitting to datasets
- Adding dependencies beyond NumPy / SciPy / Matplotlib unless clearly justified

How to contribute
1. Open an issue describing the change and why it is needed.
2. If you plan a code change, propose a minimal patch.
3. Keep changes small and focused.

Reproducibility check
Before submitting, please run:
- python make_figures.py

A successful run should generate figure files under figures_supp/ without errors.

Licence
By contributing, you agree that your contributions will be licensed under the repository licence.

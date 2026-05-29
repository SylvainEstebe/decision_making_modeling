# Legacy R / JAGS implementation

This folder preserves the **original** R code that accompanied the 2024
Decision Making exam submission at Aarhus University. The active analysis
has moved to a Python / PyMC implementation under `src/conditional_cooperation/`
at the repository root — see the top-level [`README.md`](../README.md).

## Contents

- `sylvain_analysis_worry.R` — main analysis script (worry covariate)
- `sylvain_analysis_experience.R` — main analysis script (experience covariate)
- `CC_corr.txt` — JAGS model: hierarchical CC + national-level regression
- `parameter_recovery/CC_individual.txt` — JAGS model: subject-level CC only
- `parameter_recovery/` — recovery simulation utilities
- `result/` — figures and saved MCMC samples from the original run
- `analysis/saved_JAGS/` — `.Rdata` traces for the final run

## How to re-run (historical)

These scripts target R ≥ 4.2 and depend on JAGS, `R2jags`, `polspline`,
`ggplot2`, `glue`, `pacman`. Paths inside the scripts are hard-coded to
`~/Code/decision_project/…` — adjust before running.

```r
pacman::p_load(R2jags, parallel, polspline, ggplot2, glue)
source("sylvain_analysis_worry.R")
```

For new work, prefer the Python pipeline.

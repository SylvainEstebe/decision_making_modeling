"""Sampling + convergence diagnostics for the Conditional Cooperation models."""

from __future__ import annotations

from pathlib import Path

import arviz as az
import pymc as pm


def sample_posterior(
    model: pm.Model,
    *,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    cores: int | None = None,
    target_accept: float = 0.95,
    random_seed: int = 1983,
) -> az.InferenceData:
    """Run NUTS on a CC model and return an ArviZ ``InferenceData``.

    The defaults mirror the JAGS settings (multiple chains, generous tuning)
    but use HMC/NUTS instead of Gibbs/Metropolis.
    """
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=True,
        )
    return idata


def convergence_report(idata: az.InferenceData, var_names: list[str] | None = None) -> dict:
    """Return a compact dict of R-hat, ESS and divergences for the named vars."""
    summary = az.summary(idata, var_names=var_names, kind="diagnostics")
    divergences = int(idata.sample_stats["diverging"].sum().item())
    max_rhat = float(summary["r_hat"].max())
    min_ess_bulk = float(summary["ess_bulk"].min())
    min_ess_tail = float(summary["ess_tail"].min())
    return {
        "max_rhat": max_rhat,
        "min_ess_bulk": min_ess_bulk,
        "min_ess_tail": min_ess_tail,
        "divergences": divergences,
        "passes_rhat": max_rhat < 1.05,
        "passes_ess": min_ess_bulk > 400,
    }


def save_trace(idata: az.InferenceData, path: str | Path) -> Path:
    """Persist an InferenceData object to NetCDF for later analysis."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(str(path))
    return path

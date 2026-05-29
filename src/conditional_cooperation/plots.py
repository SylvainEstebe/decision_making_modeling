"""Plotting helpers for the Conditional Cooperation analysis.

Each function takes an ``arviz.InferenceData`` (and sometimes a
:class:`~conditional_cooperation.data.CCDataset`) and saves a publication-style
figure to ``results/figures/``. Matplotlib is the only hard dependency.
"""

from __future__ import annotations

from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_traces(
    idata: az.InferenceData,
    var_names: list[str],
    out_path: str | Path,
) -> Path:
    """Trace + posterior density plot for a list of scalar parameters."""
    az.plot_trace(idata, var_names=var_names, compact=True)
    plt.tight_layout()
    out = _ensure_dir(out_path)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_posterior(
    idata: az.InferenceData,
    var_names: list[str],
    out_path: str | Path,
    hdi_prob: float = 0.95,
) -> Path:
    """Forest/posterior summary plot."""
    az.plot_posterior(idata, var_names=var_names, hdi_prob=hdi_prob)
    plt.tight_layout()
    out = _ensure_dir(out_path)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_country_means(
    idata: az.InferenceData,
    nation_names: list[str],
    covariate: np.ndarray,
    out_path: str | Path,
    covariate_label: str = "National worry index",
) -> Path:
    """Posterior country-level mu_alpha / mu_rho / mu_omega vs covariate.

    Three side-by-side scatter plots with 95 % HDI error bars.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, var, ylabel in zip(
        axes,
        ["mu_alpha", "mu_rho", "mu_omega"],
        ["Initial belief mean (α)", "Matching preference (ρ)", "Belief update (ω)"],
        strict=True,
    ):
        post = idata.posterior[var].stack(sample=("chain", "draw"))
        mean = post.mean("sample").values
        hdi = az.hdi(idata.posterior[var], hdi_prob=0.95)[var].values  # (nation, 2)
        ax.errorbar(
            covariate,
            mean,
            yerr=np.abs(hdi.T - mean),
            fmt="o",
            color="#1f77b4",
            ecolor="#aaa",
            capsize=3,
            markersize=6,
        )
        for x, y, name in zip(covariate, mean, nation_names, strict=True):
            ax.annotate(name, (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(covariate_label)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    out = _ensure_dir(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_regression_slopes(
    idata: az.InferenceData,
    out_path: str | Path,
) -> Path:
    """Posterior densities of the national-level slopes ``betaX_*``.

    A vertical line at 0 marks the null effect; anything whose HDI excludes 0
    is a credible effect of the covariate.
    """
    var_names = ["betaX_alpha", "betaX_rho", "betaX_omega"]
    az.plot_forest(idata, var_names=var_names, combined=True, hdi_prob=0.95)
    plt.axvline(0, color="red", linestyle="--", alpha=0.5)
    plt.title("National-level slopes: effect of covariate on CC parameters")
    out = _ensure_dir(out_path)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_posterior_predictive(
    idata: az.InferenceData,
    observed: np.ndarray,
    out_path: str | Path,
) -> Path:
    """Compare observed vs predicted contributions per trial (mean ± 95 % HDI)."""
    # observed shape: (subject, trial, group)
    obs_per_trial = observed.mean(axis=(0, 2))
    pp = idata.posterior_predictive["c"]  # (chain, draw, subject, trial, group)
    pp_per_trial = pp.mean(dim=("subject", "group")).stack(sample=("chain", "draw"))
    pred_mean = pp_per_trial.mean("sample").values
    pred_hdi = az.hdi(pp.mean(dim=("subject", "group")), hdi_prob=0.95)["c"].values

    trials = np.arange(1, observed.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(trials, obs_per_trial, "o-", color="black", label="Observed", linewidth=2)
    ax.plot(trials, pred_mean, "s-", color="#1f77b4", label="Posterior predictive mean")
    ax.fill_between(
        trials, pred_hdi[:, 0], pred_hdi[:, 1], color="#1f77b4", alpha=0.25, label="95 % HDI"
    )
    ax.set_xlabel("Trial")
    ax.set_ylabel("Contribution (tokens)")
    ax.set_title("Conditional Cooperation: observed vs predicted")
    ax.legend()
    ax.grid(alpha=0.3)
    out = _ensure_dir(out_path)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out

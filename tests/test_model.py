"""Tests for the PyMC CC models (prior sampling + a tiny posterior smoke test)."""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest

from conditional_cooperation.model import (
    build_hierarchical_model,
    build_individual_model,
    standardise,
)
from conditional_cooperation.recovery import simulate_cc_dataset


def test_standardise_zero_mean_unit_sd() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    z = standardise(x)
    assert abs(z.mean()) < 1e-10
    assert abs(z.std(ddof=1) - 1.0) < 1e-10


def test_individual_model_prior_sample() -> None:
    sim = simulate_cc_dataset(ngroups=3, rng=np.random.default_rng(0))
    model = build_individual_model(sim["c"], sim["Ga"])
    with model:
        prior = pm.sample_prior_predictive(draws=5, random_seed=0)
    c_prior = prior.prior_predictive["c"]
    # dims = (chain, draw, subject, trial, group)
    assert c_prior.shape[-3:] == sim["c"].shape
    assert (c_prior.values >= 0).all()


def test_hierarchical_model_prior_sample() -> None:
    sim = simulate_cc_dataset(ngroups=6, rng=np.random.default_rng(0))
    nation = np.array([0, 1, 2, 0, 1, 2])
    X = standardise(np.array([0.2, 0.4, 0.5]))
    model = build_hierarchical_model(sim["c"], sim["Ga"], X, nation)
    with model:
        prior = pm.sample_prior_predictive(draws=3, random_seed=0)
    assert "mu_alpha" in prior.prior
    assert prior.prior["mu_alpha"].shape[-1] == 3  # nnations


def test_hierarchical_model_rejects_bad_nation() -> None:
    sim = simulate_cc_dataset(ngroups=3, rng=np.random.default_rng(0))
    nation = np.array([0, 1, 5])  # out of range
    X = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="nation"):
        build_hierarchical_model(sim["c"], sim["Ga"], X, nation)


@pytest.mark.slow
def test_individual_model_smoke_posterior() -> None:
    """A tiny end-to-end sample to make sure NUTS doesn't blow up."""
    sim = simulate_cc_dataset(ngroups=4, rng=np.random.default_rng(0))
    model = build_individual_model(sim["c"], sim["Ga"])
    with model:
        idata = pm.sample(
            draws=50,
            tune=100,
            chains=2,
            cores=1,
            random_seed=0,
            progressbar=False,
            return_inferencedata=True,
        )
    assert idata.posterior["alpha"].shape[:2] == (2, 50)

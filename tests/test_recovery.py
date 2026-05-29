"""Tests for the synthetic data simulator."""

from __future__ import annotations

import numpy as np

from conditional_cooperation.recovery import recovery_correlations, simulate_cc_dataset


def test_simulate_shapes() -> None:
    sim = simulate_cc_dataset(ngroups=5, rng=np.random.default_rng(0))
    assert sim["c"].shape == (4, 10, 5)
    assert sim["Ga"].shape == sim["c"].shape
    assert sim["alpha"].shape == (4, 5)
    assert (sim["c"] >= 0).all()


def test_simulate_is_deterministic_under_seed() -> None:
    a = simulate_cc_dataset(ngroups=3, rng=np.random.default_rng(7))
    b = simulate_cc_dataset(ngroups=3, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a["c"], b["c"])


def test_recovery_correlations_perfect_when_identical() -> None:
    sim = simulate_cc_dataset(ngroups=4, rng=np.random.default_rng(0))
    corr = recovery_correlations(sim, sim)
    assert all(abs(v - 1.0) < 1e-10 for v in corr.values())

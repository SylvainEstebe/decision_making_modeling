"""Parameter recovery for the Conditional Cooperation model.

Procedure: sample true (alpha, rho, omega) from priors, simulate contributions
forward through the generative process, fit the individual-level model on the
simulated data and check whether posterior means recover the true values.
"""

from __future__ import annotations

import numpy as np

from conditional_cooperation.data import GROUP_SIZE, N_TRIALS


def simulate_cc_dataset(
    *,
    ngroups: int = 20,
    ntrials: int = N_TRIALS,
    groupSize: int = GROUP_SIZE,
    alpha_prior: tuple[float, float] = (0.1, 0.1),
    rho_prior: tuple[float, float] = (1.0, 1.0),
    omega_prior: tuple[float, float] = (1.0, 1.0),
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Generate a synthetic CC dataset from the prior.

    Returns
    -------
    dict
        Keys: ``alpha``, ``rho``, ``omega`` (true values, each ``(groupSize, ngroups)``),
        ``c`` (contributions, ``(groupSize, ntrials, ngroups)``),
        ``Ga`` (group avg excluding self, same shape as ``c``).
    """
    rng = rng or np.random.default_rng(1983)

    alpha = rng.gamma(shape=alpha_prior[0], scale=1.0 / alpha_prior[1], size=(groupSize, ngroups))
    rho = rng.beta(rho_prior[0], rho_prior[1], size=(groupSize, ngroups))
    omega = rng.beta(omega_prior[0], omega_prior[1], size=(groupSize, ngroups))

    c = np.zeros((groupSize, ntrials, ngroups), dtype=np.int64)
    Gb = np.zeros((groupSize, ntrials, ngroups))
    Ga = np.zeros_like(Gb)

    # Trial 1: initial belief = alpha (its expectation), preference = rho * alpha
    Gb[:, 0, :] = alpha
    p0 = np.clip(rho * Gb[:, 0, :], 1e-6, None)
    c[:, 0, :] = rng.poisson(p0)

    for t in range(1, ntrials):
        # group average excluding self at the previous trial
        full_mean = c[:, t - 1, :].mean(axis=0)  # (ngroups,)
        for s in range(groupSize):
            others_mean = (c[:, t - 1, :].sum(axis=0) - c[s, t - 1, :]) / (groupSize - 1)
            Ga[s, t - 1, :] = others_mean
        # update beliefs
        Gb[:, t, :] = (1.0 - omega) * Gb[:, t - 1, :] + omega * Ga[:, t - 1, :]
        p_t = np.clip(rho * Gb[:, t, :], 1e-6, None)
        c[:, t, :] = rng.poisson(p_t)
        _ = full_mean  # kept for clarity; not used directly

    # Ga at trial t uses contributions at trial t-1; trial ntrials-1 Ga unused by model
    for s in range(groupSize):
        for t in range(ntrials - 1):
            others_mean = (c[:, t, :].sum(axis=0) - c[s, t, :]) / (groupSize - 1)
            Ga[s, t, :] = others_mean

    return {"alpha": alpha, "rho": rho, "omega": omega, "c": c, "Ga": Ga}


def recovery_correlations(
    true_values: dict[str, np.ndarray],
    posterior_means: dict[str, np.ndarray],
) -> dict[str, float]:
    """Pearson r between true and recovered values for each parameter."""
    correlations = {}
    for name in ("alpha", "rho", "omega"):
        t = true_values[name].ravel()
        p = posterior_means[name].ravel()
        correlations[name] = float(np.corrcoef(t, p)[0, 1])
    return correlations

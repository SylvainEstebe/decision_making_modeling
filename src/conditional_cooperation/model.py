"""PyMC port of the Conditional Cooperation (CC) model from ``legacy_r/``.

Two model builders are exposed:

* :func:`build_individual_model` — the basic CC model (matches ``CC_individual.txt``).
  Subject-level priors only; no national covariate.

* :func:`build_hierarchical_model` — the full regression model
  (matches ``CC_corr.txt``). National-level slopes regress
  ``(alpha, rho, omega)`` on a standardised covariate (worry or experience).

Design notes
------------

The JAGS code treats ``Gb[s, 1, g]`` — the latent belief about others on the
first trial — as a Poisson-distributed integer. PyMC's NUTS sampler does not
sample latent discrete variables; rather than mixing samplers, we replace the
trial-1 belief with its expectation ``alpha``. The downstream observed
contributions ``c`` remain Poisson, so the likelihood is unchanged in form and
only the prior on ``Gb[:, 0, :]`` is tightened (point mass at the prior mean).
In practice this barely shifts posteriors on ``(alpha, rho, omega)`` because
``c`` carries the signal, and it lets us use modern HMC.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor import scan


def build_individual_model(
    c_obs: np.ndarray,
    Ga: np.ndarray,
    coords: dict[str, list] | None = None,
) -> pm.Model:
    """Build the basic Conditional Cooperation model (no national covariate).

    Parameters
    ----------
    c_obs
        Observed contributions, shape ``(groupSize, ntrials, ngroups)``.
        Must be non-negative integers.
    Ga
        Group-averaged contribution excluding self (input to the belief update),
        shape ``(groupSize, ntrials, ngroups)``.

    Returns
    -------
    pm.Model
        A PyMC model with subject- and group-indexed parameters.
    """
    groupSize, ntrials, ngroups = c_obs.shape
    if Ga.shape != c_obs.shape:
        raise ValueError(f"Ga shape {Ga.shape} != c_obs shape {c_obs.shape}")

    coords = coords or {}
    coords = {
        "subject": list(range(groupSize)),
        "group": list(range(ngroups)),
        "trial": list(range(ntrials)),
        **coords,
    }

    with pm.Model(coords=coords) as model:
        # Subject × group level priors (matches JAGS CC_individual.txt)
        alpha = pm.Gamma("alpha", alpha=0.1, beta=0.1, dims=("subject", "group"))
        rho = pm.Beta("rho", alpha=1.0, beta=1.0, dims=("subject", "group"))
        omega = pm.Beta("omega", alpha=1.0, beta=1.0, dims=("subject", "group"))

        Gb_t1 = alpha  # see "Design notes" in the module docstring
        Ga_t = pt.as_tensor_variable(Ga)  # shape (subject, trial, group)

        # Belief recursion: Gb[t] = (1 - omega) * Gb[t-1] + omega * Ga[t-1]
        def belief_step(Ga_prev, Gb_prev, omega_):
            return (1.0 - omega_) * Gb_prev + omega_ * Ga_prev

        Gb_seq = scan(
            fn=belief_step,
            sequences=[Ga_t[:, :-1, :].dimshuffle(1, 0, 2)],  # (trial-1, subject, group)
            outputs_info=[Gb_t1],
            non_sequences=[omega],
            return_updates=False,
        )
        # Gb_seq has shape (ntrials-1, subject, group). Prepend trial 1.
        Gb_all = pt.concatenate([Gb_t1[None, :, :], Gb_seq], axis=0)  # (trial, subject, group)
        Gb_all = Gb_all.dimshuffle(1, 0, 2)  # (subject, trial, group)

        # Preference and likelihood
        p = pm.Deterministic("p", rho[:, None, :] * Gb_all, dims=("subject", "trial", "group"))
        # Poisson rate must be strictly positive
        rate = pt.clip(p, 1e-6, np.inf)
        pm.Poisson("c", mu=rate, observed=c_obs, dims=("subject", "trial", "group"))

    return model


def build_hierarchical_model(
    c_obs: np.ndarray,
    Ga: np.ndarray,
    X: np.ndarray,
    nation: np.ndarray,
) -> pm.Model:
    """Build the hierarchical CC model with a national-level covariate.

    Parameters
    ----------
    c_obs
        Observed contributions, shape ``(groupSize, ntrials, ngroups)``.
    Ga
        Group-averaged contribution excluding self,
        shape ``(groupSize, ntrials, ngroups)``.
    X
        Standardised national-level covariate (e.g. worry index z-scored),
        shape ``(nnations,)``.
    nation
        Per-group nation index (0-based, in ``range(nnations)``),
        shape ``(ngroups,)``.

    Returns
    -------
    pm.Model
        A PyMC model with national-level regressions on ``(alpha, rho, omega)``
        and subject-level priors drawn from the implied national distributions.
    """
    groupSize, ntrials, ngroups = c_obs.shape
    if Ga.shape != c_obs.shape:
        raise ValueError(f"Ga shape {Ga.shape} != c_obs shape {c_obs.shape}")
    nnations = int(X.shape[0])
    if nation.shape != (ngroups,):
        raise ValueError(f"nation shape {nation.shape} != (ngroups,)={(ngroups,)}")
    if nation.max() >= nnations or nation.min() < 0:
        raise ValueError("nation values must be 0-based indices into X")

    coords = {
        "subject": list(range(groupSize)),
        "group": list(range(ngroups)),
        "trial": list(range(ntrials)),
        "nation": list(range(nnations)),
    }

    X_t = pt.as_tensor_variable(X.astype(np.float64))
    nation_t = pt.as_tensor_variable(nation.astype(np.int64))

    with pm.Model(coords=coords) as model:
        # ─── National-level regression coefficients ─────────────────────────
        # Weakly informative N(0, 1) priors (the JAGS code used a JZS prior on
        # betaX; we keep a standard normal here — close enough for our N≈13).
        beta0_alpha = pm.Normal("beta0_alpha", mu=0.0, sigma=1.0)
        betaX_alpha = pm.Normal("betaX_alpha", mu=0.0, sigma=1.0)
        beta0_rho = pm.Normal("beta0_rho", mu=0.0, sigma=1.0)
        betaX_rho = pm.Normal("betaX_rho", mu=0.0, sigma=1.0)
        beta0_omega = pm.Normal("beta0_omega", mu=0.0, sigma=1.0)
        betaX_omega = pm.Normal("betaX_omega", mu=0.0, sigma=1.0)

        # National-level means via link functions (log for alpha, probit for rates)
        mu_alpha_log = beta0_alpha + betaX_alpha * X_t
        mu_alpha = pm.Deterministic("mu_alpha", pt.exp(mu_alpha_log), dims="nation")

        mu_rho_probit = beta0_rho + betaX_rho * X_t
        mu_rho = pm.Deterministic(
            "mu_rho",
            pt.clip(0.5 * (1.0 + pt.erf(mu_rho_probit / pt.sqrt(2.0))), 1e-4, 1 - 1e-4),
            dims="nation",
        )

        mu_omega_probit = beta0_omega + betaX_omega * X_t
        mu_omega = pm.Deterministic(
            "mu_omega",
            pt.clip(0.5 * (1.0 + pt.erf(mu_omega_probit / pt.sqrt(2.0))), 1e-4, 1 - 1e-4),
            dims="nation",
        )

        # National-level dispersions
        sigma_alpha = pm.Gamma("sigma_alpha", alpha=0.01, beta=0.01, dims="nation")
        kappa_rho = pm.Uniform("kappa_rho", lower=1.0, upper=100.0, dims="nation")
        kappa_omega = pm.Uniform("kappa_omega", lower=1.0, upper=100.0, dims="nation")

        # ─── Subject-level priors per group, indexed by its nation ──────────
        # Gamma reparameterisation: mean = shape / rate
        rate_alpha_n = (mu_alpha + pt.sqrt(mu_alpha**2 + 4.0 * sigma_alpha**2)) / (
            2.0 * sigma_alpha**2
        )
        shape_alpha_n = 1.0 + mu_alpha * rate_alpha_n

        alpha = pm.Gamma(
            "alpha",
            alpha=shape_alpha_n[nation_t][None, :],
            beta=rate_alpha_n[nation_t][None, :],
            dims=("subject", "group"),
        )

        shape1_rho_n = mu_rho * kappa_rho
        shape2_rho_n = (1.0 - mu_rho) * kappa_rho
        rho = pm.Beta(
            "rho",
            alpha=shape1_rho_n[nation_t][None, :],
            beta=shape2_rho_n[nation_t][None, :],
            dims=("subject", "group"),
        )

        shape1_omega_n = mu_omega * kappa_omega
        shape2_omega_n = (1.0 - mu_omega) * kappa_omega
        omega = pm.Beta(
            "omega",
            alpha=shape1_omega_n[nation_t][None, :],
            beta=shape2_omega_n[nation_t][None, :],
            dims=("subject", "group"),
        )

        # ─── Generative process (same as individual model) ──────────────────
        Gb_t1 = alpha
        Ga_t = pt.as_tensor_variable(Ga)

        def belief_step(Ga_prev, Gb_prev, omega_):
            return (1.0 - omega_) * Gb_prev + omega_ * Ga_prev

        Gb_seq = scan(
            fn=belief_step,
            sequences=[Ga_t[:, :-1, :].dimshuffle(1, 0, 2)],
            outputs_info=[Gb_t1],
            non_sequences=[omega],
            return_updates=False,
        )
        Gb_all = pt.concatenate([Gb_t1[None, :, :], Gb_seq], axis=0)
        Gb_all = Gb_all.dimshuffle(1, 0, 2)

        p = pm.Deterministic("p", rho[:, None, :] * Gb_all, dims=("subject", "trial", "group"))
        rate = pt.clip(p, 1e-6, np.inf)
        pm.Poisson("c", mu=rate, observed=c_obs, dims=("subject", "trial", "group"))

    return model


def standardise(x: np.ndarray) -> np.ndarray:
    """Z-score a 1-D array (mean 0, sd 1) — matches the R analysis pre-step."""
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / x.std(ddof=1)

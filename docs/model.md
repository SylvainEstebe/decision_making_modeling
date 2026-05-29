# The Conditional Cooperation (CC) model

A hierarchical Bayesian cognitive model of contributions in the repeated
linear Public Goods Game (PGG). Each subject's contribution at trial $t$ is a
Poisson sample around a preference that scales linearly with the subject's
belief about what the group will contribute.

## Generative process

Let $c_{s,t,g} \in \mathbb{Z}_{\ge 0}$ be the contribution of subject $s$ in
group $g$ at trial $t$, and $\bar{c}_{-s,t,g}$ the mean contribution of the
*other* group members at trial $t$.

Subject-level parameters:

- $\alpha_{s,g} \sim \mathrm{Gamma}(0.1, 0.1)$ — initial belief about others' contribution.
- $\rho_{s,g} \sim \mathrm{Beta}(1, 1)$ — *matching preference* (slope of contribution on belief).
- $\omega_{s,g} \sim \mathrm{Beta}(1, 1)$ — *belief-update weight* (recency on actual group behaviour).

Belief update:

$$
G^b_{s,1,g} = \alpha_{s,g},
\qquad
G^b_{s,t,g} = (1 - \omega_{s,g})\,G^b_{s,t-1,g} \;+\; \omega_{s,g}\,\bar{c}_{-s,t-1,g}
\quad (t \ge 2)
$$

Preference and likelihood:

$$
p_{s,t,g} = \rho_{s,g}\,G^b_{s,t,g},
\qquad
c_{s,t,g} \sim \mathrm{Poisson}(p_{s,t,g})
$$

## National-level regression (hierarchical model)

For each national-level parameter $\theta \in \{\alpha, \rho, \omega\}$ we fit
a linear regression on a standardised covariate $X_n$ (worry index or
experience-of-harm index from the Lloyd's Register 2021 World Risk Poll):

$$
\eta_{\theta,n} \;=\; \beta^0_\theta + \beta^X_\theta\, X_n,
\qquad
\beta^0_\theta \sim \mathcal{N}(0, 1),
\quad
\beta^X_\theta \sim \mathcal{N}(0, 1)
$$

with link functions $\mu_{\alpha,n} = \exp(\eta_{\alpha,n})$ (log link) and
$\mu_{\rho,n}, \mu_{\omega,n} = \Phi(\eta_{\theta,n})$ (probit link). Subject-
level priors are then drawn from gamma/beta distributions reparameterised so
that their mean equals $\mu_{\theta,n}$, with national-level dispersions
$\sigma_\alpha \sim \mathrm{Gamma}(0.01, 0.01)$ and
$\kappa_\rho, \kappa_\omega \sim \mathrm{Uniform}(1, 100)$.

## Differences from the JAGS implementation

The JAGS code in `legacy_r/` keeps the trial-1 belief $G^b_{s,1,g}$ as a
*latent Poisson integer*. PyMC's NUTS sampler cannot sample latent discrete
variables, so we set $G^b_{s,1,g} \equiv \alpha_{s,g}$ (its expectation). The
likelihood on the observed contributions is unchanged in form, and posteriors
on $(\alpha, \rho, \omega)$ are essentially identical at this sample size.

We also replaced the JZS (Zellner–Siow) prior on the regression slopes with a
weakly informative $\mathcal{N}(0, 1)$, which is more conventional for HMC and
adequate for $n_{\text{nations}} \approx 13$.

## References

- Herrmann, B., Thöni, C., & Gächter, S. (2008). *Antisocial Punishment Across Societies.* **Science**, 319(5868), 1362–1367. <https://doi.org/10.1126/science.1153808>
- Lloyd's Register Foundation. (2021). *World Risk Poll 2021 — Risk Indexes.* <https://wrp.lrfoundation.org.uk/2021-risk-indexes/>

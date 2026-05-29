"""Parameter recovery: simulate from prior, refit, check correlations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from conditional_cooperation.fit import sample_posterior
from conditional_cooperation.model import build_individual_model
from conditional_cooperation.recovery import recovery_correlations, simulate_cc_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ngroups", type=int, default=20)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1983)
    parser.add_argument("--out-dir", type=Path, default=Path("results/parameter_recovery"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    sim = simulate_cc_dataset(ngroups=args.ngroups, rng=rng)

    model = build_individual_model(sim["c"], sim["Ga"])
    idata = sample_posterior(
        model,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        random_seed=args.seed,
    )

    posterior_means = {
        "alpha": idata.posterior["alpha"].mean(("chain", "draw")).values,
        "rho": idata.posterior["rho"].mean(("chain", "draw")).values,
        "omega": idata.posterior["omega"].mean(("chain", "draw")).values,
    }
    correlations = recovery_correlations(sim, posterior_means)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "correlations.json").write_text(json.dumps(correlations, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, name in zip(axes, ("alpha", "rho", "omega"), strict=True):
        true = sim[name].ravel()
        recovered = posterior_means[name].ravel()
        ax.scatter(true, recovered, alpha=0.6, s=30)
        lo = min(true.min(), recovered.min())
        hi = max(true.max(), recovered.max())
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.6, label="y = x")
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Recovered {name}")
        ax.set_title(f"{name} (r = {correlations[name]:.2f})")
        ax.legend()
        ax.grid(alpha=0.3)

    fig_path = args.out_dir / "recovery.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    az.summary(idata, var_names=["alpha", "rho", "omega"]).to_csv(args.out_dir / "summary.csv")

    print(json.dumps(correlations, indent=2))
    print(f"Recovery figure saved to {fig_path}")


if __name__ == "__main__":
    main()

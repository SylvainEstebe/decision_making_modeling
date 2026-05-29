"""Fast demo fit: individual CC model on the full dataset, aggregated per nation.

Sidesteps the funnel geometry of ``build_hierarchical_model`` (which needs
non-centered reparameterisation to converge cleanly under NUTS) by fitting the
*per-subject* model and aggregating posterior means per nation in pure NumPy.
Produces the scatter plots ``α / ρ / ω vs covariate`` that appear in the README.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

from conditional_cooperation.data import load_public_goods_data
from conditional_cooperation.fit import convergence_report, sample_posterior, save_trace
from conditional_cooperation.model import build_individual_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/public_good/HerrmannThoeniGaechterDATA.csv"),
    )
    parser.add_argument("--covariate", choices=["worry", "experience"], default="worry")
    parser.add_argument("--results-dir", type=Path, default=Path("results/demo"))
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=500)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.9,
        help="Lower than the hierarchical default — the individual model is well-conditioned.",
    )
    parser.add_argument("--seed", type=int, default=1983)
    args = parser.parse_args()

    ds = load_public_goods_data(args.data, covariate=args.covariate)
    c = ds.c[:, :, :, 0]  # no-punishment condition
    Ga = ds.Ga[:, :, :, 0]
    print(f"Loaded {ds.ngroups} groups, {ds.nnations} nations.")

    model = build_individual_model(c, Ga)
    idata = sample_posterior(
        model,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        random_seed=args.seed,
    )

    out = args.results_dir / args.covariate
    out.mkdir(parents=True, exist_ok=True)
    save_trace(idata, out / "trace.nc")

    diag = convergence_report(idata, var_names=["alpha", "rho", "omega"])
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    print("Diagnostics:", json.dumps(diag, indent=2))

    # ── Aggregate per nation ──────────────────────────────────────────────────
    # Posterior means: shape (subject, group)
    alpha_mean = idata.posterior["alpha"].mean(("chain", "draw")).values
    rho_mean = idata.posterior["rho"].mean(("chain", "draw")).values
    omega_mean = idata.posterior["omega"].mean(("chain", "draw")).values

    # Average across subjects and groups within each nation
    nations = np.unique(ds.nation)
    per_nation = {}
    for stat_name, mat in (("alpha", alpha_mean), ("rho", rho_mean), ("omega", omega_mean)):
        per_nation[stat_name] = np.array(
            [mat[:, ds.nation == n].mean() for n in nations]
        )

    # ── Scatter: covariate vs nation-level mean of each parameter ────────────
    cov_label = {
        "worry": "National worry index (Lloyd's Register 2021)",
        "experience": "National experience-of-harm index (Lloyd's Register 2021)",
    }[args.covariate]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, name, ylabel, color in zip(
        axes,
        ("alpha", "rho", "omega"),
        (
            "Initial belief mean (α)",
            "Matching preference (ρ)",
            "Belief update weight (ω)",
        ),
        ("#1f77b4", "#2ca02c", "#d62728"),
        strict=True,
    ):
        y = per_nation[name]
        x = ds.covariate
        ax.scatter(x, y, s=80, color=color, edgecolor="black", alpha=0.85)
        # Pearson correlation
        r = float(np.corrcoef(x, y)[0, 1])
        # Annotate each point with the nation name
        for xi, yi, lbl in zip(x, y, ds.nation_names, strict=True):
            ax.annotate(lbl, (xi, yi), xytext=(5, 4), textcoords="offset points", fontsize=8)
        # Linear fit overlay
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, "--", color="gray", alpha=0.6)
        ax.set_xlabel(cov_label)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel.split()[0]} vs {args.covariate}   (r = {r:+.2f})")
        ax.grid(alpha=0.3)

    fig_path = out / "figures" / f"{args.covariate}_per_nation.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-nation scatter saved to {fig_path}")

    # Posterior summary table
    summary = az.summary(idata, var_names=["alpha", "rho", "omega"])
    summary.to_csv(out / "posterior_summary.csv")
    print(f"Posterior summary saved to {out / 'posterior_summary.csv'}")


if __name__ == "__main__":
    main()

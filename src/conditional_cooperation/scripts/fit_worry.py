"""Fit the hierarchical CC model with the *worry* covariate."""

from __future__ import annotations

import argparse
import json

from conditional_cooperation.data import load_public_goods_data
from conditional_cooperation.fit import convergence_report, sample_posterior, save_trace
from conditional_cooperation.model import build_hierarchical_model, standardise
from conditional_cooperation.plots import (
    plot_country_means,
    plot_regression_slopes,
    plot_traces,
)
from conditional_cooperation.scripts._common import add_common_args


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_label="worry")
    args = parser.parse_args()

    ds = load_public_goods_data(args.data, covariate="worry")
    X = standardise(ds.covariate)
    # No-punishment condition is the one analysed in the original paper
    c = ds.c[:, :, :, 0]
    Ga = ds.Ga[:, :, :, 0]

    model = build_hierarchical_model(c, Ga, X, ds.nation)
    idata = sample_posterior(
        model,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        random_seed=args.seed,
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    save_trace(idata, args.results_dir / "trace.nc")

    diag = convergence_report(
        idata,
        var_names=[
            "beta0_alpha",
            "beta0_rho",
            "beta0_omega",
            "betaX_alpha",
            "betaX_rho",
            "betaX_omega",
        ],
    )
    (args.results_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2))

    figures = args.results_dir / "figures"
    plot_traces(
        idata,
        var_names=["betaX_alpha", "betaX_rho", "betaX_omega"],
        out_path=figures / "trace_slopes.png",
    )
    plot_regression_slopes(idata, out_path=figures / "regression_slopes.png")
    plot_country_means(
        idata,
        nation_names=ds.nation_names,
        covariate=ds.covariate,
        out_path=figures / "country_means.png",
        covariate_label="National worry index (Lloyd's Register 2021)",
    )

    print(f"Done. Trace saved to {args.results_dir / 'trace.nc'}")
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()

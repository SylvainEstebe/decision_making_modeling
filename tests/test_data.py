"""Tests for the data loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conditional_cooperation.data import (
    EXPERIENCE_INDEX_2021,  # noqa: F401  (used in test_index_tables_consistent)
    GROUP_SIZE,
    N_TRIALS,
    NATION_INDEX,
    WORRY_INDEX_2021,
    load_public_goods_data,
)

DATA_CSV = Path("data/public_good/HerrmannThoeniGaechterDATA.csv")


@pytest.mark.skipif(not DATA_CSV.exists(), reason="raw data CSV not present")
def test_load_shapes_worry() -> None:
    ds = load_public_goods_data(DATA_CSV, covariate="worry")
    assert ds.c.shape[:2] == (GROUP_SIZE, N_TRIALS)
    assert ds.c.shape[3] == 2
    assert ds.Ga.shape == ds.c.shape
    assert ds.nation.shape == (ds.ngroups,)
    assert ds.covariate.shape == (ds.nnations,)
    assert ds.c.dtype == np.int64
    assert (ds.c >= 0).all(), "contributions must be non-negative"
    assert ds.nation.min() >= 0
    assert ds.nation.max() < ds.nnations


@pytest.mark.skipif(not DATA_CSV.exists(), reason="raw data CSV not present")
def test_load_shapes_experience() -> None:
    ds = load_public_goods_data(DATA_CSV, covariate="experience")
    assert ds.c.shape[:2] == (GROUP_SIZE, N_TRIALS)
    assert ds.covariate.shape == (ds.nnations,)
    # nnations is at most the number of distinct nation IDs (≤ 13)
    assert ds.nnations <= len(set(NATION_INDEX.values()))


def test_index_tables_consistent() -> None:
    """NATION_INDEX must cover every city listed in the worry/experience tables."""
    assert set(NATION_INDEX) == set(WORRY_INDEX_2021)
    assert set(NATION_INDEX) == set(EXPERIENCE_INDEX_2021)


def test_invalid_covariate_raises() -> None:
    with pytest.raises(ValueError, match="covariate"):
        load_public_goods_data(DATA_CSV, covariate="bogus")  # type: ignore[arg-type]

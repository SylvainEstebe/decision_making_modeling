"""Load and preprocess the Herrmann, Thöni & Gächter (2008) Public Goods Game data.

Reshapes the long-format CSV into ``(groupSize, ntrials, ngroups)`` arrays matching
the JAGS code in ``legacy_r/`` and attaches national-level worry / experience
indices from the Lloyd's Register Foundation 2021 World Risk Poll.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Project constants ────────────────────────────────────────────────────────
GROUP_SIZE: int = 4
N_TRIALS: int = 10
N_TOKENS: int = 20
MULTIPLIER: float = 1.6  # public-good multiplier π in the original paper

# Maps cities to a nation index. St. Gallen and Zurich share index 9 (Switzerland).
NATION_INDEX: dict[str, int] = {
    "Melbourne": 1,
    "Minsk": 2,
    "Chengdu": 3,
    "Copenhagen": 4,
    "Bonn": 5,
    "Athens": 6,
    "Seoul": 7,
    "Samara": 8,
    "Zurich": 9,
    "St. Gallen": 9,
    "Istanbul": 10,
    "Nottingham": 11,
    "Dnipropetrovs'k": 12,
    "Boston": 13,
}

# 2021 World Risk Poll worry index — higher = more worried about harm.
WORRY_INDEX_2021: dict[str, float] = {
    "Melbourne": 0.30,
    "Minsk": 0.30,
    "Chengdu": 0.31,
    "Copenhagen": 0.23,
    "Bonn": 0.37,
    "Athens": 0.47,
    "Seoul": 0.47,
    "Samara": 0.40,
    "Zurich": 0.32,
    "St. Gallen": 0.32,
    "Istanbul": 0.49,
    "Nottingham": 0.30,
    "Dnipropetrovs'k": 0.41,
    "Boston": 0.35,
}

# 2021 World Risk Poll experience index — higher = more actual harm experienced.
EXPERIENCE_INDEX_2021: dict[str, float] = {
    "Melbourne": 0.18,
    "Minsk": 0.08,
    "Chengdu": 0.09,
    "Copenhagen": 0.13,
    "Bonn": 0.18,
    "Athens": 0.16,
    "Seoul": 0.11,
    "Samara": 0.14,
    "Zurich": 0.17,
    "St. Gallen": 0.17,
    "Istanbul": 0.18,
    "Nottingham": 0.17,
    "Dnipropetrovs'k": 0.14,
    "Boston": 0.21,
}


@dataclass(frozen=True)
class CCDataset:
    """Reshaped Public Goods Game data ready for the Conditional Cooperation model.

    Attributes
    ----------
    c
        Contributions, shape ``(groupSize, ntrials, ngroups, 2)``.
        Last axis: 0 = no-punishment, 1 = punishment condition.
    Ga
        Group-averaged contribution **excluding** subject ``s``,
        shape ``(groupSize, ntrials, ngroups, 2)``.
    Gga
        Group-averaged contribution (whole group),
        shape ``(ntrials, ngroups, 2)``.
    nation
        Nation index per group, shape ``(ngroups,)``, 1-based to match the R code.
    covariate
        National-level covariate (worry or experience) aggregated to nation,
        shape ``(nnations,)``.
    nation_names
        Human-readable nation labels aligned with ``covariate``.
    winnings
        Per-group payoff (sum of contributions × multiplier), shape ``(ngroups,)``.
    """

    c: np.ndarray
    Ga: np.ndarray
    Gga: np.ndarray
    nation: np.ndarray
    covariate: np.ndarray
    nation_names: list[str]
    winnings: np.ndarray

    @property
    def ngroups(self) -> int:
        return self.c.shape[2]

    @property
    def nnations(self) -> int:
        return self.covariate.shape[0]


def _city_to_nation(df: pd.DataFrame) -> pd.Series:
    return df["city"].map(NATION_INDEX).astype("Int64")


def _city_to_score(df: pd.DataFrame, score: dict[str, float]) -> pd.Series:
    return df["city"].map(score).astype("float64")


def load_public_goods_data(
    csv_path: str | Path,
    covariate: str = "worry",
) -> CCDataset:
    """Load the Herrmann, Thöni & Gächter (2008) CSV and reshape for the CC model.

    Parameters
    ----------
    csv_path
        Path to ``HerrmannThoeniGaechterDATA.csv``.
    covariate
        ``"worry"`` or ``"experience"`` — which national-level index to attach.

    Returns
    -------
    CCDataset
        Reshaped data, with groups missing the chosen covariate dropped.
    """
    if covariate not in {"worry", "experience"}:
        raise ValueError(f"covariate must be 'worry' or 'experience', got {covariate!r}")

    raw = pd.read_csv(csv_path, skiprows=3)

    # The raw file repeats each subject-period row 3 times (one per other group
    # member). The R code takes every 3rd row starting at 1 — keep only rows where
    # subjectid == otherscontribution row's owner. Simpler equivalent: take every
    # 3rd row.
    red = raw.iloc[::3].reset_index(drop=True)
    red = red.assign(
        nation=_city_to_nation(red),
        worry=_city_to_score(red, WORRY_INDEX_2021),
        experience=_city_to_score(red, EXPERIENCE_INDEX_2021),
    )

    group_names = red["groupid"].unique()
    ngroups = len(group_names)

    c_arr = np.zeros((GROUP_SIZE, N_TRIALS, ngroups, 2), dtype=np.int64)
    Gga = np.zeros((N_TRIALS, ngroups, 2), dtype=np.float64)
    Ga = np.zeros((GROUP_SIZE, N_TRIALS, ngroups, 2), dtype=np.float64)
    nation = np.zeros(ngroups, dtype=np.int64)
    cov_per_group = np.full(ngroups, np.nan)
    missing = np.zeros(ngroups, dtype=bool)

    conditions = {0: "N-experiment", 1: "P-experiment"}

    for g_idx, gid in enumerate(group_names):
        group_rows = red[red["groupid"] == gid]
        # all rows of a group share the same nation/covariate
        nation[g_idx] = (
            group_rows["nation"].iloc[0] if not pd.isna(group_rows["nation"].iloc[0]) else 0
        )
        cov_per_group[g_idx] = group_rows[covariate].iloc[0]

        for cond_idx, cond_name in conditions.items():
            sub = group_rows[group_rows["p"] == cond_name]["senderscontribution"].to_numpy()
            if sub.size < GROUP_SIZE * N_TRIALS:
                # missing data for this group/condition — flag and skip
                if cond_idx == 0:
                    missing[g_idx] = True
                continue
            c_arr[:, :, g_idx, cond_idx] = sub[: GROUP_SIZE * N_TRIALS].reshape(
                GROUP_SIZE, N_TRIALS
            )

        Gga[:, g_idx, :] = c_arr[:, :, g_idx, :].mean(axis=0)
        for s in range(GROUP_SIZE):
            others = np.delete(c_arr[:, :, g_idx, :], s, axis=0)
            Ga[s, :, g_idx, :] = others.mean(axis=0)

    # Drop groups with missing covariate or no-punish data
    keep = (~missing) & (~np.isnan(cov_per_group)) & (nation > 0)
    c_arr = c_arr[:, :, keep, :]
    Gga = Gga[:, keep, :]
    Ga = Ga[:, :, keep, :]
    nation = nation[keep]
    cov_per_group = cov_per_group[keep]

    # Aggregate covariate to nation-level (one value per nation)
    unique_nations = np.unique(nation)
    nation_names = [
        next(city for city, idx in NATION_INDEX.items() if idx == n) for n in unique_nations
    ]
    cov_by_nation = np.array(
        [cov_per_group[nation == n].mean() for n in unique_nations], dtype=np.float64
    )

    # Re-index nation to 0..nnations-1 for downstream PyMC indexing
    nation_remap = {n: i for i, n in enumerate(unique_nations)}
    nation0 = np.array([nation_remap[n] for n in nation], dtype=np.int64)

    # Winnings = sum of contributions × multiplier (no-punish condition)
    winnings = c_arr[:, :, :, 0].sum(axis=(0, 1)) * MULTIPLIER

    return CCDataset(
        c=c_arr,
        Ga=Ga,
        Gga=Gga,
        nation=nation0,
        covariate=cov_by_nation,
        nation_names=nation_names,
        winnings=winnings,
    )

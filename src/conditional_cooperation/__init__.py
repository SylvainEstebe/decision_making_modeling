"""Bayesian Conditional Cooperation model — PyMC port of a JAGS analysis.

A hierarchical model of contributions in the Public Goods Game
(Herrmann, Thöni & Gächter, 2008), regressed on national insecurity indices
(worry / experience of harm) from the Lloyd's Register Foundation World Risk Poll.
"""

from conditional_cooperation.data import (
    EXPERIENCE_INDEX_2021,
    NATION_INDEX,
    WORRY_INDEX_2021,
    CCDataset,
    load_public_goods_data,
)
from conditional_cooperation.model import (
    build_hierarchical_model,
    build_individual_model,
)

__all__ = [
    "CCDataset",
    "EXPERIENCE_INDEX_2021",
    "NATION_INDEX",
    "WORRY_INDEX_2021",
    "build_hierarchical_model",
    "build_individual_model",
    "load_public_goods_data",
]

__version__ = "0.1.0"

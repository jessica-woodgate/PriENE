import types

import matplotlib
matplotlib.use("Agg")  # headless-safe backend, must be set before data_analysis imports pyplot

import pandas as pd
import pytest

from src.harvest_model import HarvestModel
from src.data_handling.data_analysis import DataAnalysis


def make_model_stub(berries_consumed_values):
    """A duck-typed stand-in for HarvestModel exposing just what _gini_berries_consumed needs,
    so the calculation can be tested without booting a real Mesa model/grid."""
    schedule_agents = [
        types.SimpleNamespace(agent_type="baseline", berries_consumed=v)
        for v in berries_consumed_values
    ]
    return types.SimpleNamespace(
        living_agents=list(schedule_agents),  # only needs to be non-empty
        schedule=types.SimpleNamespace(agents=schedule_agents),
        num_agents=len(berries_consumed_values),
    )


GINI_CASES = [
    ([5, 5, 5, 5], 0.0),     # perfect equality
    ([0, 0, 0, 0], 0.0),     # all-zero edge case (must avoid division by zero)
    ([0, 0, 0, 10], 0.75),   # one agent has everything: (N-1)/N = 3/4
]


@pytest.mark.parametrize("values,expected_gini", GINI_CASES)
def test_harvest_model_gini_berries_consumed(values, expected_gini):
    stub = make_model_stub(values)
    assert HarvestModel._gini_berries_consumed(stub) == pytest.approx(expected_gini)


@pytest.mark.parametrize("values,expected_gini", GINI_CASES)
def test_data_analysis_calculate_gini(values, expected_gini):
    da = DataAnalysis(num_agents=len(values), filepath="")
    assert da._calculate_gini(pd.Series(values)) == pytest.approx(expected_gini)

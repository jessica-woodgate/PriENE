import matplotlib
matplotlib.use("Agg")  # headless-safe backend, must be set before data_analysis imports pyplot

import numpy as np
import pandas as pd
import pytest

from src.data_handling.data_analysis import DataAnalysis


def make_three_group_df(seed=0, effect_size=5.0):
    rng = np.random.default_rng(seed)
    df_a = pd.DataFrame({"value": rng.normal(0, 1, 30)})
    df_b = pd.DataFrame({"value": rng.normal(effect_size, 1, 30)})
    df_c = pd.DataFrame({"value": rng.normal(0, 1, 30)})
    return [df_a, df_b, df_c]


def test_perform_anova_returns_five_values_on_success():
    da = DataAnalysis(num_agents=4, filepath="")
    result = da._perform_anova(make_three_group_df(), ["a", "b", "c"], "value")
    assert len(result) == 5
    anova_table, tukey_results, anova, tukey, cohens_results = result
    assert anova is True
    assert tukey is True
    assert anova_table is not None
    assert tukey_results is not None
    assert cohens_results is not None


def test_perform_anova_detects_significant_difference():
    da = DataAnalysis(num_agents=4, filepath="")
    dfs = make_three_group_df(effect_size=10.0)
    anova_table, *_ = da._perform_anova(dfs, ["a", "b", "c"], "value")
    assert anova_table["PR(>F)"].iloc[0] < 0.05


def test_perform_anova_returns_five_values_when_posthoc_test_fails(monkeypatch):
    """
    Regression test for a bug where the post-hoc-test exception branch returned 6 values instead
    of 5, so `anova_table, tukey_results, anova, tukey, cohens_results = ...` in the caller raised
    ValueError: too many values to unpack. Forces that branch via monkeypatching rather than relying
    on naturally triggering a statsmodels internal edge case.
    """
    da = DataAnalysis(num_agents=4, filepath="")

    def broken_posthoc_test(*args, **kwargs):
        raise ValueError("simulated post-hoc test failure")

    monkeypatch.setattr("src.data_handling.data_analysis.pairwise_tukeyhsd", broken_posthoc_test)

    result = da._perform_anova(make_three_group_df(), ["a", "b", "c"], "value")
    assert len(result) == 5
    anova_table, tukey_results, anova, tukey, cohens_results = result  # must not raise
    assert anova is True
    assert tukey is False
    assert tukey_results is None
    assert cohens_results == 0


def test_perform_anova_returns_five_values_when_anova_itself_fails():
    da = DataAnalysis(num_agents=4, filepath="")
    # a single row per group makes the OLS fit degenerate (no residual degrees of freedom)
    dfs = [pd.DataFrame({"value": [1.0]}), pd.DataFrame({"value": [2.0]})]
    result = da._perform_anova(dfs, ["a", "b"], "value")
    assert len(result) == 5
    anova_table, tukey_results, anova, tukey, cohens_results = result  # must not raise
    assert anova is False
    assert tukey is False
    assert anova_table is None
    assert cohens_results is None


def test_compute_pairwise_cohens_d():
    da = DataAnalysis(num_agents=4, filepath="")
    # within-group variance is needed so the pooled std isn't zero (see the dedicated test below
    # for that edge case) -- both groups have std=1, means 0 and 10
    combined_df = pd.DataFrame({
        "value": [0, 1, -1, 10, 11, 9],
        "society": ["a", "a", "a", "b", "b", "b"],
    })
    result_df = da._compute_pairwise_cohens_d(combined_df, "value")
    assert len(result_df) == 1
    row = result_df.iloc[0]
    assert {row["group1"], row["group2"]} == {"a", "b"}
    assert row["cohens_d"] == pytest.approx(-10.0)  # group "a" (mean 0) minus group "b" (mean 10)


def test_cohens_d_zero_pooled_std_returns_zero():
    da = DataAnalysis(num_agents=4, filepath="")
    x = pd.Series([5.0, 5.0, 5.0])
    y = pd.Series([5.0, 5.0, 5.0])
    assert da._cohens_d(x, y) == 0.0


def test_cohens_d_known_value():
    da = DataAnalysis(num_agents=4, filepath="")
    x = pd.Series([1.0, 2.0, 3.0])
    y = pd.Series([2.0, 3.0, 4.0])
    # equal variances and a mean difference of -1, pooled std = 1 -> cohen's d = -1
    assert da._cohens_d(x, y) == pytest.approx(-1.0)

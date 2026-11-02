import copy
import os

import numpy as np
import pandas as pd
import pytest

from src.checkpoints import society_checkpoint_path, record_agent_types, training_checkpoint_name
from src.harvest_exception import (CheckpointMismatchException, CommonsGridException, CommonsParameterException,
                                   RegrowthDelaysNotSupported)
from src.scenarios.commons_harvest import CommonsHarvest
from tests.conftest import GRID, NUM_AGENTS, SCENARIOS, _build, _make_pretrained_checkpoint, long_lived

K = 48
R = 0.25
WIDTH, HEIGHT = 8, 8


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def build(checkpoint_path, training=True, seasons=False, society_mix="homogeneous", max_days=10, max_episodes=2, write_data=False,
          track_norms=False, norms_module_type="rule_based", filepath="commons", start_episode=1, carrying_capacity=K, growth_rate=R,
          initial_stock=None, seed=42, agent_type="utilitarian"):
    model = CommonsHarvest(society_mix, NUM_AGENTS, carrying_capacity, growth_rate, agent_type, WIDTH, HEIGHT, max_episodes, max_days,
                           training, checkpoint_path, write_data, track_norms, filepath, seed, norms_module_type, seasons, 2,
                           start_episode=start_episode, initial_stock=initial_stock)
    return long_lived(model)


def agents(model):
    return sorted((a for a in model.schedule.agents if a.agent_type != "berry"), key=lambda a: a.unique_id)


def stock_on_grid(model):
    return sum(1 for b in model.berries if not b.foraged and b.pos is not None)


@pytest.fixture
def model(in_tmp_dir):
    return build(str(in_tmp_dir / "checkpoints") + "/")


# ── the existing scenarios are unchanged ──────────────────────────

@pytest.fixture(scope="module")
def basic_checkpoint(tmp_path_factory):
    return _make_pretrained_checkpoint(tmp_path_factory.mktemp("commons"), "baseline")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_existing_scenarios_observe_the_same_features(in_tmp_dir, basic_checkpoint, scenario):
    model = _build(scenario, "homogeneous", NUM_AGENTS * 3, "baseline", False, basic_checkpoint, False, False, scenario, 1, 5, seed=42)
    assert model.num_extra_observation_features() == 0 and model.extra_observation_feature_specs() == []
    for agent in agents(model):
        observation = agent.observe()
        assert agent.n_features == len(observation) == 4 + NUM_AGENTS - 1
        assert list(observation[:4]) == [agent.health, agent.berries, agent.days_left_to_live, agent.moving_module.get_distance_to_berry()]
        assert list(observation[4:]) == list(model.get_society_well_being(agent, False, False))


@pytest.mark.parametrize("norms_module_type", ["rule_based", "bayesian"])
def test_existing_scenarios_norms_have_no_stock_feature(in_tmp_dir, basic_checkpoint, norms_module_type):
    model = _build("colours", "homogeneous", NUM_AGENTS * 3, "baseline", False, basic_checkpoint, False, True, "colours", 1, 5, seed=42,
                   norms_module_type=norms_module_type)
    agent = agents(model)[0]
    if norms_module_type == "rule_based":
        assert len(agent.norms_module.antecedent_features) == 3
    else:
        assert [spec["name"] for spec in agent.norms_module.feature_specs] == ["health", "berries", "self_wellbeing", "distance", "other_wellbeing"]


# ── observing the stock ───────────────────────────────────────────

def test_agents_observe_the_stock_as_a_fraction_of_capacity_after_the_other_features(model):
    for agent in agents(model):
        observation = agent.observe()
        assert agent.n_features == len(observation) == 4 + NUM_AGENTS - 1 + 1
        assert list(observation[:4]) == [agent.health, agent.berries, agent.days_left_to_live, agent.moving_module.get_distance_to_berry()]
        assert list(observation[4:-1]) == list(model.get_society_well_being(agent, False, False))
        assert observation[-1] == pytest.approx(model.stock() / K) == pytest.approx(0.5)


def test_the_stock_is_counted_live_as_agents_forage_within_a_step(model):
    berry = model.berries[0]
    berry.foraged = True
    assert agents(model)[0].observe()[-1] == pytest.approx((K // 2 - 1) / K)


def test_a_checkpoint_from_another_scenario_fails_clearly(in_tmp_dir, basic_checkpoint):
    with pytest.raises(CheckpointMismatchException, match=r"\(7 inputs, 5 actions\), but this run has 4 agents \(8 inputs, 5 actions\).*commons"):
        build(basic_checkpoint, training=False, agent_type="baseline")


def test_commons_checkpoint_names_hold_the_scenario_and_its_parameters():
    name = training_checkpoint_name("commons", True, 25, 1, None, 200, 42, commons_parameters={"carrying_capacity": 48, "growth_rate": 0.25, "initial_stock": 24})
    assert name == "commons_seasons25_K48_r0.25_R024_200days_seed42"
    with pytest.raises(ValueError):
        training_checkpoint_name("commons", False, 25, 1, None, 200, 42)


# ── regrowth ──────────────────────────────────────────────────────

@pytest.mark.parametrize("stock", [1, 5, 12, 24, 36, 47, 48])
def test_regrowth_is_logistic_with_random_rounding_and_never_above_capacity(model, stock):
    rng = copy.deepcopy(model.regrowth_rng)
    growth = R * stock * (1 - stock / K)
    expected = int(np.floor(growth)) + (1 if rng.random() < growth - np.floor(growth) else 0)
    assert model._regrowth(stock) == min(expected, K - stock)


def test_random_rounding_matches_the_fractional_part_on_average(model):
    # R = 10 grows by 0.25 * 10 * (1 - 10 / 48) = 1.979...: 1 or 2 berries, 2 with probability 0.979...
    draws = [model._regrowth(10) for _ in range(5000)]
    assert set(draws) == {1, 2}
    assert np.mean(draws) == pytest.approx(R * 10 * (1 - 10 / K), abs=0.02)


def test_the_stock_never_exceeds_capacity_and_regrows_at_free_cells(in_tmp_dir):
    model = build(str(in_tmp_dir / "checkpoints") + "/", max_days=60, initial_stock=K - 2, growth_rate=3.0)
    for _ in range(40):
        model.step()
        assert model.stock() <= K
        cells = [b.pos for b in model.berries]
        assert len(cells) == len(set(cells))


def test_nothing_regrows_once_the_commons_has_collapsed(model):
    for berry in list(model.berries):
        model._remove_berry(berry)
    assert model.stock() == 0
    for _ in range(3):
        model.step()
    assert model.stock() == 0 and model.collapse_day == 1
    assert [row[4] for row in model.commons_steps] == [0, 0, 0]


def test_foraged_berries_leave_the_stock_and_are_reused_when_it_regrows(model):
    berry = model.berries[0]
    berry.foraged = True
    model._handle_foraged_berry(berry)
    assert berry not in model.berries and berry.pos is None and berry in model.dormant_berries
    assert berry not in model.schedule.agents
    assert model.stock() == K // 2 - 1 and model.harvested_this_step == 1
    made = model.berry_id
    model._end_of_step_dynamics()
    # regrowth reuses dormant berries before making new ones
    assert berry in model.berries and berry.pos is not None and not berry.foraged
    assert model.berry_id == made + max(0, model.commons_steps[-1][4] - 1)


def test_regrowth_has_its_own_random_stream(in_tmp_dir):
    first = build(str(in_tmp_dir / "a") + "/")
    second = build(str(in_tmp_dir / "b") + "/")
    # the same seed gives the same regrowth draws, from a stream of its own
    assert first.regrowth_rng.random() == second.regrowth_rng.random() == np.random.default_rng([42, CommonsHarvest.COMMONS_REGROWTH_STREAM]).random()
    # a continued run reseeds it from the seed and the episode it continues from
    first.episode = 2
    first.reseed_for_continuing()
    assert first.regrowth_rng.random() == np.random.default_rng([42, CommonsHarvest.COMMONS_REGROWTH_STREAM, 2]).random()


# ── seasons and episodes ──────────────────────────────────────────

@pytest.mark.parametrize("quality,multiplier", [("normal", 1.0), ("good", 1.5), ("bad", 0.5)])
def test_seasons_change_the_growth_rate_not_the_berries(in_tmp_dir, quality, multiplier):
    model = build(str(in_tmp_dir / "checkpoints") + "/", seasons=True)
    berries = len(model.berries)
    model.current_season_quality = quality
    model._apply_season_quality()
    assert model.current_growth_rate == pytest.approx(R * multiplier)
    assert CommonsHarvest.GROWTH_RATE_MULTIPLIERS[quality] == multiplier
    assert len(model.berries) == berries


def test_each_episode_starts_with_the_initial_stock_and_its_own_seasons(in_tmp_dir):
    model = build(str(in_tmp_dir / "checkpoints") + "/", seasons=True, max_days=6)
    while model.episode <= 3:
        episode = model.episode
        model.step()
        if model.episode != episode:
            assert model.stock() == stock_on_grid(model) == len(model.berries) == K // 2
            assert model.season == 0 and model.collapse_day is None and model.commons_steps == []
            assert model.current_growth_rate == pytest.approx(R * CommonsHarvest.GROWTH_RATE_MULTIPLIERS[model.current_season_quality])
            cells = [b.pos for b in model.berries]
            assert len(cells) == len(set(cells))


def test_without_seasons_the_growth_rate_is_restored_each_episode(in_tmp_dir):
    model = build(str(in_tmp_dir / "checkpoints") + "/", max_days=3)
    model.current_growth_rate = 9.0
    while model.episode <= 1:
        model.step()
    assert model.current_growth_rate == R and model._current_season_quality() is None


# ── norms ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("norms_module_type", ["rule_based", "bayesian"])
def test_norms_can_use_the_stock_as_a_precondition(in_tmp_dir, basic_checkpoint, norms_module_type):
    model = build(str(in_tmp_dir / "checkpoints") + "/", track_norms=True, norms_module_type=norms_module_type, max_days=6, write_data=True)
    agent = agents(model)[0]
    if norms_module_type == "rule_based":
        assert agent.norms_module.antecedent_features[-1]["labels"] == CommonsHarvest.STOCK_FEATURE_SPEC["labels"]
    else:
        stock_spec = agent.norms_module.feature_specs[-1]
        assert stock_spec["name"] == "stock" and stock_spec["index"] == agent.n_features - 1
        assert [spec["index"] for spec in agent.norms_module.feature_specs[:4]] == [0, 1, 2, 3]
    # within the episode, before the norm bases are reset for the next
    for _ in range(4):
        model.step()
    if norms_module_type == "rule_based":
        behaviours = [b for a in agents(model) for b in a.norms_module.behaviour_base]
        assert behaviours and all("stock" in b for b in behaviours)
    else:
        assert any("stock" in p["name"] for p in agent.norms_module._predicates)
        observed = [view for view in agent.norms_module.iter_behaviours() if agent.norms_module.behaviour_counts(*view)[0] > 0]
        assert any("stock" in name for view in observed for name in view[0])
    while model.episode <= 2:
        model.step()


# ── parameters ────────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs,exception,match", [
    ({"growth_rate": 0}, CommonsParameterException, "growth rate must be more than 0"),
    ({"growth_rate": -0.1}, CommonsParameterException, "growth rate"),
    ({"initial_stock": 0}, CommonsParameterException, "initial stock must be at least 1"),
    ({"initial_stock": K + 1}, CommonsParameterException, "at most the carrying capacity"),
    ({"carrying_capacity": WIDTH * HEIGHT - NUM_AGENTS + 1}, CommonsGridException, "need 65 cells, but the grid has 64"),
])
def test_invalid_parameters_fail_clearly(in_tmp_dir, kwargs, exception, match):
    with pytest.raises(exception, match=match):
        build(str(in_tmp_dir / "checkpoints") + "/", **kwargs)


def test_the_largest_commons_the_grid_holds_and_its_default_initial_stock(in_tmp_dir):
    model = build(str(in_tmp_dir / "checkpoints") + "/", carrying_capacity=WIDTH * HEIGHT - NUM_AGENTS)
    assert model.initial_stock == (WIDTH * HEIGHT - NUM_AGENTS) // 2 == model.stock()


def test_regrowth_delays_are_not_supported(in_tmp_dir):
    with pytest.raises(RegrowthDelaysNotSupported, match="commons"):
        CommonsHarvest("homogeneous", NUM_AGENTS, K, R, "baseline", WIDTH, HEIGHT, 1, 5, True, str(in_tmp_dir) + "/", False, False,
                       allotment_regrowth_delays=[1])


# ── outputs ───────────────────────────────────────────────────────

def test_commons_reports_one_row_per_step_and_per_episode(in_tmp_dir):
    model = build(str(in_tmp_dir / "checkpoints") + "/", seasons=True, max_days=6, write_data=True)
    while model.episode <= 2:
        model.step()
    steps = pd.read_csv("data/results/current_run/commons_commons.csv")
    episodes = pd.read_csv("data/results/current_run/commons_episodes_commons.csv")
    assert list(steps.columns) == CommonsHarvest.STEP_COLUMNS and list(episodes.columns) == CommonsHarvest.EPISODE_COLUMNS
    assert list(episodes["episode"]) == [1, 2]
    for episode, rows in steps.groupby("episode"):
        report = episodes[episodes["episode"] == episode].iloc[0]
        assert list(rows["day"]) == list(range(1, len(rows) + 1))
        assert report["mean_stock"] == pytest.approx(rows["stock"].mean())
        assert report["final_stock"] == rows["stock"].iloc[-1]
        assert report["efficiency"] == pytest.approx(rows["harvested"].mean() / (R * K / 4))
        assert ((rows["growth_rate"] / R).round(6).isin([0.5, 1.0, 1.5])).all()
    # the other reports keep their columns
    model_report = pd.read_csv("data/results/current_run/model_episode_reports_commons.csv")
    assert "stock" not in model_report.columns


def test_a_continued_commons_run_carries_on_its_reports(in_tmp_dir):
    checkpoint_path = str(in_tmp_dir / "checkpoints") + "/"
    model = build(checkpoint_path, max_days=6, write_data=True)
    while model.episode <= 2:
        model.step()
    for _ in range(2):
        model.step()
    continued = build(checkpoint_path, max_days=6, write_data=True, start_episode=3)
    continued.reseed_for_continuing()
    while continued.episode <= 3:
        continued.step()
    for path in ["commons_commons.csv", "commons_episodes_commons.csv", "model_episode_reports_commons.csv"]:
        assert sorted(set(pd.read_csv("data/results/current_run/" + path)["episode"])) == [1, 2, 3]


# ── short runs in every combination ───────────────────────────────

@pytest.mark.parametrize("society_mix", ["homogeneous", "heterogeneous"])
@pytest.mark.parametrize("seasons", [False, True])
def test_commons_trains_and_tests_with_or_without_seasons_and_mixed_societies(in_tmp_dir, seasons, society_mix):
    root = str(in_tmp_dir / "checkpoints") + "/"
    checkpoint_path = society_checkpoint_path(root, society_mix, "utilitarian")
    trained = build(checkpoint_path, seasons=seasons, society_mix=society_mix, max_days=5, write_data=True, filepath="train_commons")
    if society_mix != "homogeneous":
        record_agent_types(checkpoint_path, trained.get_living_agents())
    while trained.episode <= 2:
        trained.step()
    tested = build(checkpoint_path, training=False, seasons=seasons, society_mix=society_mix, max_days=5, write_data=True,
                   track_norms=True)
    assert {a.agent_type for a in agents(tested)} == ({"utilitarian", "baseline"} if society_mix != "homogeneous" else {"utilitarian"})
    while tested.episode <= 2:
        tested.step()
    assert len(pd.read_csv("data/results/current_run/commons_episodes_commons.csv")) == 2

import os
from collections import Counter

import pandas as pd
import pytest

from src.harvest_exception import BerryOffGrid, SeasonsException, SeasonsNotSupported
from tests.conftest import INITIAL_HEALTHS, NUM_AGENTS, TEST_SCENARIOS, _build, _make_pretrained_checkpoint

TRAINED_CHECKPOINTS = "data/model_variables/200_days/4_agents/"
# absolute, since in_tmp_dir changes the working directory before the model is built
TRAINED_CHECKPOINTS_ABS = os.path.abspath(TRAINED_CHECKPOINTS) + "/"
SEASON_LENGTH = 3

needs_trained_checkpoints = pytest.mark.skipif(not os.path.isdir(TRAINED_CHECKPOINTS), reason="needs the trained 200_days checkpoints")


@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory):
    return _make_pretrained_checkpoint(tmp_path_factory.mktemp("seasons"), "baseline")


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def build(checkpoint_path, scenario, seasons=True, seed=42, max_days=20, max_episodes=2, initial_healths=(3.0,) * NUM_AGENTS, **kwargs):
    # agents start with enough health to live through every day of a 20-day episode, so episodes don't end early
    return _build(scenario, "homogeneous", NUM_AGENTS * 3, "baseline", False, checkpoint_path,
                  False, False, f"test_{scenario}", max_episodes, max_days, seed=seed,
                  initial_healths=list(initial_healths), seasons=seasons, season_length=SEASON_LENGTH, **kwargs)


def original_group_allocations(model):
    return {holder: allocation["berry_allocation"] for holder, allocation in model.allocations.items()}


def members_total(model):
    """Each group's total, summed here from its members' current individual allocations."""
    return {holder: sum(model.individual_allocations[i] for i in allocation["agent_ids"]) for holder, allocation in model.allocations.items()}


def berries_per_group(model):
    return dict(Counter(b.allocation_id for b in model.berries))


def agents(model):
    return sorted((a for a in model.schedule.agents if a.agent_type != "berry"), key=lambda a: a.unique_id)


def agent_groups_and_bounds(model):
    return [(a.allocation_id, a.min_width, a.max_width, a.min_height, a.max_height,
             a.moving_module.min_width, a.moving_module.max_width, a.moving_module.min_height, a.moving_module.max_height)
            for a in agents(model)]


def assert_berries_in_their_groups_region(model):
    for b in model.berries:
        assert b.min_width <= b.pos[0] < b.max_width and b.min_height <= b.pos[1] < b.max_height
        if "allotment" in model.allocations[b.allocation_id]:
            assert [b.min_width, b.max_width, b.min_height, b.max_height] == model.allocations[b.allocation_id]["allotment"]


def assert_paths_follow_groups(model):
    """Each agent heads for the nearest berry of its own group, and observes the distance to it."""
    for agent in model.get_living_agents():
        mm = agent.moving_module
        assert mm.nearest_berry.allocation_id == agent.allocation_id
        coordinates = mm._find_nearest_berry_coordinates(agent.pos)
        assert agent.observe()[3] == len(mm._find_path_to_berry(agent.pos, coordinates))


def regrow(model, berries):
    """Forage berries and regrow them, as _update_schedule does at the end of a step."""
    for b in berries:
        b.foraged = True
        model._reset_berry(b, False)


def swap_until_group_totals_change(model):
    """Swaps until some group's total differs from its original (swaps within a group can leave totals unchanged)."""
    for _ in range(20):
        model._start_new_season()
        if model._group_targets() != original_group_allocations(model):
            return
    raise AssertionError("group totals never changed")


def run_episode_recording_seasons(model):
    """Steps through the current episode, recording (day, season, individual allocations) after every step within it."""
    episode = model.episode
    record = [(model.day, model.season, list(model.individual_allocations))]
    while True:
        model.step()
        if model.episode != episode:
            return record
        record.append((model.day, model.season, list(model.individual_allocations)))


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_season_zero_group_totals_are_the_current_group_allocations(in_tmp_dir, checkpoint_path, scenario):
    for seasons in [True, False]:
        model = build(checkpoint_path, scenario, seasons=seasons)
        assert model.individual_allocations == model.resources
        assert model._group_targets() == original_group_allocations(model)
        assert berries_per_group(model) == original_group_allocations(model)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_swaps_happen_exactly_at_multiples_of_season_length(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario)
    record = run_episode_recording_seasons(model)
    assert [day for day, _, _ in record] == list(range(20))
    for day, season, _ in record:
        assert season == day // SEASON_LENGTH
    for (_, _, previous), (day, _, individual) in zip(record, record[1:]):
        assert (individual != previous) == (day % SEASON_LENGTH == 0)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_each_season_is_a_permutation_differing_from_the_last(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario)
    record = run_episode_recording_seasons(model)
    seasons = [individual for day, _, individual in record if day % SEASON_LENGTH == 0]
    assert seasons[0] == model.resources
    for previous, individual in zip(seasons, seasons[1:]):
        assert sorted(individual) == sorted(model.resources)
        assert individual != previous


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_individual_allocations_are_reproducible_with_the_same_seed(in_tmp_dir, checkpoint_path, scenario):
    first = [individual for _, _, individual in run_episode_recording_seasons(build(checkpoint_path, scenario, seed=7))]
    second = [individual for _, _, individual in run_episode_recording_seasons(build(checkpoint_path, scenario, seed=7))]
    assert first == second


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_allocations_reset_at_the_start_of_each_episode(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario)
    original_berries = {b.unique_id: (b.allocation_id, b.min_width, b.max_width) for b in model.berries}
    run_episode_recording_seasons(model)
    assert model.episode == 2
    assert model.season == 0
    assert model.individual_allocations == model.resources
    assert {b.unique_id: (b.allocation_id, b.min_width, b.max_width) for b in model.berries} == original_berries
    assert berries_per_group(model) == original_group_allocations(model)
    # the second episode swaps again
    assert run_episode_recording_seasons(model)[-1][1] > 0


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_groups_and_berries_stay_consistent_throughout_a_run(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario)
    num_berries = model.num_berries
    groups_and_bounds = agent_groups_and_bounds(model)
    while model.episode <= 2:
        model.step()
        assert sorted(model.individual_allocations) == sorted(model.resources)
        assert model._group_targets() == members_total(model)
        assert sum(model._group_targets().values()) == num_berries
        assert len(model.berries) == num_berries
        assert berries_per_group(model) == model._group_targets()
        assert agent_groups_and_bounds(model) == groups_and_bounds
        assert_berries_in_their_groups_region(model)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_swap_moves_berries_to_the_new_group_totals(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario)
    agent_positions = [a.pos for a in agents(model)]
    groups_and_bounds = agent_groups_and_bounds(model)
    berries_before = {b.unique_id: (b.allocation_id, b.pos) for b in model.berries}
    swap_until_group_totals_change(model)
    assert [a.pos for a in agents(model)] == agent_positions
    assert agent_groups_and_bounds(model) == groups_and_bounds
    assert berries_per_group(model) == model._group_targets() == members_total(model)
    assert_berries_in_their_groups_region(model)
    changed = [b for b in model.berries if b.allocation_id != berries_before[b.unique_id][0]]
    assert changed
    for b in changed:
        if scenario == "allotment":
            # relocated into the new allotment
            assert b.pos != berries_before[b.unique_id][1]
        else:
            # converted to the new group's type where it is (every type grows over the whole grid)
            assert b.pos == berries_before[b.unique_id][1]
    assert_paths_follow_groups(model)
    # regrowth uses the new group totals and regions
    regrow(model, model.berries)
    assert berries_per_group(model) == model._group_targets()
    assert_berries_in_their_groups_region(model)


def test_colours_are_groups_of_one_agent(in_tmp_dir, checkpoint_path):
    model = build(checkpoint_path, "colours")
    assert [allocation["agent_ids"] for allocation in model.allocations.values()] == [[i] for i in range(NUM_AGENTS)]
    model._start_new_season()
    for agent in agents(model):
        assert model._seasonal_group_berry_allocation(agent) == model._seasonal_berry_allocation(agent)
        assert len(model.get_uneaten_berries_coordinates(agent.allocation_id)) == model._seasonal_berry_allocation(agent)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_swap_raises_if_a_berry_is_off_the_grid(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario)
    model.grid.remove_agent(model.berries[0])
    with pytest.raises(BerryOffGrid):
        model._start_new_season()


def test_seasons_with_basic_raise(in_tmp_dir):
    with pytest.raises(SeasonsNotSupported):
        _build("basic", "homogeneous", NUM_AGENTS * 3, "baseline", True, str(in_tmp_dir / "ckpt") + "/",
               False, False, "basic", 2, 10, seasons=True)


@pytest.mark.parametrize("scenario", ["capabilities", "allotment"])
def test_seasons_with_a_single_group_raise(in_tmp_dir, checkpoint_path, scenario):
    with pytest.raises(SeasonsException):
        build(checkpoint_path, scenario, num_allocations=1)


@pytest.mark.parametrize("season_length", [0, -1, 2.5])
def test_invalid_season_length_raises(in_tmp_dir, checkpoint_path, season_length):
    with pytest.raises(SeasonsException):
        _build("colours", "homogeneous", NUM_AGENTS * 3, "baseline", False, checkpoint_path,
               False, False, "colours", 2, 10, seasons=True, season_length=season_length)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_seasons_off_keeps_the_original_allocations(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario, seasons=False)
    record = run_episode_recording_seasons(model)
    assert all(season == 0 and individual == model.resources for _, season, individual in record)
    assert model.season_rng is None


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_observation_and_actions_match_with_seasons_on_and_off(in_tmp_dir, checkpoint_path, scenario):
    on_model = build(checkpoint_path, scenario)
    off_model = build(checkpoint_path, scenario, seasons=False)
    on_model._start_new_season()
    for on_agent, off_agent in zip(agents(on_model), agents(off_model)):
        assert on_agent.observe().shape == off_agent.observe().shape
        assert on_agent.actions == off_agent.actions


@needs_trained_checkpoints
@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_trained_run_with_seasons_and_initial_healths_writes_seasonal_allocations(in_tmp_dir, scenario):
    model = _build(scenario, "homogeneous", NUM_AGENTS * 3, "utilitarian", False, TRAINED_CHECKPOINTS_ABS,
                   True, False, f"{scenario}_unequal_health_seasons", 2, 20, seed=42,
                   initial_healths=INITIAL_HEALTHS, seasons=True, season_length=SEASON_LENGTH)
    while model.episode <= 2:
        model.step()
    reports = pd.read_csv(f"data/results/current_run/agent_reports_{scenario}_unequal_health_seasons.csv")
    for agent_id, initial_health in enumerate(INITIAL_HEALTHS):
        assert (reports.loc[reports["agent_id"] == agent_id, "initial_health"] == initial_health).all()
    # rows for day d are written before the swap on day d, so they belong to season (d-1) // season_length
    reports["season"] = (reports["day"] - 1) // SEASON_LENGTH
    per_season = reports.groupby(["episode", "season", "agent_id"])[["seasonal_berry_allocation", "seasonal_group_berry_allocation"]]
    assert (per_season.nunique() == 1).all().all()
    per_season = per_season.first().reset_index()
    groups = {a.unique_id: a.allocation_id for a in agents(model)}
    for (episode, season), rows in per_season.groupby(["episode", "season"]):
        individual = dict(zip(rows["agent_id"], rows["seasonal_berry_allocation"]))
        group_total = dict(zip(rows["agent_id"], rows["seasonal_group_berry_allocation"]))
        if season == 0:
            assert individual == {i: model.resources[i] for i in individual}
        for agent_id, total in group_total.items():
            members = model.allocations[groups[agent_id]]["agent_ids"]
            if all(member in individual for member in members):
                assert total == sum(individual[member] for member in members)
    assert reports.groupby(["episode", "agent_id"])["seasonal_berry_allocation"].nunique().max() > 1


@needs_trained_checkpoints
@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_seasonal_allocations_are_the_original_allocations_when_seasons_are_off(in_tmp_dir, scenario):
    model = _build(scenario, "homogeneous", NUM_AGENTS * 3, "utilitarian", False, TRAINED_CHECKPOINTS_ABS,
                   True, False, scenario, 1, 20, seed=42)
    while model.episode <= 1:
        model.step()
    reports = pd.read_csv(f"data/results/current_run/agent_reports_{scenario}.csv")
    for agent in agents(model):
        rows = reports.loc[reports["agent_id"] == agent.unique_id]
        assert (rows["seasonal_berry_allocation"] == model.resources[agent.unique_id]).all()
        assert (rows["seasonal_group_berry_allocation"] == model.allocations[agent.allocation_id]["berry_allocation"]).all()

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


def original_values(model):
    return [allocation["berry_allocation"] for allocation in model.allocations.values()]


def berries_per_holder(model):
    return Counter(b.allocation_id for b in model.berries)


def season_targets(model):
    return dict(zip(model.allocations, model._season_values(model.season_assignment)))


def agents(model):
    return sorted((a for a in model.schedule.agents if a.agent_type != "berry"), key=lambda a: a.unique_id)


def assert_paths_follow_assignment(model):
    """Each agent heads for the nearest berry it can forage this season, and observes the distance to it."""
    for agent in model.get_living_agents():
        mm = agent.moving_module
        forage_allocation = model.get_forage_allocation(agent.allocation_id)
        assert mm.nearest_berry.allocation_id == forage_allocation
        coordinates = mm._find_nearest_berry_coordinates(agent.pos)
        assert agent.observe()[3] == len(mm._find_path_to_berry(agent.pos, coordinates))


def regrow(model, berries):
    """Forage berries and regrow them, as _update_schedule does at the end of a step."""
    for b in berries:
        b.foraged = True
        model._reset_berry(b, False)


def run_episode_recording_seasons(model):
    """Steps through the current episode, recording (day, season, assignment) after every step within it."""
    episode = model.episode
    record = [(model.day, model.season, dict(model.season_assignment))]
    while True:
        model.step()
        if model.episode != episode:
            return record
        record.append((model.day, model.season, dict(model.season_assignment)))


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_swaps_happen_exactly_at_multiples_of_season_length(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario)
    record = run_episode_recording_seasons(model)
    assert [day for day, _, _ in record] == list(range(20))
    for day, season, _ in record:
        assert season == day // SEASON_LENGTH
    for (_, _, previous), (day, _, assignment) in zip(record, record[1:]):
        assert (assignment != previous) == (day % SEASON_LENGTH == 0)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_each_season_is_a_permutation_with_different_values_from_the_last(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario)
    record = run_episode_recording_seasons(model)
    holders = list(model.allocations)
    seasons = [assignment for day, _, assignment in record if day % SEASON_LENGTH == 0]
    assert seasons[0] == {holder: holder for holder in holders}
    for previous, assignment in zip(seasons, seasons[1:]):
        assert sorted(assignment.values()) == sorted(holders)
        assert model._season_values(assignment) != model._season_values(previous)
        assert sorted(model._season_values(assignment)) == sorted(original_values(model))


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_assignments_are_reproducible_with_the_same_seed(in_tmp_dir, checkpoint_path, scenario):
    first = [assignment for _, _, assignment in run_episode_recording_seasons(build(checkpoint_path, scenario, seed=7))]
    second = [assignment for _, _, assignment in run_episode_recording_seasons(build(checkpoint_path, scenario, seed=7))]
    assert first == second


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_assignment_resets_at_the_start_of_each_episode(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario)
    original_holders = {b.unique_id: (b.allocation_id, b.min_width, b.max_width) for b in model.berries}
    run_episode_recording_seasons(model)
    assert model.episode == 2
    assert model.season == 0
    assert model.season_assignment == {holder: holder for holder in model.allocations}
    assert {b.unique_id: (b.allocation_id, b.min_width, b.max_width) for b in model.berries} == original_holders
    # the second episode swaps again
    assert run_episode_recording_seasons(model)[-1][1] > 0


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_total_berries_and_allocation_values_never_change(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario)
    num_berries = model.num_berries
    while model.episode <= 2:
        model.step()
        assert len(model.berries) == num_berries
        assert sum(berries_per_holder(model).values()) == num_berries
        assert sorted(model._season_values(model.season_assignment)) == sorted(original_values(model))


def test_colours_swap_moves_ownership_of_colours(in_tmp_dir, checkpoint_path):
    model = build(checkpoint_path, "colours")
    colours_before = {b.unique_id: b.allocation_id for b in model.berries}
    model._start_new_season()
    # berries keep their colour; agents forage the colour their home allocation holds this season
    assert {b.unique_id: b.allocation_id for b in model.berries} == colours_before
    for agent in agents(model):
        forage_allocation = model.get_forage_allocation(agent.allocation_id)
        assert forage_allocation == model.season_assignment[agent.allocation_id]
        assert len(model.get_uneaten_berries_coordinates(forage_allocation)) == model._seasonal_berry_allocation(agent)
    assert_paths_follow_assignment(model)
    # regrowth keeps the colour, so the new owner can forage it
    berry = model.berries[0]
    colour = berry.allocation_id
    regrow(model, [berry])
    assert berry.allocation_id == colour and not berry.foraged and berry.pos is not None


def test_capabilities_swap_changes_how_many_berries_of_each_kind(in_tmp_dir, checkpoint_path):
    model = build(checkpoint_path, "capabilities")
    groups_before = [a.allocation_id for a in agents(model)]
    positions_before = {b.unique_id: b.pos for b in model.berries}
    model._start_new_season()
    # groups keep their capability; berries are relabelled where they are
    assert [a.allocation_id for a in agents(model)] == groups_before
    assert all(model.get_forage_allocation(a.allocation_id) == a.allocation_id for a in agents(model))
    assert {b.unique_id: b.pos for b in model.berries} == positions_before
    assert berries_per_holder(model) == season_targets(model)
    for agent in agents(model):
        assert len(model.get_uneaten_berries_coordinates(agent.allocation_id)) == model._seasonal_berry_allocation(agent)
    assert_paths_follow_assignment(model)
    # regrowth keeps the new kind
    berry = model.berries[0]
    kind = berry.allocation_id
    regrow(model, [berry])
    assert berry.allocation_id == kind
    assert berries_per_holder(model) == season_targets(model)


def test_allotment_swap_moves_berries_between_allotments(in_tmp_dir, checkpoint_path):
    model = build(checkpoint_path, "allotment")
    agents_before = [(a.pos, a.allocation_id, a.min_width, a.max_width, a.min_height, a.max_height) for a in agents(model)]
    bounds_before = [(a.moving_module.min_width, a.moving_module.max_width) for a in agents(model)]
    model._start_new_season()
    # agents don't move between allotments and their movement bounds don't change
    assert [(a.pos, a.allocation_id, a.min_width, a.max_width, a.min_height, a.max_height) for a in agents(model)] == agents_before
    assert [(a.moving_module.min_width, a.moving_module.max_width) for a in agents(model)] == bounds_before
    assert berries_per_holder(model) == season_targets(model)
    for b in model.berries:
        allotment = model.allocations[b.allocation_id]["allotment"]
        assert [b.min_width, b.max_width, b.min_height, b.max_height] == allotment
        assert allotment[0] <= b.pos[0] < allotment[1] and allotment[2] <= b.pos[1] < allotment[3]
    assert_paths_follow_assignment(model)
    # regrowth puts the berry back inside its new allotment
    regrow(model, model.berries)
    assert berries_per_holder(model) == season_targets(model)
    for b in model.berries:
        allotment = model.allocations[b.allocation_id]["allotment"]
        assert allotment[0] <= b.pos[0] < allotment[1] and allotment[2] <= b.pos[1] < allotment[3]


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
def test_seasons_off_keeps_the_original_assignment(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario, seasons=False)
    record = run_episode_recording_seasons(model)
    assert all(season == 0 and assignment == {h: h for h in model.allocations} for _, season, assignment in record)
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
def test_trained_run_with_seasons_and_initial_healths_writes_seasonal_berry_allocation(in_tmp_dir, scenario):
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
    assert (reports.groupby(["episode", "agent_id", "season"])["seasonal_berry_allocation"].nunique() == 1).all()
    first_season = reports[reports["season"] == 0].groupby("agent_id")["seasonal_berry_allocation"].first()
    original = {a.unique_id: model.allocations[a.allocation_id]["berry_allocation"] for a in agents(model)}
    assert first_season.to_dict() == original
    assert reports.groupby(["episode", "agent_id"])["seasonal_berry_allocation"].nunique().max() > 1


@needs_trained_checkpoints
@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_seasonal_berry_allocation_is_the_original_allocation_when_seasons_are_off(in_tmp_dir, scenario):
    model = _build(scenario, "homogeneous", NUM_AGENTS * 3, "utilitarian", False, TRAINED_CHECKPOINTS_ABS,
                   True, False, scenario, 1, 20, seed=42)
    while model.episode <= 1:
        model.step()
    reports = pd.read_csv(f"data/results/current_run/agent_reports_{scenario}.csv")
    for agent in agents(model):
        expected = model.allocations[agent.allocation_id]["berry_allocation"]
        assert (reports.loc[reports["agent_id"] == agent.unique_id, "seasonal_berry_allocation"] == expected).all()

import os
from collections import Counter

import pandas as pd
import pytest

from src.harvest_exception import BerryOffGrid, SeasonsException, SeasonsNotSupported
from src.harvest_model import HarvestModel
from tests.conftest import NUM_AGENTS, TEST_SCENARIOS, _build, _make_pretrained_checkpoint, long_lived

TRAINED_CHECKPOINTS = "data/model_variables/200_days/4_agents/"
# absolute, since in_tmp_dir changes the working directory before the model is built
TRAINED_CHECKPOINTS_ABS = os.path.abspath(TRAINED_CHECKPOINTS) + "/"
SEASON_LENGTH = 3
CHANGE = HarvestModel.SEASON_QUALITY_CHANGE

needs_trained_checkpoints = pytest.mark.skipif(not os.path.isdir(TRAINED_CHECKPOINTS), reason="needs the trained 200_days checkpoints")


@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory):
    return _make_pretrained_checkpoint(tmp_path_factory.mktemp("seasons"), "baseline")


@pytest.fixture(scope="module")
def utilitarian_checkpoint_path(tmp_path_factory):
    return _make_pretrained_checkpoint(tmp_path_factory.mktemp("seasons_utilitarian"), "utilitarian")


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def build(checkpoint_path, scenario, seasons=True, seed=42, max_days=20, max_episodes=2, agent_type="baseline", **kwargs):
    return long_lived(_build(scenario, "homogeneous", NUM_AGENTS * 3, agent_type, False, checkpoint_path,
                             False, False, f"test_{scenario}", max_episodes, max_days, seed=seed,
                             seasons=seasons, season_length=SEASON_LENGTH, **kwargs))


def agents(model):
    return sorted((a for a in model.schedule.agents if a.agent_type != "berry"), key=lambda a: a.unique_id)


def agent_groups_and_bounds(model):
    return [(a.allocation_id, a.min_width, a.max_width, a.min_height, a.max_height,
             a.moving_module.min_width, a.moving_module.max_width, a.moving_module.min_height, a.moving_module.max_height)
            for a in agents(model)]


def berries_per_holder(model):
    counts = Counter(b.allocation_id for b in model.berries)
    return {holder: counts[holder] for holder in model.allocations}


def group_size(model, holder):
    return len(model.allocations[holder]["agent_ids"])


def expected_targets(model, quality):
    """Each holder's target: its own allocation, changed by its own number of agents, at least 0."""
    return {holder: max(0, data["berry_allocation"] + CHANGE[quality] * group_size(model, holder))
            for holder, data in model.allocations.items()}


def assert_berries_in_their_holders_region(model):
    for b in model.berries:
        region = model.allocations[b.allocation_id].get("allotment", [0, model.max_width, 0, model.max_height])
        assert [b.min_width, b.max_width, b.min_height, b.max_height] == region
        assert region[0] <= b.pos[0] < region[1] and region[2] <= b.pos[1] < region[3]
        assert b in model.schedule.agents and not b.foraged


def force_qualities(monkeypatch, draw):
    """Replaces the random draw of each season's quality with draw(model), from before the model is built."""
    monkeypatch.setattr(HarvestModel, "_draw_season_quality", draw)


def force_quality_sequence(monkeypatch, *qualities):
    """Seasons get qualities in order, the first being season 0 at construction."""
    sequence = iter(qualities)
    force_qualities(monkeypatch, lambda self: next(sequence))


def run_episode(model, check=None):
    """Steps through the current episode, calling check(model) after every step within it."""
    episode = model.episode
    while True:
        model.step()
        if model.episode != episode:
            return
        if check is not None:
            check(model)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_every_season_has_one_random_quality_and_targets_to_match(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario, max_episodes=3)
    qualities = {}

    def record(m):
        # seasons change after the steps on days season_length, 2*season_length, ...
        assert m.season == m.day // SEASON_LENGTH
        qualities.setdefault((m.episode, m.season), set()).add(m._current_season_quality())
        assert berries_per_holder(m) == expected_targets(m, m._current_season_quality())
    record(model)
    for _ in range(3):
        run_episode(model, record)
        if model.episode <= 3:
            assert model.day == 0
            record(model)
    assert all(len(seen) == 1 for seen in qualities.values())
    seen = [quality for seen in qualities.values() for quality in seen]
    assert set(seen) == set(HarvestModel.SEASON_QUALITIES)
    # random, not a fixed cycle: the qualities aren't the same in every episode
    assert len({tuple(frozenset(qualities[(episode, season)]) for season in range(20 // SEASON_LENGTH + 1)) for episode in (1, 2, 3)}) > 1


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_first_season_of_each_episode_gets_its_quality_from_the_original_berries(in_tmp_dir, checkpoint_path, scenario, monkeypatch):
    # bad at construction, good at the start of episode 2, normal otherwise
    force_qualities(monkeypatch, lambda self: ("bad" if self.episode == 1 else "good") if self.season == 0 else "normal")
    model = build(checkpoint_path, scenario)
    original = {holder: data["berry_allocation"] for holder, data in model.allocations.items()}
    num_original = len(model.original_berries)
    assert model.season == 0 and model._current_season_quality() == "bad"
    assert berries_per_holder(model) == {holder: original[holder] - group_size(model, holder) for holder in model.allocations}
    assert_berries_in_their_holders_region(model)

    def check(m):
        if m.season > 0:
            assert m._current_season_quality() == "normal" and berries_per_holder(m) == original
    run_episode(model, check)
    assert model.episode == 2 and model.season == 0 and model._current_season_quality() == "good"
    assert berries_per_holder(model) == {holder: original[holder] + group_size(model, holder) for holder in model.allocations}
    assert model.num_berries == len(model.berries) == num_original + NUM_AGENTS
    assert sum(1 for a in model.schedule.agents if a.agent_type == "berry") == model.num_berries
    assert all(b in model.berries for b in model.original_berries)
    assert_berries_in_their_holders_region(model)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
@pytest.mark.parametrize("num_allocations", [2, 3])
def test_allocations_stay_with_their_groups_changed_by_group_size(in_tmp_dir, checkpoint_path, scenario, num_allocations):
    # with 3 allocations capability groups and allotments have 2, 1 and 1 agents; colours are always groups of one
    model = build(checkpoint_path, scenario, num_allocations=num_allocations)
    groups_and_bounds = agent_groups_and_bounds(model)
    original = {holder: data["berry_allocation"] for holder, data in model.allocations.items()}

    def check(m):
        quality = m._current_season_quality()
        # each group's target comes from its own allocation (never cumulative), and groups never change their members or bounds;
        # berry objects a bad season makes dormant can be reused by any group in a later good season
        assert agent_groups_and_bounds(m) == groups_and_bounds
        targets = {holder: max(0, original[holder] + CHANGE[quality] * group_size(m, holder)) for holder in m.allocations}
        assert m._season_targets() == targets
        assert berries_per_holder(m) == targets
        assert len(m.berries) == m.num_berries == sum(targets.values())
        assert_berries_in_their_holders_region(m)
    run_episode(model, check)
    run_episode(model, check)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_good_season_adds_berries_in_each_groups_place_and_type(in_tmp_dir, checkpoint_path, scenario, monkeypatch):
    force_quality_sequence(monkeypatch, "normal", "good")
    model = build(checkpoint_path, scenario)
    before = berries_per_holder(model)
    original_ids = {b.unique_id for b in model.berries}
    model._start_new_season()
    assert model._current_season_quality() == "good"
    added = [b for b in model.berries if b.unique_id not in original_ids]
    assert len(added) == NUM_AGENTS
    for holder in model.allocations:
        assert berries_per_holder(model)[holder] == before[holder] + group_size(model, holder)
    assert_berries_in_their_holders_region(model)
    for agent in agents(model):
        assert len(model.get_uneaten_berries_coordinates(agent.allocation_id)) == model._seasonal_berry_target(agent)
        assert agent.moving_module.nearest_berry.allocation_id == agent.allocation_id
    # the added berries regrow in their group's region
    for b in added:
        b.foraged = True
        model._reset_berry(b, False)
    assert_berries_in_their_holders_region(model)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_bad_season_removes_uncollected_berries_from_each_group_and_not_carried_ones(in_tmp_dir, checkpoint_path, scenario, monkeypatch):
    force_quality_sequence(monkeypatch, "normal", "good", "normal", "bad", "normal")
    model = build(checkpoint_path, scenario)
    original = {holder: data["berry_allocation"] for holder, data in model.allocations.items()}
    model._start_new_season()
    model._start_new_season()
    for agent in agents(model):
        agent.berries = 2
    in_use = set(model.berries)
    model._start_new_season()
    assert model._current_season_quality() == "bad"
    assert all(agent.berries == 2 for agent in agents(model))
    for holder in model.allocations:
        assert berries_per_holder(model)[holder] == original[holder] - group_size(model, holder)
    removed = in_use - set(model.berries)
    assert len(removed) == NUM_AGENTS
    for b in removed:
        assert b in model.dormant_berries and b not in model.schedule.agents
        assert b.pos is None
    assert_berries_in_their_holders_region(model)
    # back to normal: the normal targets again, reusing dormant berries rather than making new ones
    total_objects = len(model.berries) + len(model.dormant_berries)
    model._start_new_season()
    assert berries_per_holder(model) == original
    assert len(model.berries) + len(model.dormant_berries) == total_objects


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_berry_objects_are_reused_across_seasons_and_episodes(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario, max_episodes=3)
    original = len(model.original_berries)
    while model.episode <= 3:
        model.step()
        assert len(model.berries) + len(model.dormant_berries) <= original + NUM_AGENTS
        assert len({b.unique_id for b in model.berries + model.dormant_berries}) == len(model.berries) + len(model.dormant_berries)


def test_bad_season_clamps_a_group_at_zero(in_tmp_dir, checkpoint_path, monkeypatch):
    monkeypatch.setattr(HarvestModel, "_generate_resource_allocations", lambda self, num_agents: [6, 4, 2, 0])
    force_quality_sequence(monkeypatch, "normal", "good", "normal", "bad")
    model = build(checkpoint_path, "colours")
    total = len(model.berries)
    for _ in range(3):
        model._start_new_season()
    assert model._current_season_quality() == "bad"
    assert berries_per_holder(model) == {"allocation_0": 5, "allocation_1": 3, "allocation_2": 1, "allocation_3": 0}
    # clamped, so the total falls by 3 rather than num_agents
    assert len(model.berries) == total - 3


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_two_agent_types_with_the_same_seed_get_the_same_seasons(in_tmp_dir, checkpoint_path, utilitarian_checkpoint_path, scenario):
    def boundaries(model):
        recorded = []
        while model.episode <= 2:
            model.step()
            if model.day > 0 and model.day % SEASON_LENGTH == 0:
                recorded.append((model.episode, model.day, model._current_season_quality(),
                                 sorted((b.unique_id, b.allocation_id) for b in model.berries)))
        return recorded
    baseline = boundaries(build(checkpoint_path, scenario))
    utilitarian = boundaries(build(utilitarian_checkpoint_path, scenario, agent_type="utilitarian"))
    assert len(baseline) == 2 * (20 // SEASON_LENGTH)
    assert baseline == utilitarian


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_season_change_raises_if_a_berry_is_off_the_grid(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario)
    model.grid.remove_agent(model.berries[0])
    with pytest.raises(BerryOffGrid):
        model._start_new_season()


def test_seasons_with_basic_raise(in_tmp_dir):
    with pytest.raises(SeasonsNotSupported):
        _build("basic", "homogeneous", NUM_AGENTS * 3, "baseline", True, str(in_tmp_dir / "ckpt") + "/",
               False, False, "basic", 2, 10, seasons=True)


@pytest.mark.parametrize("season_length", [0, -1, 2.5])
def test_invalid_season_length_raises(in_tmp_dir, checkpoint_path, season_length):
    with pytest.raises(SeasonsException):
        _build("colours", "homogeneous", NUM_AGENTS * 3, "baseline", False, checkpoint_path,
               False, False, "colours", 2, 10, seasons=True, season_length=season_length)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_seasons_off_keeps_one_season_with_the_original_allocations(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario, seasons=False)
    original = {holder: data["berry_allocation"] for holder, data in model.allocations.items()}

    def check(m):
        assert m.season == 0 and m._current_season_quality() is None
        assert m._season_targets() == original
    check(model)
    run_episode(model, check)
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
def test_trained_run_with_seasons_writes_the_season_columns(in_tmp_dir, scenario):
    model = long_lived(_build(scenario, "homogeneous", NUM_AGENTS * 3, "utilitarian", False, TRAINED_CHECKPOINTS_ABS,
                              True, False, f"{scenario}_seasons", 2, 20, seed=42, seasons=True, season_length=SEASON_LENGTH))
    while model.episode <= 2:
        model.step()
    reports = pd.read_csv(f"data/results/current_run/agent_reports_{scenario}_seasons.csv")
    assert "initial_health" not in reports and "seasonal_berry_allocation" not in reports
    # rows for day d are written before the season change on day d
    assert (reports["season"] == (reports["day"] - 1) // SEASON_LENGTH).all()
    assert (reports.groupby(["episode", "season"])["season_quality"].nunique() == 1).all()
    group_of = {a.unique_id: a.allocation_id for a in agents(model)}
    allocation = {a.unique_id: model.allocations[a.allocation_id]["berry_allocation"] for a in agents(model)}
    size = {a.unique_id: group_size(model, a.allocation_id) for a in agents(model)}
    change = reports["season_quality"].map(CHANGE) * reports["agent_id"].map(size)
    assert (reports["seasonal_berry_target"] == (reports["agent_id"].map(allocation) + change).clip(lower=0)).all()
    reports["group"] = reports["agent_id"].map(group_of)
    assert (reports.groupby(["episode", "day", "group"])["seasonal_berry_target"].nunique() == 1).all()
    assert set(reports["season_quality"]) == {"normal", "good", "bad"}


@needs_trained_checkpoints
@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_season_columns_without_seasons(in_tmp_dir, scenario):
    model = _build(scenario, "homogeneous", NUM_AGENTS * 3, "utilitarian", False, TRAINED_CHECKPOINTS_ABS,
                   True, False, scenario, 1, 20, seed=42)
    while model.episode <= 1:
        model.step()
    reports = pd.read_csv(f"data/results/current_run/agent_reports_{scenario}.csv")
    assert (reports["season"] == 0).all()
    assert reports["season_quality"].isna().all()
    allocation = {a.unique_id: model.allocations[a.allocation_id]["berry_allocation"] for a in agents(model)}
    assert (reports["seasonal_berry_target"] == reports["agent_id"].map(allocation)).all()

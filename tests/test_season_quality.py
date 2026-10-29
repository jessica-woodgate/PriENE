import os
from collections import Counter

import pandas as pd
import pytest

from src.harvest_exception import SeasonsNotSupported
from src.harvest_model import HarvestModel
from tests.conftest import INITIAL_HEALTHS, NUM_AGENTS, TEST_SCENARIOS, _build, _make_pretrained_checkpoint

TRAINED_CHECKPOINTS = "data/model_variables/200_days/4_agents/"
# absolute, since in_tmp_dir changes the working directory before the model is built
TRAINED_CHECKPOINTS_ABS = os.path.abspath(TRAINED_CHECKPOINTS) + "/"
SEASON_LENGTH = 3
CHANGE = HarvestModel.SEASON_QUALITY_CHANGE

needs_trained_checkpoints = pytest.mark.skipif(not os.path.isdir(TRAINED_CHECKPOINTS), reason="needs the trained 200_days checkpoints")


@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory):
    return _make_pretrained_checkpoint(tmp_path_factory.mktemp("season_quality"), "baseline")


@pytest.fixture(scope="module")
def utilitarian_checkpoint_path(tmp_path_factory):
    return _make_pretrained_checkpoint(tmp_path_factory.mktemp("season_quality_utilitarian"), "utilitarian")


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def build(checkpoint_path, scenario, seasons=False, season_quality=True, seed=42, max_days=20, max_episodes=2,
          initial_healths=(3.0,) * NUM_AGENTS, agent_type="baseline", **kwargs):
    # agents start with enough health to live through every day of a 20-day episode, so episodes don't end early
    return _build(scenario, "homogeneous", NUM_AGENTS * 3, agent_type, False, checkpoint_path,
                  False, False, f"test_{scenario}", max_episodes, max_days, seed=seed,
                  initial_healths=list(initial_healths), seasons=seasons, season_length=SEASON_LENGTH,
                  season_quality=season_quality, **kwargs)


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
    """Each holder's target: the original allocation it holds, changed by its own number of agents, at least 0."""
    return {holder: max(0, model.allocations[model.season_assignment[holder]]["berry_allocation"] + CHANGE[quality] * group_size(model, holder))
            for holder in model.allocations}


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
def test_quality_only_targets_are_the_original_allocation_plus_or_minus_group_size(in_tmp_dir, checkpoint_path, scenario):
    model = build(checkpoint_path, scenario)
    groups_and_bounds = agent_groups_and_bounds(model)
    original = {holder: data["berry_allocation"] for holder, data in model.allocations.items()}

    def check(m):
        quality = m._current_season_quality()
        # allocations stay with their groups, and the change is never cumulative
        assert m.season_assignment == {holder: holder for holder in m.allocations}
        assert agent_groups_and_bounds(m) == groups_and_bounds
        targets = {holder: max(0, original[holder] + CHANGE[quality] * group_size(m, holder)) for holder in m.allocations}
        assert m._season_targets() == targets
        assert berries_per_holder(m) == targets
        assert len(m.berries) == m.num_berries == sum(targets.values())
        assert_berries_in_their_holders_region(m)
    run_episode(model, check)
    run_episode(model, check)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_with_swaps_the_change_uses_the_size_of_the_group_receiving_the_allocation(in_tmp_dir, checkpoint_path, scenario):
    # three groups of sizes 2, 1, 1 in capabilities and allotment; groups of one in colours
    model = build(checkpoint_path, scenario, seasons=True, num_allocations=3)
    groups_and_bounds = agent_groups_and_bounds(model)
    original_total = sum(data["berry_allocation"] for data in model.allocations.values())
    seen_sizes = set()

    def check(m):
        quality = m._current_season_quality()
        assert agent_groups_and_bounds(m) == groups_and_bounds
        assert m._season_targets() == expected_targets(m, quality)
        assert berries_per_holder(m) == expected_targets(m, quality)
        # no group is clamped with 4 agents, so the society total changes by exactly num_agents
        assert len(m.berries) == original_total + CHANGE[quality] * NUM_AGENTS
        assert_berries_in_their_holders_region(m)
        seen_sizes.update(group_size(m, holder) for holder in m.allocations if m.season_assignment[holder] != holder)
    run_episode(model, check)
    run_episode(model, check)
    assert model.season_swaps
    if scenario != "colours":
        # an allocation moved to the group of two at some point, so its change was +-2
        assert 2 in seen_sizes


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_swap_comes_before_the_quality_change(in_tmp_dir, checkpoint_path, scenario, monkeypatch):
    force_quality_sequence(monkeypatch, "normal", "good")
    model = build(checkpoint_path, scenario, seasons=True)
    calls = []
    move, apply = model._move_berries_to_season_targets, model._apply_season_quality
    monkeypatch.setattr(model, "_move_berries_to_season_targets", lambda: (calls.append("swap"), move()))
    monkeypatch.setattr(model, "_apply_season_quality", lambda: (calls.append("quality"), apply()))
    model._start_new_season()
    assert calls == ["swap", "quality"]
    assert berries_per_holder(model) == expected_targets(model, "good")


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
    model = build(checkpoint_path, scenario, seasons=True, max_episodes=3)
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
@pytest.mark.parametrize("seasons", [False, True])
def test_two_agent_types_with_the_same_seed_get_the_same_seasons(in_tmp_dir, checkpoint_path, utilitarian_checkpoint_path, scenario, seasons):
    def boundaries(model):
        recorded = []
        while model.episode <= 2:
            model.step()
            if model.day > 0 and model.day % SEASON_LENGTH == 0:
                recorded.append((model.episode, model.day, model._current_season_quality(), dict(model.season_assignment),
                                 sorted((b.unique_id, b.allocation_id) for b in model.berries)))
        return recorded
    baseline = boundaries(build(checkpoint_path, scenario, seasons=seasons))
    utilitarian = boundaries(build(utilitarian_checkpoint_path, scenario, seasons=seasons, agent_type="utilitarian"))
    assert len(baseline) == 2 * (20 // SEASON_LENGTH)
    assert baseline == utilitarian


def test_season_quality_with_basic_raises(in_tmp_dir):
    with pytest.raises(SeasonsNotSupported, match="season quality"):
        _build("basic", "homogeneous", NUM_AGENTS * 3, "baseline", True, str(in_tmp_dir / "ckpt") + "/",
               False, False, "basic", 2, 10, season_quality=True)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_observation_and_actions_match_with_season_quality_on_and_off(in_tmp_dir, checkpoint_path, scenario):
    on_model = build(checkpoint_path, scenario)
    off_model = build(checkpoint_path, scenario, season_quality=False)
    on_model._start_new_season()
    for on_agent, off_agent in zip(agents(on_model), agents(off_model)):
        assert on_agent.observe().shape == off_agent.observe().shape
        assert on_agent.actions == off_agent.actions


@needs_trained_checkpoints
@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
@pytest.mark.parametrize("seasons", [False, True])
def test_trained_run_with_season_quality_and_initial_healths_writes_the_season_columns(in_tmp_dir, scenario, seasons):
    model = _build(scenario, "homogeneous", NUM_AGENTS * 3, "utilitarian", False, TRAINED_CHECKPOINTS_ABS,
                   True, False, f"{scenario}_quality", 2, 20, seed=42, initial_healths=INITIAL_HEALTHS,
                   seasons=seasons, season_length=SEASON_LENGTH, season_quality=True)
    while model.episode <= 2:
        model.step()
    reports = pd.read_csv(f"data/results/current_run/agent_reports_{scenario}_quality.csv")
    for agent_id, initial_health in enumerate(INITIAL_HEALTHS):
        assert (reports.loc[reports["agent_id"] == agent_id, "initial_health"] == initial_health).all()
    # rows for day d are written before the season change on day d
    assert (reports["season"] == (reports["day"] - 1) // SEASON_LENGTH).all()
    assert (reports.groupby(["episode", "season"])["season_quality"].nunique() == 1).all()
    group_of = {a.unique_id: a.allocation_id for a in agents(model)}
    size = {a.unique_id: group_size(model, a.allocation_id) for a in agents(model)}
    change = reports["season_quality"].map(CHANGE) * reports["agent_id"].map(size)
    assert (reports["seasonal_berry_target"] == (reports["seasonal_berry_allocation"] + change).clip(lower=0)).all()
    reports["group"] = reports["agent_id"].map(group_of)
    assert (reports.groupby(["episode", "day", "group"])["seasonal_berry_target"].nunique() == 1).all()
    assert set(reports["season_quality"]) == {"normal", "good", "bad"}


@needs_trained_checkpoints
@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_season_columns_without_season_quality(in_tmp_dir, scenario):
    model = _build(scenario, "homogeneous", NUM_AGENTS * 3, "utilitarian", False, TRAINED_CHECKPOINTS_ABS,
                   True, False, scenario, 1, 20, seed=42)
    while model.episode <= 1:
        model.step()
    reports = pd.read_csv(f"data/results/current_run/agent_reports_{scenario}.csv")
    assert (reports["season"] == 0).all()
    assert reports["season_quality"].isna().all()
    assert (reports["seasonal_berry_target"] == reports["seasonal_berry_allocation"]).all()

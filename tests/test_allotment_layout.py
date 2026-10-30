import math
from collections import Counter

import numpy as np
import pytest

from src.harvest_exception import (AllotmentCapacityException, AllotmentWidthException, MixedSocietyException,
                                   NumAllotmentsException)
from src.scenarios.allotment_harvest import AllotmentHarvest
from tests.conftest import NUM_AGENTS, _build

AGENT_COUNTS = [2, 4, 6, 8, 10, 20]
# berries in the society for each agent count: hardcoded for 2, 4, 6 and 20; Zipf with num_agents*3 berries otherwise
TOTAL_BERRIES = {2: 6, 4: 12, 6: 18, 8: 24, 10: 30, 20: 61}
SEASON_LENGTH = 3
MAX_DAYS = 10


def allotment_counts(num_agents):
    return sorted({1, 2, math.ceil(num_agents / 2), num_agents})


def grids(num_agents):
    """Widths that don't divide evenly (7, 10) and run.py's allotment grid (4n x 2n), each with a short and a tall height."""
    return sorted({(width, height) for width in [7, 10, 4 * num_agents] for height in [5, 2 * num_agents]})


LAYOUTS = [(n, k, width, height) for n in AGENT_COUNTS for k in allotment_counts(n) for width, height in grids(n)]


class RandomDecisions:
    """Stands in for DQNDecisionModule: an untrained agent choosing actions uniformly at random, without building networks."""
    def __init__(self, agent_type, unique_id, training, actions, n_features, checkpoint_path, epsilon, shared_replay_buffer=None):
        self.actions = actions

    def choose_action(self, observation):
        return np.random.randint(len(self.actions))

    def learn(self, *args):
        pass

    def save_models(self):
        pass

    def get_epsilon(self):
        return 0.0

    def start_episode(self):
        pass

    def get_mean_loss(self):
        return 0


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def random_agents(monkeypatch):
    monkeypatch.setattr("src.agent.harvest_agent.DQNDecisionModule", RandomDecisions)


def build(num_agents, num_allotments, width, height, seasons=False, delays=None, society_mix="homogeneous",
          mixed_allotments="split", agent_type="utilitarian", training=False, checkpoint_path="unused/", max_episodes=1):
    return AllotmentHarvest(society_mix, num_agents, num_agents * 3, num_allotments, agent_type, width, height, max_episodes, MAX_DAYS,
                            training, checkpoint_path, False, False, "layout", 42, "rule_based", seasons, SEASON_LENGTH, delays,
                            mixed_allotments)


def expected_columns(width, num_allotments):
    base, extra = divmod(width, num_allotments)
    edges = [0]
    for i in range(num_allotments):
        edges.append(edges[-1] + base + (1 if i < extra else 0))
    return list(zip(edges, edges[1:]))


def expected_groups(num_agents, num_allotments):
    size, extra = divmod(num_agents, num_allotments)
    groups, start = [], 0
    for i in range(num_allotments):
        end = start + size + (1 if i < extra else 0)
        groups.append(list(range(start, end)))
        start = end
    return groups


def expected_shares(num_agents, num_allotments):
    share, extra = divmod(TOTAL_BERRIES[num_agents], num_allotments)
    return [share + (1 if i < extra else 0) for i in range(num_allotments)]


def expected_error(num_agents, num_allotments, width, height, seasons):
    """The exception a layout should raise, worked out independently of the model (None if it is valid)."""
    if num_allotments > width:
        return AllotmentWidthException
    for (start, end), group, share in zip(expected_columns(width, num_allotments), expected_groups(num_agents, num_allotments),
                                          expected_shares(num_agents, num_allotments)):
        if (end - start) * height < len(group) + share + (len(group) if seasons else 0):
            return AllotmentCapacityException
    return None


def agents(model):
    return sorted((a for a in model.schedule.agents if a.agent_type != "berry"), key=lambda a: a.unique_id)


def inside(pos, allotment):
    return allotment[0] <= pos[0] < allotment[1] and allotment[2] <= pos[1] < allotment[3]


def assert_everything_inside_its_allotment(model):
    for agent in model.get_living_agents():
        allotment = model.allocations[agent.allocation_id]["allotment"]
        assert [agent.min_width, agent.max_width, agent.min_height, agent.max_height] == allotment
        mm = agent.moving_module
        assert [mm.min_width, mm.max_width, mm.min_height, mm.max_height] == allotment
        assert inside(agent.pos, allotment)
    for b in model.berries:
        allotment = model.allocations[b.allocation_id]["allotment"]
        assert [b.min_width, b.max_width, b.min_height, b.max_height] == allotment
        if b in model.regrowth_timers:
            assert b.pos is None and b.foraged
        else:
            assert inside(b.pos, allotment) and not b.foraged


@pytest.mark.parametrize("seasons", [False, True])
@pytest.mark.parametrize("num_agents,num_allotments,width,height", LAYOUTS)
def test_allotments_tile_the_grid_and_everything_starts_inside_its_allotment(in_tmp_dir, random_agents, num_agents, num_allotments, width, height, seasons):
    error = expected_error(num_agents, num_allotments, width, height, seasons)
    if error is not None:
        with pytest.raises(error):
            build(num_agents, num_allotments, width, height, seasons=seasons)
        return
    model = build(num_agents, num_allotments, width, height, seasons=seasons)
    allotments = [data["allotment"] for data in model.allocations.values()]
    # consecutive [start, end) columns from 0 to width with no gaps or overlaps, every allotment at least 1 column and the full height
    assert [(a[0], a[1]) for a in allotments] == expected_columns(width, num_allotments)
    assert allotments[0][0] == 0 and allotments[-1][1] == width
    assert all(a[1] - a[0] >= 1 and a[2] == 0 and a[3] == height for a in allotments)
    assert all(previous[1] == following[0] for previous, following in zip(allotments, allotments[1:]))
    cells = Counter((x, y) for a in allotments for x in range(a[0], a[1]) for y in range(a[2], a[3]))
    assert len(cells) == width * height and set(cells.values()) == {1}
    # every agent is in exactly one allotment
    assert [data["agent_ids"] for data in model.allocations.values()] == expected_groups(num_agents, num_allotments)
    assert [a.allocation_id for a in agents(model)] == [model._get_allocation_id(i) for i in range(num_agents)]
    # berries: an equal split of the society's total, which is num_start_berries
    assert model.num_start_berries == TOTAL_BERRIES[num_agents]
    assert [data["berry_allocation"] for data in model.allocations.values()] == expected_shares(num_agents, num_allotments)
    if not seasons:
        assert len(model.berries) == model.num_berries == model.num_start_berries
        assert Counter(b.allocation_id for b in model.berries) == {h: d["berry_allocation"] for h, d in model.allocations.items() if d["berry_allocation"]}
    assert Counter(b.allocation_id for b in model.berries) == {h: t for h, t in model._season_targets().items() if t}
    assert model.allotment_regrowth_delays == [2 if i == AllotmentHarvest.SLOW_ALLOTMENT else 0 for i in range(num_allotments)]
    assert_everything_inside_its_allotment(model)


VALID_LAYOUTS = [layout for layout in LAYOUTS if expected_error(*layout, seasons=True) is None]


@pytest.mark.parametrize("seasons,delays", [(False, "none"), (True, "none"), (False, "some"), (True, "some")])
@pytest.mark.parametrize("num_agents,num_allotments,width,height", VALID_LAYOUTS)
def test_short_run_with_untrained_agents_completes(in_tmp_dir, random_agents, num_agents, num_allotments, width, height, seasons, delays):
    allotment_delays = [0] * num_allotments if delays == "none" else [i % 3 + 1 for i in range(num_allotments)]
    model = build(num_agents, num_allotments, width, height, seasons=seasons, delays=allotment_delays)
    assert_everything_inside_its_allotment(model)
    while model.episode == 1:
        model.step()
        if model.episode == 1:
            assert_everything_inside_its_allotment(model)
            assert Counter(b.allocation_id for b in model.berries) == {h: t for h, t in model._season_targets().items() if t}
    assert model.episode == 2
    assert_everything_inside_its_allotment(model)


@pytest.mark.parametrize("num_agents", AGENT_COUNTS)
def test_short_run_with_untrained_networks_completes_for_every_agent_count(in_tmp_dir, num_agents):
    # real, newly made DQNs (training, so nothing is loaded from a checkpoint), on run.py's grid with seasons and the default delays
    model = build(num_agents, 2, 4 * num_agents, 2 * num_agents, seasons=True, training=True,
                  checkpoint_path=str(in_tmp_dir / "checkpoints") + "/", max_episodes=0)
    for agent in agents(model):
        assert agent.observe().shape == (4 + num_agents - 1,)
    for _ in range(MAX_DAYS):
        model.step()
        assert_everything_inside_its_allotment(model)
    assert model.episode == 2


@pytest.mark.parametrize("num_allotments", [0, -1, NUM_AGENTS + 1])
def test_number_of_allotments_outside_1_to_num_agents_raises(in_tmp_dir, random_agents, num_allotments):
    with pytest.raises(NumAllotmentsException, match="between 1 and the number of agents"):
        build(NUM_AGENTS, num_allotments, 16, 8)


def test_allotments_narrower_than_one_column_raise(in_tmp_dir, random_agents):
    with pytest.raises(AllotmentWidthException, match="at least 1 column wide"):
        build(10, 8, 7, 20)


@pytest.mark.parametrize("seasons", [False, True])
def test_allotment_without_enough_cells_raises(in_tmp_dir, random_agents, seasons):
    # each allotment has 2 agents and 6 berries, so needs 8 cells, or 10 with seasons (2 more berries in a good season)
    with pytest.raises(AllotmentCapacityException, match="not enough"):
        build(4, 2, 4, 2, seasons=seasons)
    # 2x4 = 8 cells: enough without seasons, not with them
    if seasons:
        with pytest.raises(AllotmentCapacityException, match="not enough"):
            build(4, 2, 4, 4, seasons=True)
    else:
        build(4, 2, 4, 4)


@pytest.mark.parametrize("width,height", [(16, 8), (8, 8)])
def test_paper_configuration_keeps_the_same_allotments(in_tmp_dir, random_agents, width, height):
    # run.py's allotment grid for 4 agents is 16x8; boundaries are what max_width // num_allotments gave before
    model = build(4, 2, width, height)
    half = width // 2
    assert model.allocations == {
        "allocation_0": {"id": 0, "berry_allocation": 6, "allotment": [0, half, 0, height], "agent_ids": [0, 1]},
        "allocation_1": {"id": 1, "berry_allocation": 6, "allotment": [half, width, 0, height], "agent_ids": [2, 3]},
    }
    assert model.allotment_regrowth_delays == [0, 2]


@pytest.mark.parametrize("num_agents,num_allotments,width,height",
                         [layout for layout in LAYOUTS if layout[2] % layout[1] == 0 and expected_error(*layout, seasons=False) is None])
def test_evenly_dividing_widths_keep_the_old_boundaries(in_tmp_dir, random_agents, num_agents, num_allotments, width, height):
    model = build(num_agents, num_allotments, width, height)
    interval = width // num_allotments
    assert [data["allotment"] for data in model.allocations.values()] == [[i * interval, (i + 1) * interval, 0, height] for i in range(num_allotments)]


def types_by_allotment(model):
    return {holder: [a.agent_type for a in agents(model) if a.allocation_id == holder] for holder in model.allocations}


@pytest.mark.parametrize("num_agents,num_allotments", [(4, 2), (6, 2), (6, 3), (8, 4), (20, 10), (4, 4)])
def test_split_mixed_allotments_alternate_agent_types(in_tmp_dir, random_agents, num_agents, num_allotments):
    model = build(num_agents, num_allotments, 4 * num_agents, 2 * num_agents, society_mix="heterogeneous")
    assert [data["agent_ids"] for data in model.allocations.values()] == expected_groups(num_agents, num_allotments)
    assert [a.agent_type for a in agents(model)] == ["utilitarian" if i % 2 == 0 else "baseline" for i in range(num_agents)]
    for holder, types in types_by_allotment(model).items():
        assert len(types) == len(model.allocations[holder]["agent_ids"])
        if len(types) >= 2:
            assert set(types) == {"utilitarian", "baseline"}


@pytest.mark.parametrize("num_agents,num_allotments,groups", [
    (4, 2, [[0, 1], [2, 3]]),
    (6, 2, [[0, 1, 2], [3, 4, 5]]),
    (8, 4, [[0, 1], [2, 3], [4, 5], [6, 7]]),
    (6, 4, [[0, 1], [2], [3, 4], [5]]),
    (10, 2, [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]),
])
def test_separate_mixed_allotments_give_each_agent_type_half_the_allotments(in_tmp_dir, random_agents, num_agents, num_allotments, groups):
    model = build(num_agents, num_allotments, 4 * num_agents, 2 * num_agents, society_mix="heterogeneous", mixed_allotments="separate")
    assert [data["agent_ids"] for data in model.allocations.values()] == groups
    types = list(types_by_allotment(model).values())
    half = num_allotments // 2
    assert all(set(t) == {"utilitarian"} for t in types[:half])
    assert all(set(t) == {"baseline"} for t in types[half:])
    assert sum(len(t) for t in types) == num_agents


def test_mixed_allotments_are_ignored_for_a_homogeneous_society(in_tmp_dir, random_agents):
    split = build(6, 3, 24, 12)
    separate = build(6, 3, 24, 12, mixed_allotments="separate")
    assert split.allocations == separate.allocations
    assert {a.agent_type for a in agents(separate)} == {"utilitarian"}


def test_separate_mixed_allotments_with_an_odd_number_of_allotments_raise(in_tmp_dir, random_agents):
    with pytest.raises(MixedSocietyException, match="even"):
        build(6, 3, 24, 12, society_mix="heterogeneous", mixed_allotments="separate")


def test_unknown_mixed_allotments_raise(in_tmp_dir, random_agents):
    with pytest.raises(MixedSocietyException, match="split"):
        build(4, 2, 16, 8, society_mix="heterogeneous", mixed_allotments="shared")


@pytest.mark.parametrize("scenario", ["basic", "colours", "capabilities"])
def test_mixed_society_with_an_odd_number_of_agents_raises(in_tmp_dir, random_agents, monkeypatch, scenario):
    monkeypatch.setattr("tests.conftest.NUM_AGENTS", 5)
    with pytest.raises(MixedSocietyException, match="even"):
        _build(scenario, "heterogeneous", 15, "utilitarian", scenario == "basic", str(in_tmp_dir) + "/", False, False, scenario, 1, 10,
               num_allocations=1)


def test_mixed_allotment_society_with_an_odd_number_of_agents_raises(in_tmp_dir, random_agents):
    with pytest.raises(MixedSocietyException, match="even"):
        build(5, 1, 20, 10, society_mix="heterogeneous")


@pytest.mark.parametrize("scenario", ["colours", "capabilities"])
def test_mixed_society_agents_are_in_the_groups_that_list_them(in_tmp_dir, random_agents, scenario):
    model = _build(scenario, "heterogeneous", NUM_AGENTS * 3, "utilitarian", False, str(in_tmp_dir) + "/", False, False, scenario, 1, 10)
    members = Counter(a.allocation_id for a in agents(model))
    for holder, data in model.allocations.items():
        assert [a.unique_id for a in agents(model) if a.allocation_id == holder] == data["agent_ids"]
        assert members[holder] == len(data["agent_ids"])
    assert [a.agent_type for a in agents(model)] == ["utilitarian"] * (NUM_AGENTS // 2) + ["baseline"] * (NUM_AGENTS // 2)

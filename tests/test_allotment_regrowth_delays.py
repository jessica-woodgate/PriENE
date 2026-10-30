from collections import Counter

import pandas as pd
import pytest

from src.harvest_exception import RegrowthDelaysException, RegrowthDelaysNotSupported
from src.harvest_model import HarvestModel
from tests.conftest import NUM_AGENTS, _build, _make_pretrained_checkpoint, long_lived

SEASON_LENGTH = 3
MAX_DAYS = 20


@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory):
    return _make_pretrained_checkpoint(tmp_path_factory.mktemp("regrowth_delays"), "baseline")


@pytest.fixture(scope="module")
def utilitarian_checkpoint_path(tmp_path_factory):
    return _make_pretrained_checkpoint(tmp_path_factory.mktemp("regrowth_delays_utilitarian"), "utilitarian")


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def build(checkpoint_path, delays, seasons=False, seed=42, max_days=MAX_DAYS, max_episodes=2,
          agent_type="baseline", scenario="allotment", **kwargs):
    return long_lived(_build(scenario, "homogeneous", NUM_AGENTS * 3, agent_type, False, checkpoint_path,
                             False, False, f"test_{scenario}", max_episodes, max_days, seed=seed,
                             seasons=seasons, season_length=SEASON_LENGTH, allotment_regrowth_delays=delays, **kwargs))


def berries_of(model, holder):
    return [b for b in model.berries if b.allocation_id == holder]


def assert_every_berry_on_the_grid_or_waiting(model):
    for b in model.berries:
        if b in model.regrowth_timers:
            assert b.foraged and b.pos is None
        else:
            assert not b.foraged and b.pos is not None
            region = model.allocations[b.allocation_id]["allotment"]
            assert region[0] <= b.pos[0] < region[1] and region[2] <= b.pos[1] < region[3]


def forage_without_agents_acting(model, berries):
    """Forage berries and start their timers as the end of a step does, without agents acting."""
    for b in berries:
        b.foraged = True
    model._update_schedule()


def run_episode(model, check=None):
    """Steps through the current episode, calling check(model) after every step within it."""
    episode = model.episode
    while True:
        model.step()
        if model.episode != episode:
            return
        if check is not None:
            check(model)


@pytest.mark.parametrize("num_allocations,shares,delays", [(1, [12], [0]), (2, [6, 6], [0, 2]), (3, [4, 4, 4], [0, 2, 0]), (4, [3, 3, 3, 3], [0, 2, 0, 0])])
def test_by_default_allotments_split_berries_equally_and_allotment_1_regrows_slower(in_tmp_dir, checkpoint_path, num_allocations, shares, delays):
    model = build(checkpoint_path, None, num_allocations=num_allocations)
    assert [data["berry_allocation"] for data in model.allocations.values()] == shares
    assert model.num_berries == sum(shares) == NUM_AGENTS * 3
    assert Counter(b.allocation_id for b in model.berries) == dict(zip(model.allocations, shares))
    assert model.allotment_regrowth_delays == delays
    assert model.holder_regrowth_delays == dict(zip(model.allocations, delays))


def test_equal_split_gives_the_first_allotments_any_berries_left_over(in_tmp_dir, checkpoint_path, monkeypatch):
    def resources(self, num_agents):
        # like the real one, the total sets num_start_berries
        self.num_start_berries = 13
        return [5, 4, 2, 2]
    monkeypatch.setattr(HarvestModel, "_generate_resource_allocations", resources)
    model = build(checkpoint_path, None, num_allocations=3)
    assert [data["berry_allocation"] for data in model.allocations.values()] == [5, 4, 4]


def test_default_gives_the_same_outputs_as_delays_0_2(in_tmp_dir, checkpoint_path):
    reports = []
    for delays in [None, [0, 2]]:
        model = _build("allotment", "homogeneous", NUM_AGENTS * 3, "baseline", False, checkpoint_path, True, False,
                       f"allotment_{delays}", 2, MAX_DAYS, seed=42, seasons=True, season_length=SEASON_LENGTH,
                       allotment_regrowth_delays=delays)
        while model.episode <= 2:
            model.step()
        reports.append({report: pd.read_csv(f"data/results/current_run/{report}_allotment_{delays}.csv")
                        for report in ["agent_reports", "model_episode_reports"]})
    default, explicit = reports
    for report in ["agent_reports", "model_episode_reports"]:
        pd.testing.assert_frame_equal(default[report], explicit[report])


def test_all_zero_delays_never_start_a_timer(in_tmp_dir, checkpoint_path):
    model = build(checkpoint_path, [0, 0], seasons=True)

    def check(m):
        assert m.regrowth_timers == {}
        assert all(not b.foraged and b.pos is not None for b in m.berries)
    run_episode(model, check)
    run_episode(model, check)


@pytest.mark.parametrize("delays", [[0, 1], [0, 3], [2, 0], [4, 4]])
def test_foraged_berry_reappears_exactly_its_allotments_delay_later(in_tmp_dir, checkpoint_path, delays):
    model = build(checkpoint_path, delays)
    model.step()
    for holder, delay in zip(model.allocations, delays):
        berry = berries_of(model, holder)[0]
        berry.foraged = True
        model.step()
        foraged_day = model.day
        if delay == 0:
            # instant regrowth, as without delays
            assert berry not in model.regrowth_timers and not berry.foraged and berry.pos is not None
            continue
        assert model.regrowth_timers[berry] == foraged_day + delay
        while model.day < foraged_day + delay:
            assert berry.pos is None and berry.foraged
            assert berry in model.regrowth_timers
            model.step()
        assert model.day == foraged_day + delay
        assert berry not in model.regrowth_timers and berry.pos is not None
        assert_every_berry_on_the_grid_or_waiting(model)


def test_berries_waiting_to_regrow_count_towards_their_allotment(in_tmp_dir, checkpoint_path):
    model = build(checkpoint_path, [3, 3])
    original = {holder: data["berry_allocation"] for holder, data in model.allocations.items()}

    def check(m):
        # without seasons every allotment keeps its allocation, counting berries waiting to regrow
        assert {holder: len(berries_of(m, holder)) for holder in m.allocations} == original
        assert_every_berry_on_the_grid_or_waiting(m)
    run_episode(model, check)


@pytest.mark.parametrize("seasons", [False, True])
def test_all_timers_are_reset_at_the_start_of_every_episode(in_tmp_dir, checkpoint_path, seasons):
    model = build(checkpoint_path, [5, 7], seasons=seasons, max_episodes=3)
    for episode in [1, 2]:
        run_episode(model)
        # nothing waits across the episode boundary: every berry is back on the grid, uncollected
        assert model.episode == episode + 1 and model.day == 0
        assert model.regrowth_timers == {}
        assert all(not b.foraged and b.pos is not None for b in model.berries)
        assert len(model.berries) == model.num_berries == sum(1 for a in model.schedule.agents if a.agent_type == "berry")
        assert_every_berry_on_the_grid_or_waiting(model)


def test_reset_clears_timers_that_are_waiting_at_the_end_of_an_episode(in_tmp_dir, checkpoint_path):
    model = build(checkpoint_path, [9, 9])
    model.step()
    forage_without_agents_acting(model, model.berries[:4])
    assert len(model.regrowth_timers) == 4
    model.finish_episode()
    assert model.episode == 2 and model.regrowth_timers == {}
    assert all(not b.foraged and b.pos is not None for b in model.berries)
    assert_every_berry_on_the_grid_or_waiting(model)


def test_delays_stay_with_their_allotments_through_seasons(in_tmp_dir, checkpoint_path):
    delays = [0, 4]
    model = build(checkpoint_path, delays, seasons=True, max_episodes=3)
    holder_delays = {holder: delays[data["id"]] for holder, data in model.allocations.items()}

    def check(m):
        assert m.holder_regrowth_delays == holder_delays
        for agent in m.living_agents:
            assert m._allotment_regrowth_delay(agent) == holder_delays[agent.allocation_id]
        for berry, due in m.regrowth_timers.items():
            # timers only run in the allotment with a delay, and never for longer than its delay
            assert berry.allocation_id == "allocation_1"
            assert m.day < due <= m.day + holder_delays["allocation_1"]
        assert_every_berry_on_the_grid_or_waiting(m)
    for _ in range(3):
        run_episode(model, check)


def test_bad_season_removes_berries_on_the_grid_before_cancelling_timers(in_tmp_dir, checkpoint_path, monkeypatch):
    sequence = iter(["normal", "bad"])
    monkeypatch.setattr(HarvestModel, "_draw_season_quality", lambda self: next(sequence))
    model = build(checkpoint_path, [6, 6], seasons=True)
    model.step()
    holder = "allocation_0"
    held = berries_of(model, holder)
    forage_without_agents_acting(model, held[1:])
    on_grid = [b for b in held if b not in model.regrowth_timers]
    assert len(on_grid) == 1
    group_size = len(model.allocations[holder]["agent_ids"])
    model._start_new_season()
    assert model._current_season_quality() == "bad"
    remaining = berries_of(model, holder)
    assert len(remaining) == len(held) - group_size
    # the berry on the grid went first, then timers were cancelled for the rest of the cut
    assert on_grid[0] in model.dormant_berries
    cancelled = [b for b in held if b not in remaining and b is not on_grid[0]]
    assert len(cancelled) == group_size - 1
    assert all(b in model.dormant_berries and b not in model.regrowth_timers and b not in model.schedule.agents for b in cancelled)
    assert all(b in model.regrowth_timers for b in remaining)


def test_seasons_keep_allotments_at_their_targets(in_tmp_dir, checkpoint_path):
    model = build(checkpoint_path, [1, 3], seasons=True, max_episodes=3)
    num_original = len(model.original_berries)

    def check(m):
        targets = m._season_targets()
        assert {holder: len(berries_of(m, holder)) for holder in m.allocations} == targets
        assert len(m.berries) == m.num_berries
        assert len(m.berries) + len(m.dormant_berries) <= num_original + NUM_AGENTS
        assert_every_berry_on_the_grid_or_waiting(m)
    check(model)
    for _ in range(3):
        run_episode(model, check)
        if model.episode <= 3:
            check(model)


def test_two_agent_types_with_the_same_seed_get_the_same_seasons(in_tmp_dir, checkpoint_path, utilitarian_checkpoint_path):
    def boundaries(model):
        recorded = []
        while model.episode <= 2:
            model.step()
            if model.day > 0 and model.day % SEASON_LENGTH == 0:
                recorded.append((model.episode, model.day, model._current_season_quality(),
                                 Counter(b.allocation_id for b in model.berries)))
        return recorded
    baseline = boundaries(build(checkpoint_path, [1, 3], seasons=True))
    utilitarian = boundaries(build(utilitarian_checkpoint_path, [1, 3], seasons=True, agent_type="utilitarian"))
    assert len(baseline) == 2 * (MAX_DAYS // SEASON_LENGTH)
    assert baseline == utilitarian


def test_observation_and_actions_match_with_and_without_delays(in_tmp_dir, checkpoint_path):
    on_model = build(checkpoint_path, [2, 5])
    off_model = build(checkpoint_path, [0, 0])
    for model in [on_model, off_model]:
        for _ in range(3):
            model.step()
    for on_agent, off_agent in zip(on_model.living_agents, off_model.living_agents):
        assert on_agent.observe().shape == off_agent.observe().shape
        assert on_agent.actions == off_agent.actions


def test_berry_allocations_and_placement_match_with_and_without_delays(in_tmp_dir, checkpoint_path):
    on_model = build(checkpoint_path, [2, 5])
    off_model = build(checkpoint_path, [0, 0])
    assert on_model.allocations == off_model.allocations
    berries = lambda model: [(b.pos, b.allocation_id) for b in model.berries]
    assert berries(on_model) == berries(off_model)


def test_output_records_each_agents_allotment_delay(in_tmp_dir, checkpoint_path):
    model = long_lived(_build("allotment", "homogeneous", NUM_AGENTS * 3, "baseline", False, checkpoint_path, True, False,
                              "allotment_regrow2-5", 1, 10, seed=42, allotment_regrowth_delays=[2, 5]))
    while model.episode <= 1:
        model.step()
    reports = pd.read_csv("data/results/current_run/agent_reports_allotment_regrow2-5.csv")
    for agent_id in range(NUM_AGENTS):
        holder = model._get_allocation_id(agent_id)
        expected = [2, 5][model.allocations[holder]["id"]]
        assert (reports.loc[reports["agent_id"] == agent_id, "allotment_regrowth_delay"] == expected).all()


@pytest.mark.parametrize("delays", [[0], [0, 1, 2], [], [0, -1], [-2, 0], [0, 1.5], [True, 0], ["1", 0]])
def test_invalid_delays_raise(in_tmp_dir, checkpoint_path, delays):
    with pytest.raises(RegrowthDelaysException):
        build(checkpoint_path, delays)


@pytest.mark.parametrize("scenario", ["colours", "capabilities"])
def test_delays_with_other_scenarios_raise(in_tmp_dir, checkpoint_path, scenario):
    with pytest.raises(RegrowthDelaysNotSupported, match="allotment"):
        build(checkpoint_path, [0, 1], scenario=scenario)


def test_delays_with_basic_raise(in_tmp_dir):
    with pytest.raises(RegrowthDelaysNotSupported, match="allotment"):
        _build("basic", "homogeneous", NUM_AGENTS * 3, "baseline", True, str(in_tmp_dir / "ckpt") + "/",
               False, False, "basic", 2, 10, allotment_regrowth_delays=[0, 1])

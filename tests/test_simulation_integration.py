import numpy as np
import pandas as pd
import pytest

from tests.conftest import AGENT_TYPES, SCENARIOS, make_scenario_model, make_training_model


# --- A: state-consistency invariants (deterministic, not dependent on agent behaviour) ---

def test_day_counter_starts_the_same_way_on_construction_and_reset(tmp_path):
    """
    Regression test for a bug where HarvestModel.__init__ set self.day = 1 while _reset() (which
    runs at the start of every subsequent episode) set self.day = 0, making the very first episode
    of a run one round shorter than every episode after it. max_days=10 is comfortably short enough
    that this check doesn't depend on it actually being reached.
    """
    model = make_training_model(tmp_path, "baseline", max_days=10)
    day_after_init = model.day
    model._reset()
    assert model.day == day_after_init


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_episodes_never_run_past_max_days_and_agent_and_berry_counts_are_conserved(tmp_path, scenario):
    """
    Episode length can legitimately vary -- a society that manages resources well survives to
    max_days, one that doesn't dies out early -- so this only asserts the upper bound: self.day must
    never be observed (after step() returns) at or beyond max_days, since finish_episode()/_reset()
    always fire within the step() that crosses it. Counts are checked at every step (not just at
    episode boundaries, where HarvestModel._reset() already self-validates via NumAgentsException/
    NumBerriesException) -- catching bookkeeping corruption that balances out by the episode boundary.
    """
    model = make_scenario_model(tmp_path, scenario, "baseline", max_days=10, max_episodes=3)
    for _ in range(40):
        model.step()
        assert model.day < model.max_days
        non_berry_agents = [a for a in model.schedule.agents if a.agent_type != "berry"]
        berries = [a for a in model.schedule.agents if a.agent_type == "berry"]
        assert len(non_berry_agents) == model.num_agents
        assert len(berries) == model.num_berries
        assert len(model.living_agents) <= model.num_agents


# --- B: crash/regression smoke tests ---

@pytest.mark.parametrize("agent_type", AGENT_TYPES)
def test_render_pygame_does_not_raise_for_any_agent_type(tmp_path, agent_type, monkeypatch):
    """
    Regression test for a bug where RenderPygame.agent_colours was missing entries for the four
    aggregation agent types (average/majoritarian/optimist/veto), causing a KeyError as soon as
    rendering was enabled for them.
    """
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")  # no real display needed
    from src.data_handling.render_pygame import RenderPygame

    model = make_training_model(tmp_path, agent_type, max_days=10, max_episodes=2)
    render_inst = RenderPygame(model.max_width, model.max_height)
    render_inst.render_pygame(model)


@pytest.mark.parametrize("scenario, reachable", [("allotment", False), ("capabilities", True)])
def test_berries_can_only_be_thrown_within_an_allotment(tmp_path, scenario, reachable):
    """
    Two agents share allocation_0 and two share allocation_1. In allotment a berry thrown
    to the other allotment is out of reach; in capabilities every agent shares the grid.
    """
    model = make_scenario_model(tmp_path, scenario, "baseline")
    agents = {a.unique_id: a for a in model.get_living_agents()}
    (thrower_id, same_id), (other_id, _) = [data["agent_ids"] for data in model.allocations.values()]
    thrower = agents[thrower_id]
    thrower.berries = 2
    thrower.health = thrower.low_health_threshold

    assert thrower._throw(same_id) == thrower.rewards["throw"]
    other_allotment_health = agents[other_id].health
    reward = thrower._throw(other_id)
    if reachable:
        assert reward == thrower.rewards["throw"]
        assert thrower.berries == 0
    else:
        assert reward == thrower.rewards["no_benefactor"]
        assert thrower.berries == 1
        assert agents[other_id].health == other_allotment_health


@pytest.mark.parametrize("scenario", ["colours", "capabilities", "allotment"])
def test_agents_are_given_allocations_afresh_each_episode(tmp_path, scenario):
    model = make_scenario_model(tmp_path, scenario, "baseline", max_days=3, max_episodes=8)
    sizes = [len(data["agent_ids"]) for data in model.allocations.values()]
    resources = {holder: data["berry_allocation"] for holder, data in model.allocations.items()}
    groupings = []
    while model.episode <= 8:
        held = [sorted(data["agent_ids"]) for data in model.allocations.values()]
        groupings.append(tuple(map(tuple, held)))
        # each allocation keeps its berries and its number of agents; only which agents hold it changes
        assert [len(ids) for ids in held] == sizes
        assert {holder: data["berry_allocation"] for holder, data in model.allocations.items()} == resources
        assert sorted(i for ids in held for i in ids) == list(range(model.num_agents))
        for agent in model.get_living_agents():
            assert agent.unique_id in model.allocations[agent.allocation_id]["agent_ids"]
            assert agent.moving_module.allocation_id == agent.allocation_id
        episode = model.episode
        while model.episode == episode:
            model.step()
    # no agent has the same allocation every episode
    assert all(len({grouping[i] for grouping in groupings}) > 1 for i in range(len(sizes)))


def test_agent_reports_record_who_a_berry_was_thrown_to(tmp_path):
    model = make_scenario_model(tmp_path, "colours", "utilitarian")
    agents = {a.unique_id: a for a in model.get_living_agents()}
    thrower = agents[0]
    throw_to_2 = thrower.actions.index("throw_2")

    def report_row(berries):
        thrower.berries = berries
        thrower.health = thrower.low_health_threshold
        thrower.perform_transition(throw_to_2, thrower.observe())
        model._collect_agent_data(thrower)
        return model.agent_reporter.iloc[-1]
    row = report_row(berries=1)
    assert row["thrown_to"] == 2 and row["action"] == throw_to_2 and agents[2].berries_consumed == 1
    # a throw without a berry gives nothing, so it records no one; nor does any other action
    assert pd.isna(report_row(berries=0)["thrown_to"])
    thrower.perform_transition(thrower.actions.index("eat"), thrower.observe())
    model._collect_agent_data(thrower)
    assert pd.isna(model.agent_reporter.iloc[-1]["thrown_to"])


def test_a_berry_cannot_be_thrown_to_an_agent_that_died_earlier_in_the_step(tmp_path):
    """
    Regression test: an agent that dies stays in living_agents until the end of the step, and a berry thrown to it was accepted
    (rewarded as a throw, and lost)
    """
    model = make_scenario_model(tmp_path, "basic", "baseline")
    agents = {a.unique_id: a for a in model.get_living_agents()}
    thrower, dead = agents[0], agents[1]
    thrower.berries = 2
    thrower.health = thrower.low_health_threshold
    dead.done = True
    dead.health = 0
    assert dead in model.get_living_agents()

    assert thrower._throw(1) == thrower.rewards["no_benefactor"]
    assert thrower.berries == 2
    assert thrower.berries_thrown == 0
    assert dead.health == 0
    assert dead.berries_consumed == 0
    #a living agent can still be thrown to
    assert thrower._throw(2) == thrower.rewards["throw"]


@pytest.mark.parametrize("agent_type", AGENT_TYPES)
def test_agent_reports_log_the_ethics_sanction_each_agent_receives_each_step(tmp_path, agent_type):
    model = make_scenario_model(tmp_path, "colours", agent_type, max_days=10, max_episodes=2)
    issued = {}
    for agent in model.get_living_agents():
        issued[agent.unique_id] = []
        if agent_type != "baseline":
            def recording(agent=agent, original=agent._ethics_sanction):
                sanction = original()
                issued[agent.unique_id].append((model.episode, float(np.sum(sanction))))
                return sanction
            agent._ethics_sanction = recording
    for _ in range(15):
        model.step()

    #the reporter holds the current episode's rows
    reports = model.agent_reporter
    assert len(reports) > 0
    if agent_type == "baseline":
        assert reports["ethics_sanction"].isna().all()
        return
    logged = {agent_id: list(rows["ethics_sanction"]) for agent_id, rows in reports.groupby("agent_id", sort=False)}
    issued = {agent_id: [sanction for episode, sanction in sanctions if episode == model.episode] for agent_id, sanctions in issued.items()}
    assert logged == {agent_id: sanctions for agent_id, sanctions in issued.items() if sanctions}

"""
The agent report's group_thrown_to: for a berry thrown, which group the receiver was in compared with the giver's: "same_group", or
another group that is "better_off_other", "worse_off_other" or "equal_other". A group is better off with more berries this season
and, with as many, a shorter regrowth delay (allotment). Empty when no berry was thrown, and in scenarios without groups (basic, commons).
"""
import pandas as pd
import pytest
from tests.conftest import make_scenario_model

def throw_row(model, thrower, receiver, berries=1):
    thrower.berries = berries
    thrower.health = thrower.low_health_threshold
    thrower.perform_transition(thrower.actions.index(f"throw_{receiver.unique_id}"), thrower.observe())
    model._collect_agent_data(thrower)
    return model.agent_reporter.iloc[-1]

def by_berries(model):
    return {a.unique_id: model._seasonal_berry_target(a) for a in model.get_living_agents()}

def agents_of(model):
    return {a.unique_id: a for a in model.get_living_agents()}

@pytest.mark.parametrize("scenario", ["capabilities", "allotment"])
def test_a_berry_thrown_within_the_group_is_same_group(tmp_path, scenario):
    model = make_scenario_model(tmp_path, scenario, "baseline")
    thrower, *others = model.get_living_agents()
    same = next(a for a in others if a.allocation_id == thrower.allocation_id)
    row = throw_row(model, thrower, same)
    assert row["thrown_to"] == same.unique_id and row["group_thrown_to"] == "same_group"

def test_colours_compare_the_berries_of_each_colour(tmp_path):
    #with 4 agents the colours have 5, 3, 2 and 2 berries, one agent each
    model = make_scenario_model(tmp_path, "colours", "baseline")
    agents, berries = agents_of(model), by_berries(model)
    richest, poorest = max(berries, key=berries.get), min(berries, key=berries.get)
    two = [i for i, b in berries.items() if b == 2]
    assert sorted(berries.values()) == [2, 2, 3, 5] and len(two) == 2
    assert throw_row(model, agents[richest], agents[poorest])["group_thrown_to"] == "worse_off_other"
    assert throw_row(model, agents[poorest], agents[richest])["group_thrown_to"] == "better_off_other"
    assert throw_row(model, agents[two[0]], agents[two[1]])["group_thrown_to"] == "equal_other"

def test_capabilities_compare_the_berries_of_each_group(tmp_path):
    model = make_scenario_model(tmp_path, "capabilities", "baseline")
    agents, berries = agents_of(model), by_berries(model)
    rich = max(berries, key=berries.get)
    poor = min(berries, key=berries.get)
    assert berries[rich] > berries[poor]
    assert throw_row(model, agents[rich], agents[poor])["group_thrown_to"] == "worse_off_other"
    assert throw_row(model, agents[poor], agents[rich])["group_thrown_to"] == "better_off_other"

def test_allotments_with_as_many_berries_compare_regrowth_delays(tmp_path):
    model = make_scenario_model(tmp_path, "allotment", "baseline")
    agents, berries = agents_of(model), by_berries(model)
    #the allotments have as many berries, but allotment 1 regrows 2 days after a berry is foraged and allotment 0 at once
    assert len(set(berries.values())) == 1
    delays = {i: model._allotment_regrowth_delay(a) for i, a in agents.items()}
    fast = next(i for i, d in delays.items() if d == 0)
    slow = next(i for i, d in delays.items() if d > 0)
    assert throw_row(model, agents[fast], agents[slow])["group_thrown_to"] == "worse_off_other"
    assert throw_row(model, agents[slow], agents[fast])["group_thrown_to"] == "better_off_other"

def test_berries_count_before_regrowth_delays(tmp_path, monkeypatch):
    model = make_scenario_model(tmp_path, "allotment", "baseline")
    agents = agents_of(model)
    fast = next(a for a in agents.values() if model._allotment_regrowth_delay(a) == 0)
    slow = next(a for a in agents.values() if model._allotment_regrowth_delay(a) > 0)
    #a season giving the slower allotment more berries makes it the better off one, whatever its delay
    targets = {fast.allocation_id: 4, slow.allocation_id: 6}
    monkeypatch.setattr(model, "_seasonal_berry_target", lambda agent: targets[agent.allocation_id])
    assert throw_row(model, fast, slow)["group_thrown_to"] == "better_off_other"

def test_nothing_is_recorded_when_no_berry_is_thrown(tmp_path):
    model = make_scenario_model(tmp_path, "colours", "baseline")
    thrower, receiver = model.get_living_agents()[:2]
    assert pd.isna(throw_row(model, thrower, receiver, berries=0)["group_thrown_to"])
    #a throw that succeeded, then another action: the next row records nothing
    throw_row(model, thrower, receiver)
    thrower.perform_transition(thrower.actions.index("eat"), thrower.observe())
    model._collect_agent_data(thrower)
    assert pd.isna(model.agent_reporter.iloc[-1]["group_thrown_to"])

def test_allotment_records_a_throw_to_another_allotment(tmp_path):
    model = make_scenario_model(tmp_path, "allotment", "baseline")
    thrower, *others = model.get_living_agents()
    other = next(a for a in others if a.allocation_id != thrower.allocation_id)
    row = throw_row(model, thrower, other)
    assert row["thrown_to"] == other.unique_id and not pd.isna(row["group_thrown_to"])

def test_scenarios_without_groups_record_nothing(tmp_path):
    model = make_scenario_model(tmp_path, "basic", "baseline")
    agents = agents_of(model)
    row = throw_row(model, agents[0], agents[1])
    assert row["thrown_to"] == 1 and pd.isna(row["group_thrown_to"])

def test_the_columns_follow_thrown_to_in_the_agent_report(tmp_path):
    model = make_scenario_model(tmp_path, "colours", "baseline")
    columns = list(model.agent_reporter.columns)
    assert "thrown_to_same_group" not in columns
    assert columns.index("group_thrown_to") == columns.index("thrown_to") + 1
    assert columns.index("agent_thrown_to") == columns.index("thrown_to") + 2


# agent_thrown_to: the receiver compared with the giver by days left to live, before the berry moves

def set_up(agent, health, berries):
    agent.health, agent.berries = health, berries
    agent.days_left_to_live = agent.get_days_left_to_live()

def agent_throw_row(model, thrower, receiver):
    thrower.perform_transition(thrower.actions.index(f"throw_{receiver.unique_id}"), thrower.observe())
    model._collect_agent_data(thrower)
    return model.agent_reporter.iloc[-1]

@pytest.mark.parametrize("scenario", ["basic", "colours", "capabilities", "allotment"])
def test_the_receiver_is_compared_with_the_giver_before_the_berry_moves(tmp_path, scenario):
    model = make_scenario_model(tmp_path, scenario, "baseline")
    thrower, *others = model.get_living_agents()
    receiver = next(a for a in others if model.can_throw_to(thrower, a))
    #before: the giver has (0.6 + 2 * 0.6) / 0.1 = 18 days left and the receiver 14, so the receiver is worse off; after the throw the
    #giver would have 12 (11 after the day's decay) and the receiver 20, so a comparison made after the throw would say better off
    set_up(thrower, 0.6, 2)
    set_up(receiver, 1.4, 0)
    row = agent_throw_row(model, thrower, receiver)
    assert row["thrown_to"] == receiver.unique_id and row["agent_thrown_to"] == "worse_off_other"
    assert receiver.days_left_to_live > thrower.days_left_to_live

def test_a_receiver_with_more_or_as_many_days_left_is_better_off_or_equal(tmp_path):
    model = make_scenario_model(tmp_path, "basic", "baseline")
    thrower, receiver = model.get_living_agents()[:2]
    set_up(thrower, 0.6, 1)
    set_up(receiver, 1.5, 0)
    assert agent_throw_row(model, thrower, receiver)["agent_thrown_to"] == "better_off_other"
    set_up(thrower, 0.6, 1)
    set_up(receiver, 1.2, 0)
    assert agent_throw_row(model, thrower, receiver)["agent_thrown_to"] == "equal_other"

def kill(model, agent):
    agent.health = 0
    agent.done = True
    model._remove_agent(agent)

@pytest.mark.parametrize("berries", [1, 0])
def test_a_throw_tried_to_a_dead_agent_is_recorded_as_dead_other(tmp_path, berries):
    model = make_scenario_model(tmp_path, "basic", "baseline")
    thrower, dead = model.get_living_agents()[:2]
    kill(model, dead)
    set_up(thrower, 0.6, berries)
    row = agent_throw_row(model, thrower, dead)
    #the throw fails (no berry moves, whatever else would have stopped it), but who it was meant for is recorded
    assert pd.isna(row["thrown_to"]) and pd.isna(row["group_thrown_to"]) and row["agent_thrown_to"] == "dead_other"

def test_a_throw_to_an_agent_that_died_earlier_in_the_step_is_dead_other(tmp_path):
    model = make_scenario_model(tmp_path, "basic", "baseline")
    thrower, dying = model.get_living_agents()[:2]
    dying.done = True
    set_up(thrower, 0.6, 1)
    assert agent_throw_row(model, thrower, dying)["agent_thrown_to"] == "dead_other"


# rule-based norms name who a throw goes to (HarvestAgent._throw_norm_label)

def norm_consequents(agent):
    return {key.split(",THEN,")[1] for key in agent.norms_module.behaviour_base}

def throw_norm(tmp_path, scenario, receiver_pick, thrower_state=(0.6, 2), receiver_state=(1.4, 0), dead=False):
    model = make_scenario_model(tmp_path, scenario, "baseline", track_norms=True)
    thrower, *others = model.get_living_agents()
    receiver = receiver_pick(model, thrower, others)
    if dead:
        kill(model, receiver)
    set_up(thrower, *thrower_state)
    set_up(receiver, *receiver_state)
    thrower.norms_module.behaviour_base = {}
    agent_throw_row(model, thrower, receiver)
    return norm_consequents(thrower), model, thrower, receiver

first_other = lambda model, thrower, others: others[0]

def test_a_throw_norm_names_the_receiver_compared_with_the_thrower_before_the_berry_moves(tmp_path):
    #the thrower has 18 days left and the receiver 14 before the throw (12 and 20 after)
    consequents, *_ = throw_norm(tmp_path / "a", "basic", first_other)
    assert consequents == {"throw,worse off agent"}
    consequents, *_ = throw_norm(tmp_path / "b", "basic", first_other, thrower_state=(0.6, 0), receiver_state=(1.4, 0))
    assert consequents == {"throw,better off agent"}
    consequents, *_ = throw_norm(tmp_path / "c", "basic", first_other, thrower_state=(0.6, 0), receiver_state=(0.6, 0))
    assert consequents == {"throw,equally off agent"}

@pytest.mark.parametrize("scenario", ["basic", "capabilities"])
def test_a_throw_norm_names_a_dead_receiver(tmp_path, scenario):
    consequents, *_ = throw_norm(tmp_path, scenario, first_other, dead=True)
    (consequent,) = consequents
    assert consequent.startswith("throw,dead agent")

def test_a_throw_norm_names_the_receivers_group(tmp_path):
    same = lambda model, thrower, others: next(a for a in others if a.allocation_id == thrower.allocation_id)
    consequents, *_ = throw_norm(tmp_path, "capabilities", same)
    assert consequents == {"throw,worse off agent,same group"}

def test_a_throw_norm_names_a_worse_off_receiver_in_a_worse_off_group(tmp_path):
    model = make_scenario_model(tmp_path, "capabilities", "baseline", track_norms=True)
    agents = model.get_living_agents()
    berries = {a: model._seasonal_berry_target(a) for a in agents}
    thrower = max(agents, key=berries.get)
    receiver = min(agents, key=berries.get)
    #the receiver is in the group with fewer berries, and has fewer days left (14 against 18)
    set_up(thrower, 0.6, 2)
    set_up(receiver, 1.4, 0)
    thrower.norms_module.behaviour_base = {}
    agent_throw_row(model, thrower, receiver)
    assert norm_consequents(thrower) == {"throw,worse off agent,worse off group"}

def test_basic_has_no_groups_to_name(tmp_path):
    consequents, *_ = throw_norm(tmp_path, "basic", first_other)
    assert consequents == {"throw,worse off agent"}

def test_other_actions_keep_their_names(tmp_path):
    model = make_scenario_model(tmp_path, "basic", "baseline", track_norms=True)
    agent = model.get_living_agents()[0]
    agent.norms_module.behaviour_base = {}
    agent.perform_transition(agent.actions.index("eat"), agent.observe())
    agent.perform_transition(agent.actions.index("move"), agent.observe())
    assert norm_consequents(agent) == {"eat", "move"}

def test_no_agent_comparison_is_recorded_when_no_berry_is_thrown(tmp_path):
    model = make_scenario_model(tmp_path, "basic", "baseline")
    thrower, receiver = model.get_living_agents()[:2]
    set_up(thrower, 0.6, 0)
    assert pd.isna(agent_throw_row(model, thrower, receiver)["agent_thrown_to"])
    set_up(thrower, 0.6, 1)
    agent_throw_row(model, thrower, receiver)
    thrower.perform_transition(thrower.actions.index("eat"), thrower.observe())
    model._collect_agent_data(thrower)
    assert pd.isna(model.agent_reporter.iloc[-1]["agent_thrown_to"])

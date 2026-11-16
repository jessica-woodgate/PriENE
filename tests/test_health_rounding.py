"""
Health and days left to live are rounded whenever they change (src/agent/harvest_agent.py DECIMALS), so steps of 0.1 that floating point
can't hold exactly never leave an agent just below a threshold it has reached: an agent at 0.6 health can throw, one at 0 health dies,
and days left to live are whole numbers of days.
"""
from tests.conftest import make_scenario_model

def an_agent(tmp_path):
    model = make_scenario_model(tmp_path, "colours", "baseline")
    agents = model.get_living_agents()
    return model, agents[0], agents[1]

def decay(agent, days):
    for _ in range(days):
        agent._update_attributes([0.0])

def test_health_reached_by_steps_of_a_tenth_is_exact(tmp_path):
    _, agent, _ = an_agent(tmp_path)
    #0.8 - 0.1 * 3 + 0.6 - 0.1 * 7 is 0.39999999999999963 in floating point without rounding
    decay(agent, 3)
    agent.berries = 1
    agent._eat()
    decay(agent, 7)
    assert agent.health == 0.4
    assert agent.days_left_to_live == 4

def test_an_agent_whose_health_has_come_down_to_the_threshold_can_throw(tmp_path):
    _, thrower, receiver = an_agent(tmp_path)
    #from 0.8: eat (1.4), then 8 days' decay brings health to 0.6, which is 0.5999999999999996 without rounding
    thrower.berries = 1
    thrower._eat()
    decay(thrower, 8)
    assert thrower.health == thrower.low_health_threshold
    thrower.berries = 1
    assert thrower._throw(receiver.unique_id) == thrower.rewards["throw"]
    assert thrower.current_thrown_to == receiver.unique_id

def test_an_agent_whose_health_has_come_down_to_zero_dies(tmp_path):
    _, agent, _ = an_agent(tmp_path)
    decay(agent, 1)
    agent.berries = 1
    agent._eat()
    agent.berries = 0
    #1.3 comes down to 0 in 13 days; without rounding it is 2.8e-17 then, and the agent lives another day
    for day in range(12):
        done, _ = agent._update_attributes([0.0])
        assert not done
    done, _ = agent._update_attributes([0.0])
    assert done and agent.health == 0 and agent.days_left_to_live == 0

def test_a_thrown_berry_raises_the_receivers_health_exactly(tmp_path):
    _, thrower, receiver = an_agent(tmp_path)
    decay(receiver, 3)
    thrower.berries = 1
    thrower._throw(receiver.unique_id)
    assert receiver.health == 1.1 and receiver.days_left_to_live == 11

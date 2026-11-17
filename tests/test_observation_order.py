"""
An agent observes the other agents' well-being in id order, so its k-th reading is always the agent its action throw_k throws to, step
after step, although Mesa reshuffles the schedule every step (before this was fixed, the readings followed the schedule's order).
"""
import pytest
from tests.conftest import make_scenario_model


@pytest.mark.parametrize("scenario", ["basic", "colours", "capabilities", "allotment"])
def test_each_observed_slot_is_the_agent_its_throw_action_targets(tmp_path, scenario):
    model = make_scenario_model(tmp_path, scenario, "baseline", max_days=20)
    agents = {a.unique_id: a for a in model.get_living_agents()}
    orders = set()
    for _ in range(8):
        model.step()
        orders.add(tuple(a.unique_id for a in model.schedule.agents if a.agent_type != "berry"))
        for agent_id, agent in agents.items():
            agent.days_left_to_live = 10 + agent_id
        for agent in agents.values():
            targets = [int(name.split("_")[1]) for name in agent.actions[2:]]
            slots = list(agent.observe()[4:4 + len(targets)])
            #a dead agent is observed with no days left, in its own slot
            assert slots == [0 if agents[target].done else 10 + target for target in targets]
    # the schedule's order did change from step to step, so the test would have caught slots that follow it
    assert len(orders) > 1

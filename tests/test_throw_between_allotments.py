"""
Throwing berries between allotments (run.py's allotment prompt "Can agents throw berries to agents in other allotments?"): off by default,
so a berry thrown to another allotment is out of reach (test_simulation_integration.py), and when on it reaches agents in any allotment;
runs with it on are named apart, in their checkpoint and their settings.
"""
from tests.conftest import make_scenario_model
from src.checkpoints import training_checkpoint_name, prepare_training_checkpoints, recorded_settings

def thrower_and_others(model):
    agents = {a.unique_id: a for a in model.get_living_agents()}
    (thrower_id, same_id), (other_id, _) = [data["agent_ids"] for data in model.allocations.values()]
    thrower = agents[thrower_id]
    thrower.berries = 2
    thrower.health = thrower.low_health_threshold
    return thrower, agents[same_id], agents[other_id]

def test_throwing_between_allotments_is_off_by_default(tmp_path):
    model = make_scenario_model(tmp_path, "allotment", "baseline")
    assert model.throw_between_allotments is False
    thrower, _, other = thrower_and_others(model)
    assert not model.can_throw_to(thrower, other)

def test_with_throwing_between_allotments_a_berry_reaches_another_allotment(tmp_path):
    model = make_scenario_model(tmp_path, "allotment", "baseline")
    model.throw_between_allotments = True
    thrower, same, other = thrower_and_others(model)
    assert thrower.allocation_id == same.allocation_id != other.allocation_id
    other_health, other_consumed = other.health, other.berries_consumed
    assert thrower._throw(other.unique_id) == thrower.rewards["throw"]
    assert thrower.current_thrown_to == other.unique_id and thrower.berries == 1
    assert other.health == other_health + thrower.berry_health_payoff and other.berries_consumed == other_consumed + 1
    assert thrower._throw(same.unique_id) == thrower.rewards["throw"] and thrower.berries == 0

def test_a_checkpoint_trained_with_throwing_between_allotments_is_named_apart():
    assert training_checkpoint_name("allotment", False, 25, 2, [0, 2], 200, 42) == "allotment_2allotments_regrow0-2_200days_seed42"
    assert (training_checkpoint_name("allotment", False, 25, 2, [0, 2], 200, 42, throw_between_allotments=True)
            == "allotment_2allotments_regrow0-2_throwbetween_200days_seed42")
    #only allotment has allotments to throw between
    assert training_checkpoint_name("capabilities", False, 25, 2, None, 200, 42, throw_between_allotments=True) == "capabilities_2capabilities_200days_seed42"

def test_recorded_settings_say_whether_agents_could_throw_between_allotments(tmp_path):
    on, off = str(tmp_path / "on") + "/", str(tmp_path / "off") + "/"
    prepare_training_checkpoints(on, ["baseline"], {"scenario": "allotment", "throw_between_allotments": True})
    prepare_training_checkpoints(off, ["baseline"], {"scenario": "allotment"})
    assert recorded_settings(on)["throw_between_allotments"] is True
    assert recorded_settings(off).get("throw_between_allotments", False) is False
    assert recorded_settings(str(tmp_path / "missing") + "/") == {}

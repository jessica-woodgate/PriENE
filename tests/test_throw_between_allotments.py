"""
Throwing berries between allotments: allotment agents can always throw berries to agents in any allotment, not only their own (there is
no option to throw only within one); allotment runs are named "throwbetween", apart from runs made when they could only throw within one,
and their checkpoints' settings record it.
"""
import subprocess
import sys
from tests.conftest import make_scenario_model
from src.checkpoints import training_checkpoint_name, prepare_training_checkpoints, recorded_settings
from src.run_naming import scenario_file_string, harvest_condition

def thrower_and_others(model):
    agents = {a.unique_id: a for a in model.get_living_agents()}
    (thrower_id, same_id), (other_id, _) = [data["agent_ids"] for data in model.allocations.values()]
    thrower = agents[thrower_id]
    thrower.berries = 2
    thrower.health = thrower.low_health_threshold
    return thrower, agents[same_id], agents[other_id]

def test_a_berry_reaches_an_agent_in_another_allotment(tmp_path):
    model = make_scenario_model(tmp_path, "allotment", "baseline")
    thrower, same, other = thrower_and_others(model)
    assert thrower.allocation_id == same.allocation_id != other.allocation_id
    assert model.can_throw_to(thrower, other) and model.can_throw_to(thrower, same)
    other_health, other_consumed = other.health, other.berries_consumed
    assert thrower._throw(other.unique_id) == thrower.rewards["throw"]
    assert thrower.current_thrown_to == other.unique_id and thrower.berries == 1
    assert other.health == other_health + thrower.berry_health_payoff and other.berries_consumed == other_consumed + 1
    assert thrower._throw(same.unique_id) == thrower.rewards["throw"] and thrower.berries == 0

def test_allotment_runs_are_named_throwbetween():
    assert training_checkpoint_name("allotment", False, 25, 2, [0, 2], 200, 42) == "allotment_2allotments_regrow0-2_throwbetween_200days_seed42"
    assert scenario_file_string("allotment", False) == "allotment_throwbetween"
    assert harvest_condition("allotment", False, 25, 2, "split", "homogeneous", 4, 200) == "allotment_throwbetween_2groups_4agents_200days"
    #only allotment has allotments to throw between
    assert training_checkpoint_name("capabilities", False, 25, 2, None, 200, 42) == "capabilities_2capabilities_200days_seed42"
    assert scenario_file_string("capabilities", False) == "capabilities"

def test_recorded_settings_say_whether_agents_could_throw_between_allotments(tmp_path):
    on, off = str(tmp_path / "on") + "/", str(tmp_path / "off") + "/"
    prepare_training_checkpoints(on, ["baseline"], {"scenario": "allotment", "throw_between_allotments": True})
    prepare_training_checkpoints(off, ["baseline"], {"scenario": "allotment"})
    assert recorded_settings(on)["throw_between_allotments"] is True
    assert recorded_settings(off).get("throw_between_allotments", False) is False
    assert recorded_settings(str(tmp_path / "missing") + "/") == {}

def test_throwing_between_allotments_is_not_a_flag():
    for value in ["y", "n"]:
        result = subprocess.run([sys.executable, "run.py", "test", "--throw-between-allotments", value], capture_output=True, text=True,
                                timeout=300)
        assert result.returncode != 0 and "is not an option" in result.stderr

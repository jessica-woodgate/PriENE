"""
The harvest scenarios' rewards in the training config (src/training_config.py): harvest.yaml holds the rewards harvest agents have always
had, a config with other rewards reaches the agents and names its checkpoint, a checkpoint's recorded config is found again for testing,
bad rewards are refused, and Fruit Market (whose reward is the substrate's own) refuses a config that changes them.
"""
import json
import pytest
from src import training_config as tc
from src.training_config import TrainingConfig, TrainingConfigError
from src.checkpoints import training_checkpoint_name, prepare_training_checkpoints, checkpoint_training_config

#the rewards HarvestAgent had written into it before they moved to the training config
BASELINE = {"death": -1, "no_berries": -0.2, "no_benefactor": -0.2, "insufficient_health": -0.2, "neutral_reward": 0,
            "throw": 0.5, "forage": 0.5, "eat": 1, "survive": 1}
ETHICS = {**BASELINE, "no_berries": -0.1, "no_benefactor": -0.1, "insufficient_health": -0.1, "eat": 0.8, "sanction": 0.4}

def harvest_agents(tmp_path, monkeypatch, agent_type, config=None):
    monkeypatch.chdir(tmp_path)
    from src.scenarios.basic_harvest import BasicHarvest
    model = BasicHarvest("homogeneous", 2, 6, agent_type, 4, 4, 1, 5, True, "checkpoints/", False, False, seed=1, training_config=config)
    return [a for a in model.schedule.agents if a.agent_type != "berry"]

def test_the_harvest_file_holds_the_rewards_harvest_agents_have_always_had():
    config = tc.for_scenario("basic")
    assert config.rewards == BASELINE and config == tc.DEFAULT
    assert config.harvest_rewards(ethics=False) == BASELINE
    assert config.harvest_rewards(ethics=True) == ETHICS

@pytest.mark.parametrize("agent_type, expected", [("baseline", BASELINE), ("utilitarian", ETHICS)])
def test_harvest_agents_get_the_harvest_files_rewards(tmp_path, monkeypatch, agent_type, expected):
    assert all(a.rewards == expected for a in harvest_agents(tmp_path, monkeypatch, agent_type, tc.for_scenario("basic")))

def test_harvest_agents_get_other_rewards_from_the_config(tmp_path, monkeypatch):
    config = tc.DEFAULT.with_values({"rewards": {**BASELINE, "throw": 0.1, "death": -5}, "ethics_rewards": {"eat": 0.5}})
    baseline = harvest_agents(tmp_path, monkeypatch, "baseline", config)
    assert all(a.rewards == {**BASELINE, "throw": 0.1, "death": -5} for a in baseline)
    ethics = harvest_agents(tmp_path, monkeypatch, "maximin", config)
    #only the ethics rewards the config gives replace the baseline's
    assert all(a.rewards == {**BASELINE, "throw": 0.1, "death": -5, "eat": 0.5, "sanction": 0.4} for a in ethics)

def test_a_config_file_with_other_rewards_loads_and_saves(tmp_path):
    config = tc.DEFAULT.with_values({"rewards": {**BASELINE, "forage": 0.2}})
    path = str(tmp_path / "R2.yaml")
    tc.save(config, path)
    assert tc.load(path) == config and tc.load(path).rewards["forage"] == 0.2

def test_a_file_without_rewards_has_the_harvest_rewards(tmp_path):
    #e.g. configs/training/fruit_market.yaml, and configs recorded before the rewards were part of the training config
    assert tc.for_scenario("fruit_market").rewards == BASELINE
    recorded = {name: value for name, value in tc.DEFAULT.to_dict().items() if name not in tc.HARVEST_ONLY_PARAMETERS}
    assert TrainingConfig(**recorded) == tc.DEFAULT

@pytest.mark.parametrize("values, message", [
    ({"rewards": {**BASELINE, "steal": 1}}, "aren't outcomes"),
    ({"rewards": {"death": -1}}, "doesn't give"),
    ({"ethics_rewards": {"steal": 1}}, "aren't outcomes"),
    ({"rewards": 1}, "must map outcomes")])
def test_bad_rewards_are_refused(values, message):
    with pytest.raises(TrainingConfigError, match=message):
        tc.DEFAULT.with_values(values)

def test_a_checkpoint_trained_with_another_config_is_named_after_it():
    assert training_checkpoint_name("basic", False, 25, 1, None, 200, 42) == "basic_200days_seed42"
    assert training_checkpoint_name("basic", False, 25, 1, None, 200, 42, config_name="R2") == "basic_200days_seed42_R2"
    assert tc.config_name("configs/training/R2.yaml") == "R2"

def test_a_checkpoints_recorded_config_is_found_for_testing(tmp_path):
    config = tc.DEFAULT.with_values({"rewards": {**BASELINE, "throw": 0.1}})
    with_config, without = str(tmp_path / "a") + "/", str(tmp_path / "b") + "/"
    prepare_training_checkpoints(with_config, ["baseline"], {"scenario": "basic", "training_config": config.to_dict()})
    prepare_training_checkpoints(without, ["baseline"], {"scenario": "basic"})
    assert TrainingConfig(**checkpoint_training_config(with_config)) == config
    with open(with_config + "settings.json") as file:
        assert json.load(file)["training_config"]["rewards"]["throw"] == 0.1
    assert checkpoint_training_config(without) is None
    assert checkpoint_training_config(str(tmp_path / "missing") + "/") is None

def test_fruit_market_refuses_a_config_changing_the_harvest_rewards(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.fruit_market.model import FruitMarketModel
    from tests.fruit_market_stand_in import StandInBackend
    config = tc.for_scenario("fruit_market").with_values({"rewards": {**BASELINE, "throw": 0.1}})
    with pytest.raises(TrainingConfigError, match="substrate's own"):
        FruitMarketModel("homogeneous", 2, "maximin", 1, 3, True, "checkpoints/", False, False, seed=1,
                         backend=StandInBackend(["apple_farmer", "banana_farmer"]), training_config=config)

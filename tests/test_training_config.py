"""
Training configs (src/training_config.py): the harvest file holds the values harvest agents have always trained with, Fruit Market's
the same with its sanction; configs refuse scenario parameters; and the values reach the DQN, the epsilon schedule and the sanction.
"""
import math
import numpy as np
import pytest
from src import training_config as tc
from src.training_config import TrainingConfig, TrainingConfigError
from src.agent.dqn.dqn import DQN
from src.agent.modules.dqn_decision_module import DQNDecisionModule

def test_the_harvest_file_holds_the_parameters_harvest_agents_have_always_trained_with():
    config = tc.for_scenario("basic")
    assert config == tc.DEFAULT
    assert config.to_dict() == {"learning_rate": 0.0001, "gamma": 0.95, "batch_size": 64, "hidden_units": 128, "huber_delta": 1.0,
                                "min_experiences": 100, "max_experiences": 100000, "target_update_interval": 50, "epsilon_start": 1.0,
                                "epsilon_min": 0.01, "epsilon_decay": 0.001, "max_training_episodes": None, "sanction": 0.4}

@pytest.mark.parametrize("scenario", ["basic", "colours", "capabilities", "allotment", "commons"])
def test_every_harvest_scenario_uses_the_harvest_file(scenario):
    assert tc.scenario_config_path(scenario).endswith("configs/training/harvest.yaml")

def test_fruit_market_trains_with_the_same_learning_values_and_its_own_sanction():
    assert tc.for_scenario("fruit_market") == tc.DEFAULT.with_values({"sanction": 3.5})

@pytest.mark.parametrize("name", ["num_players", "max_days", "grappling", "well_being", "hunger_delay", "trade_radius", "seed"])
def test_a_training_config_refuses_scenario_parameters(name):
    with pytest.raises(TrainingConfigError, match="parameter of the scenario"):
        tc.DEFAULT.with_values({name: 1})

def test_a_training_config_refuses_unknown_and_missing_parameters(tmp_path):
    with pytest.raises(TrainingConfigError, match="not a training parameter"):
        tc.DEFAULT.with_values({"learning_rat": 0.1})
    path = str(tmp_path / "partial.yaml")
    with open(path, "w") as file:
        file.write("learning_rate: 0.001\n")
    with pytest.raises(TrainingConfigError, match="doesn't give"):
        tc.load(path)

def test_a_saved_config_loads_back_unchanged(tmp_path):
    config = tc.DEFAULT.with_values({"learning_rate": 0.0003, "gamma": 0.99, "max_training_episodes": 300})
    path = str(tmp_path / "run.yaml")
    tc.save(config, path)
    assert tc.load(path) == config

def test_the_dqn_and_decision_module_use_the_configs_values(tmp_path):
    config = tc.DEFAULT.with_values({"learning_rate": 0.0005, "gamma": 0.9, "batch_size": 32, "hidden_units": 16, "huber_delta": 2.0,
                                     "min_experiences": 10, "max_experiences": 500, "target_update_interval": 7, "epsilon_min": 0.05,
                                     "epsilon_decay": 0.01})
    dqn = DQN(list(range(3)), (4,), True, training_config=config)
    assert (dqn.lr, dqn.gamma, dqn.batch_size, dqn.delta, dqn.min_experiences, dqn.max_experiences) == (0.0005, 0.9, 32, 2.0, 10, 500)
    dqn.dqn(np.zeros((1, 4), dtype="float32"))
    assert dqn.dqn.hidden_layer_1.units == 16
    module = DQNDecisionModule("baseline", 0, True, list(range(3)), 4, str(tmp_path) + "/", 1.0, training_config=config)
    assert module.replace_target_iter == 7
    assert module._decayed_epsilon(1) == 1.0
    assert module._decayed_epsilon(101) == pytest.approx(max(0.05, math.exp(-0.01 * 100)))
    assert module._decayed_epsilon(10000) == 0.05

def test_a_compressed_epsilon_schedule_reaches_its_minimum_at_the_given_fraction_of_the_run():
    decay = tc.compressed_epsilon_decay(1.0, 0.01, 300, 0.7)
    assert 1.0 * math.exp(-decay * 210) == pytest.approx(0.01)

def test_harvest_agents_get_their_sanction_from_the_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.scenarios.basic_harvest import BasicHarvest
    model = BasicHarvest("homogeneous", 2, 6, "utilitarian", 4, 4, 1, 5, True, "checkpoints/", False, False, seed=1,
                         training_config=tc.DEFAULT.with_values({"sanction": 0.7}))
    agents = [a for a in model.schedule.agents if a.agent_type != "berry"]
    assert all(a.rewards["sanction"] == 0.7 and a.ethics_module.sanction == 0.7 for a in agents)
    assert all(a.decision_module.training_config.sanction == 0.7 for a in agents)

def test_fruit_market_agents_get_their_sanction_from_the_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.fruit_market.model import FruitMarketModel
    from tests.fruit_market_stand_in import StandInBackend
    roles = ["apple_farmer", "banana_farmer"]
    model = FruitMarketModel("homogeneous", 2, "maximin", 1, 3, True, "checkpoints/", False, False, seed=1, backend=StandInBackend(roles),
                             training_config=tc.for_scenario("fruit_market").with_values({"sanction": 1.0, "epsilon_min": 0.05}))
    assert all(a.ethics_module.sanction == 1.0 for a in model.schedule.agents)
    assert model.min_epsilon == 0.05

"""
The training config's positive_sanction_only (src/training_config.py): off by default, and then everything is as before (the
sanction, and the training config as it is recorded); on, an ethics agent's sanction only rewards: where its principle or aggregation
gives -sanction it gets 0, and +sanction and 0 are unchanged.
"""
import numpy as np
import pytest
from src import training_config as tc
from src.training_config import TrainingConfig, TrainingConfigError
from src.agent.modules.ethics_module import EthicsModule

#well-being before an action, and after one that makes the worst-off worse off and society more unequal (every principle penalises
#it: the minimum falls, inequality grows, the total falls) or better off (every principle rewards it)
BEFORE = [10.0, 20.0, 30.0]
WORSE = [5.0, 20.0, 30.0]
BETTER = [15.0, 20.0, 30.0]
PRINCIPLES = EthicsModule.PRINCIPLES + EthicsModule.AGGREGATIONS


def sanction(principle, after, positive_only):
    module = EthicsModule(0.4, principle, positive_only=positive_only)
    module.update_ethics_state(np.array(BEFORE))
    return module.get_sanction(np.array(after))


@pytest.mark.parametrize("principle", PRINCIPLES)
def test_a_penalty_is_zero_with_a_positive_only_sanction(principle):
    assert np.sum(sanction(principle, WORSE, False)) == pytest.approx(-0.4)
    assert np.sum(sanction(principle, WORSE, True)) == 0


@pytest.mark.parametrize("principle", PRINCIPLES)
def test_a_reward_is_unchanged_with_a_positive_only_sanction(principle):
    assert sanction(principle, BETTER, True) == sanction(principle, BETTER, False)
    assert np.sum(sanction(principle, BETTER, True)) == pytest.approx(0.4)


def test_it_is_off_by_default_and_left_out_of_recorded_configs():
    assert tc.DEFAULT.positive_sanction_only is False
    assert tc.for_scenario("basic").positive_sanction_only is False
    #recorded configs (settings.json, run.json) are as they were before the option existed
    assert "positive_sanction_only" not in tc.DEFAULT.to_dict()
    assert TrainingConfig(**tc.DEFAULT.to_dict()) == tc.DEFAULT


def test_a_config_with_it_loads_saves_and_records_it(tmp_path):
    config = tc.DEFAULT.with_values({"positive_sanction_only": True, "sanction": 1.4})
    assert config.to_dict()["positive_sanction_only"] is True
    path = str(tmp_path / "positive.yaml")
    tc.save(config, path)
    assert tc.load(path) == config
    assert TrainingConfig(**config.to_dict()) == config


def test_it_must_be_true_or_false():
    with pytest.raises(TrainingConfigError, match="true or false"):
        tc.DEFAULT.with_values({"positive_sanction_only": "yes"})


@pytest.mark.parametrize("positive_only", [False, True])
def test_harvest_ethics_agents_get_it_from_the_config(tmp_path, monkeypatch, positive_only):
    monkeypatch.chdir(tmp_path)
    from src.scenarios.basic_harvest import BasicHarvest
    config = tc.DEFAULT.with_values({"positive_sanction_only": positive_only})
    model = BasicHarvest("homogeneous", 2, 6, "maximin", 4, 4, 1, 5, True, "checkpoints/", False, False, seed=1, training_config=config)
    agents = [a for a in model.schedule.agents if a.agent_type != "berry"]
    assert all(a.ethics_module.positive_only is positive_only for a in agents)


def test_fruit_market_refuses_it():
    #Fruit Market's agents have their own sanction; the option is the harvest scenarios' only
    from src.fruit_market.model import FruitMarketModel
    from fruit_market_stand_in import StandInBackend
    with pytest.raises(TrainingConfigError, match="positive_sanction_only"):
        FruitMarketModel("homogeneous", 2, "maximin", 1, 5, True, "checkpoints/", False, False,
                         backend=StandInBackend(["apple_farmer", "banana_farmer"]),
                         training_config=tc.for_scenario("fruit_market").with_values({"positive_sanction_only": True}))

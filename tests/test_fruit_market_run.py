"""
run.py's fruit_market scenario end to end with the real substrate (skipped without Melting Pot): training asks for Fruit Market's options
and saves checkpoints named by them, and testing loads them with the settings they were trained with and writes Fruit Market's reports.
"""
import io
import json
import os
import runpy
import sys
import pandas as pd
import pytest

pytest.importorskip("meltingpot")

from src.fruit_market.model import FruitMarketModel

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = "fruit_market_3steps_grappling-off_wb-need_sanction-attributed_obs-global_seed42"

@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path

@pytest.fixture
def short_training(monkeypatch):
    #training stops once exploration has decayed below this, i.e. a step into its second episode, instead of after ~4600 episodes
    original_init = FruitMarketModel.__init__
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.min_epsilon = 0.9995
    monkeypatch.setattr(FruitMarketModel, "__init__", init)

def run(mode, answers, monkeypatch):
    monkeypatch.setattr(sys, "argv", [os.path.join(REPO, "run.py"), mode, "-s", "42"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(answers) + "\n"))
    with pytest.raises(SystemExit) as finished:
        runpy.run_path(os.path.join(REPO, "run.py"), run_name="__main__")
    assert finished.value.code == 0

def test_fruit_market_trains_then_tests_through_run_py(in_tmp_dir, short_training, monkeypatch):
    #scenario, steps, players, grappling, well-being, sanction, spatial features, society mix, agent type, write data
    run("train", ["fruit_market", "3", "2", "n", "", "", "", "", "utilitarian", ""], monkeypatch)
    checkpoint = f"data/model_variables/trained/{CHECKPOINT}/2_agents/"
    with open(checkpoint + "settings.json") as file:
        assert json.load(file) == {"scenario": "fruit_market", "max_days": 3, "num_agents": 2, "grappling": False, "well_being": "need",
                                   "sanction": "attributed", "observation": "global", "seed": 42, "society_mix": "homogeneous"}
    assert sorted(os.listdir(checkpoint + "utilitarian")) == ["agent_0", "agent_1"]
    training_report = pd.read_csv("data/results/current_run/fm_model_episode_reports_train_fruit_market_3steps_2players_grappling-off_utilitarian.csv")
    assert list(training_report["episode"]) == [1]
    #scenario, checkpoint, test episodes, society mix, agent type, write data, norms, norms module
    run("test", ["fruit_market", CHECKPOINT, "2", "", "utilitarian", "", "y", ""], monkeypatch)
    name = "fruit_market_3steps_2players_grappling-off_utilitarian"
    agent_report = pd.read_csv(f"data/results/current_run/fm_agent_reports_{name}.csv")
    assert len(agent_report) == 2 * 3 * 2
    assert set(agent_report["action"]).isdisjoint({9, 10, 11})
    assert list(pd.read_csv(f"data/results/current_run/fm_model_episode_reports_{name}.csv")["episode"]) == [1, 2]
    assert os.path.exists(f"data/results/current_run/{name}_emerged_norms.json")

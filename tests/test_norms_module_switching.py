import json
import os

import pandas as pd
import pytest

from src.agent.harvest_agent import NORMS_MODULE_TYPES
from src.agent.modules.bayesian_norms_module import BayesianNormsModule
from src.agent.modules.norms_module import NormsModule
from src.harvest_exception import UnrecognisedNormsModule
from src.scenarios.basic_harvest import BasicHarvest
from tests.conftest import SCENARIOS, _build, make_scenario_model

MODULE_CLASSES = {"bayesian": BayesianNormsModule, "rule_based": NormsModule}


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def harvest_agents(model):
    return [a for a in model.schedule.agents if a.agent_type != "berry"]


def run_episodes(model, episodes):
    while model.episode <= episodes:
        model.step()


@pytest.mark.parametrize("norms_module_type", NORMS_MODULE_TYPES)
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_scenario_runs_with_each_norms_module(in_tmp_dir, scenario, norms_module_type):
    model = make_scenario_model(in_tmp_dir, scenario, "utilitarian", max_days=5, max_episodes=2,
                                track_norms=True, norms_module_type=norms_module_type)
    assert all(isinstance(a.norms_module, MODULE_CLASSES[norms_module_type]) for a in harvest_agents(model))
    run_episodes(model, 2)


@pytest.mark.parametrize("agent_type", ["baseline", "utilitarian"])
def test_rule_based_norms_are_tracked_and_written(in_tmp_dir, agent_type):
    model = _build("basic", "homogeneous", 12, agent_type, True, str(in_tmp_dir / "ckpt") + "/",
                   False, True, "rule", 3, 5, norms_module_type="rule_based")
    model.step()
    behaviour_base = harvest_agents(model)[0].norms_module.behaviour_base
    assert behaviour_base and all(name.startswith("IF,") and ",THEN," in name for name in behaviour_base)
    assert model.agent_reporter["num_norms"].notna().all()
    run_episodes(model, 2)
    with open("data/results/current_run/rule_emerged_norms.json") as file:
        assert list(json.load(file)) == ["1", "2"]
    assert not os.path.exists("data/results/current_run/agent_norms_rule.csv")  # Bayesian-only output


def test_bayesian_norms_write_csv_and_no_emerged_norms(in_tmp_dir):
    model = _build("basic", "homogeneous", 12, "utilitarian", True, str(in_tmp_dir / "ckpt") + "/",
                   False, True, "bayes", 3, 5, norms_module_type="bayesian")
    run_episodes(model, 2)
    assert len(pd.read_csv("data/results/current_run/agent_norms_bayes.csv")) == 2 * model.num_agents
    assert not os.path.exists("data/results/current_run/bayes_emerged_norms.json")  # rule-based-only output
    assert model.emerged_norms == {}


@pytest.mark.parametrize("norms_module_type", NORMS_MODULE_TYPES)
def test_norm_bases_are_reset_between_episodes(in_tmp_dir, norms_module_type):
    model = _build("basic", "homogeneous", 12, "utilitarian", True, str(in_tmp_dir / "ckpt") + "/",
                   False, True, "reset", 3, 5, norms_module_type=norms_module_type)
    model.step()
    model.finish_episode()
    for agent in harvest_agents(model):
        module = agent.norms_module
        if norms_module_type == "rule_based":
            assert module.behaviour_base == {}
        else:
            assert all(module.behaviour_counts(*view) == (0, 0) for view in module.iter_behaviours())


def test_norms_module_type_ignored_when_not_tracking_norms(in_tmp_dir):
    model = _build("basic", "homogeneous", 12, "utilitarian", True, str(in_tmp_dir / "ckpt") + "/",
                   False, False, "off", 3, 5, norms_module_type="rule_based")
    assert not any(hasattr(a, "norms_module") for a in harvest_agents(model))
    run_episodes(model, 2)
    assert model.agent_reporter["num_norms"].isna().all()


def test_rule_based_is_the_default(in_tmp_dir):
    """Built without a norms_module_type, as run.py and older callers do."""
    model = BasicHarvest("homogeneous", 4, 12, "utilitarian", 8, 8, 3, 5, True,
                         str(in_tmp_dir / "ckpt") + "/", False, True, "default")
    assert model.norms_module_type == "rule_based"
    assert all(isinstance(a.norms_module, NormsModule) for a in harvest_agents(model))
    assert NORMS_MODULE_TYPES[0] == "rule_based"  # run.py's default when not asked


def test_unknown_norms_module_type_raises(in_tmp_dir):
    with pytest.raises(UnrecognisedNormsModule):
        _build("basic", "homogeneous", 12, "utilitarian", True, str(in_tmp_dir / "ckpt") + "/",
               False, True, "bad", 3, 5, norms_module_type="lookup_table")

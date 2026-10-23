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


def test_emerged_norm_statistics_are_overwritten_and_days_held_tracked(in_tmp_dir):
    model = _build("basic", "homogeneous", 12, "utilitarian", True, str(in_tmp_dir / "ckpt") + "/",
                   False, True, "held", 3, 5, norms_module_type="rule_based")
    agents = harvest_agents(model)
    norm = "IF,low health,THEN,eat"

    def set_totals(numerosity, holders):
        for agent in agents:
            agent.norms_module.behaviour_base = {}
        for agent in holders:
            agent.norms_module.behaviour_base = {norm: {"reward": 2*numerosity, "numerosity": numerosity, "age": 1, "fitness": 1}}

    for day, numerosity in [(1, 1), (2, 2), (3, 3)]:
        model.day = day
        set_totals(numerosity, agents)
        model._check_emerged_norms()
    # statistics are the society's current totals, not summed across steps
    assert model.emerged_norms[norm] == {"first_day_held": 1, "last_day_held": 3, "days_held": 3,
                                         "reward": 6*len(agents), "numerosity": 3*len(agents),
                                         "fitness": len(agents), "adoption": len(agents)}
    # below the emergence threshold: kept, but neither updated nor counted as held
    model.day = 4
    set_totals(4, agents[:1])
    model._check_emerged_norms()
    assert model.emerged_norms[norm]["last_day_held"] == 3 and model.emerged_norms[norm]["days_held"] == 3
    assert model.emerged_norms[norm]["numerosity"] == 3*len(agents)


def test_dead_agents_do_not_count_towards_emergence(in_tmp_dir):
    model = _build("basic", "homogeneous", 12, "utilitarian", True, str(in_tmp_dir / "ckpt") + "/",
                   False, True, "dead", 3, 5, norms_module_type="rule_based")
    agents = harvest_agents(model)
    norm = "IF,low health,THEN,eat"
    for agent in agents:
        agent.norms_module.behaviour_base = {}
    # only the agents that die hold the norm
    for agent in agents[:-2]:
        agent.norms_module.behaviour_base = {norm: {"reward": 1, "numerosity": 1, "age": 1, "fitness": 1}}
        model._remove_agent(agent)
    model._check_emerged_norms()
    assert norm not in model.emerged_norms


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

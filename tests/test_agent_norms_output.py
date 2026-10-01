import json
import os

import numpy as np
import pandas as pd
import pytest

from src.agent.modules.bayesian_norms_module import BayesianNormsModule
from src.harvest_exception import FileExistsException
from tests.conftest import SCENARIOS, _build, make_scenario_model

CSV_COLUMNS = [
    "episode", "agent_id", "agent_type", "end_day", "num_learned_norms", "num_learned_prohibitions",
    "num_learned_obligations", "num_gained_norms", "num_lost_norms", "mean_learned_posterior",
    "num_observed_behaviours", "num_learned_cooperative_norms", "num_learned_uncooperative_norms",
]


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def make_norms_model(tmp_path, filepath="norms", max_days=5):
    return _build(
        "basic", "homogeneous", 12, "baseline", True, str(tmp_path / "checkpoints") + "/",
        False, True, filepath, 3, max_days,
    )


def harvest_agents(model):
    return sorted((a for a in model.schedule.agents if a.agent_type != "berry"), key=lambda a: a.unique_id)


def force_learned(agent, learned_ids, times_precondition_met=10):
    """Set an agent's posteriors directly so exactly learned_ids are learned."""
    for b in agent.norms_module.behaviour_base.values():
        b["posterior"] = 0.99 if b["id"] in learned_ids else 0.05
        b["times_precondition_met"] = times_precondition_met


def read_csv(filepath="norms"):
    return pd.read_csv(f"data/results/current_run/agent_norms_{filepath}.csv")


def read_agent_json(agent_id, filepath="norms"):
    with open(f"data/results/current_run/{filepath}_agent_{agent_id}_norms.json") as file:
        return json.load(file)


# --- _analyse_agent_norms (CSV) ---

def test_csv_header_written_on_construction(in_tmp_dir):
    make_norms_model(in_tmp_dir)
    df = read_csv()
    assert list(df.columns) == CSV_COLUMNS
    assert len(df) == 0


def test_csv_has_one_row_per_agent_per_episode(in_tmp_dir):
    model = make_norms_model(in_tmp_dir)
    while model.episode <= 2:
        model.step()
    df = read_csv()
    assert len(df) == 2 * model.num_agents
    assert sorted(df["episode"].unique()) == [1, 2]
    assert all(sorted(group["agent_id"]) == list(range(model.num_agents)) for _, group in df.groupby("episode"))


def test_csv_counts_gained_and_lost_norms_against_previous_episode(in_tmp_dir):
    """Ids are generated prohibition/obligation alternately, so even ids are prohibitions."""
    model = make_norms_model(in_tmp_dir)
    agent = harvest_agents(model)[0]
    model.step()
    force_learned(agent, {0, 1, 3})
    model.finish_episode()
    model.step()
    force_learned(agent, {1, 3, 5, 7})
    model.finish_episode()
    rows = read_csv().query("agent_id == 0").sort_values("episode").to_dict("records")
    assert rows[0]["num_learned_norms"] == 3
    assert rows[0]["num_learned_prohibitions"] == 1
    assert rows[0]["num_learned_obligations"] == 2
    assert (rows[0]["num_gained_norms"], rows[0]["num_lost_norms"]) == (3, 0)
    assert rows[0]["mean_learned_posterior"] == pytest.approx(0.99)
    assert rows[1]["num_learned_norms"] == 4
    assert (rows[1]["num_gained_norms"], rows[1]["num_lost_norms"]) == (2, 1)


def test_csv_counts_throw_obligations_as_cooperative_and_throw_prohibitions_as_uncooperative(in_tmp_dir):
    model = make_norms_model(in_tmp_dir)
    agent = harvest_agents(model)[0]
    base = agent.norms_module.behaviour_base.values()
    throw_obligations = [b["id"] for b in base if b["type"] == "obligation" and b["action"] == "throw"][:3]
    throw_prohibitions = [b["id"] for b in base if b["type"] == "prohibition" and b["action"] == "throw"][:2]
    eat_obligations = [b["id"] for b in base if b["type"] == "obligation" and b["action"] == "eat"][:2]
    model.step()
    force_learned(agent, set(throw_obligations + throw_prohibitions + eat_obligations))
    model.finish_episode()
    row = read_csv().query("agent_id == 0").iloc[0]
    assert row["num_learned_norms"] == 7
    assert row["num_learned_cooperative_norms"] == 3
    assert row["num_learned_uncooperative_norms"] == 2


def test_csv_excludes_behaviours_below_min_observations(in_tmp_dir):
    model = make_norms_model(in_tmp_dir)
    agent = harvest_agents(model)[0]
    model.step()
    force_learned(agent, {0, 1}, times_precondition_met=agent.norms_module.min_observations - 1)
    model.finish_episode()
    row = read_csv().query("agent_id == 0").iloc[0]
    assert row["num_learned_norms"] == 0
    assert np.isnan(row["mean_learned_posterior"])


def test_rerun_with_same_filepath_raises(in_tmp_dir):
    make_norms_model(in_tmp_dir)
    with pytest.raises(FileExistsException):
        make_norms_model(in_tmp_dir)


def test_agent_collapses_throws_to_one_norm_action(in_tmp_dir):
    model = make_norms_model(in_tmp_dir)
    for agent in harvest_agents(model):
        assert agent.norms_module.actions == ["move", "eat", "throw"]
        expected = [0 if name == "move" else 1 if name == "eat" else 2 for name in agent.actions]
        assert agent.norm_action_index == expected
        assert sum(name.startswith("throw_") for name in agent.actions) == model.num_agents - 1


def test_agent_passes_norm_actions_to_module(in_tmp_dir, monkeypatch):
    """Every update the module receives is a norm action index (0-2), matching the raw action taken."""
    received = []
    original_update = BayesianNormsModule.update

    def recording_update(self, observation, action_taken):
        received.append(action_taken)
        return original_update(self, observation, action_taken)

    monkeypatch.setattr(BayesianNormsModule, "update", recording_update)
    model = make_norms_model(in_tmp_dir)
    for _ in range(4):
        model.step()
    assert received and set(received) <= {0, 1, 2}
    assert sum(a.norms_module.action_counts.sum() for a in harvest_agents(model)) == len(received)


# --- _write_agent_norms_to_file (JSON, one per agent) ---

def test_one_json_per_agent_keyed_by_episode(in_tmp_dir):
    # called directly: the finish_episode() call to _write_agent_norms_to_file is currently disabled
    model = make_norms_model(in_tmp_dir)
    agents = harvest_agents(model)
    model.step()
    force_learned(agents[0], {0, 1, 3})
    model._write_agent_norms_to_file()
    model.finish_episode()
    model.step()
    force_learned(agents[0], {1})
    model._write_agent_norms_to_file()
    model.finish_episode()
    for agent in agents:
        assert list(read_agent_json(agent.unique_id).keys()) == ["1", "2"]
    episodes = read_agent_json(0)
    base = agents[0].norms_module.behaviour_base
    assert [list(norm)[0] for norm in episodes["1"]] == [base[i]["label"] for i in (0, 1, 3)]
    assert [list(norm)[0] for norm in episodes["2"]] == [base[1]["label"]]


def test_json_norm_entry_contents(in_tmp_dir):
    model = make_norms_model(in_tmp_dir)
    agent = harvest_agents(model)[0]
    model.step()
    force_learned(agent, {1})
    model._write_agent_norms_to_file()
    [entry] = read_agent_json(0)["1"]
    label, data = next(iter(entry.items()))
    expected = agent.norms_module.behaviour_base[1]
    assert label == expected["label"]
    assert data == {
        "type": expected["type"],
        "precondition": expected["precondition"],
        "action": expected["action"],
        "posterior": 0.99,
        "times_precondition_met": 10,
        "times_action_matched": expected["times_action_matched"],
    }


# --- integration ---

@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_runs_with_norm_tracking(in_tmp_dir, scenario):
    model = make_scenario_model(in_tmp_dir, scenario, "utilitarian", max_days=5, max_episodes=2, track_norms=True)
    while model.episode <= 2:
        model.step()
    df = pd.read_csv(f"data/results/current_run/agent_norms_test_{scenario}_utilitarian.csv")
    assert len(df) == 2 * model.num_agents
    # agents' norm bases are reset each episode, so check observations via what was recorded
    assert (df["num_observed_behaviours"] > 0).all()


TRAINED_CHECKPOINTS = "data/model_variables/200_days/4_agents/"
# absolute, since in_tmp_dir changes the working directory before the model is built
TRAINED_CHECKPOINTS_ABS = os.path.abspath(TRAINED_CHECKPOINTS) + "/"


@pytest.mark.skipif(not os.path.isdir(TRAINED_CHECKPOINTS), reason="needs the trained 200_days checkpoints")
@pytest.mark.xfail(strict=True, reason=(
    "HarvestAgent._build_norm_base has a bin that is (almost) never used under trained policies: "
    "distance>=7 (0-0.2% of observations; distance rarely exceeds 5 on an 8x8 grid). "
    "Remove this marker once the distance thresholds are changed."
))
@pytest.mark.parametrize("agent_type", ["baseline", "utilitarian"])
def test_every_predicate_is_informative_under_trained_policy(in_tmp_dir, monkeypatch, agent_type):
    """
    A bin that (almost) no observation falls into adds candidates that can never be learned, and one
    that (almost) every observation falls into adds no information. Each bin should hold between 1%
    and 99% of observations.
    Uses the trained 200_days policies: a briefly trained policy barely eats, so its observation
    ranges (e.g. health never above 0.8) are nothing like those of a real test run.
    """
    observations = []
    original_update = BayesianNormsModule.update

    def logging_update(self, observation, action_taken):
        observations.append(np.array(observation, dtype=float))
        return original_update(self, observation, action_taken)

    monkeypatch.setattr(BayesianNormsModule, "update", logging_update)
    model = _build("basic", "homogeneous", 12, agent_type, False, TRAINED_CHECKPOINTS_ABS, False, True, "informative", 3, 200)
    while model.episode <= 3:
        model.step()
    observations = np.array(observations)
    uninformative = []
    for predicate in harvest_agents(model)[0].norms_module._build_predicates():
        values = observations[:, predicate["index"]]
        rate = np.mean((predicate["lower"] <= values) & (values < predicate["upper"]))
        if not 0.01 <= rate <= 0.99:
            uninformative.append(f"{predicate['name']}: {rate:.1%}")
    assert uninformative == []

import json
import os

import numpy as np
import pandas as pd
import pytest

from src.checkpoints import (PROGRESS_FILE, finished_episode, prepare_training_checkpoints, record_agent_types, recorded_agent_types,
                             society_checkpoint_path, training_folder)
from src.harvest_exception import CheckpointExistsException, CheckpointNotFoundException, FileExistsException
from tests.conftest import NUM_AGENTS, SCENARIOS, _build, long_lived

#long enough for the replay buffer to fill to the 100 experiences learning needs within each two episodes
MAX_DAYS = 30


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def build(scenario, checkpoint_path, start_episode=1, society_mix="homogeneous", seasons=True, filepath=None):
    model = long_lived(_build(scenario, society_mix, NUM_AGENTS * 3, "utilitarian", True, checkpoint_path, True, False,
                              filepath or "train_"+scenario, 1, MAX_DAYS, seed=42, seasons=seasons, season_length=2,
                              start_episode=start_episode), health=5.0)
    if start_episode > 1:
        model.reseed_for_continuing()
    return model


def agents(model):
    return sorted((a for a in model.schedule.agents if a.agent_type != "berry"), key=lambda a: a.unique_id)


def weights(model):
    return [[w.copy() for w in network.dqn.get_weights()] for agent in agents(model)
            for network in [agent.decision_module.q_network, agent.decision_module.target_network]]


def same_weights(first, second):
    return all(np.array_equal(a, b) for x, y in zip(first, second) for a, b in zip(x, y))


def train_until(model, episode):
    while model.episode <= episode:
        model.step()


def report(scenario, filepath=None):
    return pd.read_csv(f"data/results/current_run/model_episode_reports_{filepath or 'train_'+scenario}.csv")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_stopped_run_continues_from_the_episode_after_the_last_it_saved(in_tmp_dir, scenario):
    checkpoint_path = str(in_tmp_dir / "checkpoints") + "/"
    model = build(scenario, checkpoint_path)
    initial = weights(model)
    train_until(model, 2)
    saved = weights(model)
    assert not same_weights(initial, saved)
    # stopped part way through episode 3, which is lost
    for _ in range(2):
        model.step()
    assert model.episode == 3 and model.day == 2
    report_path = f"data/results/current_run/model_episode_reports_train_{scenario}.csv"
    assert finished_episode(training_folder(checkpoint_path, "homogeneous", "utilitarian"), report_path) == 2

    continued = build(scenario, checkpoint_path, start_episode=3)
    assert continued.episode == 3
    # the networks saved after episode 2, trained further with epsilon where it left off, and a new (empty) replay buffer
    assert same_weights(saved, weights(continued))
    assert all(agent.get_epsilon() == pytest.approx(np.exp(-0.001 * 2)) for agent in agents(continued))
    assert all(len(buffer["s"]) == 0 for buffer in continued.shared_replay_buffers.values())
    train_until(continued, 4)
    assert not same_weights(saved, weights(continued))
    # one report, every episode once
    assert list(report(scenario)["episode"]) == [1, 2, 3, 4]
    assert finished_episode(training_folder(checkpoint_path, "homogeneous", "utilitarian"), report_path) == 4


def test_a_continued_run_has_the_same_agents_and_groups_but_new_random_draws(in_tmp_dir):
    checkpoint_path = str(in_tmp_dir / "checkpoints") + "/"
    model = build("colours", checkpoint_path)
    train_until(model, 1)
    continued = build("colours", checkpoint_path, start_episode=2)
    again = build("colours", checkpoint_path, start_episode=2, filepath="train_colours_again")
    fresh = build("colours", str(in_tmp_dir / "fresh") + "/", filepath="train_colours_fresh")
    # built from the run's seed, which its report still records
    assert continued.allocations == model.allocations
    assert continued.seed == 42

    def draws(run):
        return [run._draw_season_quality() for _ in range(20)] + [run.random.random() for _ in range(5)]
    # reseeded from the seed and the episode it continues from: the same each time it continues from there, not the run's first episodes
    continued_draws = draws(continued)
    assert continued_draws == draws(again)
    assert continued_draws != draws(fresh)


@pytest.mark.parametrize("layout", ["colours", "allotment-separate"])
def test_a_mixed_society_continues_with_each_agent_as_its_own_type(in_tmp_dir, layout):
    scenario = layout.split("-")[0]
    society_path = society_checkpoint_path(str(in_tmp_dir / "checkpoints") + "/", "heterogeneous", "utilitarian")

    def build_mixed(start_episode, filepath):
        kwargs = {"mixed_allotments": "separate"} if layout == "allotment-separate" else {}
        model = long_lived(_build(scenario, "heterogeneous", NUM_AGENTS * 3, "utilitarian", True, society_path, True, False, filepath, 1,
                                  MAX_DAYS, seed=42, start_episode=start_episode, **kwargs), health=5.0)
        if start_episode > 1:
            model.reseed_for_continuing()
        return model

    model = build_mixed(1, "train_mixed")
    record_agent_types(society_path, model.get_living_agents())
    train_until(model, 1)
    saved = weights(model)
    assert finished_episode(society_path, "data/results/current_run/model_episode_reports_train_mixed.csv") == 1
    continued = build_mixed(2, "train_mixed")
    assert continued.baseline_ids == model.baseline_ids
    assert {f"agent_{a.unique_id}": a.agent_type for a in agents(continued)} == recorded_agent_types(society_path)
    assert same_weights(saved, weights(continued))
    # each type still keeps its own replay buffer
    assert sorted(continued.shared_replay_buffers) == ["baseline", "utilitarian"]
    train_until(continued, 2)
    assert list(pd.read_csv("data/results/current_run/model_episode_reports_train_mixed.csv")["episode"]) == [1, 2]


def write_progress(folder, agent_episodes):
    for agent_id, episode in agent_episodes.items():
        os.makedirs(f"{folder}agent_{agent_id}", exist_ok=True)
        with open(f"{folder}agent_{agent_id}/{PROGRESS_FILE}", "w") as file:
            json.dump({"episode": episode}, file)


def write_report(path, episodes):
    pd.DataFrame({"episode": [float(e) for e in episodes], "end_day": 200.0}).to_csv(path, index=False)


def test_the_last_finished_episode_is_the_last_every_record_has(in_tmp_dir):
    folder = str(in_tmp_dir / "baseline") + "/"
    report_path = str(in_tmp_dir / "report.csv")
    assert finished_episode(folder, report_path) is None
    # agents save one by one, so a run stopped while saving has some agents an episode ahead
    write_progress(folder, {0: 7, 1: 7, 2: 6, 3: 7})
    assert finished_episode(folder) == 6
    # the report is written once every agent has saved: a run stopped before it has no row for that episode
    write_progress(folder, {2: 7})
    write_report(report_path, range(1, 7))
    assert finished_episode(folder, report_path) == 6
    # runs saved before progress was recorded go by their report
    assert finished_episode(str(in_tmp_dir / "old") + "/", report_path) == 6
    write_report(report_path, [])
    assert finished_episode(str(in_tmp_dir / "old") + "/", report_path) is None


def test_only_the_runs_being_continued_may_already_exist(in_tmp_dir):
    checkpoint_path = str(in_tmp_dir / "trained" / "basic_5days_seed42" / f"{NUM_AGENTS}_agents") + "/"
    prepare_training_checkpoints(checkpoint_path, ["utilitarian"], {"seed": 42})
    train_until(build("basic", checkpoint_path), 1)
    with pytest.raises(CheckpointExistsException):
        prepare_training_checkpoints(checkpoint_path, ["utilitarian", "maximin"], {"seed": 42})
    prepare_training_checkpoints(checkpoint_path, ["utilitarian", "maximin"], {"seed": 42}, continuing={"utilitarian": 2})
    # a continued run must have the settings it was started with
    with pytest.raises(ValueError, match="different settings"):
        prepare_training_checkpoints(checkpoint_path, ["utilitarian"], {"seed": 7}, continuing={"utilitarian": 2})


def test_continuing_a_run_that_saved_nothing_fails_clearly(in_tmp_dir):
    with pytest.raises(CheckpointNotFoundException):
        build("basic", str(in_tmp_dir / "empty") + "/", start_episode=2)


def test_a_new_run_still_refuses_an_existing_report(in_tmp_dir):
    checkpoint_path = str(in_tmp_dir / "checkpoints") + "/"
    train_until(build("basic", checkpoint_path), 1)
    with pytest.raises(FileExistsException):
        build("basic", str(in_tmp_dir / "other") + "/")

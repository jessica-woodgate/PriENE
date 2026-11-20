"""
Multi-seed runs: one seed reproduces a run exactly and another differs; each purpose's random draws come from a stream of its own;
an episode's environment depends only on the seed and the episode; test runs seeded from the test seed face the same episodes whatever
the agents' training seed; a stopped run continued from its checkpoint matches an uninterrupted one; and run.py with --output-root
runs in a folder of its own, records the run, and does nothing when relaunched once complete.
"""
import glob
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest
from src import seeding
from src.run_record import read_status, write_status
from tests.conftest import _build

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def restore_working_directory():
    #train() works inside each run's folder, as run.py does
    cwd = os.getcwd()
    yield
    os.chdir(cwd)


def train(folder, seed, agent_type="utilitarian", episodes=3, scenario="colours", start=1, checkpoint_every=1, stop_after=None):
    """A short training run in folder (its data/ written there): episodes of 10 days, or stopped stop_after steps into an episode"""
    os.makedirs(folder, exist_ok=True)
    os.chdir(folder)
    model = _build(scenario, "homogeneous", 12, agent_type, True, os.path.abspath("ck") + "/", True, False, "run", episodes, 10,
                   seed=seed, start_episode=start)
    model.checkpoint_every = checkpoint_every
    if start > 1:
        model.reseed_for_continuing()
    while model.episode <= episodes:
        model.step()
    for _ in range(stop_after or 0):
        model.step()
    return model


def report(folder):
    return pd.read_csv(os.path.join(folder, "data/results/current_run/model_episode_reports_run.csv")).drop(columns="num_emerged_norms")


def layout(model):
    """Where every agent and berry is"""
    return sorted((a.agent_type == "berry", a.unique_id, a.pos) for a in model.schedule.agents if a.pos is not None)


def test_the_same_seed_reproduces_a_training_run_and_another_seed_differs(tmp_path):
    first, again, other = (report(str(tmp_path / name)) for name in ["a", "b", "c"]
                           if train(str(tmp_path / name), {"a": 5, "b": 5, "c": 6}[name]) is not None)
    assert first.equals(again)
    assert not first.drop(columns="seed").equals(other.drop(columns="seed"))


def test_an_episodes_environment_depends_only_on_the_seed_and_the_episode(tmp_path):
    # societies of different types act differently, but start every episode in the same layout, with the same groups
    utilitarian = train(str(tmp_path / "u"), 5, "utilitarian", episodes=2)
    baseline = train(str(tmp_path / "b"), 5, "baseline", episodes=2)
    assert utilitarian.episode == baseline.episode == 3
    assert layout(utilitarian) == layout(baseline)
    assert [a.allocation_id for a in utilitarian.get_living_agents()] == [a.allocation_id for a in baseline.get_living_agents()]
    assert utilitarian.random.getstate() == baseline.random.getstate()


def test_extra_draws_from_one_stream_dont_shift_another(tmp_path):
    plain = train(str(tmp_path / "plain"), 5, episodes=1)
    os.chdir(tmp_path)
    shifted = _build("colours", "homogeneous", 12, "utilitarian", True, str(tmp_path / "shifted_ck") + "/", True, False, "shifted", 1, 10, seed=5)
    #extra exploration draws for every agent before the run starts
    for agent in shifted.get_living_agents():
        agent.decision_module.exploration_rng.random(1000)
    while shifted.episode <= 1:
        shifted.step()
    # what the agents did differs, but the next episode's environment doesn't
    assert layout(plain) == layout(shifted)
    assert plain.random.getstate() == shifted.random.getstate()


def test_streams_are_separate_and_reproducible():
    assert seeding.episode_rng(1, 3, "placement").random() == seeding.episode_rng(1, 3, "placement").random()
    draws = {stream: seeding.episode_rng(1, 3, stream).random() for stream in seeding.ENVIRONMENT_STREAMS}
    assert len(set(draws.values())) == len(draws)
    assert seeding.episode_rng(1, 3, "placement").random() != seeding.episode_rng(1, 4, "placement").random()
    assert seeding.agent_rng(1, 0, "exploration").random() != seeding.agent_rng(1, 1, "exploration").random()
    assert seeding.agent_seeds(1, 0, "weights", 3) == seeding.agent_seeds(1, 0, "weights", 3) != seeding.agent_seeds(2, 0, "weights", 3)


def test_test_runs_with_the_test_seed_face_the_same_episodes_whatever_the_training_seed(tmp_path):
    # agents trained with different seeds, tested with the same test seed: the same layout at the start of every test episode
    for name, seed in [("one", 1), ("two", 2)]:
        train(str(tmp_path / name), seed, episodes=1)
    layouts = []
    for name in ["one", "two"]:
        os.chdir(tmp_path / name)
        model = _build("colours", "homogeneous", 12, "utilitarian", False, str(tmp_path / name / "ck") + "/", False, False, "tested", 3, 10,
                       seed=1000)
        starts = []
        while model.episode <= 3:
            if model.day == 0:
                starts.append(layout(model))
            model.step()
        layouts.append(starts)
    assert layouts[0] == layouts[1] and len(layouts[0]) == 3


#one scenario: test_continue_training.py continues a stopped run in every scenario
@pytest.mark.parametrize("scenario", ["colours"])
def test_a_stopped_run_continued_from_its_checkpoint_matches_an_uninterrupted_run(tmp_path, scenario):
    uninterrupted = str(tmp_path / "whole")
    train(uninterrupted, 7, episodes=5, scenario=scenario, checkpoint_every=2)
    stopped = str(tmp_path / "stopped")
    # stopped 4 steps into episode 4: its last checkpoint is after episode 2, so episodes 3 and 4 are trained again
    train(stopped, 7, episodes=3, scenario=scenario, checkpoint_every=2, stop_after=4)
    train(stopped, 7, episodes=5, scenario=scenario, checkpoint_every=2, start=3)
    assert list(report(stopped)["episode"]) == [1, 2, 3, 4, 5]
    assert report(stopped).equals(report(uninterrupted))


# run.py with --output-root

def run_py(args, cwd=REPO):
    return subprocess.run([sys.executable, os.path.join(REPO, "run.py")] + args, cwd=cwd, capture_output=True, text=True, stdin=subprocess.DEVNULL)


@pytest.fixture
def short_config(tmp_path):
    with open(os.path.join(REPO, "configs/training/harvest.yaml")) as file:
        text = file.read().replace("max_training_episodes: null", "max_training_episodes: 2")
    path = str(tmp_path / "short.yaml")
    with open(path, "w") as file:
        file.write(text)
    return path


def train_args(root, short_config, seed=3):
    return ["train", "--output-root", root, "--experiment", "x", "--seed", str(seed), "--num-threads", "1", "--training-config", short_config,
            "--scenario", "colours", "--seasons", "n", "--days", "10", "--society-mix", "homogeneous", "--agent-type", "maximin",
            "--num-agents", "4", "--write-data", "y"]


def test_a_run_given_every_flag_runs_in_its_own_folder_records_itself_and_is_relaunched_as_it_stands(tmp_path, short_config):
    # one training run serves every check (each run.py process takes seconds just to start): its folder and record, relaunching it once
    # complete, a test run of it, and relaunching it once stopped partway
    root = str(tmp_path / "runs")
    result = run_py(train_args(root, short_config))
    assert result.returncode == 0, result.stdout + result.stderr
    (folder,) = glob.glob(f"{root}/x/train/*/seed_3/maximin")
    assert folder.endswith("x/train/colours_4agents_10days_short/seed_3/maximin")
    assert read_status(folder) == "complete"
    with open(os.path.join(folder, "run.json")) as file:
        record = json.load(file)
    assert record["configuration"]["seed"] == 3 and record["configuration"]["scenario"] == "colours"
    assert {"hostname", "cpu", "gpus", "architecture"} <= set(record["machine"]) and record["packages"]["tensorflow"]
    assert "--output-root" in record["command"]
    assert os.path.exists(os.path.join(folder, "run.log"))
    assert glob.glob(f"{folder}/data/model_variables/trained/*/4_agents/maximin/agent_0/q_model_variables.keras")
    # relaunching a complete run does nothing
    before = os.path.getmtime(glob.glob(f"{folder}/data/model_variables/trained/*/4_agents/maximin/agent_0/progress.json")[0])
    again = run_py(train_args(root, short_config))
    assert again.returncode == 0 and "is complete: nothing to do" in again.stdout
    assert os.path.getmtime(glob.glob(f"{folder}/data/model_variables/trained/*/4_agents/maximin/agent_0/progress.json")[0]) == before
    # a test run of it, from its folder, seeded from the test seed
    test = run_py(["test", "--output-root", root, "--experiment", "x", "--seed", "3", "--test-episodes", "2", "--checkpoint", folder,
                   "--scenario", "capabilities", "--seasons", "n", "--society-mix", "homogeneous", "--agent-type", "maximin", "--num-agents", "4",
                   "--groups", "2", "--write-data", "y", "--norms", "n", "--render", "n"])
    assert test.returncode == 0, test.stdout + test.stderr
    (tested,) = glob.glob(f"{root}/x/test/*/train_seed_3/maximin")
    assert read_status(tested) == "complete"
    episodes = pd.read_csv(f"{tested}/data/results/current_run/model_episode_reports_capabilities_maximin.csv")
    assert list(episodes["episode"]) == [1, 2] and set(episodes["seed"]) == {1000}
    # relaunching it as if it had stopped partway (its status says running, but its process is gone) continues it from its checkpoint
    write_status(folder, "running", pid=999999)
    assert read_status(folder) == "incomplete"
    stopped = run_py(train_args(root, short_config))
    assert stopped.returncode == 0, stopped.stdout + stopped.stderr
    assert "continuing it from its checkpoint" in stopped.stdout
    assert read_status(folder) == "complete"


def test_a_flag_the_prompt_refuses_stops_the_run():
    result = run_py(["train", "--scenario", "nowhere"])
    assert result.returncode != 0 and "--scenario 'nowhere' is not a valid answer" in result.stdout + result.stderr

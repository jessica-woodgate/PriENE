"""
The run manifest (src/analysis/run_manifest.py): one row per file of every run under an experiment's folder, with the run's mode,
scenario, society, training seed and reward config from its run.json, whatever its files are called; a test run takes its training run's
reward config.
"""
import json
import os
import pytest
from src.analysis import run_manifest as rm
from src import training_config as tc


def write_run(folder, mode, society, seed, command, training_config, checkpoint=None, test_seed=None, files=()):
    os.makedirs(folder, exist_ok=True)
    configuration = {"mode": mode, "scenario": "basic" if mode == "train" else "colours", "condition": "c", "society": society, "seed": seed,
                     "test_seed": test_seed, "checkpoint": checkpoint, "training_config": training_config}
    with open(os.path.join(folder, "run.json"), "w") as file:
        json.dump({"configuration": configuration, "command": command}, file)
    with open(os.path.join(folder, "status.json"), "w") as file:
        json.dump({"status": "complete"}, file)
    for name in files:
        path = os.path.join(folder, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").close()


@pytest.fixture
def experiment(tmp_path):
    positive = tc.DEFAULT.with_values({"sanction": 1.4, "positive_sanction_only": True}).to_dict()
    report = "data/results/current_run/model_episode_reports_train_basic_maximin.csv"
    network = "data/model_variables/trained/x/4_agents/maximin/agent_2/q_model_variables.keras"
    for seed in [101, 102]:
        write_run(str(tmp_path / "train" / "c" / f"seed_{seed}" / "maximin"), "train", "maximin", seed,
                  ["run.py", "train", "--training-config", "/somewhere/configs/training/positive.yaml"], positive, files=[report, network])
    write_run(str(tmp_path / "train" / "d" / "seed_101" / "baseline"), "train", "baseline", 101, ["run.py", "train"], tc.DEFAULT.to_dict(),
              files=[report])
    trained = tmp_path / "train" / "c" / "seed_102" / "maximin"
    write_run(str(tmp_path / "test" / "e" / "train_seed_102" / "maximin"), "test", "maximin", 102, ["run.py", "test"], positive,
              checkpoint=str(trained / "data" / "model_variables" / "trained" / "x" / "4_agents"), test_seed=1000,
              files=["data/results/current_run/model_episode_reports_colours_maximin.csv"])
    return tmp_path


def test_every_file_of_every_run_has_a_row_saying_which_run_it_came_from(experiment):
    manifest = rm.build(str(experiment))
    assert manifest["run_folder"].nunique() == 4
    reports = manifest[manifest["file_type"] == "model_episode_report"]
    #the same file name in two runs, told apart by the manifest
    trained = reports[(reports["mode"] == "train") & (reports["society"] == "maximin")]
    assert len(set(trained["file"].map(os.path.basename))) == 1 and sorted(trained["training_seed"]) == [101, 102]
    assert set(trained["reward_config"]) == {"positive"} and set(trained["sanction"]) == {1.4} and trained["positive_sanction_only"].all()
    network = manifest[manifest["file_type"] == "q_network"]
    assert sorted(network["agent_id"]) == [2, 2]


def test_a_run_given_no_training_config_has_the_scenarios_own(experiment):
    manifest = rm.build(str(experiment))
    baseline = manifest[manifest["society"] == "baseline"].iloc[0]
    assert baseline["reward_config"] == "harvest" and baseline["sanction"] == 0.4 and not baseline["positive_sanction_only"]
    assert json.loads(baseline["ethics_rewards"])["eat"] == 0.8


def test_the_export_names_every_file_by_reward_seed_and_society_and_links_it(experiment, monkeypatch):
    monkeypatch.setattr(rm, "REPO", str(experiment))
    manifest = rm.build(str(experiment))
    exported = rm.export(manifest, str(experiment / "results"))
    names = set(exported["export_file"])
    assert "results/positive/seed_101/maximin/positive_seed101_maximin_train_episode_report.csv" in names
    assert "results/positive/seed_102/maximin/positive_seed102_maximin_train_agent2_q_network.keras" in names
    #the scenario's own config is the current rewards; a test run is named by its test condition, under its training seed
    assert "results/current/seed_101/baseline/current_seed101_baseline_train_episode_report.csv" in names
    assert "results/positive/seed_102/maximin/positive_seed102_maximin_test_c_episode_report.csv" in names
    #linked, not copied, and the runs' own files are untouched
    for source, target in zip(exported["file"], exported["export_file"]):
        assert os.path.samefile(experiment / source, experiment / target)
    assert os.path.exists(experiment / "results" / "manifest.csv")
    #exporting again finds every file already there
    assert len(rm.export(manifest, str(experiment / "results"))) == len(exported)


def test_the_export_leaves_out_runs_still_in_progress(experiment, monkeypatch):
    #a run still training rewrites its checkpoints, so a link made now would go stale
    monkeypatch.setattr(rm, "REPO", str(experiment))
    with open(experiment / "train" / "c" / "seed_101" / "maximin" / "status.json", "w") as file:
        json.dump({"status": "running"}, file)
    converging = experiment / "train" / "d" / "seed_101" / "baseline" / "data" / "results" / "current_run"
    with open(converging / "convergence_train_basic_baseline.json", "w") as file:
        json.dump({"converged": False, "after_episode": 3000}, file)
    exported = rm.export(rm.build(str(experiment)), str(experiment / "results"))
    assert set(exported["run_folder"]) == {"train/c/seed_102/maximin", "test/e/train_seed_102/maximin"}
    assert not os.path.exists(experiment / "results" / "positive" / "seed_101")


def test_a_test_run_has_its_training_runs_reward_config_and_seed(experiment):
    manifest = rm.build(str(experiment))
    test = manifest[manifest["mode"] == "test"].iloc[0]
    assert test["reward_config"] == "positive" and test["training_seed"] == 102 and test["test_seed"] == 1000

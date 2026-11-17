"""
The experiment runner (src/experiments.py, run.py experiments): the commands it plans, each run in a folder of its own and a process of
its own, complete runs skipped, failures reported, test runs only after their training run completes, and the status summary.
"""
import json
import os
import sys
import pytest
import yaml
from src import experiments as ex
from src.run_record import write_status

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def experiment_file(tmp_path, **overrides):
    experiment = {"name": "exp", "output_root": str(tmp_path / "runs"), "seeds": [1, 2], "societies": ["baseline", "maximin"],
                  "train": [{"scenario": "basic", "seasons": "n", "days": 10, "num_agents": 4}],
                  "tests": [{"scenario": "colours", "seasons": "n"}, {"scenario": "allotment", "seasons": "y", "groups": 2}],
                  "test_episodes": 2, **overrides}
    path = str(tmp_path / "experiment.yaml")
    with open(path, "w") as file:
        yaml.safe_dump(experiment, file)
    return path


def flag(command, name):
    return command[command.index(name) + 1]


def test_the_plan_has_a_training_run_per_society_and_seed_and_a_test_run_per_training_run_and_condition(tmp_path):
    runs = ex.plan(ex.load_experiment(experiment_file(tmp_path)))
    train = [r for r in runs if r.mode == "train"]
    test = [r for r in runs if r.mode == "test"]
    assert len(train) == 2 * 2 and len(test) == 4 * 2
    assert runs[:4] == train
    assert len({r.folder for r in runs}) == len(runs)
    assert {(r.society, r.seed) for r in train} == {("baseline", 1), ("baseline", 2), ("maximin", 1), ("maximin", 2)}
    assert train[0].folder == str(tmp_path / "runs/exp/train/basic_4agents_10days/baseline/seed_1")
    for run in test:
        # each test run loads its own training run's checkpoints, and is seeded from the test seed
        (trained,) = [t for t in train if (t.society, t.seed) == (run.society, run.seed)]
        assert run.train_folder == trained.folder and flag(run.command, "--checkpoint") == trained.folder
        assert flag(run.command, "--test-seed") == "1000" and flag(run.command, "--seed") == str(run.seed)
        assert run.folder.endswith(f"/{run.society}/train_seed_{run.seed}")
    conditions = {r.condition for r in test}
    assert conditions == {"colours_4agents_10days", "allotment_seasons_len25_2groups_4agents_10days"}
    for run in runs:
        assert run.command[:3] == [sys.executable, os.path.join(REPO, "run.py"), run.mode]
        assert flag(run.command, "--output-root") == str(tmp_path / "runs") and flag(run.command, "--num-threads") == "1"


def test_a_mixed_society_experiment_doesnt_run_baseline_alone(tmp_path):
    runs = ex.plan(ex.load_experiment(experiment_file(tmp_path, society_mix="heterogeneous")))
    assert {r.society for r in runs} == {"maximin"}
    assert all(flag(r.command, "--society-mix") == "heterogeneous" for r in runs)


@pytest.mark.parametrize("overrides, message", [
    ({"tests": [{"scenario": "fruit_market"}]}, "harvest scenarios"),
    ({"train": [{"scenario": "basic"}]}, "must give days"),
    ({"societies": ["kind"]}, "unknown societies"),
    ({"tests": [{"scenario": "commons"}]}, "must give carrying_capacity")])
def test_bad_experiment_files_are_refused(tmp_path, overrides, message):
    with pytest.raises(ex.ExperimentError, match=message):
        ex.plan(ex.load_experiment(experiment_file(tmp_path, **overrides)))


def test_list_prints_one_command_per_run(tmp_path, capsys):
    path = experiment_file(tmp_path)
    assert ex.main(path, list_only=True) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 12 and all("run.py train " in line for line in lines[:4]) and all("run.py test " in line for line in lines[4:])


def fake_launcher(outcomes, launched):
    """Stands in for running a run.py command: records it, and writes the status it is given (complete by default)"""
    def launch(experiment, run):
        launched.append(run.name)
        status = outcomes.get((run.mode, run.society, run.seed), "complete")
        os.makedirs(run.folder, exist_ok=True)
        write_status(run.folder, status)
        return status
    return launch


def test_runs_go_training_first_then_tests_skipping_complete_runs_and_reporting_failures(tmp_path):
    experiment = ex.load_experiment(experiment_file(tmp_path))
    runs = ex.plan(experiment)
    # one training run already complete; another fails, so its two test runs aren't launched
    already = next(r for r in runs if (r.mode, r.society, r.seed) == ("train", "baseline", 1))
    os.makedirs(already.folder)
    write_status(already.folder, "complete")
    launched, messages = [], []
    outcomes = ex.run_experiment(experiment, max_workers=2, log=messages.append,
                                 launch=fake_launcher({("train", "maximin", 2): "failed"}, launched))
    assert already.name not in launched and outcomes[already.name] == "complete (skipped)"
    assert outcomes[next(r.name for r in runs if (r.mode, r.society, r.seed) == ("train", "maximin", 2))] == "failed"
    blocked = [r.name for r in runs if r.mode == "test" and (r.society, r.seed) == ("maximin", 2)]
    assert all(outcomes[name].startswith("not run") for name in blocked) and not set(blocked) & set(launched)
    # every training run was launched before any test run
    modes = [name.split("/")[0] for name in launched]
    assert modes == sorted(modes, key=lambda mode: mode != "train") and modes.count("test") == 6
    assert any("failed" in message for message in messages) and "1 failed" in messages[-2] + messages[-1]
    # run again: only the failed run (and the test runs it blocked) are left
    launched.clear()
    ex.run_experiment(experiment, max_workers=2, log=messages.append, launch=fake_launcher({}, launched))
    assert sorted(launched) == sorted([next(r.name for r in runs if (r.mode, r.society, r.seed) == ("train", "maximin", 2))] + blocked)


def test_status_summarises_every_run(tmp_path, capsys):
    path = experiment_file(tmp_path)
    experiment = ex.load_experiment(path)
    run = ex.plan(experiment)[0]
    os.makedirs(run.folder)
    write_status(run.folder, "failed", error="boom")
    table = ex.status_table(experiment)
    assert len(table) == 12 and (table["status"] == "failed").sum() == 1 and (table["status"] == "not started").sum() == 11
    ex.main(path, status=True)
    assert "failed" in capsys.readouterr().out


def test_an_experiment_runs_each_run_in_a_process_of_its_own(tmp_path, monkeypatch):
    # each run's command stands in for run.py (whose own runs test_multiseed_runs.py covers): it records its status, with its process id,
    # in its folder, as run.py does; the runner launches each as a process of its own and reads back how it ended
    experiment = ex.load_experiment(experiment_file(tmp_path, seeds=[1], societies=["maximin"], tests=[{"scenario": "colours", "seasons": "n"}]))
    planned = ex.plan(experiment)
    record = ("import os, sys; sys.path.insert(0, {repo!r}); from src.run_record import write_status; "
              "os.makedirs({folder!r}, exist_ok=True); write_status({folder!r}, {status!r})")
    for run in planned:
        status = "failed" if run.mode == "test" else "complete"
        run.command = [sys.executable, "-c", record.format(repo=REPO, folder=run.folder, status=status)] + (["--fail"] if status == "failed" else [])
    monkeypatch.setattr(ex, "plan", lambda experiment: planned)
    outcomes = ex.run_experiment(experiment, max_workers=2, log=lambda message: None)
    assert [outcomes[run.name] for run in planned] == ["complete", "failed"]
    pids = set()
    for run in planned:
        with open(os.path.join(run.folder, "status.json")) as file:
            pids.add(json.load(file)["pid"])
    assert len(pids) == 2 and os.getpid() not in pids
    # the complete run is skipped; the failed one is launched again
    again = ex.run_experiment(experiment, log=lambda message: None)
    assert again[planned[0].name] == "complete (skipped)" and again[planned[1].name] == "failed"

"""
Training until learning has converged (src/convergence.py, run.py --until-converged): only episodes at the exploration floor count;
steady metrics converge, drifting ones don't; the monitor checks every check_every episodes, records each check, and stops at
max_episodes; and the experiment runner relaunches complete training runs that haven't converged.
"""
import json
import os
import numpy as np
import pandas as pd
import pytest
from src import convergence as cv
from src import experiments as ex

CRITERION = {"window": 50, "check_every": 10, "consecutive": 3}
NEEDED = 2 * 50 + 2 * 10


def report(at_floor, above_floor=20, drift=0.0, seed=0):
    """A model episode report: above_floor episodes exploring, then at_floor at the floor, its metrics noisy and drifting by drift each episode"""
    rng = np.random.default_rng(seed)
    n = above_floor + at_floor
    epsilon = np.r_[np.linspace(0.5, 0.02, above_floor), np.full(at_floor, 0.01)]
    trend = drift * np.arange(n)
    return pd.DataFrame({"episode": np.arange(1, n + 1), "epsilon": epsilon, "mean_reward": 10 + trend + rng.normal(0, 1, n),
                         "deceased": np.clip(1 - trend / 10 + rng.normal(0, 0.3, n), 0, 4), "mean_berries_thrown": 2 + rng.normal(0, 0.5, n)})


def test_steady_learning_at_the_floor_has_converged():
    converged, details = cv.assess(report(NEEDED), 0.01, **CRITERION)
    assert converged and len(details["checks"]) == 3 and all(check["passed"] for check in details["checks"])


def test_learning_still_improving_hasnt_converged():
    converged, details = cv.assess(report(NEEDED, drift=0.05), 0.01, **CRITERION)
    assert not converged and not all(check["passed"] for check in details["checks"])


def test_only_episodes_at_the_floor_count():
    converged, details = cv.assess(report(NEEDED - 1, above_floor=500), 0.01, **CRITERION)
    assert not converged and details["episodes_at_floor"] == NEEDED - 1 and details["checks"] == []


def test_the_monitor_checks_every_check_every_episodes_records_each_check_and_stops_at_max_episodes(tmp_path):
    report_path, record_path = str(tmp_path / "report.csv"), str(tmp_path / "convergence.json")
    #improving: never converges, so it trains on until max_episodes
    data = report(NEEDED + 30, drift=0.05)
    data.to_csv(report_path, index=False)
    monitor = cv.ConvergenceMonitor(report_path, record_path, 0.01, max_episodes=len(data) + 5, **CRITERION)
    assert monitor.keep_training(len(data) + 1)
    assert json.load(open(record_path))["converged"] is False and not cv.finished(record_path)
    assert monitor.keep_training(len(data) + 2)
    assert not monitor.keep_training(len(data) + 6)
    assert "max_episodes" in json.load(open(record_path))["stopped"] and cv.finished(record_path)
    #steady: converged at the first check
    steady = report(NEEDED)
    steady.to_csv(report_path, index=False)
    monitor = cv.ConvergenceMonitor(report_path, record_path, 0.01, **CRITERION)
    assert not monitor.keep_training(len(steady) + 1)
    assert json.load(open(record_path))["converged"] is True and cv.finished(record_path)


def write_training_run(folder, convergence=None):
    results = os.path.join(folder, "data", "results", "current_run")
    os.makedirs(results)
    open(os.path.join(results, "model_episode_reports_train_basic_maximin.csv"), "w").close()
    with open(os.path.join(folder, "status.json"), "w") as file:
        json.dump({"status": "complete"}, file)
    if convergence is not None:
        with open(os.path.join(results, "convergence_train_basic_maximin.json"), "w") as file:
            json.dump(convergence, file)


def test_a_training_run_is_finished_once_its_convergence_file_says_so(tmp_path):
    write_training_run(str(tmp_path / "a"))
    write_training_run(str(tmp_path / "b"), {"converged": False})
    write_training_run(str(tmp_path / "c"), {"converged": True})
    write_training_run(str(tmp_path / "d"), {"converged": False, "stopped": "reached max_episodes"})
    assert [cv.training_finished(str(tmp_path / name)) for name in "abcd"] == [False, False, True, True]


@pytest.mark.parametrize("until_converged", [False, True])
def test_the_runner_relaunches_complete_training_runs_that_havent_converged(tmp_path, until_converged):
    path = tmp_path / "experiment.yaml"
    path.write_text(f"name: exp\noutput_root: {tmp_path / 'runs'}\nseeds: [1, 2]\nsocieties: [maximin]\nuntil_converged: {str(until_converged).lower()}\n"
                    "convergence_max_episodes: 7000\ntrain:\n  - {scenario: basic, seasons: n, days: 10, num_agents: 4}\n")
    experiment = ex.load_experiment(str(path))
    runs = ex.plan(experiment)
    assert all(("--until-converged" in run.command) == until_converged for run in runs)
    if until_converged:
        assert all(run.command[run.command.index("--convergence-max-episodes") + 1] == "7000" for run in runs)
    #both complete; seed 1's learning has converged, seed 2's hasn't
    write_training_run(runs[0].folder, {"converged": True})
    write_training_run(runs[1].folder, {"converged": False})
    launched = []
    ex.run_experiment(experiment, log=lambda message: None, launch=lambda experiment, run: launched.append(run.seed) or "complete")
    assert launched == ([2] if until_converged else [])

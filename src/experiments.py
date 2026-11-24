"""
Experiments: every training and test run of an experiment file (configs/experiments/*.yaml), each one self-contained run.py command
in a process of its own (TensorFlow's seeding and determinism are process-wide, so one process trains or tests exactly one seed):

    python run.py experiments configs/experiments/example.yaml --max-workers 4    # run them, at most 4 at a time
    python run.py experiments configs/experiments/example.yaml --list             # just print the commands (e.g. to run elsewhere)
    python run.py experiments configs/experiments/example.yaml --status           # how far each run has got

Training runs come first: one per training condition x society type x seed. Then, for each training run that completed, one test run
per test condition, loading that run's checkpoints and seeded from the test seed, so every trained society faces the same test
episodes. A complete run is skipped; a run stopped partway is continued (training) or restarted (testing) by its own command; a failed
run is reported, and the test runs of a training run that didn't complete aren't launched.

An experiment file:
    name: example                     # the folder under output_root
    output_root: runs                 # relative to the repository, or absolute
    seeds: [1, 2, 3, 4, 5]            # training seeds (default 1-5)
    societies: [baseline, utilitarian, maximin, egalitarian, average, majoritarian, optimist, veto]
    society_mix: homogeneous          # or heterogeneous (half of each society is baseline agents; baseline isn't run alone)
    training_config: null             # a training config file (run.py --training-config), or null for the scenario's own
    checkpoint_every: 10              # episodes between training checkpoints
    until_converged: false            # training trains on at the exploration floor until its learning has converged (run.py
                                      # --until-converged, src/convergence.py); complete training runs that haven't are continued
    convergence_max_episodes: 10000   # with until_converged, training stops at this episode, converged or not
    num_threads: 1                    # TensorFlow threads for each run
    train:                            # one or more training conditions
      - {scenario: basic, seasons: n, days: 200, num_agents: 4}
    test_seed: 1000
    test_episodes: 1000
    norms: y                          # write norms in test runs
    norms_module: rule_based
    tests:                            # test conditions, run for every training run (with its number of agents and days)
      - {scenario: colours, seasons: n}
      - {scenario: capabilities, seasons: n, groups: 2}
      - {scenario: allotment, seasons: n, groups: 2}
A condition may also give season_length (default 25), groups (allotment and capabilities, default 2), layout (a mixed society's
allotments) and, for commons, carrying_capacity, growth_rate and initial_stock. Allotment agents can always throw berries to agents in
other allotments.
"""
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import yaml
from .run_naming import harvest_condition
from .run_record import run_folder, read_status
from . import training_config as tc
from .convergence import MAX_EPISODES as CONVERGENCE_MAX_EPISODES, training_finished

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENT_TYPES = ["baseline", "utilitarian", "maximin", "egalitarian", "average", "majoritarian", "optimist", "veto"]
HARVEST_SCENARIOS = ["basic", "colours", "capabilities", "allotment", "commons"]
DEFAULTS = {"seeds": [1, 2, 3, 4, 5], "societies": AGENT_TYPES, "society_mix": "homogeneous", "training_config": None, "checkpoint_every": 10,
            "num_threads": 1, "test_seed": 1000, "test_episodes": 1000, "norms": "y", "norms_module": "rule_based", "tests": [],
            "until_converged": False, "convergence_max_episodes": CONVERGENCE_MAX_EPISODES}
CONDITION_DEFAULTS = {"seasons": "n", "season_length": 25, "groups": 2, "layout": "split"}


class ExperimentError(ValueError):
    pass


@dataclass
class PlannedRun:
    mode: str               # "train" or "test"
    condition: str          # the condition's folder name
    society: str            # the agent type
    seed: int               # the training seed (a test run's agents were trained with it)
    folder: str             # the run's own folder (run.py --output-root)
    command: list           # the run.py command
    train_folder: str = None    # a test run's training run, which must be complete first

    @property
    def name(self):
        #as its folder is laid out (src/run_record.py): <mode>/<condition>/<seed label>/<society>
        return f"{self.mode}/{self.condition}/{os.path.basename(os.path.dirname(self.folder))}/{self.society}"


def load_experiment(path):
    with open(path) as file:
        experiment = yaml.safe_load(file)
    if not isinstance(experiment, dict) or "name" not in experiment or "train" not in experiment:
        raise ExperimentError(f"{path} must give at least name and train")
    experiment = {**DEFAULTS, "output_root": "runs", **experiment}
    if isinstance(experiment["train"], dict):
        experiment["train"] = [experiment["train"]]
    for condition in experiment["train"] + experiment["tests"]:
        if condition.get("scenario") not in HARVEST_SCENARIOS:
            raise ExperimentError(f"{condition}: the runner launches the harvest scenarios {HARVEST_SCENARIOS} (Fruit Market runs are launched "
                                  "with run.py directly)")
    for condition in experiment["train"]:
        for key in ["days", "num_agents"]:
            if key not in condition:
                raise ExperimentError(f"training condition {condition} must give {key}")
    unknown = set(experiment["societies"]) - set(AGENT_TYPES)
    if unknown:
        raise ExperimentError(f"unknown societies {sorted(unknown)}; they are {AGENT_TYPES}")
    root = experiment["output_root"]
    experiment["output_root"] = root if os.path.isabs(root) else os.path.join(REPO, root)
    return experiment


def _yes(value):
    return value in ["y", True, "yes"]


def _condition_flags(condition, num_agents):
    """The flags for a condition's settings (run.py asks each only when it applies, so the others go unused)"""
    c = {**CONDITION_DEFAULTS, **condition}
    flags = ["--scenario", c["scenario"], "--seasons", "y" if _yes(c["seasons"]) else "n", "--num-agents", str(num_agents)]
    if _yes(c["seasons"]):
        flags += ["--season-length", str(c["season_length"])]
    if c["scenario"] in ["allotment", "capabilities"]:
        flags += ["--groups", str(c["groups"])]
    if c["scenario"] == "allotment":
        flags += ["--layout", c["layout"]]
    if c["scenario"] == "commons":
        for key in ["carrying_capacity", "growth_rate", "initial_stock"]:
            if key not in c:
                raise ExperimentError(f"commons condition {condition} must give {key}")
            flags += ["--" + key.replace("_", "-"), str(c[key])]
    return flags


def _condition_name(condition, num_agents, days, society_mix, config_name=None):
    if "throw_between_allotments" in condition:
        raise ExperimentError(f"condition {condition}: throw_between_allotments is no longer a setting (allotment agents can always throw "
                              f"berries to agents in other allotments)")
    c = {**CONDITION_DEFAULTS, **condition}
    missing = [key for key in ["carrying_capacity", "growth_rate", "initial_stock"] if c["scenario"] == "commons" and key not in c]
    if missing:
        raise ExperimentError(f"commons condition {condition} must give {', '.join(missing)}")
    commons = {key: c[key] for key in ["carrying_capacity", "growth_rate", "initial_stock"]} if c["scenario"] == "commons" else None
    groups = c["groups"] if c["scenario"] in ["allotment", "capabilities"] else 1
    return harvest_condition(c["scenario"], _yes(c["seasons"]), c["season_length"], groups, c["layout"], society_mix, num_agents, days,
                             commons, config_name)


def _config_name(experiment):
    #as run.py names a training run's condition: after its training config file, when it isn't the scenario's own
    path = experiment["training_config"]
    if path is None:
        return None
    path = path if os.path.isabs(path) else os.path.join(REPO, path)
    return None if tc.load(path) == tc.DEFAULT else tc.config_name(path)


def plan(experiment):
    """Every run of the experiment: its training runs, then its test runs (each after its training run)"""
    python, run_py = sys.executable, os.path.join(REPO, "run.py")
    root, name, mix = experiment["output_root"], experiment["name"], experiment["society_mix"]
    common = ["--output-root", root, "--experiment", name, "--num-threads", str(experiment["num_threads"])]
    societies = [s for s in experiment["societies"] if not (mix == "heterogeneous" and s == "baseline")]
    config_name = _config_name(experiment)
    config_flags = [] if experiment["training_config"] is None else ["--training-config", experiment["training_config"] if os.path.isabs(
        experiment["training_config"]) else os.path.join(REPO, experiment["training_config"])]
    runs = []
    trained = []
    for train in experiment["train"]:
        condition = _condition_name(train, train["num_agents"], train["days"], mix, config_name)
        for society in societies:
            for seed in experiment["seeds"]:
                folder = run_folder(root, name, "train", condition, society, f"seed_{seed}")
                converging = (["--until-converged", "--convergence-max-episodes", str(experiment["convergence_max_episodes"])]
                              if experiment["until_converged"] else [])
                command = [python, run_py, "train", *common, "--seed", str(seed), "--checkpoint-every", str(experiment["checkpoint_every"]),
                           *config_flags, *converging, *_condition_flags(train, train["num_agents"]), "--days", str(train["days"]),
                           "--society-mix", mix, "--agent-type", society, "--write-data", "y"]
                runs.append(PlannedRun("train", condition, society, seed, folder, command))
                trained.append((train, society, seed, folder))
    for train, society, seed, train_folder in trained:
        for test in experiment["tests"]:
            #named after the training config too (as run.py names a test given it), so tests of agents trained with different configs
            #(e.g. an experiment file per reward function) never share a folder
            condition = _condition_name(test, train["num_agents"], train["days"], mix, config_name)
            folder = run_folder(root, name, "test", condition, society, f"train_seed_{seed}")
            norms = ["--norms", "y", "--norms-module", experiment["norms_module"]] if _yes(experiment["norms"]) else ["--norms", "n"]
            command = [python, run_py, "test", *common, "--seed", str(seed), "--test-seed", str(experiment["test_seed"]),
                       "--test-episodes", str(experiment["test_episodes"]), "--checkpoint", train_folder, *config_flags,
                       *_condition_flags(test, train["num_agents"]), "--society-mix", mix, "--agent-type", society, "--write-data", "y",
                       *norms, "--render", "n"]
            runs.append(PlannedRun("test", condition, society, seed, folder, command, train_folder))
    folders = [run.folder for run in runs]
    if len(set(folders)) != len(folders):
        raise ExperimentError("two runs of the experiment would share a folder: check its conditions aren't repeated")
    return runs


def runner_log(experiment, run):
    """Where the runner keeps what a run printed before its own run.log existed (e.g. an invalid setting)"""
    return os.path.join(experiment["output_root"], experiment["name"], "runner_logs", run.name.replace("/", "__") + ".log")


def launch(experiment, run):
    """Run one run as a process of its own; returns its status afterwards"""
    log_path = runner_log(experiment, run)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as log:
        result = subprocess.run(run.command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    status = read_status(run.folder)
    if result.returncode != 0 and status != "failed":
        status = "failed"
    return status


def run_experiment(experiment, max_workers=1, log=print, launch=launch):
    """
    Run every run that isn't complete, at most max_workers at a time: training runs first, then the test runs of the training runs that
    completed. Returns {run name: status}; failed runs, and test runs whose training didn't complete, are reported
    """
    runs = plan(experiment)
    outcomes = {}

    def done(run):
        #with until_converged, a complete training run is done only once its learning has converged (or reached its max episodes)
        if read_status(run.folder) != "complete":
            return False
        return not (run.mode == "train" and experiment["until_converged"]) or training_finished(run.folder)

    def go(run):
        if done(run):
            return run, "complete (skipped)"
        log(f"starting {run.name}")
        return run, launch(experiment, run)

    for mode in ["train", "test"]:
        ready = []
        for run in [r for r in runs if r.mode == mode]:
            if run.train_folder is not None and read_status(run.train_folder) != "complete":
                outcomes[run.name] = "not run: its training run isn't complete"
                continue
            ready.append(run)
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            for run, status in pool.map(go, ready):
                outcomes[run.name] = status
                log(f"{run.name}: {status}" + (f" (see {runner_log(experiment, run)} and {run.folder}/run.log)" if status == "failed" else ""))
    failed = [name for name, status in outcomes.items() if status == "failed"]
    blocked = [name for name, status in outcomes.items() if status.startswith("not run")]
    log(f"{len(outcomes)} runs: {sum(s.startswith('complete') for s in outcomes.values())} complete, {len(failed)} failed, "
        f"{len(blocked)} not run because their training didn't complete")
    for name in failed:
        log(f"  failed: {name}")
    return outcomes


def status_table(experiment):
    """Each planned run's status, from its status file"""
    import pandas as pd
    return pd.DataFrame([{"mode": run.mode, "condition": run.condition, "society": run.society, "seed": run.seed,
                          "status": read_status(run.folder), "folder": run.folder} for run in plan(experiment)])


def main(path, max_workers=1, list_only=False, status=False):
    experiment = load_experiment(path)
    if list_only:
        for run in plan(experiment):
            print(shlex.join(run.command))
        return 0
    if status:
        table = status_table(experiment)
        print(table.groupby(["mode", "status"]).size().to_string())
        not_complete = table[table["status"] != "complete"]
        if len(not_complete):
            print(not_complete[["mode", "condition", "society", "seed", "status"]].to_string(index=False))
        return 0
    outcomes = run_experiment(experiment, max_workers)
    return 1 if any(status == "failed" for status in outcomes.values()) else 0

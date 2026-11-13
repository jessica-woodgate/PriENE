"""
Runs a sweep's training runs through run.py, each in its own folder (run.py writes to relative paths, so each run's data/ lands in its
folder and nothing is shared or overwritten). Resumable: completed and early-stopped runs are skipped; a run stopped partway is continued
from its last finished episode (run.py's continue prompt). Runs can go in parallel, each with its own seed.

Each run's folder holds:
    training_config.yaml -- the full training config it was given (run.py --training-config)
    run.json -- the run's description and status: configuration, values, agent type, seed, prompt answers, the full training config, git
        commit, times, status ("running", "done", "failed", "stopped_early", "skipped") and why it stopped
    run.log -- run.py's output
    data/ -- everything run.py wrote
"""
import datetime
import glob
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from .. import training_config as tc
from .spec import Sweep, Run, scenario_answers, scenario_report, REPO

FINISHED = ("done", "stopped_early", "skipped")
POLL_SECONDS = 20

def read_status(run):
    path = os.path.join(run.folder, "run.json")
    if not os.path.exists(path):
        return None
    with open(path) as file:
        return json.load(file)

def write_status(run, record):
    path = os.path.join(run.folder, "run.json")
    writing = path + ".writing"
    with open(writing, "w") as file:
        json.dump(record, file, indent=2, default=str)
    os.replace(writing, path)

def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def has_saved_progress(run):
    return bool(glob.glob(os.path.join(run.folder, "data", "model_variables", "**", "progress.json"), recursive=True))

def answers_for(sweep, run):
    """
    The prompt answers for a run; a run that saved progress before it stopped is continued (run.py asks whether to)
    """
    answers = scenario_answers(sweep.scenario, run.agent_type, sweep.society_mix)
    return answers + (["y"] if has_saved_progress(run) else [])

def report_frame(sweep, run):
    """
    The run's episode report so far (None if it hasn't written one)
    """
    paths = glob.glob(os.path.join(run.folder, scenario_report(sweep.scenario)["report"]))
    if not paths:
        return None
    try:
        frame = pd.read_csv(paths[0])
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None
    return frame if len(frame) else None

def early_stop_reason(sweep, frame):
    """
    Why a run should stop now (docs/fruit_market_sweep.md §3.5), or None: checked once the report reaches early_stopping's episode, on
    the window of episodes before it: mean score below min_reward, a single action above max_action_share of all actions, or a NaN loss
    """
    rules = sweep.early_stopping
    if not rules or frame is None:
        return None
    episode = rules.get("episode", 100)
    window = rules.get("window", 20)
    if frame["episode"].max() < episode:
        return None
    recent = frame[(frame["episode"] > episode - window) & (frame["episode"] <= episode)]
    columns = scenario_report(sweep.scenario)["columns"]
    score = recent[columns["score"]].mean()
    if rules.get("min_reward") is not None and score < rules["min_reward"]:
        return f"mean {columns['score']} {score:.3f} over episodes {episode - window + 1}-{episode} is below {rules['min_reward']}"
    share_column = columns.get("max_action_share")
    if rules.get("max_action_share") is not None and share_column in recent and recent[share_column].mean() > rules["max_action_share"]:
        return f"one action is {recent[share_column].mean():.2f} of all actions over episodes {episode - window + 1}-{episode}"
    if recent[columns["loss"]].isna().any():
        return "the loss is NaN"
    return None

def default_launch(command, answers, folder, log_path):
    log = open(log_path, "a")
    process = subprocess.Popen(command, cwd=folder, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True)
    process.stdin.write("\n".join(answers) + "\n")
    process.stdin.close()
    return process

class SweepRunner:
    """
    launch(command, answers, folder, log_path) starts a run and returns a Popen-like object (poll, terminate, wait, returncode); tests
    pass their own
    """
    def __init__(self, sweep: Sweep, workers=1, launch=default_launch, poll_seconds=POLL_SECONDS, log=print):
        self.sweep = sweep
        self.workers = workers
        self.launch = launch
        self.poll_seconds = poll_seconds
        self.log = log
        self.stopped_configurations = set()
        self.lock = threading.Lock()

    def prepare(self):
        """
        Write the sweep's configurations (once): a later run of the same sweep must have the same ones, so a changed sweep file can't mix
        its runs with an earlier version's
        """
        os.makedirs(self.sweep.folder, exist_ok=True)
        path = os.path.join(self.sweep.folder, "configurations.json")
        configurations = [{"id": c["id"], "values": c["values"], "training_config": self.sweep.training_config(c["values"]).to_dict()}
                          for c in self.sweep.configurations]
        if os.path.exists(path):
            with open(path) as file:
                if json.load(file) != json.loads(json.dumps(configurations)):
                    raise RuntimeError(f"{path} lists different configurations from {self.sweep.path}: the sweep file changed after this "
                                       f"sweep started. Use a new step name for the new sweep.")
        else:
            with open(path, "w") as file:
                json.dump(configurations, file, indent=2)
        with open(os.path.join(self.sweep.folder, "sweep.yaml"), "w") as file:
            with open(self.sweep.path) as source:
                file.write(source.read())

    def pending(self):
        return [run for run in self.sweep.runs() if (read_status(run) or {}).get("status") not in FINISHED]

    def run_all(self, dry_run=False):
        self.prepare()
        #a configuration stopped early in an earlier session doesn't start its remaining seeds either
        for run in self.sweep.runs():
            if (read_status(run) or {}).get("status") == "stopped_early":
                self.stopped_configurations.add((run.config_id, run.agent_type))
        pending = self.pending()
        self.log(f"{self.sweep.name}: {len(self.sweep.runs())} runs, {len(pending)} still to do")
        if dry_run:
            for run in pending:
                self.log(f"  would run {run.name} in {run.folder}: {run.values}")
            return pending
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            list(pool.map(self.run_one, pending))
        return pending

    def run_one(self, run: Run):
        key = (run.config_id, run.agent_type)
        os.makedirs(run.folder, exist_ok=True)
        previous = read_status(run) or {}
        record = {**previous, "sweep": self.sweep.name, "scenario": self.sweep.scenario, "step": self.sweep.step,
                  "config_id": run.config_id, "values": run.values, "agent_type": run.agent_type, "seed": run.seed}
        with self.lock:
            if key in self.stopped_configurations:
                write_status(run, {**record, "status": "skipped", "reason": "another seed of this configuration was stopped early"})
                return
        config = self.sweep.training_config(run.values)
        config_path = os.path.join(run.folder, "training_config.yaml")
        tc.save(config, config_path)
        answers = answers_for(self.sweep, run)
        command = [sys.executable, os.path.join(REPO, "run.py"), "train", "-s", str(run.seed), "--training-config", config_path]
        started = datetime.datetime.now().isoformat(timespec="seconds")
        record.update({"status": "running", "training_config": config.to_dict(), "answers": answers, "command": command,
                       "git_commit": git_commit(), "started": started, "attempts": previous.get("attempts", 0) + 1})
        write_status(run, record)
        self.log(f"start {run.name}")
        begin = time.time()
        process = self.launch(command, answers, run.folder, os.path.join(run.folder, "run.log"))
        reason = None
        while process.poll() is None:
            time.sleep(self.poll_seconds)
            if reason is None:
                reason = early_stop_reason(self.sweep, report_frame(self.sweep, run))
                if reason is not None:
                    process.terminate()
                    process.wait()
        record.update({"finished": datetime.datetime.now().isoformat(timespec="seconds"), "seconds": round(time.time() - begin, 1),
                       "returncode": process.returncode})
        if reason is not None:
            record.update({"status": "stopped_early", "reason": reason})
            with self.lock:
                self.stopped_configurations.add(key)
        elif process.returncode == 0:
            record["status"] = "done"
        else:
            record.update({"status": "failed", "reason": f"run.py exited with {process.returncode}; see run.log"})
        write_status(run, record)
        self.log(f"{record['status']} {run.name} after {record['seconds']} s" + (f": {record.get('reason')}" if record.get("reason") else ""))

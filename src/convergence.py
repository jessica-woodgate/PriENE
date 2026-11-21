"""
When a training run's learning has converged, for training on past the end of exploration (run.py --until-converged): only episodes
at the exploration floor (epsilon_min) count, so falling exploration isn't mistaken for learning. Every CHECK_EVERY episodes, the mean
of each of METRICS over the last WINDOW episodes is compared with its mean over the WINDOW before; learning has converged once every
metric's difference is under THRESHOLD standard errors at CONSECUTIVE checks in a row (so a single quiet stretch isn't enough). Training
also stops at max_episodes, converged or not. Each check's outcome is written to the run's convergence file.
"""
import glob
import json
import os
import numpy as np
import pandas as pd

METRICS = ["mean_reward", "deceased", "mean_berries_thrown"]
WINDOW = 500
CHECK_EVERY = 100
CONSECUTIVE = 5
THRESHOLD = 2.0
#training stops at this episode whether or not learning has converged (about twice the ~4,600 episodes exploration takes to decay)
MAX_EPISODES = 10000


def window_statistics(report, end, window=WINDOW, metrics=METRICS):
    """
    For each metric, the difference between its mean over the window episodes up to and including row end (an index into report) and its
    mean over the window before, in standard errors (Welch's t)
    """
    last = report.iloc[end - window + 1:end + 1]
    previous = report.iloc[end - 2 * window + 1:end - window + 1]
    statistics = {}
    for metric in metrics:
        a, b = previous[metric].astype(float), last[metric].astype(float)
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        difference = b.mean() - a.mean()
        statistics[metric] = 0.0 if se == 0 and difference == 0 else (np.inf if se == 0 else float(difference / se))
    return statistics


def assess(report, epsilon_min, window=WINDOW, check_every=CHECK_EVERY, consecutive=CONSECUTIVE, threshold=THRESHOLD, metrics=METRICS):
    """
    Whether learning has converged by the last episode in report (a model episode report): every metric within threshold standard errors
    at each of the last consecutive checks, every check_every episodes back from the last, each over two windows of episodes all at the
    exploration floor. Returns (converged, details)
    """
    at_floor = report["epsilon"].round(5) <= round(epsilon_min, 5)
    #the first row from which every episode is at the floor (after the last episode above it)
    above_floor = np.flatnonzero(~at_floor.to_numpy())
    floor_start = int(above_floor[-1] + 1) if len(above_floor) else 0
    needed = 2 * window + (consecutive - 1) * check_every
    details = {"episode": int(report["episode"].iloc[-1]) if len(report) else None, "episodes_at_floor": len(report) - floor_start,
               "episodes_needed_at_floor": needed, "window": window, "check_every": check_every, "consecutive": consecutive,
               "threshold": threshold, "checks": []}
    if len(report) - floor_start < needed:
        return False, details
    converged = True
    for k in range(consecutive):
        end = len(report) - 1 - k * check_every
        statistics = window_statistics(report, end, window, metrics)
        passed = all(abs(t) < threshold for t in statistics.values())
        details["checks"].append({"episode": int(report["episode"].iloc[end]), "t": statistics, "passed": passed})
        converged = converged and passed
    return converged, details


class ConvergenceMonitor:
    """
    For run.py's training loop: whether a run at the exploration floor keeps training (keep_training), assessed from its model episode
    report at the start of every check_every-th episode, and recorded in convergence_path
    """
    def __init__(self, report_path, convergence_path, epsilon_min, max_episodes=MAX_EPISODES, **criterion):
        self.report_path = report_path
        self.convergence_path = convergence_path
        self.epsilon_min = epsilon_min
        self.max_episodes = max_episodes
        self.criterion = criterion
        self._decided = (None, True)

    def keep_training(self, episode):
        """Whether the run trains episode (the episode it is about to start, or is in)"""
        decided_for, keep = self._decided
        if decided_for == episode:
            return keep
        if episode > self.max_episodes:
            keep = False
            self._record(episode - 1, False, {"stopped": f"reached max_episodes ({self.max_episodes}) before converging"})
        elif decided_for is None or (episode - 1) % self.criterion.get("check_every", CHECK_EVERY) == 0:
            converged, details = assess(pd.read_csv(self.report_path), self.epsilon_min, **self.criterion)
            keep = not converged
            self._record(episode - 1, converged, details)
        self._decided = (episode, keep)
        return keep

    def _record(self, episode, converged, details):
        record = {"converged": converged, "after_episode": episode, "max_episodes": self.max_episodes, **details}
        writing = self.convergence_path + ".writing"
        with open(writing, "w") as file:
            json.dump(record, file, indent=2)
        os.replace(writing, self.convergence_path)


#a training run's convergence file, beside its model episode report: data/results/current_run/convergence_<output name>.json
CONVERGENCE_PREFIX = "convergence_"


def finished(convergence_path):
    """Whether a run's convergence file says it is done: converged, or stopped at its max_episodes"""
    if not os.path.exists(convergence_path):
        return False
    with open(convergence_path) as file:
        record = json.load(file)
    return bool(record.get("converged")) or "stopped" in record


def training_finished(run_folder):
    """
    Whether a training run's folder (run.py --output-root) says its training is done with --until-converged: it has a convergence file
    for every model episode report, and each says converged or stopped at max_episodes. The files are where run.py writes them
    (data/results/current_run/), or moved up into data/results/ once the run is done
    """
    for results in [os.path.join(run_folder, "data", "results", "current_run"), os.path.join(run_folder, "data", "results")]:
        reports = glob.glob(os.path.join(results, "model_episode_reports_*.csv"))
        if reports:
            names = [os.path.basename(report)[len("model_episode_reports_"):-len(".csv")] for report in reports]
            return all(finished(os.path.join(results, CONVERGENCE_PREFIX + name + ".json")) for name in names)
    return False

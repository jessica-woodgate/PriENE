"""
Reference outputs: short seeded runs across the configuration space, whose outputs must not change when the code is refactored.

    python tests/reference_outputs.py --save      # run everything and save the outputs and their hashes (from the current code)
    python tests/reference_outputs.py             # run everything again and compare with the saved hashes
    python tests/reference_outputs.py --only basic   # just the runs whose name contains "basic"

Runs go through run.py, answering its prompts as a person would, so the prompts, output and checkpoint names, settings.json and the
continue-training prompt are all covered. The only change is the length of runs (see _shorten_runs): training stops after a few
episodes (run.py otherwise trains until exploration has decayed, ~4,600 episodes) and testing after a few episodes (run.py: 1,000).
Allotment regrowth delays have no prompt, so they are run directly (see DIRECT_RUNS).

Each run is in a temporary directory of its own (run.py writes to relative paths); every file it writes under data/ is hashed:
CSVs and JSONs as they are, checkpoints (.keras zips) by their model config and weights, since their metadata records the time saved.
"""
import argparse
import glob
import hashlib
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_DIR = os.path.join(REPO, "tests", "reference_data")
HASHES_FILE = os.path.join(REFERENCE_DIR, "hashes.json")
OUTPUTS_DIR = os.path.join(REFERENCE_DIR, "outputs")
SEED = 42
MAX_DAYS = 10
TRAIN_EPISODES = 10
TEST_EPISODES = 4
SEASON_LENGTH = 3
#graphs: images aren't compared (their bytes carry metadata such as the matplotlib version); the data behind them (CSVs, text) is
IMAGE_EXTENSIONS = (".png", ".pdf", ".svg", ".jpg")
AGENT_TYPES = ["baseline", "utilitarian", "maximin", "egalitarian", "average", "majoritarian", "optimist", "veto"]


# ── building runs ────────────────────────────────────────────────

def _allocation_answers(scenario, mix, layout, num_allocations):
    if scenario in ["allotment", "capabilities"]:
        answers = [str(num_allocations)]
        if scenario == "allotment" and mix == "heterogeneous" and num_allocations % 2 == 0:
            answers.append(layout)
        return answers
    return []


def _commons_answers(scenario, commons):
    if scenario != "commons":
        return []
    return list(commons) if commons else ["", "", ""]


def train_step(scenario, seasons, mix, agent_type, layout="split", num_agents=4, num_allocations=2, commons=None, episodes=TRAIN_EPISODES,
               continuing=None):
    """
    Answers to run.py train's prompts, in order: scenario, seasons, [season length], days, society mix, agent type, agents,
    [allotments or capabilities, [layout]], [commons K, r, R0], write data, [continue y for each type already trained]
    """
    answers = [scenario, "y" if seasons else "n"] + ([str(SEASON_LENGTH)] if seasons else []) + [str(MAX_DAYS), mix, agent_type, str(num_agents)]
    answers += _allocation_answers(scenario, mix, layout, num_allocations) + _commons_answers(scenario, commons) + ["y"]
    answers += ["y"] * (continuing or 0)
    return {"mode": "train", "answers": answers, "episodes": episodes}


def test_step(scenario, seasons, mix, agent_type, checkpoint, norms="rule_based", layout="split", num_agents=4, num_allocations=2,
              commons=None, episodes=TEST_EPISODES):
    """
    Answers to run.py test's prompts, in order: scenario, seasons, [season length], checkpoint, society mix, agent type, agents,
    [allotments or capabilities, [layout]], [commons K, r, R0], write data, write norms, [norms module], render
    (days come from the checkpoint's name)
    """
    answers = [scenario, "y" if seasons else "n"] + ([str(SEASON_LENGTH)] if seasons else []) + [checkpoint, mix, agent_type, str(num_agents)]
    answers += _allocation_answers(scenario, mix, layout, num_allocations) + _commons_answers(scenario, commons) + ["y"]
    answers += (["y", norms] if norms else ["n"]) + ["n"]
    return {"mode": "test", "answers": answers, "episodes": episodes}


#a test step's checkpoint answer: the name of the checkpoint its run's training saved (found when the step runs)
TRAINED = "{trained}"


def build_runs():
    """
    Every run: a name and its steps, run one after another in one directory (so tests load the checkpoints the training saved).
    Results are moved aside after each step (and before the next), except between the two steps of a continued training run
    """
    runs = {}
    types = iter(AGENT_TYPES[1:] * 10)
    for scenario in ["basic", "colours", "capabilities", "allotment", "commons"]:
        mixes = [("homogeneous", "split"), ("heterogeneous", "split")] + ([("heterogeneous", "separate")] if scenario == "allotment" else [])
        for seasons in [False, True]:
            for mix, layout in mixes:
                # a different agent type each time, so every principle and aggregation is trained and tested
                agent_type = next(types)
                name = f"{scenario}_{'seasons' if seasons else 'noseasons'}_{mix if mix == 'homogeneous' else 'mixed-' + layout}_{agent_type}"
                steps = [train_step(scenario, seasons, mix, agent_type, layout)]
                for norms in ["rule_based", "bayesian"]:
                    steps.append(test_step(scenario, seasons, mix, agent_type, TRAINED, norms, layout))
                if not seasons and mix == "homogeneous":
                    steps.append(test_step(scenario, seasons, mix, agent_type, TRAINED, None, layout))
                runs[name] = steps
    # baseline agents trained alone (the loop above trains them only alongside other types)
    runs["colours_noseasons_homogeneous_baseline"] = [train_step("colours", False, "homogeneous", "baseline"),
                                                      test_step("colours", False, "homogeneous", "baseline", TRAINED)]
    # the paper's workflow: every agent type trained in basic, then tested in the other scenarios, with graphs of the colours tests
    # (run.py graphs: current_run, colours, 4 agents, no seasons, homogeneous, rule-based norms written)
    colours_test = test_step("colours", False, "homogeneous", "all", TRAINED, episodes=3)
    colours_test["keep_results"] = True
    graphs = {"mode": "graphs", "answers": ["current_run", "colours", "4", "n", "homogeneous", "y"], "episodes": 0}
    runs["paper_all_types"] = [train_step("basic", False, "homogeneous", "all", episodes=3), colours_test, graphs] + [
        test_step(scenario, False, "homogeneous", "all", TRAINED, episodes=2) for scenario in ["capabilities", "allotment"]]
    # more agents and allocations
    runs["allotment_6agents_3allotments_mixed"] = [
        train_step("allotment", True, "heterogeneous", "maximin", num_agents=6, num_allocations=3),
        test_step("allotment", True, "heterogeneous", "maximin", TRAINED, "bayesian", num_agents=6, num_allocations=3)]
    runs["capabilities_6agents_3capabilities"] = [
        train_step("capabilities", False, "homogeneous", "egalitarian", num_agents=6, num_allocations=3),
        test_step("capabilities", False, "homogeneous", "egalitarian", TRAINED, num_agents=6, num_allocations=3)]
    # the commons with its own parameters
    commons = ("30", "0.5", "10")
    runs["commons_parameters"] = [train_step("commons", True, "homogeneous", "utilitarian", commons=commons),
                                  test_step("commons", True, "homogeneous", "utilitarian", TRAINED, "rule_based", commons=commons)]
    # continued training: stopped after 5 episodes, continued to 10 (results stay in place between the two)
    for scenario, mix, layout in [("basic", "homogeneous", "split"), ("allotment", "heterogeneous", "separate"), ("commons", "homogeneous", "split")]:
        first = train_step(scenario, True, mix, "veto", layout, episodes=5)
        second = train_step(scenario, True, mix, "veto", layout, episodes=5, continuing=1)
        first["keep_results"] = True
        runs[f"continued_{scenario}_{mix}_{layout}"] = [first, second]
    # run.py graphs "paper": the paper's graphs from its 200-day results (copied in, and not hashed, as they are its input)
    runs["paper_graphs"] = [{"mode": "graphs", "answers": ["paper", "colours"], "episodes": 0,
                             "inputs": ["data/results/200_days/4_agents/colours/agent_end_episode_df_*.csv",
                                        "data/results/200_days/4_agents/colours/end_episode_totals_*.csv"]}]
    # allotment regrowth delays (no prompt: run directly)
    runs["allotment_regrowth_delays_direct"] = [{"mode": "direct", "training": True, "episodes": TRAIN_EPISODES},
                                                {"mode": "direct", "training": False, "episodes": TEST_EPISODES}]
    return runs


# ── running a step (in its own process) ──────────────────────────

def _shorten_runs(episodes):
    """
    Stops run.py's loop once `episodes` episodes have run (from the episode the model starts at): training by raising min_epsilon
    above epsilon, testing by lowering max_episodes, once the episode counter reaches the end. Nothing else about the run changes
    """
    from src.harvest_model import HarvestModel
    original_init, original_step = HarvestModel.__init__, HarvestModel.step

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._reference_last_episode = self.episode + episodes - 1

    def step(self):
        original_step(self)
        if self.episode > self._reference_last_episode:
            self.min_epsilon = 2.0
            self.max_episodes = 0
    HarvestModel.__init__, HarvestModel.step = init, step


def _run_direct(training, episodes):
    """Allotment with regrowth delays [1, 3] (and seasons), built and run as run.py's create_and_run_model and run_simulation do."""
    from src.checkpoints import prepare_training_checkpoints, training_checkpoint_name, training_checkpoint_path
    from src.scenarios.allotment_harvest import AllotmentHarvest
    delays = [1, 3]
    checkpoint_path = training_checkpoint_path(training_checkpoint_name("allotment", True, SEASON_LENGTH, 2, delays, MAX_DAYS, SEED), 4)
    if training:
        prepare_training_checkpoints(checkpoint_path, ["utilitarian"], {"seed": SEED, "allotment_regrowth_delays": delays})
    filepath = ("train_" if training else "") + "allotment_seasons_regrow1-3_utilitarian"
    model = AllotmentHarvest("homogeneous", 4, 12, 2, "utilitarian", 16, 8, 0 if training else 1000, MAX_DAYS, training, checkpoint_path, True,
                             not training, filepath, SEED, "rule_based", True, SEASON_LENGTH, delays)
    while (model.training and round(model.epsilon, 5) > model.min_epsilon) or (not model.training and model.episode <= model.max_episodes):
        model.step()
    model.finish_episode(collect_data=False)


def run_step(step):
    """Runs one step in the current directory; called in a fresh interpreter (see _run_in_subprocess)."""
    sys.path.insert(0, REPO)
    import matplotlib
    matplotlib.use("Agg")
    _shorten_runs(step["episodes"])
    if step["mode"] == "direct":
        _run_direct(step["training"], step["episodes"])
        return
    sys.argv = [os.path.join(REPO, "run.py"), step["mode"], "-s", str(SEED)]
    sys.stdin = io.StringIO("\n".join(step["answers"]) + "\n")
    runpy.run_path(os.path.join(REPO, "run.py"), run_name="__main__")


def _run_in_subprocess(step, directory):
    result = subprocess.run([sys.executable, os.path.abspath(__file__), "--step", json.dumps(step)], cwd=directory,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"step {step} failed:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")


# ── hashing outputs ──────────────────────────────────────────────

def _hash_keras(path):
    """A checkpoint's model config and weights (each weight's name, shape, dtype and values), not the time it was saved."""
    import h5py
    digest = hashlib.sha256()
    with zipfile.ZipFile(path) as archive:
        digest.update(archive.read("config.json"))
        with h5py.File(io.BytesIO(archive.read("model.weights.h5")), "r") as weights:
            def add(name, item):
                if isinstance(item, h5py.Dataset):
                    value = item[()]
                    digest.update(name.encode())
                    digest.update(str((getattr(value, "shape", ()), str(getattr(value, "dtype", type(value))))).encode())
                    digest.update(value.tobytes() if hasattr(value, "tobytes") else repr(value).encode())
            weights.visititems(add)
    return digest.hexdigest()


def hash_file(path):
    if path.endswith(".keras"):
        return _hash_keras(path)
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


def _collect(directory, label, outputs):
    """Hashes every file written under data/ (results moved aside after each step are under data/results/<step label>/)."""
    hashes = {}
    data = os.path.join(directory, "data")
    for root, _, files in os.walk(data):
        for name in sorted(files):
            if name.endswith(IMAGE_EXTENSIONS):
                continue
            full = os.path.join(root, name)
            relative = os.path.relpath(full, directory)
            hashes[relative] = hash_file(full)
            if outputs is not None:
                target = os.path.join(outputs, label, relative)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(full, target)
    return dict(sorted(hashes.items()))


def execute(name, steps, save_outputs):
    """Runs a run's steps in a fresh directory and returns the hash of every file it wrote."""
    directory = tempfile.mkdtemp(prefix=f"reference_{name}_")
    try:
        current = os.path.join(directory, "data", "results", "current_run")
        for i, step in enumerate(steps):
            if TRAINED in step.get("answers", []):
                trained = os.listdir(os.path.join(directory, "data", "model_variables", "trained"))
                assert len(trained) == 1, trained
                step = {**step, "answers": [trained[0] if answer == TRAINED else answer for answer in step["answers"]]}
            #as in the repository, where it always exists (run.py graphs "paper" writes into it without making it)
            os.makedirs(current, exist_ok=True)
            inputs = [path for pattern in step.get("inputs", []) for path in glob.glob(os.path.join(REPO, pattern))]
            for path in inputs:
                target = os.path.join(directory, os.path.relpath(path, REPO))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(path, target)
            _run_in_subprocess(step, directory)
            for path in inputs:
                os.remove(os.path.join(directory, os.path.relpath(path, REPO)))
            if not step.get("keep_results") and os.path.isdir(current):
                os.replace(current, os.path.join(directory, "data", "results", f"step{i}_{step['mode']}"))
        return name, _collect(directory, name, OUTPUTS_DIR if save_outputs else None)
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# ── main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--save", action="store_true", help="save the outputs and hashes as the reference (from the current code)")
    parser.add_argument("--only", help="only the runs whose name contains this")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", help="write this run's hashes here as well (e.g. to compare two runs)")
    parser.add_argument("--step", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.step:
        run_step(json.loads(args.step))
        return
    runs = {name: steps for name, steps in build_runs().items() if not args.only or args.only in name}
    if args.save and os.path.exists(REFERENCE_DIR) and not args.only:
        shutil.rmtree(REFERENCE_DIR)
    print(f"{len(runs)} runs, {sum(len(steps) for steps in runs.values())} steps")
    hashes = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, run_hashes in pool.map(execute, runs, runs.values(), [args.save] * len(runs)):
            hashes[name] = run_hashes
            print(f"  {name}: {len(run_hashes)} files", flush=True)
    hashes = dict(sorted(hashes.items()))
    if args.out:
        with open(args.out, "w") as file:
            json.dump(hashes, file, indent=1)
    if args.save:
        os.makedirs(REFERENCE_DIR, exist_ok=True)
        saved = {}
        if args.only and os.path.exists(HASHES_FILE):
            with open(HASHES_FILE) as file:
                saved = json.load(file)
        saved.update(hashes)
        with open(HASHES_FILE, "w") as file:
            json.dump(dict(sorted(saved.items())), file, indent=1)
        print(f"Saved {sum(len(h) for h in hashes.values())} hashes to {HASHES_FILE}")
        return
    with open(HASHES_FILE) as file:
        reference = json.load(file)
    differences = []
    for name, run_hashes in hashes.items():
        expected = reference.get(name)
        if expected is None:
            differences.append(f"{name}: no reference")
            continue
        for path in sorted(set(expected) | set(run_hashes)):
            if expected.get(path) != run_hashes.get(path):
                state = "missing" if path not in run_hashes else "new" if path not in expected else "differs"
                differences.append(f"{name}: {path} {state}")
    if differences:
        print(f"{len(differences)} DIFFERENCES:")
        print("\n".join(differences))
        sys.exit(1)
    print(f"All {sum(len(h) for h in hashes.values())} hashes match the reference")


if __name__ == "__main__":
    main()

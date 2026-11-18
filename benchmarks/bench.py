"""
Performance benchmarks: short, seeded, representative runs, each in a fresh process and a working directory of its own, timed per step
and per episode, with peak memory and the time spent in each part of a step (docs/performance_review.md).

    python benchmarks/bench.py list                         # the configurations
    python benchmarks/bench.py all [--only NAME] [--label L]    # run every configuration, write benchmarks/results/<label>.json
    python benchmarks/bench.py run NAME [--profile FILE] [--threads N] [--no-write] [--no-norms] [--no-parts]
    python benchmarks/bench.py hash [--only NAME] [--out FILE]  # run every configuration untimed and hash every output (see hash_outputs)
    python benchmarks/bench.py compare FILE [FILE2]           # compare hashes with benchmarks/reference_hashes.json (or FILE2)

Models are built as run.py's create_and_run_model builds them (the same sizes: width and height 2 per agent, allotment width 4 per
agent, 3 berries per agent, the commons' default parameters) and run as run_simulation runs them, stopped after a set number of episodes.
Test configurations load a checkpoint the harness trains first (basic, 2 episodes; mixed: utilitarian with baseline agents).

Part timings ("parts") wrap functions in timers (inclusive times: a part called inside another is counted in both), which costs a little;
--no-parts leaves them out, for the cleanest wall time. Nothing here changes how a run behaves.
"""
import argparse
import cProfile
import glob
import hashlib
import io
import json
import os
import pstats
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(REPO, "benchmarks", "results")
REFERENCE_HASHES = os.path.join(REPO, "benchmarks", "reference_hashes.json")
SEED = 42
TEST_SEED = 1000
DAYS = 200
NUM_AGENTS = 4
TRAIN_EPISODES = 3
TEST_EPISODES = 5
#the checkpoints test configurations load: trained by the harness (prepare_checkpoints), short, in basic
CHECKPOINT_EPISODES = 2


def harvest(name, scenario, training, **options):
    return {"name": name, "kind": "harvest", "scenario": scenario, "training": training, "mix": options.get("mix", "homogeneous"),
            "agent_type": options.get("agent_type", "utilitarian"), "num_agents": options.get("num_agents", NUM_AGENTS),
            "num_allocations": options.get("num_allocations", 2), "seasons": options.get("seasons", False),
            "delays": options.get("delays"), "track_norms": options.get("track_norms", not training),
            "norms": options.get("norms", "rule_based"), "write_data": options.get("write_data", True),
            "episodes": options.get("episodes", TRAIN_EPISODES if training else TEST_EPISODES), "days": options.get("days", DAYS),
            "late": options.get("late", False)}


#late training: a training run continued at this episode from the paper's agents (copied in as the run's checkpoint), so epsilon has
#decayed (~0.02) and agents live as trained agents do, as in most of a ~4,600-episode run (early episodes, exploring at random, last a
#few steps)
LATE_EPISODE = 4000


CONFIGS = [
    #training as run.py trains (norms off), and with rule-based or Bayesian norms tracked
    harvest("train_basic", "basic", True),
    harvest("train_basic_rule_norms", "basic", True, track_norms=True),
    harvest("train_basic_bayesian_norms", "basic", True, track_norms=True, norms="bayesian"),
    harvest("train_basic_mixed", "basic", True, mix="heterogeneous"),
    harvest("train_commons", "commons", True),
    harvest("train_basic_late", "basic", True, late=True, episodes=2),
    harvest("train_basic_late_rule_norms", "basic", True, late=True, episodes=2, track_norms=True),
    harvest("train_basic_late_mixed", "basic", True, late=True, episodes=2, mix="heterogeneous"),
    #testing with data and rule-based norms written, as the paper's tests
    harvest("test_colours", "colours", False),
    harvest("test_colours_seasons", "colours", False, seasons=True),
    harvest("test_capabilities", "capabilities", False),
    harvest("test_allotment", "allotment", False),
    harvest("test_allotment_seasons_delays", "allotment", False, seasons=True, delays=[1, 3]),
    harvest("test_colours_bayesian", "colours", False, norms="bayesian"),
    harvest("test_basic_mixed", "basic", False, mix="heterogeneous"),
    harvest("test_commons", "commons", False),
    #scaling: the paper's 20-agent agents
    harvest("test_colours_20agents", "colours", False, num_agents=20, episodes=2),
    {"name": "train_fruit_market", "kind": "fruit_market", "training": True, "num_agents": 8, "days": 100, "episodes": 2,
     "agent_type": "utilitarian", "mix": "homogeneous", "track_norms": False, "norms": "rule_based", "write_data": True},
]
CONFIG = {config["name"]: config for config in CONFIGS}


# ── building and running one configuration ───────────────────────

def checkpoint_folder(config, prepared):
    """Where a training configuration saves, or a test configuration loads from (prepared: the harness's trained checkpoints)"""
    from src.checkpoints import training_checkpoint_name, training_checkpoint_path, society_checkpoint_path, loading_checkpoint_path
    if config["training"]:
        name = training_checkpoint_name(config["scenario"], config["seasons"], 25, config["num_allocations"],
                                        config["delays"] or [0, 2], config["days"], SEED, config["mix"],
                                        commons_parameters=commons_parameters(config) if config["scenario"] == "commons" else None)
        return society_checkpoint_path(training_checkpoint_path(name, config["num_agents"]), config["mix"], config["agent_type"])
    if config["scenario"] == "commons":
        #the commons adds an observation feature, so its agents can't be the paper's
        root = os.path.join(prepared, "commons", f"{config['num_agents']}_agents") + "/"
    else:
        #the paper's agents (trained in basic for 200 days), so test episodes last as long as they do in the paper's tests
        from src.checkpoints import PAPER_CHECKPOINTS
        root = os.path.join(REPO, PAPER_CHECKPOINTS, f"{config['num_agents']}_agents") + "/"
    return loading_checkpoint_path(root, config["mix"], config["agent_type"])


def commons_parameters(config):
    from src.scenarios.commons_harvest import CommonsHarvest
    n = config["num_agents"]
    capacity = min(CommonsHarvest.CARRYING_CAPACITY_PER_AGENT * n, (2 * n) * (2 * n) - n)
    return {"carrying_capacity": capacity, "growth_rate": CommonsHarvest.DEFAULT_GROWTH_RATE, "initial_stock": max(1, capacity // 2)}


def build_harvest(config, checkpoint_path):
    """The model run.py's create_and_run_model builds for these answers"""
    from src.scenarios.basic_harvest import BasicHarvest
    from src.scenarios.colours_harvest import ColoursHarvest
    from src.scenarios.capabilities_harvest import CapabilitiesHarvest
    from src.scenarios.allotment_harvest import AllotmentHarvest
    from src.scenarios.commons_harvest import CommonsHarvest
    from src.checkpoints import prepare_training_checkpoints
    n, scenario, training = config["num_agents"], config["scenario"], config["training"]
    width = n * 4 if scenario == "allotment" else n * 2
    height, berries = n * 2, n * 3
    filepath = ("train_" if training else "") + config["name"] + "_" + config["agent_type"]
    max_episodes = 0 if training else 1000
    start_episode = 1
    if training:
        society = os.path.dirname(checkpoint_path.rstrip("/")) + "/" if config["mix"] != "homogeneous" else checkpoint_path
        if config["late"]:
            from src.checkpoints import PAPER_CHECKPOINTS
            for agent_type in {config["agent_type"], "baseline"} if config["mix"] != "homogeneous" else {config["agent_type"]}:
                shutil.copytree(os.path.join(REPO, PAPER_CHECKPOINTS, f"{n}_agents", agent_type), os.path.join(checkpoint_path, agent_type))
            start_episode = LATE_EPISODE
        prepare_training_checkpoints(society, [config["agent_type"]], {"benchmark": config["name"]}, config["mix"],
                                     continuing=(config["agent_type"],) if config["late"] else ())
    common = dict(seed=SEED if training else TEST_SEED, norms_module_type=config["norms"], seasons=config["seasons"], season_length=25,
                  start_episode=start_episode)
    args = (config["mix"], n)
    rest = (config["agent_type"], width, height, max_episodes, config["days"], training, checkpoint_path, config["write_data"],
            config["track_norms"], filepath)
    if scenario == "basic":
        model = BasicHarvest(*args, berries, *rest, **common)
    elif scenario == "colours":
        model = ColoursHarvest(*args, berries, *rest, **common)
    elif scenario == "capabilities":
        model = CapabilitiesHarvest(*args, berries, config["num_allocations"], *rest, **common)
    elif scenario == "allotment":
        model = AllotmentHarvest(*args, berries, config["num_allocations"], *rest, **common, allotment_regrowth_delays=config["delays"])
    elif scenario == "commons":
        p = commons_parameters(config)
        model = CommonsHarvest(*args, p["carrying_capacity"], p["growth_rate"], *rest, **common, initial_stock=p["initial_stock"])
    if training and config["mix"] != "homogeneous":
        from src.checkpoints import record_agent_types
        record_agent_types(checkpoint_path, model.get_living_agents())
    return model


def build_fruit_market(config):
    from src.fruit_market.model import FruitMarketModel
    from src.checkpoints import fruit_market_checkpoint_name, training_checkpoint_path, prepare_training_checkpoints
    path = training_checkpoint_path(fruit_market_checkpoint_name(config["days"], True, "need", "attributed", "global", SEED), config["num_agents"])
    prepare_training_checkpoints(path, [config["agent_type"]], {"benchmark": config["name"]})
    return FruitMarketModel(config["mix"], config["num_agents"], config["agent_type"], 0, config["days"], True, path, config["write_data"],
                            config["track_norms"], "train_" + config["name"], SEED, config["norms"])


def run_episodes(model, episodes, on_step):
    """run.py's run_simulation, stopped once `episodes` episodes have run; on_step(seconds, finished_episode) after every step"""
    last = model.episode + episodes - 1
    while model.episode <= last:
        episode = model.episode
        start = time.perf_counter()
        model.step()
        on_step(time.perf_counter() - start, model.episode != episode)
    start = time.perf_counter()
    model.finish_episode(collect_data=False)
    return time.perf_counter() - start


# ── part timings ─────────────────────────────────────────────────

PARTS = {
    #TensorFlow
    "dqn.choose_action (forward pass + exploration draw)": ("src.agent.dqn.dqn", "DQN", "choose_action"),
    "dqn.train (sample + target forward + gradient step)": ("src.agent.dqn.dqn", "DQN", "train"),
    "dqn.copy_weights (target update)": ("src.agent.dqn.dqn", "DQN", "copy_weights"),
    "dqn.add_experience (replay buffer)": ("src.agent.dqn.dqn", "DQN", "add_experience"),
    "save_models (networks, optimiser, streams)": ("src.agent.modules.dqn_decision_module", "DQNDecisionModule", "save_models"),
    "save_replay_buffers": ("src.society_model", "SocietyModel", "save_replay_buffers"),
    #environment
    "observe": ("src.harvest_model", "HarvestModel", "observe"),
    "get_society_well_being": ("src.harvest_model", "HarvestModel", "get_society_well_being"),
    "moving.check_nearest_berry (search + A*)": ("src.agent.modules.moving_module", "MovingModule", "check_nearest_berry"),
    "moving._find_path_to_berry (A*)": ("src.agent.modules.moving_module", "MovingModule", "_find_path_to_berry"),
    "ethics (update + sanction)": ("src.agent.modules.ethics_module", "EthicsModule", "get_sanction"),
    "ethics.update_ethics_state": ("src.agent.modules.ethics_module", "EthicsModule", "update_ethics_state"),
    "agent_by_id": ("src.harvest_model", "HarvestModel", "agent_by_id"),
    "_update_schedule": ("src.harvest_model", "HarvestModel", "_update_schedule"),
    #reports and norms
    "_collect_agent_data (agent report rows)": ("src.harvest_model", "HarvestModel", "_collect_agent_data"),
    "_collect_model_episode_data": ("src.harvest_model", "HarvestModel", "_collect_model_episode_data"),
    "norms: get_antecedent": ("src.agent.modules.norms_module", "NormsModule", "get_antecedent"),
    "norms: update_behaviour_base": ("src.agent.modules.norms_module", "NormsModule", "update_behaviour_base"),
    "norms: _check_emerged_norms": ("src.society_model", "SocietyModel", "_check_emerged_norms"),
    "norms: _append_norm_dict_to_file": ("src.society_model", "SocietyModel", "_append_norm_dict_to_file"),
    "bayesian norms: update": ("src.agent.modules.bayesian_norms_module", "BayesianNormsModule", "update"),
    "bayesian norms: _analyse_agent_norms": ("src.society_model", "SocietyModel", "_analyse_agent_norms"),
    #Fruit Market
    "fruit market: substrate step": ("src.fruit_market.backend", "MeltingPotBackend", "step"),
    "fruit market: observe": ("src.fruit_market.model", "FruitMarketModel", "observe"),
    "fruit market: _record_step": ("src.fruit_market.model", "FruitMarketModel", "_record_step"),
}


def instrument():
    """Wrap each part in a timer; returns the totals (seconds) and call counts by part"""
    import importlib
    totals, calls = defaultdict(float), defaultdict(int)
    for label, (module_name, class_name, function_name) in PARTS.items():
        try:
            cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError):
            continue
        original = cls.__dict__.get(function_name)
        if original is None:
            continue

        def timed(*args, _original=original, _label=label, **kwargs):
            start = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                totals[_label] += time.perf_counter() - start
                calls[_label] += 1
        setattr(cls, function_name, timed)
    return totals, calls


# ── one configuration, in this process ───────────────────────────

def run_config(name, workdir, prepared, profile=None, threads=None, no_write=False, no_norms=False, parts=True, episodes=None):
    config = dict(CONFIG[name])
    if no_write:
        config["write_data"] = False
    if no_norms:
        config["track_norms"] = False
    if episodes is not None:
        config["episodes"] = episodes
    os.chdir(workdir)
    sys.path.insert(0, REPO)
    import tensorflow
    if threads is not None:
        tensorflow.config.threading.set_intra_op_parallelism_threads(threads)
        tensorflow.config.threading.set_inter_op_parallelism_threads(threads)
    totals, calls = instrument() if parts else ({}, {})
    os.makedirs("data/results/current_run", exist_ok=True)
    start = time.perf_counter()
    if config["kind"] == "fruit_market":
        model = build_fruit_market(config)
    else:
        model = build_harvest(config, checkpoint_folder(config, prepared))
    build_seconds = time.perf_counter() - start
    for key in list(totals):
        totals[key], calls[key] = 0.0, 0
    steps, episode_seconds, current = [], [], [0.0]

    def on_step(seconds, finished):
        steps.append(seconds)
        current[0] += seconds
        if finished:
            episode_seconds.append(current[0])
            current[0] = 0.0
    profiler = cProfile.Profile() if profile else None
    start = time.perf_counter()
    if profiler:
        profiler.enable()
    end_seconds = run_episodes(model, config["episodes"], on_step)
    if profiler:
        profiler.disable()
    run_seconds = time.perf_counter() - start
    if getattr(model, "backend", None) is not None:
        model.backend.close()
    if profiler:
        profiler.dump_stats(profile)
        for sort in ["cumulative", "tottime"]:
            text = io.StringIO()
            pstats.Stats(profiler, stream=text).strip_dirs().sort_stats(sort).print_stats(45)
            with open(f"{profile}.{sort}.txt", "w") as file:
                file.write(text.getvalue())
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / 2**20 if sys.platform == "darwin" else peak / 2**10
    return {"name": name, "config": config, "threads": threads, "build_s": build_seconds, "run_s": run_seconds,
            "end_of_run_save_s": end_seconds, "episodes": len(episode_seconds), "steps": len(steps),
            "s_per_episode": sum(episode_seconds) / max(1, len(episode_seconds)),
            "ms_per_step": 1000 * sum(steps) / max(1, len(steps)), "peak_rss_mb": peak_mb,
            "parts_s": dict(sorted(totals.items(), key=lambda item: -item[1])), "part_calls": dict(calls)}


# ── preparing checkpoints and driving every configuration ────────

def prepare_checkpoints(prepared):
    """Short basic (and commons) training runs whose checkpoints the test configurations load: homogeneous utilitarian, and
    utilitarian with baseline agents"""
    if os.path.exists(os.path.join(prepared, "done")):
        return
    shutil.rmtree(prepared, ignore_errors=True)
    os.makedirs(prepared)
    for scenario, mix in [("basic", "homogeneous"), ("basic", "heterogeneous"), ("commons", "homogeneous")]:
        work = tempfile.mkdtemp(prefix="bench_prepare_")
        name = f"prepare_{scenario}_{mix}"
        CONFIG[name] = harvest(name, scenario, True, mix=mix, episodes=CHECKPOINT_EPISODES)
        subprocess.run([sys.executable, __file__, "run", name, "--workdir", work, "--no-parts", "--prepare-scenario", scenario, "--prepare-mix", mix],
                       check=True, capture_output=True)
        trained = glob.glob(os.path.join(work, "data", "model_variables", "trained", "*", f"{NUM_AGENTS}_agents"))
        target = os.path.join(prepared, scenario, f"{NUM_AGENTS}_agents")
        os.makedirs(target, exist_ok=True)
        for item in os.listdir(trained[0]):
            source = os.path.join(trained[0], item)
            if os.path.isdir(source):
                shutil.copytree(source, os.path.join(target, item), dirs_exist_ok=True)
        shutil.rmtree(work, ignore_errors=True)
    open(os.path.join(prepared, "done"), "w").close()


def subprocess_run(name, prepared, extra=(), keep=None):
    work = tempfile.mkdtemp(prefix=f"bench_{name}_")
    try:
        result = subprocess.run([sys.executable, __file__, "run", name, "--workdir", work, "--prepared", prepared, "--json", *extra],
                                capture_output=True, text=True, env={**os.environ, "PYGAME_HIDE_SUPPORT_PROMPT": "1", "TF_CPP_MIN_LOG_LEVEL": "2"})
        if result.returncode != 0:
            raise RuntimeError(f"{name} failed:\n{result.stdout[-2000:]}\n{result.stderr[-4000:]}")
        line = [line for line in result.stdout.splitlines() if line.startswith("BENCH_JSON ")][-1]
        if keep is not None:
            keep(work)
        return json.loads(line[len("BENCH_JSON "):])
    finally:
        shutil.rmtree(work, ignore_errors=True)


def prepared_dir():
    return os.path.join(tempfile.gettempdir(), "priene_bench_checkpoints")


# ── output hashes (Stage 2: every change must leave them all unchanged) ──

def predictions_hash(path, n_features, rows=64):
    """A checkpoint's Q values on a fixed set of observations (float32 bytes)"""
    import numpy as np
    import keras
    from src.agent.dqn.n_network import NNetwork
    network = keras.models.load_model(path, compile=False, custom_objects={"NNetwork": NNetwork})
    observations = np.random.default_rng(0).uniform(0, 30, size=(rows, n_features)).astype("float32")
    return hashlib.sha256(np.asarray(network(observations)).astype("float32").tobytes()).hexdigest()


def hash_outputs(work):
    """Every file the run wrote under data/ (CSVs and JSONs byte for byte, checkpoints by config and weights, as
    tests/reference_outputs.py hashes them, replay buffers and optimiser states by their arrays), and each checkpoint's predictions"""
    sys.path.insert(0, os.path.join(REPO, "tests"))
    from reference_outputs import hash_file
    import numpy as np
    hashes = {}
    for root, _, files in os.walk(os.path.join(work, "data")):
        for file_name in sorted(files):
            full = os.path.join(root, file_name)
            relative = os.path.relpath(full, work)
            if file_name.endswith(".npz"):
                digest = hashlib.sha256()
                with np.load(full) as saved:
                    for key in sorted(saved.files):
                        digest.update(key.encode() + saved[key].tobytes() + str(saved[key].dtype).encode())
                hashes[relative] = digest.hexdigest()
            elif file_name == "training_state.json" or file_name.endswith(".json") or file_name.endswith(".csv") or file_name.endswith(".keras"):
                hashes[relative] = hash_file(full)
            if file_name.endswith(".keras"):
                import keras
                from src.agent.dqn.n_network import NNetwork
                n_features = keras.models.load_model(full, compile=False, custom_objects={"NNetwork": NNetwork}).n_features[0]
                hashes[relative + "#predictions"] = predictions_hash(full, n_features)
    return dict(sorted(hashes.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["list", "run", "all", "hash", "compare"])
    parser.add_argument("name", nargs="?")
    parser.add_argument("name2", nargs="?")
    parser.add_argument("--workdir")
    parser.add_argument("--prepared")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--profile")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--no-norms", action="store_true")
    parser.add_argument("--no-parts", action="store_true")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--only", help="only configurations whose name contains this")
    parser.add_argument("--label", default="latest")
    parser.add_argument("--out")
    parser.add_argument("--repeat", type=int, default=1, help="all: run each configuration this many times (the fastest is kept)")
    parser.add_argument("--prepare-scenario", help=argparse.SUPPRESS)
    parser.add_argument("--prepare-mix", help=argparse.SUPPRESS)
    parser.add_argument("--hash-into", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command == "list":
        for config in CONFIGS:
            print(config["name"], {k: v for k, v in config.items() if k != "name"})
        return
    if args.command == "run":
        if args.prepare_scenario:
            CONFIG[args.name] = harvest(args.name, args.prepare_scenario, True, mix=args.prepare_mix, episodes=CHECKPOINT_EPISODES)
        workdir = args.workdir or tempfile.mkdtemp(prefix=f"bench_{args.name}_")
        result = run_config(args.name, workdir, args.prepared or prepared_dir(), args.profile, args.threads, args.no_write, args.no_norms,
                            not args.no_parts, args.episodes)
        if args.hash_into:
            with open(args.hash_into, "w") as file:
                json.dump(hash_outputs(workdir), file, indent=1)
        print("BENCH_JSON " + json.dumps(result))
        if not args.json:
            print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=1))
        return
    if args.command == "compare":
        with open(args.name) as file:
            current = json.load(file)
        with open(args.name2 or REFERENCE_HASHES) as file:
            reference = json.load(file)
        differences = []
        for config in sorted(set(current) | set(reference)):
            a, b = current.get(config, {}), reference.get(config, {})
            for path in sorted(set(a) | set(b)):
                if a.get(path) != b.get(path):
                    differences.append(f"{config}: {path} {'missing' if path not in a else 'new' if path not in b else 'differs'}")
        print("\n".join(differences) if differences else f"All {sum(len(h) for h in current.values())} hashes match")
        sys.exit(1 if differences else 0)
    prepared = prepared_dir()
    sys.path.insert(0, REPO)
    prepare_checkpoints(prepared)
    names = [c["name"] for c in CONFIGS if not args.only or args.only in c["name"]]
    if args.command == "hash":
        hashes = {}
        for name in names:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
                into = handle.name
            subprocess_run(name, prepared, ["--no-parts", "--hash-into", into])
            with open(into) as file:
                hashes[name] = json.load(file)
            os.remove(into)
            print(f"  {name}: {len(hashes[name])} hashes", flush=True)
        out = args.out or os.path.join(RESULTS_DIR, f"hashes_{args.label}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as file:
            json.dump(hashes, file, indent=1)
        print("Hashes written to", out)
        return
    results = {}
    extra = (["--no-parts"] if args.no_parts else []) + (["--threads", str(args.threads)] if args.threads else []) + \
            (["--no-write"] if args.no_write else []) + (["--no-norms"] if args.no_norms else [])
    for name in names:
        runs = [subprocess_run(name, prepared, extra) for _ in range(args.repeat)]
        best = min(runs, key=lambda r: r["run_s"])
        best["all_run_s"] = [r["run_s"] for r in runs]
        results[name] = best
        print(f"{name:34s} {best['s_per_episode']:7.2f} s/episode {best['ms_per_step']:7.2f} ms/step  end save {best['end_of_run_save_s']:.2f}s"
              f"  peak {best['peak_rss_mb']:.0f} MB  (runs: {', '.join(f'{r:.1f}' for r in best['all_run_s'])} s)", flush=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{args.label}.json"), "w") as file:
        json.dump(results, file, indent=1)


if __name__ == "__main__":
    main()

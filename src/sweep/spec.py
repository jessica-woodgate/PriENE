"""
A sweep file (YAML) and the runs it stands for. Only training parameters (src/training_config.py) can be swept; the scenario's own
parameters, and the answers given to run.py's prompts, come from SCENARIO_ANSWERS and can't be changed by a sweep file.

    name: fruit_market_A1              # free text
    scenario: fruit_market
    step: A1                           # runs go in <output_root>/<scenario>/<step>/<configuration id>/<agent type>/seed<seed>/
    output_root: sweeps                # relative to the repository (default sweeps)
    agent_types: [baseline]
    society_mix: homogeneous           # default homogeneous
    seeds: [1, 2, 3]
    episodes: 300                      # each run's max_training_episodes
    epsilon_floor_fraction: 0.7        # epsilon reaches its minimum at this fraction of the run (null: keep the config's decay)
    base_config: null                  # training config the sweep starts from (default: the scenario's, configs/training/)
    search:
      method: random                   # random, grid or list
      configurations: 12               # random: how many (including the base configuration, if include_base)
      sampling_seed: 0
      include_base: true               # configuration c00: the base config unchanged
    parameters:                        # random and grid: each parameter's values
      learning_rate: {log_uniform: [3.0e-5, 1.0e-3]}
      gamma: [0.9, 0.95, 0.99]
    configurations:                    # list: each configuration's values
      - {sanction: 0.5}
    early_stopping: {episode: 100, window: 20, min_reward: null, max_action_share: 0.95}
    selection: {rule: step_a, last_episodes: 50}
"""
from dataclasses import dataclass, field
import itertools
import math
import os
import numpy as np
import yaml
from .. import training_config as tc

REPO = tc.REPO
SPEC_KEYS = {"name", "scenario", "step", "output_root", "agent_types", "society_mix", "seeds", "episodes", "epsilon_floor_fraction",
             "base_config", "search", "parameters", "configurations", "early_stopping", "selection"}
HARVEST_SCENARIOS = tc.HARVEST_SCENARIOS

class SweepError(ValueError):
    pass

def scenario_answers(scenario, agent_type, society_mix):
    """
    Answers to run.py train's prompts for one run: the scenario's defaults (pressing enter) for everything but the scenario, the society
    mix and the agent type, so a sweep never changes the scenario's own parameters. Writing data is always on, so the run reports.
    """
    if scenario == "fruit_market":
        #steps, players, grappling, well-being, sanction mode, spatial features: the substrate's and the design's defaults
        return ["fruit_market", "", "", "", "", "", "", society_mix, agent_type, "y"]
    if scenario in HARVEST_SCENARIOS:
        answers = [scenario, "", "", society_mix, agent_type, ""]
        if scenario in ("allotment", "capabilities"):
            answers.append("")
            if scenario == "allotment" and society_mix != "homogeneous":
                answers.append("")
        if scenario == "commons":
            answers += ["", "", ""]
        return answers + ["y"]
    raise SweepError(f"no prompt answers are known for scenario {scenario}")

#the episode report each scenario's training writes (in the run's folder), and its columns the selection rules read
SCENARIO_REPORTS = {
    "fruit_market": {"report": "data/results/current_run/fm_model_episode_reports_train_*.csv",
                     "columns": {"score": "mean_env_reward", "total_reward": "mean_reward", "fruit": "total_fruit_eaten",
                                 "trades": "total_trades", "loss": "mean_loss", "max_action_share": "max_action_share",
                                 "grapple_share": "grapple_share", "abs_sanction": "mean_abs_sanction"}},
}
HARVEST_REPORT = {"report": "data/results/current_run/model_episode_reports_train_*.csv",
                  "columns": {"score": "mean_reward", "total_reward": "mean_reward", "fruit": "mean_berries_consumed", "loss": "mean_loss"}}

def scenario_report(scenario):
    return SCENARIO_REPORTS.get(scenario, HARVEST_REPORT)

@dataclass
class Run:
    """
    One training run of a sweep: a configuration (its id and the parameter values it changes), an agent type and a seed
    """
    config_id: str
    values: dict
    agent_type: str
    seed: int
    folder: str

    @property
    def name(self):
        return f"{self.config_id}/{self.agent_type}/seed{self.seed}"

@dataclass
class Sweep:
    path: str
    name: str
    scenario: str
    step: str
    output_root: str
    agent_types: list
    society_mix: str
    seeds: list
    episodes: int
    epsilon_floor_fraction: float
    base_config: tc.TrainingConfig
    configurations: list
    early_stopping: dict = field(default_factory=dict)
    selection: dict = field(default_factory=dict)
    swept_parameters: list = field(default_factory=list)

    @property
    def folder(self):
        return os.path.join(self.output_root, self.scenario, self.step)

    def training_config(self, values):
        """
        The training config of a configuration: the base config with its values, and this sweep's training length and epsilon schedule
        """
        config = self.base_config.with_values(values).with_values({"max_training_episodes": self.episodes})
        if self.epsilon_floor_fraction is not None:
            decay = tc.compressed_epsilon_decay(config.epsilon_start, config.epsilon_min, self.episodes, self.epsilon_floor_fraction)
            config = config.with_values({"epsilon_decay": decay})
        return config

    def runs(self):
        """
        Every run, configuration by configuration, then agent type, then seed
        """
        return [Run(c["id"], c["values"], agent_type, seed, os.path.join(self.folder, c["id"], agent_type, f"seed{seed}"))
                for c in self.configurations for agent_type in self.agent_types for seed in self.seeds]

def _significant(value, digits=3):
    if value == 0:
        return 0.0
    return float(round(value, digits - 1 - int(math.floor(math.log10(abs(value))))))

def _sample(rule, rng):
    if isinstance(rule, list):
        return rule[int(rng.integers(len(rule)))]
    if isinstance(rule, dict) and set(rule) == {"log_uniform"}:
        low, high = rule["log_uniform"]
        return _significant(math.exp(rng.uniform(math.log(low), math.log(high))))
    if isinstance(rule, dict) and set(rule) == {"uniform"}:
        low, high = rule["uniform"]
        return _significant(rng.uniform(low, high))
    raise SweepError(f"unknown way of choosing values: {rule} (use a list, {{log_uniform: [low, high]}} or {{uniform: [low, high]}})")

def _expand(search, parameters, configurations):
    method = search.get("method", "list")
    include_base = search.get("include_base", True)
    chosen = [{}] if include_base else []
    if method == "list":
        chosen += [dict(c) for c in configurations or []]
    elif method == "grid":
        names = list(parameters)
        for rule in parameters.values():
            if not isinstance(rule, list):
                raise SweepError("a grid search needs a list of values for each parameter")
        chosen += [dict(zip(names, combination)) for combination in itertools.product(*parameters.values())]
    elif method == "random":
        rng = np.random.default_rng(search.get("sampling_seed", 0))
        total = search["configurations"]
        attempts = 0
        while len(chosen) < total:
            attempts += 1
            if attempts > 1000 * total:
                raise SweepError(f"the parameters' values allow fewer than {total} different configurations")
            values = {name: _sample(rule, rng) for name, rule in parameters.items()}
            if values not in chosen:
                chosen.append(values)
    else:
        raise SweepError(f"unknown search method {method} (random, grid or list)")
    return [{"id": f"c{i:02d}", "values": values} for i, values in enumerate(chosen)]

def load(path):
    with open(path) as file:
        raw = yaml.safe_load(file)
    unknown = set(raw) - SPEC_KEYS
    if unknown:
        raise SweepError(f"{path} has keys a sweep file can't have: {sorted(unknown)} (prompt answers and scenario parameters are fixed by the "
                         f"scenario; only training parameters can be swept)")
    scenario = raw["scenario"]
    scenario_answers(scenario, "baseline", raw.get("society_mix", "homogeneous"))
    parameters = raw.get("parameters") or {}
    configurations = raw.get("configurations") or []
    tc.check_parameter_names(parameters)
    for values in configurations:
        tc.check_parameter_names(values)
    for reserved in ("max_training_episodes", "epsilon_decay"):
        if reserved in parameters or any(reserved in c for c in configurations):
            raise SweepError(f"{reserved} is set from the sweep's episodes and epsilon_floor_fraction, not swept directly")
    base_path = raw.get("base_config")
    base = tc.load(base_path if os.path.isabs(base_path or "") else os.path.join(REPO, base_path)) if base_path else tc.for_scenario(scenario)
    output_root = raw.get("output_root", "sweeps")
    expanded = _expand(raw.get("search", {"method": "list"}), parameters, configurations)
    swept = sorted(set(parameters) | {name for c in configurations for name in c})
    return Sweep(path=path, name=raw.get("name", os.path.basename(path)), scenario=scenario, step=str(raw["step"]),
                 output_root=output_root if os.path.isabs(output_root) else os.path.join(REPO, output_root),
                 agent_types=list(raw["agent_types"]), society_mix=raw.get("society_mix", "homogeneous"), seeds=list(raw["seeds"]),
                 episodes=int(raw["episodes"]), epsilon_floor_fraction=raw.get("epsilon_floor_fraction", 0.7), base_config=base,
                 configurations=expanded, early_stopping=raw.get("early_stopping") or {}, selection=raw.get("selection") or {},
                 swept_parameters=swept)

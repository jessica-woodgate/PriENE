"""
Training parameters (the DQN's, the epsilon schedule, the training length and the ethics sanction), read from a file so they can be
changed without changing code: configs/training/harvest.yaml (every harvest scenario) and configs/training/fruit_market.yaml hold the
values each scenario trains with, and run.py's --training-config swaps in another file (e.g. one written by the parameter sweep,
src/sweep/). Only these parameters can be set this way; the scenario's own (substrate) parameters are refused.
"""
from dataclasses import dataclass, asdict, fields
import json
import math
import os
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO, "configs", "training")
HARVEST_SCENARIOS = ["basic", "colours", "capabilities", "allotment", "commons"]

#parameters of a scenario itself, which a training config must never change (they are chosen at run.py's prompts, or fixed by the
#substrate): asking for one is an error that says so
SCENARIO_PARAMETERS = {"scenario", "num_agents", "num_players", "players", "max_days", "steps", "episode_length", "grappling", "well_being",
                       "sanction_mode", "observation", "observation_mode", "roles", "map", "hunger", "hunger_delay", "stamina",
                       "regrowth", "regrowth_time", "trade_radius", "taste", "seasons", "season_length", "num_allocations",
                       "allotment_regrowth_delays", "carrying_capacity", "growth_rate", "initial_stock", "society_mix", "seed"}

class TrainingConfigError(ValueError):
    pass

@dataclass(frozen=True)
class TrainingConfig:
    """
    learning_rate, gamma, batch_size, hidden_units (width of each of the 2 hidden layers), huber_delta -- the DQN
    min_experiences, max_experiences -- replay buffer: learning starts once it holds min_experiences; it holds at most max_experiences
    target_update_interval -- learning calls between copying the Q network's weights to the target network
    epsilon_start, epsilon_min, epsilon_decay -- exploration: epsilon_start * exp(-epsilon_decay * (episode - 1)), never below epsilon_min
    max_training_episodes -- training stops after this many episodes; None trains until epsilon reaches epsilon_min
    sanction -- the ethics module's self-directed sanction (+ or - this much)
    """
    learning_rate: float = 0.0001
    gamma: float = 0.95
    batch_size: int = 64
    hidden_units: int = 128
    huber_delta: float = 1.0
    min_experiences: int = 100
    max_experiences: int = 100000
    target_update_interval: int = 50
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.001
    max_training_episodes: int = None
    sanction: float = 0.4

    def to_dict(self):
        return asdict(self)

    def with_values(self, values):
        """
        This config with values (a dict of parameter names to values) changed; refuses scenario parameters and unknown names
        """
        check_parameter_names(values)
        return TrainingConfig(**{**self.to_dict(), **values})

PARAMETER_NAMES = [f.name for f in fields(TrainingConfig)]

def check_parameter_names(values):
    for name in values:
        if name in SCENARIO_PARAMETERS:
            raise TrainingConfigError(f"{name} is a parameter of the scenario, not of training, so a training config can't change it")
        if name not in PARAMETER_NAMES:
            raise TrainingConfigError(f"{name} is not a training parameter; the training parameters are {PARAMETER_NAMES}")

def load(path):
    """
    The training config in path (YAML or JSON): every training parameter must be given, and nothing else
    """
    with open(path) as file:
        values = yaml.safe_load(file) if path.endswith((".yaml", ".yml")) else json.load(file)
    if not isinstance(values, dict):
        raise TrainingConfigError(f"{path} doesn't hold a mapping of parameter names to values")
    check_parameter_names(values)
    missing = [name for name in PARAMETER_NAMES if name not in values]
    if missing:
        raise TrainingConfigError(f"{path} doesn't give {missing}")
    return TrainingConfig(**values)

def save(config, path):
    with open(path, "w") as file:
        yaml.safe_dump(config.to_dict(), file, sort_keys=False)

def scenario_config_path(scenario):
    """
    The training config file a scenario uses unless another is given
    """
    if scenario in HARVEST_SCENARIOS:
        return os.path.join(CONFIG_DIR, "harvest.yaml")
    return os.path.join(CONFIG_DIR, f"{scenario}.yaml")

def for_scenario(scenario):
    return load(scenario_config_path(scenario))

def compressed_epsilon_decay(epsilon_start, epsilon_min, episodes, floor_fraction=0.7):
    """
    The epsilon_decay at which epsilon reaches epsilon_min after floor_fraction of episodes (docs/fruit_market_sweep.md §1.3)
    """
    return math.log(epsilon_start / epsilon_min) / (floor_fraction * episodes)

DEFAULT = TrainingConfig()

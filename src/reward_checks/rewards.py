"""
Candidate reward functions (configs/rewards/*.yaml): the format of the rewards in configs/training/harvest.yaml (rewards, ethics_rewards,
sanction), plus two optional entries:
    alive -- reward for each day an agent is still alive at the end of (default 0)
    shaping -- add potential-based shaping on the agent's own days left to live (default false): gamma * phi(s') - phi(s), where s and
        s' are the states of the DQN's own transition (HarvestAgent.perform_transition): phi(s) is its days left to live before it acts,
        phi(s') straight after its action (before the day's decay), and phi(s') = 0 when the transition ends the agent (it died this
        step), as for a terminal state. (Between one step's s' and the next step's s come the day's decay and any berries other agents
        throw to it, which the shaping doesn't see, as the DQN's transitions don't)

The reward a logged step (src/reward_checks/simulate.py) would have got, for an agent of a given type, is as during a run
(HarvestAgent): its outcome's reward (ethics agents: ethics_rewards in place of rewards where given), plus death if it died this step,
plus survive if it reached the episode's last day alive (not if it died that day), plus alive and shaping if
given, plus, for an ethics agent, the sanction its principle or aggregation gives the well-being before and after its action.
"""
import os
from dataclasses import dataclass, field
import numpy as np
import yaml
from src.training_config import HARVEST_REWARDS

PRINCIPLES = ["utilitarian", "maximin", "egalitarian"]
AGGREGATIONS = ["average", "majoritarian", "optimist", "veto"]
AGENT_TYPES = ["baseline"] + PRINCIPLES + AGGREGATIONS
FAILED_OUTCOMES = ["no_berries", "no_benefactor", "insufficient_health"]


class RewardFileError(ValueError):
    pass


@dataclass
class RewardFunction:
    name: str
    rewards: dict
    ethics_rewards: dict = field(default_factory=dict)
    sanction: float = 0.0
    alive: float = 0.0
    shaping: bool = False

    def table(self, agent_type):
        """Each outcome's reward (and death and survive) for an agent of agent_type"""
        return dict(self.rewards) if agent_type == "baseline" else {**self.rewards, **self.ethics_rewards}


def load(path):
    with open(path) as file:
        values = yaml.safe_load(file) or {}
    unknown = set(values) - {"rewards", "ethics_rewards", "sanction", "alive", "shaping"}
    if unknown:
        raise RewardFileError(f"{path}: unknown entries {sorted(unknown)}")
    rewards = values.get("rewards") or {}
    missing = [key for key in HARVEST_REWARDS if key not in rewards]
    if missing:
        raise RewardFileError(f"{path}: rewards doesn't give {missing}")
    for name in ["rewards", "ethics_rewards"]:
        extra = [key for key in (values.get(name) or {}) if key not in HARVEST_REWARDS]
        if extra:
            raise RewardFileError(f"{path}: {name} gives {extra}, which aren't outcomes; they are {list(HARVEST_REWARDS)}")
    return RewardFunction(os.path.splitext(os.path.basename(path))[0], dict(rewards), dict(values.get("ethics_rewards") or {}),
                          float(values.get("sanction", 0.0)), float(values.get("alive", 0.0)), bool(values.get("shaping", False)))


def step_rewards(log, reward_function, agent_type, gamma, unit_sanction=None):
    """
    Each logged step's reward for an agent of agent_type under reward_function
    unit_sanction -- for an ethics agent, the sanction its principle or aggregation gives each step with a sanction of 1 (sanctions are
        that, scaled by the sanction: see evaluate.unit_sanctions)
    """
    table = reward_function.table(agent_type)
    died = log["died"].to_numpy(dtype=bool)
    rewards = log["outcome"].map(table).to_numpy(dtype=float)
    survived = log["last_day"].to_numpy(dtype=bool) & ~died
    rewards = rewards + np.where(died, table["death"], 0.0) + np.where(survived, table["survive"], 0.0)
    if reward_function.alive:
        rewards = rewards + np.where(died, 0.0, reward_function.alive)
    if reward_function.shaping:
        before = log["days_left_before"].to_numpy(dtype=float)
        after = np.where(died, 0.0, log["days_left_acted"].to_numpy(dtype=float))
        rewards = rewards + gamma * after - before
    if agent_type != "baseline":
        if unit_sanction is None:
            raise ValueError(f"an ethics agent's rewards ({agent_type}) need its sanctions")
        rewards = rewards + reward_function.sanction * np.asarray(unit_sanction, dtype=float)
    return rewards

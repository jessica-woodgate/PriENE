"""
Runs scripted policies (src/agent/modules/scripted_decision_module.py) and logs every agent's every step (HarvestAgent._log_outcome), for
the reward check (src/reward_checks/evaluate.py):

    python -m src.reward_checks.simulate                        # basic, 200 episodes of 200 days, 4 agents, every set-up
    python -m src.reward_checks.simulate --scenarios basic colours --workers 4

Set-ups, each a run of its own (a process of its own), seeded from the validation seed (default 2000), so every episode's environment is
reseeded from [2000, episode, stream] and is the same in every run:
    homogeneous_<policy> -- every agent acts by the policy (random, passive, eater, hoarder, sharer, altruist)
    focal_<policy> -- agent 0 acts by the policy (sharer, altruist, hoarder, random) and the others are eaters
Agents are baseline agents: an agent's type changes only its rewards, which the evaluation works out afresh for every type.

Logs: <out>/<scenario>/<set-up>.csv.gz, one row per agent per step (the well-being vectors as JSON lists).
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VALIDATION_BASE_SEED = 2000
HOMOGENEOUS_POLICIES = ["random", "passive", "eater", "hoarder", "sharer", "altruist"]
FOCAL_POLICIES = ["sharer", "altruist", "hoarder", "random"]
SCENARIOS = ["basic", "colours"]


def setups():
    """Every set-up: its name, and its policy for each agent (agent 0, then the rest)"""
    return ([(f"homogeneous_{p}", "homogeneous", p, p, p) for p in HOMOGENEOUS_POLICIES]
            + [(f"focal_{p}", "focal", p, p, "eater") for p in FOCAL_POLICIES])


def build_model(scenario, policies, episodes, days, num_agents, seed, agent_type="baseline", folder=None):
    """
    A test-mode model of scenario whose agents act by policies ({agent id: policy}), logging their steps (model.outcome_log); agent_type
    is every agent's type (it decides only the rewards they get during the run)
    """
    from src.scenarios.basic_harvest import BasicHarvest
    from src.scenarios.colours_harvest import ColoursHarvest
    base = {"basic": BasicHarvest, "colours": ColoursHarvest}[scenario]
    scripted = type(f"Scripted{base.__name__}", (base,), {"scripted_policies": dict(policies)})
    folder = folder or os.getcwd()
    model = scripted("homogeneous", num_agents, num_agents * 3, agent_type, num_agents * 2, num_agents * 2, episodes, days, False,
                     os.path.join(folder, "no_checkpoints") + "/", False, False, f"scripted_{scenario}", seed)
    model.outcome_log = []
    return model


def simulate(scenario, policies, episodes=200, days=200, num_agents=4, seed=VALIDATION_BASE_SEED, agent_type="baseline"):
    """Run the scripted policies for episodes episodes; the log, one row per agent per step"""
    model = build_model(scenario, policies, episodes, days, num_agents, seed, agent_type)
    while model.episode <= episodes:
        model.step()
    return pd.DataFrame(model.outcome_log)


def run_setup(job):
    scenario, name, kind, first, rest, episodes, days, num_agents, seed, out = job
    start = time.time()
    policies = {i: first if i == 0 else rest for i in range(num_agents)}
    log = simulate(scenario, policies, episodes, days, num_agents, seed)
    log.insert(0, "setup", kind)
    log.insert(0, "run", name)
    log.insert(0, "scenario", scenario)
    for column in ["well_being_before", "well_being_after"]:
        log[column] = log[column].map(lambda values: "[" + ",".join(repr(float(v)) for v in values) + "]")
    path = os.path.join(out, scenario, f"{name}.csv.gz")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writing = path.replace(".csv.gz", ".writing.csv.gz")
    log.to_csv(writing, index=False)
    os.replace(writing, path)
    return f"{scenario}/{name}: {len(log)} steps in {time.time() - start:.0f}s"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", nargs="+", default=["basic"], choices=SCENARIOS)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--days", type=int, default=200)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--seed", type=int, default=VALIDATION_BASE_SEED, help="The validation seed every episode's environment is seeded from")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=os.path.join(REPO, "data", "reward_checks", "logs"))
    args = parser.parse_args()
    jobs = [(scenario, name, kind, first, rest, args.episodes, args.days, args.agents, args.seed, args.out)
            for scenario in args.scenarios for name, kind, _, first, rest in setups()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for message in pool.map(run_setup, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()

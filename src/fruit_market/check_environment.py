"""
The environment check (docs/fruit_market_sweep.md §2): scripted policies in the original substrate, through the adapter.
    python -m src.fruit_market.check_environment [--episodes 5] [--seed 1] [--out sweeps/fruit_market/environment_check]
Writes environment_check.csv (one row per policy and episode) and environment_check.md (means and sds per policy) in --out, and prints
whether the trader did better than autarky (if not, check the adapter before tuning anything).
"""
import argparse
import os
import time
import pandas as pd
from .scripted import POLICIES, run_policy

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLUMNS = {"mean_env_reward": "substrate reward per agent", "total_fruit_eaten": "fruit eaten (all 16)", "preferred_eat_rate": "share of preferred fruit eaten",
           "mean_hunger": "mean HUNGER", "steps_hungry": "steps hungry (all 16)", "total_trades": "trades", "mean_well_being": "mean well-being (need)",
           "min_well_being": "min well-being (need)", "mean_stamina": "mean stamina", "max_action_share": "most common action's share"}

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run scripted policies in Fruit Market")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default=os.path.join(REPO, "sweeps", "fruit_market", "environment_check"))
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    os.chdir(args.out)
    frames = []
    for policy in POLICIES:
        start = time.time()
        report = run_policy(policy, args.episodes, args.seed)
        report.insert(0, "policy", policy)
        report["seconds_per_episode"] = (time.time() - start) / args.episodes
        frames.append(report)
        print(f"{policy}: {report['mean_env_reward'].mean():.2f} substrate reward per agent per episode", flush=True)
    results = pd.concat(frames, ignore_index=True)
    results.to_csv("environment_check.csv", index=False)
    means = results.groupby("policy", sort=False)[list(COLUMNS)].agg(["mean", "std"])
    lines = ["# Fruit Market environment check", "",f"{args.episodes} episodes of 1000 steps per policy, 16 players, seed {args.seed}. "
             "Mean (sd) over episodes; per-agent measures are means over the 16 players.", "",
             "| Measure | " + " | ".join(POLICIES) + " |", "|---|" + "---|" * len(POLICIES)]
    for column, label in COLUMNS.items():
        lines.append(f"| {label} | " + " | ".join(f"{means.loc[p, (column, 'mean')]:.3g} ({means.loc[p, (column, 'std')]:.2g})" for p in POLICIES) + " |")
    trader, autarky = means.loc["trader", ("mean_env_reward", "mean")], means.loc["autarky", ("mean_env_reward", "mean")]
    verdict = ("The trader earned more than autarky, as expected." if trader > autarky else
               "**The trader did not earn more than autarky: check the adapter before tuning anything.**")
    lines += ["", verdict, "", f"Random policy's mean substrate reward per agent per episode (the sweeps' early-stopping threshold): "
              f"{means.loc['random', ('mean_env_reward', 'mean')]:.3f}"]
    with open("environment_check.md", "w") as file:
        file.write("\n".join(lines) + "\n")
    print(verdict)

if __name__ == "__main__":
    main()

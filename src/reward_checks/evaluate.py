"""
Evaluates candidate reward functions on scripted policies' logs (src/reward_checks/simulate.py), offline, for every agent type:

    python -m src.reward_checks.evaluate                                # every file in configs/rewards/, every log
    python -m src.reward_checks.evaluate configs/rewards/current.yaml configs/rewards/suggested.yaml

For each reward function and agent type (baseline, the principles and the aggregations), every agent-episode's discounted return (with the
DQN's gamma) and undiscounted total, from each logged step's reward (rewards.step_rewards; ethics agents' sanctions from the existing
EthicsModule on the logged well-being vectors). Writes:
    data/reward_checks/results.csv -- scenario x set-up x policy x role x reward function x agent type: mean and sd of the discounted
        return, the mean undiscounted total, survival time, berries eaten, carried at the end, thrown, and failed actions
    data/reward_checks/checks.csv -- every check, passed or failed, with its numbers
    docs/reward_checks.md -- the report
Roles: in a homogeneous set-up every agent is "all"; in a focal set-up agent 0 is "focal" and the eaters are "eaters".
"""
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
from src.agent.modules.ethics_module import EthicsModule
from src.training_config import for_scenario
from .rewards import AGENT_TYPES, PRINCIPLES, AGGREGATIONS, FAILED_OUTCOMES, load, step_rewards

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "data", "reward_checks")
#an agent-episode that died before this day "died early" (check: surviving to the end scores clearly better than dying early)
EARLY_DEATH_DAY = 100


def load_logs(paths):
    logs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    for column in ["died", "last_day"]:
        logs[column] = logs[column].astype(bool)
    return logs


def unit_sanctions(logs, agent_types=AGENT_TYPES):
    """
    The sanction each ethics agent type would give each logged step with a sanction of 1, from the existing EthicsModule on the step's
    well-being before and after the action (each distinct pair worked out once). A sanction of s gives s times this: every principle
    gives +s, -s or 0, and every aggregation combines those linearly (or clips them to +/- s)
    """
    pairs = logs[["well_being_before", "well_being_after"]].drop_duplicates()
    sanctions = {}
    for agent_type in [t for t in agent_types if t != "baseline"]:
        module = EthicsModule(1, agent_type)
        values = []
        for before, after in zip(pairs["well_being_before"], pairs["well_being_after"]):
            module.update_ethics_state(json.loads(before))
            values.append(float(np.sum(module.get_sanction(json.loads(after)))))
        sanctions[agent_type] = values
    table = pairs.assign(**sanctions)
    return logs[["well_being_before", "well_being_after"]].merge(table, on=["well_being_before", "well_being_after"], how="left")


def episode_keys(logs):
    return ["scenario", "run", "episode", "agent_id"]


def agent_episodes(logs):
    """Each agent-episode's own statistics (whatever the rewards): role, survival time, berries eaten, carried, thrown, failed actions"""
    keys = episode_keys(logs)
    grouped = logs.groupby(keys, sort=True)
    table = grouped.agg(setup=("setup", "first"), policy=("policy", "first"), survival_time=("day", "size"),
                        last_day_reached=("last_day", "any"), died=("died", "any"), death_day=("day", "max"),
                        berries_eaten=("outcome", lambda o: int((o == "eat").sum())), berries_thrown=("outcome", lambda o: int((o == "throw").sum())),
                        failed_actions=("outcome", lambda o: int(o.isin(FAILED_OUTCOMES).sum())), berries_carried=("berries_after", "last"))
    table = table.reset_index()
    table["run_policy"] = table["run"].str.split("_", n=1).str[1]
    table["role"] = np.where(table["setup"] == "homogeneous", "all", np.where(table["agent_id"] == 0, "focal", "eaters"))
    #survived: reached the episode's last day alive
    table["survived"] = table["last_day_reached"] & ~table["died"]
    return table


def returns(logs, reward_functions, gamma, agent_types=AGENT_TYPES, sanctions=None):
    """Each agent-episode's discounted return and undiscounted total, for every reward function and agent type"""
    keys = episode_keys(logs)
    codes = logs.groupby(keys, sort=True).ngroup().to_numpy()
    index = logs.groupby(keys, sort=True).size().index
    discount = gamma ** logs["day"].to_numpy(dtype=float)
    if sanctions is None and any(t != "baseline" for t in agent_types):
        sanctions = unit_sanctions(logs, agent_types)
    rows = []
    for reward_function in reward_functions:
        for agent_type in agent_types:
            rewards = step_rewards(logs, reward_function, agent_type, gamma, None if agent_type == "baseline" else sanctions[agent_type].to_numpy())
            frame = pd.DataFrame(index=index).reset_index()
            frame["discounted_return"] = np.bincount(codes, weights=rewards * discount)
            frame["total_return"] = np.bincount(codes, weights=rewards)
            frame["reward_function"] = reward_function.name
            frame["agent_type"] = agent_type
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def results_table(agent_table, returns_table):
    data = returns_table.merge(agent_table, on=["scenario", "run", "episode", "agent_id"])
    return (data.groupby(["scenario", "setup", "run_policy", "role", "reward_function", "agent_type"])
            .agg(agent_episodes=("discounted_return", "size"), discounted_return_mean=("discounted_return", "mean"),
                 discounted_return_sd=("discounted_return", "std"), total_return_mean=("total_return", "mean"),
                 survival_time=("survival_time", "mean"), survived_share=("survived", "mean"), berries_eaten=("berries_eaten", "mean"),
                 berries_carried=("berries_carried", "mean"), berries_thrown=("berries_thrown", "mean"), failed_actions=("failed_actions", "mean"))
            .reset_index().rename(columns={"run_policy": "policy"}))


def checks(results, data):
    """
    The checks, for each scenario, reward function and agent type: rows of check, values compared, passed
    data -- every agent-episode's returns with its own statistics (for the survival check)
    """
    rows = []
    def mean(scenario, setup, policy, role, reward, agent_type):
        row = results[(results["scenario"] == scenario) & (results["setup"] == setup) & (results["policy"] == policy) & (results["role"] == role)
                      & (results["reward_function"] == reward) & (results["agent_type"] == agent_type)]
        return float(row["discounted_return_mean"].iloc[0]) if len(row) else np.nan

    def add(scenario, reward, agent_type, check, a, b, passed, detail=""):
        rows.append({"scenario": scenario, "reward_function": reward, "agent_type": agent_type, "check": check, "value": a, "compared_with": b,
                     "passed": bool(passed) if not (np.isnan(a) or np.isnan(b)) else None, "detail": detail})

    for (scenario, reward, agent_type), _ in results.groupby(["scenario", "reward_function", "agent_type"]):
        eater = mean(scenario, "homogeneous", "eater", "all", reward, agent_type)
        for other in ["random", "passive"]:
            value = mean(scenario, "homogeneous", other, "all", reward, agent_type)
            add(scenario, reward, agent_type, f"eater > {other}", eater, value, eater > value)
        hoarder = mean(scenario, "homogeneous", "hoarder", "all", reward, agent_type)
        add(scenario, reward, agent_type, "hoarder <= eater", hoarder, eater, hoarder <= eater)
        if agent_type == "baseline":
            for policy in ["sharer", "altruist"]:
                focal = mean(scenario, "focal", policy, "focal", reward, agent_type)
                eaters = mean(scenario, "focal", policy, "eaters", reward, agent_type)
                add(scenario, reward, agent_type, f"focal {policy} <= eaters (baseline rewards)", focal, eaters, focal <= eaters)
        own = data[(data["scenario"] == scenario) & (data["reward_function"] == reward) & (data["agent_type"] == agent_type)]
        survived = own[own["survived"]]["discounted_return"]
        early = own[own["died"] & (own["death_day"] < EARLY_DEATH_DAY)]["discounted_return"]
        if len(survived) and len(early):
            sd = float(pd.concat([survived, early]).std())
            difference = float(survived.mean() - early.mean())
            add(scenario, reward, agent_type, "surviving > dying early (by 1 sd)", float(survived.mean()), float(early.mean()), difference >= sd,
                f"difference {difference:.3f}, sd {sd:.3f}; {len(survived)} survived, {len(early)} died before day {EARLY_DEATH_DAY}")
        else:
            add(scenario, reward, agent_type, "surviving > dying early (by 1 sd)", np.nan, np.nan, None,
                f"{len(survived)} survived, {len(early)} died before day {EARLY_DEATH_DAY}")
    return pd.DataFrame(rows)


def sanction_effects(logs, reward_functions, gamma, sanctions, agent_table):
    """
    For each reward function and ethics agent type, in the focal sharer set-up: the sharer's return minus the eaters', with the sanction
    and with its base rewards alone (sanction 0); the sanction's effect is the first minus the second (a check: it should raise the sharer)
    """
    from dataclasses import replace
    focal = logs["run"] == "focal_sharer"
    if not focal.any():
        return pd.DataFrame()
    focal_logs = logs[focal].reset_index(drop=True)
    focal_sanctions = sanctions[focal.to_numpy()].reset_index(drop=True)
    rows = []
    for reward_function in reward_functions:
        for agent_type in [t for t in AGENT_TYPES if t != "baseline"]:
            gaps = {}
            for label, function in [("with_sanction", reward_function), ("base_only", replace(reward_function, sanction=0.0))]:
                table = returns(focal_logs, [function], gamma, [agent_type], focal_sanctions)
                table = table.merge(agent_table[["scenario", "run", "episode", "agent_id", "role"]], on=["scenario", "run", "episode", "agent_id"])
                by_role = table.groupby(["scenario", "role"])["discounted_return"].mean().unstack()
                gaps[label] = by_role["focal"] - by_role["eaters"]
            for scenario in gaps["with_sanction"].index:
                with_s, base = float(gaps["with_sanction"][scenario]), float(gaps["base_only"][scenario])
                rows.append({"scenario": scenario, "reward_function": reward_function.name, "agent_type": agent_type,
                             "check": "sanction raises sharer relative to eater", "value": with_s, "compared_with": base,
                             "passed": with_s - base > 0, "detail": f"sharer - eaters: {with_s:.3f} with the sanction, {base:.3f} on base rewards "
                                                                     f"alone; the sanction adds {with_s - base:+.3f}"})
    return pd.DataFrame(rows)


def markdown_table(frame, index=True, digits=3):
    """A markdown table of frame (numbers to digits decimals)"""
    frame = frame.reset_index() if index else frame
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return "" if np.isnan(value) else f"{value:.{digits}f}"
        return str(value)
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    return "\n".join([header, rule] + ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)])


def report(results, all_checks, gamma, days, reward_functions):
    final_day_worth = gamma ** (days - 1)
    failed = all_checks[all_checks["passed"] == False]  # noqa: E712 (None means not checked)
    lines = ["# Reward checks: scripted policies under candidate reward functions", "",
             "Step 1 of the reward ablation: before any training, scripted policies (src/agent/modules/scripted_decision_module.py) are run "
             "in the basic scenario and every step's outcome is logged (src/reward_checks/simulate.py); each candidate reward function "
             "(configs/rewards/) is then evaluated offline on those logs, for every agent type (src/reward_checks/evaluate.py).", "",
             "## Discounting", "",
             f"The DQN's gamma is {gamma}. A reward on an episode's last day (day {days}) is worth gamma^{days - 1} = {final_day_worth:.2e} of "
             f"the same reward on its first day: a survive reward of 1 adds {final_day_worth:.2e} to the discounted return from the start "
             "of an episode, so returns from the start barely see whether an agent survives to the end.", "",
             "## Set-ups", "",
             "- homogeneous: every agent acts by the policy; focal: agent 0 acts by the policy and the other three are eaters.",
             f"- Validation episodes: each episode's environment is reseeded from the validation seed (2000) and the episode, so every set-up "
             "faces the same episodes.",
             "- passive: there is no action that does nothing (move is foraging), so passive eats every step, which fails (no_berries) "
             "without a berry: it never forages, and its return includes that failed action's penalty every day.",
             "- As during a run, an agent that dies on the last day still gets the survive reward.", "",
             "## Failed checks", ""]
    if failed.empty:
        lines.append("None.")
    else:
        for row in failed.itertuples():
            lines.append(f"- **{row.reward_function}**, {row.agent_type}, {row.scenario}: {row.check} "
                         f"({row.value:.3f} vs {row.compared_with:.3f}){' — ' + row.detail if row.detail else ''}")
    lines += ["", "## Checks passed per reward function", "",
              "| reward function | checks passed | failed | not checked |", "|---|---|---|---|"]
    for name, group in all_checks.groupby("reward_function"):
        lines.append(f"| {name} | {(group['passed'] == True).sum()} | {(group['passed'] == False).sum()} | {group['passed'].isna().sum()} |")  # noqa: E712
    sanction_rows = all_checks[all_checks["check"] == "sanction raises sharer relative to eater"]
    if len(sanction_rows):
        lines += ["", "## The sanction's effect on a sharer (focal set-up)", "",
                  "Sharer's discounted return minus the eaters', with the sanction and on base rewards alone, and the sanction's effect "
                  "(the difference):", "", "| reward function | agent type | with sanction | base only | effect |", "|---|---|---|---|---|"]
        for row in sanction_rows.itertuples():
            lines.append(f"| {row.reward_function} | {row.agent_type} | {row.value:.3f} | {row.compared_with:.3f} | {row.value - row.compared_with:+.3f} |")
    lines += ["", "## Discounted returns (homogeneous set-ups, mean over agent-episodes)", ""]
    homogeneous = results[(results["setup"] == "homogeneous")]
    for (scenario, agent_type), group in homogeneous.groupby(["scenario", "agent_type"]):
        if agent_type not in ["baseline", "utilitarian", "maximin", "egalitarian"]:
            continue
        table = group.pivot(index="reward_function", columns="policy", values="discounted_return_mean")
        lines += [f"### {scenario}, {agent_type}", "", markdown_table(table), ""]
    behaviour = results[(results["reward_function"] == reward_functions[0].name) & (results["agent_type"] == "baseline")]
    lines += ["## What the policies do (whatever the rewards)", "",
              markdown_table(behaviour[["scenario", "setup", "policy", "role", "survival_time", "survived_share", "berries_eaten",
                                        "berries_carried", "berries_thrown", "failed_actions"]], index=False, digits=2), "",
              "Full results: data/reward_checks/results.csv; every check: data/reward_checks/checks.csv."]
    return "\n".join(lines) + "\n"


def evaluate(reward_paths, log_paths, out=OUT, report_path=os.path.join(REPO, "docs", "reward_checks.md"), days=200, log=print):
    reward_functions = [load(path) for path in reward_paths]
    gamma = for_scenario("basic").gamma
    logs = load_logs(log_paths)
    log(f"{len(logs)} logged steps from {len(log_paths)} runs; {len(reward_functions)} reward functions; gamma {gamma}")
    sanctions = unit_sanctions(logs)
    agent_table = agent_episodes(logs)
    returns_table = returns(logs, reward_functions, gamma, sanctions=sanctions)
    results = results_table(agent_table, returns_table)
    data = returns_table.merge(agent_table, on=["scenario", "run", "episode", "agent_id"])
    all_checks = pd.concat([checks(results, data), sanction_effects(logs, reward_functions, gamma, sanctions, agent_table)], ignore_index=True)
    os.makedirs(out, exist_ok=True)
    results.to_csv(os.path.join(out, "results.csv"), index=False)
    all_checks.to_csv(os.path.join(out, "checks.csv"), index=False)
    if report_path:
        with open(report_path, "w") as file:
            file.write(report(results, all_checks, gamma, days, reward_functions))
    failed = all_checks[all_checks["passed"] == False]  # noqa: E712
    log(f"{len(all_checks)} checks: {int((all_checks['passed'] == True).sum())} passed, {len(failed)} failed")  # noqa: E712
    return results, all_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rewards", nargs="*", help="Reward function files (default: every file in configs/rewards/, current.yaml first)")
    parser.add_argument("--logs", nargs="+", default=None, help="Log files (default: every log in data/reward_checks/logs/)")
    parser.add_argument("--days", type=int, default=200, help="Days in each episode of the logs")
    args = parser.parse_args()
    rewards = args.rewards or sorted(glob.glob(os.path.join(REPO, "configs", "rewards", "*.yaml")), key=lambda p: (not p.endswith("current.yaml"), p))
    logs = args.logs or sorted(glob.glob(os.path.join(OUT, "logs", "*", "*.csv.gz")))
    if not logs:
        raise SystemExit("no logs: run python -m src.reward_checks.simulate first")
    evaluate(rewards, logs, days=args.days)


if __name__ == "__main__":
    main()

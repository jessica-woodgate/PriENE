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
#the policies that give berries away, each checked as the focal agent among eaters
SHARING_POLICIES = ["sharer", "altruist", "sharer_hoarder", "altruist_eater"]


def load_logs(paths):
    logs = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    for column in ["died", "last_day"]:
        logs[column] = logs[column].astype(bool)
    return with_neutral_passive(logs)


def with_neutral_passive(logs):
    """
    The logs, with a copy of the homogeneous passive runs (run homogeneous_passive_neutral, policy passive_neutral) whose failed eats
    (no_berries) count as actions with no other effect (neutral_reward): passive can't do nothing (there is no such action), so it
    tries to eat every step and pays the failed action's penalty; this scores the same steps as pure inaction
    """
    passive = logs[(logs["setup"] == "homogeneous") & (logs["run"].str.endswith("_passive"))]
    if passive.empty:
        return logs
    neutral = passive.assign(run=passive["run"] + "_neutral", policy="passive_neutral",
                             outcome=passive["outcome"].replace("no_berries", "neutral_reward"))
    return pd.concat([logs, neutral], ignore_index=True)


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
        for other in ["random", "passive", "passive_neutral"]:
            value = mean(scenario, "homogeneous", other, "all", reward, agent_type)
            add(scenario, reward, agent_type, f"eater > {other}", eater, value, eater > value)
        hoarder = mean(scenario, "homogeneous", "hoarder", "all", reward, agent_type)
        add(scenario, reward, agent_type, "hoarder <= eater", hoarder, eater, hoarder <= eater)
        if agent_type == "baseline":
            for policy in SHARING_POLICIES:
                focal = mean(scenario, "focal", policy, "focal", reward, agent_type)
                eaters = mean(scenario, "focal", policy, "eaters", reward, agent_type)
                add(scenario, reward, agent_type, f"focal {policy} <= eaters (baseline rewards)", focal, eaters, focal <= eaters)
        #every agent-episode simulated (not passive_neutral's copies of the passive ones, which would count them twice)
        own = data[(data["scenario"] == scenario) & (data["reward_function"] == reward) & (data["agent_type"] == agent_type)
                   & (data["run_policy"] != "passive_neutral")]
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
    For each reward function, ethics agent type and sharing policy (SHARING_POLICIES, each a focal set-up): the focal agent's return minus
    the eaters', with the sanction and with its base rewards alone (sanction 0); the sanction's effect is the first minus the second (a
    check: it should raise the sharing agent), and the sanction at which it catches up with the eaters
    """
    from dataclasses import replace
    rows = []
    for policy in SHARING_POLICIES:
        focal = logs["run"] == f"focal_{policy}"
        if not focal.any():
            continue
        focal_logs = logs[focal].reset_index(drop=True)
        focal_sanctions = sanctions[focal.to_numpy()].reset_index(drop=True)
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
                    effect = with_s - base
                    #the sanction's effect is linear in it (see unit_sanctions), so the focal agent catches up with the eaters at
                    #sanction * -base / effect, when the sanction raises it and it starts behind
                    break_even = reward_function.sanction * -base / effect if reward_function.sanction > 0 and effect > 0 and base < 0 else np.nan
                    rows.append({"scenario": scenario, "reward_function": reward_function.name, "agent_type": agent_type, "policy": policy,
                                 "check": f"sanction raises {policy} relative to eater", "value": with_s, "compared_with": base,
                                 #with no sanction there is nothing to check
                                 "passed": effect > 0 if reward_function.sanction > 0 else None,
                                 "sanction": reward_function.sanction, "break_even_sanction": break_even,
                                 "detail": f"{policy} - eaters: {with_s:.3f} with the sanction, {base:.3f} on base rewards alone; the sanction "
                                           f"adds {effect:+.3f}" + (f"; it catches up with the eaters at a sanction of {break_even:.2f}"
                                                                   if not np.isnan(break_even) else "")})
    return pd.DataFrame(rows)


def with_sanctions(reward_functions, sanctions):
    """Each reward function at each of sanctions, named <name>_s<sanction> (e.g. suggested_s0.8)"""
    from dataclasses import replace
    return [replace(function, name=f"{function.name}_s{sanction:g}", sanction=float(sanction))
            for function in reward_functions for sanction in sanctions]


def sanction_sweep(all_checks, reward_functions, base_names):
    """
    For a sanction sweep: each sharing policy, base reward function and ethics agent type, the focal agent's return minus the eaters' at
    each sanction, the sanction's effect for a sanction of 1, and the sanction at which it catches up with the eaters
    """
    rows = all_checks[all_checks["check"].str.startswith("sanction raises ")].copy()
    if "policy" not in rows.columns:
        rows["policy"] = "sharer"
    rows["base_reward_function"] = rows["reward_function"].map(base_names)
    rows["effect_per_unit"] = np.where(rows["sanction"] > 0, (rows["value"] - rows["compared_with"]) / rows["sanction"].replace(0, np.nan), np.nan)
    index = ["policy", "base_reward_function", "agent_type"]
    table = rows.pivot_table(index=index, columns="sanction", values="value")
    table.columns = [f"focal - eaters at s={c:g}" for c in table.columns]
    summary = rows.groupby(index).agg(effect_per_unit=("effect_per_unit", "mean"), break_even_sanction=("break_even_sanction", "min"))
    return table.join(summary).reset_index()


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


def sweep_report(all_checks, sweep, gamma, days, sanctions, base_names):
    """The report of a sanction sweep: the sharer against the eaters at every sanction, and the failed checks by sanction"""
    checks_with = all_checks.assign(base_reward_function=all_checks["reward_function"].map(base_names),
                                    sanction=all_checks["reward_function"].map({name: float(name.rsplit("_s", 1)[1]) for name in base_names}))
    failed = checks_with[checks_with["passed"] == False]  # noqa: E712
    lines = ["# Reward checks: varying the ethics sanction", "",
             "The scripted-policy check (docs/reward_checks.md) with every reward function in configs/rewards/ at each sanction "
             f"{', '.join(f'{s:g}' for s in sanctions)} (the reward functions named <file>_s<sanction>). The scripted policies don't "
             "depend on the rewards, so the same logs (basic, 200 validation episodes per set-up) are evaluated afresh for each.", "",
             f"Gamma is {gamma}; a reward on day {days} is worth {gamma ** (days - 1):.2e} of one on the first day.", "",
             "A sanction of s is s times a sanction of 1 at every step (every principle gives +s, -s or 0, and every aggregation combines those "
             "linearly or clips them to +/- s), so its effect on any return is linear in s: it can't change sign as the sanction grows.", "",
             "## Sharing agents against the eaters (focal set-ups)", "",
             "For each sharing policy as the focal agent among eaters: its discounted return minus the eaters' at each sanction; the "
             "sanction's effect for a sanction of 1; and the sanction at which it catches up with the eaters (blank: it never does, as "
             "the sanction lowers it or it isn't behind).", "",
             markdown_table(sweep, index=False), "",
             "## Failed checks by sanction", ""]
    if failed.empty:
        lines.append("None.")
    else:
        counts = failed.pivot_table(index="check", columns="sanction", values="reward_function", aggfunc="count", fill_value=0)
        counts.columns = [f"s={c:g}" for c in counts.columns]
        lines += ["Failed checks (reward functions x agent types) at each sanction:", "", markdown_table(counts), ""]
        by_type = failed.pivot_table(index="check", columns="agent_type", values="reward_function", aggfunc="count", fill_value=0)
        lines += ["By agent type:", "", markdown_table(by_type), ""]
    lines += ["## Checks passed per reward function", "", "| reward function | checks passed | failed | not checked |", "|---|---|---|---|"]
    for name, group in all_checks.groupby("reward_function"):
        lines.append(f"| {name} | {(group['passed'] == True).sum()} | {(group['passed'] == False).sum()} | {group['passed'].isna().sum()} |")  # noqa: E712
    lines += ["", "Full results: data/reward_checks/sanctions/results.csv, checks.csv and sweep.csv."]
    return "\n".join(lines) + "\n"


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
             "without a berry: it never forages, and its return includes that failed action's penalty every day. passive_neutral scores "
             "the same steps with those failed eats as actions with no other effect (neutral_reward): pure inaction.",
             "- An agent that dies on the last day doesn't get the survive reward (as during a run).", "",
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
    sanction_rows = all_checks[all_checks["check"].str.startswith("sanction raises ")]
    if len(sanction_rows):
        lines += ["", "## The sanction's effect on sharing agents (focal set-ups)", "",
                  "The focal agent's discounted return minus the eaters', with the sanction and on base rewards alone, and the sanction's "
                  "effect (the difference):", "", "| policy | reward function | agent type | with sanction | base only | effect |",
                  "|---|---|---|---|---|---|"]
        for row in sanction_rows.itertuples():
            lines.append(f"| {row.policy} | {row.reward_function} | {row.agent_type} | {row.value:.3f} | {row.compared_with:.3f} | "
                         f"{row.value - row.compared_with:+.3f} |")
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


def evaluate(reward_paths, log_paths, out=OUT, report_path=os.path.join(REPO, "docs", "reward_checks.md"), days=200, log=print, sanctions=None):
    """
    sanctions -- a sanction sweep: every reward function at each of these sanctions (with_sanctions), reported by sweep_report
    """
    reward_functions = [load(path) for path in reward_paths]
    base_names = None
    if sanctions:
        swept = with_sanctions(reward_functions, sanctions)
        base_names = {s.name: f.name for f in reward_functions for s in swept if s.name.startswith(f.name + "_s")
                      and s.name.rsplit("_s", 1)[0] == f.name}
        reward_functions = swept
    gamma = for_scenario("basic").gamma
    logs = load_logs(log_paths)
    log(f"{len(logs)} logged steps from {len(log_paths)} runs; {len(reward_functions)} reward functions; gamma {gamma}")
    unit = unit_sanctions(logs)
    agent_table = agent_episodes(logs)
    returns_table = returns(logs, reward_functions, gamma, sanctions=unit)
    results = results_table(agent_table, returns_table)
    data = returns_table.merge(agent_table, on=["scenario", "run", "episode", "agent_id"])
    all_checks = pd.concat([checks(results, data), sanction_effects(logs, reward_functions, gamma, unit, agent_table)], ignore_index=True)
    os.makedirs(out, exist_ok=True)
    results.to_csv(os.path.join(out, "results.csv"), index=False)
    all_checks.to_csv(os.path.join(out, "checks.csv"), index=False)
    sweep = None
    if sanctions:
        sweep = sanction_sweep(all_checks, reward_functions, base_names)
        sweep.to_csv(os.path.join(out, "sweep.csv"), index=False)
    if report_path:
        with open(report_path, "w") as file:
            file.write(sweep_report(all_checks, sweep, gamma, days, sanctions, base_names) if sanctions
                       else report(results, all_checks, gamma, days, reward_functions))
    failed = all_checks[all_checks["passed"] == False]  # noqa: E712
    log(f"{len(all_checks)} checks: {int((all_checks['passed'] == True).sum())} passed, {len(failed)} failed")  # noqa: E712
    return results, all_checks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rewards", nargs="*", help="Reward function files (default: every file in configs/rewards/, current.yaml first)")
    parser.add_argument("--logs", nargs="+", default=None, help="Log files (default: every log in data/reward_checks/logs/)")
    parser.add_argument("--days", type=int, default=200, help="Days in each episode of the logs")
    parser.add_argument("--sanctions", nargs="+", type=float, default=None,
                        help="A sanction sweep: every reward function at each of these sanctions (into data/reward_checks/sanctions/ and "
                             "docs/reward_checks_sanctions.md), e.g. --sanctions 0 0.1 0.2 0.4 0.8 1 2 5")
    args = parser.parse_args()
    rewards = args.rewards or sorted(glob.glob(os.path.join(REPO, "configs", "rewards", "*.yaml")), key=lambda p: (not p.endswith("current.yaml"), p))
    logs = args.logs or sorted(glob.glob(os.path.join(OUT, "logs", "*", "*.csv.gz")))
    if not logs:
        raise SystemExit("no logs: run python -m src.reward_checks.simulate first")
    if args.sanctions:
        evaluate(rewards, logs, out=os.path.join(OUT, "sanctions"), report_path=os.path.join(REPO, "docs", "reward_checks_sanctions.md"),
                 days=args.days, sanctions=args.sanctions)
    else:
        evaluate(rewards, logs, days=args.days)


if __name__ == "__main__":
    main()

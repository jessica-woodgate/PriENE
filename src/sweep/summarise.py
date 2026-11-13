"""
Summarise a sweep: python -m src.sweep.summarise configs/sweeps/<sweep>.yaml
Writes, in the sweep's folder (<output_root>/<scenario>/<step>/):
    sweep_results.csv -- one row per run (configuration, agent type, seed), then a mean and an sd row per configuration and agent type
    curves/<configuration>.png -- learning curves (20-episode moving average of the score, one line per seed and agent type)
    ranking.csv -- configurations ranked by the sweep's selection rule
    summary.md -- the recommended configuration, how it was chosen, and how sensitive the score is to each parameter
The selection rules are docs/fruit_market_sweep.md §3.2 (step_a) and §3.3 (step_b), fixed before any run.
"""
import argparse
import json
import math
import os
import numpy as np
import pandas as pd
from .spec import load, scenario_report
from .runner import read_status, report_frame

MOVING_AVERAGE = 20
#step A: exclusions and ties (docs/fruit_market_sweep.md §3.2)
MAX_ACTION_SHARE = 0.8
MAX_GRAPPLE_SHARE = 0.5
LOSS_BLOWUP = 100
#step B: eligibility and ties (§3.3)
SANCTION_SHARE_RANGE = (0.05, 0.5)
SETTLED_CHANGE = 0.10
INSTABILITY_TIE = 0.10
TARGET_SANCTION_SHARE = 0.25

def _moving_average(series):
    return series.rolling(MOVING_AVERAGE, min_periods=1).mean()

def run_metrics(sweep, run, frame, last_episodes):
    """
    A run's selection measures over its last last_episodes episodes (fewer if it has fewer)
    """
    columns = scenario_report(sweep.scenario)["columns"]
    frame = frame.sort_values("episode")
    last = frame.tail(last_episodes)
    score = frame[columns["score"]]
    final = last[columns["score"]].mean()
    average = _moving_average(score)
    reached = frame["episode"][average >= 0.9 * final] if final > 0 else frame["episode"].iloc[-1:]
    total = _moving_average(frame[columns["total_reward"]])
    earlier = total.iloc[-last_episodes - 1] if len(total) > last_episodes else total.iloc[0]
    metrics = {"episodes": int(frame["episode"].max()), "score": final, "total_reward": last[columns["total_reward"]].mean(),
               "fruit": last[columns["fruit"]].mean() if "fruit" in columns else np.nan,
               "loss": last[columns["loss"]].mean(), "loss_nan": bool(frame[columns["loss"]].isna().any()),
               "convergence_episode": int(reached.iloc[0]) if len(reached) else np.nan,
               "settled": bool(abs(total.iloc[-1] - earlier) <= SETTLED_CHANGE * max(abs(total.iloc[-1]), 1e-9))}
    for name in ("trades", "max_action_share", "grapple_share", "abs_sanction"):
        metrics[name] = last[columns[name]].mean() if name in columns and columns[name] in last else np.nan
    metrics["trades_total"] = last[columns["trades"]].sum() if "trades" in columns and columns["trades"] in last else np.nan
    if not math.isnan(metrics["abs_sanction"]):
        absolute_score = last[columns["score"]].abs().sum()
        absolute_sanction = last[columns["abs_sanction"]].sum()
        metrics["sanction_share"] = absolute_sanction / (absolute_score + absolute_sanction) if absolute_score + absolute_sanction else 0.0
    else:
        metrics["sanction_share"] = np.nan
    metrics["degenerate"] = bool(metrics["loss_nan"] or (metrics["max_action_share"] > MAX_ACTION_SHARE)
                                 or (metrics["grapple_share"] > MAX_GRAPPLE_SHARE))
    return metrics

def collect(sweep):
    """
    One row per run (with its status and, if it reported, its measures), and each run's learning curve
    """
    last_episodes = sweep.selection.get("last_episodes", 50)
    rows, curves = [], {}
    for run in sweep.runs():
        status = read_status(run) or {"status": "not run"}
        row = {"config_id": run.config_id, "agent_type": run.agent_type, "seed": run.seed, "status": status.get("status"),
               "reason": status.get("reason"), "seconds": status.get("seconds"), **{f"param_{k}": v for k, v in run.values.items()}}
        frame = report_frame(sweep, run)
        if frame is not None:
            row.update(run_metrics(sweep, run, frame, last_episodes))
            curves[(run.config_id, run.agent_type, run.seed)] = frame.sort_values("episode")
        rows.append(row)
    return pd.DataFrame(rows), curves

def with_aggregates(results):
    numeric = [c for c in results.columns if c not in ("config_id", "agent_type", "seed", "status", "reason") and pd.api.types.is_numeric_dtype(results[c])]
    rows = []
    for (config_id, agent_type), group in results.groupby(["config_id", "agent_type"], sort=True):
        for name, function in (("mean", "mean"), ("sd", "std")):
            row = {"config_id": config_id, "agent_type": agent_type, "seed": name, "status": f"{(group['status'] == 'done').sum()}/{len(group)} done"}
            row.update(getattr(group[numeric], function)().to_dict())
            rows.append(row)
    return pd.concat([results, pd.DataFrame(rows)], ignore_index=True)

def _changed(sweep, values):
    base = sweep.base_config.to_dict()
    return sum(1 for name, value in values.items() if base.get(name) != value)

def rank_step_a(sweep, results):
    """
    docs/fruit_market_sweep.md §3.2: score = mean over seeds of the last-episodes substrate reward; excluded if degenerate in any seed or
    its loss blows up; ties within one pooled standard error of the best, decided by stability, trading, convergence, fewer changes, size
    """
    config_values = {c["id"]: c["values"] for c in sweep.configurations}
    finished = results[results["status"] == "done"]
    median_loss = finished.groupby("config_id")["loss"].mean().median() if len(finished) else np.nan
    rows = []
    for config_id, group in results.groupby("config_id", sort=True):
        done = group[group["status"] == "done"]
        n = len(done)
        config = sweep.training_config(config_values[config_id])
        row = {"config_id": config_id, **config_values[config_id], "seeds_done": n, "score": done["score"].mean() if n else np.nan,
               "sd": done["score"].std(ddof=1) if n > 1 else np.nan, "fruit": done["fruit"].mean() if n else np.nan,
               "trades": done["trades"].mean() if n else np.nan, "never_trades": bool(n and (done["trades_total"].fillna(0) == 0).all()),
               "convergence_episode": done["convergence_episode"].mean() if n else np.nan, "changes": _changed(sweep, config_values[config_id]),
               "hidden_units": config.hidden_units, "batch_size": config.batch_size}
        excluded = []
        if n < len(group):
            excluded.append("not every seed finished (stopped early, failed or not run)")
        if n and done["degenerate"].any():
            excluded.append("degenerate policy in a seed")
        if n and not np.isnan(median_loss) and done["loss"].mean() > LOSS_BLOWUP * median_loss:
            excluded.append("loss blew up")
        row["excluded"] = "; ".join(excluded)
        rows.append(row)
    table = pd.DataFrame(rows)
    eligible = table[(table["excluded"] == "") & table["score"].notna()]
    if eligible.empty:
        return table.assign(rank=np.nan, tied_with_best=False)
    best = eligible.loc[eligible["score"].idxmax()]
    n = max(int(best["seeds_done"]), 1)
    def standard_error(row):
        return math.sqrt((best["sd"] if not np.isnan(best["sd"]) else 0) ** 2 / n + (row["sd"] if not np.isnan(row["sd"]) else 0) ** 2 / n)
    table["tied_with_best"] = [bool(row["excluded"] == "" and not np.isnan(row["score"]) and best["score"] - row["score"] <= standard_error(row))
                               for _, row in table.iterrows()]
    tied = table[table["tied_with_best"]].sort_values(["sd", "never_trades", "convergence_episode", "changes", "hidden_units", "batch_size"],
                                                      na_position="last")
    rest = (table[~table["tied_with_best"]].assign(is_excluded=lambda frame: frame["excluded"] != "")
            .sort_values(["is_excluded", "score"], ascending=[True, False]).drop(columns="is_excluded"))
    ranked = pd.concat([tied, rest], ignore_index=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked

def rank_step_b(sweep, results):
    """
    docs/fruit_market_sweep.md §3.3: an f (configuration) is eligible only if for every agent type the sanction share is within
    SANCTION_SHARE_RANGE, no seed is degenerate and learning has settled; the chosen f has the lowest instability (sd/|mean| of the
    score over seeds) averaged over agent types with equal weight; ties within INSTABILITY_TIE go to the mean sanction share closest to
    TARGET_SANCTION_SHARE, then to the f closest to the base config's. No measure compares agent types with each other.
    """
    config_values = {c["id"]: c["values"] for c in sweep.configurations}
    rows = []
    for config_id, group in results.groupby("config_id", sort=True):
        sanction = sweep.training_config(config_values[config_id]).sanction
        failures, instabilities, shares = [], [], []
        for agent_type, runs in group.groupby("agent_type"):
            done = runs[runs["status"] == "done"]
            if len(done) < len(runs) or done.empty:
                failures.append(f"{agent_type}: not every seed finished")
                continue
            share = done["sanction_share"].mean()
            mean = done["score"].mean()
            instability = done["score"].std(ddof=1) / abs(mean) if mean else math.inf
            shares.append(share)
            instabilities.append(instability)
            if not SANCTION_SHARE_RANGE[0] <= share <= SANCTION_SHARE_RANGE[1]:
                failures.append(f"{agent_type}: sanction share {share:.2f}")
            if done["degenerate"].any():
                failures.append(f"{agent_type}: degenerate")
            if not done["settled"].all():
                failures.append(f"{agent_type}: not settled")
        rows.append({"config_id": config_id, "sanction": sanction, "mean_instability": float(np.mean(instabilities)) if instabilities else math.inf,
                     "mean_sanction_share": float(np.mean(shares)) if shares else np.nan, "failures": len(failures),
                     "failed": "; ".join(failures), "eligible": not failures})
    table = pd.DataFrame(rows)
    base_sanction = sweep.base_config.sanction
    pool = table[table["eligible"]] if table["eligible"].any() else table[table["failures"] == table["failures"].min()]
    lowest = pool["mean_instability"].min()
    tied = pool[pool["mean_instability"] <= lowest * (1 + INSTABILITY_TIE)].copy()
    tied["share_distance"] = (tied["mean_sanction_share"] - TARGET_SANCTION_SHARE).abs()
    tied["base_distance"] = (tied["sanction"] - base_sanction).abs()
    order = list(tied.sort_values(["share_distance", "base_distance"])["config_id"])
    order += list(table[~table["config_id"].isin(order)].sort_values(["eligible", "failures", "mean_instability"], ascending=[False, True, True])["config_id"])
    ranked = table.set_index("config_id").loc[order].reset_index()
    ranked["rank"] = range(1, len(ranked) + 1)
    ranked["chosen_from"] = "eligible" if table["eligible"].any() else "fewest failures (no f eligible)"
    return ranked

RANKINGS = {"step_a": rank_step_a, "step_b": rank_step_b}

def sensitivity(sweep, ranking):
    """
    For each swept parameter: the mean score of the configurations with each of its values (log-scale terciles for continuous ones)
    """
    lines = []
    if "score" not in ranking:
        return lines
    for name in sweep.swept_parameters:
        if name not in ranking:
            continue
        data = ranking[["score", name]].dropna()
        if data.empty:
            continue
        values = data[name]
        if values.nunique() > 4:
            groups = pd.qcut(np.log(values.astype(float)), 3, duplicates="drop")
            means = data.groupby(groups, observed=True)["score"].mean()
            labels = [f"{math.exp(interval.left):.3g}–{math.exp(interval.right):.3g}" for interval in means.index]
        else:
            means = data.groupby(name)["score"].mean()
            labels = [str(v) for v in means.index]
        spread = means.max() - means.min()
        lines.append(f"| {name} | " + ", ".join(f"{label}: {mean:.2f}" for label, mean in zip(labels, means)) + f" | {spread:.2f} |")
    return lines

def plot_curves(sweep, curves, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    column = scenario_report(sweep.scenario)["columns"]["score"]
    os.makedirs(folder, exist_ok=True)
    for config_id in sorted({key[0] for key in curves}):
        figure, axes = plt.subplots(figsize=(7, 4))
        for (config, agent_type, seed), frame in sorted(curves.items()):
            if config == config_id:
                axes.plot(frame["episode"], _moving_average(frame[column]), label=f"{agent_type} seed {seed}")
        axes.set_xlabel("episode")
        axes.set_ylabel(f"{column} ({MOVING_AVERAGE}-episode moving average)")
        axes.set_title(f"{sweep.step} {config_id}")
        axes.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(os.path.join(folder, f"{config_id}.png"), dpi=100)
        plt.close(figure)

def summarise(sweep, plots=True):
    results, curves = collect(sweep)
    os.makedirs(sweep.folder, exist_ok=True)
    with_aggregates(results).to_csv(os.path.join(sweep.folder, "sweep_results.csv"), index=False)
    rule = sweep.selection.get("rule", "step_a")
    ranking = RANKINGS[rule](sweep, results) if "score" in results else pd.DataFrame()
    ranking.to_csv(os.path.join(sweep.folder, "ranking.csv"), index=False)
    if plots and curves:
        plot_curves(sweep, curves, os.path.join(sweep.folder, "curves"))
    write_markdown(sweep, results, ranking, rule)
    return results, ranking

def markdown_table(frame):
    def cell(value):
        if isinstance(value, float):
            return "" if math.isnan(value) else f"{value:.3g}"
        return str(value).replace("|", "/")
    lines = ["| " + " | ".join(str(c) for c in frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)

def write_markdown(sweep, results, ranking, rule):
    values = {c["id"]: c["values"] for c in sweep.configurations}
    lines = [f"# Sweep {sweep.name}: {sweep.scenario}, step {sweep.step}", "",
             f"{len(results)} runs: " + ", ".join(f"{count} {status}" for status, count in results["status"].value_counts().items()) + ".",
             f"Selection rule: `{rule}` (docs/fruit_market_sweep.md), over the last {sweep.selection.get('last_episodes', 50)} episodes "
             f"of {sweep.episodes}.", ""]
    if ranking.empty or "rank" not in ranking or ranking["rank"].isna().all():
        lines.append("No configuration could be ranked yet.")
    else:
        best = ranking.iloc[0]
        config = sweep.training_config(values[best["config_id"]])
        lines += [f"**Recommended: {best['config_id']}**, changing {values[best['config_id']] or 'nothing'} from the base config.", "",
                  "Its full training config (training length and epsilon decay are the sweep's):", "", "```",
                  json.dumps(config.to_dict(), indent=2), "```", "", "## Ranking", "", markdown_table(ranking), ""]
        table = sensitivity(sweep, ranking)
        if table:
            lines += ["## Sensitivity", "", "Mean score of the configurations with each value, and the spread between values:", "",
                      "| Parameter | Mean score by value | Spread |", "|---|---|---|", *table, ""]
    stopped = results[results["status"].isin(["stopped_early", "failed", "skipped"])]
    if len(stopped):
        lines += ["## Runs that didn't finish", ""] + [f"- {r.config_id}/{r.agent_type}/seed{r.seed}: {r.status} ({r.reason})" for r in stopped.itertuples()]
    with open(os.path.join(sweep.folder, "summary.md"), "w") as file:
        file.write("\n".join(lines) + "\n")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarise a training parameter sweep")
    parser.add_argument("sweep", help="the sweep file (configs/sweeps/*.yaml)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    sweep = load(args.sweep)
    summarise(sweep, plots=not args.no_plots)
    print(f"Written to {sweep.folder}: sweep_results.csv, ranking.csv, summary.md" + ("" if args.no_plots else ", curves/"))

if __name__ == "__main__":
    main()

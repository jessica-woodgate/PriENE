"""
Analysis of a multi-seed experiment (src/experiments.py), with the training seed as the unit of replication:

    python -m src.analysis.multiseed_analysis runs/example                         # an experiment's folder
    python -m src.analysis.multiseed_analysis configs/experiments/example.yaml     # or its file (which also says which runs are missing)

writes into <experiment folder>/analysis/:
    episodes.csv -- one row per society x training seed x test condition x test episode, with every metric (the long table)
    per_seed.csv, per_seed.tex -- level 1: each society x seed x condition x metric, its mean and median over the test episodes
    across_seeds.csv -- level 2: each society x condition x metric, the mean, sd, 95% t-interval and bootstrap interval of the per-seed means
    mixed_model.csv -- each condition x main metric (every metric with --mixed-model-all): metric ~ society + (1 | training_seed) +
        (1 | episode) on the episode-level data, with pairwise contrasts between societies, Holm-corrected. Each fit takes a few minutes
        at 1000 test episodes; --max-model-episodes fits on fewer
    paired_tests.csv -- the simpler check: paired t-tests on per-seed means (seed s is shared by every society), Holm-corrected, and
        Cohen's d from the per-seed means
    consistency.csv -- for each pair of societies, in how many seeds society A beats society B
    norms.csv -- for each society and condition, each emerged norm and the number of seeds it emerged in
    learning_curves.csv, learning_curves_<condition>.png -- each society's training reward, mean +/- sd across seeds
    boxplots/<condition>_<metric>.png -- episode-level distributions per society, per-seed means overlaid
    table_<condition>.tex -- the paper's Table 4 style: each society's mean, median and Cohen's d (against baseline) across seeds
    summary.md -- where results are and aren't consistent across seeds, and any missing, incomplete or mixed-machine runs

Metrics (one value per test episode, from each agent's last row in the episode; an agent's berries are those it holds and has eaten):
    M1_inequality -- Gini coefficient of the agents' berries (lower is better)
    M2_minimum -- the fewest berries an agent has
    M3_maximum -- the most berries an agent has
    M4_total -- the society's total berries
    M5_robustness -- how long agents survive: the mean number of days they live in the episode
    emerged_norms -- the norms that emerged in the episode (the episode report's num_emerged_norms)
    and every other numeric column of the episode report a scenario adds (e.g. commons' mean_stock, efficiency), prefixed episode_.
Fruit Market's test runs report their own episode metrics (its fm_model_episode_reports), which are taken as they are.
"""
import argparse
import glob
import itertools
import json
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from src.run_record import RUN_FILE, read_status

MAIN_METRICS = ["M1_inequality", "M2_minimum", "M3_maximum", "M4_total", "M5_robustness", "emerged_norms"]
#metrics for which lower is better (everything else: higher is better)
LOWER_IS_BETTER = {"M1_inequality"}
#episode report columns that describe the run rather than the society's outcomes
NOT_METRICS = {"episode", "seed", "epsilon", "mean_loss", "num_emerged_norms"}
BASELINE = "baseline"


def gini(values):
    """As the paper's analysis (DataAnalysis._calculate_gini): 0 for perfect equality"""
    x = sorted(values)
    n, total = len(x), sum(x)
    if n == 0 or total == 0:
        return 0.0
    b = sum(xi * (n - i) for i, xi in enumerate(x)) / (n * total)
    return 1 + 1 / n - 2 * b


# ── collecting ──────────────────────────────────────────────────

def experiment_folder(path):
    """An experiment's folder, from the folder itself or its experiment file"""
    if os.path.isdir(path):
        return os.path.abspath(path)
    from src.experiments import load_experiment
    experiment = load_experiment(path)
    return os.path.join(experiment["output_root"], experiment["name"])


def run_folders(folder, mode):
    return sorted(os.path.dirname(path) for path in glob.glob(os.path.join(folder, mode, "*", "*", "*", RUN_FILE)))


def read_record(run):
    with open(os.path.join(run, RUN_FILE)) as file:
        return json.load(file)


def episode_metrics(run):
    """One row per test episode of a run, with every metric"""
    results = os.path.join(run, "data", "results", "current_run")
    agent_reports = glob.glob(os.path.join(results, "agent_reports_*.csv"))
    episode_reports = glob.glob(os.path.join(results, "model_episode_reports_*.csv")) + glob.glob(os.path.join(results, "fm_model_episode_reports_*.csv"))
    if not episode_reports:
        return None
    episodes = pd.read_csv(episode_reports[0])
    numeric = [c for c in episodes.columns if c not in NOT_METRICS and pd.api.types.is_numeric_dtype(episodes[c])]
    table = episodes[["episode"]].copy()
    if "num_emerged_norms" in episodes.columns:
        table["emerged_norms"] = episodes["num_emerged_norms"]
    for column in numeric:
        table["episode_" + column] = episodes[column]
    if agent_reports:
        agents = pd.read_csv(agent_reports[0], usecols=["agent_id", "episode", "day", "berries", "berries_consumed"])
        last = agents.groupby(["episode", "agent_id"]).tail(1).assign(total=lambda d: d["berries"] + d["berries_consumed"])
        per_episode = last.groupby("episode").agg(M1_inequality=("total", gini), M2_minimum=("total", "min"), M3_maximum=("total", "max"),
                                                  M4_total=("total", "sum"), M5_robustness=("day", "mean")).reset_index()
        table = per_episode.merge(table, on="episode", how="left")
    return table


def collect(folder, expected=None):
    """
    The long table of every test run's episodes (society, training_seed, test_condition, episode, metrics), and a report of the runs:
    those not complete, those an experiment file expects but that don't exist, and the machines they ran on
    expected -- the planned runs (src/experiments.plan), or None
    """
    rows, report = [], {"incomplete": [], "missing": [], "machines": {}}
    for run in run_folders(folder, "test"):
        record = read_record(run)
        config = record["configuration"]
        status = read_status(run)
        if status != "complete":
            report["incomplete"].append(f"{os.path.relpath(run, folder)} ({status})")
            continue
        machine = record.get("machine", {})
        report["machines"].setdefault((machine.get("architecture"), tuple(machine.get("gpus") or [])), []).append(os.path.relpath(run, folder))
        table = episode_metrics(run)
        if table is None:
            report["incomplete"].append(f"{os.path.relpath(run, folder)} (no episode report)")
            continue
        table.insert(0, "test_condition", config["condition"])
        table.insert(0, "training_seed", config["seed"])
        table.insert(0, "society", config["society"])
        table["run"] = os.path.relpath(run, folder)
        rows.append(table)
    if expected is not None:
        report["missing"] = [os.path.relpath(r.folder, folder) for r in expected if not os.path.exists(os.path.join(r.folder, RUN_FILE))]
    episodes = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return episodes, report


def metric_columns(episodes):
    return [c for c in episodes.columns if c in MAIN_METRICS or c.startswith("episode_")]


def to_long(episodes):
    return episodes.melt(id_vars=["society", "training_seed", "test_condition", "episode"], value_vars=metric_columns(episodes),
                         var_name="metric", value_name="value").dropna(subset=["value"])


# ── levels 1 and 2 ──────────────────────────────────────────────

def per_seed(long):
    """Level 1: each society x seed x condition x metric, the mean and median over the test episodes"""
    return (long.groupby(["test_condition", "metric", "society", "training_seed"])["value"]
            .agg(mean="mean", median="median", episodes="count").reset_index())


def bootstrap_interval(values, reps=10000, seed=0):
    """Percentile 95% interval of the mean, resampling the seeds"""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def across_seeds(seed_means, reps=10000):
    """Level 2: each society x condition x metric, over its per-seed means: mean, sd, 95% t-interval and bootstrap interval"""
    rows = []
    for (condition, metric, society), group in seed_means.groupby(["test_condition", "metric", "society"]):
        values = group["mean"].to_numpy(dtype=float)
        n, mean = len(values), values.mean()
        sd = values.std(ddof=1) if n > 1 else np.nan
        half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
        low, high = bootstrap_interval(values, reps)
        rows.append({"test_condition": condition, "metric": metric, "society": society, "seeds": n, "mean": mean,
                     "median_of_seed_means": float(np.median(values)), "sd": sd, "t_ci_low": mean - half, "t_ci_high": mean + half,
                     "bootstrap_ci_low": low, "bootstrap_ci_high": high})
    return pd.DataFrame(rows)


# ── inference ───────────────────────────────────────────────────

def holm(p_values):
    """Holm-adjusted p-values, in the order given"""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[index]))
        adjusted[index] = running
    return adjusted


def mixed_model(data):
    """
    metric ~ society + (1 | training_seed) + (1 | episode) on one condition and metric's episode-level data (training seeds and test
    episodes are shared by every society, so both are crossed random effects, fitted as variance components of one group). Returns
    the fixed effects and their covariance, the variance components, and whether the fit converged
    """
    import statsmodels.formula.api as smf
    data = data.assign(group=1, society=data["society"].astype(str), training_seed=data["training_seed"].astype(str),
                       episode=data["episode"].astype(str))
    societies = sorted(data["society"].unique())
    reference = BASELINE if BASELINE in societies else societies[0]
    formula = f"value ~ C(society, Treatment(reference='{reference}'))"
    model = smf.mixedlm(formula, data, groups="group", re_formula="0",
                        vc_formula={"training_seed": "0 + C(training_seed)", "episode": "0 + C(episode)"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = model.fit(reml=True, method=["lbfgs", "powell"])
    names = fit.fe_params.index
    effects = {reference: 0.0}
    for society in societies:
        if society != reference:
            effects[society] = fit.fe_params[[n for n in names if f"[T.{society}]" in n][0]]
    return fit, reference, effects, societies


def society_contrasts(fit, reference, societies):
    """Every pair of societies' difference in the mixed model's fixed effects, with its Wald test"""
    names = list(fit.fe_params.index)
    covariance = fit.cov_params().loc[names, names].to_numpy()
    params = fit.fe_params.to_numpy()

    def vector(society):
        v = np.zeros(len(names))
        if society != reference:
            v[[i for i, n in enumerate(names) if f"[T.{society}]" in n][0]] = 1.0
        return v
    rows = []
    for a, b in itertools.combinations(societies, 2):
        c = vector(a) - vector(b)
        estimate = float(c @ params)
        se = float(np.sqrt(c @ covariance @ c))
        z = estimate / se if se > 0 else np.nan
        rows.append({"society_a": a, "society_b": b, "difference": estimate, "se": se, "z": z, "p": 2 * stats.norm.sf(abs(z))})
    table = pd.DataFrame(rows)
    if len(table):
        table["p_holm"] = holm(table["p"])
    return table


def mixed_models(long, max_episodes=None, metrics=MAIN_METRICS):
    """
    The mixed model and its contrasts for every condition and metric in metrics (None: every metric), on the first max_episodes test
    episodes if given. A fit has a random effect per test episode, so it grows with them: about 3 minutes for 8 societies x 5 seeds x
    1000 episodes (M1 Pro), hence the main metrics only by default
    """
    tables = []
    for (condition, metric), data in long.groupby(["test_condition", "metric"]):
        if metrics is not None and metric not in metrics:
            continue
        if max_episodes is not None:
            data = data[data["episode"] <= max_episodes]
        if data["society"].nunique() < 2 or data["value"].nunique() < 2:
            continue
        try:
            fit, reference, _, societies = mixed_model(data)
            contrasts = society_contrasts(fit, reference, societies)
            contrasts.insert(0, "metric", metric)
            contrasts.insert(0, "test_condition", condition)
            contrasts["converged"] = bool(fit.converged)
            contrasts["seed_variance"] = float(fit.vcomp[0]) if len(fit.vcomp) > 0 else np.nan
            contrasts["episode_variance"] = float(fit.vcomp[1]) if len(fit.vcomp) > 1 else np.nan
            contrasts["residual_variance"] = float(fit.scale)
            tables.append(contrasts)
        except (np.linalg.LinAlgError, ValueError) as error:
            tables.append(pd.DataFrame([{"test_condition": condition, "metric": metric, "error": str(error)}]))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def cohens_d(a, b):
    """Cohen's d of two societies' per-seed means (pooled sd)"""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2)) if len(a) + len(b) > 2 else np.nan
    return (a.mean() - b.mean()) / pooled if pooled and pooled > 0 else np.nan


def paired_tests(seed_means):
    """Paired t-tests between societies on per-seed means (matched by seed), Holm-corrected per condition and metric; Cohen's d"""
    tables = []
    for (condition, metric), group in seed_means.groupby(["test_condition", "metric"]):
        wide = group.pivot(index="training_seed", columns="society", values="mean")
        rows = []
        for a, b in itertools.combinations(sorted(wide.columns), 2):
            pair = wide[[a, b]].dropna()
            diff = pair[a] - pair[b]
            if len(pair) >= 2 and diff.std(ddof=1) > 0:
                t, p = stats.ttest_rel(pair[a], pair[b])
            else:
                t, p = np.nan, np.nan
            rows.append({"test_condition": condition, "metric": metric, "society_a": a, "society_b": b, "seeds": len(pair),
                         "mean_difference": diff.mean(), "t": t, "p": p, "cohens_d": cohens_d(pair[a], pair[b])})
        table = pd.DataFrame(rows)
        if len(table):
            table["p_holm"] = np.nan
            tested = table["p"].notna()
            table.loc[tested, "p_holm"] = holm(table.loc[tested, "p"])
        tables.append(table)
    columns = ["test_condition", "metric", "society_a", "society_b", "seeds", "mean_difference", "t", "p", "cohens_d", "p_holm"]
    tables = [t for t in tables if len(t)]
    #with a single society there are no pairs: an empty table with its columns
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=columns)


def consistency(seed_means):
    """For each pair of societies, in how many seeds society A beats society B (the better per-seed mean, by the metric's direction)"""
    rows = []
    for (condition, metric), group in seed_means.groupby(["test_condition", "metric"]):
        wide = group.pivot(index="training_seed", columns="society", values="mean")
        sign = -1 if metric in LOWER_IS_BETTER else 1
        for a, b in itertools.permutations(sorted(wide.columns), 2):
            pair = wide[[a, b]].dropna()
            rows.append({"test_condition": condition, "metric": metric, "society_a": a, "society_b": b, "seeds": len(pair),
                         "a_beats_b": int((sign * (pair[a] - pair[b]) > 0).sum()), "ties": int((pair[a] == pair[b]).sum())})
    return pd.DataFrame(rows, columns=["test_condition", "metric", "society_a", "society_b", "seeds", "a_beats_b", "ties"])


# ── norms and learning curves ───────────────────────────────────

def norms_by_seed(folder):
    """For each society and test condition, each norm that emerged (in any test episode) and in how many training seeds"""
    seen = {}
    for run in run_folders(folder, "test"):
        if read_status(run) != "complete":
            continue
        config = read_record(run)["configuration"]
        for path in glob.glob(os.path.join(run, "data", "results", "current_run", "*_emerged_norms.json")):
            with open(path) as file:
                episodes = json.load(file)
            norms = {name for norm_list in episodes.values() for norm in norm_list for name in norm}
            for norm in norms:
                seen.setdefault((config["society"], config["condition"], norm), set()).add(config["seed"])
    rows = [{"society": s, "test_condition": c, "norm": n, "seeds": len(seeds)} for (s, c, n), seeds in seen.items()]
    return pd.DataFrame(rows, columns=["society", "test_condition", "norm", "seeds"]).sort_values(["test_condition", "society", "seeds"],
                                                                                                    ascending=[True, True, False])


def learning_curves(folder):
    """Each society's training reward per episode (mean_reward of its training report): mean and sd across seeds"""
    rows = []
    for run in run_folders(folder, "train"):
        config = read_record(run)["configuration"]
        reports = glob.glob(os.path.join(run, "data", "results", "current_run", "*model_episode_reports_train_*.csv"))
        if not reports:
            continue
        report = pd.read_csv(reports[0], usecols=lambda c: c in ["episode", "mean_reward"])
        rows.append(report.assign(society=config["society"], training_seed=config["seed"], train_condition=config["condition"]))
    if not rows:
        return pd.DataFrame()
    curves = pd.concat(rows, ignore_index=True)
    return (curves.groupby(["train_condition", "society", "episode"])["mean_reward"]
            .agg(mean="mean", sd="std", seeds="count").reset_index())


# ── tables, plots and summary ───────────────────────────────────

def seed_table_latex(seed_means):
    table = seed_means.pivot_table(index=["test_condition", "metric", "society"], columns="training_seed", values="mean")
    table.columns = [f"seed {s}" for s in table.columns]
    return table.to_latex(float_format="%.3f", multirow=True)


def paper_table(levels, paired, condition, metrics=MAIN_METRICS):
    """The paper's Table 4 style for one condition: each society's mean (median) across seeds, and Cohen's d against baseline"""
    rows = []
    level = levels[(levels["test_condition"] == condition) & (levels["metric"].isin(metrics))]
    for society, group in level.groupby("society"):
        row = {"Society": society}
        for metric in metrics:
            m = group[group["metric"] == metric]
            if m.empty:
                continue
            d = np.nan
            if society != BASELINE:
                pair = paired[(paired["test_condition"] == condition) & (paired["metric"] == metric)
                              & (((paired["society_a"] == society) & (paired["society_b"] == BASELINE))
                                 | ((paired["society_a"] == BASELINE) & (paired["society_b"] == society)))]
                if not pair.empty:
                    d = pair["cohens_d"].iloc[0] * (1 if pair["society_a"].iloc[0] == society else -1)
            row[metric] = f"{m['mean'].iloc[0]:.2f} ({m['median_of_seed_means'].iloc[0]:.2f})" + ("" if np.isnan(d) else f", d={d:.2f}")
        rows.append(row)
    return pd.DataFrame(rows).to_latex(index=False, escape=True,
                                       caption=f"{condition}: mean (median) of per-seed means across training seeds; d is Cohen's d against baseline")


def box_plots(long, seed_means, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(out, "boxplots"), exist_ok=True)
    for (condition, metric), data in long.groupby(["test_condition", "metric"]):
        societies = sorted(data["society"].unique())
        figure, axes = plt.subplots(figsize=(max(6, 1.1 * len(societies)), 4))
        axes.boxplot([data[data["society"] == s]["value"] for s in societies], showfliers=False)
        means = seed_means[(seed_means["test_condition"] == condition) & (seed_means["metric"] == metric)]
        for i, society in enumerate(societies, start=1):
            values = means[means["society"] == society]["mean"]
            axes.scatter(np.full(len(values), i) + np.linspace(-0.12, 0.12, len(values)), values, s=18, zorder=3, label="seed means" if i == 1 else None)
        axes.set_xticks(range(1, len(societies) + 1), societies, rotation=30, ha="right")
        axes.set_title(f"{condition}: {metric}")
        axes.legend(loc="best", fontsize=8)
        figure.tight_layout()
        figure.savefig(os.path.join(out, "boxplots", f"{condition}_{metric}.png"), dpi=110)
        plt.close(figure)


def learning_curve_plots(curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for condition, data in curves.groupby("train_condition"):
        figure, axes = plt.subplots(figsize=(8, 4))
        for society, group in data.groupby("society"):
            axes.plot(group["episode"], group["mean"], label=society, linewidth=1)
            axes.fill_between(group["episode"], group["mean"] - group["sd"].fillna(0), group["mean"] + group["sd"].fillna(0), alpha=0.2)
        axes.set_xlabel("training episode")
        axes.set_ylabel("mean reward (mean +/- sd across seeds)")
        axes.set_title(condition)
        axes.legend(fontsize=7, ncol=2)
        figure.tight_layout()
        figure.savefig(os.path.join(out, f"learning_curves_{condition}.png"), dpi=110)
        plt.close(figure)


def summary(levels, paired, consistent, report, norms):
    """Markdown: per condition and metric, the best society across seeds and whether it is best in every seed; pairs whose order
    changes between seeds; and any problems with the runs"""
    lines = ["# Multi-seed results", ""]
    if report["incomplete"] or report["missing"]:
        lines += ["## Runs not included", ""] + [f"- missing: {r}" for r in report["missing"]] + [f"- {r}" for r in report["incomplete"]] + [""]
    if len(report["machines"]) > 1:
        lines += ["## Warning: runs on different machines", "",
                  "Runs of this experiment were produced on different architectures or devices, so they may not be exactly comparable:", ""]
        lines += [f"- {arch}, GPUs {list(gpus) or 'none'}: {len(runs)} runs" for (arch, gpus), runs in report["machines"].items()] + [""]
    for (condition, metric), level in levels.groupby(["test_condition", "metric"]):
        if level.empty:
            continue
        lower = metric in LOWER_IS_BETTER
        best = level.sort_values("mean", ascending=lower).iloc[0]
        pairs = consistent[(consistent["test_condition"] == condition) & (consistent["metric"] == metric) & (consistent["society_a"] == best["society"])]
        always = pairs[pairs["a_beats_b"] == pairs["seeds"]]["society_b"].tolist()
        not_always = pairs[pairs["a_beats_b"] < pairs["seeds"]]
        lines.append(f"## {condition}: {metric}")
        lines.append(f"- Best across seeds: **{best['society']}** (mean {best['mean']:.3f}, sd {best['sd']:.3f} over {int(best['seeds'])} seeds; "
                     f"{'lower' if lower else 'higher'} is better).")
        if len(pairs):
            lines.append(f"- It beats {len(always)} of {len(pairs)} other societies in every seed"
                         + (f"; not in every seed against: " + ", ".join(f"{r.society_b} ({r.a_beats_b}/{r.seeds})" for r in not_always.itertuples())
                            if len(not_always) else "") + ".")
        flips = consistent[(consistent["test_condition"] == condition) & (consistent["metric"] == metric) & (consistent["society_a"] < consistent["society_b"])
                           & (consistent["a_beats_b"] > 0) & (consistent["a_beats_b"] + consistent["ties"] < consistent["seeds"])]
        if len(flips):
            lines.append(f"- Pairs whose order changes between seeds: {len(flips)} (" + ", ".join(f"{r.society_a} vs {r.society_b} {r.a_beats_b}/{r.seeds}"
                                                                                               for r in flips.head(8).itertuples()) + ("…" if len(flips) > 8 else "") + ").")
        significant = paired[(paired["test_condition"] == condition) & (paired["metric"] == metric) & (paired["p_holm"] < 0.05)]
        lines.append(f"- Paired tests across seeds (Holm): {len(significant)} of {len(paired[(paired['test_condition'] == condition) & (paired['metric'] == metric)])} pairs differ at p < 0.05.")
        lines.append("")
    if len(norms):
        lines += ["## Norms", "", "Norms that emerged in every seed, by society and condition:", ""]
        for (condition, society), group in norms.groupby(["test_condition", "society"]):
            seeds = int(levels[(levels["test_condition"] == condition) & (levels["society"] == society)]["seeds"].max()) if len(levels) else 0
            everywhere = group[group["seeds"] >= seeds]
            lines.append(f"- {condition}, {society}: {len(everywhere)} of {len(group)} norms emerged in all {seeds} seeds")
    return "\n".join(lines) + "\n"


def analyse(path, out=None, max_model_episodes=None, bootstrap_reps=10000, plots=True, mixed=True, log=print, mixed_all_metrics=False):
    folder = experiment_folder(path)
    expected = None
    if not os.path.isdir(path):
        from src.experiments import load_experiment, plan
        expected = [r for r in plan(load_experiment(path)) if r.mode == "test"]
    out = out or os.path.join(folder, "analysis")
    os.makedirs(out, exist_ok=True)
    episodes, report = collect(folder, expected)
    for problem in report["missing"]:
        log(f"missing run: {problem}")
    for problem in report["incomplete"]:
        log(f"not included (not complete): {problem}")
    if len(report["machines"]) > 1:
        log("WARNING: runs of this experiment were produced on different architectures or devices: "
            + "; ".join(f"{arch} GPUs {list(gpus) or 'none'} ({len(runs)} runs)" for (arch, gpus), runs in report["machines"].items()))
    if episodes.empty:
        log("no complete test runs to analyse")
        return None
    episodes.to_csv(os.path.join(out, "episodes.csv"), index=False)
    long = to_long(episodes)
    seed_means = per_seed(long)
    seed_means.to_csv(os.path.join(out, "per_seed.csv"), index=False)
    with open(os.path.join(out, "per_seed.tex"), "w") as file:
        file.write(seed_table_latex(seed_means))
    levels = across_seeds(seed_means, bootstrap_reps)
    levels.to_csv(os.path.join(out, "across_seeds.csv"), index=False)
    paired = paired_tests(seed_means)
    paired.to_csv(os.path.join(out, "paired_tests.csv"), index=False)
    consistent = consistency(seed_means)
    consistent.to_csv(os.path.join(out, "consistency.csv"), index=False)
    if mixed:
        mixed_models(long, max_model_episodes, None if mixed_all_metrics else MAIN_METRICS).to_csv(os.path.join(out, "mixed_model.csv"), index=False)
    norms = norms_by_seed(folder)
    norms.to_csv(os.path.join(out, "norms.csv"), index=False)
    curves = learning_curves(folder)
    curves.to_csv(os.path.join(out, "learning_curves.csv"), index=False)
    for condition in levels["test_condition"].unique():
        with open(os.path.join(out, f"table_{condition}.tex"), "w") as file:
            file.write(paper_table(levels, paired, condition))
    if plots:
        box_plots(long, seed_means, out)
        if len(curves):
            learning_curve_plots(curves, out)
    with open(os.path.join(out, "summary.md"), "w") as file:
        file.write(summary(levels, paired, consistent, report, norms))
    log(f"analysis written to {out}")
    return {"episodes": episodes, "per_seed": seed_means, "across_seeds": levels, "paired": paired, "consistency": consistent,
            "norms": norms, "learning_curves": curves, "report": report, "out": out}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", help="An experiment's folder (<output_root>/<name>) or its experiment file")
    parser.add_argument("--out", help="Where to write the analysis (default <experiment folder>/analysis)")
    parser.add_argument("--max-model-episodes", type=int, default=None,
                        help="Fit the mixed models on the first this many test episodes (they grow with the number of episodes)")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap resamples of the seeds")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-mixed-model", action="store_true")
    parser.add_argument("--mixed-model-all", action="store_true",
                        help="Fit the mixed model for every metric, not only the main ones (M1-M5 and emerged norms); each fit takes minutes")
    args = parser.parse_args()
    analyse(args.experiment, args.out, args.max_model_episodes, args.bootstrap, not args.no_plots, not args.no_mixed_model,
            mixed_all_metrics=args.mixed_model_all)


if __name__ == "__main__":
    main()

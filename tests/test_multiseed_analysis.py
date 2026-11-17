"""
The multi-seed analysis (src/analysis/multiseed_analysis.py) on a synthetic experiment with a known difference: society "kind" gives
every agent 2 more berries than "baseline" in every episode, on top of a training-seed effect and a test-episode effect shared by both
(as when both are tested on the same test episodes). The aggregation levels, the mixed model, the paired tests, consistency, norms,
learning curves and the report of incomplete runs and mixed machines are checked.
"""
import json
import os
import numpy as np
import pandas as pd
import pytest
from src.analysis import multiseed_analysis as msa
from src.run_record import write_status

SEEDS = [1, 2, 3, 4, 5]
EPISODES = 20
CONDITION = "colours_4agents_10days"
EFFECT = 2


def write_run(folder, mode, society, seed, condition, agent_rows=None, episode_rows=None, norms=None, train_rewards=None,
              status="complete", architecture="arm64"):
    os.makedirs(os.path.join(folder, "data", "results", "current_run"), exist_ok=True)
    with open(os.path.join(folder, "run.json"), "w") as file:
        json.dump({"configuration": {"mode": mode, "society": society, "seed": seed, "condition": condition},
                   "machine": {"architecture": architecture, "gpus": []}}, file)
    write_status(folder, status, **({"pid": 999999} if status == "running" else {}))
    results = os.path.join(folder, "data", "results", "current_run")
    if agent_rows is not None:
        pd.DataFrame(agent_rows).to_csv(os.path.join(results, f"agent_reports_colours_{society}.csv"), index=False)
        pd.DataFrame(episode_rows).to_csv(os.path.join(results, f"model_episode_reports_colours_{society}.csv"), index=False)
    if norms is not None:
        with open(os.path.join(results, f"colours_{society}_emerged_norms.json"), "w") as file:
            json.dump(norms, file)
    if train_rewards is not None:
        pd.DataFrame({"episode": range(1, len(train_rewards) + 1), "mean_reward": train_rewards}).to_csv(
            os.path.join(results, f"model_episode_reports_train_basic_{society}.csv"), index=False)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("analysis") / "exp"
    rng = np.random.default_rng(0)
    seed_effect = {s: rng.normal(0, 1.5) for s in SEEDS}
    episode_effect = {e: rng.normal(0, 1.0) for e in range(1, EPISODES + 1)}
    for society, extra in [("baseline", 0), ("kind", EFFECT)]:
        for seed in SEEDS:
            agent_rows, episode_rows = [], []
            for episode in range(1, EPISODES + 1):
                for agent in range(4):
                    berries = max(0, round(3 + agent + extra + seed_effect[seed] + episode_effect[episode] + rng.normal(0, 0.5)))
                    agent_rows += [{"agent_id": agent, "episode": episode, "day": day, "berries": 0, "berries_consumed": berries if day == 10 else 0}
                                   for day in range(1, 11)]
                episode_rows.append({"episode": episode, "num_emerged_norms": 3 + extra, "seed": 1000, "epsilon": 0, "mean_reward": 1.0 + extra})
            norms = {"1": [{"IF,a,THEN,eat": {}}, {f"IF,b,THEN,{society}": {}}]}
            if seed == 1:
                norms["2"] = [{"IF,rare,THEN,move": {}}]
            write_run(str(root / "test" / CONDITION / society / f"train_seed_{seed}"), "test", society, seed, CONDITION, agent_rows, episode_rows, norms)
            write_run(str(root / "train" / "basic_4agents_10days" / society / f"seed_{seed}"), "train", society, seed, "basic_4agents_10days",
                      train_rewards=[float(e + extra + seed) for e in range(5)])
    # one test run stopped partway, and one training seed's test produced on another machine, are reported
    write_run(str(root / "test" / CONDITION / "kind" / "train_seed_9"), "test", "kind", 9, CONDITION, status="running")
    with open(root / "test" / CONDITION / "baseline" / "train_seed_5" / "run.json") as file:
        record = json.load(file)
    record["machine"]["architecture"] = "x86_64"
    with open(root / "test" / CONDITION / "baseline" / "train_seed_5" / "run.json", "w") as file:
        json.dump(record, file)
    messages = []
    results = msa.analyse(str(root), bootstrap_reps=500, log=messages.append)
    return root, results, messages


def test_every_complete_runs_episodes_are_collected_into_one_long_table(experiment):
    _, results, _ = experiment
    episodes = results["episodes"]
    assert len(episodes) == 2 * len(SEEDS) * EPISODES
    assert {"society", "training_seed", "test_condition", "episode", "M1_inequality", "M2_minimum", "M3_maximum", "M4_total",
            "M5_robustness", "emerged_norms"} <= set(episodes.columns)
    assert set(episodes["training_seed"]) == set(SEEDS)
    # robustness: every agent lived all 10 days
    assert (episodes["M5_robustness"] == 10).all()


def test_level_one_is_each_seeds_mean_and_median_over_its_episodes(experiment):
    _, results, _ = experiment
    per_seed = results["per_seed"]
    episodes = results["episodes"]
    row = per_seed[(per_seed["society"] == "kind") & (per_seed["training_seed"] == 3) & (per_seed["metric"] == "M4_total")].iloc[0]
    values = episodes[(episodes["society"] == "kind") & (episodes["training_seed"] == 3)]["M4_total"]
    assert row["mean"] == pytest.approx(values.mean()) and row["median"] == pytest.approx(values.median()) and row["episodes"] == EPISODES


def test_level_two_summarises_the_per_seed_means(experiment):
    _, results, _ = experiment
    levels, per_seed = results["across_seeds"], results["per_seed"]
    row = levels[(levels["society"] == "kind") & (levels["metric"] == "M4_total")].iloc[0]
    means = per_seed[(per_seed["society"] == "kind") & (per_seed["metric"] == "M4_total")]["mean"]
    assert row["seeds"] == len(SEEDS) and row["mean"] == pytest.approx(means.mean()) and row["sd"] == pytest.approx(means.std(ddof=1))
    assert row["t_ci_low"] < row["mean"] < row["t_ci_high"] and row["bootstrap_ci_low"] <= row["mean"] <= row["bootstrap_ci_high"]


def test_the_mixed_model_recovers_the_societies_difference(experiment):
    _, results, _ = experiment
    long = msa.to_long(results["episodes"])
    data = long[(long["metric"] == "M4_total")]
    fit, reference, effects, societies = msa.mixed_model(data)
    assert reference == "baseline"
    # 4 agents x 2 more berries each
    assert effects["kind"] == pytest.approx(4 * EFFECT, abs=1.0)
    contrasts = msa.society_contrasts(fit, reference, societies)
    assert contrasts.iloc[0]["p_holm"] < 0.001
    # the seed effect is a variance component of its own
    assert fit.vcomp[0] > 0


def test_paired_tests_consistency_and_effect_sizes(experiment):
    _, results, _ = experiment
    paired = results["paired"]
    row = paired[(paired["metric"] == "M4_total")].iloc[0]
    assert (row["society_a"], row["society_b"]) == ("baseline", "kind")
    assert row["p_holm"] < 0.01 and row["cohens_d"] < -1 and row["seeds"] == len(SEEDS)
    consistent = results["consistency"]
    kind_beats = consistent[(consistent["metric"] == "M4_total") & (consistent["society_a"] == "kind")].iloc[0]
    assert kind_beats["a_beats_b"] == len(SEEDS) and kind_beats["seeds"] == len(SEEDS)


def test_norms_are_counted_by_the_seeds_they_emerged_in(experiment):
    _, results, _ = experiment
    norms = results["norms"]
    kind = norms[norms["society"] == "kind"].set_index("norm")["seeds"]
    assert kind["IF,a,THEN,eat"] == len(SEEDS) and kind["IF,rare,THEN,move"] == 1


def test_learning_curves_are_mean_and_sd_across_seeds(experiment):
    _, results, _ = experiment
    curves = results["learning_curves"]
    first = curves[(curves["society"] == "kind") & (curves["episode"] == 1)].iloc[0]
    assert first["seeds"] == len(SEEDS) and first["mean"] == pytest.approx(np.mean([0 + EFFECT + s for s in SEEDS]))


def test_incomplete_runs_and_mixed_machines_are_reported_and_outputs_written(experiment):
    root, results, messages = experiment
    assert any("train_seed_9" in message and "incomplete" in message for message in messages)
    assert any("different architectures" in message for message in messages)
    out = results["out"]
    for name in ["episodes.csv", "per_seed.csv", "per_seed.tex", "across_seeds.csv", "mixed_model.csv", "paired_tests.csv",
                 "consistency.csv", "norms.csv", "learning_curves.csv", f"table_{CONDITION}.tex", "summary.md",
                 "learning_curves_basic_4agents_10days.png", f"boxplots/{CONDITION}_M4_total.png"]:
        assert os.path.exists(os.path.join(out, name)), name
    with open(os.path.join(out, "summary.md")) as file:
        summary = file.read()
    assert "Best across seeds: **kind**" in summary and "different machines" in summary and "train_seed_9" in summary
    with open(os.path.join(out, f"table_{CONDITION}.tex")) as file:
        assert "d=" in file.read()

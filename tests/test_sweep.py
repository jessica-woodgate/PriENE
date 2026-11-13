"""
The parameter sweep tooling (src/sweep/): sweep files, the configurations they expand to, refusing scenario parameters, run folders,
resuming, early stopping, per-run seeding, and the summary and selection rules on a small fake sweep. Runs here are faked (a stand-in
for run.py that writes a report), except the last test, which runs the real thing with the stand-in substrate.
"""
import json
import os
import pandas as pd
import pytest
from src.sweep import spec as sweep_spec
from src.sweep.spec import load, SweepError, scenario_answers
from src.sweep.runner import SweepRunner, read_status, early_stop_reason
from src.sweep.summarise import summarise
from src import training_config as tc

REPO = sweep_spec.REPO

def write_sweep(tmp_path, **overrides):
    raw = {"name": "test", "scenario": "fruit_market", "step": "T", "output_root": str(tmp_path / "sweeps"), "agent_types": ["baseline"],
           "seeds": [1, 2, 3], "episodes": 60, "epsilon_floor_fraction": 0.7,
           "search": {"method": "random", "configurations": 4, "sampling_seed": 0, "include_base": True},
           "parameters": {"learning_rate": {"log_uniform": [3e-5, 1e-3]}, "gamma": [0.9, 0.95, 0.99]},
           "selection": {"rule": "step_a", "last_episodes": 20}}
    raw.update(overrides)
    path = tmp_path / "sweep.yaml"
    import yaml
    with open(path, "w") as file:
        yaml.safe_dump(raw, file)
    return str(path)

class FakeProcess:
    def __init__(self, returncode=0):
        self.returncode = returncode
    def poll(self):
        return self.returncode
    def terminate(self):
        pass
    def wait(self):
        return self.returncode

def fake_launch(score_of, calls, episodes=60, failing=()):
    """
    A stand-in for run.py: writes an episode report whose score is score_of(training config, seed, episode), and saved progress
    """
    def launch(command, answers, folder, log_path):
        config = tc.load(command[command.index("--training-config") + 1])
        seed = int(command[command.index("-s") + 1])
        calls.append((folder, answers))
        if folder in failing:
            return FakeProcess(1)
        rows = [{"episode": e, "mean_env_reward": score_of(config, seed, e), "mean_reward": score_of(config, seed, e),
                 "total_fruit_eaten": 5, "total_trades": 1, "mean_loss": 0.1, "max_action_share": 0.2, "grapple_share": 0.1,
                 "mean_abs_sanction": 0.0} for e in range(1, episodes + 1)]
        os.makedirs(os.path.join(folder, "data/results/current_run"), exist_ok=True)
        pd.DataFrame(rows).to_csv(os.path.join(folder, "data/results/current_run/fm_model_episode_reports_train_fruit_market_baseline.csv"), index=False)
        os.makedirs(os.path.join(folder, "data/model_variables/trained/x/16_agents/baseline/agent_0"), exist_ok=True)
        with open(os.path.join(folder, "data/model_variables/trained/x/16_agents/baseline/agent_0/progress.json"), "w") as file:
            json.dump({"episode": episodes}, file)
        return FakeProcess(0)
    return launch

def runner(sweep, launch):
    return SweepRunner(sweep, workers=2, launch=launch, poll_seconds=0, log=lambda message: None)

# --- sweep files ---

def test_a_random_sweep_expands_to_the_same_configurations_every_time(tmp_path):
    sweep = load(write_sweep(tmp_path))
    again = load(write_sweep(tmp_path))
    assert sweep.configurations == again.configurations
    assert [c["id"] for c in sweep.configurations] == ["c00", "c01", "c02", "c03"]
    assert sweep.configurations[0]["values"] == {}
    for c in sweep.configurations[1:]:
        assert 3e-5 <= c["values"]["learning_rate"] <= 1e-3 and c["values"]["gamma"] in (0.9, 0.95, 0.99)

def test_each_runs_training_config_has_the_sweeps_length_and_compressed_epsilon_schedule(tmp_path):
    sweep = load(write_sweep(tmp_path))
    config = sweep.training_config({"gamma": 0.99})
    assert config.max_training_episodes == 60 and config.gamma == 0.99 and config.sanction == 3.5
    assert config.epsilon_decay == pytest.approx(tc.compressed_epsilon_decay(1.0, 0.01, 60, 0.7))

def test_grid_and_list_searches(tmp_path):
    grid = load(write_sweep(tmp_path, search={"method": "grid", "include_base": False}, parameters={"gamma": [0.9, 0.99], "batch_size": [32, 64]}))
    assert len(grid.configurations) == 4
    listed = load(write_sweep(tmp_path, search={"method": "list", "include_base": False}, parameters={},
                              configurations=[{"sanction": 0.5}, {"sanction": 6.0}]))
    assert [c["values"] for c in listed.configurations] == [{"sanction": 0.5}, {"sanction": 6.0}]

@pytest.mark.parametrize("parameters", [{"num_players": [8, 16]}, {"grappling": [True, False]}, {"max_days": [100]}, {"hunger_delay": [25]}])
def test_a_sweep_refuses_to_change_scenario_parameters(tmp_path, parameters):
    with pytest.raises(tc.TrainingConfigError, match="parameter of the scenario"):
        load(write_sweep(tmp_path, parameters=parameters))

def test_a_sweep_file_cant_give_prompt_answers_or_unknown_keys(tmp_path):
    with pytest.raises(SweepError, match="can't have"):
        load(write_sweep(tmp_path, answers=["fruit_market", "500"]))

def test_fruit_market_runs_always_answer_the_substrates_defaults():
    assert scenario_answers("fruit_market", "maximin", "homogeneous") == ["fruit_market", "", "", "", "", "", "", "homogeneous", "maximin", "y"]

# --- runs ---

def test_runs_are_in_folders_named_by_scenario_step_configuration_agent_type_and_seed(tmp_path):
    sweep = load(write_sweep(tmp_path))
    run = sweep.runs()[0]
    assert run.folder == os.path.join(str(tmp_path / "sweeps"), "fruit_market", "T", "c00", "baseline", "seed1")
    assert len(sweep.runs()) == 4 * 3

def test_every_run_logs_its_full_configuration_and_result(tmp_path):
    sweep = load(write_sweep(tmp_path))
    calls = []
    runner(sweep, fake_launch(lambda c, s, e: 1.0, calls)).run_all()
    for run in sweep.runs():
        record = read_status(run)
        assert record["status"] == "done" and record["seed"] == run.seed and record["values"] == run.values
        assert record["training_config"] == sweep.training_config(run.values).to_dict()
        assert tc.load(os.path.join(run.folder, "training_config.yaml")) == sweep.training_config(run.values)
        assert record["command"][-2:] == ["--training-config", os.path.join(run.folder, "training_config.yaml")]
        assert record["command"][record["command"].index("-s") + 1] == str(run.seed)

def test_a_resumed_sweep_skips_finished_runs_and_continues_unfinished_ones(tmp_path):
    sweep = load(write_sweep(tmp_path))
    calls = []
    failing = {sweep.runs()[5].folder}
    runner(sweep, fake_launch(lambda c, s, e: 1.0, calls, failing=failing)).run_all()
    assert len(calls) == 12 and read_status(sweep.runs()[5])["status"] == "failed"
    calls.clear()
    runner(sweep, fake_launch(lambda c, s, e: 1.0, calls)).run_all()
    assert [folder for folder, _ in calls] == [sweep.runs()[5].folder]
    #it had saved no progress, so it starts afresh; a run that had would be continued
    assert calls[0][1][-1] == "y" and len(calls[0][1]) == len(scenario_answers("fruit_market", "baseline", "homogeneous"))
    run = sweep.runs()[0]
    record = read_status(run)
    record["status"] = "running"
    json.dump(record, open(os.path.join(run.folder, "run.json"), "w"))
    calls.clear()
    runner(sweep, fake_launch(lambda c, s, e: 1.0, calls)).run_all()
    assert len(calls) == 1 and calls[0][1] == scenario_answers("fruit_market", "baseline", "homogeneous") + ["y"]

def test_a_changed_sweep_file_cant_continue_an_earlier_sweep(tmp_path):
    path = write_sweep(tmp_path)
    runner(load(path), fake_launch(lambda c, s, e: 1.0, [])).run_all(dry_run=True)
    with pytest.raises(RuntimeError, match="sweep file changed"):
        runner(load(write_sweep(tmp_path, search={"method": "random", "configurations": 4, "sampling_seed": 1})),
               fake_launch(lambda c, s, e: 1.0, [])).run_all(dry_run=True)

def test_a_random_sweep_with_too_few_possible_values_says_so(tmp_path):
    with pytest.raises(SweepError, match="fewer than 4 different configurations"):
        load(write_sweep(tmp_path, parameters={"gamma": [0.9]}))

def test_early_stopping_rules(tmp_path):
    sweep = load(write_sweep(tmp_path, early_stopping={"episode": 10, "window": 5, "min_reward": 2.0, "max_action_share": 0.9}))
    frame = pd.DataFrame({"episode": range(1, 11), "mean_env_reward": [3.0] * 10, "mean_loss": [0.1] * 10, "max_action_share": [0.2] * 10})
    assert early_stop_reason(sweep, frame) is None
    assert "below" in early_stop_reason(sweep, frame.assign(mean_env_reward=1.0))
    assert "of all actions" in early_stop_reason(sweep, frame.assign(max_action_share=0.97))
    assert early_stop_reason(sweep, frame.head(8).assign(mean_env_reward=0.0)) is None

def test_runs_of_a_configuration_are_reproducible_and_seeded_apart(tmp_path, monkeypatch):
    """
    PriENE's side of a run is fixed by its seed: two Fruit Market training runs with the same training config and seed choose the same
    actions (with the deterministic stand-in substrate); another seed chooses differently
    """
    from src.fruit_market.model import FruitMarketModel
    from tests.fruit_market_stand_in import StandInBackend
    sweep = load(write_sweep(tmp_path))
    config = sweep.training_config(sweep.configurations[1]["values"]).with_values({"min_experiences": 5})
    roles = ["apple_farmer", "banana_farmer"]
    def actions(seed, folder):
        monkeypatch.chdir(tmp_path)
        os.makedirs(folder, exist_ok=True)
        monkeypatch.chdir(folder)
        backend = StandInBackend(roles)
        model = FruitMarketModel("homogeneous", 2, "baseline", 1, 30, True, "checkpoints/", False, False, seed=seed, backend=backend,
                                 training_config=config)
        for _ in range(30):
            model.step()
        return backend.actions
    assert actions(1, "a") == actions(1, "b")
    assert actions(1, "c") != actions(2, "d")

# --- summary ---

def test_the_summary_ranks_by_the_step_a_rule_and_writes_every_output(tmp_path):
    sweep = load(write_sweep(tmp_path))
    best = sweep.configurations[2]["values"]
    def score(config, seed, episode):
        bonus = 2.0 if config.learning_rate == best["learning_rate"] and config.gamma == best["gamma"] else 0.0
        return min(episode / 20, 1.0) * (3.0 + bonus) + 0.01 * seed
    runner(sweep, fake_launch(score, [])).run_all()
    results, ranking = summarise(sweep)
    assert ranking.iloc[0]["config_id"] == "c02"
    table = pd.read_csv(os.path.join(sweep.folder, "sweep_results.csv"))
    assert len(table) == 12 + 2 * 4
    assert set(table["seed"].astype(str)) >= {"1", "2", "3", "mean", "sd"}
    for name in ["ranking.csv", "summary.md", "curves/c00.png", "curves/c03.png"]:
        assert os.path.exists(os.path.join(sweep.folder, name))
    summary = open(os.path.join(sweep.folder, "summary.md")).read()
    assert "Recommended: c02" in summary and "## Sensitivity" in summary

def test_step_a_excludes_degenerate_configurations_and_breaks_ties_by_stability(tmp_path):
    sweep = load(write_sweep(tmp_path))
    results = pd.DataFrame([{"config_id": c, "agent_type": "baseline", "seed": s, "status": "done", "score": score, "fruit": 1, "trades": 1,
                             "trades_total": 10, "loss": 0.1, "convergence_episode": 10, "degenerate": c == "c03"}
                            for c, scores in {"c00": [5.0, 5.0, 5.0], "c01": [5.3, 4.9, 5.1], "c02": [1.0, 1.0, 1.0], "c03": [9.0, 9.0, 9.0]}.items()
                            for s, score in zip([1, 2, 3], scores)])
    from src.sweep.summarise import rank_step_a
    ranking = rank_step_a(sweep, results)
    #c03 scores best but is degenerate; c01's mean is a little higher than c00's, within a standard error, and c00 is more stable
    assert list(ranking["config_id"])[:2] == ["c00", "c01"]
    assert ranking.set_index("config_id").loc["c03", "excluded"] == "degenerate policy in a seed"

def test_step_b_chooses_one_sanction_for_every_type_by_eligibility_then_stability(tmp_path):
    sweep = load(write_sweep(tmp_path, agent_types=["utilitarian", "maximin"], search={"method": "list", "include_base": False}, parameters={},
                             configurations=[{"sanction": 0.5}, {"sanction": 2.0}, {"sanction": 6.0}], selection={"rule": "step_b", "last_episodes": 20}))
    shares = {"c00": 0.02, "c01": 0.3, "c02": 0.7}
    rows = []
    for c in ["c00", "c01", "c02"]:
        for t in ["utilitarian", "maximin"]:
            for s in [1, 2, 3]:
                rows.append({"config_id": c, "agent_type": t, "seed": s, "status": "done", "score": 5.0 + 0.1 * s, "sanction_share": shares[c],
                             "degenerate": False, "settled": True})
    from src.sweep.summarise import rank_step_b
    ranking = rank_step_b(sweep, pd.DataFrame(rows))
    #only 2.0 keeps the sanction between 5% and 50% of the reward for both types
    assert ranking.iloc[0]["sanction"] == 2.0 and ranking.iloc[0]["eligible"]
    assert not ranking.set_index("config_id").loc["c00", "eligible"] and not ranking.set_index("config_id").loc["c02", "eligible"]

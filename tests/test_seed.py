import numpy as np
import pandas as pd
import pytest

from tests.conftest import SCENARIOS, _build, _make_pretrained_checkpoint


def run_trajectory(model, steps):
    """Everything random in a step: agent order/actions/positions/attributes and berry positions."""
    trajectory = []
    for _ in range(steps):
        model.step()
        agents = sorted((a for a in model.schedule.agents if a.agent_type != "berry"), key=lambda a: a.unique_id)
        berries = sorted((b for b in model.schedule.agents if b.agent_type == "berry"), key=lambda b: b.unique_id)
        trajectory.append((
            model.episode, model.day,
            tuple((a.pos, a.current_action, round(float(a.health), 6), a.berries) for a in agents),
            tuple(b.pos for b in berries),
        ))
    return trajectory


def network_weights(model):
    agents = sorted((a for a in model.schedule.agents if a.agent_type != "berry"), key=lambda a: a.unique_id)
    return [w for a in agents for w in a.decision_module.q_network.dqn.get_weights()]


def make_training_model(tmp_path, seed, name):
    # training=True: exercises weight initialisation, exploration and replay sampling
    return _build("basic", "homogeneous", 12, "utilitarian", True, str(tmp_path / name) + "/",
                  False, False, name, 3, 10, seed=seed)


def test_same_seed_reproduces_training_run(tmp_path):
    first = make_training_model(tmp_path, 42, "first")
    first_trajectory = run_trajectory(first, 30)
    second = make_training_model(tmp_path, 42, "second")
    second_trajectory = run_trajectory(second, 30)
    assert first_trajectory == second_trajectory
    for w1, w2 in zip(network_weights(first), network_weights(second)):
        np.testing.assert_array_equal(w1, w2)


def test_different_seeds_give_different_runs(tmp_path):
    first = run_trajectory(make_training_model(tmp_path, 1, "first"), 30)
    second = run_trajectory(make_training_model(tmp_path, 2, "second"), 30)
    assert first != second


@pytest.mark.parametrize("scenario", SCENARIOS[1:])
def test_same_seed_reproduces_test_run(tmp_path, scenario):
    """colours/allotment/capabilities are evaluated with training=False (and allotment/capabilities
    draw random resource allocations), so check those are reproducible too."""
    checkpoint_path = _make_pretrained_checkpoint(tmp_path, "utilitarian")
    trajectories = [
        run_trajectory(_build(scenario, "homogeneous", 12, "utilitarian", False, checkpoint_path,
                              False, False, f"{scenario}_{i}", 3, 10, seed=7), 25)
        for i in range(2)
    ]
    assert trajectories[0] == trajectories[1]


def test_seed_is_stored_and_none_leaves_run_unseeded(tmp_path):
    assert make_training_model(tmp_path, 5, "seeded").seed == 5
    unseeded = make_training_model(tmp_path, None, "unseeded")
    assert unseeded.seed is None
    run_trajectory(unseeded, 5)


@pytest.mark.parametrize("seed", [42, None])
def test_seed_written_to_model_episode_reports(tmp_path, monkeypatch, seed):
    monkeypatch.chdir(tmp_path)  # reports are written to the relative path data/results/current_run
    model = _build("basic", "homogeneous", 12, "utilitarian", True, str(tmp_path / "ckpt") + "/",
                   True, False, "seeded", 3, 5, seed=seed)
    while model.episode <= 2:
        model.step()
    df = pd.read_csv("data/results/current_run/model_episode_reports_seeded.csv")
    assert list(df.columns)[-1] == "seed"
    assert len(df) == 2
    if seed is None:
        assert df["seed"].isna().all()
    else:
        assert (df["seed"] == seed).all()

import os

import pytest

from src.harvest_exception import InitialHealthsException
from src.scenarios.health_harvest import HealthHarvest
from tests.conftest import INITIAL_HEALTHS, NUM_AGENTS, _build, _make_pretrained_checkpoint

TRAINED_CHECKPOINTS = "data/model_variables/200_days/4_agents/"
# absolute, since in_tmp_dir changes the working directory before the model is built
TRAINED_CHECKPOINTS_ABS = os.path.abspath(TRAINED_CHECKPOINTS) + "/"


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def build(tmp_path, scenario, seed=None, checkpoint_path=None, max_days=10):
    if checkpoint_path is None:
        checkpoint_path = _make_pretrained_checkpoint(tmp_path, "baseline")
    return _build(scenario, "homogeneous", NUM_AGENTS * 3, "baseline", False, checkpoint_path,
                  False, False, f"test_{scenario}", 2, max_days, seed=seed)


def agent_healths(model):
    agents = sorted((a for a in model.schedule.agents if a.agent_type != "berry"), key=lambda a: a.unique_id)
    return [a.health for a in agents]


def test_initial_healths_applied_at_construction_and_after_episode_reset(in_tmp_dir):
    model = build(in_tmp_dir, "health")
    assert agent_healths(model) == INITIAL_HEALTHS
    for agent in model.get_living_agents():
        assert agent.days_left_to_live == agent.get_days_left_to_live()
        agent.health = 0.1
    model.finish_episode()
    assert model.episode == 2
    assert agent_healths(model) == INITIAL_HEALTHS


@pytest.mark.parametrize("initial_healths", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5, 0.5], [1.0, 1.0, 1.0, 0.0], [1.0, -1.0, 1.0, 0.5]])
def test_invalid_initial_healths_raise(in_tmp_dir, initial_healths):
    with pytest.raises(InitialHealthsException):
        HealthHarvest("homogeneous", NUM_AGENTS, NUM_AGENTS * 3, "baseline", 8, 8, 2, 10, False,
                      str(in_tmp_dir) + "/", False, False, initial_healths)


def test_observation_shape_matches_colours(in_tmp_dir):
    health_model = build(in_tmp_dir, "health")
    colours_model = build(in_tmp_dir, "colours")
    for health_agent, colours_agent in zip(health_model.get_living_agents(), colours_model.get_living_agents()):
        assert health_agent.observe().shape == colours_agent.observe().shape
        assert health_agent.actions == colours_agent.actions


def test_berry_allocations_match_colours_for_same_seed(in_tmp_dir):
    health_model = build(in_tmp_dir, "health", seed=42)
    colours_model = build(in_tmp_dir, "colours", seed=42)
    assert health_model.allocations == colours_model.allocations
    berries = lambda model: [(b.pos, b.allocation_id) for b in model.berries]
    assert berries(health_model) == berries(colours_model)


@pytest.mark.skipif(not os.path.isdir(TRAINED_CHECKPOINTS), reason="needs the trained 200_days checkpoints")
def test_short_run_with_trained_checkpoint(in_tmp_dir):
    model = build(in_tmp_dir, "health", seed=42, checkpoint_path=TRAINED_CHECKPOINTS_ABS, max_days=20)
    while model.episode <= 2:
        model.step()
    assert agent_healths(model) == INITIAL_HEALTHS

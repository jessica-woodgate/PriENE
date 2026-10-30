import hashlib
import os

import pandas as pd
import pytest

from src.harvest_exception import InitialHealthsException, InitialHealthsNotSupported
from tests.conftest import INITIAL_HEALTHS, NUM_AGENTS, TEST_SCENARIOS, _build, _make_pretrained_checkpoint, use_unequal_instant_allotments

TRAINED_CHECKPOINTS = "data/model_variables/200_days/4_agents/"
# absolute, since in_tmp_dir changes the working directory before the model is built
TRAINED_CHECKPOINTS_ABS = os.path.abspath(TRAINED_CHECKPOINTS) + "/"
DEFAULT_START_HEALTH = 0.8

# sha256 of the reports (read and rewritten by pandas) from a seeded run of each scenario, made with
# the code before unequal starting health became an option -- see run_reference_scenario
REFERENCE_REPORT_HASHES = {
    "agent_reports_allotment.csv": "a713b6e3f6f34ec27259436ab2f6ce437393a0247830e90454aa53b8f7a24cc1",
    "agent_reports_capabilities.csv": "4dc0236c5b50dfaf8cccff8e9c82f788bb43701247bd599de8c6e8a36a7748da",
    "agent_reports_colours.csv": "cc7e9e9f9e481b34d9f87a9fdb39cc1b1d581dfc096bcd849087e387e0bd0bd3",
    "model_episode_reports_allotment.csv": "551d284d15a8276d0cf5d592e27d2737d14e401fb41bc9a93d4af3035ec44285",
    "model_episode_reports_capabilities.csv": "66680a3af0792640d8a098d509eac606f924fe40298b199d46e5015f84030309",
    "model_episode_reports_colours.csv": "f2ba7f29c1f5595b8ff5d098e827dae756046e004e12156697c261c76bb5be6d",
}

needs_trained_checkpoints = pytest.mark.skipif(not os.path.isdir(TRAINED_CHECKPOINTS), reason="needs the trained 200_days checkpoints")


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def build(tmp_path, scenario, initial_healths=None, seed=None, checkpoint_path=None, max_days=10, filepath=None):
    if checkpoint_path is None:
        checkpoint_path = _make_pretrained_checkpoint(tmp_path, "baseline")
    return _build(scenario, "homogeneous", NUM_AGENTS * 3, "baseline", False, checkpoint_path,
                  False, False, filepath or f"test_{scenario}", 2, max_days, seed=seed, initial_healths=initial_healths)


def agent_healths(model):
    agents = sorted((a for a in model.schedule.agents if a.agent_type != "berry"), key=lambda a: a.unique_id)
    return [a.health for a in agents]


def report_hash(path):
    # columns added since the reference outputs were made
    df = pd.read_csv(path).drop(columns=["initial_health", "seasonal_berry_allocation", "season", "season_quality", "seasonal_berry_target", "allotment_regrowth_delay"], errors="ignore")
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


@needs_trained_checkpoints
@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_option_off_reproduces_reference_outputs(in_tmp_dir, scenario, monkeypatch):
    # the references predate allotments' equal split and slower allotment, so allotment is run as it was then
    use_unequal_instant_allotments(monkeypatch)
    model = _build(scenario, "homogeneous", NUM_AGENTS * 3, "utilitarian", False, TRAINED_CHECKPOINTS_ABS,
                   True, False, scenario, 2, 20, seed=42)
    while model.episode <= 2:
        model.step()
    for report in ["agent_reports", "model_episode_reports"]:
        filename = f"{report}_{scenario}.csv"
        assert report_hash(f"data/results/current_run/{filename}") == REFERENCE_REPORT_HASHES[filename]
    agent_reports = pd.read_csv(f"data/results/current_run/agent_reports_{scenario}.csv")
    assert (agent_reports["initial_health"] == DEFAULT_START_HEALTH).all()


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_initial_healths_applied_at_construction_and_after_episode_reset(in_tmp_dir, scenario):
    model = build(in_tmp_dir, scenario, INITIAL_HEALTHS)
    assert agent_healths(model) == INITIAL_HEALTHS
    for agent in model.get_living_agents():
        assert agent.days_left_to_live == agent.get_days_left_to_live()
        agent.health = 0.1
    model.finish_episode()
    assert model.episode == 2
    assert agent_healths(model) == INITIAL_HEALTHS


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_option_off_keeps_default_start_health(in_tmp_dir, scenario):
    model = build(in_tmp_dir, scenario)
    assert agent_healths(model) == [DEFAULT_START_HEALTH] * NUM_AGENTS


def test_initial_healths_with_basic_raise(in_tmp_dir):
    with pytest.raises(InitialHealthsNotSupported):
        _build("basic", "homogeneous", NUM_AGENTS * 3, "baseline", True, str(in_tmp_dir / "ckpt") + "/",
               False, False, "basic", 2, 10, initial_healths=INITIAL_HEALTHS)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
@pytest.mark.parametrize("initial_healths", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.5, 0.5], [1.0, 1.0, 1.0, 0.0], [1.0, -1.0, 1.0, 0.5]])
def test_invalid_initial_healths_raise(in_tmp_dir, scenario, initial_healths):
    with pytest.raises(InitialHealthsException):
        _build(scenario, "homogeneous", NUM_AGENTS * 3, "baseline", False, str(in_tmp_dir) + "/",
               False, False, scenario, 2, 10, initial_healths=initial_healths)


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_observation_shape_matches_with_option_on_and_off(in_tmp_dir, scenario):
    on_model = build(in_tmp_dir, scenario, INITIAL_HEALTHS)
    off_model = build(in_tmp_dir, scenario)
    for on_agent, off_agent in zip(on_model.get_living_agents(), off_model.get_living_agents()):
        assert on_agent.observe().shape == off_agent.observe().shape
        assert on_agent.actions == off_agent.actions


@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_berry_allocations_match_with_option_on_and_off(in_tmp_dir, scenario):
    on_model = build(in_tmp_dir, scenario, INITIAL_HEALTHS, seed=42)
    off_model = build(in_tmp_dir, scenario, seed=42)
    assert on_model.allocations == off_model.allocations
    berries = lambda model: [(b.pos, b.allocation_id) for b in model.berries]
    assert berries(on_model) == berries(off_model)


@needs_trained_checkpoints
@pytest.mark.parametrize("scenario", TEST_SCENARIOS)
def test_short_run_with_trained_checkpoint_writes_initial_healths(in_tmp_dir, scenario):
    model = _build(scenario, "homogeneous", NUM_AGENTS * 3, "baseline", False, TRAINED_CHECKPOINTS_ABS,
                   True, False, f"{scenario}_unequal_health", 2, 20, seed=42, initial_healths=INITIAL_HEALTHS)
    while model.episode <= 2:
        model.step()
    assert agent_healths(model) == INITIAL_HEALTHS
    agent_reports = pd.read_csv(f"data/results/current_run/agent_reports_{scenario}_unequal_health.csv")
    for agent_id, initial_health in enumerate(INITIAL_HEALTHS):
        assert (agent_reports.loc[agent_reports["agent_id"] == agent_id, "initial_health"] == initial_health).all()

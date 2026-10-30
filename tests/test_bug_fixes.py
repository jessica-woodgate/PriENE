import os

import numpy as np
import pandas as pd
import pytest

from src.agent.modules.ethics_module import EthicsModule
from src.harvest_exception import UnrecognisedPrinciple
from tests.conftest import NUM_AGENTS, _build, _make_pretrained_checkpoint, long_lived


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def checkpoint_path(tmp_path_factory):
    return _make_pretrained_checkpoint(tmp_path_factory.mktemp("bug_fixes"), "baseline")


def agents(model):
    return sorted((a for a in model.schedule.agents if a.agent_type != "berry"), key=lambda a: a.unique_id)


@pytest.mark.parametrize("scenario", ["basic", "colours", "allotment"])
def test_episode_report_statistics_are_over_the_episodes_agent_rows(in_tmp_dir, checkpoint_path, scenario):
    model = long_lived(_build(scenario, "homogeneous", NUM_AGENTS * 3, "baseline", False, checkpoint_path, True, False, scenario, 2, 15, seed=42))
    while model.episode <= 2:
        model.step()
    agent_reports = pd.read_csv(f"data/results/current_run/agent_reports_{scenario}.csv")
    episode_reports = pd.read_csv(f"data/results/current_run/model_episode_reports_{scenario}.csv")
    for episode, rows in agent_reports.groupby("episode"):
        report = episode_reports[episode_reports["episode"] == episode].iloc[0]
        assert report["mean_berries"] == pytest.approx(rows["berries"].mean())
        assert report["max_berries"] == rows["berries"].max()
        assert report["mean_berries_consumed"] == pytest.approx(rows["berries_consumed"].mean())
        assert report["mean_berries_thrown"] == pytest.approx(rows["berries_thrown"].mean())
        assert report["mean_health"] == pytest.approx(rows["health"].mean())
        assert report["median_health"] == pytest.approx(rows["health"].median())
        assert report["variance_health"] == pytest.approx(rows["health"].var())


def test_gini_is_measured_when_every_agent_has_died(in_tmp_dir, checkpoint_path):
    model = _build("colours", "homogeneous", NUM_AGENTS * 3, "baseline", False, checkpoint_path, False, False, "colours", 1, 10, seed=42)
    for agent, consumed in zip(agents(model), [6, 0, 0, 0]):
        agent.berries_consumed = consumed
    for agent in list(model.living_agents):
        model._remove_agent(agent)
    assert model.living_agents == []
    # one agent ate everything: as unequal as 4 agents can be
    assert model._gini_berries_consumed() == pytest.approx(0.75)


def test_mean_loss_is_over_the_episode_just_ended(in_tmp_dir):
    model = long_lived(_build("basic", "homogeneous", NUM_AGENTS * 3, "baseline", True, str(in_tmp_dir / "checkpoints") + "/",
                              False, False, "basic", 3, 40, seed=42))
    while model.episode <= 2:
        model.step()
    # episode 2 has ended (model is in episode 3): each agent's losses are only episode 3's so far, i.e. none
    assert all(agent.decision_module.losses == [] for agent in agents(model))
    # one learning step each in episode 3
    model.step()
    assert all(len(agent.decision_module.losses) == 1 for agent in agents(model))


def test_rendering_skips_berries_waiting_to_regrow(in_tmp_dir, checkpoint_path, monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from src.data_handling.render_pygame import RenderPygame
    model = _build("allotment", "homogeneous", NUM_AGENTS * 3, "baseline", False, checkpoint_path, False, False, "allotment", 1, 10,
                   seed=42, allotment_regrowth_delays=[3, 3])
    model.step()
    for berry in model.berries[:3]:
        berry.foraged = True
    model._update_schedule()
    assert any(b.pos is None for b in model.berries)
    render = RenderPygame(model.max_width, model.max_height)
    assert render.render_pygame(model)


def test_an_unknown_principle_raises_clearly():
    with pytest.raises(UnrecognisedPrinciple, match="multiobjective"):
        EthicsModule(0.4, "multiobjective")
    for principle in EthicsModule.PRINCIPLES + EthicsModule.AGGREGATIONS:
        EthicsModule(0.4, principle)

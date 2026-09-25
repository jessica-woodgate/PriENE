import pytest

from tests.conftest import AGENT_TYPES, SCENARIOS, make_scenario_model, make_training_model


# --- A: state-consistency invariants (deterministic, not dependent on agent behaviour) ---

def test_day_counter_starts_the_same_way_on_construction_and_reset(tmp_path):
    """
    Regression test for a bug where HarvestModel.__init__ set self.day = 1 while _reset() (which
    runs at the start of every subsequent episode) set self.day = 0, making the very first episode
    of a run one round shorter than every episode after it. max_days=10 is comfortably short enough
    that this check doesn't depend on it actually being reached.
    """
    model = make_training_model(tmp_path, "baseline", max_days=10)
    day_after_init = model.day
    model._reset()
    assert model.day == day_after_init


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_episode_never_runs_longer_than_max_days(tmp_path, scenario):
    """
    Episode length can legitimately vary -- a society that manages resources well survives to
    max_days, one that doesn't dies out early -- so this only asserts the upper bound, never
    equality: self.day must never be observed (from outside the model, i.e. after step() returns)
    at or beyond max_days, since finish_episode()/_reset() always fire within the same step() call
    that crosses the threshold.
    """
    model = make_scenario_model(tmp_path, scenario, "baseline", max_days=10, max_episodes=3)
    for _ in range(40):
        model.step()
        assert model.day < model.max_days


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_agent_and_berry_counts_are_conserved_across_steps_and_episodes(tmp_path, scenario):
    """
    Checked at every single step (not just at episode boundaries, where HarvestModel._reset()
    already self-validates via NumAgentsException/NumBerriesException) -- catches bookkeeping
    corruption that happens to still balance out by the time an episode boundary is reached.
    """
    model = make_scenario_model(tmp_path, scenario, "baseline", max_days=10, max_episodes=3)
    for _ in range(40):
        model.step()
        non_berry_agents = [a for a in model.schedule.agents if a.agent_type != "berry"]
        berries = [a for a in model.schedule.agents if a.agent_type == "berry"]
        assert len(non_berry_agents) == model.num_agents
        assert len(berries) == model.num_berries
        assert len(model.living_agents) <= model.num_agents


# --- B: crash/regression smoke tests ---

@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("agent_type", AGENT_TYPES)
def test_scenario_runs_for_every_agent_type_without_raising(tmp_path, scenario, agent_type):
    model = make_scenario_model(tmp_path, scenario, agent_type, max_days=10, max_episodes=2)
    for _ in range(20):
        model.step()


@pytest.mark.parametrize("agent_type", AGENT_TYPES)
def test_render_pygame_does_not_raise_for_any_agent_type(tmp_path, agent_type, monkeypatch):
    """
    Regression test for a bug where RenderPygame.agent_colours was missing entries for the four
    aggregation agent types (average/majoritarian/optimist/veto), causing a KeyError as soon as
    rendering was enabled for them.
    """
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")  # no real display needed
    from src.data_handling.render_pygame import RenderPygame

    model = make_training_model(tmp_path, agent_type, max_days=10, max_episodes=2)
    render_inst = RenderPygame(model.max_width, model.max_height)
    render_inst.render_pygame(model)

from src.scenarios.basic_harvest import BasicHarvest
from src.scenarios.colours_harvest import ColoursHarvest
from src.scenarios.allotment_harvest import AllotmentHarvest
from src.scenarios.capabilities_harvest import CapabilitiesHarvest

# the 8 agent types actually reachable through run.py's CLI (PRINCIPLES + AGGREGATIONS);
# "multiobjective" is deliberately excluded -- it isn't reachable through any documented entry point
AGENT_TYPES = [
    "baseline",
    "utilitarian",
    "maximin",
    "egalitarian",
    "average",
    "majoritarian",
    "optimist",
    "veto",
]

SCENARIOS = ["basic", "colours", "allotment", "capabilities"]
# scenarios with berry allocations, which take seasons
TEST_SCENARIOS = ["colours", "allotment", "capabilities"]

NUM_AGENTS = 4
NUM_ALLOCATIONS = 2
GRID = {
    "basic": (8, 8),
    "colours": (8, 8),
    "allotment": (16, 8),
    "capabilities": (8, 8),
}


def _build(scenario, society_mix, num_start_berries, agent_type, training, checkpoint_path,
           write_data, write_norms, filepath, max_episodes, max_days, seed=None,
           norms_module_type="rule_based", seasons=False, season_length=25,
           num_allocations=NUM_ALLOCATIONS, allotment_regrowth_delays=None, mixed_allotments="split"):
    max_width, max_height = GRID[scenario]
    if scenario == "basic":
        return BasicHarvest(
            society_mix, NUM_AGENTS, num_start_berries, agent_type, max_width, max_height,
            max_episodes, max_days, training, checkpoint_path, write_data, write_norms, filepath, seed, norms_module_type,
            seasons, season_length, allotment_regrowth_delays,
        )
    elif scenario == "colours":
        return ColoursHarvest(
            society_mix, NUM_AGENTS, num_start_berries, agent_type, max_width, max_height,
            max_episodes, max_days, training, checkpoint_path, write_data, write_norms, filepath, seed, norms_module_type,
            seasons, season_length, allotment_regrowth_delays,
        )
    elif scenario == "allotment":
        return AllotmentHarvest(
            society_mix, NUM_AGENTS, num_start_berries, num_allocations, agent_type, max_width,
            max_height, max_episodes, max_days, training, checkpoint_path, write_data,
            write_norms, filepath, seed, norms_module_type, seasons, season_length, allotment_regrowth_delays, mixed_allotments,
        )
    elif scenario == "capabilities":
        return CapabilitiesHarvest(
            society_mix, NUM_AGENTS, num_start_berries, num_allocations, agent_type, max_width,
            max_height, max_episodes, max_days, training, checkpoint_path, write_data,
            write_norms, filepath, seed, norms_module_type, seasons, season_length, allotment_regrowth_delays,
        )
    raise ValueError(f"unknown scenario {scenario}")


def long_lived(model, health=3.0):
    """
    Gives every agent enough start health to live through every day of a short episode (restored by agent.reset() each
    episode), so episodes in tests don't end early; health draws nothing from any random stream
    """
    for agent in model.schedule.agents:
        if agent.agent_type != "berry":
            agent.start_health = health
            agent.health = health
            agent.days_left_to_live = agent.get_days_left_to_live()
            agent.total_days_left_to_live = agent.days_left_to_live
    return model


def use_unequal_instant_allotments(monkeypatch):
    """
    The allotment scenario before the equal split: each allotment holds the sum of its agents' unequal resources and every
    berry regrows in the step it is foraged, as when the reference outputs were made. Delays passed explicitly still apply
    """
    def assign_unequal_allocations(self, columns):
        resources = self._generate_resource_allocations(self.num_agents)
        allocations = {}
        for i, ((start, end), agent_ids) in enumerate(zip(columns, self._agent_groups())):
            allocations["allocation_"+str(i)] = {"id": i, "berry_allocation": sum(resources[agent_id] for agent_id in agent_ids),
                                                 "allotment": [start, end, 0, self.max_height], "agent_ids": agent_ids}
        return allocations
    monkeypatch.setattr(AllotmentHarvest, "_assign_allocations", assign_unequal_allocations)
    monkeypatch.setattr(AllotmentHarvest, "_default_regrowth_delays", lambda self: [0] * self.num_allotments)


def make_training_model(tmp_path, agent_type, max_days=10, max_episodes=2, society_mix="homogeneous"):
    """A training model in basic_harvest, where the paper's agents were trained (any scenario can train; see test_modes.py)."""
    checkpoint_path = str(tmp_path / "checkpoints") + "/"
    return _build(
        "basic", society_mix, NUM_AGENTS * 3, agent_type, True, checkpoint_path,
        False, False, f"train_{agent_type}", max_episodes, max_days,
    )


def _make_pretrained_checkpoint(tmp_path, agent_type, max_days=5):
    """
    Briefly trains on basic_harvest and returns the checkpoint_path it saved to, so a
    training=False model can load from it. Mirrors the real workflow: train a policy on
    basic_harvest, then evaluate it on the other, test-only scenarios.
    """
    checkpoint_path = str(tmp_path / "checkpoints") + "/"
    model = _build(
        "basic", "homogeneous", NUM_AGENTS * 3, agent_type, True, checkpoint_path,
        False, False, f"pretrain_{agent_type}", 0, max_days,
    )
    for _ in range(max_days + 1):
        model.step()  # runs past max_days at least once, triggering finish_episode -> save_models
    return checkpoint_path


def make_scenario_model(tmp_path, scenario, agent_type, max_days=10, max_episodes=2, society_mix="homogeneous", track_norms=False,
                        norms_module_type="rule_based"):
    """
    Builds a model for the given scenario/agent_type in the paper's workflow: basic_harvest trains
    (training=True); colours/allotment/capabilities test (training=False) a policy pretrained on
    basic_harvest. Every scenario can run in either mode; test_modes.py covers the other combinations.
    """
    num_start_berries = NUM_AGENTS * 3
    if scenario == "basic":
        checkpoint_path = str(tmp_path / "checkpoints") + "/"
        training = True
    else:
        checkpoint_path = _make_pretrained_checkpoint(tmp_path, agent_type)
        training = False
    return _build(
        scenario, society_mix, num_start_berries, agent_type, training, checkpoint_path,
        False, track_norms, f"test_{scenario}_{agent_type}", max_episodes, max_days,
        norms_module_type=norms_module_type,
    )

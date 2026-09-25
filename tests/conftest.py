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

NUM_AGENTS = 4
NUM_ALLOCATIONS = 2
GRID = {
    "basic": (8, 8),
    "colours": (8, 8),
    "allotment": (16, 8),
    "capabilities": (8, 8),
}


def _build(scenario, society_mix, num_start_berries, agent_type, training, checkpoint_path,
           write_data, write_norms, filepath, max_episodes, max_days):
    max_width, max_height = GRID[scenario]
    if scenario == "basic":
        return BasicHarvest(
            society_mix, NUM_AGENTS, num_start_berries, agent_type, max_width, max_height,
            max_episodes, max_days, training, checkpoint_path, write_data, write_norms, filepath,
        )
    elif scenario == "colours":
        return ColoursHarvest(
            society_mix, NUM_AGENTS, num_start_berries, agent_type, max_width, max_height,
            max_episodes, max_days, training, checkpoint_path, write_data, write_norms, filepath,
        )
    elif scenario == "allotment":
        return AllotmentHarvest(
            society_mix, NUM_AGENTS, num_start_berries, NUM_ALLOCATIONS, agent_type, max_width,
            max_height, max_episodes, max_days, training, checkpoint_path, write_data,
            write_norms, filepath,
        )
    elif scenario == "capabilities":
        return CapabilitiesHarvest(
            society_mix, NUM_AGENTS, num_start_berries, NUM_ALLOCATIONS, agent_type, max_width,
            max_height, max_episodes, max_days, training, checkpoint_path, write_data,
            write_norms, filepath,
        )
    raise ValueError(f"unknown scenario {scenario}")


def make_training_model(tmp_path, agent_type, max_days=10, max_episodes=2, society_mix="homogeneous"):
    """basic_harvest is the only scenario ever actually trained on (run.py's "train" CLI option
    hardcodes scenario="basic") -- this mirrors that."""
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


def make_scenario_model(tmp_path, scenario, agent_type, max_days=10, max_episodes=2, society_mix="homogeneous"):
    """
    Builds a model for the given scenario/agent_type the way it is actually used in practice:
    basic_harvest trains (training=True); colours/allotment/capabilities are test-only
    (training=False), evaluating a policy pretrained on basic_harvest. Constructing
    colours/allotment/capabilities with training=True is not a supported configuration -- in
    particular, allotment agents' berry search only respects their own allotment boundary when
    training=False, so training=True can raise NoPathFound.
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
        False, False, f"test_{scenario}_{agent_type}", max_episodes, max_days,
    )

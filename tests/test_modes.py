import hashlib
import itertools
import os

import numpy as np
import pytest

from src.agent.dqn.n_network import NNetwork
from src.agent.modules.dqn_decision_module import DQNDecisionModule
from src.checkpoints import (AGENT_TYPES_FILE, RUN_OPTIONS, TRAINED_DIR, prepare_training_checkpoints, record_agent_types,
                             recorded_agent_types, resolve_checkpoint, society_checkpoint_path, loading_checkpoint_path,
                             training_checkpoint_name, training_checkpoint_path)
from src.harvest_exception import CheckpointExistsException, CheckpointMismatchException, CheckpointNotFoundException
from src.scenarios.allotment_harvest import AllotmentHarvest
from tests.conftest import NUM_AGENTS, SCENARIOS, _build, _make_pretrained_checkpoint, long_lived

REPO_CHECKPOINTS = os.path.abspath("data/model_variables") + "/"
PAPER_CHECKPOINTS = REPO_CHECKPOINTS + "200_days/4_agents/"
SEASON_LENGTH = 2

needs_paper_checkpoints = pytest.mark.skipif(not os.path.isdir(PAPER_CHECKPOINTS), reason="needs the trained 200_days checkpoints")


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """HarvestModel writes to the relative path data/results/current_run -- keep that inside tmp_path."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope="module")
def basic_checkpoint(tmp_path_factory):
    return _make_pretrained_checkpoint(tmp_path_factory.mktemp("modes"), "baseline")


def build(scenario, training, checkpoint_path, seasons=False, max_days=5, max_episodes=2, agent_type="baseline", health=3.0, **kwargs):
    return long_lived(_build(scenario, "homogeneous", NUM_AGENTS * 3, agent_type, training, checkpoint_path, False, False, scenario,
                             max_episodes, max_days, seed=42, seasons=seasons, season_length=SEASON_LENGTH, **kwargs), health)


def agents(model):
    return sorted((a for a in model.schedule.agents if a.agent_type != "berry"), key=lambda a: a.unique_id)


def network_weights(model):
    return [[w.copy() for w in agent.decision_module.q_network.dqn.get_weights()] for agent in agents(model)]


def same_weights(first, second):
    return all(np.array_equal(a, b) for x, y in zip(first, second) for a, b in zip(x, y))


def directory_hash(path):
    digest = hashlib.sha256()
    for root, _, files in sorted(os.walk(path)):
        for name in sorted(files):
            full = os.path.join(root, name)
            digest.update(full.encode())
            with open(full, "rb") as file:
                digest.update(file.read())
    return digest.hexdigest()


@pytest.mark.parametrize("seasons", [False, True])
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("training", [True, False], ids=["train", "test"])
def test_every_scenario_runs_in_either_mode_with_or_without_seasons(in_tmp_dir, basic_checkpoint, training, scenario, seasons):
    checkpoint_path = str(in_tmp_dir / "checkpoints") + "/" if training else basic_checkpoint
    model = build(scenario, training, checkpoint_path, seasons=seasons, max_episodes=3)
    while model.episode <= 3:
        model.step()
    assert model.episode == 4
    if training:
        # saved at the end of every episode
        for agent in agents(model):
            assert os.path.exists(agent.decision_module.q_checkpoint_path) and os.path.exists(agent.decision_module.target_checkpoint_path)
    else:
        assert model.epsilon == 0 and all(agent.get_epsilon() == 0 for agent in agents(model))


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_training_updates_the_networks_and_decays_epsilon(in_tmp_dir, scenario):
    model = build(scenario, True, str(in_tmp_dir / "checkpoints") + "/", max_days=30, max_episodes=3, health=5.0)
    assert model.epsilon == 1.0
    before = network_weights(model)
    epsilons = {}
    while model.episode <= 3:
        episode = model.episode
        model.step()
        # every agent explores with the same epsilon throughout an episode
        epsilons.setdefault(episode, set()).update(agent.get_epsilon() for agent in agents(model) if model.episode == episode)
    assert not same_weights(before, network_weights(model))
    # epsilon is 1 (fully exploring) for all of episode 1, then decays from it episode by episode
    assert {episode: [pytest.approx(e) for e in values] for episode, values in epsilons.items()} == {
        episode: [pytest.approx(np.exp(-0.001 * (episode - 1)))] for episode in [1, 2, 3]}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_testing_neither_learns_nor_explores_nor_saves(in_tmp_dir, basic_checkpoint, scenario, monkeypatch):
    checkpoints_before = directory_hash(basic_checkpoint)
    model = build(scenario, False, basic_checkpoint, seasons=True, max_days=30, max_episodes=2)
    before = network_weights(model)

    def no_learning(*args):
        raise AssertionError("learn() was called in test mode")
    monkeypatch.setattr(DQNDecisionModule, "learn", no_learning)
    while model.episode <= 2:
        model.step()
        assert all(agent.get_epsilon() == 0 for agent in agents(model))
    assert same_weights(before, network_weights(model))
    assert directory_hash(basic_checkpoint) == checkpoints_before
    assert all(len(buffer["s"]) == 0 for buffer in model.shared_replay_buffers.values())


@pytest.fixture(scope="module")
def trained_in_each_scenario(tmp_path_factory):
    """A checkpoint from a short training run in each scenario (with seasons), keyed by scenario."""
    paths = {}
    for scenario in SCENARIOS:
        directory = tmp_path_factory.mktemp(f"trained_{scenario}")
        cwd = os.getcwd()
        os.chdir(directory)
        try:
            checkpoint_path = str(directory / "checkpoints") + "/"
            model = build(scenario, True, checkpoint_path, seasons=True, max_episodes=1)
            while model.episode <= 1:
                model.step()
            paths[scenario] = (checkpoint_path, network_weights(model))
        finally:
            os.chdir(cwd)
    return paths


@pytest.mark.parametrize("tested_in", SCENARIOS)
@pytest.mark.parametrize("trained_in", SCENARIOS)
def test_a_checkpoint_trained_in_any_scenario_tests_in_every_scenario(in_tmp_dir, trained_in_each_scenario, trained_in, tested_in):
    checkpoint_path, trained_weights = trained_in_each_scenario[trained_in]
    model = build(tested_in, False, checkpoint_path, seasons=True, max_episodes=1)
    assert same_weights(network_weights(model), trained_weights)
    while model.episode <= 1:
        model.step()
    assert model.episode == 2


def test_observations_and_actions_are_the_same_in_every_scenario_with_or_without_seasons(in_tmp_dir, basic_checkpoint):
    shapes = set()
    for scenario, seasons in itertools.product(SCENARIOS, [False, True]):
        model = build(scenario, False, basic_checkpoint, seasons=seasons)
        model.step()
        shapes.add(tuple((agent.observe().shape, tuple(agent.actions), agent.decision_module.n_features) for agent in agents(model)))
    assert len(shapes) == 1
    (shape, actions, n_features), *_ = next(iter(shapes))
    assert shape == (NUM_AGENTS + 3,) == (n_features,)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_seeding_and_season_streams_are_the_same_in_both_modes(in_tmp_dir, basic_checkpoint, scenario):
    def record(model):
        start = sorted((b.unique_id, b.pos) for b in model.berries) + [(a.unique_id, a.pos) for a in agents(model)]
        qualities = [model._current_season_quality()]
        while model.episode <= 2:
            model.step()
            if model.day > 0 and model.day % SEASON_LENGTH == 0:
                qualities.append(model._current_season_quality())
        return start, qualities
    trained = record(build(scenario, True, str(in_tmp_dir / "checkpoints") + "/", seasons=True, max_days=10))
    tested = record(build(scenario, False, basic_checkpoint, seasons=True, max_days=10))
    assert trained == tested


def test_checkpoint_names_are_unique_across_settings():
    settings = list(itertools.product(SCENARIOS, [False, True], [25, 10], [1, 2, 3], ["default", "all 1"], [50, 200], [42, 7]))
    names = {}
    for scenario, seasons, season_length, num_allocations, delay_choice, max_days, seed in settings:
        delays = None
        if scenario == "allotment":
            delays = AllotmentHarvest.default_regrowth_delays(num_allocations) if delay_choice == "default" else [1] * num_allocations
        name = training_checkpoint_name(scenario, seasons, season_length, num_allocations, delays, max_days, seed)
        # settings that make no difference (season length without seasons, allocations in basic/colours, delays outside allotment)
        # give the same name
        key = (scenario, seasons, season_length if seasons else None, num_allocations if scenario in ["capabilities", "allotment"] else None,
               tuple(delays) if delays is not None else None, max_days, seed)
        names.setdefault(name, set()).add(key)
    assert all(len(keys) == 1 for keys in names.values())
    assert len(names) == len({key for keys in names.values() for key in keys})
    assert training_checkpoint_name("allotment", True, 25, 2, [0, 2], 200, 42) == "allotment_seasons25_2allotments_regrow0-2_200days_seed42"
    with pytest.raises(ValueError, match="regrowth delays"):
        training_checkpoint_name("allotment", True, 25, 2, None, 200, 42)
    assert training_checkpoint_name("basic", False, 25, 1, None, 200, 42) == "basic_200days_seed42"
    # trained checkpoints only ever go under trained/, so never into the checkpoints that already exist
    for name in names:
        path = training_checkpoint_path(name, NUM_AGENTS)
        assert path.startswith(TRAINED_DIR) and path.endswith(f"/{NUM_AGENTS}_agents/")
        assert not any(path.startswith(f"data/model_variables/{run}/") for run in RUN_OPTIONS)


@needs_paper_checkpoints
def test_training_never_writes_into_existing_checkpoints(in_tmp_dir):
    before = directory_hash(REPO_CHECKPOINTS)
    with pytest.raises(CheckpointExistsException, match="already exists"):
        build("basic", True, PAPER_CHECKPOINTS)
    with pytest.raises(CheckpointExistsException):
        prepare_training_checkpoints(PAPER_CHECKPOINTS, ["baseline"], {})
    assert directory_hash(REPO_CHECKPOINTS) == before


def test_a_second_training_run_into_the_same_checkpoints_is_refused(in_tmp_dir):
    checkpoint_path = str(in_tmp_dir / "trained" / "basic_5days_seed42" / f"{NUM_AGENTS}_agents") + "/"
    prepare_training_checkpoints(checkpoint_path, ["baseline"], {"seed": 42})
    model = build("basic", True, checkpoint_path, max_episodes=1)
    while model.episode <= 1:
        model.step()
    saved = directory_hash(checkpoint_path)
    with pytest.raises(CheckpointExistsException):
        prepare_training_checkpoints(checkpoint_path, ["baseline"], {"seed": 42})
    with pytest.raises(CheckpointExistsException):
        build("basic", True, checkpoint_path)
    assert directory_hash(checkpoint_path) == saved
    # another agent type can be trained alongside, with the same settings
    prepare_training_checkpoints(checkpoint_path, ["utilitarian"], {"seed": 42})
    with pytest.raises(ValueError, match="different settings"):
        prepare_training_checkpoints(checkpoint_path, ["maximin"], {"seed": 7})


def test_checkpoints_are_chosen_by_run_trained_name_or_path(in_tmp_dir):
    trained_dir = str(in_tmp_dir / "trained") + "/"
    os.makedirs(trained_dir + "colours_200days_seed42")
    assert resolve_checkpoint("") == "data/model_variables/200_days/"
    assert resolve_checkpoint("50_days") == "data/model_variables/50_days/"
    assert resolve_checkpoint("colours_200days_seed42", trained_dir=trained_dir) == trained_dir + "colours_200days_seed42/"
    assert resolve_checkpoint(trained_dir) == trained_dir
    assert resolve_checkpoint("not_a_checkpoint", trained_dir=trained_dir) is None


@needs_paper_checkpoints
def test_existing_checkpoints_still_load_with_their_names(in_tmp_dir):
    for scenario in SCENARIOS:
        model = build(scenario, False, PAPER_CHECKPOINTS, agent_type="utilitarian")
        model.step()


def test_a_checkpoint_for_another_number_of_agents_fails_clearly(in_tmp_dir, basic_checkpoint, monkeypatch):
    monkeypatch.setattr("tests.conftest.NUM_AGENTS", 6)
    with pytest.raises(CheckpointMismatchException, match=r"trained with 4 agents \(7 inputs, 5 actions\), but this run has 6 agents \(9 inputs, 7 actions\)"):
        _build("colours", "homogeneous", 18, "baseline", False, basic_checkpoint, False, False, "colours", 1, 5)


def test_a_checkpoint_with_another_input_size_fails_clearly(in_tmp_dir):
    checkpoint_path = str(in_tmp_dir / "odd") + "/"
    network = NNetwork(NUM_AGENTS + 1, (NUM_AGENTS + 5,), 128)
    network(np.zeros((1, NUM_AGENTS + 5), dtype="float32"))
    os.makedirs(checkpoint_path + "baseline/agent_0")
    network.save(checkpoint_path + "baseline/agent_0/q_model_variables.keras")
    with pytest.raises(CheckpointMismatchException, match=r"\(9 inputs, 5 actions\), but this run has 4 agents \(7 inputs, 5 actions\)"):
        build("basic", False, checkpoint_path)


def test_a_missing_checkpoint_fails_clearly(in_tmp_dir):
    with pytest.raises(CheckpointNotFoundException, match="No checkpoint at .*baseline/agent_0/q_model_variables.keras"):
        build("colours", False, str(in_tmp_dir / "empty") + "/")


# the agent type of each agent id in a mixed society of utilitarian and baseline agents
MIXED_TYPES = {
    "basic": ["utilitarian", "utilitarian", "baseline", "baseline"],
    "colours": ["utilitarian", "utilitarian", "baseline", "baseline"],
    "capabilities": ["utilitarian", "utilitarian", "baseline", "baseline"],
    "allotment": ["utilitarian", "baseline", "utilitarian", "baseline"],
    "allotment-separate": ["utilitarian", "utilitarian", "baseline", "baseline"],
}


def build_mixed(layout, training, checkpoint_path, seasons=True, max_episodes=1):
    scenario = layout.split("-")[0]
    kwargs = {"mixed_allotments": "separate"} if layout == "allotment-separate" else {}
    return long_lived(_build(scenario, "heterogeneous", NUM_AGENTS * 3, "utilitarian", training, checkpoint_path, False, False, scenario,
                             max_episodes, 5, seed=42, seasons=seasons, season_length=SEASON_LENGTH, **kwargs))


def train_mixed(layout, root):
    """Trains a mixed society for an episode as run.py does, returning the folder its checkpoints are saved in."""
    society_path = society_checkpoint_path(root, "heterogeneous", "utilitarian")
    model = build_mixed(layout, True, society_path)
    record_agent_types(society_path, model.get_living_agents())
    while model.episode <= 1:
        model.step()
    return society_path, model


@pytest.mark.parametrize("layout", list(MIXED_TYPES))
def test_mixed_training_saves_each_agent_under_its_own_type(in_tmp_dir, layout):
    root = str(in_tmp_dir / "checkpoints") + "/"
    society_path, model = train_mixed(layout, root)
    assert society_path == root + "utilitarian_with_baseline/"
    types = MIXED_TYPES[layout]
    assert [agent.agent_type for agent in agents(model)] == types
    # every agent's networks are saved under its own type, and the recorded types say which agent is which
    expected = {f"{agent_type}/agent_{agent_id}" for agent_id, agent_type in enumerate(types)}
    saved = {os.path.relpath(os.path.dirname(path), society_path)
             for path in (os.path.join(r, f) for r, _, files in os.walk(society_path) for f in files) if path.endswith("q_model_variables.keras")}
    assert saved == expected
    assert sorted(os.listdir(society_path)) == sorted({"baseline", "utilitarian", AGENT_TYPES_FILE})
    assert recorded_agent_types(society_path) == {f"agent_{agent_id}": agent_type for agent_id, agent_type in enumerate(types)}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_homogeneous_society_shares_one_replay_buffer(in_tmp_dir, scenario):
    model = build(scenario, True, str(in_tmp_dir / "checkpoints") + "/", max_episodes=1)
    while model.episode <= 1:
        model.step()
    assert list(model.shared_replay_buffers) == ["baseline"]
    buffer = model.shared_replay_buffers["baseline"]
    for agent in agents(model):
        assert agent.decision_module.q_network.experience is buffer and agent.decision_module.target_network.experience is buffer
    # every agent adds its experiences to the one buffer
    assert len(buffer["s"]) > 0


@pytest.mark.parametrize("layout", ["colours", "allotment"])
def test_mixed_training_keeps_each_types_experiences_apart(in_tmp_dir, layout):
    _, model = train_mixed(layout, str(in_tmp_dir / "checkpoints") + "/")
    assert set(model.shared_replay_buffers) == {"utilitarian", "baseline"}
    for agent in agents(model):
        buffer = model.shared_replay_buffers[agent.agent_type]
        assert agent.decision_module.q_network.experience is buffer and agent.decision_module.target_network.experience is buffer
    assert all(len(buffer["s"]) > 0 for buffer in model.shared_replay_buffers.values())
    assert model.shared_replay_buffers["utilitarian"] is not model.shared_replay_buffers["baseline"]


@pytest.mark.parametrize("layout", list(MIXED_TYPES))
def test_a_mixed_society_trained_together_tests_together(in_tmp_dir, layout):
    root = str(in_tmp_dir / "checkpoints") + "/"
    society_path, trained = train_mixed(layout, root)
    assert loading_checkpoint_path(root, "heterogeneous", "utilitarian") == society_path
    model = build_mixed(layout, False, loading_checkpoint_path(root, "heterogeneous", "utilitarian"))
    assert [agent.agent_type for agent in agents(model)] == MIXED_TYPES[layout]
    assert same_weights(network_weights(model), network_weights(trained))
    while model.episode <= 1:
        model.step()


def test_a_mixed_society_tested_with_other_agent_types_fails_clearly(in_tmp_dir):
    # trained split between allotments (baseline agents are 1 and 3), tested in colours (baseline agents are 2 and 3)
    root = str(in_tmp_dir / "checkpoints") + "/"
    society_path, _ = train_mixed("allotment", root)
    with pytest.raises(CheckpointNotFoundException, match=r"utilitarian/agent_1/.*trained together as .*'agent_1': 'baseline'"):
        build_mixed("colours", False, loading_checkpoint_path(root, "heterogeneous", "utilitarian"))


@needs_paper_checkpoints
def test_a_mixed_society_without_mixed_checkpoints_tests_each_types_own_agents(in_tmp_dir):
    # the paper's agents were each trained in a society of their own type, so a mixed society loads each type's own folder, as before
    assert loading_checkpoint_path(PAPER_CHECKPOINTS, "heterogeneous", "utilitarian") == PAPER_CHECKPOINTS
    model = build_mixed("colours", False, PAPER_CHECKPOINTS)
    model.step()


def test_mixed_training_never_overwrites_and_names_say_mixed(in_tmp_dir):
    root = str(in_tmp_dir / "checkpoints") + "/"
    prepare_training_checkpoints(root, ["utilitarian"], {"seed": 42}, "heterogeneous")
    train_mixed("colours", root)
    with pytest.raises(CheckpointExistsException, match="utilitarian_with_baseline"):
        prepare_training_checkpoints(root, ["utilitarian", "maximin"], {"seed": 42}, "heterogeneous")
    with pytest.raises(CheckpointExistsException):
        build_mixed("colours", True, society_checkpoint_path(root, "heterogeneous", "utilitarian"))
    # another agent type with baseline agents has its own folder, so it can train alongside
    prepare_training_checkpoints(root, ["maximin"], {"seed": 42}, "heterogeneous")
    names = {training_checkpoint_name("allotment", False, 25, 2, [0, 2], 200, 42, society_mix, layout)
             for society_mix in ["homogeneous", "heterogeneous"] for layout in ["split", "separate"]}
    assert names == {"allotment_2allotments_regrow0-2_200days_seed42", "allotment_2allotments_regrow0-2_mixed_200days_seed42",
                     "allotment_2allotments_regrow0-2_mixed-separate_200days_seed42"}
    assert training_checkpoint_name("colours", False, 25, 1, None, 200, 42, "heterogeneous", "separate") == "colours_mixed_200days_seed42"

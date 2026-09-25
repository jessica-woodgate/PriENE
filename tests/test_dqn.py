import keras
import numpy as np

from src.agent.dqn.dqn import DQN
from src.agent.dqn.n_network import NNetwork


ACTIONS = ["move", "eat", "throw_0", "throw_1"]
N_FEATURES = 5


def make_dqn(training=True, checkpoint_path="/tmp/unused_checkpoint"):
    return DQN(ACTIONS, (N_FEATURES,), training, checkpoint_path=checkpoint_path)


# --- NNetwork ---

def test_nnetwork_forward_pass_output_shape():
    net = NNetwork(n_actions=4, n_features=(5,), hidden_units=8)
    inputs = np.zeros((1, 5), dtype="float32")
    output = net(inputs)
    assert output.shape == (1, 4)


def test_nnetwork_get_config_round_trip():
    net = NNetwork(n_actions=4, n_features=(5,), hidden_units=8)
    config = net.get_config()
    assert config["n_actions"] == 4
    assert config["n_features"] == (5,)
    assert config["hidden_units"] == 8
    rebuilt = NNetwork.from_config(config)
    assert rebuilt.n_actions == 4
    assert rebuilt.n_features == (5,)
    assert rebuilt.hidden_units == 8


def test_nnetwork_save_and_load_round_trip(tmp_path):
    """
    The whole training pipeline depends on NNetwork's keras serialization (via
    @register_keras_serializable + get_config/from_config) actually working, since checkpoints
    are saved/loaded through exactly this path in DQNDecisionModule. Exercise it directly.
    """
    net = NNetwork(n_actions=4, n_features=(5,), hidden_units=8)
    inputs = np.random.rand(1, 5).astype("float32")
    net(inputs)  # build the model's variables via a forward pass before saving
    path = tmp_path / "model.keras"
    net.save(str(path))
    loaded = keras.models.load_model(str(path), compile=True, custom_objects={"NNetwork": NNetwork})
    original_output = net(inputs).numpy()
    loaded_output = loaded(inputs).numpy()
    np.testing.assert_allclose(original_output, loaded_output, rtol=1e-5)


# --- DQN ---

def test_dqn_choose_action_epsilon_1_always_explores_within_valid_range():
    dqn = make_dqn()
    observation = np.zeros(N_FEATURES)
    actions_seen = {dqn.choose_action(observation, epsilon=1.0) for _ in range(50)}
    assert actions_seen.issubset(set(range(len(ACTIONS))))


def test_dqn_choose_action_epsilon_0_is_deterministic():
    dqn = make_dqn()
    observation = np.random.rand(N_FEATURES)
    action1 = dqn.choose_action(observation, epsilon=0.0)
    action2 = dqn.choose_action(observation, epsilon=0.0)
    assert action1 == action2
    assert 0 <= action1 < len(ACTIONS)


def test_dqn_predict_output_shape():
    dqn = make_dqn()
    observation = np.zeros(N_FEATURES)
    output = dqn.predict(np.atleast_2d(observation.astype("float32")))
    assert output.shape == (1, len(ACTIONS))


def test_dqn_add_experience_appends():
    dqn = make_dqn()
    dqn.add_experience({"s": [1, 2], "a": 0, "r": 1.0, "s_": [1, 3], "done": False})
    assert len(dqn.experience["s"]) == 1
    assert dqn.experience["a"] == [0]


def test_dqn_add_experience_caps_at_max_experiences():
    dqn = make_dqn()
    dqn.max_experiences = 3
    for i in range(5):
        dqn.add_experience({"s": [i], "a": i, "r": float(i), "s_": [i + 1], "done": False})
    assert len(dqn.experience["s"]) == 3
    # oldest entries should have been dropped, keeping only the most recent ones
    assert dqn.experience["a"] == [2, 3, 4]


def test_dqn_train_returns_zero_below_min_experiences():
    dqn = make_dqn()
    dqn.min_experiences = 100
    target_net = make_dqn()
    assert dqn.train(target_net) == 0


def test_dqn_copy_weights_makes_networks_produce_identical_output():
    dqn = make_dqn()
    target = make_dqn()
    observation = np.random.rand(1, N_FEATURES).astype("float32")
    target.copy_weights(dqn)
    q_output = dqn.predict(observation).numpy()
    target_output = target.predict(observation).numpy()
    np.testing.assert_allclose(q_output, target_output, rtol=1e-5)

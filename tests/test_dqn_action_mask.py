"""
DQN's action mask (used to turn Fruit Market's grappling off): masked actions are never explored, never chosen greedily, and never the
best next action in a learning target; without a mask, choosing makes exactly the random draws it always has.
"""
import numpy as np
import tensorflow as tf
from src.agent.dqn.dqn import DQN

ACTIONS = list(range(6))
MASK = np.array([True, True, False, True, False, True])

def network(action_mask=None):
    tf.keras.utils.set_random_seed(3)
    return DQN(ACTIONS, (4,), True, action_mask=action_mask)

def test_masked_actions_are_never_explored():
    dqn = network(MASK)
    np.random.seed(0)
    chosen = {dqn.choose_action(np.zeros(4), 1.0) for _ in range(2000)}
    assert chosen == {0, 1, 3, 5}

def test_masked_actions_are_never_chosen_greedily_even_when_their_value_is_highest():
    dqn = network(MASK)
    dqn.predict = lambda inputs: np.array([[0.0, 1.0, 9.0, 2.0, 8.0, 0.5]])
    assert dqn.choose_action(np.zeros(4), 0.0) == 3

def test_the_learning_target_bootstraps_from_the_best_allowed_next_action():
    dqn, target = network(MASK), network()
    target.predict = lambda inputs: np.tile([0.0, 1.0, 9.0, 2.0, 8.0, 0.5], (len(inputs), 1))
    recorded = {}
    original_huber = tf.keras.losses.Huber
    for _ in range(dqn.min_experiences):
        dqn.add_experience({"s": np.zeros(4), "a": 0, "r": 0.0, "s_": np.zeros(4), "done": False})
    import src.agent.dqn.dqn as dqn_module
    class RecordingHuber:
        def __init__(self, delta):
            self.huber = original_huber(delta)
        def __call__(self, actual, predicted):
            recorded["actual"] = np.array(actual)
            return self.huber(actual, predicted)
    dqn_module.losses.Huber, saved = RecordingHuber, dqn_module.losses.Huber
    try:
        dqn.train(target)
    finally:
        dqn_module.losses.Huber = saved
    assert np.allclose(recorded["actual"], dqn.gamma * 2.0)

def test_without_a_mask_choosing_makes_the_same_random_draws_as_before():
    dqn = network()
    np.random.seed(5)
    chosen = [dqn.choose_action(np.zeros(4), 1.0) for _ in range(50)]
    np.random.seed(5)
    expected = []
    for _ in range(50):
        np.random.uniform(0, 1)
        expected.append(ACTIONS.index(np.random.choice(ACTIONS)))
    assert chosen == expected

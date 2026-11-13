"""
DQN's compiled training step (used by Fruit Market) learns exactly as the eager one does: from the same weights and the same sampled
batch, the same loss and the same updated weights, with or without an action mask.
"""
import numpy as np
import pytest
import tensorflow as tf
from src.agent.dqn.dqn import DQN

ACTIONS = list(range(5))
MASK = np.array([True, False, True, True, False])

def networks(compiled, action_mask, buffer):
    tf.keras.utils.set_random_seed(11)
    q = DQN(ACTIONS, (6,), True, shared_replay_buffer=buffer, action_mask=action_mask, compiled=compiled)
    target = DQN(ACTIONS, (6,), True, shared_replay_buffer=buffer, action_mask=action_mask)
    return q, target

def filled_buffer():
    rng = np.random.default_rng(0)
    buffer = {"s": [], "a": [], "r": [], "s_": [], "done": []}
    for _ in range(300):
        buffer["s"].append(rng.random(6))
        buffer["a"].append(int(rng.choice(np.flatnonzero(MASK))))
        buffer["r"].append(float(rng.normal()))
        buffer["s_"].append(rng.random(6))
        buffer["done"].append(bool(rng.random() < 0.1))
    return buffer

@pytest.mark.parametrize("action_mask", [None, MASK])
def test_compiled_training_matches_eager_training(action_mask):
    buffer = filled_buffer()
    results = []
    for compiled in (False, True):
        q, target = networks(compiled, action_mask, buffer)
        np.random.seed(3)
        losses = [float(np.asarray(q.train(target))) for _ in range(5)]
        results.append((losses, [v.numpy() for v in q.dqn.trainable_variables]))
    (eager_losses, eager_weights), (compiled_losses, compiled_weights) = results
    assert compiled_losses == pytest.approx(eager_losses, rel=1e-4)
    for eager, compiled in zip(eager_weights, compiled_weights):
        assert np.allclose(eager, compiled, rtol=1e-4, atol=1e-6)

def test_compiled_training_waits_for_enough_experiences_like_eager_training():
    q, target = networks(True, None, {"s": [], "a": [], "r": [], "s_": [], "done": []})
    assert q.train(target) == 0

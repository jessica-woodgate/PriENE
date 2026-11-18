"""
Is a compiled (tf.function) forward pass, and a compiled training step, bit-identical to the eager ones DQN uses, on this machine's
device (CPU here; run it on the cluster's GPU too)? Prints the time each takes.

    python benchmarks/check_tf_identity.py [n_features] [steps]

Training step: DQN.train as it is (eager), against the same algorithm with the targets computed exactly as DQN.train computes them (the
target network's forward pass, then numpy in float64) and only the gradient step (forward, Huber loss, gradients, Adam) compiled; the
two run from the same weights, replay buffer and replay stream, with the target network updated every 50 steps, and every weight, the
optimiser's state and every loss are compared. (DQN's existing compiled=True step computes the targets in float32 inside the graph,
which differs.)
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import keras
from keras import losses
from src.agent.dqn.dqn import DQN
from src import seeding

keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()
N_FEATURES = int(sys.argv[1]) if len(sys.argv) > 1 else 7
N_ACTIONS = N_FEATURES - 2
ACTIONS = ["move", "eat"] + [f"throw_{i}" for i in range(N_ACTIONS - 2)]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 600
rng = np.random.default_rng(0)
experience = {"s": [], "a": [], "r": [], "s_": [], "done": []}
for _ in range(2000):
    experience["s"].append(rng.uniform(0, 40, N_FEATURES))
    experience["a"].append(int(rng.integers(N_ACTIONS)))
    experience["r"].append(float(rng.choice([-1, -0.2, 0, 0.5, 1, 1.4, 0.9])))
    experience["s_"].append(rng.uniform(0, 40, N_FEATURES))
    experience["done"].append(bool(rng.random() < 0.05))


def pair():
    seeds = seeding.agent_seeds(42, 0, "weights", 6)
    q = DQN(ACTIONS, (N_FEATURES,), True, shared_replay_buffer=experience, weight_seeds=seeds[:3],
            exploration_rng=np.random.default_rng(1), replay_rng=np.random.default_rng(2))
    t = DQN(ACTIONS, (N_FEATURES,), True, shared_replay_buffer=experience, weight_seeds=seeds[3:])
    x = np.zeros((1, N_FEATURES), dtype="float32")
    q.dqn(x)
    t.dqn(x)
    return q, t


forwards, steps_cache = {}, {}


def forward(net, x):
    if id(net) not in forwards:
        forwards[id(net)] = tf.function(lambda inputs: net.dqn(inputs), input_signature=[tf.TensorSpec([None, N_FEATURES], tf.float32)])
    return forwards[id(net)](x)


def gradient_step(self, states, actions, actual_values):
    if id(self) not in steps_cache:
        variables = self.dqn.trainable_variables
        self.optimiser.build(variables)
        huber = losses.Huber(self.delta)

        def step(states, actions, actual_values):
            with tf.GradientTape() as tape:
                selected = tf.math.reduce_sum(self.dqn(states) * tf.one_hot(actions, self.n_actions), axis=1)
                loss = huber(actual_values, selected)
            self.optimiser.apply_gradients(zip(tape.gradient(loss, variables), variables))
            return loss
        steps_cache[id(self)] = tf.function(step, input_signature=[tf.TensorSpec([None, N_FEATURES], tf.float32), tf.TensorSpec([None], tf.int64),
                                                                   tf.TensorSpec([None], tf.float64)])
    return steps_cache[id(self)](states, actions, actual_values)


def compiled_train(self, target):
    """DQN.train, with the target network's forward pass and the gradient step compiled"""
    if len(self.experience["s"]) < self.min_experiences:
        return 0
    ids = self.replay_rng.integers(low=0, high=len(self.experience["s"]), size=self.batch_size)
    states = np.asarray([self.experience["s"][i] for i in ids])
    actions = np.asarray([self.experience["a"][i] for i in ids])
    rewards = np.asarray([self.experience["r"][i] for i in ids])
    states_next = np.asarray([self.experience["s_"][i] for i in ids])
    dones = np.asarray([self.experience["done"][i] for i in ids])
    value_next = np.max(forward(target, states_next.astype("float32")), axis=1)
    actual_values = np.where(dones, rewards, rewards + self.gamma * value_next)
    return gradient_step(self, states.astype("float32"), actions, actual_values)


def main():
    print("devices:", [d.name for d in tf.config.list_physical_devices()])
    results = {}
    for name in ["eager", "compiled"]:
        q, t = pair()
        losses_ = []
        start = time.perf_counter()
        for i in range(STEPS):
            losses_.append(float(np.asarray(q.train(t) if name == "eager" else compiled_train(q, t))))
            if (i + 1) % 50 == 0:
                t.copy_weights(q)
        results[name] = ([v.numpy() for v in q.dqn.trainable_variables] + [v.numpy() for v in q.optimiser.variables], losses_,
                         (time.perf_counter() - start) / STEPS * 1000)
    same = all(np.array_equal(a, b) for a, b in zip(results["eager"][0], results["compiled"][0]))
    print(f"training step ({N_FEATURES} features, {STEPS} steps): eager {results['eager'][2]:.2f} ms, compiled {results['compiled'][2]:.2f} ms; "
          f"weights and optimiser state identical: {same}; losses identical: {results['eager'][1] == results['compiled'][1]}")
    q, _ = pair()
    observations = [rng.uniform(0, 40, N_FEATURES) for _ in range(2000)]
    outputs, timings = {}, {}
    for name, f in [("eager", lambda x: q.dqn(x)), ("compiled", lambda x: forward(q, x))]:
        f(np.atleast_2d(observations[0].astype("float32")))
        start = time.perf_counter()
        outputs[name] = np.stack([np.asarray(f(np.atleast_2d(o.astype("float32"))))[0] for o in observations])
        timings[name] = (time.perf_counter() - start) / len(observations) * 1000
    print(f"forward pass (one observation): eager {timings['eager']:.3f} ms, compiled {timings['compiled']:.3f} ms; "
          f"bit-identical: {np.array_equal(outputs['eager'], outputs['compiled'])}")


if __name__ == "__main__":
    main()

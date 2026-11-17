"""
Every random draw in a run comes from one run seed, through a separate stream for each purpose, so no draw for one purpose shifts
another's (numpy's SeedSequence keeps derived streams independent):

    environment streams, reseeded at the start of every episode from SeedSequence([run_seed, episode, stream]), so an episode's
        environment doesn't depend on what agents did in earlier episodes, and societies run with the same seed see the same
        environment draws for as long as their actions coincide:
            placement -- the cells agents and berries are put in at the start of an episode
            respawn -- the cells foraged berries regrow in during an episode
            seasons -- season qualities and the berries they add or remove
            membership -- which agents are in which group (colour, capability or allotment) each episode
            activation -- the order agents act in each step (Mesa's model RNG, which RandomActivation shuffles with)
            commons_regrowth -- how much the commons regrows each step, and where
            fruit_market_substrate -- the seed Melting Pot's Fruit Market substrate is reset with
    run streams, drawn once per run from SeedSequence([run_seed, stream]):
            resources -- the Zipf fallback for starting resources (agent counts without a hand-picked distribution)
            mixed_society -- which agents are baseline in a mixed society
    agent streams, from SeedSequence([run_seed, agent_id, stream]) and carried on across episodes, as they are part of learning (a
    continued run restores them from its checkpoint, see DQNDecisionModule):
            weights -- each network layer's initial weights
            exploration -- epsilon-greedy exploration
            replay -- which experiences a learning step samples

A test run uses the test seed as its run seed, so every trained society faces the same test episodes. seed=None leaves a run unseeded.
"""
import numpy as np

ENVIRONMENT_STREAMS = {"placement": 1, "respawn": 2, "seasons": 3, "membership": 4, "activation": 5, "commons_regrowth": 6,
                       "fruit_market_substrate": 7}
RUN_STREAMS = {"resources": 101, "mixed_society": 102}
AGENT_STREAMS = {"weights": 201, "exploration": 202, "replay": 203}


def _sequence(entropy):
    return None if entropy is None else np.random.SeedSequence(entropy)


def episode_rng(seed, episode, stream):
    """An environment stream's generator for one episode"""
    return np.random.default_rng(None if seed is None else _sequence([seed, episode, ENVIRONMENT_STREAMS[stream]]))


def episode_seed(seed, episode, stream):
    """An environment stream for one episode as a single integer (for generators that take one, e.g. Mesa's and Melting Pot's)"""
    if seed is None:
        return None
    return int(_sequence([seed, episode, ENVIRONMENT_STREAMS[stream]]).generate_state(1)[0])


def run_rng(seed, stream):
    """A run stream's generator"""
    return np.random.default_rng(None if seed is None else _sequence([seed, RUN_STREAMS[stream]]))


def agent_rng(seed, agent_id, stream):
    """An agent stream's generator, carried on across episodes"""
    return np.random.default_rng(None if seed is None else _sequence([seed, agent_id, AGENT_STREAMS[stream]]))


def agent_seeds(seed, agent_id, stream, count):
    """count integer seeds from an agent stream (e.g. one per network layer's weight initialiser); None for an unseeded run"""
    if seed is None:
        return [None] * count
    return [int(s) for s in _sequence([seed, agent_id, AGENT_STREAMS[stream]]).generate_state(count)]

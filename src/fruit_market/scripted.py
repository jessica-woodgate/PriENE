"""
Scripted policies for the environment check (docs/fruit_market_sweep.md §2): random, autarky (harvest and eat its own fruit) and trader
(harvest its own fruit, offer it 1:1 for the other, go to partners, eat the other fruit). Each chooses one of the substrate's 25
actions from the state the agents' observations are built from (features.py), so it also checks that a good policy can be computed from
what the agents observe.
"""
import numpy as np
from . import substrate as fm
from .features import egocentric, nearest, ripe_trees, partners

POLICIES = ["random", "autarky", "trader"]
#the fair 1:1 offer giving one of a player's own fruit for one of the other: (apple -1, banana +1) or (apple +1, banana -1)
FAIR_OFFER = {"apple": fm.FIRST_OFFER_ACTION + fm.OFFER_QUANTITIES.index((-1, 1)),
              "banana": fm.FIRST_OFFER_ACTION + fm.OFFER_QUANTITIES.index((1, -1))}
EAT = {fruit: action for action, fruit in fm.EAT_ACTIONS.items()}

def step_towards(player, cell):
    """
    The movement action bringing player one cell closer to cell (forward first, then sideways, then back); NOOP if it is there
    """
    forward, right = egocentric((cell[0] - player.position[0], cell[1] - player.position[1]), player.orientation)
    if forward > 0:
        return fm.ACTION_NAMES.index("FORWARD")
    if right > 0:
        return fm.ACTION_NAMES.index("STEP_RIGHT")
    if right < 0:
        return fm.ACTION_NAMES.index("STEP_LEFT")
    if forward < 0:
        return fm.ACTION_NAMES.index("BACKWARD")
    return 0

def _eat_when_hungry(player, prefer):
    if player.hunger < fm.HUNGER_LIMIT:
        return None
    for fruit in prefer:
        if player.held(fruit) > 0:
            return EAT[fruit]
    return None

def _harvest_own(state, player):
    found = nearest(player.position, ripe_trees(state.trees, player.own_fruit))
    return 0 if found is None else step_towards(player, found[1])

def autarky(state, index, rng):
    player = state.players[index]
    eat = _eat_when_hungry(player, [player.own_fruit, player.other_fruit])
    return eat if eat is not None else _harvest_own(state, player)

def trader(state, index, rng):
    player = state.players[index]
    eat = _eat_when_hungry(player, [player.other_fruit, player.own_fruit])
    if eat is not None:
        return eat
    if player.held(player.own_fruit) > 0 and player.held(player.other_fruit) == 0:
        offer = fm.offer_of(FAIR_OFFER[player.own_fruit])
        if tuple(player.offer) != offer:
            return FAIR_OFFER[player.own_fruit]
        found = nearest(player.position, [p.position for p in partners(state, index)])
        if found is not None:
            return step_towards(player, found[1])
    return _harvest_own(state, player)

def random_policy(state, index, rng):
    return int(rng.integers(fm.NUM_ACTIONS))

POLICY_FUNCTIONS = {"random": random_policy, "autarky": autarky, "trader": trader}

class ScriptedDecisions:
    """
    Stands in for an agent's DQNDecisionModule: chooses by a scripted policy from the model's current state, and never learns
    """
    def __init__(self, policy, model, index, rng):
        self.policy = POLICY_FUNCTIONS[policy]
        self.model = model
        self.index = index
        self.rng = rng

    def choose_action(self, observation):
        return self.policy(self.model.state, self.index, self.rng)

    def learn(self, *args):
        pass

    def start_episode(self):
        pass

    def save_models(self, episode):
        pass

    def get_epsilon(self):
        return 0.0

    def get_mean_loss(self):
        return 0.0

def run_policy(policy, episodes, seed, folder_name="check"):
    """
    episodes Fruit Market episodes of 16 players following policy, through FruitMarketModel and the real substrate (in the working
    directory, which gets the model's reports); returns the episode report
    """
    import pandas as pd
    from .model import FruitMarketModel
    model = FruitMarketModel("homogeneous", fm.NUM_PLAYERS, "baseline", episodes, fm.EPISODE_LENGTH, True, f"{folder_name}_{policy}/", True,
                             False, filepath=f"check_{policy}", seed=seed)
    rng = np.random.default_rng(seed)
    for agent in model.schedule.agents:
        agent.decision_module = ScriptedDecisions(policy, model, agent.unique_id, rng)
    while model.episode <= episodes:
        model.step()
    model.backend.close()
    return pd.read_csv(f"data/results/current_run/fm_model_episode_reports_check_{policy}.csv")

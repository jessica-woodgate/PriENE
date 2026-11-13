"""
The Fruit Market substrate itself: Melting Pot's fruit_market__concentric_rivers, built exactly as Melting Pot builds it, plus read-only
observers (docs/fruit_market_design.md §2.9 and §5.0):
    - Melting Pot's own LocationObserver on each avatar, for POSITION and ORIENTATION;
    - PriENE's TreeStateObserver (WORLD.TREES) and HeldByObserver (HELD_BY), in lua/priene_trade/priene_components.lua, loaded through
      lua/priene_trade/init.lua, which is the trade level's init.lua with that one file added.
They only report; they change nothing in the substrate. Only this module imports Melting Pot (an optional dependency, see
requirements-fruit-market.txt).
"""
import copy
import os
import numpy as np
from .substrate import SUBSTRATE_NAME, ACTION_COMPONENTS, MAP_WIDTH, MAP_HEIGHT
from .state import PlayerState, FruitMarketState, Trade
from ..harvest_exception import HarvestException

LUA_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lua")
LEVEL_NAME = "priene_trade"
INDIVIDUAL_OBSERVATIONS = ["READY_TO_SHOOT", "STAMINA", "INVENTORY", "MY_OFFER", "HUNGER", "POSITION", "ORIENTATION", "HELD_BY"]
GLOBAL_OBSERVATIONS = ["WORLD.TREES"]
INSTALL_HINT = ("The fruit_market scenario needs Melting Pot, which isn't installed. Install it with\n"
                "    pip install -c constraints-prenv.txt -r requirements-fruit-market.txt\n"
                "    pip install --no-deps dm-meltingpot==2.4.1")

class MeltingPotMissing(HarvestException):
    def __init__(self):
        super().__init__(INSTALL_HINT)

def _meltingpot():
    try:
        from meltingpot.configs import substrates as substrate_configs
        from meltingpot.utils.substrates import builder, substrate
        from meltingpot.utils.substrates.wrappers import (observables_wrapper, multiplayer_wrapper, discrete_action_wrapper,
                                                          collective_reward_wrapper)
    except ImportError as error:
        raise MeltingPotMissing() from error
    return substrate_configs, builder, substrate, observables_wrapper, multiplayer_wrapper, discrete_action_wrapper, collective_reward_wrapper

def _meltingpot_root():
    builder = _meltingpot()[1]
    return builder._MELTINGPOT_ROOT

def meltingpot_available():
    try:
        _meltingpot()
    except MeltingPotMissing:
        return False
    return True

def substrate_config():
    substrate_configs = _meltingpot()[0]
    return substrate_configs.get_config(SUBSTRATE_NAME)

def check_action_set(config):
    """
    The installed substrate's actions must be the 25 this package uses, in the same order (substrate.ACTION_COMPONENTS)
    """
    installed = tuple(dict(action) for action in config.action_set)
    if installed != ACTION_COMPONENTS:
        raise HarvestException("the installed Melting Pot's Fruit Market actions differ from the ones PriENE uses (src/fruit_market/substrate.py)")

def lab2d_settings(roles, with_observers=True):
    """
    The substrate's settings as Melting Pot builds them for roles; with_observers adds PriENE's read-only observers and level
    """
    config = substrate_config()
    settings = copy.deepcopy(dict(config.lab2d_settings_builder(roles=roles, config=config)))
    if not with_observers:
        return config, settings
    #dmlab2d opens the level relative to Melting Pot's root (the folder holding the meltingpot package), so ours is given relative to it
    settings["levelDirectory"] = os.path.relpath(LUA_DIRECTORY, _meltingpot_root())
    settings["levelName"] = LEVEL_NAME
    settings["simulation"]["scene"]["components"].append(
        {"component": "TreeStateObserver", "kwargs": {"width": MAP_WIDTH, "height": MAP_HEIGHT}})
    for game_object in settings["simulation"]["gameObjects"]:
        if game_object["name"] == "avatar":
            game_object["components"].append({"component": "LocationObserver", "kwargs": {"objectIsAvatar": True, "alsoReportOrientation": True}})
            game_object["components"].append({"component": "HeldByObserver"})
    return config, settings

def build(roles, seed, with_observers=True, individual_observations=None, global_observations=None):
    """
    A Melting Pot Substrate (built as meltingpot.utils.substrates.substrate.build_substrate does), seeded with seed (None = random)
    """
    _, builder, substrate, observables_wrapper, multiplayer_wrapper, discrete_action_wrapper, collective_reward_wrapper = _meltingpot()
    config, settings = lab2d_settings(roles, with_observers)
    check_action_set(config)
    env = builder.builder(settings, env_seed=seed)
    env = observables_wrapper.ObservablesWrapper(env)
    env = multiplayer_wrapper.Wrapper(env, individual_observation_names=individual_observations or INDIVIDUAL_OBSERVATIONS,
                                      global_observation_names=global_observations if global_observations is not None else GLOBAL_OBSERVATIONS)
    env = discrete_action_wrapper.Wrapper(env, action_table=config.action_set)
    env = collective_reward_wrapper.CollectiveRewardWrapper(env)
    return substrate.Substrate(env)

def parse_trade(event_values):
    """
    A Trade from a Melting Pot "trade" event's values: [b'dict', key, value, key, value, ...]; player indices are 1-based there
    """
    fields = {}
    items = list(event_values)[1:]
    for key, value in zip(items[0::2], items[1::2]):
        key = key.decode() if isinstance(key, bytes) else str(key)
        fields[key] = value.decode() if isinstance(value, bytes) else value
    def number(key):
        return int(np.asarray(fields[key]).item())
    a_received = (number("player_a_traded_item_0"), number("player_a_traded_item_1"))
    b_received = (number("player_b_traded_item_0"), number("player_b_traded_item_1"))
    return Trade(number("player_a_index") - 1, number("player_b_index") - 1, a_received, b_received)

class MeltingPotBackend:
    """
    Runs the substrate for FruitMarketModel: reset(seed) builds a fresh substrate for an episode, step(actions) steps every player at once;
    both return a FruitMarketState
    """
    def __init__(self, roles):
        _meltingpot()
        self.roles = list(roles)
        self.substrate = None
        self.events = []

    def reset(self, seed):
        if self.substrate is not None:
            self.substrate.close()
        self.substrate = build(self.roles, seed)
        self.events = []
        self.substrate.observables().events.subscribe(on_next=self.events.append)
        timestep = self.substrate.reset()
        return self._state(timestep)

    def step(self, actions):
        self.events.clear()
        timestep = self.substrate.step([int(action) for action in actions])
        return self._state(timestep)

    def close(self):
        if self.substrate is not None:
            self.substrate.close()
            self.substrate = None

    def _state(self, timestep):
        players = []
        for index, (role, obs) in enumerate(zip(self.roles, timestep.observation)):
            held_by = int(np.asarray(obs["HELD_BY"]).item())
            position = np.asarray(obs["POSITION"])
            players.append(PlayerState(index, role, (int(position[0]), int(position[1])), int(np.asarray(obs["ORIENTATION"]).item()),
                                       tuple(int(n) for n in obs["INVENTORY"]), float(obs["HUNGER"]), float(obs["STAMINA"]),
                                       float(obs["READY_TO_SHOOT"]), tuple(int(n) for n in obs["MY_OFFER"]), None if held_by == 0 else held_by - 1))
        trades = [parse_trade(values) for name, values in self.events if name == "trade"]
        trees = np.asarray(timestep.observation[0]["WORLD.TREES"]).astype(int)
        rewards = [float(reward) for reward in timestep.reward]
        return FruitMarketState(players, trees, trades, rewards)

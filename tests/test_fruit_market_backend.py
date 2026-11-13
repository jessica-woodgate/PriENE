"""
The Fruit Market backend against the real substrate (skipped without Melting Pot): the substrate's facts PriENE relies on, that PriENE's
read-only observers don't change the substrate, reproducibility from a seed, and what the observers and trade events report.
"""
import os
import re
import numpy as np
import pytest

meltingpot = pytest.importorskip("meltingpot")
pytest.importorskip("dmlab2d")

from src.fruit_market import substrate as fm
from src.fruit_market.backend import lab2d_settings, MeltingPotBackend, check_action_set, LUA_DIRECTORY

def avatar_component(settings, name):
    avatar = next(o for o in settings["simulation"]["gameObjects"] if o["name"] == "avatar")
    return next(c for c in avatar["components"] if c["component"] == name).get("kwargs", {})

def test_the_installed_substrate_has_the_actions_and_parameters_priene_uses():
    config, settings = lab2d_settings(fm.DEFAULT_ROLES, with_observers=False)
    check_action_set(config)
    assert tuple(config.default_player_roles) == fm.DEFAULT_ROLES
    assert settings["maxEpisodeLengthFrames"] == fm.EPISODE_LENGTH
    assert avatar_component(settings, "PeriodicNeed")["delay"] == fm.HUNGER_DELAY
    taste = avatar_component(settings, "Taste")
    assert (taste["mostTastyReward"], taste["defaultReward"]) == (fm.MOST_TASTY_REWARD, fm.DEFAULT_TASTE_REWARD)
    assert avatar_component(settings, "Trading")["radius"] == fm.TRADE_RADIUS
    assert avatar_component(settings, "Stamina")["maxStamina"] == fm.MAX_STAMINA
    tree = next(c for c in settings["simulation"]["prefabs"]["potential_tree"]["components"] if c["component"] == "Harvestable")
    assert tree["kwargs"]["regrowthTime"] == fm.REGROWTH_TIME
    assert settings["simulation"]["map"].strip("\n").split("\n") == fm.MAP_ROWS

def lua_functions(source):
    return set(re.findall(r"^function \w+:(\w+)\(", source, flags=re.MULTILINE))

def test_priene_observers_only_report():
    """
    Melting Pot's substrate isn't reproducible from its seed (the same env_seed gives a different start each time it is built), so two
    runs can't be compared step by step. Instead: PriENE's components define observations and nothing else (no updaters, no
    reaction to contacts or hits, no random draws, no state changes), and Melting Pot's LocationObserver likewise.
    """
    with open(os.path.join(LUA_DIRECTORY, "priene_trade", "priene_components.lua")) as file:
        source = file.read()
    assert lua_functions(source) == {"__init__", "_trees", "reset", "addObservations"}
    for forbidden in ["require 'system.random'", "random:", "setState","registerUpdaters", "addReward", "moveAbs", "moveRel", "teleport", "setOrientation", ":add("]:
        assert forbidden not in source
    with open(os.path.join(os.path.dirname(meltingpot.__file__), "lua", "modules", "component_library.lua")) as file:
        library = file.read()
    location_observer = library[library.index("local LocationObserver"):library.index("local StochasticEpisodeEnding")]
    assert lua_functions(location_observer) == {"__init__", "addObservations"}

def test_priene_observers_add_only_observations_to_the_substrates_settings():
    _, original = lab2d_settings(fm.DEFAULT_ROLES, with_observers=False)
    _, observed = lab2d_settings(fm.DEFAULT_ROLES)
    added = [c["component"] for c in observed["simulation"]["scene"]["components"][len(original["simulation"]["scene"]["components"]):]]
    assert added == ["TreeStateObserver"]
    for before, after in zip(original["simulation"]["gameObjects"], observed["simulation"]["gameObjects"]):
        extra = [c["component"] for c in after["components"][len(before["components"]):]]
        assert after["components"][:len(before["components"])] == before["components"]
        assert extra == (["LocationObserver", "HeldByObserver"] if before["name"] == "avatar" else [])
    for key in original:
        if key not in ("simulation", "levelDirectory", "levelName"):
            assert observed[key] == original[key]
    assert observed["simulation"]["prefabs"] == original["simulation"]["prefabs"]
    assert observed["simulation"]["map"] == original["simulation"]["map"]

def test_players_start_on_spawn_points_and_trees_start_ripe_on_potential_tree_tiles():
    backend = MeltingPotBackend(fm.DEFAULT_ROLES)
    state = backend.reset(2)
    assert all(fm.MAP_ROWS[p.position[1]][p.position[0]] == "P" for p in state.players)
    assert all(p.orientation in (0, 1, 2, 3) and p.held_by is None for p in state.players)
    ys, xs = np.nonzero(state.trees)
    assert all(fm.MAP_ROWS[y][x] in "~P" for y, x in zip(ys, xs))
    assert set(np.unique(state.trees)) <= {fm.NO_TREE, fm.RIPE_APPLE, fm.RIPE_BANANA}
    backend.close()

def test_trade_events_match_the_traders_inventory_changes():
    backend = MeltingPotBackend(fm.DEFAULT_ROLES)
    state = backend.reset(7)
    rng = np.random.default_rng(1)
    trades = 0
    for _ in range(400):
        actions = [int(rng.choice([1, 2, 3, 4, 5, 6])) if rng.random() < 0.5 else (17 if p.role == "apple_farmer" else 21) for p in state.players]
        before, state = state, backend.step(actions)
        for trade in state.trades:
            trades += 1
            for index in (trade.player_a, trade.player_b):
                change = tuple(after - prior for after, prior in zip(state.players[index].inventory, before.players[index].inventory))
                #what the trade doesn't explain is a harvest (2 of one fruit) in the same step, if anything
                rest = tuple(c - r for c, r in zip(change, state.received(index)))
                assert rest in [(0, 0), (2, 0), (0, 2)]
    backend.close()
    assert trades > 0

def test_a_hold_is_reported_on_the_held_player():
    backend = MeltingPotBackend(fm.DEFAULT_ROLES)
    state = backend.reset(4)
    held = 0
    for _ in range(100):
        state = backend.step([9] * len(fm.DEFAULT_ROLES))
        held += sum(p.held_by is not None for p in state.players)
        assert all(p.held_by is None or 0 <= p.held_by < len(fm.DEFAULT_ROLES) and p.held_by != p.index for p in state.players)
    backend.close()
    assert held > 0

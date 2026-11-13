"""
The environment check's scripted policies (src/fruit_market/scripted.py) choose what they say they do, from a hand-built state.
"""
import numpy as np
from src.fruit_market import substrate as fm
from src.fruit_market.state import PlayerState, FruitMarketState
from src.fruit_market.scripted import step_towards, autarky, trader, FAIR_OFFER

RNG = np.random.default_rng(0)

def player(index, role="apple_farmer", position=(10, 10), orientation=0, inventory=(0, 0), hunger=0.0, offer=(0, 0)):
    return PlayerState(index, role, position, orientation, inventory, hunger, 1.0, 1.0, offer)

def state(players, ripe=()):
    trees = np.zeros((fm.MAP_HEIGHT, fm.MAP_WIDTH), dtype=int)
    for (x, y), code in ripe:
        trees[y, x] = code
    return FruitMarketState(players, trees, [], [0.0] * len(players))

def test_steps_go_forward_then_sideways_then_back_as_the_player_faces():
    facing_north = player(0)
    assert fm.ACTION_NAMES[step_towards(facing_north, (10, 5))] == "FORWARD"
    assert fm.ACTION_NAMES[step_towards(facing_north, (13, 10))] == "STEP_RIGHT"
    assert fm.ACTION_NAMES[step_towards(facing_north, (8, 10))] == "STEP_LEFT"
    assert fm.ACTION_NAMES[step_towards(facing_north, (10, 12))] == "BACKWARD"
    assert fm.ACTION_NAMES[step_towards(player(0, orientation=1), (13, 10))] == "FORWARD"
    assert step_towards(facing_north, (10, 10)) == 0

def test_autarky_walks_to_its_own_fruit_and_eats_it_when_hungry():
    s = state([player(0), player(1, position=(20, 20))], ripe=[((10, 7), fm.RIPE_APPLE), ((10, 11), fm.RIPE_BANANA)])
    assert fm.ACTION_NAMES[autarky(s, 0, RNG)] == "FORWARD"
    hungry = state([player(0, inventory=(1, 1), hunger=1.0), player(1)])
    assert fm.ACTION_NAMES[autarky(hungry, 0, RNG)] == "EAT_APPLE"
    assert fm.ACTION_NAMES[autarky(state([player(0, inventory=(1, 1), hunger=0.5), player(1)]), 0, RNG)] != "EAT_APPLE"

def test_the_trader_offers_its_fruit_fairly_goes_to_a_partner_and_eats_the_other_fruit():
    holding = state([player(0, inventory=(2, 0)), player(1, role="banana_farmer", position=(10, 4), inventory=(0, 2), offer=(1, -1))])
    assert trader(holding, 0, RNG) == FAIR_OFFER["apple"]
    assert fm.offer_of(FAIR_OFFER["apple"]) == (-1, 1) and fm.offer_of(FAIR_OFFER["banana"]) == (1, -1)
    offered = state([player(0, inventory=(2, 0), offer=(-1, 1)), player(1, role="banana_farmer", position=(10, 4), inventory=(0, 2), offer=(1, -1))])
    assert fm.ACTION_NAMES[trader(offered, 0, RNG)] == "FORWARD"
    hungry = state([player(0, inventory=(1, 1), hunger=1.2), player(1)])
    assert fm.ACTION_NAMES[trader(hungry, 0, RNG)] == "EAT_BANANA"

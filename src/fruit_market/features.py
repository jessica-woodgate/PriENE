"""
Scalar observations for Fruit Market agents (docs/fruit_market_design.md §4), computed from a FruitMarketState. Directions are
egocentric, (forward, right) as the player faces, because the substrate's movement is egocentric.
"""
import numpy as np
from .substrate import (FORWARD, RIGHT, RIVER, MAP_WIDTH, MAP_HEIGHT, MAX_DISTANCE, RIPE_CODE, TRADE_RADIUS, MAX_OFFER_QUANTITY,
                        VIEW_SIDE, VIEW_FORWARD, VIEW_BACKWARD, FRUITS, MOST_TASTY_REWARD)
from .well_being import STEPS_PER_FRUIT

OBSERVATION_MODES = ["global", "view"]
FRUIT_SCALE = 10
COUNT_SCALE = 15
HUNGRY_TIME_SCALE = 500
#others' well-being is divided by this: a full stomach plus 10 fruit, in steps (times the largest fruit value in preference mode)
WELL_BEING_SCALE = STEPS_PER_FRUIT * (1 + FRUIT_SCALE)
#names of the features before others' well-being, in order (FEATURE_INDEX gives each one's index)
FEATURE_NAMES = ["hunger", "hungry_time", "stamina", "own_fruit", "other_fruit", "role", "ready_to_shoot",
                 "facing_n", "facing_e", "facing_s", "facing_w", "offer_own", "offer_other",
                 "own_tree_distance", "own_tree_forward", "own_tree_right", "other_tree_distance", "other_tree_forward", "other_tree_right",
                 "partner_distance", "partner_forward", "partner_right", "compatible_offers_nearby", "on_river", "river_ahead", "held",
                 "hungry_players_nearby"]
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

def observation_size(num_players):
    return len(FEATURE_NAMES) + num_players - 1

def egocentric(offset, orientation):
    """
    (forward, right) components of a map offset (dx, dy) for a player facing orientation
    """
    fx, fy = FORWARD[orientation]
    rx, ry = RIGHT[orientation]
    return offset[0] * fx + offset[1] * fy, offset[0] * rx + offset[1] * ry

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def within_radius(a, b, radius=TRADE_RADIUS):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 <= radius ** 2

def in_view(player, cell):
    forward, right = egocentric((cell[0] - player.position[0], cell[1] - player.position[1]), player.orientation)
    return -VIEW_BACKWARD <= forward <= VIEW_FORWARD and -VIEW_SIDE <= right <= VIEW_SIDE

def ripe_trees(trees, fruit):
    ys, xs = np.nonzero(trees == RIPE_CODE[fruit])
    return list(zip(xs.tolist(), ys.tolist()))

def nearest(position, cells):
    """
    The nearest of cells to position by path length (Manhattan; other players are ignored and rivers can be walked through), first in
    cells on ties; None if there are none
    """
    best = None
    for cell in cells:
        distance = manhattan(position, cell)
        if best is None or distance < best[0]:
            best = (distance, cell)
    return best

def is_valid(offer):
    return any(q > 0 for q in offer) and any(q < 0 for q in offer)

def _wants_covered(mine, theirs):
    #Trading:isCompatible: for everything I want, they offer to give at least as much
    return all(not (q > 0 and q + t > 0) for q, t in zip(mine, theirs))

def compatible(mine, theirs):
    """
    Whether two public offers would trade if their players were within the trading radius of each other
    """
    return is_valid(mine) and is_valid(theirs) and _wants_covered(mine, theirs) and _wants_covered(theirs, mine)

def could_fill(inventory, theirs):
    """
    Whether a player holding inventory could give what a public offer wants
    """
    return is_valid(theirs) and all(q <= 0 or held >= q for q, held in zip(theirs, inventory))

def partners(state, index):
    """
    Players player index could trade with: those whose public offer is compatible with its own; or, if it has no public offer, those whose
    public offer it could fill from its inventory
    """
    me = state.players[index]
    others = [p for p in state.players if p.index != index]
    if is_valid(me.offer):
        return [p for p in others if compatible(me.offer, p.offer)]
    return [p for p in others if could_fill(me.inventory, p.offer)]

def _target_features(player, cells, view_limited):
    if view_limited:
        cells = [cell for cell in cells if in_view(player, cell)]
    found = nearest(player.position, cells)
    if found is None:
        return [1.0, 0.0, 0.0]
    distance, cell = found
    if distance == 0:
        return [0.0, 0.0, 0.0]
    forward, right = egocentric((cell[0] - player.position[0], cell[1] - player.position[1]), player.orientation)
    return [distance / MAX_DISTANCE, forward / distance, right / distance]

def _cell_is_river(cell):
    x, y = cell
    return 0 <= x < MAP_WIDTH and 0 <= y < MAP_HEIGHT and bool(RIVER[y, x])

def own_targets(state, index):
    """
    The cells each movement label is about, for player index: ripe trees of its own fruit, of the other fruit, and its partners' positions
    """
    player = state.players[index]
    return {"own": ripe_trees(state.trees, player.own_fruit), "other": ripe_trees(state.trees, player.other_fruit),
            "partner": [p.position for p in partners(state, index)]}

def hungry_players_nearby(state, index):
    me = state.players[index]
    return sum(1 for p in state.players if p.index != index and p.hunger >= 1 and within_radius(me.position, p.position))

def well_being_scale(well_being_mode):
    return WELL_BEING_SCALE * (MOST_TASTY_REWARD if well_being_mode == "preference" else 1)

def observation(state, index, others_well_being, view_limited=False, well_being_mode="need"):
    """
    The observation vector of player index: FEATURE_NAMES, then others_well_being (one value per other player, in index order)
    """
    p = state.players[index]
    own, other = FRUITS.index(p.own_fruit), FRUITS.index(p.other_fruit)
    targets = own_targets(state, index)
    facing = [0.0] * 4
    facing[p.orientation] = 1.0
    ahead = (p.position[0] + FORWARD[p.orientation][0], p.position[1] + FORWARD[p.orientation][1])
    compatible_nearby = sum(1 for q in state.players if q.index != index and within_radius(p.position, q.position) and compatible(p.offer, q.offer))
    features = ([min(p.hunger, 1.0), min(p.hungry_steps / HUNGRY_TIME_SCALE, 1.0), p.stamina,
                 min(p.inventory[own] / FRUIT_SCALE, 1.0), min(p.inventory[other] / FRUIT_SCALE, 1.0),
                 1.0 if p.own_fruit == "apple" else 0.0, float(p.ready_to_shoot)] + facing
                + [p.offer[own] / MAX_OFFER_QUANTITY, p.offer[other] / MAX_OFFER_QUANTITY]
                + _target_features(p, targets["own"], view_limited) + _target_features(p, targets["other"], view_limited)
                + _target_features(p, targets["partner"], view_limited)
                + [compatible_nearby / COUNT_SCALE, float(_cell_is_river(p.position)), float(_cell_is_river(ahead)),
                   0.0 if p.held_by is None else 1.0, hungry_players_nearby(state, index) / COUNT_SCALE])
    well_being = np.clip(np.asarray(others_well_being, dtype=float) / well_being_scale(well_being_mode), -1.0, 1.0)
    return np.concatenate([np.array(features, dtype=float), well_being])

"""
Facts about Melting Pot's fruit_market__concentric_rivers substrate that the rest of the package needs, written out so they can be used
without Melting Pot installed. backend.py checks the action set against the installed substrate's, and tests check the rest against its
config (see tests/test_fruit_market_backend.py).
"""
import numpy as np

SUBSTRATE_NAME = "fruit_market__concentric_rivers"
NUM_PLAYERS = 16
EPISODE_LENGTH = 1000
#the substrate's own role assignment (config.default_player_roles): 8 apple farmers, then 8 banana farmers
DEFAULT_ROLES = ("apple_farmer",) * 8 + ("banana_farmer",) * 8
FRUITS = ("apple", "banana")
ROLE_SPECIALTY = {"apple_farmer": "apple", "banana_farmer": "banana"}

#PeriodicNeed: HUNGER = max(1 - level/HUNGER_DELAY, 0), where level starts at HUNGER_DELAY and falls by 1 each step until the player eats.
#HUNGER reaches 1 after HUNGER_DELAY steps and keeps rising after that (by 1/HUNGER_DELAY a step); from 1 on, the player loses 1 stamina a step
HUNGER_DELAY = 50
HUNGER_RATE = 1 / HUNGER_DELAY
HUNGER_LIMIT = 1.0
#Taste: eating the fruit a player doesn't specialise in gives MOST_TASTY_REWARD, its own fruit DEFAULT_TASTE_REWARD
MOST_TASTY_REWARD = 8
DEFAULT_TASTE_REWARD = 1
#Trading: offers are matched with players within this Euclidean radius
TRADE_RADIUS = 4
MAX_OFFER_QUANTITY = 3
#Harvestable: steps a harvested tree stays unripe
REGROWTH_TIME = 50
#Stamina: the largest value (STAMINA is reported as value / MAX_STAMINA)
MAX_STAMINA = 18
#Avatar view: cells visible to the left and right, ahead and behind
VIEW_SIDE, VIEW_FORWARD, VIEW_BACKWARD = 5, 9, 1

#orientations as LocationObserver reports them, and each one's forward and right unit vectors in (x, y) map coordinates (y grows southwards)
ORIENTATIONS = ("N", "E", "S", "W")
FORWARD = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}
RIGHT = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

def _offer_quantities():
    """
    The 12 offers in the order of the substrate's ACTION_SET (fruit_market.py), as (apple, banana): positive is wanted, negative given
    """
    offers = []
    for a in range(-MAX_OFFER_QUANTITY, MAX_OFFER_QUANTITY):
        for b in range(-MAX_OFFER_QUANTITY, MAX_OFFER_QUANTITY):
            if (a > 0 and b < 0) or (a < 0 and b > 0):
                offers.append((a, b))
    return offers

OFFER_QUANTITIES = _offer_quantities()
FIRST_OFFER_ACTION = 13

def _offer_name(apple, banana):
    return f"OFFER_APPLE{apple:+d}_BANANA{banana:+d}"

#the substrate's 25 actions, in its order; each name says what the action's components are (see ACTION_COMPONENTS)
ACTION_NAMES = (["NOOP", "FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT", "TURN_LEFT", "TURN_RIGHT", "EAT_APPLE", "EAT_BANANA",
                 "HOLD", "SHOVE", "PULL", "CANCEL_OFFER"] + [_offer_name(a, b) for a, b in OFFER_QUANTITIES])
NUM_ACTIONS = len(ACTION_NAMES)
GRAPPLING_ACTIONS = (9, 10, 11)
MOVE_ACTIONS = (1, 2, 3, 4)
TURN_ACTIONS = (5, 6)
EAT_ACTIONS = {7: "apple", 8: "banana"}
CANCEL_ACTION = 12

def _components(index):
    """
    The substrate's component dict for action index (as in fruit_market.ACTION_SET), so backend.py can check ours against the installed one
    """
    components = {"move": 0, "turn": 0, "eat_apple": 0, "eat_banana": 0, "offer_apple": 0, "offer_banana": 0, "offer_cancel": 0, "hold": 0, "shove": 0}
    name = ACTION_NAMES[index]
    moves = {"FORWARD": 1, "STEP_RIGHT": 2, "BACKWARD": 3, "STEP_LEFT": 4}
    if name in moves:
        components["move"] = moves[name]
    elif name in ("TURN_LEFT", "TURN_RIGHT"):
        components["turn"] = -1 if name == "TURN_LEFT" else 1
    elif name.startswith("EAT_"):
        components["eat_" + name[4:].lower()] = 1
    elif name in ("HOLD", "SHOVE", "PULL"):
        components["hold"] = 1
        components["shove"] = {"HOLD": 0, "SHOVE": 1, "PULL": -1}[name]
    elif name == "CANCEL_OFFER":
        components["offer_cancel"] = 1
    elif index >= FIRST_OFFER_ACTION:
        components["offer_apple"], components["offer_banana"] = OFFER_QUANTITIES[index - FIRST_OFFER_ACTION]
    return components

ACTION_COMPONENTS = tuple(_components(i) for i in range(NUM_ACTIONS))

def offer_of(action):
    """
    (apple, banana) of an offer action, or None for any other action
    """
    if action >= FIRST_OFFER_ACTION:
        return OFFER_QUANTITIES[action - FIRST_OFFER_ACTION]
    return None

def taste_value(role, fruit):
    """
    Reward role gets for eating fruit: the most for the fruit it doesn't specialise in
    """
    return DEFAULT_TASTE_REWARD if ROLE_SPECIALTY[role] == fruit else MOST_TASTY_REWARD

ASCII_MAP = """
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
x/___________________________+x
x'###########################`x
x!~~~~~~~~~~~~~~~~~~~~~~~~~~~|x
x!~~~~~~~~~~~~~~~~~~~~~~~~~~~|x
x!~~~LLLLLLLLLLLLLLLLLLLLL~~~|x
x!~~~L~~~~~~~~~~~~~~~~~~~L~~~|x
x!~~~L~~~~~~~~~~~~~~~~~~~L~~~|x
x!~~~L~~LLLLLLLLLLLLLLL~~L~~~|x
x!~~~L~~L~~~~~~~~~~~~~L~~L~~~|x
x!~~~L~~L~~~~~~~~~~~~~L~~L~~~|x
x!~~~L~~L~~LLLLLLLLL~~L~~L~~~|x
x!~~~L~~L~~LP~P~P~PL~~L~~L~~~|x
x!~~~L~~L~~L~P~P~P~L~~L~~L~~~|x
x!~~~L~~L~~L~~P~P~~L~~L~~L~~~|x
x!~~~L~~L~~L~P~P~P~L~~L~~L~~~|x
x!~~~L~~L~~L~~P~P~~L~~L~~L~~~|x
x!~~~L~~L~~L~P~P~P~L~~L~~L~~~|x
x!~~~L~~L~~LP~P~P~PL~~L~~L~~~|x
x!~~~L~~L~~LLLLLLLLL~~L~~L~~~|x
x!~~~L~~L~~~~~~~~~~~~~L~~L~~~|x
x!~~~L~~L~~~~~~~~~~~~~L~~L~~~|x
x!~~~L~~LLLLLLLLLLLLLLL~~L~~~|x
x!~~~L~~~~~~~~~~~~~~~~~~~L~~~|x
x!~~~L~~~~~~~~~~~~~~~~~~~L~~~|x
x!~~~LLLLLLLLLLLLLLLLLLLLL~~~|x
x!~~~~~~~~~~~~~~~~~~~~~~~~~~~|x
x!~~~~~~~~~~~~~~~~~~~~~~~~~~~|x
x!~~~~~~~~~~~~~~~~~~~~~~~~~~~|x
x(---------------------------)x
x<###########################>x
"""
MAP_ROWS = ASCII_MAP.strip("\n").split("\n")
MAP_HEIGHT = len(MAP_ROWS)
MAP_WIDTH = len(MAP_ROWS[0])
#river tiles, indexed [y, x]
RIVER = np.array([[cell == "L" for cell in row] for row in MAP_ROWS])
#the largest Manhattan distance between two cells of the map
MAX_DISTANCE = MAP_WIDTH + MAP_HEIGHT - 2

#tree codes in the WORLD.TREES observation (see lua/priene_trade/priene_components.lua)
NO_TREE, RIPE_APPLE, RIPE_BANANA, UNRIPE_APPLE, UNRIPE_BANANA = 0, 1, 2, 3, 4
RIPE_CODE = {"apple": RIPE_APPLE, "banana": RIPE_BANANA}

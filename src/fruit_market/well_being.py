"""
Well-being for the ethics module (docs/fruit_market_design.md §7), in steps:
    need:       w = max(1 - hunger, 0)/rate + (apples + bananas) * limit/rate - HUNGRY_DECAY * hungry_steps
    preference: w = max(1 - hunger, 0)/rate * v_last/v_min + limit/rate * sum_f(n_f * v_f/v_min) - HUNGRY_DECAY * hungry_steps
where rate is HUNGER's rise per step, limit the HUNGER at which hunger costs begin (so limit/rate = 50 steps per fruit), v_f the value of
fruit f to the player (the substrate's taste reward), v_min the smaller of its two values, and v_last the value of the fruit it last ate.
"""
import numpy as np
from .substrate import HUNGER_LIMIT, HUNGER_RATE, FRUITS, taste_value

WELL_BEING_MODES = ["need", "preference"]
#well-being lost per step a player stays hungry: a tenth of the rate it falls before hunger (1 per step)
HUNGRY_WELL_BEING_DECAY = 0.1
STEPS_PER_FRUIT = HUNGER_LIMIT / HUNGER_RATE

def _satiation(hunger):
    #steps until hunger costs begin (0 once they have)
    return max(HUNGER_LIMIT - hunger, 0) / HUNGER_RATE

def _hungry_steps(hunger):
    return max(round((hunger - HUNGER_LIMIT) / HUNGER_RATE), 0)

def need_well_being(hunger, inventory):
    return _satiation(hunger) + sum(inventory) * STEPS_PER_FRUIT - HUNGRY_WELL_BEING_DECAY * _hungry_steps(hunger)

def preference_well_being(hunger, inventory, role, last_eaten):
    """
    last_eaten -- the fruit the player last ate this episode, or None if it hasn't eaten yet (its current cycle then counts at v_min)
    """
    values = {fruit: taste_value(role, fruit) for fruit in FRUITS}
    v_min = min(values.values())
    v_last = v_min if last_eaten is None else values[last_eaten]
    held = sum(n * values[fruit] / v_min for fruit, n in zip(FRUITS, inventory))
    return _satiation(hunger) * v_last / v_min + held * STEPS_PER_FRUIT - HUNGRY_WELL_BEING_DECAY * _hungry_steps(hunger)

def well_being(mode, hunger, inventory, role, last_eaten):
    if mode == "need":
        return need_well_being(hunger, inventory)
    return preference_well_being(hunger, inventory, role, last_eaten)

def passive_hunger(hunger):
    """
    HUNGER one step later if the player doesn't eat
    """
    return hunger + HUNGER_RATE

def society_well_being(mode, hungers, inventories, roles, last_eaten):
    return np.array([well_being(mode, h, inv, role, last) for h, inv, role, last in zip(hungers, inventories, roles, last_eaten)], dtype=float)

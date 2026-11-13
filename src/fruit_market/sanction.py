"""
The two well-being vectors an ethics agent's sanction compares (docs/fruit_market_design.md §6.3). Each player record is
(hunger, inventory, role, last_eaten), before or after the step.
    attributed -- before: every player advanced one step with nobody acting (passive); after: the same, except the agent's own entry and its
        trade partner's (if it traded this step), which take their actual values. They differ only by what the agent's action did.
    joint -- before: every player's actual value before the step; after: every player's actual value after it.
"""
from .well_being import society_well_being, passive_hunger

SANCTION_MODES = ["attributed", "joint"]

def _vector(mode, records):
    hungers, inventories, roles, last_eaten = zip(*records)
    return society_well_being(mode, hungers, inventories, roles, last_eaten)

def passive(records):
    return [(passive_hunger(hunger), inventory, role, last_eaten) for hunger, inventory, role, last_eaten in records]

def sanction_vectors(sanction_mode, well_being_mode, before, after, agent, partner=None):
    """
    (before, after) well-being vectors for agent's sanction; partner is the player it traded with this step, if any
    """
    if sanction_mode == "joint":
        return _vector(well_being_mode, before), _vector(well_being_mode, after)
    passive_before = passive(before)
    attributed_after = list(passive_before)
    for index in [agent] + ([] if partner is None else [partner]):
        attributed_after[index] = after[index]
    return _vector(well_being_mode, passive_before), _vector(well_being_mode, attributed_after)

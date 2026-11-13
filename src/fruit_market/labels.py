"""
Norm labels (docs/fruit_market_design.md §9): what each step's primitive action did, as a higher-level behaviour, for the norms modules
only; agents never choose labels.
"""
import math
from .substrate import (MOVE_ACTIONS, TURN_ACTIONS, EAT_ACTIONS, GRAPPLING_ACTIONS, CANCEL_ACTION, ACTION_NAMES, FORWARD, offer_of,
                        RIPE_CODE)
from .features import own_targets, nearest, manhattan

NORM_LABELS = ["wait", "wait at other tree", "harvest own fruit", "harvest other fruit", "approach partner", "go to own fruit",
               "go to other fruit", "move", "eat own fruit", "eat other fruit", "hold", "shove", "pull", "cancel offer",
               "offer generous", "offer fair", "offer demanding"]
NORM_LABEL_INDEX = {label: i for i, label in enumerate(NORM_LABELS)}
#movement effects in tie-break order: the first one a step has gives its label
MOVE_EFFECTS = [("partner", "approach partner"), ("own", "go to own fruit"), ("other", "go to other fruit")]
COOPERATIVE_NORM_LABELS = {"offer generous", "offer fair"}
AGGRESSIVE_NORM_LABELS = {"hold", "shove", "pull"}

def offer_label(offer):
    """
    An offer's terms relative to a 1:1 trade: generous if it gives more than it wants, demanding if less
    """
    give = -sum(q for q in offer if q < 0)
    want = sum(q for q in offer if q > 0)
    if give > want:
        return "offer generous"
    if give < want:
        return "offer demanding"
    return "offer fair"

def _facing_cosine(position, orientation, cell):
    dx, dy = cell[0] - position[0], cell[1] - position[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return 1.0
    fx, fy = FORWARD[orientation]
    return (dx * fx + dy * fy) / length

def move_effects(before, after, index):
    """
    Which targets player index's step brought it closer to, by kind ("partner", "own", "other"): a move that shortened the path to the
    nearest one, or a turn that faced it more directly. Targets are as they were before the step.
    """
    targets = own_targets(before, index)
    start, end = before.players[index], after.players[index]
    effects = set()
    for kind, cells in targets.items():
        found = nearest(start.position, cells)
        if found is None or found[0] == 0:
            continue
        if end.position != start.position:
            if min(manhattan(end.position, cell) for cell in cells) < found[0]:
                effects.add(kind)
        elif end.orientation != start.orientation:
            if _facing_cosine(end.position, end.orientation, found[1]) > _facing_cosine(start.position, start.orientation, found[1]):
                effects.add(kind)
    return effects

def _on_ripe_other_tree(state, index):
    player = state.players[index]
    x, y = player.position
    return state.trees[y, x] == RIPE_CODE[player.other_fruit]

def norm_label(action, before, after, index, harvested=None):
    """
    The label of player index's action this step. harvested is "own" or "other" if the step harvested that fruit, otherwise None
    """
    player = before.players[index]
    if action == 0 or action in MOVE_ACTIONS or action in TURN_ACTIONS:
        if harvested is not None:
            return f"harvest {harvested} fruit"
        if action == 0:
            return "wait at other tree" if _on_ripe_other_tree(before, index) else "wait"
        effects = move_effects(before, after, index)
        for kind, label in MOVE_EFFECTS:
            if kind in effects:
                return label
        return "move"
    if action in EAT_ACTIONS:
        return "eat own fruit" if EAT_ACTIONS[action] == player.own_fruit else "eat other fruit"
    if action in GRAPPLING_ACTIONS:
        return ACTION_NAMES[action].lower()
    if action == CANCEL_ACTION:
        return "cancel offer"
    return offer_label(offer_of(action))

def move_effect_flags(before, after, index):
    """
    The move_effects as a string for the report, e.g. "partner+own" ("" if none)
    """
    effects = move_effects(before, after, index)
    return "+".join(kind for kind, _ in MOVE_EFFECTS if kind in effects)

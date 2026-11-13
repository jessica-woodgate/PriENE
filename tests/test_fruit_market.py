"""
Fruit Market's pure parts (src/fruit_market/): the substrate's action set and offers, egocentric features, offer compatibility,
well-being, the ethics sanction's two well-being vectors, and norm labels. None of these need Melting Pot installed.
"""
import numpy as np
import pytest
from src.fruit_market import substrate as fm
from src.fruit_market.state import PlayerState, FruitMarketState, Trade
from src.fruit_market.features import (egocentric, compatible, could_fill, observation, observation_size, FEATURE_INDEX, partners,
                                       WELL_BEING_SCALE)
from src.fruit_market.well_being import need_well_being, preference_well_being, HUNGRY_WELL_BEING_DECAY
from src.fruit_market.sanction import sanction_vectors
from src.fruit_market.labels import norm_label, offer_label, move_effects, NORM_LABELS
from src.agent.modules.ethics_module import EthicsModule

def player(index, role="apple_farmer", position=(5, 5), orientation=0, inventory=(0, 0), hunger=0.0, offer=(0, 0), held_by=None):
    return PlayerState(index, role, position, orientation, inventory, hunger, 1.0, 1.0, offer, held_by)

def empty_trees():
    return np.zeros((fm.MAP_HEIGHT, fm.MAP_WIDTH), dtype=int)

def state(players, trees=None, trades=None):
    return FruitMarketState(players, empty_trees() if trees is None else trees, trades or [], [0.0] * len(players))

# --- the substrate's actions ---

def test_the_action_set_is_the_substrates_25_actions_in_its_order():
    assert fm.NUM_ACTIONS == 25
    assert fm.ACTION_NAMES[:13] == ["NOOP", "FORWARD", "BACKWARD", "STEP_LEFT", "STEP_RIGHT", "TURN_LEFT", "TURN_RIGHT", "EAT_APPLE",
                                    "EAT_BANANA", "HOLD", "SHOVE", "PULL", "CANCEL_OFFER"]
    assert fm.OFFER_QUANTITIES == [(-3, 1), (-3, 2), (-2, 1), (-2, 2), (-1, 1), (-1, 2), (1, -3), (1, -2), (1, -1), (2, -3), (2, -2), (2, -1)]

@pytest.mark.parametrize("index, components", [
    (1, {"move": 1}), (2, {"move": 3}), (3, {"move": 4}), (4, {"move": 2}), (5, {"turn": -1}), (6, {"turn": 1}), (7, {"eat_apple": 1}),
    (8, {"eat_banana": 1}), (9, {"hold": 1}), (10, {"hold": 1, "shove": 1}), (11, {"hold": 1, "shove": -1}), (12, {"offer_cancel": 1}),
    (13, {"offer_apple": -3, "offer_banana": 1}), (24, {"offer_apple": 2, "offer_banana": -1})])
def test_each_action_has_the_substrates_components(index, components):
    expected = {"move": 0, "turn": 0, "eat_apple": 0, "eat_banana": 0, "offer_apple": 0, "offer_banana": 0, "offer_cancel": 0, "hold": 0, "shove": 0}
    expected.update(components)
    assert fm.ACTION_COMPONENTS[index] == expected

def test_offers_are_6_generous_4_fair_and_2_demanding():
    labels = [offer_label(offer) for offer in fm.OFFER_QUANTITIES]
    assert (labels.count("offer generous"), labels.count("offer fair"), labels.count("offer demanding")) == (6, 4, 2)
    assert offer_label((-3, 1)) == "offer generous" and offer_label((-1, 2)) == "offer demanding" and offer_label((2, -2)) == "offer fair"

def test_the_preferred_fruit_is_the_one_a_player_does_not_specialise_in():
    assert fm.taste_value("apple_farmer", "banana") == 8 and fm.taste_value("apple_farmer", "apple") == 1
    assert fm.taste_value("banana_farmer", "apple") == 8

def test_the_map_has_its_river_rings():
    assert (fm.MAP_WIDTH, fm.MAP_HEIGHT) == (31, 31)
    assert fm.RIVER.sum() == 168

# --- features ---

@pytest.mark.parametrize("orientation, offset, expected", [
    (0, (0, -3), (3, 0)), (0, (2, 0), (0, 2)), (1, (2, 0), (2, 0)), (1, (0, 2), (0, 2)), (2, (0, 3), (3, 0)), (2, (-1, 0), (0, 1)),
    (3, (-2, 0), (2, 0)), (3, (0, -1), (0, 1))])
def test_egocentric_directions_follow_the_players_facing(orientation, offset, expected):
    assert egocentric(offset, orientation) == expected

@pytest.mark.parametrize("mine, theirs, expected", [
    ((-1, 1), (1, -1), True), ((-3, 1), (1, -1), True), ((-1, 2), (1, -1), False), ((-1, 2), (1, -2), True), ((-1, 1), (-1, 1), False),
    ((0, 0), (1, -1), False), ((2, -1), (-1, 1), False), ((2, -1), (-3, 1), True), ((2, -1), (-2, 1), True)])
def test_offers_are_compatible_when_each_gives_at_least_what_the_other_wants(mine, theirs, expected):
    assert compatible(mine, theirs) == expected

def test_a_player_with_no_offer_can_partner_anyone_whose_offer_it_could_fill():
    assert could_fill((2, 0), (2, -1)) and not could_fill((1, 0), (2, -1))
    s = state([player(0, inventory=(2, 0)), player(1, role="banana_farmer", offer=(2, -1)), player(2, offer=(0, 0))])
    assert [p.index for p in partners(s, 0)] == [1]

def test_the_observation_has_every_feature_then_each_other_players_well_being():
    trees = empty_trees()
    trees[2, 5] = fm.RIPE_APPLE
    s = state([player(0, position=(5, 5), offer=(-1, 2), inventory=(4, 0)), player(1, role="banana_farmer", position=(6, 5))], trees)
    obs = observation(s, 0, [WELL_BEING_SCALE / 2])
    assert len(obs) == observation_size(2) == len(FEATURE_INDEX) + 1
    #the ripe apple tree (own fruit) 3 cells north of a player facing north: straight ahead
    assert obs[FEATURE_INDEX["own_tree_distance"]] == pytest.approx(3 / fm.MAX_DISTANCE)
    assert (obs[FEATURE_INDEX["own_tree_forward"]], obs[FEATURE_INDEX["own_tree_right"]]) == (1.0, 0.0)
    #no ripe banana tree: the furthest distance and no direction
    assert obs[FEATURE_INDEX["other_tree_distance"]] == 1.0
    #offers role-relative: an apple farmer giving 1 apple (own fruit) for 2 bananas
    assert (obs[FEATURE_INDEX["offer_own"]], obs[FEATURE_INDEX["offer_other"]]) == pytest.approx((-1 / 3, 2 / 3))
    assert obs[FEATURE_INDEX["own_fruit"]] == pytest.approx(0.4)
    assert obs[-1] == pytest.approx(0.5)

def test_the_view_limited_observation_ignores_trees_out_of_view():
    trees = empty_trees()
    trees[8, 5] = fm.RIPE_APPLE
    s = state([player(0, position=(5, 5), orientation=0), player(1)], trees)
    #3 cells behind a player facing north: out of its view (1 cell behind), but known in the global observation
    assert observation(s, 0, [0.0], view_limited=True)[FEATURE_INDEX["own_tree_distance"]] == 1.0
    assert observation(s, 0, [0.0])[FEATURE_INDEX["own_tree_distance"]] == pytest.approx(3 / fm.MAX_DISTANCE)

# --- well-being ---

def test_hunger_keeps_rising_after_it_reaches_the_limit_and_counts_hungry_steps():
    assert player(0, hunger=1.0).hungry_steps == 0
    assert player(0, hunger=1.6).hungry_steps == 30

def test_need_well_being_is_steps_until_hunger_plus_fifty_per_fruit():
    assert need_well_being(0.0, (0, 0)) == pytest.approx(50)
    assert need_well_being(0.6, (2, 1)) == pytest.approx(20 + 150)
    assert need_well_being(1.0, (0, 0)) == pytest.approx(0)

def test_time_spent_hungry_slowly_reduces_well_being():
    assert need_well_being(1.6, (0, 0)) == pytest.approx(-30 * HUNGRY_WELL_BEING_DECAY)
    assert need_well_being(1.6, (1, 0)) == pytest.approx(50 - 3)

def test_eating_at_the_hunger_limit_is_neutral_earlier_is_a_loss_and_later_a_gain():
    assert need_well_being(0.0, (0, 0)) - need_well_being(1.0, (1, 0)) == pytest.approx(0)
    assert need_well_being(0.0, (0, 0)) - need_well_being(0.6, (1, 0)) == pytest.approx(-20)
    assert need_well_being(0.0, (0, 0)) - need_well_being(1.6, (1, 0)) == pytest.approx(3)

def test_preference_well_being_counts_preferred_fruit_and_eating_it_at_the_limit_is_neutral():
    #an apple farmer prefers bananas (8 to an apple's 1)
    holding = preference_well_being(1.0, (0, 1), "apple_farmer", None)
    assert holding == pytest.approx(400)
    assert preference_well_being(0.0, (0, 0), "apple_farmer", "banana") == pytest.approx(holding)
    assert preference_well_being(0.0, (0, 0), "apple_farmer", "apple") == pytest.approx(preference_well_being(1.0, (1, 0), "apple_farmer", None))

# --- the sanction's well-being vectors (the worked example in docs/fruit_market_design.md §6.3) ---

def worked_example():
    roles = ["apple_farmer"] * 4
    #well-being [120, 60, 20, -3]: player 3 has been hungry for 30 steps
    before = [(0.6, (2, 0)), (0.8, (1, 0)), (0.6, (0, 0)), (1.6, (0, 0))]
    #player 0 walks, player 1 harvests 2 apples, players 2 and 3 wait
    after = [(0.62, (2, 0)), (0.82, (3, 0)), (0.62, (0, 0)), (1.62, (0, 0))]
    return ([(h, inv, role, None) for (h, inv), role in zip(before, roles)], [(h, inv, role, None) for (h, inv), role in zip(after, roles)])

def sanction(principle, vectors):
    ethics = EthicsModule(3.5, principle)
    ethics.update_ethics_state(vectors[0])
    return ethics.get_sanction(vectors[1])[0]

def test_the_worked_example_has_the_documented_well_being():
    before, after = worked_example()
    joint_before, joint_after = sanction_vectors("joint", "need", before, after, 0)
    assert joint_before == pytest.approx([120, 60, 20, -3])
    assert joint_after == pytest.approx([119, 159, 19, -3.1])

def test_an_attributed_sanction_reflects_only_the_agents_own_action():
    before, after = worked_example()
    #player 0 only walked; player 1 harvested
    assert sanction("utilitarian", sanction_vectors("attributed", "need", before, after, 0)) == 0
    assert sanction("utilitarian", sanction_vectors("attributed", "need", before, after, 1)) == 3.5
    #a joint sanction rewards player 0 for player 1's harvest
    assert sanction("utilitarian", sanction_vectors("joint", "need", before, after, 0)) == 3.5

def test_with_a_joint_sanction_maximin_agents_are_punished_while_anyone_stays_hungry():
    before, after = worked_example()
    assert sanction("maximin", sanction_vectors("joint", "need", before, after, 2)) == -3.5
    assert sanction("maximin", sanction_vectors("attributed", "need", before, after, 2)) == 0

def test_a_trade_counts_for_both_traders_in_an_attributed_sanction():
    before, after = worked_example()
    #player 1 gives player 3 (the worst off) an apple for nothing else changing... as part of a 2-for-1 trade both entries change
    after[1] = (0.82, (0, 0), "apple_farmer", None)
    after[3] = (1.62, (1, 0), "apple_farmer", None)
    assert sanction("maximin", sanction_vectors("attributed", "need", before, after, 1, partner=3)) == 3.5
    assert sanction("maximin", sanction_vectors("attributed", "need", before, after, 3, partner=1)) == 3.5

# --- norm labels ---

def test_every_label_is_in_the_norm_action_list():
    assert len(NORM_LABELS) == len(set(NORM_LABELS)) == 17

@pytest.mark.parametrize("action, expected", [(7, "eat own fruit"), (8, "eat other fruit"), (9, "hold"), (10, "shove"), (11, "pull"),
                                              (12, "cancel offer"), (13, "offer generous"), (16, "offer fair"), (18, "offer demanding")])
def test_eating_grappling_and_offers_are_labelled_by_the_action(action, expected):
    s = state([player(0), player(1)])
    assert norm_label(action, s, s, 0) == expected

def test_moves_are_labelled_by_their_effect():
    trees = empty_trees()
    trees[2, 5] = fm.RIPE_APPLE
    trees[5, 9] = fm.RIPE_BANANA
    before = state([player(0, position=(5, 5)), player(1, position=(20, 20))], trees)
    north = state([player(0, position=(5, 4)), player(1, position=(20, 20))], trees)
    east = state([player(0, position=(6, 5)), player(1, position=(20, 20))], trees)
    south = state([player(0, position=(5, 6)), player(1, position=(20, 20))], trees)
    assert norm_label(1, before, north, 0) == "go to own fruit"
    assert norm_label(4, before, east, 0) == "go to other fruit"
    assert norm_label(2, before, south, 0) == "move"
    #blocked: no change of position
    assert norm_label(1, before, before, 0) == "move"

def test_approaching_a_partner_takes_precedence_over_going_to_fruit():
    trees = empty_trees()
    trees[2, 5] = fm.RIPE_APPLE
    before = state([player(0, position=(5, 5), offer=(-1, 1), inventory=(1, 0)),
                    player(1, role="banana_farmer", position=(5, 0), offer=(1, -1), inventory=(0, 1))], trees)
    after = state([player(0, position=(5, 4), offer=(-1, 1), inventory=(1, 0)),
                   player(1, role="banana_farmer", position=(5, 0), offer=(1, -1), inventory=(0, 1))], trees)
    assert move_effects(before, after, 0) == {"own", "partner"}
    assert norm_label(1, before, after, 0) == "approach partner"

def test_a_turn_towards_a_target_counts_as_going_to_it():
    trees = empty_trees()
    trees[5, 9] = fm.RIPE_APPLE
    before = state([player(0, position=(5, 5), orientation=0), player(1, position=(20, 20))], trees)
    after = state([player(0, position=(5, 5), orientation=1), player(1, position=(20, 20))], trees)
    assert norm_label(6, before, after, 0) == "go to own fruit"
    assert norm_label(5, after, before, 0) == "move"

def test_harvesting_and_waiting_on_another_fruits_tree_have_their_own_labels():
    trees = empty_trees()
    trees[5, 5] = fm.RIPE_BANANA
    s = state([player(0, position=(5, 5)), player(1, position=(20, 20))], trees)
    assert norm_label(0, s, s, 0) == "wait at other tree"
    assert norm_label(0, s, s, 0, harvested="other") == "harvest other fruit"
    assert norm_label(1, s, s, 0, harvested="own") == "harvest own fruit"
    assert norm_label(0, state([player(0), player(1)]), state([player(0), player(1)]), 0) == "wait"

def test_trades_say_who_received_what():
    s = state([player(0), player(1)], trades=[Trade(0, 1, (-1, 1), (1, -1))])
    assert s.trade_partner(0) == 1 and s.trade_partner(1) == 0
    assert s.received(1) == (1, -1) and s.received(2) == (0, 0)

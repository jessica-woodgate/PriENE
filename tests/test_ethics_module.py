import numpy as np
import pytest

from src.agent.modules.ethics_module import EthicsModule


# --- welfare calculations ---

def test_calculate_egalitarian_welfare_perfect_equality():
    em = EthicsModule(sanction=0.4, principle="egalitarian")
    assert em._calculate_egalitarian_welfare(np.array([10.0, 10.0, 10.0])) == 0


def test_calculate_egalitarian_welfare_inequality():
    em = EthicsModule(sanction=0.4, principle="egalitarian")
    # mean = 10; absolute deviations = 10, 0, 10 -> sum = 20
    assert em._calculate_egalitarian_welfare(np.array([0.0, 10.0, 20.0])) == 20


def test_calculate_maximin_welfare_single_min():
    em = EthicsModule(sanction=0.4, principle="maximin")
    min_value, num_mins = em._calculate_maximin_welfare(np.array([5.0, 2.0, 8.0]))
    assert min_value == 2.0
    assert num_mins == 1


def test_calculate_maximin_welfare_multiple_mins():
    em = EthicsModule(sanction=0.4, principle="maximin")
    min_value, num_mins = em._calculate_maximin_welfare(np.array([2.0, 2.0, 8.0]))
    assert min_value == 2.0
    assert num_mins == 2


def test_calculate_utilitarian_welfare():
    em = EthicsModule(sanction=0.4, principle="utilitarian")
    assert em._calculate_utilitarian_welfare(np.array([1.0, 2.0, 3.0])) == 6.0


# --- sanction logic (Algorithm 1): compares well-being before vs after acting ---

def test_egalitarian_sanction_improved_welfare():
    em = EthicsModule(sanction=0.4, principle="egalitarian")
    em.update_ethics_state(society_well_being=np.array([0.0, 10.0, 20.0]))  # loss=20
    assert em.get_sanction(np.array([10.0, 10.0, 10.0])) == [0.4]  # loss=0, improved


def test_egalitarian_sanction_worsened_welfare():
    em = EthicsModule(sanction=0.4, principle="egalitarian")
    em.update_ethics_state(society_well_being=np.array([10.0, 10.0, 10.0]))  # loss=0
    assert em.get_sanction(np.array([0.0, 10.0, 20.0])) == [-0.4]  # loss=20, worsened


def test_egalitarian_sanction_unchanged_welfare():
    em = EthicsModule(sanction=0.4, principle="egalitarian")
    em.update_ethics_state(society_well_being=np.array([10.0, 10.0, 10.0]))
    assert em.get_sanction(np.array([10.0, 10.0, 10.0])) == [0]


def test_maximin_sanction_min_increased():
    em = EthicsModule(sanction=0.4, principle="maximin")
    em.update_ethics_state(society_well_being=np.array([2.0, 5.0, 8.0]))  # min=2
    assert em.get_sanction(np.array([3.0, 5.0, 8.0])) == [0.4]  # min=3, improved


def test_maximin_sanction_min_decreased():
    em = EthicsModule(sanction=0.4, principle="maximin")
    em.update_ethics_state(society_well_being=np.array([2.0, 5.0, 8.0]))  # min=2
    assert em.get_sanction(np.array([1.0, 5.0, 8.0])) == [-0.4]  # min=1, worsened


def test_maximin_sanction_fewer_instances_of_min_is_positive():
    em = EthicsModule(sanction=0.4, principle="maximin")
    em.update_ethics_state(society_well_being=np.array([2.0, 2.0, 8.0]))  # min=2, 2 instances
    assert em.get_sanction(np.array([2.0, 5.0, 8.0])) == [0.4]  # min still 2, now only 1 instance


def test_maximin_sanction_more_instances_of_min_is_negative():
    em = EthicsModule(sanction=0.4, principle="maximin")
    em.update_ethics_state(society_well_being=np.array([2.0, 5.0, 8.0]))  # min=2, 1 instance
    assert em.get_sanction(np.array([2.0, 2.0, 8.0])) == [-0.4]  # min still 2, now 2 instances


def test_utilitarian_sanction_total_increased():
    em = EthicsModule(sanction=0.4, principle="utilitarian")
    em.update_ethics_state(society_well_being=np.array([1.0, 2.0, 3.0]))  # total=6
    assert em.get_sanction(np.array([2.0, 2.0, 3.0])) == [0.4]  # total=7, improved


def test_utilitarian_sanction_total_decreased():
    em = EthicsModule(sanction=0.4, principle="utilitarian")
    em.update_ethics_state(society_well_being=np.array([1.0, 2.0, 3.0]))
    assert em.get_sanction(np.array([0.0, 2.0, 3.0])) == [-0.4]  # total=5, worsened


# --- aggregation methods used by the combined principles ---

def test_veto_aggregation_any_negative_vetoes():
    em = EthicsModule(sanction=0.4, principle="veto")
    assert em._veto_aggregation([0.4, 0.4, -0.4]) == [-0.4]
    assert em._veto_aggregation([0.4, 0.4, 0]) == [0.4]
    assert em._veto_aggregation([0, 0, 0]) == [0]


def test_optimist_aggregation_any_positive_wins():
    em = EthicsModule(sanction=0.4, principle="optimist")
    assert em._optimist_aggregation([0.4, -0.4, -0.4]) == [0.4]
    assert em._optimist_aggregation([-0.4, -0.4, 0]) == [-0.4]
    assert em._optimist_aggregation([0, 0, 0]) == [0]


def test_majoritarian_aggregation_clamps_to_sanction_bounds():
    em = EthicsModule(sanction=0.4, principle="majoritarian")
    assert em._majoritarian_aggregation([0.4, 0.4, 0.4]) == [0.4]  # sum=1.2, clamped to 0.4
    assert em._majoritarian_aggregation([-0.4, -0.4, -0.4]) == [-0.4]  # clamped to -0.4
    assert em._majoritarian_aggregation([0.4, -0.4, 0]) == [0.0]


def test_average_aggregation_value():
    em = EthicsModule(sanction=0.4, principle="average")
    assert em._average_aggregation([0.4, 0.4, 0.4]) == pytest.approx([0.4])
    assert em._average_aggregation([0.4, -0.4, 0]) == pytest.approx([0.0])


@pytest.mark.parametrize("principle", ["veto", "optimist", "majoritarian", "average"])
def test_combined_sanction_returns_single_element_list(principle):
    em = EthicsModule(sanction=0.4, principle=principle)
    em.update_ethics_state(society_well_being=np.array([1.0, 2.0, 3.0]))
    sanction = em.get_sanction(np.array([2.0, 2.0, 3.0]))
    assert isinstance(sanction, list)
    assert len(sanction) == 1


ALL_PRINCIPLES = EthicsModule.PRINCIPLES + EthicsModule.AGGREGATIONS


def _days_left(berries, health):
    #HarvestAgent.get_days_left_to_live with its default berry_health_payoff (0.6) and health_decay (0.1)
    return max((0.6 * berries + health) / 0.1, 0)


def _health_after_decays(decays):
    #health as an agent's arithmetic reaches it: start health 0.8, then 0.1 decay a step, topped up with a 0.6 berry when low
    health = 0.8
    for _ in range(decays):
        health -= 0.1
        if health <= 0.05:
            health += 0.6
    return health


def _eating_cases():
    #(berries, health) where eating a berry leaves days left to live unchanged in exact arithmetic but not in floats
    return [(berries, _health_after_decays(decays)) for berries in range(1, 5) for decays in range(40)
            if _days_left(berries, _health_after_decays(decays)) != _days_left(berries - 1, _health_after_decays(decays) + 0.6)]


@pytest.mark.parametrize("principle", ALL_PRINCIPLES)
def test_eating_is_never_sanctioned_despite_float_error(principle):
    """
    Regression test: eating turns a berry (6 days) into health (6 days), so days left to live is unchanged, but health's float
    arithmetic left it differing by ~1e-15, and the exact comparison gave eating a +/-0.4 sanction
    """
    cases = _eating_cases()
    assert cases, "expected float error in some eating cases, or this test checks nothing"
    others = [7.0, 12.0, 20.0]
    for berries, health in cases:
        em = EthicsModule(sanction=0.4, principle=principle)
        em.update_ethics_state(np.array([_days_left(berries, health)] + others))
        assert em.get_sanction(np.array([_days_left(berries - 1, health + 0.6)] + others)) == [0]


def test_utilitarian_throw_is_never_sanctioned_despite_float_error():
    """
    A throw moves 6 days from the thrower to the benefactor, so the society's total is unchanged, but float error used to sanction it
    """
    noisy = 0
    for thrower_decays in range(40):
        for benefactor_decays in range(40):
            thrower, benefactor = _health_after_decays(thrower_decays), _health_after_decays(benefactor_decays)
            before = np.array([_days_left(2, thrower), _days_left(0, benefactor), 15.0])
            after = np.array([_days_left(1, thrower), _days_left(0, benefactor + 0.6), 15.0])
            noisy += sum(before) != sum(after)
            em = EthicsModule(sanction=0.4, principle="utilitarian")
            em.update_ethics_state(before)
            assert em.get_sanction(after) == [0]
    assert noisy, "expected float error in some throws, or this test checks nothing"


def test_maximin_counts_agents_tied_at_the_minimum_despite_float_error():
    # 5 days stored with and without float error: both agents are at the minimum
    almost_five = (0.8 - 0.1 - 0.1 - 0.1) / 0.1
    assert almost_five != 5.0
    em = EthicsModule(sanction=0.4, principle="maximin")
    em.update_ethics_state(np.array([5.0, almost_five, 9.0]))
    assert em.number_of_minimums == 2
    #one of the two tied agents is lifted off the minimum: fewer agents at it is better
    assert em.get_sanction(np.array([11.0, almost_five, 9.0])) == [0.4]


@pytest.mark.parametrize("principle", ALL_PRINCIPLES)
def test_rounding_keeps_a_one_day_change(principle):
    #the smallest real change (one step of health decay) is still seen
    em = EthicsModule(sanction=0.4, principle=principle)
    em.update_ethics_state(np.array([5.0, 9.0, 12.0]))
    assert em.get_sanction(np.array([4.0, 9.0, 12.0])) != [0]

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
    em.update_ethics_state(can_help=True, society_well_being=np.array([0.0, 10.0, 20.0]))  # loss=20
    assert em.get_sanction(np.array([10.0, 10.0, 10.0])) == [0.4]  # loss=0, improved


def test_egalitarian_sanction_worsened_welfare_when_could_help():
    em = EthicsModule(sanction=0.4, principle="egalitarian")
    em.update_ethics_state(can_help=True, society_well_being=np.array([10.0, 10.0, 10.0]))  # loss=0
    assert em.get_sanction(np.array([0.0, 10.0, 20.0])) == [-0.4]  # loss=20, worsened


def test_egalitarian_sanction_worsened_welfare_when_could_not_help():
    em = EthicsModule(sanction=0.4, principle="egalitarian")
    em.update_ethics_state(can_help=False, society_well_being=np.array([10.0, 10.0, 10.0]))
    assert em.get_sanction(np.array([0.0, 10.0, 20.0])) == [0]


def test_egalitarian_sanction_unchanged_welfare():
    em = EthicsModule(sanction=0.4, principle="egalitarian")
    em.update_ethics_state(can_help=True, society_well_being=np.array([10.0, 10.0, 10.0]))
    assert em.get_sanction(np.array([10.0, 10.0, 10.0])) == [0]


def test_maximin_sanction_min_increased():
    em = EthicsModule(sanction=0.4, principle="maximin")
    em.update_ethics_state(can_help=True, society_well_being=np.array([2.0, 5.0, 8.0]))  # min=2
    assert em.get_sanction(np.array([3.0, 5.0, 8.0])) == [0.4]  # min=3, improved


def test_maximin_sanction_min_decreased_when_could_help():
    em = EthicsModule(sanction=0.4, principle="maximin")
    em.update_ethics_state(can_help=True, society_well_being=np.array([2.0, 5.0, 8.0]))  # min=2
    assert em.get_sanction(np.array([1.0, 5.0, 8.0])) == [-0.4]  # min=1, worsened


def test_maximin_sanction_fewer_instances_of_min_is_positive():
    em = EthicsModule(sanction=0.4, principle="maximin")
    em.update_ethics_state(can_help=True, society_well_being=np.array([2.0, 2.0, 8.0]))  # min=2, 2 instances
    assert em.get_sanction(np.array([2.0, 5.0, 8.0])) == [0.4]  # min still 2, now only 1 instance


def test_maximin_sanction_more_instances_of_min_is_negative_when_could_help():
    em = EthicsModule(sanction=0.4, principle="maximin")
    em.update_ethics_state(can_help=True, society_well_being=np.array([2.0, 5.0, 8.0]))  # min=2, 1 instance
    assert em.get_sanction(np.array([2.0, 2.0, 8.0])) == [-0.4]  # min still 2, now 2 instances


def test_utilitarian_sanction_total_increased():
    em = EthicsModule(sanction=0.4, principle="utilitarian")
    em.update_ethics_state(can_help=True, society_well_being=np.array([1.0, 2.0, 3.0]))  # total=6
    assert em.get_sanction(np.array([2.0, 2.0, 3.0])) == [0.4]  # total=7, improved


def test_utilitarian_sanction_total_decreased_when_could_help():
    em = EthicsModule(sanction=0.4, principle="utilitarian")
    em.update_ethics_state(can_help=True, society_well_being=np.array([1.0, 2.0, 3.0]))
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


def test_average_aggregation_returns_a_single_element_list():
    """
    Regression test: average_aggregation used to return a bare np.mean(...) scalar instead of a
    single-element list like the other three aggregations. That only avoided crashing because
    `list + numpy.float64` happens to trigger numpy's broadcasting __radd__ fallback, whereas
    `list + float(...)` (a plain python float) raises TypeError -- one innocent-looking refactor
    away from breaking every "average" agent run. Now it's consistent with veto/optimist/majoritarian.
    """
    em = EthicsModule(sanction=0.4, principle="average")
    result = em._average_aggregation([0.4, 0.4, 0.4])
    assert isinstance(result, list)
    assert len(result) == 1
    reward_vector = [0.5]
    combined = reward_vector + result
    assert combined == pytest.approx([0.5, 0.4])


@pytest.mark.parametrize("principle", ["veto", "optimist", "majoritarian", "average"])
def test_combined_sanction_returns_single_element_list(principle):
    em = EthicsModule(sanction=0.4, principle=principle)
    em.update_ethics_state(can_help=True, society_well_being=np.array([1.0, 2.0, 3.0]))
    sanction = em.get_sanction(np.array([2.0, 2.0, 3.0]))
    assert isinstance(sanction, list)
    assert len(sanction) == 1

import pytest

from src.harvest_exception import (
    FileExistsException,
    NoBerriesException,
    NumAgentsException,
    NumAllotmentsException,
    AgentTypeException,
    OutOfBounds,
    NoEmptyCells,
    NoAllocationException,
    UnrecognisedPrinciple,
    NoPathFound,
    IllegalBerry,
    NumBerriesException,
    NumFeaturesException,
    ImpossibleNormException,
)


def test_file_exists_exception_message():
    e = FileExistsException("data/results/current_run/foo.csv")
    assert str(e) == "File data/results/current_run/foo.csv already exists"


def test_no_berries_exception_agent_only():
    e = NoBerriesException(agent_id=3)
    assert str(e) == "Agent 3 has found no berries in the grid"


def test_no_berries_exception_coordinates_only():
    e = NoBerriesException(coordinates=(2, 4))
    assert str(e) == "Coordinates (2, 4) has no berries"


def test_no_berries_exception_agent_and_coordinates():
    e = NoBerriesException(agent_id=3, coordinates=(2, 4))
    assert str(e) == "Agent 3 trying to forage at (2, 4) which has no berry"


def test_num_agents_exception_message():
    e = NumAgentsException(4, 3)
    assert str(e) == "Expected 4 agents and got 3"


def test_num_allotments_exception_message():
    e = NumAllotmentsException(4, 6)
    assert str(e) == (
        "Can't have more agents than allotments: currently asking for 4 agents and 6 allotments"
    )


def test_agent_type_exception_message():
    e = AgentTypeException("berry", "baseline")
    assert str(e) == "Expected type berry and got baseline"


def test_out_of_bounds_message():
    e = OutOfBounds(1, (5, 5))
    assert str(e) == "Coordinates (5, 5) are out of bounds for agent 1"


def test_no_empty_cells_message():
    e = NoEmptyCells()
    assert str(e) == "No empty cells in grid"


def test_no_allocation_exception_message():
    e = NoAllocationException(2)
    assert str(e) == "No allocation found for agent 2"


def test_unrecognised_principle_message():
    e = UnrecognisedPrinciple("unknown_principle")
    assert str(e) == "Do not recognise principle unknown_principle"


def test_no_path_found_message():
    e = NoPathFound(1, (0, 0), (3, 3))
    assert str(e) == "Agent 1 couldn't find a path from (0, 0) to (3, 3)"


def test_illegal_berry_message():
    e = IllegalBerry(1, "allocated to allocation_2")
    assert str(e) == "Agent 1 is trying to access illegal berry: allocated to allocation_2"


def test_num_berries_exception_message():
    e = NumBerriesException(10, 8)
    assert str(e) == "Expected 10 berries and got 8"


def test_num_features_exception_message():
    # regression test: this message previously (incorrectly) said "berries" instead of "features"
    e = NumFeaturesException(6, 5)
    assert str(e) == "Expected 6 features and got 5"
    assert "berries" not in str(e)


def test_impossible_norm_exception_message():
    e = ImpossibleNormException(3, 1, "IF,no berries,low health", "throw_2", -0.2)
    assert str(e) == (
        "Day 3, agent 1 saving an impossible behaviour! "
        "Antecedent IF,no berries,low health consequent throw_2; got reward -0.2"
    )


@pytest.mark.parametrize(
    "exception_cls,args",
    [
        (FileExistsException, ("file.csv",)),
        (NoBerriesException, ()),
        (NumAgentsException, (4, 3)),
        (NumAllotmentsException, (4, 6)),
        (AgentTypeException, ("berry", "baseline")),
        (OutOfBounds, (1, (5, 5))),
        (NoEmptyCells, ()),
        (NoAllocationException, (2,)),
        (UnrecognisedPrinciple, ("unknown",)),
        (NoPathFound, (1, (0, 0), (3, 3))),
        (IllegalBerry, (1, "bad")),
        (NumBerriesException, (10, 8)),
        (NumFeaturesException, (6, 5)),
        (ImpossibleNormException, (3, 1, "IF,x", "throw_2", -0.2)),
    ],
)
def test_all_exceptions_are_raisable_and_catchable(exception_cls, args):
    with pytest.raises(exception_cls):
        raise exception_cls(*args)

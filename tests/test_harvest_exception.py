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
)


@pytest.mark.parametrize("exception,message", [
    (FileExistsException("data/results/current_run/foo.csv"), "File data/results/current_run/foo.csv already exists"),
    (NoBerriesException(agent_id=3), "Agent 3 has found no berries in the grid"),
    (NoBerriesException(coordinates=(2, 4)), "Coordinates (2, 4) has no berries"),
    (NoBerriesException(agent_id=3, coordinates=(2, 4)), "Agent 3 trying to forage at (2, 4) which has no berry"),
    (NumAgentsException(4, 3), "Expected 4 agents and got 3"),
    (NumAllotmentsException(4, 6),
     "The number of allotments must be between 1 and the number of agents: currently asking for 4 agents and 6 allotments"),
    (AgentTypeException("berry", "baseline"), "Expected type berry and got baseline"),
    (OutOfBounds(1, (5, 5)), "Coordinates (5, 5) are out of bounds for agent 1"),
    (NoEmptyCells(), "No empty cells in grid"),
    (NoAllocationException(2), "No allocation found for agent 2"),
    (UnrecognisedPrinciple("unknown_principle"), "Do not recognise principle unknown_principle"),
    (NoPathFound(1, (0, 0), (3, 3)), "Agent 1 couldn't find a path from (0, 0) to (3, 3)"),
    (IllegalBerry(1, "allocated to allocation_2"), "Agent 1 is trying to access illegal berry: allocated to allocation_2"),
    (NumBerriesException(10, 8), "Expected 10 berries and got 8"),
    # regression test: this message previously (incorrectly) said "berries" instead of "features"
    (NumFeaturesException(6, 5), "Expected 6 features and got 5"),
], ids=lambda value: type(value).__name__ if isinstance(value, Exception) else "")
def test_exception_messages(exception, message):
    assert str(exception) == message

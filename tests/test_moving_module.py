import types

import pytest

from src.agent.modules.moving_module import MovingModule
from src.harvest_exception import OutOfBounds, NoPathFound, IllegalBerry


def make_module(min_width=0, max_width=10, min_height=0, max_height=10):
    # model=None is safe here: none of the methods under test touch self.model
    return MovingModule(
        agent_id=1,
        model=None,
        training=True,
        allotment=[min_width, max_width, min_height, max_height],
        allocation_id=None,
        restrict_to_allocation=False,
    )


# --- _move: standard convention is north/south change y, east/west change x ---

def test_move_north_increases_y():
    assert make_module()._move((3, 3), "north") == (3, 4)


def test_move_south_decreases_y():
    assert make_module()._move((3, 3), "south") == (3, 2)


def test_move_east_increases_x():
    assert make_module()._move((3, 3), "east") == (4, 3)


def test_move_west_decreases_x():
    assert make_module()._move((3, 3), "west") == (2, 3)


def test_move_north_out_of_bounds_at_top_edge():
    with pytest.raises(OutOfBounds):
        make_module(max_height=5)._move((0, 4), "north")


def test_move_south_out_of_bounds_at_bottom_edge():
    with pytest.raises(OutOfBounds):
        make_module(min_height=0)._move((0, 0), "south")


def test_move_east_out_of_bounds_at_right_edge():
    with pytest.raises(OutOfBounds):
        make_module(max_width=5)._move((4, 0), "east")


def test_move_west_out_of_bounds_at_left_edge():
    with pytest.raises(OutOfBounds):
        make_module(min_width=0)._move((0, 0), "west")


def test_calculate_distance():
    mm = make_module()
    assert mm._calculate_distance((0, 0), (3, 4)) == pytest.approx(5.0)
    assert mm._calculate_distance((1, 1), (1, 1)) == 0


def test_direction_to_string_all_four_directions():
    mm = make_module()
    # start is later along the path than end (its parent in came_from)
    assert mm._direction_to_string((0, 1), (0, 0)) == "north"  # step was +1 in y
    assert mm._direction_to_string((0, 0), (0, 1)) == "south"  # step was -1 in y
    assert mm._direction_to_string((1, 0), (0, 0)) == "east"   # step was +1 in x
    assert mm._direction_to_string((0, 0), (1, 0)) == "west"   # step was -1 in x


@pytest.mark.parametrize(
    "start,goal",
    [
        ((0, 0), (0, 3)),  # straight north
        ((0, 3), (0, 0)),  # straight south
        ((0, 0), (3, 0)),  # straight east
        ((3, 0), (0, 0)),  # straight west
        ((0, 0), (3, 3)),  # requires a mix of directions
        ((3, 3), (0, 0)),  # requires a mix of directions, reversed
        ((2, 2), (2, 2)),  # already there
        ((0, 0), (9, 9)),  # corner to corner on a 10x10 grid
    ],
)
def test_path_following_the_computed_directions_arrives_at_the_goal(start, goal):
    """
    Regression test for the north/south/east/west relabelling: whatever the internal string labels
    mean, following the computed path step-by-step via _move must land exactly on the goal.
    """
    mm = make_module(max_width=10, max_height=10)
    path = mm._find_path_to_berry(start, goal)
    pos = start
    for step in path:
        pos = mm._move(pos, step)
    assert pos == goal
    manhattan_distance = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
    assert len(path) == manhattan_distance


def test_find_path_to_berry_raises_when_unreachable():
    # a 1x1 allotment can't reach a point outside it
    mm = make_module(min_width=0, max_width=1, min_height=0, max_height=1)
    with pytest.raises(NoPathFound):
        mm._find_path_to_berry((0, 0), (5, 5))


def test_get_distance_to_berry_no_path():
    assert make_module().get_distance_to_berry() == 0


def test_get_distance_to_berry_with_path():
    mm = make_module()
    mm.path = ["north", "north", "east"]
    mm.path_step = 1
    assert mm.get_distance_to_berry() == 2


# --- restrict_to_allocation: berry-access restriction is independent of training ---

def _make_module_with_berry(allocation_id, berry_allocation_id, restrict_to_allocation, training=False):
    berry = types.SimpleNamespace(agent_type="berry", unique_id=5, allocation_id=berry_allocation_id, foraged=False)
    fake_model = types.SimpleNamespace(get_cell_contents=lambda cell: [berry])
    mm = MovingModule(
        agent_id=1, model=fake_model, training=training, allotment=[0, 10, 0, 10],
        allocation_id=allocation_id, restrict_to_allocation=restrict_to_allocation,
    )
    mm.nearest_berry = berry
    return mm, berry


def test_forage_raises_illegal_berry_when_restricted_and_allocation_mismatches():
    mm, _ = _make_module_with_berry(allocation_id="allocation_1", berry_allocation_id="allocation_0",
                                     restrict_to_allocation=True)
    with pytest.raises(IllegalBerry):
        mm._forage((0, 0))


def test_forage_succeeds_when_restricted_and_allocation_matches():
    mm, berry = _make_module_with_berry(allocation_id="allocation_1", berry_allocation_id="allocation_1",
                                         restrict_to_allocation=True)
    assert mm._forage((0, 0)) is True
    assert berry.foraged is True


def test_forage_succeeds_regardless_of_allocation_when_unrestricted():
    """
    Regression test: restrict_to_allocation (not training) now gates berry-allocation checks, so
    an unrestricted agent (e.g. basic_harvest, even if evaluated with training=False) can still
    forage any berry regardless of an allocation mismatch.
    """
    mm, berry = _make_module_with_berry(allocation_id="allocation_1", berry_allocation_id="allocation_0",
                                         restrict_to_allocation=False, training=False)
    assert mm._forage((0, 0)) is True
    assert berry.foraged is True

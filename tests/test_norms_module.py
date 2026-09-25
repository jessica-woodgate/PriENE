import pytest

from src.agent.modules.norms_module import NormsModule


# --- get_antecedent: berries/health/well-being thresholds ---

def test_antecedent_no_berries_low_health_low_days():
    nm = NormsModule(agent_id=1)
    assert nm.get_antecedent(berries=0, health=0.5, well_being=[5]) == "IF,no berries,low health,low days"


def test_antecedent_medium_berries_medium_health_mixed_days():
    nm = NormsModule(agent_id=1)
    # berries=2 -> medium (1 <= 2 < 3); health=1.0 -> medium (0.6 <= 1.0 < 2.0)
    assert nm.get_antecedent(berries=2, health=1.0, well_being=[15, 35]) == (
        "IF,medium berries,medium health,medium days,high days"
    )


def test_antecedent_high_berries_high_health_no_other_agents():
    nm = NormsModule(agent_id=1)
    assert nm.get_antecedent(berries=5, health=3.0, well_being=[]) == "IF,high berries,high health"


def test_antecedent_low_berries_branch_is_only_reachable_for_non_integer_counts():
    """
    low_berries_threshold=1, so `berries > 0 and berries < 1` can never be true for an integer
    berry count (0 or >=1) -- this branch is effectively dead code in the running system, since
    HarvestAgent.berries is always a non-negative integer. Documented here rather than changed.
    """
    nm = NormsModule(agent_id=1)
    assert nm.get_antecedent(berries=0.5, health=1.0, well_being=[]) == "IF,low berries,medium health"


@pytest.mark.parametrize(
    "health,expected",
    [
        (0.59, "low health"),
        (0.6, "medium health"),   # lower bound is inclusive
        (1.99, "medium health"),
        (2.0, "high health"),     # upper bound flips to high
    ],
)
def test_antecedent_health_thresholds(health, expected):
    nm = NormsModule(agent_id=1)
    antecedent = nm.get_antecedent(berries=5, health=health, well_being=[])
    assert antecedent.split(",")[2] == expected


@pytest.mark.parametrize(
    "berries,expected",
    [
        (0, "no berries"),
        (1, "medium berries"),  # lower bound is inclusive
        (2, "medium berries"),
        (3, "high berries"),    # upper bound flips to high
    ],
)
def test_antecedent_berries_thresholds(berries, expected):
    nm = NormsModule(agent_id=1)
    antecedent = nm.get_antecedent(berries=berries, health=3.0, well_being=[])
    assert antecedent.split(",")[1] == expected


@pytest.mark.parametrize(
    "well_being_value,expected",
    [
        (9.99, "low days"),
        (10, "medium days"),   # lower bound is inclusive
        (29.99, "medium days"),
        (30, "high days"),     # upper bound flips to high
    ],
)
def test_antecedent_well_being_thresholds(well_being_value, expected):
    nm = NormsModule(agent_id=1)
    antecedent = nm.get_antecedent(berries=5, health=3.0, well_being=[well_being_value])
    assert antecedent.split(",")[3] == expected


# --- get_consequent ---

def test_consequent_move_action():
    nm = NormsModule(agent_id=1)
    assert nm.get_consequent("move") == "THEN,move"


def test_consequent_eat_action():
    nm = NormsModule(agent_id=1)
    assert nm.get_consequent("eat") == "THEN,eat"


def test_consequent_throw_action():
    nm = NormsModule(agent_id=1)
    assert nm.get_consequent("throw_2") == "THEN,throw"


def test_consequent_cardinal_directions_also_map_to_move():
    """
    In practice HarvestAgent only ever passes "move"/"eat"/"throw_X" (its DQN action names) into
    get_consequent -- "north"/"south"/"east"/"west" are internal to MovingModule and never reach
    here -- but the function itself still handles them if called directly, so pin that down too.
    """
    nm = NormsModule(agent_id=1)
    for direction in ["north", "south", "east", "west"]:
        assert nm.get_consequent(direction) == "THEN,move"


# --- behaviour base updates ---

def test_update_behaviour_base_creates_new_norm():
    nm = NormsModule(agent_id=1)
    nm.update_behaviour_base("IF,no berries,low health", "move", 0.5, day=1, episode=1)
    key = "IF,no berries,low health,THEN,move"
    assert key in nm.behaviour_base
    norm = nm.behaviour_base[key]
    assert norm["reward"] == 0.5
    assert norm["numerosity"] == 1
    assert norm["age"] == 1  # _update_behaviours_age runs once immediately after creation
    assert norm["fitness"] == 0  # fitness is only (re)computed when an *existing* norm is updated


def test_update_behaviour_base_accumulates_on_repeat():
    nm = NormsModule(agent_id=1)
    nm.update_behaviour_base("IF,no berries,low health", "move", 0.5, day=1, episode=1)
    nm.update_behaviour_base("IF,no berries,low health", "move", 0.5, day=2, episode=1)
    key = "IF,no berries,low health,THEN,move"
    norm = nm.behaviour_base[key]
    assert norm["reward"] == pytest.approx(1.0)
    assert norm["numerosity"] == 2
    assert norm["age"] == 2
    # fitness computed on the second call using age=1 (before that call's age increment):
    # fitness = numerosity(2) * reward(1.0) * (decay_rate(0.3) * age(1)) = 0.6
    assert norm["fitness"] == pytest.approx(0.6)


def test_update_norm_fitness_zero_age_leaves_fitness_unchanged():
    nm = NormsModule(agent_id=1)
    norm = {"reward": 5, "numerosity": 2, "age": 0, "fitness": 0}
    nm._update_norm_fitness(norm)
    assert norm["fitness"] == 0


def test_update_norm_fitness_formula():
    nm = NormsModule(agent_id=1)
    norm = {"reward": 2, "numerosity": 3, "age": 4, "fitness": 0}
    nm._update_norm_fitness(norm)
    # fitness = numerosity * reward * (decay_rate * age) = 3 * 2 * (0.3 * 4) = 7.2
    assert norm["fitness"] == pytest.approx(7.2)


def test_clip_behaviour_base_noop_when_under_limit():
    nm = NormsModule(agent_id=1)
    nm.max_norms = 10
    nm.behaviour_base = {"a": {"reward": 1, "numerosity": 1, "age": 1, "fitness": 0}}
    nm._clip_behaviour_base(day=10, episode=1)
    assert len(nm.behaviour_base) == 1


def test_clip_behaviour_base_keeps_highest_fitness_entries():
    nm = NormsModule(agent_id=1)
    nm.max_norms = 3
    # same numerosity/age for every entry so fitness ranking is driven purely by reward
    nm.behaviour_base = {
        "a": {"reward": 1, "numerosity": 1, "age": 1, "fitness": 0},
        "b": {"reward": 5, "numerosity": 1, "age": 1, "fitness": 0},
        "c": {"reward": 9, "numerosity": 1, "age": 1, "fitness": 0},
        "d": {"reward": 3, "numerosity": 1, "age": 1, "fitness": 0},
        "e": {"reward": 7, "numerosity": 1, "age": 1, "fitness": 0},
    }
    nm._clip_behaviour_base(day=10, episode=1)
    assert len(nm.behaviour_base) == 3
    assert set(nm.behaviour_base.keys()) == {"c", "e", "b"}


def test_update_behaviour_base_triggers_clipping_on_clipping_frequency_day():
    nm = NormsModule(agent_id=1)
    nm.max_norms = 1
    nm.norm_clipping_frequency = 10
    nm.update_behaviour_base("IF,a", "move", 1.0, day=1, episode=1)
    nm.update_behaviour_base("IF,b", "eat", 5.0, day=10, episode=1)  # day % 10 == 0 -> clips
    assert len(nm.behaviour_base) == 1

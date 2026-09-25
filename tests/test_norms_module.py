import pytest

from src.agent.modules.norms_module import NormsModule

# matches exactly what HarvestAgent._antecedent_features()/_consequent_rules() register in practice
BERRIES_HEALTH_DAYS_FEATURES = [
    {"boundaries": [1, 3], "labels": ["low berries", "medium berries", "high berries"], "zero_label": "no berries"},
    {"boundaries": [0.6, 2.0], "labels": ["low health", "medium health", "high health"]},
    {"boundaries": [10, 30], "labels": ["low days", "medium days", "high days"], "repeated": True},
]
THROW_CONSEQUENT_RULES = [{"prefix": "throw", "label": "throw"}]


def make_norms_module(antecedent_features=BERRIES_HEALTH_DAYS_FEATURES, consequent_rules=THROW_CONSEQUENT_RULES):
    return NormsModule(agent_id=1, antecedent_features=antecedent_features, consequent_rules=consequent_rules)


# --- get_antecedent: berries/health/well-being thresholds (the feature set HarvestAgent registers) ---

def test_antecedent_no_berries_low_health_low_days():
    nm = make_norms_module()
    assert nm.get_antecedent([0, 0.5, [5]]) == "IF,no berries,low health,low days"


def test_antecedent_medium_berries_medium_health_mixed_days():
    nm = make_norms_module()
    # berries=2 -> medium (1 <= 2 < 3); health=1.0 -> medium (0.6 <= 1.0 < 2.0)
    assert nm.get_antecedent([2, 1.0, [15, 35]]) == (
        "IF,medium berries,medium health,medium days,high days"
    )


def test_antecedent_high_berries_high_health_no_other_agents():
    nm = make_norms_module()
    assert nm.get_antecedent([5, 3.0, []]) == "IF,high berries,high health"


def test_antecedent_low_berries_branch_is_only_reachable_for_non_integer_counts():
    """
    The berries spec's first boundary is 1, so `0 < berries < 1` can never be true for an integer
    berry count (0 or >=1) -- this branch is effectively dead code in the running system, since
    HarvestAgent.berries is always a non-negative integer. Documented here rather than changed.
    """
    nm = make_norms_module()
    assert nm.get_antecedent([0.5, 1.0, []]) == "IF,low berries,medium health"


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
    nm = make_norms_module()
    antecedent = nm.get_antecedent([5, health, []])
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
    nm = make_norms_module()
    antecedent = nm.get_antecedent([berries, 3.0, []])
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
    nm = make_norms_module()
    antecedent = nm.get_antecedent([5, 3.0, [well_being_value]])
    assert antecedent.split(",")[3] == expected


def test_antecedent_is_driven_entirely_by_registered_features_not_hardcoded_names():
    """
    NormsModule has no built-in knowledge of "berries"/"health"/"days" -- it just buckets whatever
    feature specs it was constructed with. Prove that with a completely unrelated feature set.
    """
    features = [
        {"boundaries": [50], "labels": ["cold", "hot"]},
        {"boundaries": [1, 5], "labels": ["quiet", "busy", "crowded"], "repeated": True},
    ]
    nm = NormsModule(agent_id=1, antecedent_features=features, consequent_rules=[])
    assert nm.get_antecedent([70, [0, 3, 8]]) == "IF,hot,quiet,busy,crowded"


# --- get_consequent ---

def test_consequent_move_action():
    nm = make_norms_module()
    assert nm.get_consequent("move") == "THEN,move"


def test_consequent_eat_action():
    nm = make_norms_module()
    assert nm.get_consequent("eat") == "THEN,eat"


def test_consequent_throw_action():
    nm = make_norms_module()
    assert nm.get_consequent("throw_2") == "THEN,throw"


def test_consequent_falls_back_to_the_raw_action_when_no_rule_matches():
    nm = make_norms_module()
    assert nm.get_consequent("jump") == "THEN,jump"


def test_consequent_rules_are_driven_entirely_by_registration_not_hardcoded_actions():
    """
    NormsModule has no built-in knowledge of "throw" specifically -- it just applies whatever
    prefix rules it was constructed with, in order, falling back to the raw action string.
    """
    nm = NormsModule(
        agent_id=1,
        antecedent_features=[],
        consequent_rules=[{"prefix": "attack_", "label": "attack"}],
    )
    assert nm.get_consequent("attack_goblin") == "THEN,attack"
    assert nm.get_consequent("flee") == "THEN,flee"


# --- behaviour base updates (independent of antecedent_features/consequent_rules) ---

def test_update_behaviour_base_creates_new_norm():
    nm = make_norms_module()
    nm.update_behaviour_base("IF,no berries,low health", "move", 0.5, day=1, episode=1)
    key = "IF,no berries,low health,THEN,move"
    assert key in nm.behaviour_base
    norm = nm.behaviour_base[key]
    assert norm["reward"] == 0.5
    assert norm["numerosity"] == 1
    assert norm["age"] == 1  # _update_behaviours_age runs once immediately after creation
    assert norm["fitness"] == 0  # fitness is only (re)computed when an *existing* norm is updated


def test_update_behaviour_base_accumulates_on_repeat():
    nm = make_norms_module()
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
    nm = make_norms_module()
    norm = {"reward": 5, "numerosity": 2, "age": 0, "fitness": 0}
    nm._update_norm_fitness(norm)
    assert norm["fitness"] == 0


def test_update_norm_fitness_formula():
    nm = make_norms_module()
    norm = {"reward": 2, "numerosity": 3, "age": 4, "fitness": 0}
    nm._update_norm_fitness(norm)
    # fitness = numerosity * reward * (decay_rate * age) = 3 * 2 * (0.3 * 4) = 7.2
    assert norm["fitness"] == pytest.approx(7.2)


def test_clip_behaviour_base_noop_when_under_limit():
    nm = make_norms_module()
    nm.max_norms = 10
    nm.behaviour_base = {"a": {"reward": 1, "numerosity": 1, "age": 1, "fitness": 0}}
    nm._clip_behaviour_base(day=10, episode=1)
    assert len(nm.behaviour_base) == 1


def test_clip_behaviour_base_keeps_highest_fitness_entries():
    nm = make_norms_module()
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
    nm = make_norms_module()
    nm.max_norms = 1
    nm.norm_clipping_frequency = 10
    nm.update_behaviour_base("IF,a", "move", 1.0, day=1, episode=1)
    nm.update_behaviour_base("IF,b", "eat", 5.0, day=10, episode=1)  # day % 10 == 0 -> clips
    assert len(nm.behaviour_base) == 1

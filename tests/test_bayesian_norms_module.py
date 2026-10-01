import numpy as np
import pytest

from src.agent.modules.bayesian_norms_module import BayesianNormsModule
from src.harvest_exception import UnmappedActionException

# the action set a 4-agent HarvestAgent actually has (see HarvestAgent._generate_actions)
ACTIONS = ["move", "eat", "throw_1", "throw_2", "throw_3"]
ACTION_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
EAT = ACTION_INDEX["eat"]
MOVE = ACTION_INDEX["move"]

# a single predicate, berries<1, so every candidate shares one precondition
NO_BERRIES_SPEC = [{"name": "berries", "index": 0, "thresholds": [1], "direction": "<"}]
PRECONDITION_MET = np.array([0.0])
PRECONDITION_NOT_MET = np.array([5.0])

# matches exactly what HarvestAgent._build_norm_base registers for a 4-agent society
HARVEST_SPECS = [
    {"name": "health", "index": 0, "thresholds": [1, 5, 10], "direction": "<", "repeated": False},
    {"name": "berries", "index": 1, "thresholds": [1, 5, 10], "direction": "<", "repeated": False},
    {"name": "self_wellbeing", "index": 2, "thresholds": [10, 50, 100], "direction": "<", "repeated": False},
    {"name": "distance", "index": 3, "thresholds": [1, 5, 10], "direction": ">", "repeated": False},
    {"name": "other_wellbeing", "index": [4, 5, 6], "thresholds": [10, 50, 100], "direction": "<", "repeated": True},
]


def make_module(feature_specs=NO_BERRIES_SPEC, max_predicates=1, epsilon=0.1, **kwargs):
    return BayesianNormsModule(
        feature_specs, ACTIONS, max_predicates=max_predicates,
        action_name_to_index=ACTION_INDEX, epsilon=epsilon, **kwargs,
    )


def behaviour(module, behaviour_type, action):
    """The single behaviour with this type/action (only unambiguous for single-predicate specs)."""
    matches = [b for b in module.behaviour_base.values() if b["type"] == behaviour_type and b["action"] == action]
    assert len(matches) == 1
    return matches[0]


def learned_labels(module):
    """Learned behaviours as short labels, e.g. "eat" / "NOT move" (single-predicate specs only)."""
    return sorted(b["label"].split(" THEN ")[1] for b in module.get_learned_behaviours())


def run_policy(module, policy, steps, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        module.update(PRECONDITION_MET, policy(rng))


def always_eat(rng):
    return EAT


def random_action(rng):
    return int(rng.integers(len(ACTIONS)))


def eat_ninety_percent(rng):
    return EAT if rng.random() < 0.9 else random_action(rng)


def eat_or_move(rng):
    return EAT if rng.random() < 0.5 else MOVE


# --- candidate generation ---

def test_harvest_specs_generate_expected_number_of_candidates():
    """
    21 predicates (3 thresholds x (4 scalar features + 3 other-agent well-being readings)),
    preconditions of 1 or 2 predicates = 21 + C(21,2) = 231, x 5 actions x 2 types = 2310.
    """
    module = make_module(HARVEST_SPECS, max_predicates=2)
    assert len(module._build_predicates()) == 21
    assert len(module.behaviour_base) == 2310


def test_small_spec_candidate_count():
    # 2 predicates -> 2 singles + 1 pair = 3 preconditions, x 5 actions x 2 types
    spec = [{"name": "berries", "index": 0, "thresholds": [1, 3], "direction": "<"}]
    assert len(make_module(spec, max_predicates=2).behaviour_base) == 3 * 5 * 2


def test_candidate_ids_are_unique_and_consecutive():
    module = make_module(HARVEST_SPECS, max_predicates=2)
    assert sorted(module.behaviour_base.keys()) == list(range(len(module.behaviour_base)))
    assert all(behaviour_id == b["id"] for behaviour_id, b in module.behaviour_base.items())


def test_every_precondition_action_pair_has_one_prohibition_and_one_obligation():
    module = make_module(HARVEST_SPECS, max_predicates=2)
    pairs = {}
    for b in module.behaviour_base.values():
        pairs.setdefault((tuple(b["precondition"]), b["action"]), []).append(b["type"])
    assert all(sorted(types) == ["obligation", "prohibition"] for types in pairs.values())


def test_labels():
    spec = [
        {"name": "berries", "index": 0, "thresholds": [1], "direction": "<"},
        {"name": "distance", "index": 1, "thresholds": [5], "direction": ">"},
    ]
    labels = {b["label"] for b in make_module(spec, max_predicates=2).behaviour_base.values()}
    assert "IF berries<1 THEN NOT eat" in labels
    assert "IF distance>5 THEN throw_2" in labels
    assert "IF berries<1 AND distance>5 THEN NOT move" in labels


def test_repeated_feature_creates_one_predicate_per_index():
    spec = [{"name": "other_wellbeing", "index": [4, 5], "thresholds": [10], "direction": "<", "repeated": True}]
    predicates = make_module(spec)._build_predicates()
    assert [p["index"] for p in predicates] == [4, 5]
    assert [p["name"] for p in predicates] == ["other_wellbeing[0]<10", "other_wellbeing[1]<10"]


def test_labels_are_unique():
    """
    Regression test: repeated-feature predicates used to share a name across indices, so 480 of the
    2310 harvest labels were ambiguous -- and learned norms keyed by label in the per-agent norms
    JSON silently overwrote each other.
    """
    module = make_module(HARVEST_SPECS, max_predicates=2)
    labels = [b["label"] for b in module.behaviour_base.values()]
    assert len(set(labels)) == len(labels)


def test_initial_state():
    module = make_module()
    for b in module.behaviour_base.values():
        assert b["posterior"] == module.prior
        assert b["times_precondition_met"] == 0
        assert b["times_action_matched"] == 0


@pytest.mark.parametrize("action_name_to_index", [None, {"move": 0, "eat": 1}])
def test_unmapped_action_raises(action_name_to_index):
    with pytest.raises(UnmappedActionException):
        BayesianNormsModule(NO_BERRIES_SPEC, ACTIONS, max_predicates=1, action_name_to_index=action_name_to_index)


def test_initialise_resets_posteriors_and_counts():
    module = make_module()
    run_policy(module, always_eat, 20)
    module.initialise()
    for b in module.behaviour_base.values():
        assert b["posterior"] == module.prior
        assert b["times_precondition_met"] == 0
        assert b["times_action_matched"] == 0


def test_reset_restores_prior_and_counts_but_keeps_candidates():
    module = make_module()
    run_policy(module, always_eat, 20)
    ids_before = list(module.behaviour_base)
    module.reset()
    assert list(module.behaviour_base) == ids_before
    for b in module.behaviour_base.values():
        assert b["posterior"] == module.prior
        assert b["times_precondition_met"] == 0
        assert b["times_action_matched"] == 0


# --- _precondition_satisfied ---

@pytest.mark.parametrize(
    "direction,value,expected",
    [
        ("<", 4.99, True),
        ("<", 5.0, False),   # strict: equal to threshold is not below it
        ("<", 5.01, False),
        (">", 5.01, True),
        (">", 5.0, False),   # strict: equal to threshold is not above it
        (">", 4.99, False),
        ("<=", 5.0, True),   # inclusive: the label "x<=5" must hold at exactly 5
        ("<=", 5.01, False),
        (">=", 5.0, True),   # inclusive: the label "x>=5" must hold at exactly 5
        (">=", 4.99, False),
    ],
)
def test_precondition_threshold_boundaries(direction, value, expected):
    spec = [{"name": "x", "index": 0, "thresholds": [5], "direction": direction}]
    module = make_module(spec)
    assert module._precondition_satisfied(module.behaviour_base[0], np.array([value])) == expected


def test_unknown_direction_raises():
    """Regression test: an unrecognised direction used to fall through silently to a ">" check."""
    with pytest.raises(ValueError):
        make_module([{"name": "x", "index": 0, "thresholds": [5], "direction": "=<"}])


def test_two_predicate_precondition_requires_both():
    spec = [
        {"name": "berries", "index": 0, "thresholds": [1], "direction": "<"},
        {"name": "distance", "index": 1, "thresholds": [5], "direction": ">"},
    ]
    module = make_module(spec, max_predicates=2)
    pair = next(b for b in module.behaviour_base.values() if len(b["predicates"]) == 2)
    assert module._precondition_satisfied(pair, np.array([0, 6]))
    assert not module._precondition_satisfied(pair, np.array([0, 4]))
    assert not module._precondition_satisfied(pair, np.array([2, 6]))


# --- update() mechanics ---

def test_update_ignores_behaviours_whose_precondition_is_not_met():
    module = make_module()
    module.update(PRECONDITION_NOT_MET, EAT)
    for b in module.behaviour_base.values():
        assert b["posterior"] == module.prior
        assert b["times_precondition_met"] == 0


def test_update_counts():
    module = make_module()
    module.update(PRECONDITION_MET, EAT)
    module.update(PRECONDITION_MET, MOVE)
    eat_obligation = behaviour(module, "obligation", "eat")
    eat_prohibition = behaviour(module, "prohibition", "eat")
    assert eat_obligation["times_precondition_met"] == 2
    assert eat_obligation["times_action_matched"] == 1     # consistent only when eating
    assert eat_prohibition["times_action_matched"] == 1    # consistent only when not eating


def test_single_update_matches_hand_computed_bayes():
    """
    Obligation "eat", agent eats (consistent), epsilon=0.1, 5 actions:
    P(consistent | learned) = 0.9, P(consistent | not learned) = 1/5 = 0.2
    posterior = 0.9 * 0.05 / (0.9 * 0.05 + 0.2 * 0.95) = 0.045 / 0.235
    """
    module = make_module(epsilon=0.1)
    module.update(PRECONDITION_MET, EAT)
    assert behaviour(module, "obligation", "eat")["posterior"] == pytest.approx(0.045 / 0.235)


def test_likelihood_floor_holds_when_epsilon_is_zero():
    """
    In testing, epsilon can be 0 -- the 0.01 floor on the likelihoods must stop a single slip from
    driving the posterior to exactly 0, and give the same result as epsilon=0.01.
    """
    posteriors = []
    for epsilon in [0.0, 1e-6, 0.01]:
        module = make_module(epsilon=epsilon)
        module.update(PRECONDITION_MET, MOVE)  # a slip against obligation "eat"
        posteriors.append(behaviour(module, "obligation", "eat")["posterior"])
    assert posteriors[0] > 0
    assert posteriors == pytest.approx([posteriors[0]] * 3)


@pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.1])
def test_posteriors_stay_finite_and_in_unit_interval(epsilon):
    module = make_module(epsilon=epsilon)
    run_policy(module, eat_ninety_percent, 500)
    for b in module.behaviour_base.values():
        assert np.isfinite(b["posterior"])
        assert 0.0 <= b["posterior"] <= 1.0


# --- posterior behaviour under synthetic policies ---

EXPECTED_ALWAYS_EAT = sorted(["eat", "NOT move", "NOT throw_1", "NOT throw_2", "NOT throw_3"])


@pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.1])
def test_random_policy_learns_nothing(epsilon):
    module = make_module(epsilon=epsilon)
    run_policy(module, random_action, 500)
    assert module.get_learned_behaviours() == []


@pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.1])
@pytest.mark.parametrize("policy", [always_eat, eat_ninety_percent])
def test_consistent_eating_learns_exactly_the_eating_behaviours(epsilon, policy):
    module = make_module(epsilon=epsilon)
    run_policy(module, policy, 500)
    assert learned_labels(module) == EXPECTED_ALWAYS_EAT


@pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.1])
def test_mixed_eat_move_policy_only_learns_not_throwing(epsilon):
    module = make_module(epsilon=epsilon)
    run_policy(module, eat_or_move, 500)
    assert learned_labels(module) == ["NOT throw_1", "NOT throw_2", "NOT throw_3"]


@pytest.mark.parametrize("policy", [always_eat, random_action, eat_ninety_percent, eat_or_move])
def test_behaviour_and_its_opposite_are_never_both_learned(policy):
    module = make_module(epsilon=0.1)
    run_policy(module, policy, 500)
    learned = {(b["action"], b["type"]) for b in module.get_learned_behaviours()}
    for action in ACTIONS:
        assert not {(action, "prohibition"), (action, "obligation")} <= learned


def test_epsilon_at_or_above_inverse_num_actions_inverts_prohibitions():
    """
    Documents a known limit rather than changing it: with P(inconsistent | learned) = epsilon and
    P(inconsistent | not learned) = 1/n_actions, any epsilon > 1/n_actions (0.2 here) makes every
    violation of a prohibition count as evidence FOR it. Callers must keep epsilon below
    1/n_actions (a training run's starting epsilon of 0.9 would break this).
    """
    module = make_module(epsilon=0.3)
    run_policy(module, always_eat, 200)
    assert "NOT eat" in learned_labels(module)


# --- get_learned_behaviours / get_all_behaviours_by_certainty ---

def test_learned_behaviours_require_min_observations():
    module = make_module(epsilon=0.1, min_observations=5)
    run_policy(module, always_eat, 4)   # obligation "eat" crosses the threshold before 5 steps
    assert behaviour(module, "obligation", "eat")["posterior"] >= module.learned_threshold
    assert module.get_learned_behaviours() == []
    run_policy(module, always_eat, 1)
    assert "eat" in learned_labels(module)


def test_learned_behaviours_sorted_by_posterior_descending():
    module = make_module()
    run_policy(module, always_eat, 50)
    posteriors = [b["posterior"] for b in module.get_learned_behaviours()]
    assert posteriors == sorted(posteriors, reverse=True)


def test_all_behaviours_by_certainty_returns_every_behaviour_sorted():
    module = make_module()
    run_policy(module, eat_ninety_percent, 50)
    ordered = module.get_all_behaviours_by_certainty()
    assert len(ordered) == len(module.behaviour_base)
    posteriors = [b["posterior"] for b in ordered]
    assert posteriors == sorted(posteriors, reverse=True)

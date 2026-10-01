import numpy as np
import pytest

from src.agent.modules.bayesian_norms_module import BayesianNormsModule
from src.harvest_exception import UnmappedActionException

# the norm actions HarvestAgent registers (every throw_X collapses to "throw")
ACTIONS = ["move", "eat", "throw"]
ACTION_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}
MOVE = ACTION_INDEX["move"]
EAT = ACTION_INDEX["eat"]
THROW = ACTION_INDEX["throw"]

# one threshold -> two bins (two "states"): berries<1 and berries>=1
NO_BERRIES_SPEC = [{"name": "berries", "index": 0, "thresholds": [1]}]
NO_BERRIES = "berries<1"
HAS_BERRIES = "berries>=1"
PRECONDITION_MET = np.array([0.0])      # in the berries<1 bin
PRECONDITION_NOT_MET = np.array([5.0])  # in the berries>=1 bin

# matches exactly what HarvestAgent._build_norm_base registers for a 4-agent society
HARVEST_SPECS = [
    {"name": "health", "index": 0, "thresholds": [1, 5, 10], "repeated": False},
    {"name": "berries", "index": 1, "thresholds": [1, 2, 3], "repeated": False},
    {"name": "self_wellbeing", "index": 2, "thresholds": [10, 50, 100], "repeated": False},
    {"name": "distance", "index": 3, "thresholds": [1, 3, 7], "repeated": False},
    {"name": "other_wellbeing", "index": [4, 5, 6], "thresholds": [0.1, 10, 50, 100], "repeated": True},
]


def make_module(feature_specs=NO_BERRIES_SPEC, max_predicates=1, epsilon=0.1, **kwargs):
    return BayesianNormsModule(
        feature_specs, ACTIONS, max_predicates=max_predicates,
        action_name_to_index=ACTION_INDEX, epsilon=epsilon, **kwargs,
    )


def behaviour(module, behaviour_type, action, precondition=NO_BERRIES):
    """The single behaviour with this type/action and one-predicate precondition."""
    matches = [b for b in module.behaviour_base.values()
               if b["type"] == behaviour_type and b["action"] == action and b["precondition"] == [precondition]]
    assert len(matches) == 1
    return matches[0]


def learned_labels(module):
    """Learned behaviours as labels without the "IF ", e.g. "berries<1 THEN NOT move"."""
    return sorted(b["label"].removeprefix("IF ") for b in module.get_learned_behaviours())


def run_policy(module, policy_without_berries, policy_with_berries, steps, seed=0):
    """Alternate between the two states, acting by the given policy in each."""
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        module.update(PRECONDITION_MET, policy_without_berries(rng))
        module.update(PRECONDITION_NOT_MET, policy_with_berries(rng))


def always_eat(rng):
    return EAT


def always_move(rng):
    return MOVE


def random_action(rng):
    return int(rng.integers(len(ACTIONS)))


def eat_ninety_percent(rng):
    return EAT if rng.random() < 0.9 else random_action(rng)


def move_ninety_percent(rng):
    return MOVE if rng.random() < 0.9 else random_action(rng)


def eat_or_move(rng):
    return EAT if rng.random() < 0.5 else MOVE


def rarely_throw(rng):
    return THROW if rng.random() < 0.05 else eat_or_move(rng)


def throw_or_move(rng):
    return THROW if rng.random() < 0.5 else MOVE


# --- candidate generation ---

def test_harvest_specs_generate_expected_number_of_candidates():
    """
    4 scalar features x 4 bins + 3 other-agent well-being readings x 5 bins (incl. <0.1 = dead) =
    31 predicates. Pairs: C(31,2) = 465, minus pairs of two bins of the same feature
    (4 x C(4,2) + 3 x C(5,2) = 54) = 411. (31 + 411) preconditions x 3 norm actions x 2 types = 2652.
    """
    module = make_module(HARVEST_SPECS, max_predicates=2)
    assert len(module._build_predicates()) == 31
    assert len(module.behaviour_base) == 2652


def test_bins_of_one_feature_are_never_paired():
    """Bins don't overlap, so e.g. "health<1 AND 1<=health<5" could never be met."""
    spec = [{"name": "berries", "index": 0, "thresholds": [1, 3]}]
    module = make_module(spec, max_predicates=2)
    assert len(module.behaviour_base) == 3 * 3 * 2  # 3 bins, no pairs
    harvest = make_module(HARVEST_SPECS, max_predicates=2)
    for b in harvest.behaviour_base.values():
        indices = [p["index"] for p in b["predicates"]]
        assert len(set(indices)) == len(indices)


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


def test_bin_names():
    spec = [{"name": "health", "index": 0, "thresholds": [5, 1, 10]}]  # unsorted on purpose
    names = [p["name"] for p in make_module(spec)._build_predicates()]
    assert names == ["health<1", "1<=health<5", "5<=health<10", "health>=10"]


def test_labels():
    spec = [
        {"name": "berries", "index": 0, "thresholds": [1]},
        {"name": "distance", "index": 1, "thresholds": [5]},
    ]
    labels = {b["label"] for b in make_module(spec, max_predicates=2).behaviour_base.values()}
    assert "IF berries<1 THEN NOT eat" in labels
    assert "IF distance>=5 THEN throw" in labels
    assert "IF berries<1 AND distance>=5 THEN NOT move" in labels


def test_repeated_feature_creates_one_set_of_bins_per_index():
    spec = [{"name": "other_wellbeing", "index": [4, 5], "thresholds": [10], "repeated": True}]
    predicates = make_module(spec)._build_predicates()
    assert [p["index"] for p in predicates] == [4, 4, 5, 5]
    assert [p["name"] for p in predicates] == [
        "other_wellbeing[0]<10", "other_wellbeing[0]>=10", "other_wellbeing[1]<10", "other_wellbeing[1]>=10",
    ]


def test_labels_are_unique():
    """
    Regression test: repeated-feature predicates used to share a name across indices, so 480 of the
    harvest labels were ambiguous -- and learned norms keyed by label in the per-agent norms
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


def test_lookup_matches_every_behaviour_whose_precondition_is_met():
    """update() only visits behaviours found via the precondition lookup table -- it must find
    exactly those a full scan of the behaviour base would, each once, including values on bin edges."""
    module = make_module(HARVEST_SPECS, max_predicates=2)
    thresholds = [spec["thresholds"] for spec in HARVEST_SPECS[:4]] + [HARVEST_SPECS[4]["thresholds"] + [0]] * 3
    rng = np.random.default_rng(0)
    for _ in range(500):
        observation = np.array([
            rng.choice(t) if rng.random() < 0.3 else rng.uniform(-5, 150) for t in thresholds
        ], dtype=float)
        found = [b["id"] for b in module._matching_behaviours(observation)]
        expected = [b["id"] for b in module.behaviour_base.values() if module._precondition_satisfied(b, observation)]
        assert sorted(found) == sorted(expected)
        assert len(found) == len(set(found))


def test_lookup_table_shares_behaviour_dicts_with_behaviour_base():
    """reset(), the model's norm writers and the tests all read/modify behaviour_base -- the lookup
    must hold the same dict objects, not copies, or updates and resets would diverge."""
    module = make_module(HARVEST_SPECS, max_predicates=2)
    in_lookup = [b for behaviours in module._rules_by_precondition.values() for b in behaviours]
    assert len(in_lookup) == len(module.behaviour_base)
    assert all(b is module.behaviour_base[b["id"]] for b in in_lookup)


def test_reset_restores_prior_and_counts_but_keeps_candidates_and_action_counts():
    module = make_module()
    run_policy(module, always_eat, always_move, 20)
    ids_before = list(module.behaviour_base)
    module.reset()
    assert list(module.behaviour_base) == ids_before
    for b in module.behaviour_base.values():
        assert b["posterior"] == module.prior
        assert b["times_precondition_met"] == 0
        assert b["times_action_matched"] == 0
    # the agent's action frequencies are a property of its (fixed) policy, kept across episodes
    assert list(module.action_counts) == [20, 20, 0]


# --- _precondition_satisfied ---

def satisfied_bins(module, value):
    return [b["precondition"][0] for b in module.behaviour_base.values()
            if b["type"] == "prohibition" and b["action"] == "move"
            and module._precondition_satisfied(b, np.array([value]))]


@pytest.mark.parametrize(
    "value,expected_bin",
    [
        (-3.0, "x<1"),
        (0.99, "x<1"),
        (1.0, "1<=x<5"),   # a value on a threshold belongs to the bin above it
        (4.99, "1<=x<5"),
        (5.0, "x>=5"),
        (1e6, "x>=5"),
    ],
)
def test_value_falls_in_exactly_one_bin(value, expected_bin):
    module = make_module([{"name": "x", "index": 0, "thresholds": [1, 5]}])
    assert satisfied_bins(module, value) == [expected_bin]


def test_dead_agent_has_its_own_bin():
    """A dead agent's well-being is observed as exactly 0; living agents' is a whole number of days
    (>= 1, up to float noise), so the 0.1 threshold puts dead agents, and only them, in <0.1."""
    module = make_module([HARVEST_SPECS[4] | {"index": [0], "repeated": True}])
    assert satisfied_bins(module, 0.0) == ["other_wellbeing[0]<0.1"]
    assert satisfied_bins(module, 1.0) == ["0.1<=other_wellbeing[0]<10"]
    assert satisfied_bins(module, 0.9999999999999404) == ["0.1<=other_wellbeing[0]<10"]  # float noise


def test_random_values_always_fall_in_exactly_one_bin_per_feature():
    module = make_module(HARVEST_SPECS)
    rng = np.random.default_rng(0)
    predicates = module._build_predicates()
    for _ in range(200):
        observation = rng.uniform(-10, 200, size=7)
        for index in range(7):
            in_bin = [p for p in predicates if p["index"] == index
                      and p["lower"] <= observation[index] < p["upper"]]
            assert len(in_bin) == 1


def test_two_predicate_precondition_requires_both():
    spec = [
        {"name": "berries", "index": 0, "thresholds": [1]},
        {"name": "distance", "index": 1, "thresholds": [5]},
    ]
    module = make_module(spec, max_predicates=2)
    pair = next(b for b in module.behaviour_base.values() if b["precondition"] == ["berries<1", "distance>=5"])
    assert module._precondition_satisfied(pair, np.array([0, 6]))
    assert not module._precondition_satisfied(pair, np.array([0, 4]))
    assert not module._precondition_satisfied(pair, np.array([2, 6]))


# --- update() mechanics ---

def test_update_ignores_behaviours_whose_precondition_is_not_met():
    module = make_module()
    module.update(PRECONDITION_NOT_MET, EAT)  # berries>=1 bin only
    for b in module.behaviour_base.values():
        if b["precondition"] == [NO_BERRIES]:
            assert b["posterior"] == module.prior
            assert b["times_precondition_met"] == 0
        else:
            assert b["times_precondition_met"] == 1


def test_update_counts():
    module = make_module()
    module.update(PRECONDITION_MET, EAT)
    module.update(PRECONDITION_MET, MOVE)
    eat_obligation = behaviour(module, "obligation", "eat")
    eat_prohibition = behaviour(module, "prohibition", "eat")
    assert eat_obligation["times_precondition_met"] == 2
    assert eat_obligation["times_action_matched"] == 1     # consistent only when eating
    assert eat_prohibition["times_action_matched"] == 1    # consistent only when not eating


def test_action_counts_track_every_update():
    module = make_module()
    for action in [EAT, EAT, MOVE, THROW, EAT]:
        module.update(PRECONDITION_MET, action)
    assert list(module.action_counts) == [1, 3, 1]


def test_single_update_matches_hand_computed_bayes():
    """
    First step, so no actions have been counted yet and the smoothed frequencies are uniform (1/3).
    Obligation "eat", agent eats (consistent), epsilon=0.1:
    P(consistent | learned) = max(0.9, 1/3) = 0.9, P(consistent | not learned) = 1/3
    posterior = 0.9 * 0.05 / (0.9 * 0.05 + 1/3 * 0.95)
    """
    module = make_module(epsilon=0.1)
    module.update(PRECONDITION_MET, EAT)
    expected = 0.9 * 0.05 / (0.9 * 0.05 + (1 / 3) * 0.95)
    assert behaviour(module, "obligation", "eat")["posterior"] == pytest.approx(expected)


def test_update_uses_agents_own_action_frequencies():
    """
    After the agent has eaten 3 times and moved once (smoothed frequency of eat = (3+1)/(4+3) = 4/7),
    eating again is weaker evidence for obligation "eat" than under a uniform baseline:
    P(consistent | not learned) = 4/7.
    """
    module = make_module(epsilon=0.1)
    for action in [EAT, EAT, EAT, MOVE]:
        module.update(PRECONDITION_NOT_MET, action)  # counted, but outside the berries<1 bin
    module.update(PRECONDITION_MET, EAT)
    expected = 0.9 * 0.05 / (0.9 * 0.05 + (4 / 7) * 0.95)
    assert behaviour(module, "obligation", "eat")["posterior"] == pytest.approx(expected)


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
    run_policy(module, eat_ninety_percent, move_ninety_percent, 500)
    for b in module.behaviour_base.values():
        assert np.isfinite(b["posterior"])
        assert 0.0 <= b["posterior"] <= 1.0


# --- posterior behaviour under synthetic policies ---
# A norm is only learned if the agent acts differently whenever its precondition holds than it does
# overall, so the policies differ between the two states (berries<1 / berries>=1).

EATS_WITHOUT_BERRIES_MOVES_WITH = [
    "berries<1 THEN NOT move", "berries<1 THEN eat", "berries>=1 THEN NOT eat", "berries>=1 THEN move",
]


@pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.1])
def test_random_policy_learns_nothing(epsilon):
    module = make_module(epsilon=epsilon)
    run_policy(module, random_action, random_action, 500)
    assert module.get_learned_behaviours() == []


@pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.1])
def test_same_behaviour_in_every_state_learns_nothing(epsilon):
    """Eating everywhere isn't a norm conditioned on any state, so nothing is learned."""
    module = make_module(epsilon=epsilon)
    run_policy(module, always_eat, always_eat, 500)
    assert module.get_learned_behaviours() == []


@pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.1])
@pytest.mark.parametrize("policies", [(always_eat, always_move), (eat_ninety_percent, move_ninety_percent)])
def test_state_dependent_behaviour_learns_exactly_the_state_specific_norms(epsilon, policies):
    """Never throwing anywhere doesn't make "NOT throw" a norm in either state."""
    module = make_module(epsilon=epsilon)
    run_policy(module, *policies, 500)
    assert learned_labels(module) == EATS_WITHOUT_BERRIES_MOVES_WITH


@pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.1])
def test_rare_action_everywhere_is_not_prohibited(epsilon):
    """
    Regression test: against a uniform baseline, any action an agent took in under ~15% of steps
    became "prohibited" in every state (e.g. NOT throw_X for each of 3 targets at ~6% each).
    """
    module = make_module(epsilon=epsilon)
    run_policy(module, rarely_throw, rarely_throw, 500)
    assert module.get_learned_behaviours() == []


@pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.1])
def test_not_throwing_in_one_state_only_is_a_norm(epsilon):
    """Throwing half the time with berries but never without: NOT throw is learned without berries."""
    module = make_module(epsilon=epsilon)
    run_policy(module, eat_or_move, throw_or_move, 500)
    assert "berries<1 THEN NOT throw" in learned_labels(module)
    assert "berries>=1 THEN throw" not in learned_labels(module)  # only 50%, not consistent


@pytest.mark.parametrize("policies", [
    (always_eat, always_move), (random_action, random_action), (eat_ninety_percent, move_ninety_percent),
    (eat_or_move, throw_or_move),
])
def test_behaviour_and_its_opposite_are_never_both_learned(policies):
    module = make_module(epsilon=0.1)
    run_policy(module, *policies, 500)
    learned = {(tuple(b["precondition"]), b["action"], b["type"]) for b in module.get_learned_behaviours()}
    for precondition, action, _ in learned:
        assert not {(precondition, action, "prohibition"), (precondition, action, "obligation")} <= learned


@pytest.mark.parametrize("epsilon", [0.2, 0.3, 0.5])
def test_large_epsilon_does_not_invert_evidence(epsilon):
    """
    Regression test: with P(inconsistent | learned) = epsilon fixed, an epsilon above the baseline
    rate of an action made every violation of its prohibition count as evidence FOR it. P(consistent |
    learned) is now never below P(consistent | not learned), so violations never support a norm.
    """
    module = make_module(epsilon=epsilon)
    run_policy(module, always_eat, always_move, 500)
    learned = learned_labels(module)
    assert "berries<1 THEN NOT eat" not in learned
    assert "berries>=1 THEN NOT move" not in learned


# --- get_learned_behaviours / get_all_behaviours_by_certainty ---

def test_learned_behaviours_require_min_observations():
    module = make_module(min_observations=5)
    eat = behaviour(module, "obligation", "eat")
    eat["posterior"] = 0.99
    eat["times_precondition_met"] = 4
    assert module.get_learned_behaviours() == []
    eat["times_precondition_met"] = 5
    assert module.get_learned_behaviours() == [eat]


def test_learned_behaviours_sorted_by_posterior_descending():
    module = make_module()
    run_policy(module, eat_ninety_percent, move_ninety_percent, 100)
    assert len(module.get_learned_behaviours()) > 1
    posteriors = [b["posterior"] for b in module.get_learned_behaviours()]
    assert posteriors == sorted(posteriors, reverse=True)


def test_all_behaviours_by_certainty_returns_every_behaviour_sorted():
    module = make_module()
    run_policy(module, eat_ninety_percent, move_ninety_percent, 50)
    ordered = module.get_all_behaviours_by_certainty()
    assert len(ordered) == len(module.behaviour_base)
    posteriors = [b["posterior"] for b in ordered]
    assert posteriors == sorted(posteriors, reverse=True)

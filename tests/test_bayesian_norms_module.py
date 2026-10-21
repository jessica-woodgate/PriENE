import numpy as np
import pytest

from src.agent.modules.bayesian_norms_module import OBLIGATION, PROHIBITION, BayesianNormsModule, BehaviourView
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
    """The BehaviourView with this type/action and one-predicate precondition."""
    return BehaviourView(frozenset([precondition]), module.action_name_to_index[action], behaviour_type)


def all_behaviours(module):
    """Every behaviour as a dict (id, type, action, precondition, label, posterior, ...)."""
    return [module.behaviour_dict(view) for view in module.iter_behaviours()]


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
    assert len(list(module.iter_behaviours())) == 2652


def test_total_behaviours_match_old_structure():
    """
    One behaviour_base entry per precondition must still expose preconditions x actions x 2 behaviours --
    the old one-entry-per-behaviour len(behaviour_base) of 2652 for the harvest specs.
    """
    module = make_module(HARVEST_SPECS, max_predicates=2)
    predicates = module._build_predicates()
    n_singles = sum(len(e["predicates"]) == 1 for e in module.behaviour_base.values())
    n_pairs = sum(len(e["predicates"]) == 2 for e in module.behaviour_base.values())
    n_behaviours = len(list(module.iter_behaviours()))
    print(f"\n{len(predicates)} predicates (4 scalar features x 4 bins + 3 other-agent well-being x 5 bins)"
          f"\n{len(module.behaviour_base)} preconditions = {n_singles} single predicates + {n_pairs} pairs"
          f" (C({len(predicates)},2) = {len(predicates) * (len(predicates) - 1) // 2} minus same-feature pairs)"
          f"\n{len(ACTIONS)} actions x 2 types (prohibition, obligation)"
          f"\n{len(module.behaviour_base)} x {len(ACTIONS)} x 2 = {n_behaviours} behaviours")
    assert (len(predicates), n_singles, n_pairs) == (31, 31, 411)
    assert n_behaviours == len(module.behaviour_base) * len(ACTIONS) * 2 == 2652


def test_behaviour_base_entries():
    module = make_module(HARVEST_SPECS, max_predicates=2)
    for key, entry in module.behaviour_base.items():
        assert key == frozenset(entry["precondition"])
        assert [p["name"] for p in entry["predicates"]] == entry["precondition"]
        assert entry["action_counts"].shape == (len(ACTIONS),) and entry["action_counts"].dtype.kind == "i"
        assert entry["posteriors"].shape == (len(ACTIONS), 2) and (entry["posteriors"] == module.prior).all()


def test_bins_of_one_feature_are_never_paired():
    """Bins don't overlap, so e.g. "health<1 AND 1<=health<5" could never be met."""
    spec = [{"name": "berries", "index": 0, "thresholds": [1, 3]}]
    module = make_module(spec, max_predicates=2)
    assert len(module.behaviour_base) == 3  # 3 bins, no pairs
    harvest = make_module(HARVEST_SPECS, max_predicates=2)
    for entry in harvest.behaviour_base.values():
        indices = [p["index"] for p in entry["predicates"]]
        assert len(set(indices)) == len(indices)


def test_behaviour_ids_are_unique_and_stable():
    """String ids "type|action|sorted precondition" are the same for every agent and run."""
    module = make_module(HARVEST_SPECS, max_predicates=2)
    ids = [module.behaviour_id(view) for view in module.iter_behaviours()]
    assert len(set(ids)) == len(ids)
    assert ids == [make_module(HARVEST_SPECS, max_predicates=2).behaviour_id(v) for v in module.iter_behaviours()]
    assert module.behaviour_id(BehaviourView(frozenset(["distance>=7", "health<1"]), THROW, "obligation")) == \
        "obligation|throw|distance>=7 AND health<1"


def test_views_cover_every_entry_action_and_type():
    module = make_module(HARVEST_SPECS, max_predicates=2)
    views = list(module.iter_behaviours())
    assert set(views) == {BehaviourView(key, a, t) for key in module.behaviour_base
                          for a in range(len(ACTIONS)) for t in ("prohibition", "obligation")}
    # per (precondition, action): prohibition then obligation, the original generation order
    assert [v.type for v in views[:4]] == ["prohibition", "obligation"] * 2


def test_posterior_columns():
    module = make_module()
    entry = module.behaviour_base[frozenset([NO_BERRIES])]
    entry["posteriors"][EAT, PROHIBITION] = 0.2
    entry["posteriors"][EAT, OBLIGATION] = 0.7
    assert module.posterior(behaviour(module, "prohibition", "eat")) == 0.2
    assert module.posterior(behaviour(module, "obligation", "eat")) == 0.7


def test_every_precondition_action_pair_has_one_prohibition_and_one_obligation():
    module = make_module(HARVEST_SPECS, max_predicates=2)
    pairs = {}
    for b in all_behaviours(module):
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
    labels = {b["label"] for b in all_behaviours(make_module(spec, max_predicates=2))}
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
    labels = [b["label"] for b in all_behaviours(module)]
    assert len(set(labels)) == len(labels)


def test_initial_state():
    module = make_module()
    for view in module.iter_behaviours():
        assert module.posterior(view) == module.prior
        assert module.behaviour_counts(*view) == (0, 0)


@pytest.mark.parametrize("action_name_to_index", [None, {"move": 0, "eat": 1}])
def test_unmapped_action_raises(action_name_to_index):
    with pytest.raises(UnmappedActionException):
        BayesianNormsModule(NO_BERRIES_SPEC, ACTIONS, max_predicates=1, action_name_to_index=action_name_to_index)


def test_lookup_matches_every_precondition_that_is_met():
    """update() only visits preconditions found via the behaviour_base lookup -- it must find exactly
    those a full scan of the behaviour base would, each once, including values on bin edges."""
    module = make_module(HARVEST_SPECS, max_predicates=2)
    thresholds = [spec["thresholds"] for spec in HARVEST_SPECS[:4]] + [HARVEST_SPECS[4]["thresholds"] + [0]] * 3
    rng = np.random.default_rng(0)
    for _ in range(500):
        observation = np.array([
            rng.choice(t) if rng.random() < 0.3 else rng.uniform(-5, 150) for t in thresholds
        ], dtype=float)
        found = list(module._matching_preconditions(observation))
        expected = [key for key, entry in module.behaviour_base.items()
                    if module._precondition_satisfied(entry, observation)]
        assert sorted(found, key=sorted) == sorted(expected, key=sorted)
        assert len(found) == len(set(found))


def test_reset_restores_prior_and_counts_but_keeps_candidates_and_action_counts():
    module = make_module()
    run_policy(module, always_eat, always_move, 20)
    keys_before = list(module.behaviour_base)
    module.reset()
    assert list(module.behaviour_base) == keys_before
    for view in module.iter_behaviours():
        assert module.posterior(view) == module.prior
        assert module.behaviour_counts(*view) == (0, 0)
    assert all((entry["action_counts"] == 0).all() for entry in module.behaviour_base.values())
    # the agent's action frequencies are a property of its (fixed) policy, kept across episodes
    assert list(module.action_counts) == [20, 20, 0]


# --- _precondition_satisfied ---

def satisfied_bins(module, value):
    return [entry["precondition"][0] for entry in module.behaviour_base.values()
            if module._precondition_satisfied(entry, np.array([value]))]


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
    for view in module.iter_behaviours():
        if view.precondition_key == frozenset([NO_BERRIES]):
            assert module.posterior(view) == module.prior
            assert module.behaviour_counts(*view)[0] == 0
        else:
            assert module.behaviour_counts(*view)[0] == 1


def test_update_counts():
    module = make_module()
    module.update(PRECONDITION_MET, EAT)
    module.update(PRECONDITION_MET, MOVE)
    eat_obligation = behaviour(module, "obligation", "eat")
    eat_prohibition = behaviour(module, "prohibition", "eat")
    assert module.behaviour_counts(*eat_obligation) == (2, 1)   # consistent only when eating
    assert module.behaviour_counts(*eat_prohibition) == (2, 1)  # consistent only when not eating


def test_action_counts_are_stored_once_per_precondition():
    """One count per action for each precondition, shared by its obligations and prohibitions."""
    module = make_module()
    for action in [EAT, EAT, MOVE, THROW, EAT]:
        module.update(PRECONDITION_MET, action)
    module.update(PRECONDITION_NOT_MET, MOVE)
    assert len(module.behaviour_base) == 2  # berries<1, berries>=1
    assert list(module.behaviour_base[frozenset([NO_BERRIES])]["action_counts"]) == [1, 3, 1]
    assert list(module.behaviour_base[frozenset([HAS_BERRIES])]["action_counts"]) == [1, 0, 0]
    assert module.behaviour_counts(*behaviour(module, "obligation", "throw")) == (5, 1)
    assert module.behaviour_counts(*behaviour(module, "prohibition", "throw")) == (5, 4)


def test_two_predicate_precondition_counts_once_per_step():
    spec = [
        {"name": "berries", "index": 0, "thresholds": [1]},
        {"name": "distance", "index": 1, "thresholds": [5]},
    ]
    module = make_module(spec, max_predicates=2)
    module.update(np.array([0, 6]), EAT)
    module.update(np.array([0, 2]), EAT)
    base = module.behaviour_base
    assert base[frozenset(["berries<1"])]["action_counts"].sum() == 2
    assert base[frozenset(["berries<1", "distance>=5"])]["action_counts"].sum() == 1
    assert base[frozenset(["berries<1", "distance<5"])]["action_counts"].sum() == 1


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
    assert module.posterior(behaviour(module, "obligation", "eat")) == pytest.approx(expected)


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
    assert module.posterior(behaviour(module, "obligation", "eat")) == pytest.approx(expected)


def test_likelihood_floor_holds_when_epsilon_is_zero():
    """
    In testing, epsilon can be 0 -- the 0.01 floor on the likelihoods must stop a single slip from
    driving the posterior to exactly 0, and give the same result as epsilon=0.01.
    """
    posteriors = []
    for epsilon in [0.0, 1e-6, 0.01]:
        module = make_module(epsilon=epsilon)
        module.update(PRECONDITION_MET, MOVE)  # a slip against obligation "eat"
        posteriors.append(module.posterior(behaviour(module, "obligation", "eat")))
    assert posteriors[0] > 0
    assert posteriors == pytest.approx([posteriors[0]] * 3)


@pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.1])
def test_posteriors_stay_finite_and_in_unit_interval(epsilon):
    module = make_module(epsilon=epsilon)
    run_policy(module, eat_ninety_percent, move_ninety_percent, 500)
    for entry in module.behaviour_base.values():
        assert np.isfinite(entry["posteriors"]).all()
        assert ((0.0 <= entry["posteriors"]) & (entry["posteriors"] <= 1.0)).all()


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


# --- episode_report ---

REPORT_COLUMNS = [
    "behaviour_id", "precondition", "action", "type", "is_cooperative", "n_precon_met",
    "n_action_matched", "rate", "exhibited", "posterior",
]


def report_row(rows, action):
    """The row for this action ("eat" / "NOT eat"), for the single-predicate spec."""
    [row] = [r for r in rows if r["action"] == action]
    return row


def test_episode_report_only_includes_behaviours_observed_this_episode():
    module = make_module()
    for action in [EAT, EAT, MOVE]:
        module.update(PRECONDITION_MET, action)
    rows = module.episode_report()
    assert len(rows) == len(ACTIONS) * 2  # berries<1 bin only: 3 actions x 2 types
    assert {r["precondition"] for r in rows} == {NO_BERRIES}
    assert all(list(r) == REPORT_COLUMNS for r in rows)


def test_episode_report_values():
    module = make_module(cooperative_actions=["throw"])
    for action in [EAT, EAT, MOVE]:
        module.update(PRECONDITION_MET, action)
    rows = module.episode_report()
    eat = report_row(rows, "eat")
    assert eat["behaviour_id"] == module.behaviour_id(behaviour(module, "obligation", "eat")) == "obligation|eat|berries<1"
    assert (eat["n_precon_met"], eat["n_action_matched"]) == (3, 2)
    assert eat["rate"] == pytest.approx(2 / 3)
    assert eat["posterior"] == pytest.approx(module.posterior(behaviour(module, "obligation", "eat")))
    not_eat = report_row(rows, "NOT eat")  # consistent only when not eating
    assert (not_eat["n_precon_met"], not_eat["n_action_matched"]) == (3, 1)
    assert [r["action"] for r in rows if r["is_cooperative"]] == ["throw"]  # not "NOT throw"


def test_episode_report_type_and_cooperative_flags():
    """Every row's type matches its "NOT " prefix; only obligations to throw are cooperative --
    NOT throw is never cooperative, even when the agent consistently avoids throwing."""
    module = make_module(cooperative_actions=["throw"])
    for _ in range(10):
        module.update(PRECONDITION_MET, EAT)  # consistent with NOT throw every step
    rows = module.episode_report()
    for row in rows:
        assert row["type"] == ("prohibition" if row["action"].startswith("NOT ") else "obligation")
    not_throw = report_row(rows, "NOT throw")
    assert not_throw["rate"] == 1.0
    assert not_throw["is_cooperative"] is False
    assert report_row(rows, "throw")["is_cooperative"] is True


def test_episode_report_preconditions_are_joined_with_and():
    spec = [
        {"name": "berries", "index": 0, "thresholds": [1]},
        {"name": "distance", "index": 1, "thresholds": [5]},
    ]
    module = make_module(spec, max_predicates=2)
    module.update(np.array([0, 6]), EAT)
    assert "berries<1 AND distance>=5" in {r["precondition"] for r in module.episode_report()}


@pytest.mark.parametrize("steps,action,expected", [
    (5, EAT, False),   # n_precon_met must exceed min_observations, not equal it
    (6, EAT, True),
    (6, MOVE, False),  # rate 0: never ate
])
def test_episode_report_exhibited(steps, action, expected):
    module = make_module(min_observations=5)
    for _ in range(steps):
        module.update(PRECONDITION_MET, action)
    assert report_row(module.episode_report(), "eat")["exhibited"] is expected


def test_episode_report_is_empty_after_reset():
    module = make_module()
    module.update(PRECONDITION_MET, EAT)
    module.reset()
    assert module.episode_report() == []


# --- get_learned_behaviours / get_all_behaviours_by_certainty ---

def test_learned_behaviours_require_min_observations():
    module = make_module(min_observations=5)
    eat = behaviour(module, "obligation", "eat")
    entry = module.behaviour_base[eat.precondition_key]
    entry["posteriors"][EAT, OBLIGATION] = 0.99
    entry["action_counts"][EAT] = 4
    assert module.get_learned_behaviours() == []
    entry["action_counts"][EAT] = 5
    assert module.get_learned_behaviours() == [module.behaviour_dict(eat)]


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
    assert len(ordered) == len(module.behaviour_base) * len(ACTIONS) * 2
    posteriors = [b["posterior"] for b in ordered]
    assert posteriors == sorted(posteriors, reverse=True)

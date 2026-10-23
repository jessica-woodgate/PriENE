"""
Temporary: checks the one-entry-per-precondition BayesianNormsModule reproduces the previous
one-entry-per-(precondition, action, type) implementation exactly. _OldBayesianNormsModule is a
verbatim copy of that previous class; delete this file once the refactor is accepted.
"""
from itertools import combinations

import numpy as np
import pytest

from src.agent.modules.bayesian_norms_module import BayesianNormsModule
from src.harvest_exception import UnmappedActionException
from tests.test_bayesian_norms_module import ACTION_INDEX, ACTIONS, EAT, HARVEST_SPECS, MOVE, THROW


class _OldBayesianNormsModule():
    def __init__(
        self,
        feature_specs,
        actions,
        max_predicates=2,
        prior=0.05,
        learned_threshold=0.95,
        min_observations=5,
        action_name_to_index=None,
        epsilon=0.1,
        cooperative_actions=None
    ):
        self.feature_specs = feature_specs
        self.actions = actions
        self.action_name_to_index = action_name_to_index or {}
        self.max_predicates = max_predicates
        self.prior = prior
        self.learned_threshold = learned_threshold
        self.min_observations = min_observations
        self.epsilon = epsilon
        self.cooperative_actions = set(cooperative_actions or [])
        self.action_counts = np.zeros(len(actions))
        self.behaviour_base = {}

        self._initialise()

    def reset(self):
        for behaviour in self.behaviour_base.values():
            behaviour["posterior"] = self.prior
        for counts in self.precondition_action_counts.values():
            counts[:] = 0

    def update(self, observation, action_taken):
        action_frequencies = (self.action_counts + 1) / (self.action_counts.sum() + len(self.actions))

        for key in self._matching_preconditions(observation):
            self.precondition_action_counts[key][action_taken] += 1

            for behaviour in self._rules_by_precondition[key]:
                action_idx = behaviour["action_idx"]

                if behaviour["type"] == "prohibition":
                    consistent = (action_taken != action_idx)
                    p_consistent_given_not_learned = 1.0 - action_frequencies[action_idx]
                else:
                    consistent = (action_taken == action_idx)
                    p_consistent_given_not_learned = action_frequencies[action_idx]
                p_consistent_given_learned = max(1.0 - self.epsilon, p_consistent_given_not_learned)

                if consistent:
                    p_evidence_given_learned     = p_consistent_given_learned
                    p_evidence_given_not_learned = p_consistent_given_not_learned
                else:
                    p_evidence_given_learned     = 1.0 - p_consistent_given_learned
                    p_evidence_given_not_learned = 1.0 - p_consistent_given_not_learned

                p_evidence_given_learned     = np.clip(p_evidence_given_learned,     0.01, 0.99)
                p_evidence_given_not_learned = np.clip(p_evidence_given_not_learned, 0.01, 0.99)

                prior = behaviour["posterior"]
                numerator = p_evidence_given_learned * prior
                denominator = numerator + p_evidence_given_not_learned * (1.0 - prior)
                behaviour["posterior"] = numerator / denominator

        self.action_counts[action_taken] += 1

    def behaviour_counts(self, behaviour):
        counts = self.precondition_action_counts[behaviour["precondition_key"]]
        n_precon_met = int(counts.sum())
        n_action_taken = int(counts[behaviour["action_idx"]])
        if behaviour["type"] == "prohibition":
            return n_precon_met, n_precon_met - n_action_taken
        return n_precon_met, n_action_taken

    def get_learned_behaviours(self):
        learned = [
            b for b in self.behaviour_base.values()
            if b["posterior"] >= self.learned_threshold
            and self.behaviour_counts(b)[0] >= self.min_observations
        ]
        return sorted(learned, key=lambda b: b["posterior"], reverse=True)

    def get_all_behaviours_by_certainty(self):
        observed = [
            b for b in self.behaviour_base.values()
        ]
        return sorted(observed, key=lambda b: b["posterior"], reverse=True)

    def episode_report(self):
        rows = []
        for b in self.behaviour_base.values():
            n_precon_met, n_action_matched = self.behaviour_counts(b)
            if n_precon_met == 0:
                continue
            rate = n_action_matched / n_precon_met
            rows.append({
                "behaviour_id":     b["id"],
                "precondition":     " AND ".join(b["precondition"]),
                "action":           b["action"] if b["type"] == "commitment" else f"NOT {b['action']}",
                "type":             b["type"],
                "is_cooperative":   b["type"] == "commitment" and b["action"] in self.cooperative_actions,
                "n_precon_met":     n_precon_met,
                "n_action_matched": n_action_matched,
                "rate":             rate,
                "exhibited":        rate > 0 and n_precon_met > self.min_observations,
                "posterior":        float(b["posterior"]),
            })
        return rows

    def _initialise(self):
        self._predicates = self._build_predicates()
        candidates = self._generate_candidates(self._predicates)
        self._validate_action_mapping(candidates)

        self.behaviour_base = {}
        self._rules_by_precondition = {}
        self.precondition_action_counts = {}

        for b in candidates:
            key = frozenset(b["precondition"])
            behaviour = {
                **b,
                "action_idx": self.action_name_to_index[b["action"]],
                "precondition_key": key,
                "posterior": self.prior,
            }
            self.behaviour_base[b["id"]] = behaviour
            self._rules_by_precondition.setdefault(key, []).append(behaviour)
            self.precondition_action_counts.setdefault(key, np.zeros(len(self.actions), dtype=int))

        return self

    def _build_predicates(self):
        predicates = []
        for spec in self.feature_specs:
            repeated = spec.get("repeated", False)
            indices = spec["index"] if repeated else [spec["index"]]
            for position, idx in enumerate(indices):
                for lower, upper in self._intervals(spec["thresholds"]):
                    predicates.append(self._make_predicate(
                        spec, idx, lower, upper,
                        position if repeated else None
                    ))
        return predicates

    def _intervals(self, thresholds):
        bounds = [-np.inf] + sorted(thresholds) + [np.inf]
        return list(zip(bounds[:-1], bounds[1:]))

    def _make_predicate(self, spec, index, lower, upper, position=None):
        feature = spec["name"] if position is None else f"{spec['name']}[{position}]"
        if lower == -np.inf:
            name = f"{feature}<{upper}"
        elif upper == np.inf:
            name = f"{feature}>={lower}"
        else:
            name = f"{lower}<={feature}<{upper}"
        return {
            "name":  name,
            "index": index,
            "lower": lower,
            "upper": upper,
        }

    def _generate_candidates(self, predicates):
        candidates = []
        behaviour_id = 0

        for length in range(1, self.max_predicates + 1):
            for pred_combo in combinations(predicates, length):
                if len({p["index"] for p in pred_combo}) < length:
                    continue
                precondition_names = [p["name"] for p in pred_combo]

                for action in self.actions:
                    candidates.append({
                        "id":           behaviour_id,
                        "type":         "prohibition",
                        "predicates":   pred_combo,
                        "precondition": precondition_names,
                        "action":       action,
                        "label": (
                            f"IF {' AND '.join(precondition_names)} "
                            f"THEN NOT {action}"
                        ),
                    })
                    behaviour_id += 1

                    candidates.append({
                        "id":           behaviour_id,
                        "type":         "commitment",
                        "predicates":   pred_combo,
                        "precondition": precondition_names,
                        "action":       action,
                        "label": (
                            f"IF {' AND '.join(precondition_names)} "
                            f"THEN {action}"
                        ),
                    })
                    behaviour_id += 1

        return candidates

    def _validate_action_mapping(self, candidates):
        referenced_actions = {c["action"] for c in candidates}
        missing = referenced_actions - set(self.action_name_to_index.keys())
        if missing:
            raise UnmappedActionException(missing)

    def _matching_preconditions(self, observation):
        active = [
            p for p in self._predicates
            if p["lower"] <= observation[p["index"]] < p["upper"]
        ]
        for length in range(1, self.max_predicates + 1):
            for combo in combinations(active, length):
                key = frozenset(p["name"] for p in combo)
                if key in self._rules_by_precondition:
                    yield key


# --- comparison ---

FEATURE_RANGES = [(0, 12), (0, 5), (0, 120), (0, 8), (0, 120), (0, 120), (0, 120)]
THRESHOLDS = [spec["thresholds"] for spec in HARVEST_SPECS[:4]] + [HARVEST_SPECS[4]["thresholds"] + [0]] * 3


def old_style_id(b):
    """The new string id of an old behaviour dict, to pair behaviours across the two implementations."""
    return f"{b['type']}|{b['action']}|{' AND '.join(sorted(b['precondition']))}"


def random_step(rng):
    """An observation (30% of values exactly on a bin edge) and a state-dependent action, so that
    some behaviours are learned and the comparison isn't trivially all-prior."""
    observation = np.array([
        rng.choice(t) if rng.random() < 0.3 else rng.uniform(lo, hi)
        for (lo, hi), t in zip(FEATURE_RANGES, THRESHOLDS)
    ], dtype=float)
    if observation[1] < 1 and rng.random() < 0.9:
        action = EAT
    elif observation[3] >= 3 and rng.random() < 0.8:
        action = MOVE
    elif observation[4] < 10 and rng.random() < 0.6:
        action = THROW
    else:
        action = int(rng.integers(len(ACTIONS)))
    return observation, action


def make_pair(epsilon):
    kwargs = dict(max_predicates=2, action_name_to_index=ACTION_INDEX, epsilon=epsilon, cooperative_actions=["throw"])
    return _OldBayesianNormsModule(HARVEST_SPECS, ACTIONS, **kwargs), BayesianNormsModule(HARVEST_SPECS, ACTIONS, **kwargs)


def assert_equivalent(old, new):
    old_behaviours = list(old.behaviour_base.values())
    views = list(new.iter_behaviours())
    # same behaviours, in the same order
    assert [new.behaviour_id(v) for v in views] == [old_style_id(b) for b in old_behaviours]
    for b, view in zip(old_behaviours, views):
        assert new.behaviour_counts(*view) == old.behaviour_counts(b)
    old_posteriors = np.array([b["posterior"] for b in old_behaviours], dtype=float)
    new_posteriors = np.array([new.posterior(v) for v in views], dtype=float)
    assert np.array_equal(new_posteriors, old_posteriors)
    old_rows = [{**row, "behaviour_id": old_style_id(old.behaviour_base[row["behaviour_id"]])}
                for row in old.episode_report()]
    assert new.episode_report() == old_rows
    assert [b["id"] for b in new.get_learned_behaviours()] == [old_style_id(b) for b in old.get_learned_behaviours()]
    assert [b["id"] for b in new.get_all_behaviours_by_certainty()] == \
        [old_style_id(b) for b in old.get_all_behaviours_by_certainty()]
    assert np.array_equal(new.action_counts, old.action_counts)


@pytest.mark.parametrize("epsilon", [0.0, 0.01, 0.1, 0.3])
def test_new_structure_matches_old_implementation(epsilon):
    old, new = make_pair(epsilon)
    rng = np.random.default_rng(0)
    for step in range(600):
        observation, action = random_step(rng)
        old.update(observation, action)
        new.update(observation, action)
        if step in (150, 299, 599):
            assert_equivalent(old, new)
        if step == 299:  # mid-run episode boundary
            old.reset()
            new.reset()
            assert_equivalent(old, new)
    assert new.get_learned_behaviours(), "sequence should produce some learned behaviours"

from itertools import combinations
import numpy as np

from src.harvest_exception import UnmappedActionException

class BayesianNormsModule():
    """
    Domain-agnostic norm base with Bayesian updating.

    Generates a candidate set of prohibition and obligation behaviours from
    bucket specifications provided at construction time, then performs
    Bayesian updating at each step using the agent's own (observation, action)
    pairs to estimate which behaviours the agent has learned.

    A behaviour is considered learned when the agent consistently acts in
    accordance with it whenever its precondition is met. Bayesian updating
    maintains a posterior P(behaviour learned | observations so far) for
    each candidate, starting from a low prior and rising as supporting
    evidence accumulates.

    "Not learned" is judged against the agent's own overall action frequencies,
    so a behaviour is only learned when the agent acts differently whenever its
    precondition is met than it does in general: an action the agent rarely
    takes anywhere is not "prohibited" in every state.

    Instance variables:
        feature_specs        -- ordered list of feature specification dicts
        actions              -- list of action name strings
        action_name_to_index -- dict mapping action name -> action index
        max_predicates       -- maximum number of predicates per precondition
        prior                -- initial P(behaviour learned), low by default
        learned_threshold    -- posterior above which a behaviour is returned
                                as learned (default 0.95)
        min_observations     -- minimum number of times a behaviour's
                                precondition must be met before it can be
                                considered learned (default 5)
        epsilon              -- exploration probability for agent's action selection
        action_counts        -- number of times the agent has taken each action, over
                                every update (not reset per episode: the policy, and so
                                its action frequencies, don't change between episodes)
        behaviour_base       -- dict mapping behaviour_id -> behaviour dict,
                                populated by _initialise()
    """

    def __init__(
        self,
        feature_specs,
        actions,
        max_predicates=2,
        prior=0.05,
        learned_threshold=0.95,
        min_observations=5,
        action_name_to_index=None,
        epsilon=0.1
    ):
        self.feature_specs = feature_specs
        self.actions = actions
        self.action_name_to_index = action_name_to_index or {}
        self.max_predicates = max_predicates
        self.prior = prior
        self.learned_threshold = learned_threshold
        self.min_observations = min_observations
        self.epsilon = epsilon
        self.action_counts = np.zeros(len(actions))
        self.behaviour_base = {}

        self._initialise()

    def reset(self):
        """
        Reset posteriors and counts for a new episode. action_counts is kept, so
        the agent's action frequencies keep being estimated across episodes.
        """
        for behaviour in self.behaviour_base.values():
            behaviour["posterior"] = self.prior
            behaviour["times_precondition_met"] = 0
            behaviour["times_action_matched"] = 0

    def update(self, observation, action_taken):
        """
        Bayesian update on one (observation, action) pair from the agent's own
        step. Only behaviours whose precondition is met are visited; all others
        receive no information from this step and are left unchanged.

        P(consistent | not learned) is the agent's overall frequency of acting
        consistently with the behaviour (from action_counts, with +1 smoothing
        so it starts uniform); P(consistent | learned) is 1 - epsilon, but never
        below P(consistent | not learned) -- otherwise a violation of a rare
        action's prohibition would count as evidence for it.

        observation  -- flat numpy array
        action_taken -- int, index into actions of the action the agent chose
        """
        # frequencies from steps before this one
        action_frequencies = (self.action_counts + 1) / (self.action_counts.sum() + len(self.actions))

        for behaviour in self._matching_behaviours(observation):
            action_idx = behaviour["action_idx"]
            behaviour["times_precondition_met"] += 1

            if behaviour["type"] == "prohibition":
                # learned <=> agent avoids the prohibited action
                consistent = (action_taken != action_idx)
                p_consistent_given_not_learned = 1.0 - action_frequencies[action_idx]
            else:
                # learned <=> agent takes the obligated action
                consistent = (action_taken == action_idx)
                p_consistent_given_not_learned = action_frequencies[action_idx]
            p_consistent_given_learned = max(1.0 - self.epsilon, p_consistent_given_not_learned)

            if consistent:
                behaviour["times_action_matched"] += 1
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

    def get_learned_behaviours(self):
        """
        Return behaviours whose posterior exceeds the learned threshold,
        sorted by posterior descending — most certain first.

        These are the behaviours the agent has most consistently exhibited
        whenever their precondition was met.
        """
        learned = [
            b for b in self.behaviour_base.values()
            if b["posterior"] >= self.learned_threshold
            and b["times_precondition_met"] >= self.min_observations
        ]
        return sorted(learned, key=lambda b: b["posterior"], reverse=True)

    def get_all_behaviours_by_certainty(self):
        """
        Return all behaviours that have been observed, sorted by posterior descending.
        Useful for inspecting the full learned/not-learned spectrum.
        """
        observed = [
            b for b in self.behaviour_base.values()
        ]
        return sorted(observed, key=lambda b: b["posterior"], reverse=True)

    def _initialise(self):
        """
        Generate candidate behaviours, assign priors, and build a lookup table
        from each precondition to the behaviours that share it, so update()
        only visits behaviours whose precondition can match the observation.
        """
        self._predicates = self._build_predicates()
        candidates = self._generate_candidates(self._predicates)
        self._validate_action_mapping(candidates)

        self.behaviour_base = {}
        # frozenset of predicate names -> list of behaviour dicts
        self._rules_by_precondition = {}

        for b in candidates:
            behaviour = {
                **b,
                # resolved once here rather than looked up on every step
                "action_idx": self.action_name_to_index[b["action"]],
                # posterior: P(this behaviour has been learned)
                "posterior": self.prior,
                # counts used to compute the likelihood at each step
                "times_precondition_met": 0,
                "times_action_matched":   0,
            }
            self.behaviour_base[b["id"]] = behaviour
            key = frozenset(b["precondition"])
            self._rules_by_precondition.setdefault(key, []).append(behaviour)

        return self

    def _build_predicates(self):
        """
        Turn each feature's thresholds into mutually exclusive intervals.
        Thresholds [t0, t1, t2] produce:
            x < t0,  t0 <= x < t1,  t1 <= x < t2,  x >= t2
        Repeated features produce one set of intervals per index.
        """
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
        """Consecutive [lower, upper) pairs, open-ended at both extremes."""
        bounds = [-np.inf] + sorted(thresholds) + [np.inf]
        return list(zip(bounds[:-1], bounds[1:]))

    def _make_predicate(self, spec, index, lower, upper, position=None):
        # repeated features need their position in the name, else each other
        # agent's well-being predicate would share one label
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
        """
        Generate both a prohibition and an obligation candidate for every
        combination of predicates and every action. Both types are always
        generated — the Bayesian update determines which are learned,
        not the generation step. Combinations containing two bins of the same
        feature are skipped: bins don't overlap, so they could never be met.
        """
        candidates = []
        behaviour_id = 0

        for length in range(1, self.max_predicates + 1):
            for pred_combo in combinations(predicates, length):
                if len({p["index"] for p in pred_combo}) < length:
                    continue
                precondition_names = [p["name"] for p in pred_combo]

                for action in self.actions:
                    # Prohibition: IF precondition THEN NOT action
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

                    # Obligation: IF precondition THEN action
                    candidates.append({
                        "id":           behaviour_id,
                        "type":         "obligation",
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
        """
        Raise immediately if any generated candidate's action isn't covered by
        action_name_to_index. Without this check, update() would silently skip every behaviour
        using an unmapped action forever (action_idx would always resolve to None), with no error
        or warning -- times_precondition_met would stay at 0 and learned_behaviours()/
        all_behaviours_by_certainty() would silently never return anything for it.
        """
        referenced_actions = {c["action"] for c in candidates}
        missing = referenced_actions - set(self.action_name_to_index.keys())
        if missing:
            raise UnmappedActionException(missing)

    def _matching_behaviours(self, observation):
        """
        Yield every behaviour whose precondition is satisfied by this observation.

        Finds the predicates the observation satisfies (one bin per feature),
        then looks up each combination of them up to max_predicates.
        Same-feature combinations can't occur here because each feature has
        only one active bin.
        """
        active = [
            p for p in self._predicates
            if p["lower"] <= observation[p["index"]] < p["upper"]
        ]
        for length in range(1, self.max_predicates + 1):
            for combo in combinations(active, length):
                key = frozenset(p["name"] for p in combo)
                yield from self._rules_by_precondition.get(key, [])

    def _precondition_satisfied(self, behaviour, observation):
        for pred in behaviour["predicates"]:
            value = observation[pred["index"]]
            if not (pred["lower"] <= value < pred["upper"]):
                return False
        return True
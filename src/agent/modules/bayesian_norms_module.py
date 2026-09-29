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

    Instance variables:
        feature_specs        -- ordered list of feature specification dicts
        actions              -- list of action name strings
        action_name_to_index -- dict mapping action name -> action index
        max_predicates       -- maximum number of predicates per precondition
        prior                -- initial P(behaviour learned), low by default
        learned_threshold    -- posterior above which a behaviour is returned
                                as learned (default 0.95)
        behaviour_base            -- dict mapping behaviour_id -> behaviour dict,
                                populated by initialise()
    """

    def __init__(
        self,
        feature_specs,
        actions,
        max_predicates=2,
        prior=0.05,
        learned_threshold=0.95,
        action_name_to_index=None
    ):
        self.feature_specs = feature_specs
        self.actions = actions
        self.action_name_to_index = action_name_to_index or {}
        self.max_predicates = max_predicates
        self.prior = prior
        self.learned_threshold = learned_threshold
        self.behaviour_base = {}

    def initialise(self):
        """
        Generate candidate behaviours and assign priors. Call once before testing begins.
        """
        predicates = self._build_predicates()
        candidates = self._generate_candidates(predicates)
        self._validate_action_mapping(candidates)
        self.behaviour_base = {
            b["id"]: {
                **b,
                # posterior: P(this behaviour has been learned)
                "posterior": self.prior,
                # counts used to compute the likelihood at each step
                "times_precondition_met": 0,
                "times_action_matched":   0,
            }
            for b in candidates
        }
        return self

    def update(self, observation, action_taken):
        """
        Bayesian update on one (observation, action) pair from the
        agent's own step. For each candidate behaviour:

          - if its precondition is not met, skip (no information)
          - if its precondition is met and the agent's action is
            consistent with the behaviour, this is supporting evidence
          - if its precondition is met and the agent's action contradicts
            the behaviour, this is counter-evidence

        The update follows Bayes' rule:

            P(learned | evidence) ∝ P(evidence | learned) * P(learned)

        P(evidence | learned) is estimated empirically from the running
        match rate across all steps where the precondition was met.

        observation  -- flat numpy array
        action_taken -- int, index of the action the agent chose
        """
        for behaviour in self.behaviour_base.values():
            if not self._precondition_satisfied(behaviour, observation):
                continue

            action_idx = self.action_name_to_index.get(behaviour["action"])
            if action_idx is None:
                continue

            behaviour["times_precondition_met"] += 1

            # Determine whether this step is supporting evidence
            if behaviour["type"] == "prohibition":
                # Behaviour learned <=> agent avoids the prohibited action
                consistent = (action_taken != action_idx)
            else:
                # Behaviour learned <=> agent takes the obligated action
                consistent = (action_taken == action_idx)

            if consistent:
                behaviour["times_action_matched"] += 1

            # ── Bayesian update ────────────────────────────────────────
            # Empirical match rate as the likelihood estimate.
            # P(consistent | learned)    = match_rate (agent acts this way
            #                              because it learned the behaviour)
            # P(consistent | not learned) = base_rate (agent acts this way
            #                              by chance or for other reasons)
            n   = behaviour["times_precondition_met"]
            k   = behaviour["times_action_matched"]

            # Running match rate: how often the agent acted consistently
            match_rate = k / n

            # Base rate: for a prohibition, the agent avoids the action
            # some fraction of the time regardless of norms (1 - 1/n_actions
            # is a simple uninformed estimate). For an obligation, the
            # probability of coincidentally taking one specific action.
            n_actions = len(self.actions)
            if behaviour["type"] == "prohibition":
                base_rate = 1.0 - (1.0 / n_actions)
            else:
                base_rate = 1.0 / n_actions

            # Likelihood of this step's evidence under each hypothesis
            if consistent:
                p_evidence_given_learned     = match_rate
                p_evidence_given_not_learned = base_rate
            else:
                p_evidence_given_learned     = 1.0 - match_rate
                p_evidence_given_not_learned = 1.0 - base_rate

            # Avoid degenerate likelihoods on first observations
            p_evidence_given_learned     = np.clip(
                p_evidence_given_learned,     0.01, 0.99
            )
            p_evidence_given_not_learned = np.clip(
                p_evidence_given_not_learned, 0.01, 0.99
            )

            # Bayes' rule
            prior = behaviour["posterior"]
            numerator = p_evidence_given_learned * prior
            denominator = (
                numerator
                + p_evidence_given_not_learned * (1.0 - prior)
            )
            behaviour["posterior"] = numerator / denominator

    def learned_behaviours(self):
        """
        Return behaviours whose posterior exceeds the learned threshold,
        sorted by posterior descending — most certain first.

        These are the behaviours the agent has most consistently exhibited
        whenever their precondition was met.
        """
        learned = [
            b for b in self.behaviour_base.values()
            if b["posterior"] >= self.learned_threshold
            and b["times_precondition_met"] > 0
        ]
        return sorted(learned, key=lambda b: b["posterior"], reverse=True)

    def all_behaviours_by_certainty(self, min_observations=5):
        """
        Return all behaviours that have been observed at least
        min_observations times, sorted by posterior descending.
        Useful for inspecting the full learned/not-learned spectrum.
        """
        observed = [
            b for b in self.behaviour_base.values()
            if b["times_precondition_met"] >= min_observations
        ]
        return sorted(observed, key=lambda b: b["posterior"], reverse=True)

    def _build_predicates(self):
        predicates = []
        for spec in self.feature_specs:
            if spec.get("repeated"):
                for idx in spec["index"]:
                    for threshold in spec["thresholds"]:
                        predicates.append(
                            self._make_predicate(spec, idx, threshold)
                        )
            else:
                for threshold in spec["thresholds"]:
                    predicates.append(
                        self._make_predicate(spec, spec["index"], threshold)
                    )
        return predicates

    def _make_predicate(self, spec, index, threshold):
        direction = spec["direction"]
        name = f"{spec['name']}{direction}{threshold}"
        return {
            "name":      name,
            "index":     index,
            "threshold": threshold,
            "direction": direction,
        }

    def _generate_candidates(self, predicates):
        """
        Generate both a prohibition and an obligation candidate for every
        combination of predicates and every action. Both types are always
        generated — the Bayesian update determines which are learned,
        not the generation step.
        """
        candidates = []
        behaviour_id = 0

        for length in range(1, self.max_predicates + 1):
            for pred_combo in combinations(predicates, length):
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

    def _precondition_satisfied(self, behaviour, observation):
        for pred in behaviour["predicates"]:
            value = observation[pred["index"]]
            if pred["direction"] == "<":
                if not (value < pred["threshold"]):
                    return False
            else:
                if not (value > pred["threshold"]):
                    return False
        return True
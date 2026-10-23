from collections import namedtuple
from itertools import combinations
import numpy as np

from src.harvest_exception import UnmappedActionException

# columns of each behaviour_base entry's posteriors array
PROHIBITION = 0
COMMITMENT = 1
TYPE_COLUMNS = {"prohibition": PROHIBITION, "commitment": COMMITMENT}
# order behaviours of one (precondition, action) are listed in
TYPES = ("prohibition", "commitment")

# one behaviour, "IF precondition THEN (NOT) action": a view onto a behaviour_base entry
BehaviourView = namedtuple("BehaviourView", ["precondition_key", "action_idx", "type"])

class BayesianNormsModule():
    """
    Domain-agnostic norm base with Bayesian updating.

    Generates a candidate set of prohibition and commitment behaviours from
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
        cooperative_actions  -- set of action names whose commitments count as
                                cooperative in episode_report() (e.g. {"throw"})
        action_counts        -- number of times the agent has taken each action, over
                                every update (not reset per episode: the policy, and so
                                its action frequencies, don't change between episodes)
        behaviour_base       -- dict mapping precondition (frozenset of predicate names)
                                -> entry, populated by _initialise(). One entry per
                                precondition holds every behaviour with it:
                                    predicates    -- tuple of predicate dicts
                                    precondition  -- list of predicate names
                                    action_counts -- int array (len(actions)): how often
                                                     the agent took each action while the
                                                     precondition held this episode
                                    posteriors    -- float array (len(actions), 2):
                                                     P(learned) per action, column
                                                     PROHIBITION / COMMITMENT
                                Individual behaviours are BehaviourViews (see
                                iter_behaviours()), identified by behaviour_id()
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
        """
        Reset posteriors and counts for a new episode. action_counts is kept, so
        the agent's action frequencies keep being estimated across episodes.
        """
        for entry in self.behaviour_base.values():
            entry["action_counts"][:] = 0
            entry["posteriors"][:] = self.prior

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
        n_actions = len(self.actions)
        # frequencies from steps before this one
        action_frequencies = (self.action_counts + 1) / (self.action_counts.sum() + n_actions)

        # likelihoods for every (action, type) cell -- they depend only on the action taken and the
        # agent's overall action frequencies, so they are shared by every matching precondition
        p_consistent_given_not_learned = np.empty((n_actions, 2))
        # prohibition learned <=> agent avoids the action; commitment learned <=> agent takes it
        p_consistent_given_not_learned[:, PROHIBITION] = 1.0 - action_frequencies
        p_consistent_given_not_learned[:, COMMITMENT]  = action_frequencies
        p_consistent_given_learned = np.maximum(1.0 - self.epsilon, p_consistent_given_not_learned)

        took_action = np.arange(n_actions) == action_taken
        consistent = np.empty((n_actions, 2), dtype=bool)
        consistent[:, PROHIBITION] = ~took_action
        consistent[:, COMMITMENT]  = took_action

        p_evidence_given_learned     = np.where(consistent, p_consistent_given_learned,
                                                1.0 - p_consistent_given_learned)
        p_evidence_given_not_learned = np.where(consistent, p_consistent_given_not_learned,
                                                1.0 - p_consistent_given_not_learned)
        p_evidence_given_learned     = np.clip(p_evidence_given_learned,     0.01, 0.99)
        p_evidence_given_not_learned = np.clip(p_evidence_given_not_learned, 0.01, 0.99)

        for key in self._matching_preconditions(observation):
            entry = self.behaviour_base[key]
            entry["action_counts"][action_taken] += 1

            prior = entry["posteriors"]
            numerator = p_evidence_given_learned * prior
            denominator = numerator + p_evidence_given_not_learned * (1.0 - prior)
            entry["posteriors"] = numerator / denominator

        self.action_counts[action_taken] += 1

    def behaviour_counts(self, precondition_key, action_idx, behaviour_type):
        """
        (n_precon_met, n_action_matched) for a behaviour this episode, derived from
        its precondition's action counts: n_action_matched is how often the agent
        took the committed action, or avoided the prohibited one.
        """
        counts = self.behaviour_base[precondition_key]["action_counts"]
        n_precon_met = int(counts.sum())
        n_action_taken = int(counts[action_idx])
        if behaviour_type == "prohibition":
            return n_precon_met, n_precon_met - n_action_taken
        return n_precon_met, n_action_taken

    def iter_behaviours(self):
        """
        Yield a BehaviourView for every behaviour: each precondition, each action, then
        prohibition before commitment (the order behaviours were originally generated in).
        """
        for key in self.behaviour_base:
            yield from self._entry_behaviours(key)

    def behaviour_id(self, view):
        """Stable id, identical across agents and runs: "type|action|sorted precondition"."""
        precondition = sorted(self.behaviour_base[view.precondition_key]["precondition"])
        return f"{view.type}|{self._action_names[view.action_idx]}|{' AND '.join(precondition)}"

    def posterior(self, view):
        return self.behaviour_base[view.precondition_key]["posteriors"][view.action_idx, TYPE_COLUMNS[view.type]]

    def label(self, view):
        precondition = self.behaviour_base[view.precondition_key]["precondition"]
        negation = "NOT " if view.type == "prohibition" else ""
        return f"IF {' AND '.join(precondition)} THEN {negation}{self._action_names[view.action_idx]}"

    def behaviour_dict(self, view):
        """A behaviour as a dict, in the shape callers (e.g. HarvestModel's norm writers) read."""
        entry = self.behaviour_base[view.precondition_key]
        return {
            "id":               self.behaviour_id(view),
            "type":             view.type,
            "action":           self._action_names[view.action_idx],
            "predicates":       entry["predicates"],
            "precondition":     entry["precondition"],
            "label":            self.label(view),
            "posterior":        float(self.posterior(view)),
            "precondition_key": view.precondition_key,
            "action_idx":       view.action_idx,
        }

    def get_learned_behaviours(self):
        """
        Return behaviours whose posterior exceeds the learned threshold,
        sorted by posterior descending — most certain first.

        These are the behaviours the agent has most consistently exhibited
        whenever their precondition was met.
        """
        learned = []
        for key, entry in self.behaviour_base.items():
            # n_precon_met is shared by every behaviour of a precondition
            if entry["action_counts"].sum() < self.min_observations:
                continue
            learned.extend(
                self.behaviour_dict(view) for view in self._entry_behaviours(key)
                if self.posterior(view) >= self.learned_threshold
            )
        return sorted(learned, key=lambda b: b["posterior"], reverse=True)

    def get_all_behaviours_by_certainty(self):
        """
        Return all behaviours that have been observed, sorted by posterior descending.
        Useful for inspecting the full learned/not-learned spectrum.
        """
        observed = [self.behaviour_dict(view) for view in self.iter_behaviours()]
        return sorted(observed, key=lambda b: b["posterior"], reverse=True)

    def episode_report(self):
        """
        One row per behaviour whose precondition was met at least once this
        episode. Call at the end of each episode, before reset() clears the counts.

        Columns:
            behaviour_id     -- behaviour_id(), "type|action|sorted precondition"
            precondition     -- predicate names joined with " AND "
            action           -- the action, prefixed "NOT " for prohibitions
            type             -- "commitment" or "prohibition"
            is_cooperative   -- commitment to take one of cooperative_actions
                                (prohibitions, e.g. NOT throw, are never cooperative)
            n_precon_met     -- steps this episode where the precondition held
            n_action_matched -- of those, steps where the agent acted consistently
                                (took the committed / avoided the prohibited action)
            rate             -- n_action_matched / n_precon_met
            exhibited        -- rate > 0 and n_precon_met > min_observations, i.e.
                                seen often enough for the rate to mean something
            posterior        -- P(behaviour learned) at the end of the episode
        """
        rows = []
        for key, entry in self.behaviour_base.items():
            if entry["action_counts"].sum() == 0:
                continue
            for view in self._entry_behaviours(key):
                n_precon_met, n_action_matched = self.behaviour_counts(*view)
                action = self._action_names[view.action_idx]
                rate = n_action_matched / n_precon_met
                rows.append({
                    "behaviour_id":     self.behaviour_id(view),
                    "precondition":     " AND ".join(entry["precondition"]),
                    "action":           action if view.type == "commitment" else f"NOT {action}",
                    "type":             view.type,
                    "is_cooperative":   view.type == "commitment" and action in self.cooperative_actions,
                    "n_precon_met":     n_precon_met,
                    "n_action_matched": n_action_matched,
                    "rate":             rate,
                    "exhibited":        rate > 0 and n_precon_met > self.min_observations,
                    "posterior":        float(self.posterior(view)),
                })
        return rows

    def _initialise(self):
        """
        Build one behaviour_base entry per candidate precondition, with zeroed action
        counts and every posterior at the prior. Keyed by precondition, behaviour_base is
        also the lookup table update() uses to visit only matching preconditions.
        """
        self._validate_action_mapping()
        # action index -> name; rows of action_counts / posteriors are action indices
        self._action_names = {self.action_name_to_index[action]: action for action in self.actions}
        self._predicates = self._build_predicates()

        n_actions = len(self.actions)
        self.behaviour_base = {}
        for pred_combo in self._generate_preconditions(self._predicates):
            precondition_names = [p["name"] for p in pred_combo]
            self.behaviour_base[frozenset(precondition_names)] = {
                "predicates":    pred_combo,
                "precondition":  precondition_names,
                "action_counts": np.zeros(n_actions, dtype=int),
                "posteriors":    np.full((n_actions, 2), self.prior, dtype=float),
            }

        return self

    def _entry_behaviours(self, precondition_key):
        """BehaviourViews of one precondition: each action, prohibition before commitment."""
        for action in self.actions:
            action_idx = self.action_name_to_index[action]
            for behaviour_type in TYPES:
                yield BehaviourView(precondition_key, action_idx, behaviour_type)

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

    def _generate_preconditions(self, predicates):
        """
        Yield every combination of up to max_predicates predicates as a candidate
        precondition; each gets a prohibition and a commitment for every action (the
        Bayesian update determines which are learned, not the generation step).
        Combinations containing two bins of the same feature are skipped: bins don't
        overlap, so they could never be met.
        """
        for length in range(1, self.max_predicates + 1):
            for pred_combo in combinations(predicates, length):
                if len({p["index"] for p in pred_combo}) < length:
                    continue
                yield pred_combo

    def _validate_action_mapping(self):
        """
        Raise immediately if any action isn't covered by action_name_to_index, rather than
        failing later with a KeyError when its behaviours are first indexed.
        """
        missing = set(self.actions) - set(self.action_name_to_index.keys())
        if missing:
            raise UnmappedActionException(missing)

    def _matching_preconditions(self, observation):
        """
        Yield the key of every precondition satisfied by this observation.

        Finds the predicates the observation satisfies (one bin per feature),
        then each combination of them up to max_predicates. Same-feature
        combinations can't occur here because each feature has only one active bin.
        """
        active = [
            p for p in self._predicates
            if p["lower"] <= observation[p["index"]] < p["upper"]
        ]
        for length in range(1, self.max_predicates + 1):
            for combo in combinations(active, length):
                key = frozenset(p["name"] for p in combo)
                if key in self.behaviour_base:
                    yield key

    def _precondition_satisfied(self, entry, observation):
        """Whether observation satisfies a behaviour_base entry's precondition (full check, no lookup)."""
        for pred in entry["predicates"]:
            value = observation[pred["index"]]
            if not (pred["lower"] <= value < pred["upper"]):
                return False
        return True
from mesa import Agent
from .modules.norms_module import NormsModule

# norm bases an agent can track its behaviour with (see _build_norms_module); the first is the default
NORMS_MODULE_TYPES = ["rule_based", "bayesian"]

class LearningAgent(Agent):
    """
    What every learning agent shares, whatever its environment: a DQNDecisionModule (decision_module), an EthicsModule (ethics_module,
    absent for "baseline") and, if track_norms, a norm base (norms_module) of norms_module_type. Subclasses build those modules and give
    the environment's side: what the agent observes, what its actions do, and how its state is bucketed for norms, via
    _antecedent_features, _consequent_rules, _build_bayesian_norms_module, _norm_action_label and _norm_action_index.
    """
    def get_epsilon(self):
        return self.decision_module.get_epsilon()

    def get_mean_loss(self):
        return self.decision_module.get_mean_loss()

    def finish_episode(self):
        if self.training:
            self.decision_module.save_models(self.model.episode)

    def _ethics_sanction(self):
        society_well_being = self.model.get_society_well_being(self, False, True)
        sanction = self.ethics_module.get_sanction(society_well_being)
        return sanction

    def _build_norms_module(self):
        """
        Construct the norm base selected by norms_module_type. Domain knowledge (features, thresholds,
        action groupings) lives in the subclass; both norms modules stay domain-agnostic.
        """
        if self.norms_module_type == "bayesian":
            return self._build_bayesian_norms_module(self.model.get_num_agents())
        #the model rejects any other type before agents are built
        return NormsModule(self.unique_id, self._antecedent_features(), self._consequent_rules())

    def _update_norms(self, antecedent, observation, action, reward):
        """
        Record this step's behaviour in the norm base: rule-based norms are keyed on the antecedent (the state before acting) and the
        action's label, with this step's reward; Bayesian norms on the observation before acting and the action's norm action index
        """
        if self.norms_module_type == "rule_based":
            self.norms_module.update_behaviour_base(antecedent, self._norm_action_label(action), reward, self.model.get_day(), self.model.episode)
        else:
            self.norms_module.update(observation, self._norm_action_index(action))

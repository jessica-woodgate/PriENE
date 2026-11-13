from ..agent.learning_agent import LearningAgent
from ..agent.modules.dqn_decision_module import DQNDecisionModule
from ..agent.modules.ethics_module import EthicsModule
from ..agent.modules.bayesian_norms_module import BayesianNormsModule
from .labels import NORM_LABELS, NORM_LABEL_INDEX

#the ethics sanction, + or - this much (docs/fruit_market_design.md §6.1)
FRUIT_MARKET_SANCTION = 3.5

class FruitMarketAgent(LearningAgent):
    """
    A learning agent playing one Fruit Market player. The substrate steps every player at once, so the model asks each agent to choose
    (from the observation before the step), steps the substrate, then has each agent settle (learn from the step's reward and next
    observation, and record its behaviour for norms). Its reward is the substrate's, plus the ethics sanction unless it is "baseline".
        role -- "apple_farmer" or "banana_farmer"
        current_action, current_label -- this step's action and its norm label (see labels.norm_label)
        current_env_reward, current_sanction, current_reward -- this step's substrate reward, ethics sanction (None for baseline) and their sum
    """
    def __init__(self, unique_id, model, agent_type, role, training, checkpoint_path, epsilon, track_norms, shared_replay_buffer=None,
                 norms_module_type="rule_based"):
        super().__init__(unique_id, model)
        self.agent_type = agent_type
        self.role = role
        self.training = training
        self.done = False
        self.off_grid = False
        self.track_norms = track_norms
        self.norms_module_type = norms_module_type
        self.n_features = model.observation_size(self)
        self.actions = model.agent_actions(self)
        self.decision_module = DQNDecisionModule(agent_type, unique_id, training, self.actions, self.n_features, checkpoint_path, epsilon,
                                                 shared_replay_buffer, model.episode, action_mask=model.action_mask(self))
        if self.track_norms:
            self.norms_module = self._build_norms_module()
        if agent_type != "baseline":
            self.ethics_module = EthicsModule(FRUIT_MARKET_SANCTION, agent_type)
        self.reset()

    def reset(self):
        self.done = False
        self.total_episode_reward = 0
        self.current_reward = 0
        self.current_env_reward = 0
        self.current_sanction = None
        self.current_action = None
        self.current_label = None
        self.decision_module.start_episode()
        if self.track_norms:
            self.norms_module.reset()

    def choose(self, observation, antecedent_values):
        """
        Choose this step's action from the observation before the step; antecedent_values are the norm features of the state before it
        """
        self.current_observation = observation
        self.current_antecedent = None
        if self.track_norms and self.norms_module_type == "rule_based":
            self.current_antecedent = self.norms_module.get_antecedent(antecedent_values)
        self.current_action = self.decision_module.choose_action(observation)
        return self.current_action

    def settle(self, env_reward, sanction, next_observation, label):
        """
        Learn from the step: the substrate's reward plus the sanction (None for baseline), and the observation after the step; then record
        the step's behaviour, label, for norms
        """
        self.current_env_reward = env_reward
        self.current_sanction = sanction
        self.current_label = label
        self.current_reward = env_reward + (0 if sanction is None else sanction)
        if self.training:
            self.decision_module.learn(self.current_observation, self.current_action, self.current_reward, next_observation, False, self.model.episode)
        self.total_episode_reward += self.current_reward
        if self.track_norms:
            self._update_norms(self.current_antecedent, self.current_observation, self.current_action, self.current_reward)

    def ethics_sanction(self, before, after):
        """
        The sanction for this step from society's well-being before and after it (see sanction.sanction_vectors)
        """
        self.ethics_module.update_ethics_state(before)
        return float(sum(self.ethics_module.get_sanction(after)))

    # ── Norms ───────────────────────────────────────────────────

    def _norm_action_label(self, action):
        return self.current_label

    def _norm_action_index(self, action):
        return NORM_LABEL_INDEX[self.current_label]

    def _antecedent_features(self):
        return self.model.norm_antecedent_specs()

    def _consequent_rules(self):
        #labels are already behaviours (see labels.norm_label), so they are used as they are
        return []

    def _build_bayesian_norms_module(self, n_agents):
        return BayesianNormsModule(feature_specs=self.model.bayesian_feature_specs(), actions=list(NORM_LABELS), max_predicates=2,
                                   action_name_to_index=dict(NORM_LABEL_INDEX))

from .learning_agent import LearningAgent
from .learning_agent import NORMS_MODULE_TYPES  # noqa: F401 (imported from here by HarvestModel)
from .modules.dqn_decision_module import DQNDecisionModule
from .modules.moving_module import MovingModule
from .modules.bayesian_norms_module import BayesianNormsModule
from .modules.ethics_module import EthicsModule
from src.training_config import DEFAULT as DEFAULT_TRAINING_CONFIG
from src.harvest_exception import NumFeaturesException
from src.harvest_exception import AgentTypeException
import numpy as np

#health changes in steps of 0.1 (eating or being thrown a berry, and each day's decay), which floating point can't hold exactly: after
#a few steps health that should be 0.6 is stored as 0.5999999999999996, so an agent at the throwing threshold couldn't throw, and an
#agent at 0 health could be stored just above 0 and live another day. Health and days left to live are rounded to this many decimals
#whenever they change, so every comparison (throwing, dying, the norms' thresholds) sees the value it should
DECIMALS = 10

class HarvestAgent(LearningAgent):
    """
    HarvestAgent observes, chooses an action via its DQNDecisionModule, performs
    the action (move/eat/throw), updates its own attributes (health, berries, days left to
    live), receives an ethics sanction from its EthicsModule (unless agent_type is "baseline"),
    and optionally tracks its own behaviour/norms via a NormsModule (see perform_transition,
    which is the Interaction Module described as Algorithm 3 in the paper).
    Instance variables:
        agent_type -- normative ethics principle (or aggregation, or "baseline") this agent follows
        training -- boolean training or testing (also gates whether DQNDecisionModule learns)
        n_features -- length of the observation vector (see _calculate_n_features)
        actions -- this agent's action vocabulary: ["move", "eat"] plus one "throw_X" per other agent
        health / start_health / health_decay / berry_health_payoff -- agent's vitality; health
            decays every step and is restored by eating/receiving a thrown berry
        low/high_health_threshold, low/high_berries_threshold, low/high_days_left_threshold --
            bucket boundaries registered with NormsModule (see _antecedent_features) for turning
            this agent's state into a natural language precondition string
        berries / berries_consumed / berries_thrown -- this agent's current berry count and
            lifetime totals of berries eaten/given away
        days_left_to_live / total_days_left_to_live -- this agent's own well-being measure
            (Equation 4), and its running total across the episode
        min/max_width, min/max_height -- the grid bounds this agent may move within
        allocation_id -- which allocation of berries this agent may forage from (colours/allotment/
            capabilities scenarios only; defaults to this agent's own unique_id when unrestricted)
        decision_module -- DQNDecisionModule choosing actions and (if training) learning from them
        moving_module -- MovingModule handling pathfinding/movement towards berries
        ethics_module -- EthicsModule computing a self-directed sanction (absent for "baseline")
        norms_module_type -- which norm base tracks this agent's behaviour: "rule_based"
            (NormsModule, the default) or "bayesian" (BayesianNormsModule); only used if track_norms
        norms_module -- norm base tracking this agent's behaviour (present only if track_norms)
        rewards -- this agent's reward table (see _baseline_rewards/_ethics_rewards)
    """
    def __init__(self,unique_id,model,agent_type,allotment,training,checkpoint_path,epsilon,track_norms,shared_replay_buffer=None,allocation_id=None,norms_module_type="rule_based"):
        super().__init__(unique_id,model)
        self.done = False
        self.current_reward = 0
        #the ethics sanction this step's action received (already part of current_reward); None for "baseline", which has none
        self.current_sanction = None
        #id of the agent this step's action threw a berry to; None if it didn't throw one (or the throw failed)
        self.current_thrown_to = None
        #the group that agent was in when the berry was thrown, compared with this agent's (see HarvestModel.group_thrown_to); None if no
        #berry was thrown or the scenario has no groups (basic, commons)
        self.current_group_thrown_to = None
        #that agent compared with this one by days left to live, before the berry moved ("better_off_other", "worse_off_other" or
        #"equal_other"); "dead_other" for a throw tried to an agent that has died (which always fails); None otherwise
        self.current_agent_thrown_to = None
        #this step's action as rule-based norms label it, worked out before acting (see _throw_norm_label)
        self.current_norm_label = None
        self.total_episode_reward = 0
        self.training = training
        self.agent_type = agent_type
        self.n_features = self._calculate_n_features()
        self.start_health = 0.8
        self.health = self.start_health
        self.health_decay = 0.1
        self.low_health_threshold = 0.6
        self.high_health_threshold = 2.0
        self.low_berries_threshold = 1
        self.high_berries_threshold = 3
        self.low_days_left_threshold = 10
        self.high_days_left_threshold = 30
        self.berry_health_payoff = 0.6
        self.berries = 0
        self.berries_consumed = 0
        self.berries_thrown = 0
        self.days_left_to_live = self.get_days_left_to_live()
        self.total_days_left_to_live = self.days_left_to_live
        self.max_days = self.model.get_max_days()
        self.actions = self._generate_actions(self.unique_id, model.get_num_agents())
        self.off_grid = False
        self.min_width = allotment[0]
        self.max_width = allotment[1]
        self.min_height = allotment[2]
        self.max_height = allotment[3]
        restrict_to_allocation = allocation_id is not None
        self.allocation_id = self.unique_id if allocation_id is None else allocation_id
        #a continued training run starts from a later episode (see HarvestModel)
        #the training config is only passed when it differs from the default, which the decision module uses anyway
        config = {} if model.training_config == DEFAULT_TRAINING_CONFIG else {"training_config": model.training_config}
        self.decision_module = DQNDecisionModule(agent_type,unique_id,training,self.actions,self.n_features,checkpoint_path,epsilon,shared_replay_buffer,model.episode,**config)
        self.moving_module = MovingModule(self.unique_id, model, allotment, self.allocation_id, restrict_to_allocation)
        self.track_norms = track_norms
        self.norms_module_type = norms_module_type
        if self.track_norms:
            self.norms_module = self._build_norms_module()
        if agent_type != "baseline":
            self.rewards = self._ethics_rewards()
            self.ethics_module = EthicsModule(self.rewards["sanction"],agent_type)
        else:
            self.rewards = self._baseline_rewards()
    
    def step(self):
        if self.done == False:
            observation = self.observe()
            action = self.decision_module.choose_action(observation)
            self.current_reward, next_state, self.done = self.perform_transition(action, observation)
            self.current_reward = np.sum(self.current_reward)
            if self.training:
                self.decision_module.learn(observation, action, self.current_reward, next_state, self.done, self.model.episode)
            self.total_episode_reward += self.current_reward

    def perform_transition(self, action, observation):
        """
        Interaction Module (Algorithm 3) receives action from DQN and performs transition
        Observes state before acting and passes view to Norms Module for behaviour and norms handling (Algorithm 2)
        Performs action and observes next state
        Receives sanction from Ethics Module (Algorithm 1)
        Updates attributes and passess to Norms Module
        Returns reward, next state, done to DQN for learning
        """
        self.current_action = action
        self.current_thrown_to = None
        self.current_group_thrown_to = None
        self.current_agent_thrown_to = None
        antecedent = None
        if self.track_norms and self.norms_module_type == "rule_based":
            # rule-based norms are keyed on the state before acting, and so is the action's label
            antecedent = self.norms_module.get_antecedent([self.berries, self.health, self.model.get_society_well_being(self, True, False)]
                                                          + list(self.model.extra_observation_features(self)))
            self.current_norm_label = self._throw_norm_label(action)
        if self.agent_type != "baseline":
            society_well_being = self.model.get_society_well_being(self, False, True)
            self.ethics_module.update_ethics_state(society_well_being)
        reward_vector = [self._perform_action(action)]
        #recalculated straight after acting, so the sanction (and the next state) see the effect of every action on this agent's
        #well-being, e.g. a foraged berry; health decay comes after, in _update_attributes
        self.days_left_to_live = self.get_days_left_to_live()
        next_state = self.observe()
        if self.agent_type != "baseline":
            sanction = self._ethics_sanction()
            self.current_sanction = float(np.sum(sanction))
            reward_vector = reward_vector + sanction
        done, reward_vector = self._update_attributes(reward_vector)
        if self.track_norms:
            self._update_norms(antecedent, observation, action, np.sum(reward_vector).item())
        return reward_vector, next_state, done
    
    def observe(self):
        """
        The observation the model gives this agent (see HarvestModel.observe)
        """
        observation = self.model.observe(self)
        if len(observation) != self.n_features:
            raise NumFeaturesException(self.n_features, len(observation))
        return observation
    
    def get_days_left_to_live(self):
        """
        Get the days an agent has left to live (Equation 4)
        """
        days_left_to_live = (self.berry_health_payoff * self.berries) + self.health
        days_left_to_live = round(days_left_to_live / self.health_decay, DECIMALS)
        if days_left_to_live < 0:
            return 0
        return days_left_to_live

    def reset(self):
        """
        Reset agent for new episode
        """
        self.berries = 0
        self.berries_consumed = 0
        self.berries_thrown = 0
        self.health = self.start_health
        self.days_left_to_live = self.get_days_left_to_live()
        self.total_days_left_to_live = self.days_left_to_live
        self.done = False
        self.total_episode_reward = 0
        self.current_reward = 0
        self.current_sanction = None
        self.current_thrown_to = None
        self.current_group_thrown_to = None
        self.current_agent_thrown_to = None
        self.moving_module.reset()
        self.decision_module.start_episode()
        if self.track_norms:
            self.norms_module.reset()
    
    def _calculate_n_features(self):
        """
        Number of features in this agent's observation (see HarvestModel.observation_size)
        """
        return self.model.observation_size(self)

    def _generate_actions(self, unique_id, num_agents):
        """
        This agent's actions (see HarvestModel.agent_actions)
        """
        return self.model.agent_actions(self)
    
    def _perform_action(self, action_index):
        reward = 0
        action = self.actions[action_index]
        #action 0
        if action == "move":
            reward = self._move()
        #action 1
        elif action == "eat":
            reward = self._eat()
        #action 2+ (throw)
        else:
            agent_id = int(action.split("_")[1])
            reward = self._throw(agent_id)
        return reward
    
    def _move(self):
        if not self.moving_module.check_nearest_berry(self.pos):
            #if no berries have been found to walk towards, have to wait
            return self.rewards["neutral_reward"]
        #otherwise, we have a path, move towards the berry; returns True if we are at the end of the path and find a berry
        berry_found, new_pos = self.moving_module.move_towards_berry(self.pos)
        if berry_found:
            self.berries += 1
            return self.rewards["forage"]
        if new_pos != self.pos:
            self.model.move_agent_to_cell(self, new_pos)
        return self.rewards["neutral_reward"]
    
    def is_dead(self):
        """
        Whether this agent has died: earlier this step (done), or on an earlier day (taken off the grid)
        """
        return self.done or self.off_grid

    def _throw(self, benefactor_id):
        #a throw to a dead agent always fails, so it is recorded when it is tried, whatever else would have stopped it
        if self.model.agent_by_id(benefactor_id).is_dead():
            self.current_agent_thrown_to = "dead_other"
        if self.berries <= 0:
            return self.rewards["no_berries"]
        #have to have a minimum amount of health to throw
        if self.health < self.low_health_threshold:
            return self.rewards["insufficient_health"]
        for a in self.model.get_living_agents():
            if a.unique_id == benefactor_id:
                if a.agent_type == "berry":
                    raise AgentTypeException("agent to throw to", "berry")
                if a.done:
                    #died earlier this step (it leaves living_agents at the end of the step): no living benefactor
                    return self.rewards["no_benefactor"]
                if not self.model.can_throw_to(self, a):
                    #out of reach (e.g. in another allotment) is treated the same as no living benefactor
                    return self.rewards["no_benefactor"]
                #who the berry goes to, compared before it moves, so the giver still holds it and the receiver hasn't been fed
                self.current_thrown_to = a.unique_id
                self.current_group_thrown_to = self.model.group_thrown_to(self, a)
                self.current_agent_thrown_to = self._compare_well_off(a.get_days_left_to_live(), self.get_days_left_to_live())
                a.health = round(a.health + self.berry_health_payoff, DECIMALS)
                a.berries_consumed += 1
                a.days_left_to_live = a.get_days_left_to_live()
                self.berries -= 1
                self.berries_thrown += 1
                self.days_left_to_live = self.get_days_left_to_live()
                return self.rewards["throw"]
        return self.rewards["no_benefactor"]

    @staticmethod
    def _compare_well_off(receiver, giver):
        """
        "better_off_other", "worse_off_other" or "equal_other": the receiver's measure of how well off it is compared with the giver's
        """
        if receiver > giver:
            return "better_off_other"
        if receiver < giver:
            return "worse_off_other"
        return "equal_other"

    def _eat(self):
        if self.berries > 0:
            self.health = round(self.health + self.berry_health_payoff, DECIMALS)
            self.berries -= 1
            self.berries_consumed += 1
            return self.rewards["eat"]
        else:
            return self.rewards["no_berries"]

    def _update_attributes(self, reward_vector):
        done = False
        self.health = round(self.health - self.health_decay, DECIMALS)
        self.days_left_to_live = self.get_days_left_to_live()
        self.total_days_left_to_live += self.days_left_to_live 
        day = self.model.get_day()
        if self.health <= 0:
            #environment class checks for dead agents to remove at the end of each step
            done = True
            self.health = 0
            reward_vector[0] += self.rewards["death"]
        if day == self.max_days - 1:
            reward_vector[0] += self.rewards["survive"]
        reward_vector = np.array(reward_vector)
        return done, reward_vector

    def _baseline_rewards(self):
        #from the training config (configs/training/harvest.yaml unless run.py's --training-config gives another)
        return self.model.training_config.harvest_rewards(ethics=False)

    def _ethics_rewards(self):
        #the baseline's, with the training config's ethics rewards in their place, and the sanction the ethics module gives
        return self.model.training_config.harvest_rewards(ethics=True)

    # ── Norms ───────────────────────────────────────────────────

    def _norm_action_label(self, action):
        #the label worked out before acting (see _throw_norm_label), so it describes the state the norm's antecedent does
        return self.current_norm_label

    #how a norm about throwing names who the berry goes to
    AGENT_NORM_LABELS = {"better_off_other": "better off agent", "worse_off_other": "worse off agent", "equal_other": "equally off agent"}
    GROUP_NORM_LABELS = {"same_group": "same group", "better_off_other": "better off group", "worse_off_other": "worse off group",
                         "equal_other": "equally off group"}

    def _throw_norm_label(self, action):
        """
        The action's label for rule-based norms, worked out before acting: move and eat by name, and a throw by who it goes to,
        compared with this agent as it is before the berry moves: the receiver ("worse off agent", "better off agent", "equally off
        agent", or "dead agent"), then, in scenarios with groups, its group ("same group", or a "worse off", "better off" or "equally
        off group"), e.g. "throw,worse off agent,same group". A tried throw is labelled by who it was meant for, whether or not it
        succeeds, as norms are about what agents choose
        """
        name = self.actions[action]
        if not name.startswith("throw_"):
            return name
        target = self.model.agent_by_id(int(name.split("_")[1]))
        if target.is_dead():
            parts = ["throw", "dead agent"]
        else:
            parts = ["throw", self.AGENT_NORM_LABELS[self._compare_well_off(target.get_days_left_to_live(), self.get_days_left_to_live())]]
        group = self.model.group_thrown_to(self, target)
        if group is not None:
            parts.append(self.GROUP_NORM_LABELS[group])
        return ",".join(parts)

    def _norm_action_index(self, action):
        return self.norm_action_index[action]

    def _antecedent_features(self):
        """
        Registers with NormsModule how to bucket this agent's state into a natural language
        precondition string: berries and health as single scalar readings, well-being (one
        reading per other currently-observed agent) as a repeated feature, then any features the scenario adds.
        """
        return [
            {"boundaries": [self.low_berries_threshold, self.high_berries_threshold],
             "labels": ["low berries", "medium berries", "high berries"], "zero_label": "no berries"},
            {"boundaries": [self.low_health_threshold, self.high_health_threshold],
             "labels": ["low health", "medium health", "high health"]},
            {"boundaries": [self.low_days_left_threshold, self.high_days_left_threshold],
             "labels": ["low days", "medium days", "high days"], "repeated": True},
        ] + [{"boundaries": spec["thresholds"], "labels": spec["labels"]} for spec in self.model.extra_observation_feature_specs()]

    def _consequent_rules(self):
        """
        Registers with NormsModule how to generalise this agent's raw action names (move/eat/
        throw_X) into a natural language postcondition string: a raw throw_X collapses to "throw"; everything else (move, eat,
        and throws already named by who they go to, "throw,...", see _throw_norm_label) is used as-is.
        """
        return [{"prefix": "throw_", "label": "throw"}]

    def _build_bayesian_norms_module(self, n_agents):
        """
        Construct a BayesianNormsModule from this agent's observation layout and bucket thresholds.
        """
        n_others = n_agents - 1
        other_wellbeing_indices = list(range(4, 4 + n_others))

        #(name, observation index, thresholds): the agent's own features, then other agents' well-being (one reading each, so repeated),
        #then any features the scenario adds, which come after the other agents' well-being in the observation
        own_features = [("health", 0, [1, 5, 10]), ("berries", 1, [1, 2, 3]), ("self_wellbeing", 2, [10, 50, 100]), ("distance", 3, [1, 2, 3])]
        extra_features = [(spec["name"], 4 + n_others + i, spec["thresholds"]) for i, spec in enumerate(self.model.extra_observation_feature_specs())]
        feature_specs = [{"name": name, "index": index, "thresholds": thresholds, "repeated": False} for name, index, thresholds in own_features]
        feature_specs.append({"name": "other_wellbeing", "index": other_wellbeing_indices, "thresholds": [0.1, 10, 50, 100], "repeated": True})
        feature_specs += [{"name": name, "index": index, "thresholds": thresholds, "repeated": False} for name, index, thresholds in extra_features]
        max_predicates = 2

        # norms are about whether an agent shares at all, not who it shares with: every throw_X
        # collapses to a single "throw" norm action. norm_action_index maps each raw action index
        # (into self.actions) to its norm action index, for converting actions before update()
        norm_actions = ["move", "eat", "throw"]
        action_name_to_index = {name: idx for idx, name in enumerate(norm_actions)}
        self.norm_action_index = [
            action_name_to_index["throw" if name.startswith("throw_") else name]
            for name in self.actions
        ]

        return BayesianNormsModule(
            feature_specs=feature_specs,
            actions=norm_actions,
            max_predicates=max_predicates,
            action_name_to_index=action_name_to_index
        )
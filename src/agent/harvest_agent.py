from mesa import Agent
from .modules.dqn_decision_module import DQNDecisionModule
from .modules.momp_dqn_decision_module import MPDQNDecisionModule
from .modules.moving_module import MovingModule
from .modules.norms_module import NormsModule
from .modules.ethics_module import EthicsModule
from src.harvest_exception import NumFeaturesException
from src.harvest_exception import AgentTypeException
import numpy as np

class HarvestAgent(Agent):
    def __init__(self,unique_id,model,agent_type,min_width,max_width,min_height,max_height,training,checkpoint_path,epsilon,write_norms,n_rewards=1,shared_replay_buffer=None):
        super().__init__(unique_id,model)
        self.done = False
        self.current_reward = 0
        self.total_episode_reward = 0
        self.training = training
        self.agent_type = agent_type
        self.n_features = self._calculate_n_features()
        self.n_rewards = n_rewards
        self.start_health = 0.8
        self.health = self.start_health
        self.health_decay = 0.1
        self.low_health_threshold = 0.6
        self.berry_health_payoff = 0.6
        self.berries = 0
        self.berries_consumed = 0
        self.berries_thrown = 0
        self.days_survived = 0
        self.days_left_to_live = self.get_days_left_to_live()
        self.total_days_left_to_live = self.days_left_to_live
        self.max_days = self.model.get_max_days()
        self.actions = self._generate_actions(self.unique_id, model.get_num_agents())
        self.off_grid = False
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        if self.agent_type == "multiobjective_mp":
            self.decision_module = MPDQNDecisionModule(agent_type,unique_id,training,self.actions,self.n_features,checkpoint_path,epsilon,shared_replay_buffer)
        else:
            self.decision_module = DQNDecisionModule(agent_type,unique_id,training,self.actions,self.n_features,checkpoint_path,epsilon,n_rewards,shared_replay_buffer)
        self.moving_module = MovingModule(self.unique_id, model, training, min_width, max_width, min_height, max_height)
        self.write_norms = write_norms
        if self.write_norms:
            self.norms_module = NormsModule(self.unique_id)
        if agent_type != "baseline":
            self.rewards = self._ethics_rewards()
            self.ethics_module = EthicsModule(self.rewards["sanction"],agent_type)
        else:
            self.rewards = self._baseline_rewards()
    
    def step(self):
        if self.done == False:
            observation = self.observe()
            action = self.decision_module.choose_action(observation)
            self.current_reward, next_state, self.done = self.perform_transition(action)
            if self.n_rewards == 1:
                self.current_reward = np.sum(self.current_reward)
            if self.training:
                self.decision_module.learn(observation, action, self.current_reward, next_state, self.done, self.model.episode)
            self.total_episode_reward += sum(self.current_reward) if self.n_rewards > 1 else self.current_reward

    def perform_transition(self, action):
        """
        Interaction Module (Algorithm 3) receives action from DQN and performs transition
        Observes state before acting and passes view to Norms Module for behaviour and norms handling (Algorithm 2)
        Performs action and observes next state
        Receives sanction from Ethics Module (Algorithm 1)
        Updates attributes and passess to Norms Module
        Returns reward, next state, done to DQN for learning
        """
        done = False
        self.current_action = action
        if self.write_norms:
            antecedent = self.norms_module.get_antecedent(self.berries, self.health, self.model.get_society_well_being(self, True, False))
        if self.agent_type != "baseline":
            self.ethics_module.day = self.model.get_day()
            self._update_ethics()
        reward_vector = [self._perform_action(action)]
        next_state = self.observe()
        if self.agent_type != "baseline":
            reward_vector = reward_vector + self._ethics_sanction()
        done, reward_vector = self._update_attributes(reward_vector)
        if self.write_norms:
            self.norms_module.update_behaviour_base(antecedent, self.actions[action], reward_vector, self.model.get_day())
            if ("no berries" in antecedent and action == "throw") or ("eat" in antecedent and self.actions[action] == "throw"):
                #raise ImpossibleNormException(self.unique_id, antecedent, self.actions[action], reward)
                print(self.model.episode, self.model.day, "agent", self.agent_id, antecedent, "reward", reward_vector, "berries", self.berries, "health", self.health)
        return reward_vector, next_state, done
    
    def observe(self):
        """
        Agents observe their attributes, distance to nearest berry, well-being of other agents in society
        """
        distance_to_berry = self.moving_module.get_distance_to_berry()
        observer_features = np.array([self.health, self.berries, self.days_left_to_live, distance_to_berry])
        agent_well_being = self.model.get_society_well_being(self, False, False)
        observation = np.append(observer_features, agent_well_being)
        if len(observation) != self.n_features:
            raise NumFeaturesException(self.n_features, len(observation))
        return observation
    
    def get_days_left_to_live(self):
        """
        Get the days an agent has left to live (Equation 4)
        """
        days_left_to_live = (self.berry_health_payoff * self.berries) + self.health
        days_left_to_live = days_left_to_live / self.health_decay
        if days_left_to_live < 0:
            return 0
        return days_left_to_live

    def get_epsilon(self):
        return self.decision_module.get_epsilon()

    def get_mean_loss(self):
        return self.decision_module.get_mean_loss()
    
    def finish_episode(self, end_day):
        if self.off_grid == False:
            self.days_survived = end_day
        self.decision_module.save_models()

    def reset(self):
        """
        Reset agent for new episode
        """
        self.berries = 0
        self.berries_consumed = 0
        self.berries_thrown = 0
        self.max_berries = 0
        self.health = self.start_health
        self.days_left_to_live = self.get_days_left_to_live()
        self.total_days_left_to_live = self.days_left_to_live
        self.days_survived = 0
        self.done = False
        self.total_episode_reward = 0
        self.current_reward = 0
        self.moving_module.reset()
        if self.write_norms:
            self.norms_module.behaviour_base  = {}
    
    def _calculate_n_features(self):
        """
        Get number of features in observation (agent's health, days left to live, distance to berry, well-being of other agents in society)
        """
        n_features = 4
        n_features += self.model.get_num_agents() -1
        return n_features
    
    def _generate_actions(self, unique_id, num_agents):
        actions = ["move", "eat"]
        for agent_id in range(num_agents):
            if agent_id != unique_id:
                actions.append(f"throw_{agent_id}")
        return actions
    
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
    
    def _throw(self, benefactor_id):
        if self.berries <= 0:
            return self.rewards["no_berries"]
        #have to have a minimum amount of health to throw
        if self.health < self.low_health_threshold:
            return self.rewards["insufficient_health"]
        for a in self.model.get_living_agents():
            if a.unique_id == benefactor_id:
                if a.agent_type == "berry":
                    raise AgentTypeException("agent to throw to", "berry")
                a.health += self.berry_health_payoff 
                a.berries_consumed += 1
                a.days_left_to_live = a.get_days_left_to_live()
                self.berries -= 1
                self.berries_thrown += 1
                self.days_left_to_live = self.get_days_left_to_live()
                return self.rewards["throw"]
        return self.rewards["no_benefactor"]
    
    def _eat(self):
        if self.berries > 0:
            self.health += self.berry_health_payoff
            self.berries -= 1
            self.berries_consumed += 1
            return self.rewards["eat"]
        else:
            return self.rewards["no_berries"]
    
    def _ethics_sanction(self):
        society_well_being = self.model.get_society_well_being(self, False, True)
        sanction = self.ethics_module.get_sanction(society_well_being)
        return sanction
    
    def _update_ethics(self):
        society_well_being = self.model.get_society_well_being(self, False, True)
        if self.berries > 0 and self.health >= self.low_health_threshold:
            can_help = True
            self.ethics_module.update_ethics_state(can_help, society_well_being)
        else:
            can_help = True
            self.ethics_module.update_ethics_state(can_help, society_well_being)
    
    def _update_attributes(self, reward_vector):
        done = False
        self.health -= self.health_decay
        self.days_left_to_live = self.get_days_left_to_live()
        self.total_days_left_to_live += self.days_left_to_live 
        day = self.model.get_day()
        if self.health <= 0:
            #environment class checks for dead agents to remove at the end of each step
            done = True
            self.days_survived = day
            self.health = 0
            reward_vector[0] += self.rewards["death"]
        if day == self.max_days - 1:
            reward_vector[0] += self.rewards["survive"]
        reward_vector = np.array(reward_vector)
        return done, reward_vector

    def _baseline_rewards(self):
        rewards = {"death": -1,
                   "no_berries": -0.2,
                   "no_benefactor": -0.2,
                   "insufficient_health": -0.2,
                   "neutral_reward": 0,
                   "throw": 0.5,
                   "forage": 1,
                   "eat": 1,
                   "survive": 1
                   }
        return rewards
    
    def _ethics_rewards(self):
        rewards = {"death": -1,
                   "no_berries": -0.1,
                   "no_benefactor": -0.1,
                   "insufficient_health": -0.1,
                   "neutral_reward": 0,
                   "sanction": 0.4,
                   "throw": 0.5,
                   "forage": 0.8,
                   "eat": 0.8,
                   "survive": 1
                   }
        return rewards
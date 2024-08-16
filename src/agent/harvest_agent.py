from .dqn.dqn_agent import DQNAgent
from .moving_module import MovingModule
from .norms_module import NormsModule
from .ethics_module import EthicsModule
from src.harvest_exception import NumFeaturesException
from src.harvest_exception import AgentTypeException
import numpy as np

class HarvestAgent(DQNAgent):
    """
    Agent acts in an environment and receives a reward
    Modules:
        Interaction module -- receives action from DQN, calls norms and ethics module and updates attributes (Algorithm 3)
        Moving module -- handles pathfinding and returns coordinates for agent to move to
        Norms module -- stores and generates norms from view of state (Algorithm 2)
        Ethics module -- evaluates societal well-being before and after acting and generates a self-directed sanction (Algorithm 1)
    Instance variables:
        actions -- possible actions available to an agent (move, eat, throw to each agent)
        health -- current health
        berries -- number of berries currently carrying
        berries_consumed -- history of eaten berries of current episode
        days_survived -- number of days survived of current episode
        max_days -- maximum number of days in an episode
        max/min width/height -- dimensions of the grid agent can access
        health_decay -- decay of health at each timestep
        days_left_to_live -- number of days an agent can live for given their health, health decay, and number of berries they are carrying
        total_days_left_to_live -- cumulative days left to live of current episode
        berry_health_payoff -- payoff received from eating a berry
        low_health_threshold -- minimum health required to throw a berry
        agent_type -- baseline or maximin
        write_norms -- boolean whether agent is tracking norms
        rewards -- dictionary of rewards received
        off_grid -- status of agent on the grid; agent is removed from the grid upon death
        current_action -- the current action being performed
    """
    def __init__(self,unique_id,model,agent_type,max_days,min_width,max_width,min_height,max_height,training,checkpoint_path,epsilon,write_norms,shared_replay_buffer=None):
        self.actions = self._generate_actions(unique_id, model.get_num_agents())
        #dqn agent class handles learning and action selection
        super().__init__(unique_id,model,agent_type,self.actions,training,checkpoint_path,epsilon,shared_replay_buffer=shared_replay_buffer)
        self.start_health = 0.8
        self.health = self.start_health
        self.berries = 0
        self.berries_consumed = 0
        self.berries_thrown = 0
        self.days_survived = 0
        self.max_days = max_days
        self.max_width = max_width
        self.min_width = min_width
        self.width = max_width - min_width + 1
        self.max_height = max_height
        self.min_height = min_height
        #self.height = max_height - min_height + 1
        self.health_decay = 0.1
        self.days_left_to_live = self.health/self.health_decay
        self.total_days_left_to_live = self.days_left_to_live
        self.berry_health_payoff = 0.6
        self.low_health_threshold = 0.6
        self.agent_type = agent_type
        self.write_norms = write_norms
        self.moving_module = MovingModule(self.unique_id, model, training, max_width, max_height)
        self.norms_module = NormsModule(self.unique_id)
        if agent_type != "baseline":
            self.rewards = self._ethics_rewards()
            self.ethics_module = EthicsModule(self.unique_id,self.rewards["sanction"])
        else:
            self.rewards = self._baseline_rewards()
        self.off_grid = False
        self.current_action = None
        
    def interaction_module(self, action):
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
        society_well_being = self.model.get_society_well_being(self, True)
        if self.write_norms:
            antecedent = self.norms_module.get_antecedent(self.health, self.berries, society_well_being)
        if self.agent_type != "baseline":
            self.ethics_module.day = self.model.get_day()
            can_help = self._update_ethics(society_well_being)
        reward = self._perform_action(action)
        next_state = self.observe()
        if self.agent_type != "baseline":
            reward += self._ethics_sanction(can_help)
            #print("day", self.model.get_day(), "agent", self.unique_id, "reward after sanction", reward)
        done, reward = self._update_attributes(reward)
        #print("day", self.model.get_day(), "agent", self.unique_id, "action", action, "reward", reward)
        if self.write_norms:
            self.norms_module.update_behaviour_base(antecedent, self.actions[action], reward, self.model.get_day())
        return reward, next_state, done
        
    def observe(self):
        """
        Agents observe their attributes, distance to nearest berry, well-being of other agents in society
        """
        distance_to_berry = self.moving_module.get_distance_to_berry()
        observer_features = np.array([self.health, self.berries, self.days_left_to_live, distance_to_berry])
        agent_well_being = self.model.get_society_well_being(self, False)
        observation = np.append(observer_features, agent_well_being)
        if len(observation) != self.n_features:
            raise NumFeaturesException(self.n_features, len(observation))
        return observation
    
    def get_n_features(self):
        """
        Get number of features in observation (agent's health, days left to live, distance to berry, well-being of other agents in society)
        """
        n_features = 4
        n_features += self.model.get_num_agents() -1
        return n_features
      
    def reset(self):
        """
        Reset agent for new episode
        """
        self.done = False
        self.total_episode_reward = 0
        self.berries = 0
        self.berries_consumed = 0
        self.berries_thrown = 0
        self.max_berries = 0
        self.health = self.start_health
        self.current_reward = 0
        self.days_left_to_live = self.get_days_left_to_live()
        self.total_days_left_to_live = self.days_left_to_live
        self.days_survived = 0
        self.norms_module.behaviour_base  = {}
        self.moving_module.reset()

    def get_days_left_to_live(self):
        """
        Get the days an agent has left to live (Equation 4)
        """
        days_left_to_live = (self.berry_health_payoff * self.berries) + self.health
        days_left_to_live = days_left_to_live / self.health_decay
        if days_left_to_live < 0:
            return 0
        return days_left_to_live
    
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
            #print("day", self.model.get_day(), "agent", self.unique_id, "performing action", action, "wellbeing", self.model.get_society_well_being(self, True))
            agent_id = int(action.split("_")[1])
            reward = self._throw(agent_id)
            #print("day", self.model.get_day(), "agent", self.unique_id, "after performing action", action, "wellbeing", self.model.get_society_well_being(self, True), "reward", reward)
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

    def _ethics_sanction(self, can_help):
        if not can_help:
            return 0
        society_well_being = self.model.get_society_well_being(self, True)
        sanction = self.ethics_module.get_sanction(society_well_being)
        #print("day", self.model.get_day(), "agent", self.unique_id, "sanction", sanction, "well being", society_well_being)
        return sanction
    
    def _update_ethics(self, society_well_being):
        if self.berries > 0 and self.health >= self.low_health_threshold:
            can_help = True
            self.ethics_module.update_social_welfare(self.agent_type, society_well_being)
        else:
            can_help = False
        #print("day", self.model.get_day(), "agent", self.unique_id, "berries", self.berries, "health", self.health, "can help", can_help, "social welfare", society_well_being, "measure", self.ethics_module._measure_of_well_being, "minimums", self.ethics_module._number_of_minimums)
        return can_help
    
    def _update_attributes(self, reward):
        done = False
        self.health -= self.health_decay
        self.days_left_to_live = self.get_days_left_to_live()
        self.total_days_left_to_live += self.days_left_to_live 
        day = self.model.get_day()
        # if len(self.model.get_living_agents()) < self.model.get_num_agents():
        #     reward += self.rewards["death"]
        #     self.days_survived = day
        #     done = True
        if self.health <= 0:
            #environment class checks for dead agents to remove at the end of each step
            done = True
            self.days_survived = day
            self.health = 0
            reward += self.rewards["death"]
        if day == self.max_days - 1:
            reward += self.rewards["survive"]
        return done, reward
    
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
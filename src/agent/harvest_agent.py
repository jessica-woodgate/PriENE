from .dqn.dqn_agent import DQNAgent
from .moving_module import MovingModule
from .norms_module import NormsModule
from .ethics_module import EthicsModule
import numpy as np

class HarvestAgent(DQNAgent):
    def __init__(self,unique_id,model,agent_type,min_width,max_width,min_height,max_height,n_features,training,epsilon,shared_replay_buffer=None):
        self.actions = ["move", "eat", "random_throw", "egalitarian_throw", "maximin_throw", "utilitarian_throw"]
        #dqn agent class handles learning and action selection
        super().__init__(unique_id,model,agent_type,self.actions,n_features,training,epsilon,shared_replay_buffer=shared_replay_buffer)
        self.health = 0.8
        self.berries = 0
        self.berries_consumed = 0
        self.berries_thrown = 0
        self.days_survived = 0
        self.max_width = max_width
        self.min_width = min_width
        self.width = max_width - min_width + 1
        self.max_height = max_height
        self.min_height = min_height
        self.height = max_height - min_height + 1
        self.health_decay = 0.1
        self.days_left_to_live = self.health/self.health_decay
        self.berry_health_payoff = 0.6
        self.low_health_threshold = 0.6
        self.agent_type = agent_type
        self._moving_module = MovingModule(model)
        self.norm_module = NormsModule(self.unique_id)
        self._norm_clipping_frequency = 10
        if agent_type != "baseline":
            self._rewards = self._ethics_rewards()
            self._ethics_module = EthicsModule(self._rewards["shaped_reward"])
        else:
            self._rewards = self._baseline_rewards()
        self.off_grid = False
        self.current_action = None
        
    def execute_transition(self, action):
        """
        execute_transition updates the ethics module with its ability to act ethically (has berries)
        calls ethics module to store measure of social welfare appropriate for the principle
        performs action, gets sanction from ethics module
        updates attributes and writes norms
        """
        done = False
        self.current_action = action
        if self.model.write_norms:
            self.norm_module.update_norm_age()
            antecedent = self.norm_module.get_antecedent(self.health, self.berries, society_well_being)
        if self.agent_type != "baseline" and self.agent_type != "berry":
            if self.berries > 0:
                have_berries = True
            else:
                have_berries = False
            society_well_being = [x.days_left_to_live for x in self.model.living_agents]
            self._ethics_module.update_state(self.agent_type, society_well_being, have_berries)
        reward = self._perform_action(action)
        next_state = self.model.observe(self)
        if self.agent_type != "baseline" and self.agent_type != "berry":
            society_well_being = [x.days_left_to_live for x in self.model.living_agents]
            reward += self._ethics_module.get_sanction(society_well_being)
        done, reward = self._update_attributes(reward)
        if self.model.write_norms:
            self.norm_module.update_norm(antecedent, self.actions[action], reward)
            if self.model.day % self._norm_clipping_frequency == 0:
                self.norm_module.clip_norm_base()
        return reward, next_state, done
    
    def _perform_action(self, action):
        reward = 0
        #action 0
        if self.actions[action] == "move":
            reward = self._move()
        #action 1
        elif self.actions[action] == "eat":
            reward = self._eat()
        #action 2
        elif self.actions[action] == "random_throw":
            benefactor = np.random.choice(self.model.living_agents)
            reward = self._throw(benefactor)
        #action 3
        elif self.actions[action] == "egalitarian_throw":
            benefactor = None
            reward = self._throw(benefactor)
        #action 4
        elif self.actions[action] == "maximin_throw":
            benefactor = None
            reward = self._throw(benefactor)
        #action 5
        elif self.actions[action] == "utilitarian_throw":
            benefactor = None
            reward = self._throw(benefactor)
        return reward
    
    def _move(self):
        if not self._moving_module.check_nearest_berry():
            #if no berries have been found to walk towards, have to wait
            return self._rewards["empty_forage"]
        if self._moving_module.move_towards_berry():
            self.berries += 1
            return self._rewards["forage"]
        return self._rewards["empty_forage"]
    
    def _throw(self):
        if self.berries <= 0:
            return self._rewards["no_berries"]
        #have to have a minimum amount of health to throw
        if self.health < self.low_health_threshold:
            return self._rewards["insufficient_health"]
        benefactor = self.choose_benefactor()
        if not benefactor:
            return self._rewards["no_benefactor"]
        assert(benefactor.agent_type != "berry")
        benefactor.health += self.berry_health_payoff 
        benefactor.berries_consumed += 1
        benefactor.days_left_to_live = benefactor.get_days_left_to_live()
        self.berries -= 1
        self.berries_thrown += 1
        return self._rewards["throw"]
    
    def choose_benefactor(self):
        benefactor = [a for a in self.model.living_agents if a.unique_id != self.unique_id]
        if len(benefactor) > 0:
            return benefactor[0]
        else:
            return False
    
    def _eat(self):
        if self.berries > 0:
            self.health += self.berry_health_payoff
            self.berries -= 1
            self.berries_consumed += 1
            return self._rewards["eat"]
        else:
            return self._rewards["no_berries"]
    
    def _baseline_rewards(self):
        rewards = {"crash": -0.2,
                   "no_berries": -0.2,
                   "no_benefactor": -0.2,
                   "insufficient_health": -0.2,
                   "empty_forage": 0,
                   "throw": 0.5,
                   "forage": 1,
                   "eat": 1
                   }
        return rewards
    
    def _ethics_rewards(self):
        rewards = {"crash": -0.1,
                   "no_berries": -0.1,
                   "no_benefactor": -0.1,
                   "insufficient_health": -0.1,
                   "empty_forage": 0,
                   "shaped_reward": 0.4,
                   "throw": 0.5,
                   "forage": 0.8,
                   "eat": 0.8
                   }
        return rewards

    def get_days_left_to_live(self):
        health = self.health
        health += self.berry_health_payoff * self.berries
        days_left_to_live = health / self.health_decay
        if days_left_to_live < 0:
            return 0
        return days_left_to_live
    
    def _update_attributes(self, reward):
        done = False
        self.health -= self.health_decay
        self.days_left_to_live = self.get_days_left_to_live()
        if len(self.model.living_agents) < self.model.num_agents:
            reward -= 1
            self.days_survived = self.model.day
            done = True
        if self.health <= 0:
            #environment class checks for dead agents to remove at the end of each step
            done = True
            self.days_survived = self.model.day
            self.health = 0
            reward -= 1
        if self.model.day == self.model.max_days - 1:
            reward += 1
        return done, reward
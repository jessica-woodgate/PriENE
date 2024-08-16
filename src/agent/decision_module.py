from .moving_module import MovingModule
from .norms_module import NormsModule
from .ethics_module import EthicsModule
from src.harvest_exception import NumFeaturesException
import numpy as np

class DecisionModule():
    def __init__(self,model,agent_type,max_width,max_height,training,write_norms,n_features):
        self.model = model
        self.agent_type = agent_type
        self.n_features = n_features
        self.write_norms = write_norms
        self.dqn_module = DQNAgent(unique_id,model,agent_type,self.actions,training,checkpoint_path,epsilon,shared_replay_buffer=shared_replay_buffer)
        self.moving_module = MovingModule(self.unique_id, model, training, max_width, max_height)
        if self.write_norms:
            self.norms_module = NormsModule(self.unique_id)
        if agent_type != "baseline":
            self.rewards = self._ethics_rewards()
            self.ethics_module = EthicsModule(self.unique_id,self.rewards["sanction"])
        else:
            self.rewards = self._baseline_rewards()

    def get_action(self, observation, action, health, berries, society_well_being):
        """
        execute_transition updates the ethics module with its ability to act ethically (has berries)
        calls ethics module to store measure of social welfare appropriate for the principle
        performs action, gets sanction from ethics module
        updates attributes and writes norms
        """
        action = self.dqn_module()
        if self.write_norms:
            self.antecedent = self.norms_module.get_antecedent(health, berries, society_well_being)
        if self.agent_type != "baseline":
            self.can_help = self._update_ethics(society_well_being)
        return action

    def after_acting(self, next_state, action_string):
        if self.agent_type != "baseline":
            reward += self._ethics_sanction(self.can_help)
            #print("day", self.model.get_day(), "agent", self.unique_id, "reward after sanction", reward)
        done, reward = self._update_attributes(reward)
        #print("day", self.model.get_day(), "agent", self.unique_id, "action", action, "reward", reward)
        if self.write_norms:
            self.norms_module.update_behaviour_base(self.antecedent, action_string, reward, self.model.get_day())
        return reward, next_state, done
    
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
    
from .modules.interaction_module import InteractionModule
from .dqn.dqn import DQN
from .dqn.mosp_dqn import MODQN
import numpy as np
import os

class DQNAgent():
    def __init__(self,unique_id,model,agent_type,training,checkpoint_path,epsilon,min_width,max_width,min_height,max_height,write_norms,n_rewards=1,shared_replay_buffer=None):
        self.unique_id = unique_id
        self.model = model
        self.agent_type = agent_type
        self.training = training
        self.n_features = self._calculate_n_features()
        self.interaction_module = InteractionModule(unique_id,model,agent_type,self.n_features,min_width,max_width,min_height,max_height,training,write_norms)
        self.actions = self.interaction_module.get_actions()
        self.epsilon = epsilon
        self.min_exploration_prob = 0.01
        self.expl_decay = 0.001
        self.total_episode_reward = 0
        self.n_actions = len(self.actions)
        self.n_rewards = n_rewards
        self.done = False
        self.shared_replay_buffer = shared_replay_buffer
        self.learn_step = 0
        self.replace_target_iter = 50
        self.current_reward = 0
        self.training = training
        self._init_networks(checkpoint_path)

    def step(self):
        """
        Step oberves current state, chooses an action using Q network, performs action using interaction module and learns if training
        """
        if self.done == False:
            observation = self.interaction_module.observe()
            action = self.q_network.choose_action(observation,self.epsilon)
            self.current_reward, next_state, self.done = self.interaction_module.perform_transition(action)
            if self.n_rewards == 1:
                self.current_reward = np.sum(self.current_reward)
            if self.training:
                self._learn(observation, action, self.current_reward, next_state, self.done)
                self.epsilon = max(self.min_exploration_prob, np.exp(-self.expl_decay*self.model.episode))
            self.total_episode_reward += sum(self.current_reward)

    def save_models(self):
        """
        Save q and target networks to file
        """
        self.q_network.dqn.save(self.q_checkpoint_path)
        self.target_network.dqn.save(self.target_checkpoint_path)
    
    def reset(self):
        self.done = False
        self.total_episode_reward = 0
        self.current_reward = 0
        self.interaction_module.reset()
    
    def _calculate_n_features(self):
        """
        Get number of features in observation (agent's health, days left to live, distance to berry, well-being of other agents in society)
        """
        n_features = 4
        n_features += self.model.get_num_agents() -1
        return n_features

    def _init_networks(self, checkpoint_path):
        #need to init a network for each objective
        if self.training:
            self.q_checkpoint_path = checkpoint_path+self.agent_type+"/agent_"+str(self.unique_id)+"/q_model_variables.keras"
            self.target_checkpoint_path = checkpoint_path+self.agent_type+"/agent_"+str(self.unique_id)+"/target_model_variables.keras"
            os.makedirs(os.path.dirname(self.q_checkpoint_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.target_checkpoint_path), exist_ok=True)
        else:
            self.q_checkpoint_path = checkpoint_path+self.agent_type+"/agent_"+str(self.unique_id)+"/q_model_variables.keras"
            self.target_checkpoint_path = checkpoint_path+self.agent_type+"/agent_"+str(self.unique_id)+"/target_model_variables.keras"
        self.q_network = DQN(self.actions,(self.n_features,),self.training,checkpoint_path=self.q_checkpoint_path,shared_replay_buffer=self.shared_replay_buffer)
        self.target_network = DQN(self.actions,(self.n_features,),self.training,checkpoint_path=self.target_checkpoint_path,shared_replay_buffer=self.shared_replay_buffer)
        if self.training:
            inputs = np.zeros(self.n_features)
            self.q_network.dqn(np.atleast_2d(inputs.astype('float32')))
            self.target_network.dqn(np.atleast_2d(inputs.astype('float32')))
            self.losses = list()
    
    def _learn(self, observation, action, reward, next_state, done):
        #vectorised rewards
        experience = {"s":observation, "a":action, "r":reward, "s_":next_state, "done":done}
        self.q_network.add_experience(experience)
        loss = self.q_network.train(self.target_network)
        self._append_losses(loss)
        self.learn_step += 1
        if self.learn_step % self.replace_target_iter == 0:
            self.target_network.copy_weights(self.q_network)
    
    def _append_losses(self, loss):
        if isinstance(loss, int):
            self.losses.append(loss)
        else:
            self.losses.append(loss.numpy())
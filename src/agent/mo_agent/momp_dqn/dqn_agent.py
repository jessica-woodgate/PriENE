from mesa import Agent
import numpy as np
from .dqn import DQN
from abc import abstractmethod
from src.harvest_exception import NumFeaturesException
import os

class DQNAgent(Agent):
    """
    DQN Agent learns and selects actions using Q network
    Instance variables:
        epsilon -- probability of exploration
        min_exploration_prob -- smallest that epsilon can decay to during training
        expl_decay -- rate of exponential decay of epsilon
        total_episode_reward -- total reward of current episode
        actions -- possible actions available to agent
        n_actions -- number of actions
        n_features -- number of features in DQN (length of observation)
        done -- whether agent has finished
        shared_replay_buffer -- experience replay buffer shared amongst agents
        learn_step -- current step of learning
        replace_target_iter -- interval for updating weights of target network
        agent_type -- baseline or maximin, for file name to saving model weights
        current_reward -- reward of current step
        training -- boolean training or testing
        q_checkpoint_path -- file path for q network (saving or loading)
        target_checkpoint_path -- file path for target network (saving or loading)
        losses -- history of losses
    """
    def __init__(self,unique_id,model,agent_type,actions,training,checkpoint_path,epsilon,shared_replay_buffer=None):
        super().__init__(unique_id, model)
        self.epsilon = epsilon
        self.min_exploration_prob = 0.01
        self.expl_decay = 0.001
        self.total_episode_reward = 0
        self.actions = actions
        self.n_actions = len(self.actions)
        self.n_features = self.get_n_features()
        self.done = False
        self.shared_replay_buffer = shared_replay_buffer
        self.learn_step = 0
        self.replace_target_iter = 50
        self.agent_type = agent_type
        self.current_reward = 0
        self.training = training
        if self.training:
            self.q_checkpoint_path = checkpoint_path+self.agent_type+"/agent_"+str(unique_id)+"/q_model_variables.keras"
            self.target_checkpoint_path = checkpoint_path+self.agent_type+"/agent_"+str(unique_id)+"/target_model_variables.keras"
            os.makedirs(os.path.dirname(self.q_checkpoint_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.target_checkpoint_path), exist_ok=True)
        else:
            self.q_checkpoint_path = checkpoint_path+self.agent_type+"/agent_"+str(unique_id)+"/q_model_variables.keras"
            self.target_checkpoint_path = checkpoint_path+self.agent_type+"/agent_"+str(unique_id)+"/target_model_variables.keras"

        self.q_network = DQN(self.actions,(self.n_features,),self.training,checkpoint_path=self.q_checkpoint_path,shared_replay_buffer=self.shared_replay_buffer)
        self.target_network = DQN(self.actions,(self.n_features,),self.training,checkpoint_path=self.target_checkpoint_path,shared_replay_buffer=self.shared_replay_buffer)
        if self.training:
            inputs = np.zeros(self.n_features)
            self.q_network.dqn(np.atleast_2d(inputs.astype('float32')))
            self.target_network.dqn(np.atleast_2d(inputs.astype('float32')))
            self.losses = list()
    
    @abstractmethod
    def interaction_module(self):
        raise NotImplementedError
    
    @abstractmethod
    def observe(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_n_features(self):
        raise NotImplementedError
    
    def step(self):
        """
        Step oberves current state, chooses an action using Q network, performs action using interaction module and learns if training
        """
        if self.done == False:
            observation = self.observe()
            if len(observation) != self.n_features:
                raise NumFeaturesException(self.n_features, len(observation))
            action = self.q_network.choose_action(observation,self.epsilon)
            self.current_reward, next_state, self.done = self.interaction_module(action)
            if self.training:
                self._learn(observation, action, self.current_reward, next_state, self.done)
                self.epsilon = max(self.min_exploration_prob, np.exp(-self.expl_decay*self.model.episode))
            self.total_episode_reward += self.current_reward

    def save_models(self):
        """
        Save q and target networks to file
        """
        self.q_network.dqn.save(self.q_checkpoint_path)
        self.target_network.dqn.save(self.target_checkpoint_path)
    
    def _learn(self, observation, action, reward, next_state, done):
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
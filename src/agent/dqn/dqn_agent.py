from mesa import Agent
import numpy as np
from .dqn import DQN
from abc import abstractmethod
import os

class DQNAgent(Agent):
    def __init__(self,unique_id,model,agent_type,actions,training,epsilon,shared_replay_buffer=None):
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
            self.q_checkpoint_path = "model_variables/current_run/"+self.agent_type+"/agent_"+str(unique_id)+"/q_model_variables.keras"
            self.target_checkpoint_path = "model_variables/current_run/"+self.agent_type+"/agent_"+str(unique_id)+"/target_model_variables.keras"
            os.makedirs(os.path.dirname(self.q_checkpoint_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.target_checkpoint_path), exist_ok=True)
        else:
            self.q_checkpoint_path = "model_variables/current_run/"+self.agent_type+"/agent_"+str(unique_id)+"/q_model_variables.keras"
            self.target_checkpoint_path = "model_variables/current_run/"+self.agent_type+"/agent_"+str(unique_id)+"/target_model_variables.keras"

        self.hidden_units = round(((self.n_features/3) * 2) + (2 * self.n_actions))
        self.qNetwork = DQN(self.actions,(self.n_features,),self.training,checkpoint_path=self.q_checkpoint_path,shared_replay_buffer=self.shared_replay_buffer)
        self.targetNetwork = DQN(self.actions,(self.n_features,),self.training,checkpoint_path=self.target_checkpoint_path,shared_replay_buffer=self.shared_replay_buffer)
        if self.training:
            inputs = np.zeros(self.n_features)
            self.qNetwork.dqn(np.atleast_2d(inputs.astype('float32')))
            self.targetNetwork.dqn(np.atleast_2d(inputs.astype('float32')))
            self.losses = list()
    
    @abstractmethod
    def execute_transition(self):
        raise NotImplementedError
    
    @abstractmethod
    def observe(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_n_features(self):
        raise NotImplementedError
    
    def step(self):
        """
        if agents are being tested, they do not learn
        """
        if self.done == False:
            observation = self.observe()
            assert(observation.size == self.n_features), f"expected {self.n_features}, got {observation.size}"
            action = self.qNetwork.choose_action(observation,self.epsilon)
            self.current_reward, next_state, self.done = self.execute_transition(action)
            if self.training:
                self._learn(observation, action, self.current_reward, next_state, self.done)
                self.epsilon = max(self.min_exploration_prob, np.exp(-self.expl_decay*self.model.episode))
            self.total_episode_reward += self.current_reward

    def save_models(self):
        self.qNetwork.dqn.save(self.q_checkpoint_path)
        self.targetNetwork.dqn.save(self.target_checkpoint_path)
    
    def _learn(self, observation, action, reward, next_state, done):
        experience = {"s":observation, "a":action, "r":reward, "s_":next_state, "done":done}
        self.qNetwork.add_experience(experience)
        loss = self.qNetwork.train(self.targetNetwork)
        self._append_losses(loss)
        self.learn_step += 1
        if self.learn_step % self.replace_target_iter == 0:
            self.targetNetwork.copy_weights(self.qNetwork)
    
    def _append_losses(self, loss):
        if isinstance(loss, int):
            self.losses.append(loss)
        else:
            self.losses.append(loss.numpy())
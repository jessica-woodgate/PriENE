from ..dqn.dqn import DQN
import numpy as np
import os

class DQNDecisionModule():
    def __init__(self,agent_type,unique_id,training,actions,n_features,checkpoint_path,epsilon,shared_replay_buffer=None):
        self.n_features = n_features
        self.agent_type = agent_type
        self.unique_id = unique_id
        self.training = training
        self.epsilon = epsilon
        self.min_exploration_prob = 0.01
        self.expl_decay = 0.001
        self.total_episode_reward = 0
        self.actions = actions
        self.n_actions = len(actions)
        self.shared_replay_buffer = shared_replay_buffer
        self.learn_step = 0
        self.replace_target_iter = 50
        self._init_networks(checkpoint_path)

    def choose_action(self, observation):
        return self.q_network.choose_action(observation, self.epsilon)
    
    def learn(self, observation, action, reward, next_state, done, episode):
        #vectorised rewards
        experience = {"s":observation, "a":action, "r":reward, "s_":next_state, "done":done}
        self.q_network.add_experience(experience)
        loss = self.q_network.train(self.target_network)
        self._append_losses(loss)
        self.learn_step += 1
        if self.learn_step % self.replace_target_iter == 0:
            self.target_network.copy_weights(self.q_network)
        self.epsilon = max(self.min_exploration_prob, np.exp(-self.expl_decay*episode))

    def save_models(self):
        """
        Save q and target networks to file
        """
        self.q_network.dqn.save(self.q_checkpoint_path)
        self.target_network.dqn.save(self.target_checkpoint_path)
    
    def get_epsilon(self):
        return self.epsilon
    
    def get_mean_loss(self):
        if len(self.losses) > 1:
            return np.mean(self.losses)
        return 0
    
    def _calculate_n_features(self, model):
        """
        Get number of features in observation (agent's health, days left to live, distance to berry, well-being of other agents in society)
        """
        n_features = 4
        n_features += model.get_num_agents() - 1
        return n_features

    def _init_networks(self, checkpoint_path):
        self.q_checkpoint_path = checkpoint_path+self.agent_type+"/agent_"+str(self.unique_id)+"/q_model_variables.keras"
        self.target_checkpoint_path = checkpoint_path+self.agent_type+"/agent_"+str(self.unique_id)+"/target_model_variables.keras"
        if self.training:
            os.makedirs(os.path.dirname(self.q_checkpoint_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.target_checkpoint_path), exist_ok=True)
        self.q_network = DQN(self.actions,(self.n_features,),self.training,checkpoint_path=self.q_checkpoint_path,shared_replay_buffer=self.shared_replay_buffer)
        self.target_network = DQN(self.actions,(self.n_features,),self.training,checkpoint_path=self.target_checkpoint_path,shared_replay_buffer=self.shared_replay_buffer)
        if self.training:
            inputs = np.zeros(self.n_features)
            self.q_network.dqn(np.atleast_2d(inputs.astype('float32')))
            self.target_network.dqn(np.atleast_2d(inputs.astype('float32')))
            self.losses = list()
    
    def _append_losses(self, loss):
        if isinstance(loss, int):
            self.losses.append(loss)
        else:
            self.losses.append(loss.numpy())
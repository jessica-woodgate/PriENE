from ..dqn.dqn import DQN
from collections import Counter
import numpy as np
import os

class MPDQNDecisionModule():
    def __init__(self,agent_type,training,actions,n_features,checkpoint_path,epsilon,shared_replay_buffer=None):
        self.n_features = n_features
        self.agent_type = agent_type
        self.training = training
        self.actions = actions
        self.n_actions = len(self.actions)
        self.objectives = ["baseline","egalitarian","maximin","utilitarian"]
        self.n_rewards = len(self.objectives)
        self.epsilon = epsilon
        self.min_exploration_prob = 0.01
        self.expl_decay = 0.001
        self.shared_replay_buffer = shared_replay_buffer
        self.replace_target_iter = 50
        self.training = training
        self._init_networks(checkpoint_path)
    
    def choose_action(self, observation):
        best_action, selected_actions = self._apply_utility(observation)
        return best_action

    def save_models(self):
        """
        Save q and target networks to file
        """
        for objective in self.networks:
            objective["q_network"].dqn.save(objective["q_checkpoint_path"])
            objective["target_network"].dqn.save(objective["target_checkpoint_path"])

    def _init_networks(self, checkpoint_path):
        #need to init a network for each objective
        self.q_checkpoint_paths = []
        self.target_checkpoint_paths = []
        self.networks = {}
        for objective in self.objectives:
            q_checkpoint_path = checkpoint_path+objective+"/agent_"+str(self.unique_id)+"/q_model_variables.keras"
            self.q_checkpoint_paths.append(q_checkpoint_path)
            target_checkpoint_path = checkpoint_path+objective+"/agent_"+str(self.unique_id)+"/target_model_variables.keras"
            self.target_checkpoint_paths.append(target_checkpoint_path)
            # if self.training:
            #     os.makedirs(os.path.dirname(q_checkpoint_path), exist_ok=True)
            #     os.makedirs(os.path.dirname(target_checkpoint_path), exist_ok=True)
            q_network = DQN(self.actions,(self.n_features,),self.training,checkpoint_path=q_checkpoint_path,shared_replay_buffer=self.shared_replay_buffer)
            target_network = DQN(self.actions,(self.n_features,),self.training,checkpoint_path=target_checkpoint_path,shared_replay_buffer=self.shared_replay_buffer)
            # if self.training:
            #     inputs = np.zeros(self.n_features)
            #     q_network.dqn(np.atleast_2d(inputs.astype('float32')))
            #     target_network.dqn(np.atleast_2d(inputs.astype('float32')))
            #     losses = list()
            # else:
            #     losses = None
            self.networks[objective] = {"q_network": q_network,
                                        "q_checkpoint_path": q_checkpoint_path,
                                        "target_network": target_network,
                                        "target_checkpoint_path": target_checkpoint_path,
                                        #"losses": losses,
                                        "learn_step": 0,
                                        "epsilon": self.epsilon}
    
    def _apply_utility(self, observation):
        actions = {}
        for objective, value in self.networks.items():
            #have to think about what to do if training: if your action is selected, then you can learn from the experience
            #can't learn from the experience if your action isn't selected
            actions[objective] = value["q_network"].choose_action(observation,value["epsilon"])
        #if even split between choices, select random best choice
        most_common = Counter(actions.values()).most_common()
        most_common_values = [value for value, count in most_common if count == most_common[0][1]]
        most_common = np.random.choice(most_common_values)
        return most_common, actions

    # def _learn_networks(self, observation, best_action, selected_actions, reward, next_state, done):
    #     for objective in selected_actions.keys():
    #         if selected_actions[objective] == best_action:
    #             self._learn(self.networks[objective], observation, best_action, reward, next_state, done)
        
    # def _learn(self, network, observation, action, reward, next_state, done):
    #     #vectorised rewards
    #     experience = {"s":observation, "a":action, "r":reward, "s_":next_state, "done":done}
    #     network["q_network"].add_experience(experience)
    #     loss = network["q_network"].train(network["target_network"])
    #     self._append_losses(loss, network["losses"])
    #     network["learn_step"] += 1
    #     if self.learn_step % self.replace_target_iter == 0:
    #         network["target_network"].copy_weights(network["q_network"])
    #     network["epsilon"] = max(self.min_exploration_prob, np.exp(-self.expl_decay*self.model.episode))
    
    # def _append_losses(self, loss, losses):
    #     if isinstance(loss, int):
    #         losses.append(loss)
    #     else:
    #         losses.append(loss.numpy())
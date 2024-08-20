import numpy as np
from .n_network import NNetwork
import tensorflow as tf
import keras
from keras import losses

class DQN:
    """
    DQN handles network training, choosing actions, prediction
    Instance variables:
        gamma -- discount factor of future rewards
        lr -- learning rate
        total_episode_reward -- total reward received in current episode
        batch_size -- size of batch of experiences for training
        hidden_units -- number of hidden units in networks
        n_features -- number of input features for network (length of observation)
        actions -- possible actions
        n_actions -- number of possible actions (output of network)
        training -- boolean for training (learning) or testing (no learning)
        checkpoint_path -- name of path to save and load model
        experience -- buffer for experience history; shared or individual
        min_experiences -- minimum number of experiences before learning can happen
        max_experiences -- maximum size of experience replay buffer
        optimiser -- learning optimiser
        delta -- parameter for Huber loss
    """
    def __init__(self,actions,n_features,training,checkpoint_path=None,shared_replay_buffer=None):
        self.gamma = 0.95
        self.lr = 0.0001
        self.total_episode_reward = 0
        self.batch_size = 64
        self.hidden_units = 128
        self.n_features = n_features
        self.actions = actions
        self.n_actions = len(actions)
        self.training = training
        self.checkpoint_path = checkpoint_path
        if shared_replay_buffer == None:
            self.experience = {"s": [], "a": [], "r": [], "s_": [], "done": []} #experience replay buffer
        else:
            self.experience = shared_replay_buffer
        self.min_experiences = 100
        self.max_experiences = 100000
        self.optimiser = keras.optimizers.Adam(learning_rate=self.lr)
        self.delta = 1.0
        
        if self.training:
            self.dqn = NNetwork(self.n_features,self.hidden_units, self.n_actions)
        else:
            self.dqn = keras.models.load_model(self.checkpoint_path,compile=True)
    
    def train(self, TargetNet):
        """
        Train takes a batch of random experiences, predicts Q values for them using target network, and computes loss
        """
        if len(self.experience['s']) < self.min_experiences:
            return 0
        #get a batch of experiences
        ids = np.random.randint(low=0, high=len(self.experience["s"]), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s_'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        #predict q value using target net
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        #where done, actual value is reward; if not done, actual value is discounted rewards
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        #gradient tape uses automatic differentiation to compute gradients of loss and records operations for back prop
        with tf.GradientTape() as tape:
            #one hot to select the action which was chosen; find predicted q value; reduce to tensor of the batch size
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.n_actions), axis=1) #mask logits through one hot
            huber = losses.Huber(self.delta)
            loss = huber(actual_values, selected_action_values)
        #trainable variables are automatically watched
        variables = self.dqn.trainable_variables
        #compute gradients w.r.t. loss
        gradients = tape.gradient(loss, variables)
        self.optimiser.apply_gradients(grads_and_vars=zip(gradients, variables))
        return loss

    def choose_action(self, observation, epsilon):
        """
        Choose an action randomly or using network with e-greedy probability
        """
        if np.random.uniform(0,1) < epsilon:
            a = np.random.choice(self.actions)
            action = self.actions.index(a)
        else:
            action_values = self.predict(np.atleast_2d(observation))
            action = np.argmax(action_values)
        return action
    
    def predict(self, inputs):
        """
        Predict runs forward pass of network and returns logits (non-normalised predictions) for actions
        Keras model by default recognises input as batch so want to have at least 2 dimensions even if a single state
        """
        actions = self.dqn(np.atleast_2d(inputs.astype('float32')))
        return actions
    
    def add_experience(self, experience):
        """
        Add experience to experience replay buffer
        """
        #check we haven't exceeded size of replay buffer
        if len(self.experience["s"]) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        #add experience to replay buffer
        for key, value in experience.items():
            self.experience[key].append(value)
    
    def copy_weights(self, QNet):
        """
        Copy weights of q net to target net every n steps
        """
        variables1 = self.dqn.trainable_variables
        variables2 = QNet.dqn.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
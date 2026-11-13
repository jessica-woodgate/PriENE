from .n_network import NNetwork
from src.harvest_exception import CheckpointNotFoundException
from src.harvest_exception import CheckpointMismatchException
from src.checkpoints import recorded_agent_types
import numpy as np
import os
import tensorflow as tf
import keras
from keras import losses

class DQN:
    """
    DQN handles network training, choosing actions, prediction
    Instance variables:
        gamma -- discount factor of future rewards
        lr -- learning rate
        batch_size -- size of batch of experiences for training
        hidden_units -- number of hidden units in networks
        n_features -- number of input features for network (length of observation)
        actions -- possible actions
        n_actions -- number of possible actions (output of network)
        training -- boolean for training (learning) or testing (no learning)
        checkpoint_path -- name of path to save and load model; a new network is created to train, unless continuing a training run
        experience -- buffer for experience history; shared or individual
        min_experiences -- minimum number of experiences before learning can happen
        max_experiences -- maximum size of experience replay buffer
        optimiser -- learning optimiser
        delta -- parameter for Huber loss
        action_mask -- boolean per action, False for actions never to be chosen (not explored, not chosen greedily, and left out of the
            best next action in learning targets), or None to allow every action; the network keeps an output for every action either way
        compiled -- whether train runs as one compiled TensorFlow graph (about 13 times faster; the same algorithm and the same random draws,
            but floating point results can differ in the last bits from the eager version, so the harvest scenarios keep the eager one)
    """
    def __init__(self,actions,n_features,training,checkpoint_path=None,shared_replay_buffer=None,continuing=False,action_mask=None,compiled=False):
        self.action_mask = None if action_mask is None else np.asarray(action_mask, dtype=bool)
        self.compiled = compiled
        self._compiled_steps = {}
        self.gamma = 0.95
        self.lr = 0.0001
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
        if self.training and not continuing:
            self.dqn = NNetwork(self.n_actions,self.n_features,self.hidden_units)
        else:
            self.dqn = self._load(self.checkpoint_path)

    def _load(self, checkpoint_path):
        """
        Load a saved network, checking it has this agent's input and output sizes (they depend only on the number of agents)
        """
        if not os.path.exists(checkpoint_path):
            #<society folder>/<agent type>/agent_<id>/<network>.keras; agents trained together in a mixed society record their types there
            society_path = os.path.dirname(os.path.dirname(os.path.dirname(checkpoint_path)))
            raise CheckpointNotFoundException(checkpoint_path, recorded_agent_types(society_path+"/"))
        dqn = keras.models.load_model(checkpoint_path,compile=True, custom_objects={"NNetwork": NNetwork})
        if tuple(dqn.n_features) != tuple(self.n_features) or dqn.n_actions != self.n_actions:
            raise CheckpointMismatchException(checkpoint_path, self.n_features[0], self.n_actions, tuple(dqn.n_features)[0], dqn.n_actions)
        return dqn
    
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
        if self.compiled:
            return self._compiled_step(TargetNet)(states.astype("float32"), actions, rewards.astype("float32"), states_next.astype("float32"), dones)
        #predict q value using target net
        if self.action_mask is None:
            value_next = np.max(TargetNet.predict(states_next), axis=1)
        else:
            value_next = np.max(self._masked(TargetNet.predict(states_next)), axis=1)
        #where done, actual value is reward; if not done, actual value is discounted rewards
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        #gradient tape uses automatic differentiation to compute gradients of loss and records operations for back prop
        with tf.GradientTape() as tape:
            #one hot to select the action which was chosen; find predicted q value; reduce to tensor of the batch size
            one_hot = tf.one_hot(actions, self.n_actions)
            selected_action_values = tf.math.reduce_sum(self.predict(states) * one_hot, axis=1) #mask logits through one hot
            huber = losses.Huber(self.delta)
            loss = huber(actual_values, selected_action_values)
        #trainable variables are automatically watched
        variables = self.dqn.trainable_variables
        #compute gradients w.r.t. loss
        gradients = tape.gradient(loss, variables)
        self.optimiser.apply_gradients(grads_and_vars=zip(gradients, variables))
        return loss

    def _compiled_step(self, TargetNet):
        """
        train's learning step from a sampled batch, as one compiled graph: targets from TargetNet (the best allowed next action), Huber
        loss on the chosen actions' Q values, one optimiser step. Built once per target network
        """
        if id(TargetNet) not in self._compiled_steps:
            variables = self.dqn.trainable_variables
            #the optimiser's own variables must exist before the graph is traced
            self.optimiser.build(variables)
            huber = losses.Huber(self.delta)
            mask = None if self.action_mask is None else tf.constant(self.action_mask)
            def step(states, actions, rewards, states_next, dones):
                next_values = TargetNet.dqn(states_next)
                if mask is not None:
                    next_values = tf.where(mask, next_values, tf.fill(tf.shape(next_values), -np.inf))
                actual_values = tf.where(dones, rewards, rewards + self.gamma * tf.reduce_max(next_values, axis=1))
                with tf.GradientTape() as tape:
                    selected_action_values = tf.math.reduce_sum(self.dqn(states) * tf.one_hot(actions, self.n_actions), axis=1)
                    loss = huber(actual_values, selected_action_values)
                self.optimiser.apply_gradients(zip(tape.gradient(loss, variables), variables))
                return loss
            self._compiled_steps[id(TargetNet)] = tf.function(step, reduce_retracing=True)
        return self._compiled_steps[id(TargetNet)]

    def choose_action(self, observation, epsilon):
        """
        Choose an action randomly or using network with e-greedy probability (only among allowed actions, if there is an action mask)
        """
        if self.action_mask is not None:
            return self._choose_allowed_action(observation, epsilon)
        if np.random.uniform(0,1) < epsilon:
            a = np.random.choice(self.actions)
            action = self.actions.index(a)
        else:
            action_values = self.predict(np.atleast_2d(observation))
            action = np.argmax(action_values)
        return action

    def _choose_allowed_action(self, observation, epsilon):
        if np.random.uniform(0,1) < epsilon:
            return int(np.random.choice(np.flatnonzero(self.action_mask)))
        return int(np.argmax(self._masked(self.predict(np.atleast_2d(observation)))))

    def _masked(self, action_values):
        """
        Q values with every masked action's set to -inf, so it is never the best
        """
        values = np.array(action_values, dtype=float)
        values[..., ~self.action_mask] = -np.inf
        return values
    
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
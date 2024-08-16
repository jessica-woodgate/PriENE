import tensorflow
from tensorflow import keras
from keras import layers
from keras import initializers as initialiser

class NNetwork(keras.Model):
    """
    NNetwork handles the network
    Instance variables:
        n_features -- number of features for input (size of observation)
        hidden_units -- number of hidden units
        n_actions -- number of possible actions (size of output)
        input_layer -- input layer of network
        hidden_layer_1 and hidden_layer 2 -- hidden layers
        output_layer -- output layer of network
        model -- the sequential model
    """
    def __init__(self, n_features, hidden_units, n_actions, trainable=True, dtype="float32", **kwargs):
        super().__init__()
        self.n_features = n_features
        self.hidden_units = hidden_units
        self.n_actions = n_actions
        self.input_layer = layers.InputLayer(shape=n_features,)
        self.hidden_layer_1 = layers.Dense(hidden_units, activation="relu", kernel_initializer=initialiser.HeNormal())
        self.hidden_layer_2 = layers.Dense(hidden_units, activation="relu", kernel_initializer=initialiser.HeNormal())
        self.output_layer = layers.Dense(n_actions, activation="linear", kernel_initializer=initialiser.HeNormal())
        
        self.model = keras.Sequential([
            self.input_layer,
            self.hidden_layer_1,
            self.hidden_layer_2,
            self.output_layer
        ])
        
    def call(self, inputs):
        z = self.model(inputs)
        return z
    
    def get_config(self):
        """
        when saving a model that includes custom objects, you must call a get_config() method on the object class
        https://keras.io/guides/serialization_and_saving/
        """
        config = super().get_config()
        #Update the config with the custom model's parameters
        config.update(
            {
                "n_features": self.n_features,
                "hidden_units": self.hidden_units,
                "n_actions": self.n_actions,
            }
        )
        return config
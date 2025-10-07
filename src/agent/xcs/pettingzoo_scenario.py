from .scenarios import Scenario
from .bitstrings import BitString

class PettingZooScenario(Scenario):

    def __init__(self, agent_id, env, input_size, possible_actions):
        """
        In XCS a scenario is a series of situations (inputs to the classifier) that the algorithm must respond to with appropriate action to maximise reward
        For PettingZoo env input_size of an agent is obs_dims
        possible_actions is env.action_space(agent_id)
        """
        self.agent_id = agent_id
        self.input_size = input_size
        self.possible_actions = possible_actions
        self.env = env
    
    @property
    def is_dynamic(self):
        """
        Later training cycles are affected by earlier actions
        """
        return True
    
    def get_possible_actions(self):
        return self.possible_actions
    
    def reset(self):
        pass

    def sense(self):
        """
        Returns a new input (current situation)
        """
        obs = self.env.observe(self.agent_id)
        obs_bitstring = BitString(obs)
        return obs_bitstring
    
    def execute(self, action):
        if action not in self.possible_actions:
            raise ValueError(f"Unrecognised action{action}")
        self.env.step(action)
        return self.env.rewards[self.agent_id]
    
    def more(self):
        return self.env.terminations[self.agent_id] or self.env.truncations[self.agent_id]
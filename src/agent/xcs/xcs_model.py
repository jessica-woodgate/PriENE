import numpy as np
import logging
import time

from policies.xcs.scenarios import Scenario, ScenarioObserver
from policies.xcs.framework import LCSAlgorithm, ClassifierSet
from policies.xcs.algorithms.xcs import XCSAlgorithm
from policies.xcs.pettingzoo_scenario import PettingZooScenario

from pettingzoo.mpe import simple_spread_v3
from gymnasium.spaces import Discrete

from rl_models.RL_model import RLModel
from data_handling.results_tracking import ResultsTracking

class XCSModel(RLModel):
    def __init__(self, training, write_data, run_name, max_episodes=50000):
        self.env = simple_spread_v3.env()
        self.agents, self.scenarios = self._init_agents()
        self.run_name = run_name
        self.results_tracking = ResultsTracking(training, write_data, run_name)
        self.max_episodes = max_episodes
        self.episode_length = 25
        self.epsilon = 0.99
        self.min_epsilon = 0.1
        self.epsilon_decay = 0.0001
    
    def run_episodes(self):
        start_time = time.time()
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        epsiode = 0
        while (self.training and self.epsilon > self.min_epsilon) or (not self.training and episode <= self.max_episodes):
        #for episode in range(self.max_episodes):
            self.env.reset(seed=42)
            # record reward of each agent in this episode
            episode_reward = np.zeros((self.episode_length, len(self.env.possible_agents)))
            previous_match_sets = None
            for step in range(self.episode_length):  # interact with the env for an episode
                previous_match_sets, episode_reward[step] = self._step_env(step, previous_match_sets, self.episode_length, episode)
            #episode finishes
            self.epsilon = min(agent.exploration_probability for agent in self.agents)
            self.results_tracking.store_episode_data(episode, episode_reward)
            episode += 1
        end_time = time.time()
        for i, agent in enumerate(self.agents):
            logger.info('Classifiers:\n\n%s\n', agent)
            logger.info("Total time: %.5f seconds", end_time - start_time)
            with open(f"data/results/current_run/{self.run_name}_agent_{i}_classifiers.txt", "w") as f:
                f.write(str(agent))

    def _step_env(self, step, previous_match_sets, episode_length, episode):
        last_step = step == episode_length - 1
        previous_match_sets = self._step_agents(previous_match_sets, episode, last_step)
        #for xcs model only need rewards for tracking learning results; pettingzoo_scenario handles environment interaction
        rewards_list = self._get_transition_info(self.env, just_rewards=True)
        return previous_match_sets, rewards_list
    
    def _step_agents(self, previous_match_sets, episode, last_step=False):
        current_match_sets = []
        for i, agent in enumerate(self.agents):
            previous_match_set = None if previous_match_sets is None else previous_match_sets[i]
            match_set = agent.run_step(self.scenarios[i], previous_match_set, last_step, learn=True)
            current_match_sets.append(match_set)
            agent.exploration_probability = max(self.min_epsilon, np.exp(-self.epsilon_decay*episode))
        return current_match_sets
            
    def _init_agents(self):
        """
        Returns a list of xcs agents with LCS algorithm
        Returns a list of the pettingzoo scenario which performs necessary functionality to interact with xcs
        """
        agents = []
        scenarios = []
        for agent_id in self.env.possible_agents:
            input_size = self.env.observation_space(agent_id).shape[0]
            action_space = self.env.action_space(agent_id)
            if isinstance(action_space, Discrete):
                possible_actions = list(range(action_space.n))
            else:
                raise ValueError(f"Unknown action type {action_space}")
            petting_zoo_scenario = PettingZooScenario(agent_id, self.env, input_size, possible_actions)
            xcs_agent = self._setup_xcs_agent(scenario=ScenarioObserver(petting_zoo_scenario))
            agents.append(xcs_agent)
            scenarios.append(petting_zoo_scenario)
        return agents, scenarios
    
    def _setup_xcs_agent(self, algorithm=None, scenario=None):
        assert algorithm is None or isinstance(algorithm, LCSAlgorithm)
        assert scenario is None or isinstance(scenario, Scenario)

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        if not isinstance(scenario, ScenarioObserver):
            #wrapper that will report things back for visibility.
            scenario = ScenarioObserver(scenario)

        if algorithm is None:
            # Define the algorithm.
            algorithm = XCSAlgorithm()
            algorithm.exploration_probability = .99
            algorithm.do_ga_subsumption = True
            algorithm.do_action_set_subsumption = True

        assert isinstance(algorithm, LCSAlgorithm)
        assert isinstance(scenario, ScenarioObserver)

        # Create the classifier system from the algorithm.
        agent = ClassifierSet(algorithm, scenario.get_possible_actions())
        return agent
        
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import pandas as pd
import numpy as np
import json
import os
from .agent.harvest_agent import HarvestAgent
from .berry import Berry
from .harvest_exception import FileExistsException
from .harvest_exception import NoEmptyCells
from .harvest_exception import NumAgentsException
from .harvest_exception import AgentTypeException
from .harvest_exception import NoBerriesException
from .harvest_exception import NumBerriesException
from os.path import exists
from abc import abstractmethod

class HarvestModel(Model):
    """
    Harvest model handles the environment, stepping agents, and data collection
    Instance variables:
        num_agents -- number of agents at beginning of each episode
        num_berries -- number of berries at beginning of each episode
        end_day -- last day of episode
        day -- current day
        max_days -- max days in a single episode
        max_episode -- maximum number of episodes (for testing; training runs until epsilon is min epsilon)
        min_epsilon -- minimum epsilon to end training
        schedule -- schedule of agents
        max_with -- width of grid
        max_height -- height of grid
        grid -- grid object
        shared_replay_buffer -- replay buffer to share amongst agents to reduce training time
        agent_id -- tracker for unique agent ids
        berry_id -- tracker for unique berry ids
        episode -- current episode
        training -- boolean training or testing
        filepath -- file path for current run
        write_data -- boolean to write data to file
        write_norms -- boolean to track norms and write to file
        societal_norm_emergence_threshold -- percentage of society required to have adopted a behaviour for it to become a norm
        emerged_norms -- all norms which emerge in current episode
        min_fitness -- minimum fitness required for a behaviour to become a norm
        epsilon -- probability of exploration for agents (tracks when to end training)
    """
    def __init__(self,num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath=""):
        super().__init__()
        self.num_agents = num_agents
        if self.num_agents <= 0:
            raise NumAgentsException(">0", 0)
        self.num_berries = 0
        self.end_day = 0
        self.day = 1
        self.max_days = max_days
        self.max_episodes = max_episodes
        self.min_epsilon = 0.01
        self.schedule = RandomActivation(self)
        self.max_width = max_width
        self.max_height = max_height
        self.grid = MultiGrid(self.max_width, self.max_height, False)
        self.shared_replay_buffer = {"s": [], "a": [], "r": [], "s_": [], "done": []}
        self.agent_id = 0
        self.berry_id = 0
        self.episode = 1
        self.training = training
        self.filepath = filepath
        self.write_data = write_data
        self.write_norms = write_norms
        self.societal_norm_emergence_threshold = 0.9
        self.emerged_norms = {}
        #self.min_fitness = 0.0
        if self.training:
            self.epsilon = 0.9
        else:
            self.epsilon = 0.0
        self._init_reporters()
            
    def step(self):
        """
        Steps the schedule of agents, updates data collection, handles dead agents and foraged berries
        """
        self.schedule.step()
        self.day += 1
        self._update_schedule()
        self.epsilon = self._mean_epsilon()
        if self.write_norms:
            self._check_emerged_norms()
        #if exceeded max days or all agents died, reset for new episode
        if self.day >= self.max_days or len(self.living_agents) <= 0:
            self.finish_episode()
    
    def finish_episode(self, collect_data=True):
        self.end_day = self.day
        if self.write_norms and collect_data:
            self._append_norm_dict_to_file(self.emerged_norms, "data/results/current_run/"+self.filepath+"_emerged_norms.json")
        for a in self.schedule.agents:
            if a.agent_type != "berry":
                a.finish_episode(self.day)
        if collect_data:
            self._collect_model_episode_data()
        self._reset()

    def move_agent_to_cell(self, agent, new_pos):
        """
        Move an agent to a specified cell
        """
        self.grid.move_agent(agent, new_pos)
    
    def get_cell_contents(self, cell):
        """
        Get the coordinates of a specified cell
        """
        return self.grid.iter_cell_list_contents(cell)

    def get_uneaten_berries_coordinates(self, allocation_id=None):
        """
        Get the coordinates of uneaten berries
        """
        berries_coordinates = []
        for b in self.berries:
            if b.foraged == False:
                if allocation_id==None:
                    berries_coordinates.append(b.pos)
                else:
                    if b.allocation_id == allocation_id:
                        berries_coordinates.append(b.pos)
        return berries_coordinates
    
    def get_uneaten_berry_by_coords(self, coords, allocation_id=None):
        """
        Get an uneaten berry by its coordinates
        """
        for b in self.berries:
            if b.pos == coords and b.foraged == False:
                if allocation_id == None:
                    return b
                elif b.allocation_id == allocation_id:
                    return b
        raise NoBerriesException(coordinates=coords)
    
    def get_society_well_being(self, observer, norms_observation, ethics_observation):
        """
        Get the well-being of a society; iterates over all agents in the society, excluding the observer if the agent is observing
        """
        society_well_being = np.array([])
        for a in self.schedule.agents:
            if (not ethics_observation and (a.unique_id == observer.unique_id)) or a.agent_type == "berry":
                continue
            elif a.done == False:
                society_well_being = np.append(society_well_being, a.days_left_to_live)
            elif a.done == True and (norms_observation or ethics_observation):
                continue
            else:
                society_well_being = np.append(society_well_being, 0)
        return society_well_being
    
    def get_num_agents(self):
        return self.num_agents
    
    def get_num_living_agents(self):
        return len(self.living_agents)
    
    def get_living_agents(self):
        return self.living_agents
    
    def get_day(self):
        return self.day
    
    def get_max_days(self):
        return self.max_days
    
    @abstractmethod
    def _init_berries(self):
        raise NotImplementedError
    
    def _init_agents(self, society_mix, agent_type, checkpoint_path):
        self.living_agents = []
        allotment = [0,self.max_width,0,self.max_height]
        if society_mix == "homogeneous":
            for i in range(self.num_agents):
                self._add_agent(i, agent_type, allotment, checkpoint_path)
        else:
            assert self.num_agents%2 == 0
            half_pop = int(self.num_agents/2)
            for i in range(half_pop):
                self._add_agent(i, agent_type, allotment, checkpoint_path)
            for i in range(half_pop):
                self._add_agent(i+half_pop, "baseline", allotment, checkpoint_path)
        assert self.num_agents == len(self.living_agents)
        self.berry_id = len(self.living_agents) + 1
    # def _init_agents(self, scenario, agent_type, checkpoint_path):
    #     self.living_agents = []
    #     allotment = [0,self.max_width,0,self.max_height]
    #     for i in range(self.num_agents):
    #         a = HarvestAgent(i,self,agent_type,allotment,self.training,checkpoint_path,self.epsilon,self.write_norms,shared_replay_buffer=self.shared_replay_buffer)
    #         self._add_agent(a)
    #     self.berry_id = len(self.living_agents) + 1

    def _add_agent(self, i, agent_type, allotment, checkpoint_path, allocation_id=None):
        a = HarvestAgent(i,self,agent_type,allotment,self.training,checkpoint_path,self.epsilon,self.write_norms,shared_replay_buffer=self.shared_replay_buffer,allocation_id=allocation_id)
        self.schedule.add(a)
        self._place_agent_in_allotment(a)
        if a.agent_type != "berry":
            self.agent_id += 1
            self.living_agents.append(a)
    # def _add_agent(self, a):
    #     self.schedule.add(a)
    #     self._place_agent_in_allotment(a)
    #     if a.agent_type != "berry":
    #         self.agent_id += 1
    #         self.living_agents.append(a)

    def _reset(self):
        self.living_agents = []
        self.emerged_norms = {}
        self.day = 0
        num_agents = 0
        num_berries = 0
        self.episode += 1
        self.total_episode_reward = 0
        self._clear_grid()
        for a in self.schedule.agents:
            if a.agent_type != "berry":
                self._reset_agent(a)
                num_agents += 1
            elif a.agent_type == "berry":
                self._reset_berry(a, True)
                num_berries += 1
        if num_agents != self.num_agents:
            raise NumAgentsException(self.num_agents, num_agents)
        if num_berries != self.num_berries:
            raise NumBerriesException(self.num_berries, num_berries)
        self.agent_reporter = pd.DataFrame({"agent_id": [],
                               "agent_type": [],
                               "episode": [],
                               "day": [],
                               "berries": [],
                               "berries_consumed": [],
                               "berries_thrown": [],
                               "health": [],
                               "days_left_to_live": [],
                               "total_days_left_to_live": [],
                               "action": [],
                               "reward": [],
                               "num_norms": []})
    
    def _reset_agent(self, agent):
        if agent.agent_type == "berry":
            raise AgentTypeException(agent.agent_type, "berry")
        agent.reset()
        self._place_agent_in_allotment(agent)
        agent.off_grid = False
        self.living_agents.append(agent)
    
    def _reset_berry(self, berry, end_of_episode):
        if berry.agent_type != "berry":
            raise AgentTypeException("berry", berry.agent_type)
        if not end_of_episode:
            self.grid.remove_agent(berry)
        berry.reset()
        self._place_agent_in_allotment(berry)
        
    def _init_reporters(self):
        os.makedirs("data/results/current_run", exist_ok=True)
        self.agent_reporter = pd.DataFrame({"agent_id": [],
                               "agent_type": [],
                               "episode": [],
                               "day": [],
                               "berries": [],
                               "berries_consumed": [],
                               "berries_thrown": [],
                               "health": [],
                               "days_left_to_live": [],
                               "total_days_left_to_live": [],
                               "action": [],
                               "reward": [],
                               "num_norms": []})
        if self.write_data and not self.training:
           if exists("data/results/current_run/agent_reports_"+self.filepath+".csv"):
               raise FileExistsException("data/results/current_run/agent_reports_"+self.filepath+".csv")
           self.agent_reporter.to_csv("data/results/current_run/agent_reports_"+self.filepath+".csv", mode='a',index=False)
        self.model_episode_reporter = pd.DataFrame({"episode": [], 
                               "end_day": [],
                               "epsilon": [],
                               "mean_reward": [],
                               "mean_loss": [],
                               "max_berries": [],
                               "mean_berries": [],
                               "max_berries_consumed": [],
                               "mean_berries_consumed": [],
                               "gini_berries_consumed": [],
                               "mean_berries_thrown": [],
                               "max_health": [],
                               "mean_health": [],
                               "median_health": [],
                               "variance_health": [],
                               "deceased": [],
                               "num_emerged_norms": []})
        if self.write_data:
            if exists("data/results/current_run/model_episode_reports_"+self.filepath+".csv"):
                raise FileExistsException("data/results/current_run/model_episode_reports_"+self.filepath+".csv")
            self.model_episode_reporter.to_csv("data/results/current_run/model_episode_reports_"+self.filepath+".csv", mode='a',index=False)

    def _collect_agent_data(self, agent):
        new_entry = pd.DataFrame({"agent_id": [agent.unique_id],
                               "agent_type": [agent.agent_type],
                               "episode": [self.episode],
                               "day": [self.day],
                               "berries": [agent.berries],
                               "berries_consumed": [agent.berries_consumed],
                               "berries_thrown": [agent.berries_thrown],
                               "health": [agent.health],
                               "days_left_to_live": [agent.days_left_to_live],
                               "total_days_left_to_live": [agent.total_days_left_to_live],
                               "action": [agent.current_action],
                               "reward": [agent.current_reward],
                               "num_norms": [len(agent.norms_module.behaviour_base) if self.write_norms else None]})
        self.agent_reporter = pd.concat([self.agent_reporter, new_entry])
        if self.write_data and not self.training:
           new_entry.to_csv("data/results/current_run/agent_reports_"+self.filepath+".csv", header=None, mode='a',index=False)
    
    def _collect_model_episode_data(self):
        row_index_list = self.agent_reporter.index[self.agent_reporter["episode"] == self.episode].tolist()
        new_entry = pd.DataFrame({"episode": [self.episode], 
                               "end_day": [self.day],
                               "epsilon": [self.epsilon],
                               "mean_reward": [self._mean_reward()],
                               "mean_loss": [self._mean_loss()],
                               "max_berries": [self.agent_reporter["berries"].loc[row_index_list].max()],
                               "mean_berries": [self.agent_reporter["berries"].iloc[row_index_list].mean(axis=0)],
                               "max_berries_consumed": [self.agent_reporter["berries_consumed"].loc[row_index_list].max()],
                               "mean_berries_consumed": [self.agent_reporter["berries_consumed"].loc[row_index_list].mean(axis=0)],
                               "gini_berries_consumed": [self._gini_berries_consumed()],
                               "mean_berries_thrown": [self.agent_reporter["berries_thrown"].loc[row_index_list].mean(axis=0)],
                               "max_health": [self.agent_reporter["health"].loc[row_index_list].max()],
                               "mean_health": [self.agent_reporter["health"].loc[row_index_list].mean(axis=0)],
                               "median_health": [self.agent_reporter["health"].loc[row_index_list].median()],
                               "variance_health": [self.agent_reporter["health"].loc[row_index_list].var(axis=0)],
                               "deceased": [self.num_agents - len(self.living_agents)],
                               "num_emerged_norms": [len(self.emerged_norms) if self.write_norms else None]})
        if self.write_data:
            new_entry.to_csv("data/results/current_run/model_episode_reports_"+self.filepath+".csv", header=None, mode='a',index=False)
        return new_entry

    def _append_norm_dict_to_file(self, norm_dictionary, filename):
        with open(filename, "a+") as file:
            file.seek(0)
            if not file.read(1):
                file.write("{")
            file.seek(0, 2)
            norm_list = []
            for key, value in norm_dictionary.items():
                dict = {key: value}
                norm_list.append(dict)
            file.write(f"\"{self.episode}\": {json.dumps(norm_list, indent=4)}\n")
            if self.episode != self.max_episodes:
                file.write(",")
            else:
                file.write("}")

    def _check_emerged_norms(self):
        if len(self.living_agents) < 2:
            return
        emergence_count = len(self.living_agents) * self.societal_norm_emergence_threshold
        current_emerged_norms = {}
        for agent in self.schedule.agents:
            if agent.agent_type != "berry":
                for norm_name, norm_value in agent.norms_module.behaviour_base.items():
                    current_emerged_norms = self._update_norm(norm_name, norm_value, current_emerged_norms)
        #current_emerged_norms = {norm: norm_value for norm, norm_value in current_emerged_norms.items() if norm_value["adoption"] >= emergence_count and norm_value["fitness"] >= self.min_fitness}
        current_emerged_norms = {norm: norm_value for norm, norm_value in current_emerged_norms.items() if norm_value["adoption"] >= emergence_count}
        for norm_name, norm_value in current_emerged_norms.items():
            self.emerged_norms = self._update_norm(norm_name, norm_value, self.emerged_norms)
    
    def _update_norm(self, norm_name, norm_value, norm_base):
        if norm_name not in norm_base.keys():
            norm_base[norm_name] = {"reward": 0,
                                        "numerosity": 0,
                                        "fitness": 0,
                                        "adoption": 0}
        norm_base[norm_name]["reward"] += norm_value["reward"]
        norm_base[norm_name]["numerosity"] += norm_value["numerosity"]
        norm_base[norm_name]["fitness"] += norm_value["fitness"]
        norm_base[norm_name]["adoption"] += 1
        return norm_base
    
    def _update_schedule(self):
        #check for dead agents & foraged berries
        for a in self.schedule.agents:
            if a.agent_type == "berry" and a.foraged == True:
                self._reset_berry(a, False)
            if a.agent_type != "berry":
                if a.off_grid == False:
                    self._collect_agent_data(a)
                    if a.done == True:
                        self._remove_agent(a)
    
    def _check_bounds(self, cell):
        if cell[0] >= 0 and cell[0] < self.max_width:
            if cell[1] >= 0 and cell[1] < self.max_height:
                return True
        return False

    def _place_agent_in_allotment(self, agent):    
        #for new agents who aren't yet on the grid
        if not self.grid.exists_empty_cells:
            raise NoEmptyCells
        cell = self._random_allotment_cell(agent)
        self.grid.place_agent(agent, cell)

    def _move_agent_in_allotment(self, agent, cell=None):
        #for agents who are on the grid
        if not self.grid.exists_empty_cells:
            raise NoEmptyCells
        if cell == None:
            cell = self._random_allotment_cell(agent)
        self.grid.move_agent(agent, cell)
    
    def _clear_grid(self):
        for a in self.schedule.agents:
            if a.agent_type != "berry" and a.off_grid:
                continue
            self.grid.remove_agent(a)
    
    def _remove_agent(self, agent):
        self.grid.remove_agent(agent)
        agent.off_grid = True
        agent.days_left_to_live = 0
        self.living_agents = [a for a in self.schedule.agents if a.agent_type != "berry" and a.off_grid == False]
    
    def _new_berry(self,allotment,allocation_id=None):
        if allocation_id != None:
            allocation_id = "allocation_"+str(allocation_id)
        berry = Berry(self.berry_id,self,allotment,allocation_id)
        self.schedule.add(berry)
        self.berry_id += 1
        return berry
    
    def _random_allotment_cell(self, agent):
        width = np.random.randint(agent.min_width, agent.max_width)
        height = np.random.randint(agent.min_height, agent.max_height)
        return (width, height)
    
    def _generate_resource_allocations(self, num_agents):
        if num_agents == 2:
            resources = [5, 1]
        elif num_agents == 4:
            resources = [5, 3, 2, 2]
        elif num_agents == 6:
            resources = [5, 2, 3, 2, 5, 1]
        elif num_agents == 20:
            resources = [5, 5, 3, 5, 5, 3, 5, 2, 3, 2, 5, 1, 5, 2, 1, 2, 2, 1, 2, 2]
        else:
            resources = self._generate_zipf_distribution(num_agents)
        self.num_start_berries = sum(resources)
        return resources
    
    def _generate_zipf_distribution(self, num_agents):
        total_resources = self.num_start_berries

        # Generate Zipf-like distribution
        weights = 1 / np.arange(1, num_agents + 1)  # [1, 1/2, 1/3, ...]
        weights /= weights.sum()
        allocations = (weights * total_resources).astype(int)

        # Adjust rounding error
        diff = total_resources - allocations.sum()
        for _ in range(abs(diff)):
            i = np.random.randint(num_agents)
            allocations[i] += np.sign(diff)
            
        np.random.shuffle(allocations)  # Shuffle to avoid order bias
        self.num_start_berries = int(allocations.sum())
        return allocations.tolist()

    def _gini_berries_consumed(self):
        if len(self.living_agents) == 0:
            return 0
        berries_consumed = [a.berries_consumed for a in self.schedule.agents if a.agent_type != "berry"]
        x = sorted(berries_consumed)
        s = sum(x)
        if s == 0:
            return 0
        N = self.num_agents
        B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * s)
        return 1 + (1 / N) - 2 * B
    
    def _mean_loss(self):
        m = 0
        if self.training:
            for agent in self.schedule.agents:
                if agent.agent_type != "berry":
                    m += agent.get_mean_loss()
            if m == 0:
                return 0
            m /= self.num_agents
        return m
    
    def _mean_reward(self):
        m = 0
        for agent in self.schedule.agents:
            if agent.agent_type != "berry":
                m += agent.total_episode_reward
        if m == 0:
            return 0
        m /= self.num_agents
        return m
    
    def _mean_epsilon(self):
        m = 0
        for agent in self.schedule.agents:
            if agent.agent_type != "berry":
                m += agent.get_epsilon()
        if m == 0:
            return 0
        m /= self.num_agents
        return m
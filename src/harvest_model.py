from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
import pandas as pd
import numpy as np
import json
import os
import warnings
import tensorflow
from tensorflow import keras
from .agent.harvest_agent import HarvestAgent
from .agent.harvest_agent import NORMS_MODULE_TYPES
from .berry import Berry
from .harvest_exception import FileExistsException
from .harvest_exception import NoEmptyCells
from .harvest_exception import NumAgentsException
from .harvest_exception import AgentTypeException
from .harvest_exception import NoBerriesException
from .harvest_exception import NumBerriesException
from .harvest_exception import UnrecognisedNormsModule
from .harvest_exception import InitialHealthsException
from .harvest_exception import SeasonsException
from .harvest_exception import BerryOffGrid
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
        track_norms -- boolean to track norms and write to file
        norms_module_type -- norm base agents track norms with: "rule_based" (default; tracks societally
            emerged norms, writes <filepath>_emerged_norms.json) or "bayesian" (writes agent_norms_<filepath>.csv)
        previous_learned_norms -- dict mapping agent unique_id -> set of behaviour ids learned at the end of the previous episode
        societal_norm_emergence_threshold -- percentage of society required to have adopted a behaviour for it to become a norm
        emerged_norms -- all norms which emerge in current episode
        epsilon -- probability of exploration for agents (tracks when to end training)
        seed -- random seed for the run (None = unseeded)
        initial_healths -- health each agent starts every episode with, indexed by agent id (None = every agent starts with the agent's default start health)
        seasons -- boolean to randomly reassign the berry allocations between holders (groups: capability groups, allotments, or in colours single agents) every season_length days
        season_length -- days in each season; swaps happen after the steps on days season_length, 2*season_length, ...
        season_swaps -- whether swaps can change anything; False with seasons on if there is one holder or every holder has the same allocation, and swaps are skipped
        season -- index of the current season in this episode (0 = the original assignment)
        season_assignment -- dict mapping each holder (an allocation key) to the original allocation whose berries it holds this season (None for scenarios without allocations)
        season_rng -- generator for season swaps, seeded from seed and separate from the other random streams so seasons don't shift them
    """
    def __init__(self,num_agents,max_width,max_height,max_episodes,max_days,training,write_data,track_norms,filepath="",seed=None,norms_module_type="rule_based",initial_healths=None,seasons=False,season_length=25):
        super().__init__()
        self._set_seed(seed)
        self.num_agents = num_agents
        if self.num_agents <= 0:
            raise NumAgentsException(">0", 0)
        if initial_healths is not None and (len(initial_healths) != num_agents or any(health <= 0 for health in initial_healths)):
            raise InitialHealthsException(num_agents, initial_healths)
        self.initial_healths = None if initial_healths is None else list(initial_healths)
        self.num_berries = 0
        self.end_day = 0
        self.day = 0
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
        self.track_norms = track_norms
        if norms_module_type not in NORMS_MODULE_TYPES:
            raise UnrecognisedNormsModule(norms_module_type, NORMS_MODULE_TYPES)
        self.norms_module_type = norms_module_type
        self.societal_norm_emergence_threshold = 0.9
        self.emerged_norms = {}
        self.previous_learned_norms = {}
        if self.training:
            self.epsilon = 0.9
        else:
            self.epsilon = 0.0
        if seasons and (not isinstance(season_length, int) or season_length < 1):
            raise SeasonsException(f"season length must be a whole number of days of at least 1, got {season_length}")
        self.seasons = seasons
        self.season_length = season_length
        self.season = 0
        self.season_assignment = None
        self.season_swaps = False
        self.season_rng = np.random.default_rng(seed) if seasons else None
        self._init_reporters()

    def step(self):
        """
        Steps the schedule of agents, updates data collection, handles dead agents and foraged berries
        """
        self.schedule.step()
        self.day += 1
        self._update_schedule()
        self.epsilon = self._mean_epsilon()
        if self.track_norms and self.norms_module_type == "rule_based":
            self._check_emerged_norms()
        #if exceeded max days or all agents died, reset for new episode
        if self.day >= self.max_days or len(self.living_agents) <= 0:
            self.finish_episode()
        elif self.season_swaps and self.day % self.season_length == 0:
            self._start_new_season()

    def finish_episode(self, collect_data=True):
        self.end_day = self.day
        if self.track_norms and collect_data:
            # record norms before agents' finish_episode(), so any end-of-episode norm update (e.g. decay) is excluded
            if self.norms_module_type == "rule_based":
                self._append_norm_dict_to_file(self.emerged_norms, "data/results/current_run/"+self.filepath+"_emerged_norms.json")
            else:
                self._analyse_agent_norms()
                #self._write_agent_norms_to_file()
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
    
    def _set_seed(self, seed):
        """
        Seeds every source of randomness so a run can be reproduced: python's random module, numpy's
        global RNG (grid placement, resource allocations, DQN replay sampling and exploration),
        tensorflow (network weight initialisation), and mesa's own model RNG (RandomActivation's
        agent ordering). Must run before any agents are created. Op determinism is process-wide and
        stays on for any later models in the same process. seed=None leaves the run unseeded.
        """
        self.seed = seed
        if seed is None:
            return
        keras.utils.set_random_seed(seed)
        tensorflow.config.experimental.enable_op_determinism()
        self.reset_randomizer(seed)

    def get_num_agents(self):
        return self.num_agents
    
    def get_num_living_agents(self):
        return len(self.living_agents)
    
    def get_living_agents(self):
        return self.living_agents

    def can_throw_to(self, thrower, benefactor):
        """
        Whether thrower can reach benefactor with a berry; any agent can by default, scenarios restrict this by overriding
        """
        return True

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

    def _add_agent(self, i, agent_type, allotment, checkpoint_path, allocation_id=None):
        a = HarvestAgent(i,self,agent_type,allotment,self.training,checkpoint_path,self.epsilon,self.track_norms,shared_replay_buffer=self.shared_replay_buffer,allocation_id=allocation_id,norms_module_type=self.norms_module_type)
        if self.initial_healths is not None:
            self._apply_initial_health(a)
        self.schedule.add(a)
        self._place_agent_in_allotment(a)
        if a.agent_type != "berry":
            self.agent_id += 1
            self.living_agents.append(a)

    def _apply_initial_health(self, agent):
        """
        Set the agent's start health, which agent.reset() restores at the start of every episode,
        and its current health for the first episode
        """
        agent.start_health = self.initial_healths[agent.unique_id]
        agent.health = agent.start_health
        agent.days_left_to_live = agent.get_days_left_to_live()
        agent.total_days_left_to_live = agent.days_left_to_live

    def _reset(self):
        self.living_agents = []
        self.emerged_norms = {}
        self.day = 0
        num_agents = 0
        num_berries = 0
        self.episode += 1
        self.total_episode_reward = 0
        self._clear_grid()
        if self.seasons:
            self._restore_original_season()
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
                               "num_norms": [],
                               "initial_health": [],
                               "seasonal_berry_allocation": []})

    def _init_seasons(self):
        """
        Called by scenarios with allocations once their agents and berries exist: season 0 is the original
        assignment, where every holder (allocation key) holds its own allocation
        """
        self.season = 0
        self.season_assignment = {holder: holder for holder in self.allocations}
        if not self.seasons:
            return
        #which holder each berry belongs to originally, restored at the start of every episode
        self.original_berry_holders = {b.unique_id: b.allocation_id for b in self.berries}
        if len(self.allocations) < 2:
            warnings.warn(f"seasons: needs at least 2 agents, capability groups or allotments to swap berry allocations between, got {len(self.allocations)}; no swaps this run")
        elif len(set(self._season_values(self.season_assignment))) < 2:
            warnings.warn(f"seasons: every holder has the same berry allocation, so a swap can't change anything: {self._season_values(self.season_assignment)}; no swaps this run")
        else:
            self.season_swaps = True

    def _season_values(self, assignment):
        """
        Number of berries each holder has under assignment, in holder order
        """
        return [self.allocations[assignment[holder]]["berry_allocation"] for holder in self.allocations]

    def _seasonal_berry_allocation(self, agent):
        """
        Number of berries allocated to the agent's holder this season: its own allocation in colours, its
        group's or allotment's in capabilities and allotment (None for scenarios without allocations)
        """
        if self.season_assignment is None:
            return None
        return self.allocations[self.season_assignment[agent.allocation_id]]["berry_allocation"]

    def _start_new_season(self):
        """
        Randomly reassign the original allocations between holders, redrawing until the number of berries
        each holder has differs from the previous season; then move berries to match and update agents' paths
        so they head for berries they can now forage
        """
        #foraged berries regrow in the same step, so every berry should be on the grid at a swap
        for b in self.berries:
            if b.pos is None or b.foraged:
                raise BerryOffGrid(b.unique_id, self.day)
        holders = list(self.allocations.keys())
        previous_values = self._season_values(self.season_assignment)
        while True:
            assignment = {holder: holders[i] for holder, i in zip(holders, self.season_rng.permutation(len(holders)))}
            if self._season_values(assignment) != previous_values:
                break
        self.season += 1
        self.season_assignment = assignment
        self._move_berries_to_season_targets()
        for agent in self.living_agents:
            agent.moving_module.reset()
            agent.moving_module.check_nearest_berry(agent.pos)

    def _move_berries_to_season_targets(self):
        """
        Give each holder as many berries as the allocation it holds this season: randomly chosen surplus
        berries from holders whose allocation shrank go to holders whose allocation grew. Holders never change
        their members, capability or movement bounds, so in colours a berry changes colour, in capabilities it
        changes between tree and ground, and in allotment a berry outside its new allotment is moved to a
        random cell inside it; otherwise berries stay where they are
        """
        targets = dict(zip(self.allocations, self._season_values(self.season_assignment)))
        held = {holder: [b for b in self.berries if b.allocation_id == holder] for holder in self.allocations}
        surplus = []
        for holder, berries in held.items():
            excess = len(berries) - targets[holder]
            if excess > 0:
                chosen = sorted(self.season_rng.choice(len(berries), excess, replace=False))
                surplus += [berries[i] for i in chosen]
        for holder, berries in held.items():
            for _ in range(targets[holder] - len(berries)):
                berry = surplus.pop()
                self._set_berry_holder(berry, holder)
                if not self._inside_bounds(berry, berry.pos):
                    self.grid.remove_agent(berry)
                    self._place_agent_in_allotment(berry)

    def _set_berry_holder(self, berry, holder):
        """
        Give a berry to holder; scenarios where holders have their own region of the grid also set its bounds
        """
        berry.allocation_id = holder

    def _inside_bounds(self, agent, pos):
        return agent.min_width <= pos[0] < agent.max_width and agent.min_height <= pos[1] < agent.max_height

    def _restore_original_season(self):
        """
        Back to the original assignment (season 0) at the start of an episode; berries get their original
        holders back before they are placed on the grid
        """
        self.season = 0
        self.season_assignment = {holder: holder for holder in self.allocations}
        for b in self.berries:
            self._set_berry_holder(b, self.original_berry_holders[b.unique_id])

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
                               "num_norms": [],
                               "initial_health": [],
                               "seasonal_berry_allocation": []})
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
                               "num_emerged_norms": [],
                               "seed": []})
        if self.write_data:
            if exists("data/results/current_run/model_episode_reports_"+self.filepath+".csv"):
                raise FileExistsException("data/results/current_run/model_episode_reports_"+self.filepath+".csv")
            self.model_episode_reporter.to_csv("data/results/current_run/model_episode_reports_"+self.filepath+".csv", mode='a',index=False)
        if self.track_norms and self.norms_module_type == "bayesian":
            if exists(self._agent_norms_csv_filename()):
                raise FileExistsException(self._agent_norms_csv_filename())
            pd.DataFrame(columns=self._agent_norms_columns()).to_csv(self._agent_norms_csv_filename(), mode='a', index=False)

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
                               "num_norms": [self._num_agent_norms(agent)],
                               "initial_health": [agent.start_health],
                               "seasonal_berry_allocation": [self._seasonal_berry_allocation(agent)]})
        self.agent_reporter = pd.concat([self.agent_reporter, new_entry])
        if self.write_data and not self.training:
           new_entry.to_csv("data/results/current_run/agent_reports_"+self.filepath+".csv", header=None, mode='a',index=False)
    
    def _num_agent_norms(self, agent):
        """
        Size of an agent's norm base: behaviours in its behaviour base (rule-based), or behaviours
        it has learned (Bayesian, whose candidate set is fixed). None if norms aren't tracked.
        """
        if not self.track_norms:
            return None
        if self.norms_module_type == "rule_based":
            return len(agent.norms_module.behaviour_base)
        return len(agent.norms_module.get_learned_behaviours())

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
                               "num_emerged_norms": [len(self.emerged_norms) if self.track_norms else None],
                               "seed": [self.seed]})
        if self.write_data:
            new_entry.to_csv("data/results/current_run/model_episode_reports_"+self.filepath+".csv", header=None, mode='a',index=False)
        return new_entry

    def _append_norm_dict_to_file(self, norm_dictionary, filename):
        """
        Persists the current episode's emerged norms into filename, keyed by episode number
        Rewrites the whole file from a full in-memory dict each call so the file on disk is always valid,
        complete JSON, rather than relying on the run reaching its final episode to write a closing brace
        """
        if exists(filename):
            with open(filename, "r") as file:
                try:
                    all_episodes = json.load(file)
                except json.JSONDecodeError:
                    print(f"Warning: {filename} was not valid JSON (likely left over from an interrupted run); starting a new norms file")
                    all_episodes = {}
        else:
            all_episodes = {}
        norm_list = [{key: value} for key, value in norm_dictionary.items()]
        all_episodes[str(self.episode)] = norm_list
        with open(filename, "w") as file:
            json.dump(all_episodes, file, indent=4)

    def _analyse_agent_norms(self):
        """
        Collects one row per agent summarising the norms it has learned by the end of this episode and
        appends them to agent_norms_<filepath>.csv. Gained/lost compare against the behaviours the same
        agent had learned at the end of the previous episode (posteriors decay between episodes, so
        behaviours can be lost as well as gained). Cooperative norms are learned commitments to throw a
        berry to another agent ("IF ... THEN throw"); uncooperative norms are learned prohibitions on
        throwing ("IF ... THEN NOT throw"). Agents' throw_X actions are collapsed to "throw" for norms.
        """
        rows = []
        for agent in self.schedule.agents:
            if agent.agent_type == "berry":
                continue
            learned = agent.norms_module.get_learned_behaviours()
            learned_ids = {b["id"] for b in learned}
            previous_ids = self.previous_learned_norms.get(agent.unique_id, set())
            rows.append({"episode": self.episode,
                         "agent_id": agent.unique_id,
                         "agent_type": agent.agent_type,
                         "end_day": self.day,
                         "num_learned_norms": len(learned),
                         "num_learned_prohibitions": sum(b["type"] == "prohibition" for b in learned),
                         "num_learned_commitments": sum(b["type"] == "commitment" for b in learned),
                         "num_gained_norms": len(learned_ids - previous_ids),
                         "num_lost_norms": len(previous_ids - learned_ids),
                         "mean_learned_posterior": float(np.mean([b["posterior"] for b in learned])) if learned else None,
                         "num_observed_behaviours": sum(agent.norms_module.behaviour_counts(*view)[0] > 0 for view in agent.norms_module.iter_behaviours()),
                         "num_learned_cooperative_norms": sum(b["type"] == "commitment" and b["action"] == "throw" for b in learned),
                         "num_learned_uncooperative_norms": sum(b["type"] == "prohibition" and b["action"] == "throw" for b in learned)})
            self.previous_learned_norms[agent.unique_id] = learned_ids
        pd.DataFrame(rows, columns=self._agent_norms_columns()).to_csv(self._agent_norms_csv_filename(), header=None, mode='a', index=False)

    def _write_agent_norms_to_file(self):
        """
        Writes the norms each agent has learned by the end of this episode to that agent's own JSON file
        (<filepath>_agent_<id>_norms.json), keyed by episode number
        """
        for agent in self.schedule.agents:
            if agent.agent_type == "berry":
                continue
            learned_norms = {}
            for b in agent.norms_module.get_learned_behaviours():
                times_precondition_met, times_action_matched = agent.norms_module.behaviour_counts(b["precondition_key"], b["action_idx"], b["type"])
                learned_norms[b["label"]] = {"type": b["type"],
                                             "precondition": b["precondition"],
                                             "action": b["action"],
                                             "posterior": float(b["posterior"]),
                                             "times_precondition_met": times_precondition_met,
                                             "times_action_matched": times_action_matched}
            self._append_norm_dict_to_file(learned_norms, "data/results/current_run/"+self.filepath+"_agent_"+str(agent.unique_id)+"_norms.json")

    def _agent_norms_csv_filename(self):
        return "data/results/current_run/agent_norms_"+self.filepath+".csv"

    def _agent_norms_columns(self):
        return ["episode", "agent_id", "agent_type", "end_day", "num_learned_norms", "num_learned_prohibitions",
                "num_learned_commitments", "num_gained_norms", "num_lost_norms", "mean_learned_posterior",
                "num_observed_behaviours", "num_learned_cooperative_norms",
                "num_learned_uncooperative_norms"]

    def _check_emerged_norms(self):
        """
        function to check the current emerged norms at this step and update the norms that have emerged across the whole episode
        for each living agent, add the behaviours in its behaviour base to society_norms_snapshot
        filter society_norms_snapshot by emergence threshold
        update norms emerged across the whole episode with the norms emerged at this step
        """
        if len(self.living_agents) < 2:
            return
        emergence_threshold = len(self.living_agents) * self.societal_norm_emergence_threshold
        society_norms_snapshot = {}
        for agent in self.schedule.agents:
            if agent.agent_type != "berry" and not agent.off_grid:
                for norm_name, norm_value in agent.norms_module.behaviour_base.items():
                    society_norms_snapshot = self._add_agent_behaviour_to_snapshot(norm_name, norm_value, society_norms_snapshot)
        society_norms_snapshot = {norm: norm_value for norm, norm_value in society_norms_snapshot.items() if norm_value["adoption"] >= emergence_threshold}
        for norm_name, norm_value in society_norms_snapshot.items():
            self._update_episode_emerged_norm(norm_name, norm_value)

    def _update_episode_emerged_norm(self, norm_name, norm_value):
        """
        Overwrite an emerged norm's statistics with the society's current totals (agents' behaviour
        bases already hold running totals for the episode) and track when the norm was held: 
        first and last day above the emergence threshold, and number of days held
        """
        norm = self.emerged_norms.get(norm_name)
        if norm is None:
            norm = {"first_day_held": self.day, "days_held": 0}
            self.emerged_norms[norm_name] = norm
        norm["reward"] = norm_value["reward"]
        norm["numerosity"] = norm_value["numerosity"]
        norm["fitness"] = norm_value["fitness"]
        #number of living agents with the norm in their behaviour base on the last day it was held
        norm["adoption"] = norm_value["adoption"]
        norm["last_day_held"] = self.day
        norm["days_held"] += 1

    def _add_agent_behaviour_to_snapshot(self, norm_name, norm_value, society_norms_snapshot):
        """
        Add one agent's totals for a behaviour to the society's totals at this step
        """
        if norm_name not in society_norms_snapshot.keys():
            society_norms_snapshot[norm_name] = {"reward": 0,
                                        "numerosity": 0,
                                        "fitness": 0,
                                        "adoption": 0}
        society_norms_snapshot[norm_name]["reward"] += norm_value["reward"]
        #number times it has been used by individual agents
        society_norms_snapshot[norm_name]["numerosity"] += norm_value["numerosity"]
        society_norms_snapshot[norm_name]["fitness"] += norm_value["fitness"]
        #number of agents in the society with it in their behaviour base
        society_norms_snapshot[norm_name]["adoption"] += 1
        return society_norms_snapshot
    
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
    
    def _place_agent_in_allotment(self, agent):
        #for new agents who aren't yet on the grid
        if not self.grid.exists_empty_cells:
            raise NoEmptyCells
        cell = self._random_allotment_cell(agent)
        self.grid.place_agent(agent, cell)

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
        """
        Returns a list of starting berry counts, one per agent, modelling an unequal distribution
        of resources across the society (used by colours/allotment/capabilities scenarios; never
        called by basic_harvest). For num_agents in {2,4,6,20}, returns a hand-picked, hardcoded
        inequality pattern -- these are the exact, reproducible configurations behind the paper's
        published results (matching run.py's "select N to replicate results in the paper" prompts),
        not arbitrary sample data. For any other agent count, falls back to _generate_zipf_distribution
        to generate an equivalently-shaped inequality programmatically.
        Side effect: overwrites self.num_start_berries with sum(resources) -- the num_start_berries
        passed into the model's constructor does NOT directly control the final berry count for
        these scenarios; it's silently discarded here for the four hardcoded agent counts, and only
        used as a target total for the Zipf fallback (see _generate_zipf_distribution).
        """
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
        """
        Distributes self.num_start_berries (the target total, read here before
        _generate_resource_allocations overwrites it) across num_agents agents in a Zipf's-law
        ("few haves, many have-nots") shape, matching the character of the hardcoded distributions
        this is a fallback for.
        """
        total_resources = self.num_start_berries

        # Generate Zipf-like distribution
        weights = 1 / np.arange(1, num_agents + 1)  # [1, 1/2, 1/3, ...] -- rank 0 gets the largest share
        weights /= weights.sum()
        allocations = (weights * total_resources).astype(int)

        #.astype(int) truncates each share toward zero, which can only ever lose berries to
        #rounding (never gain), so diff is always >= 0; redistribute that shortfall one berry at a
        #time to a random agent so the total exactly matches total_resources
        diff = total_resources - allocations.sum()
        for _ in range(abs(diff)):
            i = np.random.randint(num_agents)
            allocations[i] += np.sign(diff)

        #without this, agent index/unique_id 0 would always be the richest agent, since the Zipf
        #weighting always puts the largest share at rank 0 -- shuffle to decouple "who is agent 0"
        #from "who is rich" while keeping the overall few-rich-many-poor distribution shape
        np.random.shuffle(allocations)
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
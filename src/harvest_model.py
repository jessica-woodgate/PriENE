from mesa.time import RandomActivation
from mesa.space import MultiGrid
import pandas as pd
import numpy as np
import os
from tensorflow import keras
from .society_model import SocietyModel
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
from .harvest_exception import SeasonsException
from .harvest_exception import BerryOffGrid
from .harvest_exception import NoAllocationException
from os.path import exists

class HarvestModel(SocietyModel):
    """
    Harvest model handles the environment, stepping agents, and data collection
    Instance variables:
        num_agents -- number of agents at beginning of each episode
        num_berries -- number of berries in use (at the beginning of each episode, and throughout unless a good or bad season changes it)
        day -- current day
        max_days -- max days in a single episode
        max_episodes -- maximum number of episodes (for testing; training runs until epsilon is min epsilon)
        min_epsilon -- minimum epsilon to end training
        schedule -- schedule of agents
        max_width -- width of grid
        max_height -- height of grid
        grid -- grid object
        shared_replay_buffers -- dict mapping each agent type to the replay buffer its agents share to reduce training time; agents of
            different types (in a mixed society) are rewarded differently, so they never learn from each other's experiences
        berry_id -- tracker for unique berry ids
        episode -- current episode, starting from start_episode: 1, or the episode a stopped training run continues from (its agents then
            load the networks it saved, and its report is continued)
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
        allocations -- dict mapping each holder (an allocation key: a capability group, an allotment, in colours a single agent, or in basic
            the whole society) to its agents and berry allocation
        seasons -- boolean for every season, including the first of each episode, to get a quality drawn at random from SEASON_QUALITIES;
            in a good season each holder's target is its allocation plus one berry per agent in the holder, in a bad season minus one per
            agent (at least 0); allocations always stay with the same holder
        season_length -- days in each season; quality changes happen after the steps on days season_length, 2*season_length, ...
        season -- index of the current season in this episode (0 = the first)
        current_season_quality -- quality of the current season (None without seasons)
        season_rng -- generator for season qualities and the berries they add or remove, seeded from seed and separate from the other random streams so seasons don't shift them
        dormant_berries -- berry objects not used this season (off the grid and out of the schedule), reused before new berries are made
        allotment_regrowth_delays -- steps a foraged berry waits before regrowing, one per allotment in allotment order (None = every berry
            regrows in the step it is foraged); only the allotment scenario takes it
        holder_regrowth_delays -- dict mapping each holder (allotment key) to its regrowth delay (None without allotment regrowth delays)
        regrowth_timers -- dict mapping each berry waiting to regrow (foraged, off the grid) to the day it is due back; emptied at the start of every episode
        baseline_ids -- ids of the baseline agents in a mixed society, chosen from the seed (empty in a homogeneous society)
    """
    SEASON_QUALITIES = ["normal", "good", "bad"]
    SEASON_QUALITY_CHANGE = {"normal": 0, "good": 1, "bad": -1}
    #spawn key for the random stream that puts agents in allocations each episode (commons uses 2)
    ALLOCATION_MEMBERSHIP_STREAM = 3
    #whether agents are put in allocations afresh each episode (see _episode_groups): scenarios with more than one allocation
    SHUFFLES_ALLOCATIONS = False
    #columns of the agent report (one row per agent per step) and the model episode report (one row per episode), in order, before any
    #a scenario adds; each row's values are listed in the same order in _collect_agent_data and _collect_model_episode_data
    AGENT_REPORT_FIELDS = ["agent_id", "agent_type", "episode", "day", "berries", "berries_consumed", "berries_thrown", "health",
                           "days_left_to_live", "total_days_left_to_live", "action", "thrown_to", "reward", "ethics_sanction", "num_norms",
                           "season", "season_quality", "seasonal_berry_target", "allotment_regrowth_delay"]
    EPISODE_REPORT_FIELDS = ["episode", "end_day", "epsilon", "mean_reward", "mean_loss", "max_berries", "mean_berries", "max_berries_consumed",
                             "mean_berries_consumed", "gini_berries_consumed", "mean_berries_thrown", "max_health", "mean_health",
                             "median_health", "variance_health", "deceased", "num_emerged_norms", "seed"]

    def __init__(self,num_agents,max_width,max_height,max_episodes,max_days,training,write_data,track_norms,filepath="",seed=None,norms_module_type="rule_based",seasons=False,season_length=25,allotment_regrowth_delays=None,start_episode=1):
        super().__init__()
        self._set_seed(seed)
        self.num_agents = num_agents
        if self.num_agents <= 0:
            raise NumAgentsException(">0", 0)
        self.num_berries = 0
        self.day = 0
        self.max_days = max_days
        self.max_episodes = max_episodes
        self.min_epsilon = 0.01
        self.schedule = RandomActivation(self)
        self.max_width = max_width
        self.max_height = max_height
        self.grid = MultiGrid(self.max_width, self.max_height, False)
        self.shared_replay_buffers = {}
        self.berry_id = 0
        self.episode = start_episode
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
            #agents start training fully exploring, and epsilon decays from there each episode
            self.epsilon = 1.0
        else:
            self.epsilon = 0.0
        if seasons and (not isinstance(season_length, int) or season_length < 1):
            raise SeasonsException(f"season length must be a whole number of days of at least 1, got {season_length}")
        self.allocations = None
        self.seasons = seasons
        self.season_length = season_length
        self.season = 0
        self.current_season_quality = None
        self.season_rng = np.random.default_rng(seed) if seasons else None
        self.dormant_berries = []
        self.allotment_regrowth_delays = None if allotment_regrowth_delays is None else list(allotment_regrowth_delays)
        self.holder_regrowth_delays = None
        self.regrowth_timers = {}
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
        elif self.seasons and self.day % self.season_length == 0:
            self._start_new_season()

    def finish_episode(self, collect_data=True):
        if self.track_norms and collect_data:
            # record norms before agents' finish_episode(), so any end-of-episode norm update (e.g. decay) is excluded
            if self.norms_module_type == "rule_based":
                self._append_norm_dict_to_file(self.emerged_norms, "data/results/current_run/"+self.filepath+"_emerged_norms.json")
            else:
                self._analyse_agent_norms()
        for a in self.schedule.agents:
            if a.agent_type != "berry":
                a.finish_episode()
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
        Get the coordinates of uneaten berries (of one allocation, or of any if allocation_id is None)
        """
        return [b.pos for b in self.berries if not b.foraged and (allocation_id is None or b.allocation_id == allocation_id)]
    
    def get_uneaten_berry_by_coords(self, coords, allocation_id=None):
        """
        Get an uneaten berry by its coordinates (of one allocation, or of any if allocation_id is None)
        """
        for b in self.berries:
            if b.pos == coords and not b.foraged and (allocation_id is None or b.allocation_id == allocation_id):
                return b
        raise NoBerriesException(coordinates=coords)
    
    def get_society_well_being(self, observer, norms_observation, ethics_observation):
        """
        Get the well-being of a society; iterates over all agents in the society, excluding the observer if the agent is observing
        """
        society_well_being = []
        for a in self.schedule.agents:
            if (not ethics_observation and (a.unique_id == observer.unique_id)) or a.agent_type == "berry":
                continue
            elif a.done == False:
                society_well_being.append(a.days_left_to_live)
            elif not (norms_observation or ethics_observation):
                #a dead agent is seen with no days left to live, except by norms and ethics, which leave it out
                society_well_being.append(0)
        return np.array(society_well_being, dtype=float)

    def reseed_for_continuing(self):
        """
        A continued training run is built from its seed, so its agents and their groups are the same as when it started; call this once it
        is built to reseed every random stream from the seed and the episode it continues from, so it doesn't repeat the run's first episodes
        """
        if self.seed is None:
            return
        seed = int(np.random.SeedSequence([self.seed, self.episode]).generate_state(1)[0])
        keras.utils.set_random_seed(seed)
        self.reset_randomizer(seed)
        if self.seasons:
            self.season_rng = np.random.default_rng([self.seed, self.episode])

    def get_living_agents(self):
        return self.living_agents

    def can_throw_to(self, thrower, benefactor):
        """
        Whether thrower can reach benefactor with a berry; any agent can by default, scenarios restrict this by overriding
        """
        return True

    def num_extra_observation_features(self):
        """
        Number of features a scenario adds to the end of every agent's observation (none by default)
        """
        return 0

    def extra_observation_features(self, agent):
        """
        Features a scenario adds to the end of agent's observation, after its own and other agents' well-being, so the existing features
        keep their indices (none by default)
        """
        return np.array([])

    def extra_observation_feature_specs(self):
        """
        One spec per extra observation feature, so norms can use it as a precondition: name, thresholds (ascending) and labels (one
        more than thresholds, for values below the first threshold up to values at or above the last); none by default
        """
        return []

    def extra_agent_report_columns(self):
        """
        Columns a scenario adds to the end of the agent report, one row per agent per step (none by default)
        """
        return []

    def extra_agent_report_values(self, agent):
        """
        This step's value for each of extra_agent_report_columns in agent's row, by column
        """
        return {}

    def extra_episode_report_columns(self):
        """
        Columns a scenario adds to the end of the model episode report (none by default)
        """
        return []

    def extra_episode_report_values(self):
        """
        This episode's value for each of extra_episode_report_columns, by column
        """
        return {}

    def _end_of_step_dynamics(self):
        """
        Changes to the environment once every agent has acted and this step's foraged berries are handled, before agents' rows are
        reported and the end of episode check (none by default)
        """
        pass

    def get_day(self):
        return self.day
    
    def get_max_days(self):
        return self.max_days
    
    def _init_berries(self):
        """
        Each allocation's berries in turn, each placed as it is made, in its allocation's region of the grid (the commons makes its own)
        """
        berries = []
        self.num_berries = 0
        for allocation_id, allocation_data in self.allocations.items():
            for _ in range(allocation_data["berry_allocation"]):
                b = self._new_berry(self._allocation_region(allocation_id), allocation_data["id"])
                self._place_agent_in_allotment(b)
                self.num_berries += 1
                berries.append(b)
        if self.num_berries != self.num_start_berries:
            raise NumBerriesException(self.num_start_berries, self.num_berries)
        return berries
    
    def _init_agents(self, society_mix, agent_type, checkpoint_path):
        """
        Agents in id order, each placed as it is made; in scenarios with allocations (SHUFFLES_ALLOCATIONS) each agent is given one, whose
        berries it forages, in its region of the grid (see _allocation_region); in a mixed society the baseline agents are chosen from the seed
        """
        self.living_agents = []
        self._choose_agent_types(society_mix)
        if self.SHUFFLES_ALLOCATIONS:
            self._assign_agents_to_allocations()
        for i in range(self.num_agents):
            allocation_id = self._get_allocation_id(i) if self.SHUFFLES_ALLOCATIONS else None
            self._add_agent(i, self._agent_type_of(i, agent_type), self._allocation_region(allocation_id), checkpoint_path, allocation_id=allocation_id)
        if len(self.living_agents) != self.num_agents:
            raise NumAgentsException(self.num_agents, len(self.living_agents))
        self.berry_id = len(self.living_agents) + 1

    def _allocation_region(self, allocation_id):
        """
        The part of the grid an allocation's agents and berries are in, as [min_width, max_width, min_height, max_height]: the whole grid,
        unless the scenario has allotments
        """
        return [0, self.max_width, 0, self.max_height]

    def _add_agent(self, i, agent_type, allotment, checkpoint_path, allocation_id=None):
        shared_replay_buffer = self.shared_replay_buffers.setdefault(agent_type, {"s": [], "a": [], "r": [], "s_": [], "done": []})
        a = HarvestAgent(i,self,agent_type,allotment,self.training,checkpoint_path,self.epsilon,self.track_norms,shared_replay_buffer=shared_replay_buffer,allocation_id=allocation_id,norms_module_type=self.norms_module_type)
        self.schedule.add(a)
        self._place_agent_in_allotment(a)
        self.living_agents.append(a)

    def _reset(self):
        self.living_agents = []
        self.emerged_norms = {}
        self.day = 0
        num_agents = 0
        num_berries = 0
        self.episode += 1
        #every berry waiting to regrow is placed again below, so no timer carries over to the new episode
        self.regrowth_timers = {}
        self._clear_grid()
        self._restore_episode_state()
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
        if self.seasons:
            self._start_season_quality()
        self.agent_reporter = self._empty_agent_reporter()

    def _restore_episode_state(self):
        """
        Restore the scenario's state for a new episode, after the grid is cleared and before agents and berries are reset and placed:
        by default, the berries each season started with; and with SHUFFLES_ALLOCATIONS, agents are put in allocations afresh
        """
        if self.seasons:
            self._restore_original_berries()
        if self.SHUFFLES_ALLOCATIONS:
            self._assign_agents_to_allocations()
            for agent in self.schedule.agents:
                if agent.agent_type != "berry":
                    self._move_to_allocation(agent, self._get_allocation_id(agent.unique_id))

    def _allocation_membership_rng(self):
        #from the seed and the episode, in a stream of its own, so it shifts no other random draws, every agent type gets the same groups
        #in the same episode, and a continued run gets the groups the run it continues would have had
        return np.random.default_rng(None if self.seed is None else [self.seed, self.ALLOCATION_MEMBERSHIP_STREAM, self.episode])

    def _episode_groups(self):
        """
        Agent ids in each allocation this episode: every agent is shuffled among the allocations, each keeping its number of agents, so no
        agent has the same allocation (and its resources) every episode
        """
        order = [int(agent_id) for agent_id in self._allocation_membership_rng().permutation(self.num_agents)]
        return self._split_into_groups(order, [len(data["agent_ids"]) for data in self.allocations.values()])

    def _even_sizes(self, total, num_parts):
        """
        num_parts sizes summing to total, as equal as they can be: the first ones are one bigger when total doesn't divide evenly
        """
        size, extra = divmod(total, num_parts)
        return [size + (1 if i < extra else 0) for i in range(num_parts)]

    def _consecutive_groups(self, ids, num_groups):
        """
        ids split, in order, into num_groups groups as equal in size as they can be (the first ones bigger)
        """
        ids = list(ids)
        return self._split_into_groups(ids, self._even_sizes(len(ids), num_groups))

    def _split_into_groups(self, ids, sizes):
        groups = []
        start = 0
        for size in sizes:
            groups.append(ids[start:start + size])
            start += size
        return groups

    def _assign_agents_to_allocations(self):
        for allocation_data, group in zip(self.allocations.values(), self._episode_groups()):
            allocation_data["agent_ids"] = group

    def _get_allocation_id(self, agent_id):
        for allocation_id, allocation_data in self.allocations.items():
            if agent_id in allocation_data["agent_ids"]:
                return allocation_id
        raise NoAllocationException(agent_id)

    def _move_to_allocation(self, agent, allocation_id):
        """
        An agent moving allocation forages only its new allocation's berries (scenarios with allotments also change its bounds)
        """
        agent.allocation_id = allocation_id
        agent.moving_module.allocation_id = allocation_id

    def _init_seasons(self):
        """
        Called by scenarios with allocations once their agents and berries exist: with seasons on, season 0
        gets a random quality
        """
        self.season = 0
        if not self.seasons:
            return
        #the berries and which holder each belongs to originally, restored at the start of every episode
        self.original_berries = list(self.berries)
        self.original_berry_holders = {b.unique_id: b.allocation_id for b in self.berries}
        self._start_season_quality()

    def _current_season_quality(self):
        """
        Quality of the current season (None without seasons)
        """
        if not self.seasons:
            return None
        return self.current_season_quality

    def _draw_season_quality(self):
        """
        Quality for a new season, each of SEASON_QUALITIES equally likely (one draw from season_rng)
        """
        return self.SEASON_QUALITIES[self.season_rng.integers(len(self.SEASON_QUALITIES))]

    def _start_season_quality(self):
        """
        Draw the quality of the season that is starting, and add or remove berries to match it
        """
        self.current_season_quality = self._draw_season_quality()
        self._apply_season_quality()

    def _season_targets(self):
        """
        Number of berries each holder has this season: its allocation, changed by one berry per agent in the
        holder in good and bad seasons (never below 0)
        """
        change = self.SEASON_QUALITY_CHANGE[self._current_season_quality()] if self.seasons else 0
        return {holder: max(0, allocation_data["berry_allocation"] + change * len(allocation_data["agent_ids"]))
                for holder, allocation_data in self.allocations.items()}

    def _seasonal_berry_target(self, agent):
        """
        Number of berries the agent's holder has this season (None for scenarios without allocations)
        """
        if self.allocations is None:
            return None
        return self._season_targets()[self._holder_of(agent)]

    def _holder_of(self, agent):
        """
        The holder (allocation key) whose berries the agent forages; basic overrides this, since its agents aren't restricted to an allocation
        """
        return agent.allocation_id

    def _start_new_season(self):
        """
        Draw the new season's quality and add or remove berries to match it, then update agents' paths so
        they head for berries that are still there
        """
        #foraged berries regrow in the same step or wait on a regrowth timer, so every other berry should be on the grid at a season change
        for b in self.berries:
            if (b.pos is None or b.foraged) and b not in self.regrowth_timers:
                raise BerryOffGrid(b.unique_id, self.day)
        self.season += 1
        self._start_season_quality()
        for agent in self.living_agents:
            agent.moving_module.reset()
            agent.moving_module.check_nearest_berry(agent.pos)

    def _choose_surplus_berries(self, berries, excess):
        """
        Randomly choose excess of a holder's berries to give up: uncollected berries on the grid first, then, only
        if there aren't enough of those, berries waiting to regrow
        """
        #with no delay above 0 no berry ever waits, so the draw is the same as without regrowth delays
        if self.holder_regrowth_delays is None or not any(self.holder_regrowth_delays.values()):
            return [berries[i] for i in sorted(self.season_rng.choice(len(berries), excess, replace=False))]
        #one permutation of all the holder's berries however many are waiting, so the stream doesn't depend on agents' behaviour
        order = [berries[i] for i in self.season_rng.permutation(len(berries))]
        ranked = [b for b in order if b not in self.regrowth_timers] + [b for b in order if b in self.regrowth_timers]
        return ranked[:excess]

    def _apply_season_quality(self):
        """
        Give each holder as many berries as its target this season: randomly chosen uncollected berries are
        removed from holders above their target, and holders below it get new berries at random cells in their
        region of the grid; carried berries are only counts on agents, so they are never removed. Berries waiting
        to regrow count as their holder's, and are only removed (cancelling their timers) once no uncollected
        berries are left to remove. Every removal comes before any addition, so berries removed from one holder
        can be reused by another
        """
        targets = self._season_targets()
        shortfalls = {}
        for holder in self.allocations:
            held = [b for b in self.berries if b.allocation_id == holder and (not b.foraged or b in self.regrowth_timers)]
            excess = len(held) - targets[holder]
            if excess > 0:
                for berry in self._choose_surplus_berries(held, excess):
                    self._remove_berry(berry)
            shortfalls[holder] = max(0, -excess)
        for holder, shortfall in shortfalls.items():
            for _ in range(shortfall):
                self._add_season_berry(holder)

    def _remove_berry(self, berry):
        """
        Take a berry out of use (off the grid and out of the schedule), keeping it to reuse; a berry waiting to regrow has its timer cancelled
        """
        self.regrowth_timers.pop(berry, None)
        if berry.pos is not None:
            self.grid.remove_agent(berry)
        self.schedule.remove(berry)
        self.berries.remove(berry)
        self.dormant_berries.append(berry)
        self.num_berries -= 1

    def _add_season_berry(self, holder):
        """
        Give holder another berry, placed at a random cell in its region
        """
        def season_cell(berry):
            #two draws per berry from season_rng whatever is on the grid, so the stream doesn't depend on agents' behaviour
            return (int(self.season_rng.integers(berry.min_width, berry.max_width)), int(self.season_rng.integers(berry.min_height, berry.max_height)))
        self._add_berry(holder, season_cell)

    def _add_berry(self, holder, choose_cell):
        """
        Put another berry into use for holder, reusing a dormant one if there is one, at the cell choose_cell(berry) returns
        """
        if self.dormant_berries:
            berry = self.dormant_berries.pop(0)
            self.schedule.add(berry)
        else:
            berry = self._new_berry([0, self.max_width, 0, self.max_height])
        berry.reset()
        self._set_berry_holder(berry, holder)
        self.grid.place_agent(berry, choose_cell(berry))
        self.berries.append(berry)
        self.num_berries += 1
        return berry

    def _set_berry_holder(self, berry, holder):
        """
        Give a berry to holder; scenarios where holders have their own region of the grid also set its bounds
        """
        berry.allocation_id = holder

    def _restore_original_berries(self):
        """
        Back to the original berries (season 0) at the start of an episode, after the grid is cleared: berries
        added this episode become dormant and removed original berries come back. A dormant berry can be reused
        by another holder, so berries get their original holders back before they are placed on the grid
        """
        self.season = 0
        original_ids = set(self.original_berry_holders)
        for b in self.berries:
            if b.unique_id not in original_ids:
                self.schedule.remove(b)
                self.dormant_berries.append(b)
        for b in self.original_berries:
            if b in self.dormant_berries:
                self.dormant_berries.remove(b)
                self.schedule.add(b)
        self.berries = list(self.original_berries)
        self.num_berries = len(self.berries)
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
        
    def _empty_agent_reporter(self):
        agent_reporter = pd.DataFrame({column: [] for column in self.AGENT_REPORT_FIELDS})
        for column in self.extra_agent_report_columns():
            agent_reporter[column] = []
        return agent_reporter

    def _init_reporters(self):
        os.makedirs("data/results/current_run", exist_ok=True)
        self.agent_reporter = self._empty_agent_reporter()
        if self.write_data and not self.training:
           if exists("data/results/current_run/agent_reports_"+self.filepath+".csv"):
               raise FileExistsException("data/results/current_run/agent_reports_"+self.filepath+".csv")
           self.agent_reporter.to_csv("data/results/current_run/agent_reports_"+self.filepath+".csv", mode='a',index=False)
        self.model_episode_reporter = pd.DataFrame({column: [] for column in self.EPISODE_REPORT_FIELDS})
        for column in self.extra_episode_report_columns():
            self.model_episode_reporter[column] = []
        if self.write_data:
            self._start_report("data/results/current_run/model_episode_reports_"+self.filepath+".csv", list(self.model_episode_reporter.columns))
        if self.track_norms and self.norms_module_type == "bayesian":
            if exists(self._agent_norms_csv_filename()):
                raise FileExistsException(self._agent_norms_csv_filename())
            pd.DataFrame(columns=self._agent_norms_columns()).to_csv(self._agent_norms_csv_filename(), mode='a', index=False)

    def _collect_agent_data(self, agent):
        #in the order of AGENT_REPORT_FIELDS
        values = [agent.unique_id, agent.agent_type, self.episode, self.day, agent.berries, agent.berries_consumed, agent.berries_thrown,
                  agent.health, agent.days_left_to_live, agent.total_days_left_to_live, agent.current_action, agent.current_thrown_to,
                  agent.current_reward, agent.current_sanction, self._num_agent_norms(agent), self.season, self._current_season_quality(),
                  self._seasonal_berry_target(agent), self._allotment_regrowth_delay(agent)]
        new_entry = self._report_row(self.AGENT_REPORT_FIELDS, values, self.extra_agent_report_values(agent))
        self.agent_reporter = pd.concat([self.agent_reporter, new_entry])
        if self.write_data and not self.training:
           new_entry.to_csv("data/results/current_run/agent_reports_"+self.filepath+".csv", header=None, mode='a',index=False)

    def _collect_model_episode_data(self):
        #every row of the agent reporter is added with index 0, so select this episode's rows by value, not by index
        rows = self.agent_reporter[self.agent_reporter["episode"] == self.episode]
        #in the order of EPISODE_REPORT_FIELDS
        values = [self.episode, self.day, self.epsilon, self._mean_reward(), self._mean_loss(), rows["berries"].max(), rows["berries"].mean(axis=0),
                  rows["berries_consumed"].max(), rows["berries_consumed"].mean(axis=0), self._gini_berries_consumed(),
                  rows["berries_thrown"].mean(axis=0), rows["health"].max(), rows["health"].mean(axis=0), rows["health"].median(),
                  rows["health"].var(axis=0), self.num_agents - len(self.living_agents), len(self.emerged_norms) if self.track_norms else None,
                  self.seed]
        new_entry = self._report_row(self.EPISODE_REPORT_FIELDS, values, self.extra_episode_report_values())
        if self.write_data:
            new_entry.to_csv("data/results/current_run/model_episode_reports_"+self.filepath+".csv", header=None, mode='a',index=False)
        return new_entry

    def _update_schedule(self):
        if self.regrowth_timers:
            self._regrow_due_berries()
        scheduled = self.schedule.agents
        #foraged berries, then the step's end-of-step changes, so every agent's row sees the whole step; then dead agents
        for a in scheduled:
            if a.agent_type == "berry" and a.foraged == True:
                self._handle_foraged_berry(a)
        self._end_of_step_dynamics()
        for a in scheduled:
            if a.agent_type != "berry":
                if a.off_grid == False:
                    self._collect_agent_data(a)
                    if a.done == True:
                        self._remove_agent(a)

    def _handle_foraged_berry(self, berry):
        """
        A berry foraged this step regrows straight away at a random cell, or, with an allotment regrowth delay, waits off the grid until
        its allotment's delay has passed
        """
        if self.holder_regrowth_delays is None or self.holder_regrowth_delays[berry.allocation_id] == 0:
            self._reset_berry(berry, False)
        elif berry not in self.regrowth_timers:
            self.grid.remove_agent(berry)
            self.regrowth_timers[berry] = self.day + self.holder_regrowth_delays[berry.allocation_id]

    def _regrow_due_berries(self):
        """
        Berries whose regrowth timer is due today regrow at a random cell in their allotment, as long as the
        allotment has fewer berries on the grid than its target this season; otherwise the timer is discarded
        and the berry is kept to reuse
        """
        targets = self._season_targets()
        for berry in [b for b, due in self.regrowth_timers.items() if due <= self.day]:
            del self.regrowth_timers[berry]
            on_grid = sum(1 for b in self.berries if b.allocation_id == berry.allocation_id and b.pos is not None)
            if on_grid < targets[berry.allocation_id]:
                berry.reset()
                self._place_agent_in_allotment(berry)
            else:
                self._remove_berry(berry)

    def _allotment_regrowth_delay(self, agent):
        """
        Regrowth delay of the agent's allotment (None without allotment regrowth delays)
        """
        if self.holder_regrowth_delays is None:
            return None
        return self.holder_regrowth_delays[agent.allocation_id]

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
            #berries waiting to regrow are already off the grid
            if a.agent_type == "berry" and a.pos is None:
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
        published results, not arbitrary sample data. For any other agent count, falls back to _generate_zipf_distribution
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
        #over every agent, dead or alive, so an episode that ends with every agent dead still has its inequality measured
        berries_consumed = [a.berries_consumed for a in self.schedule.agents if a.agent_type != "berry"]
        x = sorted(berries_consumed)
        s = sum(x)
        if s == 0:
            return 0
        N = self.num_agents
        B = sum(xi * (N - i) for i, xi in enumerate(x)) / (N * s)
        return 1 + (1 / N) - 2 * B
from src.harvest_model import HarvestModel
from src.harvest_exception import RegrowthDelaysNotSupported
from src.harvest_exception import CommonsParameterException
from src.harvest_exception import CommonsGridException
import numpy as np
import os
import pandas as pd

class CommonsHarvest(HarvestModel):
    """
    Commons harvest scenario: a common-pool resource that any agent can forage (open access), whose berries regrow according to the
    current stock with logistic dynamics; agents observe the stock as a fraction of carrying capacity, after their other features
    Instance variables:
        carrying_capacity -- K, the most berries the commons holds
        growth_rate -- r, the commons' growth rate in a normal season (and without seasons)
        initial_stock -- R0, the berries the commons starts every episode with
        current_growth_rate -- growth rate this season: growth_rate changed by GROWTH_RATE_MULTIPLIERS for the season's quality
        regrowth_rng -- generator for the rounding of regrowth and the cells berries regrow at, seeded from the seed and its own stream
            (COMMONS_REGROWTH_STREAM) so regrowth doesn't shift other random draws
        harvested_this_step -- berries foraged in the current step
        commons_steps -- this episode's steps, each its stock, harvest, regrowth and growth rate at the end of the step: added to every
            agent's row of the agent report (AGENT_REPORT_COLUMNS), and summarised at the end of the model episode report's rows
            (EPISODE_SUMMARY_COLUMNS)
        collapse_day -- the first day this episode the stock was 0 (None if it hasn't collapsed); once collapsed, nothing regrows
    """
    COMMONS = "allocation_commons"
    #spawn key for the commons' regrowth random stream (see HarvestModel.MIXED_SOCIETY_STREAM)
    COMMONS_REGROWTH_STREAM = 2
    #in good seasons the commons regrows faster, in bad seasons slower; seasons never add or remove berries
    GROWTH_RATE_MULTIPLIERS = {"normal": 1.0, "good": 1.5, "bad": 0.5}
    #defaults for run.py
    CARRYING_CAPACITY_PER_AGENT = 12
    DEFAULT_GROWTH_RATE = 0.25
    #the stock (as a fraction of carrying capacity) as a norm precondition
    STOCK_FEATURE_SPEC = {"name": "stock", "thresholds": [0.25, 0.5, 0.75],
                          "labels": ["very low stock", "low stock", "high stock", "very high stock"]}
    #added to the end of the agent report (which already has the season and its quality): the stock after the step's foraging and
    #regrowth, the berries every agent harvested and the commons regrew this step, and the growth rate
    AGENT_REPORT_COLUMNS = ["stock", "harvested_this_step", "regrown_this_step", "growth_rate"]
    #added to the end of the model episode report
    EPISODE_SUMMARY_COLUMNS = ["collapse_day", "mean_stock", "final_stock", "efficiency"]

    def __init__(self,society_mix,num_agents,carrying_capacity,growth_rate,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath="",seed=None,norms_module_type="rule_based",seasons=False,season_length=25,allotment_regrowth_delays=None,start_episode=1,initial_stock=None):
        if allotment_regrowth_delays is not None:
            raise RegrowthDelaysNotSupported("commons")
        if initial_stock is None:
            initial_stock = carrying_capacity // 2
        self._check_parameters(num_agents, carrying_capacity, growth_rate, initial_stock, max_width, max_height)
        self.carrying_capacity = carrying_capacity
        self.growth_rate = growth_rate
        self.initial_stock = initial_stock
        super().__init__(num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath,seed,norms_module_type,seasons,season_length,start_episode=start_episode)
        self.regrowth_rng = np.random.default_rng(None if seed is None else [seed, self.COMMONS_REGROWTH_STREAM])
        self.current_growth_rate = growth_rate
        self.harvested_this_step = 0
        self.commons_steps = []
        self.collapse_day = None
        self._init_commons_reporters()
        self._init_agents(society_mix, agent_type, checkpoint_path)
        self.berries = self._init_berries()
        self.season = 0
        if self.seasons:
            self._start_season_quality()

    @staticmethod
    def _check_parameters(num_agents, carrying_capacity, growth_rate, initial_stock, max_width, max_height):
        if not growth_rate > 0:
            raise CommonsParameterException(f"the growth rate must be more than 0, got {growth_rate}")
        if not 1 <= initial_stock <= carrying_capacity:
            raise CommonsParameterException(f"the initial stock must be at least 1 and at most the carrying capacity {carrying_capacity}, got {initial_stock}")
        if carrying_capacity + num_agents > max_width * max_height:
            raise CommonsGridException(carrying_capacity, num_agents, max_width * max_height)

    def _init_berries(self):
        berries = []
        self.num_berries = 0
        for _ in range(self.initial_stock):
            b = self._new_berry([0, self.max_width, 0, self.max_height])
            self._set_berry_holder(b, self.COMMONS)
            self._place_agent_in_allotment(b)
            self.num_berries += 1
            berries.append(b)
        return berries

    def stock(self):
        """
        R_t: the uncollected berries on the grid, counted live (agents act in turn within a step)
        """
        return sum(1 for b in self.berries if not b.foraged)

    def num_extra_observation_features(self):
        return 1

    def extra_observation_features(self, agent):
        return np.array([self.stock() / self.carrying_capacity])

    def extra_observation_feature_specs(self):
        return [self.STOCK_FEATURE_SPEC]

    def reseed_for_continuing(self):
        super().reseed_for_continuing()
        if self.seed is not None:
            self.regrowth_rng = np.random.default_rng([self.seed, self.COMMONS_REGROWTH_STREAM, self.episode])

    def _place_agent_in_allotment(self, agent):
        #berries take a free cell, so the commons can hold up to its carrying capacity; agents are placed as in the other scenarios
        if agent.agent_type != "berry":
            return super()._place_agent_in_allotment(agent)
        cells = self._free_cells()
        self.grid.place_agent(agent, cells[np.random.randint(len(cells))])

    def _free_cells(self):
        #sorted, so which cell a draw picks doesn't depend on the order the grid keeps them in
        return sorted(self.grid.empties)

    def _handle_foraged_berry(self, berry):
        """
        A foraged berry leaves the stock (off the grid and out of the schedule), kept to reuse when the commons regrows
        """
        self._remove_berry(berry)
        self.harvested_this_step += 1

    def _regrowth(self, stock):
        """
        Berries the commons regrows this step from logistic growth r * R * (1 - R / K), rounded down and up with probability equal to its
        fractional part, never above carrying capacity; nothing regrows from a stock of 0
        """
        growth = self.current_growth_rate * stock * (1 - stock / self.carrying_capacity)
        new = int(np.floor(growth))
        if self.regrowth_rng.random() < growth - new:
            new += 1
        return min(new, self.carrying_capacity - stock)

    def _end_of_step_dynamics(self):
        """
        Once every agent has acted and foraged berries have left the stock, the commons regrows at random free cells
        """
        def regrowth_cell(berry):
            cells = self._free_cells()
            return cells[self.regrowth_rng.integers(len(cells))]
        regrown = self._regrowth(self.stock())
        for _ in range(regrown):
            self._add_berry(self.COMMONS, regrowth_cell)
        stock = self.stock()
        if stock == 0 and self.collapse_day is None:
            self.collapse_day = self.day
        self.commons_steps.append({"stock": stock, "harvested": self.harvested_this_step, "regrown": regrown, "growth_rate": self.current_growth_rate})
        self.harvested_this_step = 0

    def _apply_season_quality(self):
        """
        Season quality changes how fast the commons regrows, not its berries
        """
        self.current_growth_rate = self.growth_rate * self.GROWTH_RATE_MULTIPLIERS[self._current_season_quality()]

    def _restore_episode_state(self):
        """
        Every episode starts with initial_stock berries (placed again with the agents), growing at the normal rate
        """
        for berry in list(self.berries):
            self._remove_berry(berry)
        for _ in range(self.initial_stock):
            berry = self.dormant_berries.pop(0)
            self.schedule.add(berry)
            self.berries.append(berry)
        self.num_berries = self.initial_stock
        self.season = 0
        self.current_growth_rate = self.growth_rate
        self.harvested_this_step = 0
        self.commons_steps = []
        self.collapse_day = None

    def _init_commons_reporters(self):
        if self.write_data and self.training and self.episode > 1:
            self._merge_earlier_episode_summaries()

    def _model_episode_report_path(self):
        return "data/results/current_run/model_episode_reports_"+self.filepath+".csv"

    def _merge_earlier_episode_summaries(self):
        """
        Runs started before the episode summary was part of the model episode report wrote it to commons_episodes_<filepath>.csv: a
        continued run fills those episodes' summaries into its report from there (the file is left as it is)
        """
        earlier_path = "data/results/current_run/commons_episodes_"+self.filepath+".csv"
        report_path = self._model_episode_report_path()
        if not (os.path.exists(earlier_path) and os.path.exists(report_path)):
            return
        reports = pd.read_csv(report_path)
        earlier = pd.read_csv(earlier_path).set_index("episode")
        missing = reports[self.EPISODE_SUMMARY_COLUMNS].isna().all(axis=1) & reports["episode"].isin(earlier.index)
        if not missing.any():
            return
        for column in self.EPISODE_SUMMARY_COLUMNS:
            reports.loc[missing, column] = reports.loc[missing, "episode"].map(earlier[column])
        writing_path = report_path+".writing"
        reports.to_csv(writing_path, index=False)
        os.replace(writing_path, report_path)

    def extra_agent_report_columns(self):
        return self.AGENT_REPORT_COLUMNS

    def extra_agent_report_values(self, agent):
        """
        This step's stock, harvest, regrowth and growth rate (the commons regrows before agents' rows are reported, so every row of a
        step has the same values)
        """
        step = self.commons_steps[-1]
        return {"stock": step["stock"], "harvested_this_step": step["harvested"], "regrown_this_step": step["regrown"], "growth_rate": step["growth_rate"]}

    def extra_episode_report_columns(self):
        return self.EPISODE_SUMMARY_COLUMNS

    def extra_episode_report_values(self):
        """
        The episode's collapse day, mean and final stock, and efficiency: harvest per step as a fraction of the maximum sustainable
        yield, r * K / 4 (at stock K / 2, with the normal growth rate)
        """
        stocks = [step["stock"] for step in self.commons_steps]
        harvested = [step["harvested"] for step in self.commons_steps]
        return {"collapse_day": self.collapse_day,
                "mean_stock": np.mean(stocks) if stocks else None,
                "final_stock": stocks[-1] if stocks else None,
                "efficiency": np.mean(harvested) / (self.growth_rate * self.carrying_capacity / 4) if harvested else None}

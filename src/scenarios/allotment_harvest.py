from src.harvest_model import HarvestModel
from src.harvest_exception import NumBerriesException
from src.harvest_exception import NumAgentsException
from src.harvest_exception import NumAllotmentsException
from src.harvest_exception import AllotmentWidthException
from src.harvest_exception import AllotmentCapacityException
from src.harvest_exception import MixedSocietyException
from src.harvest_exception import NoAllocationException
from src.harvest_exception import RegrowthDelaysException

class AllotmentHarvest(HarvestModel):
    """
    Allotment harvest scenario agents have only access to specific parts of the grid; every allotment starts with an equal share of the
    berries, but berries regrow more slowly in some allotments than in others
    Instance variables:
        num_start_berries -- the number of berries initiated at the beginning of an episode
        allocations -- dictionary of agent ids, the part of the grid they have access to, and the berries assigned to that agent
        berries -- list of active berry objects
        society_mix -- "homogeneous" (every agent is agent_type) or "heterogeneous" (half agent_type, half baseline)
        mixed_allotments -- how a heterogeneous society shares the allotments (see MIXED_ALLOTMENT_LAYOUTS); ignored when homogeneous
    """
    #by default allotment 1 regrows SLOW_ALLOTMENT_DELAY steps after a berry is foraged, and every other allotment instantly
    SLOW_ALLOTMENT = 1
    SLOW_ALLOTMENT_DELAY = 2
    #"split": allotments hold consecutive agent ids, and the two agent types alternate by id, so every allotment of 2 or more agents has
    #both types; "separate": the agent_type agents share the first half of the allotments and the baseline agents the second half
    MIXED_ALLOTMENT_LAYOUTS = ["split", "separate"]

    def __init__(self,society_mix,num_agents,num_start_berries,num_allotments,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath="",seed=None,norms_module_type="rule_based",seasons=False,season_length=25,allotment_regrowth_delays=None,mixed_allotments="split"):
        super().__init__(num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath,seed,norms_module_type,seasons,season_length,allotment_regrowth_delays)
        self.num_start_berries = num_start_berries
        if num_allotments > num_agents or num_allotments < 1:
            raise NumAllotmentsException(num_agents, num_allotments)
        if num_allotments > max_width:
            raise AllotmentWidthException(max_width, num_allotments)
        self.num_allotments = num_allotments
        self.society_mix = society_mix
        self.mixed_allotments = mixed_allotments
        self._check_mixed_society()
        if allotment_regrowth_delays is None:
            allotment_regrowth_delays = self._default_regrowth_delays()
            self.allotment_regrowth_delays = allotment_regrowth_delays
        elif (len(allotment_regrowth_delays) != num_allotments
                or any(not isinstance(delay, int) or isinstance(delay, bool) or delay < 0 for delay in allotment_regrowth_delays)):
            raise RegrowthDelaysException(num_allotments, allotment_regrowth_delays)
        self.allocations = self._assign_allocations(self._allotment_columns())
        self._check_capacity()
        self.holder_regrowth_delays = {holder: self.allotment_regrowth_delays[data["id"]] for holder, data in self.allocations.items()}
        self._init_agents(society_mix, agent_type, checkpoint_path)
        self.berries = self._init_berries()
        self._init_seasons()

    def _set_berry_holder(self, berry, holder):
        """
        A berry given to an allotment (a new or reused berry in a good season) takes that allotment's bounds,
        so it is placed and regrows inside it
        """
        berry.allocation_id = holder
        berry.min_width, berry.max_width, berry.min_height, berry.max_height = self.allocations[holder]["allotment"]

    def _default_regrowth_delays(self):
        return self.default_regrowth_delays(self.num_allotments)

    @classmethod
    def default_regrowth_delays(cls, num_allotments):
        """
        Regrowth delay of each allotment when none are given: SLOW_ALLOTMENT_DELAY in SLOW_ALLOTMENT (if there is one), 0 elsewhere
        """
        delays = [0] * num_allotments
        if num_allotments > cls.SLOW_ALLOTMENT:
            delays[cls.SLOW_ALLOTMENT] = cls.SLOW_ALLOTMENT_DELAY
        return delays

    def _check_mixed_society(self):
        if self.mixed_allotments not in self.MIXED_ALLOTMENT_LAYOUTS:
            raise MixedSocietyException(f"mixed_allotments must be one of {self.MIXED_ALLOTMENT_LAYOUTS}, got {self.mixed_allotments}")
        if self.society_mix == "homogeneous":
            return
        if self.num_agents % 2 != 0:
            raise MixedSocietyException(f"half the agents are baseline, so the number of agents must be even, got {self.num_agents}")
        if self.mixed_allotments == "separate" and self.num_allotments % 2 != 0:
            raise MixedSocietyException(f"with separate allotments each agent type has half of them, so the number of allotments must be even, got {self.num_allotments}")

    def _allotment_columns(self):
        """
        [start, end) columns of each allotment, tiling the whole grid; when max_width doesn't divide evenly, the first allotments are one column wider
        """
        width, extra = divmod(self.max_width, self.num_allotments)
        columns = []
        start = 0
        for i in range(self.num_allotments):
            end = start + width + (1 if i < extra else 0)
            columns.append((start, end))
            start = end
        return columns

    def _agent_groups(self):
        """
        Agent ids in each allotment: consecutive ids, with any agents left over going to the first allotments; with separate allotments
        for a mixed society, the agent_type half (the lower ids) share the first half of the allotments and the baseline half the rest
        """
        if self.society_mix != "homogeneous" and self.mixed_allotments == "separate":
            half_pop = self.num_agents // 2
            half_allotments = self.num_allotments // 2
            return self._consecutive_groups(range(half_pop), half_allotments) + self._consecutive_groups(range(half_pop, self.num_agents), half_allotments)
        return self._consecutive_groups(range(self.num_agents), self.num_allotments)

    def _consecutive_groups(self, ids, num_groups):
        ids = list(ids)
        group_size, remainder = divmod(len(ids), num_groups)
        groups = []
        start = 0
        for i in range(num_groups):
            end = start + group_size + (1 if i < remainder else 0)
            groups.append(ids[start:end])
            start = end
        return groups

    def _assign_allocations(self, columns):
        """
        Every allotment gets an equal share of the society's berries (the same total as the other scenarios); when the total
        doesn't divide evenly, the first allotments get one more
        """
        total = sum(self._generate_resource_allocations(self.num_agents))
        share, extra = divmod(total, self.num_allotments)
        allocations = {}
        for i, ((start, end), agent_ids) in enumerate(zip(columns, self._agent_groups())):
            current_resource = share + (1 if i < extra else 0)
            key = "allocation_"+str(i)
            allocations[key] = {"id": i, "berry_allocation": current_resource, "allotment":[start,end,0,self.max_height], "agent_ids": agent_ids}
        return allocations

    def _check_capacity(self):
        """
        Every allotment needs a cell for each of its agents and for the most berries it can hold (one more per agent in a good season)
        """
        for data in self.allocations.values():
            min_width, max_width, min_height, max_height = data["allotment"]
            cells = (max_width - min_width) * (max_height - min_height)
            most_berries = data["berry_allocation"] + (len(data["agent_ids"]) if self.seasons else 0)
            needed = len(data["agent_ids"]) + most_berries
            if cells < needed:
                raise AllotmentCapacityException(data["id"], data["allotment"], cells, len(data["agent_ids"]), most_berries)

    def _init_berries(self):
        self.num_berries = 0
        berries = []
        for allocation_data in self.allocations.values():
            for i in range(allocation_data["berry_allocation"]):
                b = self._new_berry(allocation_data["allotment"],allocation_data["id"])
                self._place_agent_in_allotment(b)
                self.num_berries += 1
                berries.append(b)
        if self.num_berries != self.num_start_berries:
            raise NumBerriesException(self.num_start_berries, self.num_berries)
        return berries

    def _init_agents(self, society_mix, agent_type, checkpoint_path):
        self.living_agents = []
        for id in range(self.num_agents):
            allocation_id = self._get_allocation_id(id)
            allotment = self.allocations[allocation_id]["allotment"]
            self._add_agent(id, self._agent_type_of(id, agent_type), allotment, checkpoint_path, allocation_id=allocation_id)
        self.num_living_agents = len(self.living_agents)
        self.berry_id = self.num_living_agents + 1
        if self.num_living_agents != self.num_agents:
            raise NumAgentsException(self.num_agents, self.num_living_agents)

    def _agent_type_of(self, agent_id, agent_type):
        """
        In a mixed society half the agents are baseline: odd ids with split allotments, the upper half of ids with separate allotments
        """
        if self.society_mix == "homogeneous":
            return agent_type
        if self.mixed_allotments == "split":
            return agent_type if agent_id % 2 == 0 else "baseline"
        return agent_type if agent_id < self.num_agents // 2 else "baseline"

    def _get_allocation_id(self, agent_id):
        for allocation_id, allocation_data in self.allocations.items():
            if agent_id in allocation_data["agent_ids"]:
                return allocation_id
        raise NoAllocationException(agent_id)

    def can_throw_to(self, thrower, benefactor):
        """
        Berries can only be thrown to agents in the same allotment
        """
        return thrower.allocation_id == benefactor.allocation_id

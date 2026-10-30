from src.harvest_model import HarvestModel
from ..agent.harvest_agent import HarvestAgent
from src.harvest_exception import NumBerriesException
from src.harvest_exception import NumAgentsException
from src.harvest_exception import NumAllotmentsException
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
    """
    #by default allotment 1 regrows SLOW_ALLOTMENT_DELAY steps after a berry is foraged, and every other allotment instantly
    SLOW_ALLOTMENT = 1
    SLOW_ALLOTMENT_DELAY = 2

    def __init__(self,society_mix,num_agents,num_start_berries,num_allotments,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath="",seed=None,norms_module_type="rule_based",initial_healths=None,seasons=False,season_length=25,season_quality=False,allotment_regrowth_delays=None):
        super().__init__(num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath,seed,norms_module_type,initial_healths,seasons,season_length,season_quality,allotment_regrowth_delays)
        self.num_start_berries = num_start_berries
        if num_allotments > num_agents or num_allotments < 1:
            raise NumAllotmentsException(num_agents, num_allotments)
        self.num_allotments = num_allotments
        if allotment_regrowth_delays is None:
            allotment_regrowth_delays = self._default_regrowth_delays()
            self.allotment_regrowth_delays = allotment_regrowth_delays
        elif (len(allotment_regrowth_delays) != num_allotments
                or any(not isinstance(delay, int) or isinstance(delay, bool) or delay < 0 for delay in allotment_regrowth_delays)):
            raise RegrowthDelaysException(num_allotments, allotment_regrowth_delays)
        allotment_interval = int(max_width // num_allotments)
        self.allocations = self._assign_allocations(allotment_interval)
        #keyed by allotment, so a delay stays with its land when swaps move allocations
        self.holder_regrowth_delays = {holder: self.allotment_regrowth_delays[data["id"]] for holder, data in self.allocations.items()}
        self._init_agents(society_mix, agent_type, checkpoint_path)
        self.berries = self._init_berries()
        self._init_seasons()

    def _set_berry_holder(self, berry, holder):
        """
        Agents stay in their allotments with the same movement bounds; a berry given to another allotment
        takes that allotment's bounds, so it is placed and regrows inside it
        """
        berry.allocation_id = holder
        berry.min_width, berry.max_width, berry.min_height, berry.max_height = self.allocations[holder]["allotment"]

    def _default_regrowth_delays(self):
        """
        Regrowth delay of each allotment when none are given: SLOW_ALLOTMENT_DELAY in SLOW_ALLOTMENT (if there is one), 0 elsewhere
        """
        delays = [0] * self.num_allotments
        if self.num_allotments > self.SLOW_ALLOTMENT:
            delays[self.SLOW_ALLOTMENT] = self.SLOW_ALLOTMENT_DELAY
        return delays

    def _assign_allocations(self, allotment_interval):
        """
        Every allotment gets an equal share of the society's berries (the same total as the other scenarios); when the total
        doesn't divide evenly, the first allotments get one more
        """
        total = sum(self._generate_resource_allocations(self.num_agents))
        share, extra = divmod(total, self.num_allotments)
        allocations = {}
        allotment_start = 0
        allotment_end = allotment_interval
        group_size = self.num_agents // self.num_allotments
        remainder = self.num_agents % self.num_allotments
        start_index = 0
        for i in range(self.num_allotments):
            end_index = start_index + group_size + (1 if i < remainder else 0)
            current_resource = share + (1 if i < extra else 0)
            agent_ids = list(range(start_index, end_index))
            start_index = end_index
            key = "allocation_"+str(i)
            allocations[key] = {"id": i, "berry_allocation": current_resource, "allotment":[allotment_start,allotment_end,0,self.max_height], "agent_ids": agent_ids}
            allotment_start += allotment_interval
            allotment_end += allotment_interval
        return allocations

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
        if society_mix == "homogeneous":
            for id in range(self.num_agents):
                allocation_id = self._get_allocation_id(id)
                allotment = self.allocations[allocation_id]["allotment"]
                self._add_agent(id, agent_type, allotment, checkpoint_path, allocation_id=allocation_id)
        else:
            assert self.num_agents%2 == 0
            half_pop = int(self.num_agents/2)
            for id in range(half_pop):
                allocation_id = self._get_allocation_id(id)
                allotment = self.allocations[allocation_id]["allotment"]
                self._add_agent(id, agent_type, allotment, checkpoint_path, allocation_id=allocation_id)
            for id in range(half_pop):
                allocation_id = self._get_allocation_id(id)
                allotment = self.allocations[allocation_id]["allotment"]
                self._add_agent(id+half_pop, "baseline", allotment, checkpoint_path, allocation_id=allocation_id)
        self.num_living_agents = len(self.living_agents)
        self.berry_id = self.num_living_agents + 1
        if self.num_living_agents != self.num_agents:
            raise NumAgentsException(self.num_agents, self.num_living_agents)
    
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
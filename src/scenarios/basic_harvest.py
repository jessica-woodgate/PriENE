from src.harvest_model import HarvestModel
from src.harvest_exception import RegrowthDelaysNotSupported

class BasicHarvest(HarvestModel):
    """
    Basic harvest scenario; any agent can access any berry, so the whole society is one group holding every berry
    Instance variables:
        num_start_berries -- the number of berries initiated at the beginning of an episode
        allocations -- the society as a single group, so seasons change its berries by one per agent
        berries -- list of active berry objects
    """
    SOCIETY = "allocation_0"

    def __init__(self,society_mix,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath="",seed=None,norms_module_type="rule_based",seasons=False,season_length=25,allotment_regrowth_delays=None):
        if allotment_regrowth_delays is not None:
            raise RegrowthDelaysNotSupported("basic")
        super().__init__(num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath,seed,norms_module_type,seasons,season_length)
        self.num_start_berries = num_start_berries
        self.allocations = {self.SOCIETY: {"id": 0, "berry_allocation": num_start_berries, "agent_ids": list(range(num_agents))}}
        self._init_agents(society_mix,agent_type,checkpoint_path)
        self.berries = self._init_berries()
        self._init_seasons()

    def _init_berries(self):
        berries = []
        self.num_berries = 0
        allotment = [0,self.max_width,0,self.max_height]
        for i in range(self.num_start_berries):
            #labelled as the society's, but agents aren't restricted to an allocation, so any agent can forage any berry
            b = self._new_berry(allotment, self.allocations[self.SOCIETY]["id"])
            self._place_agent_in_allotment(b)
            self.num_berries += 1
            berries.append(b)
        assert(self.num_berries == self.num_start_berries)
        return berries

    def _holder_of(self, agent):
        return self.SOCIETY

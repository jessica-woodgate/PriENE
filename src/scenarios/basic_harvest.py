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

    def __init__(self,society_mix,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath="",seed=None,norms_module_type="rule_based",seasons=False,season_length=25,allotment_regrowth_delays=None,start_episode=1,training_config=None):
        if allotment_regrowth_delays is not None:
            raise RegrowthDelaysNotSupported("basic")
        super().__init__(num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath,seed,norms_module_type,seasons,season_length,start_episode=start_episode,training_config=training_config)
        self.num_start_berries = num_start_berries
        self.allocations = {self.SOCIETY: {"id": 0, "berry_allocation": num_start_berries, "agent_ids": list(range(num_agents))}}
        self._init_agents(society_mix,agent_type,checkpoint_path)
        self.berries = self._init_berries()
        self._init_seasons()

    def _holder_of(self, agent):
        #berries are labelled as the society's, but agents aren't restricted to an allocation, so any agent can forage any berry
        return self.SOCIETY

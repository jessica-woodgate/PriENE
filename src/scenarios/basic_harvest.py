from src.harvest_model import HarvestModel
from src.harvest_exception import InitialHealthsNotSupported

class BasicHarvest(HarvestModel):
    """
    Basic harvest scenario for training agents in; any agent can access any berry
    Instance variables:
        num_start_berries -- the number of berries initiated at the beginning of an episode
        berries -- list of active berry objects
    """
    def __init__(self,society_mix,num_agents,num_start_berries,agent_type,max_width,max_height,max_episodes,max_days,training,checkpoint_path,write_data,write_norms,filepath="",seed=None,norms_module_type="rule_based",initial_healths=None):
        #agents are trained starting with equal health
        if initial_healths is not None:
            raise InitialHealthsNotSupported("basic", initial_healths)
        super().__init__(num_agents,max_width,max_height,max_episodes,max_days,training,write_data,write_norms,filepath,seed,norms_module_type)
        self.num_start_berries = num_start_berries
        self._init_agents(society_mix,agent_type,checkpoint_path)
        self.berries = self._init_berries()

    def _init_berries(self):
        berries = []
        self.num_berries = 0
        allotment = [0,self.max_width,0,self.max_height]
        for i in range(self.num_start_berries):
            b = self._new_berry(allotment)
            self._place_agent_in_allotment(b)
            self.num_berries += 1
            berries.append(b)
        assert(self.num_berries == self.num_start_berries)
        return berries